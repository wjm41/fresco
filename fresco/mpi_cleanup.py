import os.path
import pickle
import math
import random
import sys
import logging

from tqdm import tqdm
import numpy as np

from rdkit import Chem
import os.path

from mpi4py import MPI

from frag_funcs import return_pcore_dataframe, get_pair_distances, clean_smiles

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

tranch_file = open(sys.argv[1],'r')
tranches = tranch_file.read().splitlines()

pickle_index = str(sys.argv[2])
# tranches = [x.split('/')[-1][:-4] for x in tranch_file.read().splitlines()]
    
#zinc_dir = '/rds-d2/user/wjm41/hpc-work/datasets/ZINC/real/'
zinc_dir = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/'

def return_borders(index, dat_len, mpi_size):
    mpi_borders = np.linspace(0, dat_len, mpi_size + 1).astype('int')

    border_low = mpi_borders[index]
    border_high = mpi_borders[index+1]
    return border_low, border_high


folder_names = [zinc_dir + x for x in tranches]
# folder_names.sort()

important = ['Donor-Aromatic',
            'Aromatic-Aromatic']
unimportant = ['Donor-Donor',
               'Donor-Acceptor',
            'Aromatic-Acceptor',
               'Acceptor-Acceptor']

pairs = important+unimportant

for folder in tqdm(folder_names):
    logging.warning(folder)
    if not os.path.isfile(folder+'/mols'+pickle_index+'.smi') or os.stat(folder+'/mols'+pickle_index+'.smi').st_size==0:

       suppl = Chem.SDMolSupplier(folder+'/mols.sdf', sanitize=True, removeHs=False)
       my_border_low, my_border_high = return_borders(int(pickle_index), len(suppl), mpi_size)

       my_mols = [None]*(my_border_high-my_border_low)

       new_border_low, new_border_high = return_borders(mpi_rank, len(my_mols), mpi_size)
       my_mols = [None]*(new_border_high-new_border_low)

       for i,n in enumerate(range(my_border_low + new_border_low, my_border_low + new_border_high+1)):
           try:
               mol = suppl[n]
               conf = mol.GetConformer()

               mol_data = [mol]
               for j,atom in enumerate(mol.GetAtoms()):
                   mol_data.append([atom.GetSymbol(),
                                       conf.GetPositions()[j]
                                       ])
               my_mols[i] = mol_data
           except:
               pass

       my_mols = [mol for mol in my_mols if mol is not None]

       interesting_pcores = ['Donor', 'Acceptor', 'Aromatic']
       
       pcore_df = return_pcore_dataframe(my_mols, interesting_pcores, hit=False)
       smis = pcore_df['smiles'].unique()

       f = open(folder+'/mols_cleanup_'+pickle_index+'_mpi_'+str(mpi_rank)+'.smi', 'w')
       for smi in smis:
           f.write(smi+'\n')
       f.close()

       print('Number of valid molecules for tranch {} :{}'.format(folder.split('/')[-1],len(smis)))
       
       my_mols = pcore_df.groupby('smiles')
       histograms = {}

       for j,smi in enumerate(smis):
           mol_histogram = {}
           match = my_mols.get_group(smi)

           for combo in pairs:
               core_a,core_b = combo.split('-')

               mol_histogram[combo]= get_pair_distances(match, core_a, core_b, frag=False, active=None)
           histograms[smi] = mol_histogram

       assert len(smis) == len(histograms)
                         
       pickle.dump(histograms, open(folder+'/pairs_cleanup_'+pickle_index+'_mpi_'+str(mpi_rank)+'.pickle', 'wb')) 

       mpi_comm.Barrier()
