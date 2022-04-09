import os.path
import pickle
import math
import random
import sys
import logging

from tqdm import tqdm
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # removes annoying RDKit warnings

import os.path

from mpi4py import MPI

from frag_funcs import return_pcore_dataframe, get_pair_distances, file_size

logging.basicConfig(level=logging.INFO)

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

folder = sys.argv[1]
pickle_index = str(sys.argv[2])
index_size = int(sys.argv[3])

def return_borders(index, dat_len, mpi_size):
    mpi_borders = np.linspace(0, dat_len, mpi_size + 1).astype('int')

    border_low = mpi_borders[index]
    border_high = mpi_borders[index+1]
    return border_low, border_high

important = ['Donor-Aromatic',
            'Aromatic-Aromatic']
unimportant = ['Donor-Donor',
               'Donor-Acceptor',
            'Aromatic-Acceptor',
               'Acceptor-Acceptor']

pairs = important+unimportant

if mpi_rank==0:
    logging.info('{} SDF file size: {}'.format(folder, file_size(folder+'/mols.sdf')))

suppl = Chem.SDMolSupplier(folder+'/mols.sdf', sanitize=True, removeHs=False)
my_border_low, my_border_high = return_borders(int(pickle_index), len(suppl), index_size)

my_mols = [None]*(my_border_high-my_border_low)

new_border_low, new_border_high = return_borders(mpi_rank, len(my_mols), mpi_size)
my_mols = [None]*(new_border_high-new_border_low)

for i,n in enumerate(range(my_border_low + new_border_low, my_border_low + new_border_high)):
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

if mpi_rank==0:
    logging.info('Generating dataframe...')

pcore_df = return_pcore_dataframe(my_mols, interesting_pcores, hit=False)
if pcore_df is None:
    logging.info('Molecules for MPI process {} do not contain relevant pharmacophores. Skipping!'.format(mpi_rank))

else:
    smis = pcore_df['smiles'].unique()
    
    f = open(folder+'/mols_cleanup_'+pickle_index+'_mpi_'+str(mpi_rank)+'.smi', 'w')
    for smi in smis:
        f.write(smi+'\n')
    f.close()
    
    if mpi_rank==0:
        logging.info('Generating histograms...')
    
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

kde_dict = pickle.load(open('kde_dict_mac1.pickle', 'rb'))

histograms = {}

for n in range(mpi_size):
    filename = folder+'/pairs_cleanup_'+pickle_index+'_mpi_'+str(n)+'.pickle' 
    try:
    	histograms.update((pickle.load(open(filename, 'rb'))))
    except FileNotFoundError:
        continue
          
if mpi_rank==0:
    logging.info('Beginning scoring...')
    pickle.dump(histograms, open(folder+'/pairs_mpi_'+pickle_index+'.pickle', 'wb'))

my_low, my_high = return_borders(mpi_rank, len(histograms), mpi_size)
my_mols = list(histograms.keys())[my_low:my_high]

my_hist = {smi:histograms[smi] for smi in my_mols}
scores = {}

for smi in tqdm(my_hist):
    score = {}
    for n, combo in enumerate(pairs):
        kde = kde_dict[combo]
        try:
            ith_score = np.abs(kde.score_samples(histograms[smi][combo][0].reshape(-1,1)))
            score[combo] = np.mean(ith_score)
        except:
            score[combo] = np.nan

    scores[smi] = score

full_scores = mpi_comm.gather(scores,root=0)

if mpi_rank==0:
    for score in full_scores[1:]:
        scores.update(score)
    assert len(scores) == len(histograms)
    df = pd.DataFrame.from_dict(scores, orient='index')
    df.index = df.index.rename('smiles')
    df = df.reset_index()
    df.to_csv(folder+'/scores'+pickle_index+'_mac.csv', index=False)
    logging.info('Scoring Complete!')
