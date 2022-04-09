import os
import os.path
import sys
import logging

from tqdm import tqdm
import pandas as pd
import numpy as np

from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')

from frag_funcs import clean_smiles

tranch_file = open(sys.argv[1],'r')
tranches = tranch_file.read().splitlines()
# tranches = [x.split('/')[-1][:-4] for x in tranch_file.read().splitlines()]

#zinc_dir = '/rds-d2/user/wjm41/hpc-work/datasets/ZINC/real/'
zinc_dir = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/'

folder_names = [zinc_dir + x for x in tranches]

important = ['Donor-Aromatic',
             'Donor-Acceptor',
             'Aromatic-Aromatic']
unimportant = ['Donor-Donor',
               'Aromatic-Acceptor',
               'Acceptor-Acceptor']

pairs = important + unimportant
for folder in tqdm(folder_names):
    for mpi_ind in range(12):
        try:
            zinc_smi = open(folder+'/mols'+str(mpi_ind)+'.smi', 'r').read().splitlines()
            scores = np.load(folder+'/scores'+str(mpi_ind)+'.npy')
            if len(zinc_smi)==len(scores):
               continue
            elif len(zinc_smi)==0:
               logging.warning(folder)
               continue

            zinc_smi = clean_smiles(zinc_smi) # filters mols without pharmacophores!

            assert len(zinc_smi) == len(scores)
            
            f = open(folder+'/mols'+str(mpi_ind)+'.smi', 'w')
            for smi in zinc_smi:
                f.write(smi+'\n')
            f.close()

            df = pd.DataFrame(data=scores, columns = pairs)
            df['smiles'] = zinc_smi
            df['2body_score'] = df[important].mean(axis=1)
            df = df[['smiles', '2body_score']+pairs]


            df.to_csv(folder+'/scores'+str(mpi_ind)+'_acc.csv', index=False)
        
        except FileNotFoundError:
           continue 
