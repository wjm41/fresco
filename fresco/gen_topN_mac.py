import os
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import Descriptors, MolFromSmiles, Lipinski, rdMolDescriptors

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from mpi4py import MPI

logging.basicConfig(level=logging.INFO)

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def return_borders(index, dat_len, mpi_size):
    mpi_borders = np.linspace(0, dat_len, mpi_size + 1).astype('int')

    border_low = mpi_borders[index]
    border_high = mpi_borders[index+1]
    return border_low, border_high
 

dir = '/rds-d2/user/wjm41/hpc-work/datasets/ZINC/real/'

dir2 = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/'

scored_dirs1 = open(dir+'missing.txt', 'r').read().splitlines()
scored_dirs2 = open(dir2+'missing.txt', 'r').read().splitlines()

def filter_physchem(row):
   
    row['keep'] = True
    mol = MolFromSmiles(row['smiles'])

    if mol.GetNumHeavyAtoms() > 25 or mol.GetNumHeavyAtoms() < 8:
        row['keep'] = False
    elif Descriptors.MolWt(mol) > 400 or Descriptors.MolWt(mol) < 109:
        row['keep'] = False
    elif Descriptors.MolLogP(mol) > 3.5 or Descriptors.MolLogP(mol) < -2.7:
        row['keep'] = False
    elif Descriptors.TPSA(mol) > 179 or Descriptors.TPSA(mol) < 3:
        row['keep'] = False
    elif Lipinski.NumHAcceptors(mol) > 8 or Lipinski.NumHAcceptors(mol) < 1:
        row['keep'] = False
    elif Lipinski.NumHDonors(mol) > 4:
        row['keep'] = False
    elif rdMolDescriptors.CalcNumRotatableBonds(mol) > 10:
        row['keep'] = False

    return row

def proc_df(path, N=1000):
    important = ['Donor-Aromatic',
                 'Aromatic-Aromatic',
                 'Acceptor-Acceptor']

    df = pd.read_csv(path)
    df['2body_score'] = np.mean(df[important], axis=1)
    df = df[df['2body_score'].notnull()]
    df = df[df['smiles'].notnull()]
    df = df.drop_duplicates(subset=['smiles'], keep='first')
    if len(df)!=0:
        df = df.apply(filter_physchem, axis=1)
        df = df[df['keep']]
        df = df.sort_values(by='2body_score')
        df = df.iloc[:min(N,len(df))]
    return df

N = 10000

my_border_low, my_border_high = return_borders(mpi_rank, len(scored_dirs2), mpi_size)
scored_dirs2 = scored_dirs2[my_border_low: my_border_high]

for i,folder in tqdm(enumerate(scored_dirs2), total = len(scored_dirs2)):
    logging.info(folder)
    if not os.path.isfile(dir2+folder+'/topN_mac.csv'):
        pairs = [file for file in os.listdir(dir2+folder) if "pairs_mpi_" in file]
        scores = [file for file in os.listdir(dir2+folder) if "_mac" in file and ".csv" in file]

        if os.path.isfile(dir2+folder+'/scores_mac.csv') and os.stat(dir2+folder+'/scores_mac.csv').st_size!=0:
            df = proc_df(dir2+folder+'/scores_mac.csv', N=N)
            df.to_csv(dir2+folder+'/topN_mac.csv', index=False)
            
        elif len(pairs) <= len(scores) and len(pairs)!=0 :
            df_all = []
            for score_csv in tqdm(scores):
                if os.stat(dir2+folder+'/' + score_csv).st_size!=0:
                    df = proc_df(dir2+folder+'/'+score_csv, N=N)
                    df_all.append(df)
                
            df_all = pd.concat(df_all).drop_duplicates(subset=['smiles'], keep='first')
            df_all = df_all.sort_values(by='2body_score').iloc[:min(N, len(df_all))]
            df_all.to_csv(dir2+folder+'/topN_mac.csv', index=False)
        
