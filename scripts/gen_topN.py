import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import Descriptors, MolFromSmiles

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from mpi4py import MPI

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

dirs1 = [x for x in os.listdir(dir) if os.path.isdir(dir+x)]

dirs2 = [x for x in os.listdir(dir2) if os.path.isdir(dir2+x)]

def filter_small(row):
   
    row['keep'] = True
    mol = MolFromSmiles(row['smiles'])

    if mol.GetNumHeavyAtoms() >= 25 or Descriptors.MolLogP(mol) >= 3.5:
        row['keep'] = False
    return row

def proc_df(path, N=1000):
    df = pd.read_csv(path)
    df = df[df['2body_score'].notnull()]
    df = df[df['smiles'].notnull()]
    if len(df)!=0:
        df = df.apply(filter_small, axis=1)
        df = df[df['keep']]
        df = df.sort_values(by='2body_score')
        df = df.iloc[:min(N,len(df))]
    return df

N = 1000

my_border_low, my_border_high = return_borders(mpi_rank, len(dirs2), mpi_size)
dirs2 = dirs2[my_border_low: my_border_high]

for i,folder in tqdm(enumerate(dirs2), total = len(dirs2)):
     #if not os.path.isfile(dir2+folder+'/topN_new.csv'):
   pairs = [file for file in os.listdir(dir2+folder) if "pairs_mpi_" in file]
   scores = [file for file in os.listdir(dir2+folder) if ".csv" in file]

   if os.path.isfile(dir2+folder+'/scores.csv'):
       df = proc_df(dir2+folder+'/scores.csv')
       df.to_csv(dir2+folder+'/topN_new.csv', index=False)
       
   elif len(pairs) == len(scores) and len(pairs)!=0 :
       df_all = []
       for score_csv in scores:
           df = proc_df(dir2+folder+'/'+score_csv)
           df_all.append(df)
           
       df_all = pd.concat(df_all).sort_values(by='2body_score')
       df_all = df_all.sort_values(by='2body_score').iloc[:min(N, len(df_all))]
       df_all.to_csv(dir2+folder+'/topN_new.csv', index=False)

