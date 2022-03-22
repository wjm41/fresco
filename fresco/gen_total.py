import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import Descriptors, MolFromSmiles

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

dir = '/rds-d2/user/wjm41/hpc-work/datasets/ZINC/real/'

dir2 = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/'

dirs1 = [x for x in os.listdir(dir) if os.path.isdir(dir+x)]

dirs2 = [x for x in os.listdir(dir2) if os.path.isdir(dir2+x)]

def filter_small(row):
   
    row['keep'] = True
    try:
        mol = MolFromSmiles(row['smiles'])

        if mol.GetNumHeavyAtoms() >= 25 or Descriptors.MolLogP(mol) >= 3.5:
            row['keep'] = False
    except Exception as ex:
        row['keep'] = False
        print(ex)
    return row

N = 1000

df_all_2 = []
for i,folder in tqdm(enumerate(dirs2), total = len(dirs2)):
    if os.path.isfile(dir2+folder+'/topN.csv'):
        df = pd.read_csv(dir2+folder+'/topN.csv')
#         print(df)
        df = df[df['2body_score'].notnull()]
        df = df[df['smiles'].notnull()]
        if len(df)!=0:
            df = df.apply(filter_small, axis=1)
            if len(df)!=0:
                df = df[df['keep']]
                if len(df)!=0:
                    df_all_2.append(df)
        
df_all_2 = pd.concat(df_all_2).sort_values(by='2body_score').iloc[:min(N,len(df_all_2))]
df_all_2.to_csv(dir2+'topN_new.csv', index=False)
