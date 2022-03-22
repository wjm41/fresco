from os import listdir
import os.path
import pickle
import math
import random
import sys
import logging 

import numpy as np
import pandas as pd
from tqdm import tqdm

import os.path

from frag_funcs import return_pcore_dataframe, get_pair_distances

tranch_file = open(sys.argv[1],'r')
#tranches = [x.split('/')[-1][:-4] for x in tranch_file.read().splitlines()]
tranches = [x for x in tranch_file.read().splitlines()]
    
#zinc_dir = '/rds-d2/user/wjm41/hpc-work/datasets/ZINC/real/'
zinc_dir = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/'

folder_names = [zinc_dir + x for x in tranches]
# folder_names.sort()

important = ['Donor-Aromatic',
             'Donor-Acceptor',
             'Aromatic-Aromatic']
unimportant = ['Donor-Donor',
               'Aromatic-Acceptor',
               'Acceptor-Acceptor']

pairs = important+unimportant

kde_dict = pickle.load(open('kde_dict_mac1.pickle', 'rb'))

for folder in tqdm(folder_names):
    logging.warning(folder)
    if os.path.isfile(folder+'/pairs.pickle') and not os.path.isfile(folder+'/scores_mac.csv'): # check file existence 
       with open(folder+'/pairs.pickle', 'rb') as handle:
           zinc_pairs = pickle.load(handle)    
       try:    
           file = open(folder+'/mols.smi', 'r')
           zinc_smi = file.read().splitlines()    
           assert len(zinc_smi) == len(zinc_pairs)

           scores = {}
           
           for n, combo in enumerate(pairs):
               kde = kde_dict[combo]
               
               combo_scores = np.empty((len(zinc_pairs)))    
               for i in tqdm(range(len(zinc_pairs))):
                   try:
                       ith_score = np.abs(kde.score_samples(zinc_pairs[i][combo][0].reshape(-1,1)))
                       combo_scores[i] = np.mean(ith_score)
                   except:
                       combo_scores[i] = np.nan

               scores[combo] = combo_scores 
       
           df = pd.DataFrame.from_dict(scores)
           df['smiles'] = zinc_smi
           df = df[['smiles']+pairs]
           df.to_csv(folder+'/scores_mac.csv', index=False)

       except AssertionError:
           logging.warning('{} assertion error'.format(folder))
           continue
       except FileNotFoundError:
           logging.warning('{} file not found error'.format(folder)) 
           continue
