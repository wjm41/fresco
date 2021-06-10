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

with open('kde_dict_opt.pickle', 'rb') as handle:
    kde_dict = pickle.load(handle)

for folder in tqdm(folder_names):
    logging.warning(folder)
    try:
        if os.path.isfile(folder+'/pairs.pickle'): # check file existence 
            
            with open(folder+'/pairs.pickle', 'rb') as handle:
                zinc_pairs = pickle.load(handle)    
                
            scores = np.empty((len(zinc_pairs), len(pairs)))    
            
            for n, combo in enumerate(pairs):
                kde = kde_dict[combo]

                for i in range(len(zinc_pairs)):
                    try:
                        ith_score = np.abs(kde.score_samples(zinc_pairs[i][combo][0].reshape(-1,1)))
                        scores[i,n] = np.mean(ith_score)
                    except:
                        scores[i,n] = np.nan

            np.save(folder+'/scores.npy', scores)
            scores_imp = scores[:,:len(important)]
            np.save(folder+'/scores_imp.npy', scores_imp)
            
            if os.path.isfile(folder+'/mols.smi'):
                file = open(folder+'/mols.smi', 'r')
                zinc_smi = file.read().splitlines()    

                df = pd.DataFrame(list(zip(zinc_smi, np.mean(scores_imp, axis=1))), 
                                 columns = ['smiles','2body_score'])
                df.to_csv(folder+'/scores_acc.csv', index=False)
    except:
        continue
