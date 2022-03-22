import os.path
import pickle
import math
import random
import sys
import logging
import subprocess
from tqdm import tqdm
import numpy as np

from rdkit import Chem
import os.path

tranch_file = open(sys.argv[1],'r')
tranches = tranch_file.read().splitlines()

pickle_index = str(sys.argv[2])
mpi_size = int(sys.argv[3])
# tranches = [x.split('/')[-1][:-4] for x in tranch_file.read().splitlines()]
    
#zinc_dir = '/rds-d2/user/wjm41/hpc-work/datasets/ZINC/real/'
zinc_dir = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/'

def bash_command(cmd):
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    return p.wait()

folder_names = [zinc_dir + x for x in tranches]
# folder_names.sort()

important = ['Donor-Aromatic',
            'Aromatic-Acceptor',
            'Aromatic-Aromatic']
unimportant = ['Donor-Donor',
               'Donor-Acceptor',
               'Acceptor-Acceptor']

pairs = important+unimportant

for folder in tqdm(folder_names):
    logging.warning(folder)
    try:
        #del_cmd = 'rm '+folder+'/mols_cleanup_'+pickle_index+'_mpi_*'
        #print(bash_command(del_cmd))
        to_cat = []
        for mpi_rank in range(mpi_size):
            
            with open(folder+'/pairs_cleanup_'+pickle_index+'_mpi_'+str(mpi_rank)+'.pickle', 'rb') as handle:
                to_cat.append(pickle.load(handle))
            try:
                os.remove(folder+'/pairs_cleanup_'+pickle_index+'_mpi_'+str(mpi_rank)+'.pickle') #delete mols.sdf afterwards
            except:
                pass
                  
        pickle.dump(to_cat, open(folder+'/pairs_mpi_'+pickle_index+'.pickle', 'wb'))
    except Exception as ex:
        print(ex)
        pass
    
