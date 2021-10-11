import os.path
import pickle
import math
import random
import sys
import logging
import subprocess

from tqdm import tqdm
import pandas as pd
import numpy as np

from rdkit import Chem
import os.path

from mpi4py import MPI

from frag_funcs import return_borders

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

logging.basicConfig(level=logging.INFO)

folder = sys.argv[1]
pickle_index = str(sys.argv[2])

def bash_command(cmd):
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    return p.wait()

important = ['Donor-Aromatic',
            'Aromatic-Acceptor',
            'Aromatic-Aromatic']
unimportant = ['Donor-Donor',
               'Donor-Acceptor',
               'Acceptor-Acceptor']

pairs = important+unimportant

kde_dict = pickle.load(open('kde_dict_mac1.pickle', 'rb'))

histograms = {}

for n in range(mpi_size):
    
    with open(folder+'/pairs_cleanup_'+pickle_index+'_mpi_'+str(n)+'.pickle', 'rb') as handle:
        histograms.update((pickle.load(handle)))
          
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
    df.to_csv(folder+'/scores'+str(pickle_index)+'_mac.csv', index=False)
    logging.info('Scoring Complete!')
