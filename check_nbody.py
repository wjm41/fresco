import pickle
import math
import sys
import random
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product

from sklearn.model_selection import GridSearchCV
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon as jsd
from scipy.stats import ks_2samp

from mpi4py import MPI

from frag_funcs import return_random_dataframe, return_pcore_dataframe, get_pair_distances, get_trip_distances, fit_pair_kde, fit_trip_kde

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

logging.basicConfig()
logging.root.setLevel(logging.INFO)

n_rand = 10
with open('all_ligands.list', 'rb') as pickle_file:
    frags = pickle.load(pickle_file)
with open('frag_pair_distance_dict.pickle', 'rb') as handle:
    frag_pair_distance_dict = pickle.load(handle)
with open('frag_trip_distance_dict.pickle', 'rb') as handle:
    frag_trip_distance_dict = pickle.load(handle)

# real_pair_dicts = [{} for i in range(n_rand)]  
# rand_pair_dicts = [{} for i in range(n_rand)]  

# interesting_pcores = ['Donor', 'Acceptor', 'Aromatic']

# fragpcore_dfs = [return_pcore_dataframe(frags, interesting_pcores, jiggle=True) for i in range(n_rand)]
# rand_dfs = [return_random_dataframe(frags, interesting_pcores) for i in range(n_rand)]

# for pcore_pair in tqdm(product(interesting_pcores,repeat=2)):
#     core_a,core_b = pcore_pair
#     combo = core_a+'-'+core_b
    
# #     frag_pair_distance_dict[combo] = np.hstack(get_pair_distances(fragpcore_df, core_a, core_b, frag=True))
#     for i,fragpcore_df in enumerate(fragpcore_dfs):
#         real_pair_dicts[i][combo] = np.hstack(get_pair_distances(fragpcore_df, core_a, core_b, frag=True))
        
#     for i,rand_df in enumerate(rand_dfs):
#         rand_pair_dicts[i][combo] = np.hstack(get_pair_distances(rand_df, core_a, core_b, frag=True))
        
trips = ['Donor-Donor-Donor',
         'Donor-Acceptor-Acceptor',
         'Acceptor-Acceptor-Acceptor',
         'Acceptor-Acceptor-Aromatic',
         'Acceptor-Aromatic-Aromatic',
         'Aromatic-Aromatic-Aromatic',
         'Donor-Donor-Acceptor',
         'Donor-Donor-Aromatic',
         'Donor-Acceptor-Aromatic',
         'Donor-Aromatic-Aromatic']  
      
rand_trip_dicts = [{} for i in range(n_rand)]  

# for combo in tqdm(trips):
#     core_a, core_b, core_c = combo.split('-')
    
# #     frag_trip_distance_dict[combo] = np.hstack(get_trip_distances(fragpcore_df, core_a, core_b, core_c, frag=True, active=False))
#     for i,fragpcore_df in enumerate(fragpcore_dfs):
#         real_trip_dicts[i][combo] = np.hstack(get_trip_distances(fragpcore_df, core_a, core_b, core_c, frag=True))
        
#     for i,rand_df in enumerate(rand_dfs):
#         rand_trip_dicts[i][combo] = np.hstack(get_trip_distances(rand_df, core_a, core_b, core_c, frag=True))

# with open('rand_pair_dicts.pickle', 'wb') as handle:
#     pickle.dump(rand_pair_dicts, handle)     
    
# with open('rand_trip_dicts.pickle', 'wb') as handle:
#     pickle.dump(rand_trip_dicts, handle)     

# kde_dict = {}
# rand_kde_dicts = [{} for i in range(n_rand)]

# for combo in tqdm(trips):
#     core_a, core_b, core_c = combo.split('-')
#     combo_list.append(combo)
    
#     pair1 = core_a+'-'+core_b
#     pair2 = core_b+'-'+core_c
#     pair3 = core_c+'-'+core_a
    
#     if pair1 not in kde_dict:
#         kde_dict[pair1] = 'placeholder'
#         for i in range(n_rand):
#             rand_kde_dicts[i][pair1] = fit_pair_kde(rand_pair_dicts[i][pair1])
#     if pair2 not in kde_dict:
#         kde_dict[pair2] = 'placeholder'
#         for i in range(n_rand):
#             rand_kde_dicts[i][pair2] = fit_pair_kde(rand_pair_dicts[i][pair2])
#     if pair3 not in kde_dict:
#         kde_dict[pair3] = 'placeholder'
#         for i in range(n_rand):
#             rand_kde_dicts[i][pair3] = fit_pair_kde(rand_pair_dicts[i][pair3])
#     if combo not in kde_dict:
#         kde_dict[combo] = 'placeholder'
#         for i in range(n_rand):
#             rand_kde_dicts[i][combo] = fit_trip_kde(rand_trip_dicts[i][combo])
        
# with open('rand_kde_dicts.pickle', 'wb') as handle:
#     pickle.dump(rand_kde_dicts, handle)        
    
with open('kde_dict.pickle', 'rb') as pickle_file:
    kde_dict = pickle.load(pickle_file)
    
with open('rand_kde_dicts.pickle', 'rb') as pickle_file:
    rand_kde_dicts = pickle.load(pickle_file)
    
combo_list = []

n = int(sys.argv[1])
experiment = sys.argv[2]

for i, combo in tqdm(enumerate(trips), total = len(trips)):
    core_a, core_b, core_c = combo.split('-')
    combo_list.append(combo)

    if mpi_rank==i:
        pair1 = core_a+'-'+core_b
        pair2 = core_b+'-'+core_c
        pair3 = core_c+'-'+core_a
        
        x = np.linspace(0, np.amax(frag_pair_distance_dict[pair1]), n)
        y = np.linspace(0, np.amax(frag_pair_distance_dict[pair2]), n)
        z = np.linspace(0, np.amax(frag_pair_distance_dict[pair3]), n)
        xv, yv, zv = np.meshgrid(x, y, z)
        xyz = np.array([xv.ravel(), yv.ravel(), zv.ravel()])
        
        if experiment == '3v2':
            kde_pair1 = kde_dict[pair1]
            kde_pair2 = kde_dict[pair2]
            kde_pair3 = kde_dict[pair3]

            x_score = np.exp(kde_pair1.score_samples(xv.reshape(-1,1)))
            y_score = np.exp(kde_pair2.score_samples(yv.reshape(-1,1)))
            z_score = np.exp(kde_pair3.score_samples(zv.reshape(-1,1)))
            
            pair_dist = (x_score.ravel()*y_score.ravel()*z_score.ravel()).flatten()
            pair_dist = pair_dist/np.sum(pair_dist)

            kde_trip = kde_dict[combo]
            trip_dist = np.exp(kde_trip.score_samples(xyz.T)).flatten()
            trip_dist = trip_dist.flatten()
            trip_dist = trip_dist/np.sum(trip_dist)
            
            exp_val = ks_2samp(trip_dist, pair_dist)[1]

        elif experiment == '3vr':
            kde_trip = kde_dict[combo]
            
            trip_dist = np.exp(kde_trip.score_samples(xyz.T)).flatten()
            trip_dist = trip_dist.flatten()
            trip_dist = trip_dist/np.sum(trip_dist)
            
            exp_list = []
            for i in range(n_rand):
                kde_rand = rand_kde_dicts[i][combo]
                rand_dist = np.exp(kde_rand.score_samples(xyz.T)).flatten()
                rand_dist = rand_dist.flatten()
                rand_dist = rand_dist/np.sum(rand_dist)
                
                exp_list.append(ks_2samp(trip_dist, rand_dist)[1])
                
            exp_val = np.mean(exp_list)

        elif experiment == '2vr':
            kde_pair1 = kde_dict[pair1]
            kde_pair2 = kde_dict[pair2]
            kde_pair3 = kde_dict[pair3]

            x_score = np.exp(kde_pair1.score_samples(xv.reshape(-1,1)))
            y_score = np.exp(kde_pair2.score_samples(yv.reshape(-1,1)))
            z_score = np.exp(kde_pair3.score_samples(zv.reshape(-1,1)))
            
            pair_dist = (x_score.ravel()*y_score.ravel()*z_score.ravel()).flatten()
            pair_dist = pair_dist/np.sum(pair_dist)
            
            exp_list = []
            for i in range(n_rand):

                kde_rand = rand_kde_dicts[i][combo]
                rand_dist = np.exp(kde_rand.score_samples(xyz.T)).flatten()
                rand_dist = rand_dist.flatten()
                rand_dist = rand_dist/np.sum(rand_dist)
                
                exp_list.append(ks_2samp(pair_dist, rand_dist)[1])
                
            exp_val = np.mean(exp_list) 
            
        elif experiment == 'rvr':
            exp_list = []
            for i in range(n_rand):
                kde_pair1 = rand_kde_dicts[i][pair1]
                kde_pair2 = rand_kde_dicts[i][pair2]
                kde_pair3 = rand_kde_dicts[i][pair3]

                x_score = np.exp(kde_pair1.score_samples(xv.reshape(-1,1)))
                y_score = np.exp(kde_pair2.score_samples(yv.reshape(-1,1)))
                z_score = np.exp(kde_pair3.score_samples(zv.reshape(-1,1)))

                pair_dist = (x_score.ravel()*y_score.ravel()*z_score.ravel()).flatten()
                pair_dist = pair_dist/np.sum(pair_dist)

                kde_rand = rand_kde_dicts[i][combo]
                trip_dist = np.exp(kde_rand.score_samples(xyz.T)).flatten()
                trip_dist = trip_dist.flatten()
                trip_dist = trip_dist/np.sum(trip_dist)

                exp_list.append(ks_2samp(trip_dist, pair_dist)[1])
                
            exp_val = np.mean(exp_list) 
        else:
            print('Please choose an appropriate experiment')
            raise Exception

val_list = mpi_comm.gather(exp_val, root=0)


if mpi_rank==0:
    df = pd.DataFrame(list(zip(combo_list, val_list)), 
                             columns = ['combo', experiment + ' p-value'])
    print('nx, ny, nz = {},{},{}'.format(n, n, n))
    print(df.round({experiment + ' p-value': 2}))
    df.to_csv('{}_{}.csv'.format(experiment, n), index=False)


