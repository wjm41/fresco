import logging
import argparse
import os.path
import pickle
import dill as pickle

import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
from tqdm import tqdm

from frag_funcs import score_dist

import os.path

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def score_mpi_folders(folder_names, kde_dict, target_name):
    pairs = ['Donor-Aromatic',
             'Donor-Acceptor',
             'Aromatic-Aromatic',
             'Donor-Donor',
             'Aromatic-Acceptor',
             'Acceptor-Acceptor']

    for folder in tqdm(folder_names):
        for mpi_ind in range(12):
            try:
                if not os.path.isfile(folder+'/scores'+str(mpi_ind)+'_'+target_name+'.csv'):
                    # if True:
                    twobody_dist = pickle.load(
                        open(folder+'/pairs_mpi_'+str(mpi_ind)+'.pickle', 'rb'))
                    if len(twobody_dist) != 0:

                        # fix silly shit
                        if isinstance(twobody_dist, list) and isinstance(twobody_dist[0], list):
                            concat_dist = []
                            for x in twobody_dist:
                                concat_dist = concat_dist + x
                            twobody_dist = concat_dist

                        elif isinstance(list(twobody_dist)[0], dict):
                            # print('dict')
                            real_smi = open(folder+'/mols'+str(mpi_ind)+'.smi',
                                            'r').read().splitlines()
                            assert len(real_smi) == len(twobody_dist)

                            df = pd.DataFrame(columns=['smiles']+pairs)
                            df['smiles'] = real_smi
                            # scores = {}

                            for combo in pairs:
                                kde = kde_dict[combo]

                                # combo_scores = np.empty((len(twobody_dist)))
                                for i in range(len(twobody_dist)):
                                    real_dist = twobody_dist[i][combo][0].reshape(
                                        -1, 1)
                                    df.at[i, combo] = score_dist(
                                        kde, real_dist)
                                #     try:
                                #         ith_score = np.abs(kde.score_samples(
                                #             twobody_dist[i][combo][0].reshape(-1, 1)))
                                #         combo_scores[i] = np.mean(ith_score)

                                #         print(np.amin(combo_scores))
                                #         print(np.amax(combo_scores))
                                #     except:
                                #         combo_scores[i] = np.nan

                                # scores[combo] = combo_scores

                                # df = pd.DataFrame.from_dict(scores)
                                # df['smiles'] = real_smi
                                # df = df[['smiles']+pairs]

                            scores = df[pairs].to_numpy().astype(float)
                            scores[np.all(np.isnan(scores), axis=1)] = -100
                            df['mean_score'] = np.nanmean(
                                scores, axis=1)
                            df.to_csv(folder+'/scores'+str(mpi_ind) +
                                      '_'+target_name+'.csv', index=False)

                        elif isinstance(list(twobody_dist)[0], str):
                            # print('string')
                            scores = {}

                            for smi in twobody_dist:
                                score = {}
                                for combo in pairs:
                                    kde = kde_dict[combo]
                                    real_dist = twobody_dist[smi][combo][0].reshape(
                                        -1, 1)
                                    score[combo] = score_dist(kde, real_dist)
                                    # try:
                                    #     ith_score = np.abs(kde.score_samples(
                                    #         twobody_dist[smi][combo][0].reshape(-1, 1)))
                                    #     score[combo] = np.mean(ith_score)
                                    # except:
                                    #     score[combo] = np.nan

                                scores[smi] = score

                            df = pd.DataFrame.from_dict(scores, orient='index')
                            df.index = df.index.rename('smiles')
                            df = df.reset_index()
                            scores = df[pairs].to_numpy().astype(float)
                            scores[np.all(np.isnan(scores), axis=1)] = -100
                            df['mean_score'] = np.nanmean(
                                scores, axis=1)
                            df.to_csv(folder+'/scores'+str(mpi_ind) +
                                      '_'+target_name+'.csv', index=False)
                        else:
                            raise TypeError
                    else:
                        logging.warning(
                            '{} zero pickle length!'.format(folder))
            except AssertionError:
                logging.warning(
                    '{} length assertion error: {} vs {}'.format(
                        folder, len(real_smi), len(twobody_dist)))

                continue
            except FileNotFoundError as not_found:
                logging.warning('{}: {} not found error'.format(
                    folder, not_found.filename))
                continue
            except TypeError as e:
                logging.warning('{}: pickle type error {}'.format(
                    folder, e))
                continue
        logging.info('{} Done!'.format(folder))


def main(args):
    tranch_file = open(args.fname, 'r')
    # tranches = [x.split('/')[-1][:-4] for x in tranch_file.read().splitlines()]
    tranches = [x for x in tranch_file.read().splitlines()]

    data_dir = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/data/'

    folder_names = [data_dir + x for x in tranches]

    pickle_dir = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/pickles/'
    # kdes = {'mpro': 'kde_dict_mpro.pickle',
    #         'mac1': 'kde_dict_mac1.pickle',
    #         'dpp11': 'kde_dict_dpp11.pickle'}
    kdes = {'mpro': 'kde_dict_spl_mpro.pickle',
            'mac1': 'kde_dict_spl_mac1.pickle',
            'dpp11': 'kde_dict_spl_dpp11.pickle'}
    kde_dict = pickle.load(open(pickle_dir+kdes[args.target], 'rb'))

    score_mpi_folders(folder_names, kde_dict, args.target)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-fname', type=str, required=True,
                        help='Location of file containing names of EnamineREAL tranches to score.')
    parser.add_argument('-target', type=str, required=True,
                        help='Name of protein target to score.')
    args = parser.parse_args()

    main(args)
