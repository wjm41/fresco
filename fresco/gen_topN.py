from frag_funcs import return_borders

import argparse
import os
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
# from mpi4py import MPI
from rdkit.Chem import Descriptors, MolFromSmiles, Lipinski, rdMolDescriptors

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.INFO)


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
    elif Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol) > 8:
        row['keep'] = False
    elif rdMolDescriptors.CalcNumRotatableBonds(mol) > 10:
        row['keep'] = False

    return row


def proc_df(path, N):
    df = pd.read_csv(path)
    df = df[df['mean_score'].notnull()]
    df = df[df['smiles'].notnull()]
    df = df.drop_duplicates(subset=['smiles'], keep='first')
    if len(df) != 0:
        df = df.apply(filter_physchem, axis=1)
        df = df[df['keep']]
        df = df.nlargest(n=min(args.N, len(df)), columns='mean_score')
    return df


def main(args):

    # mpi_comm = MPI.COMM_WORLD
    # mpi_rank = mpi_comm.Get_rank()
    # mpi_size = mpi_comm.Get_size()

    folder_names = open(args.fname, 'r')
    my_dirs = [x for x in folder_names.read().splitlines()]

    data_dir = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/data/'

    # my_dirs = [data_dir + x for x in folder_names]
    # all_folds = open(project_dir+'/folds/all_folds.txt',
    #                  'r').read().splitlines()

    # my_border_low, my_border_high = return_borders(
    #     mpi_rank, len(all_folds), mpi_size)

    # my_dirs = all_folds[my_border_low: my_border_high]

    for folder in tqdm(my_dirs, total=len(my_dirs)):

        if not os.path.isfile(data_dir+folder+'/topN_'+args.target+'.csv'):
            # pairs = [file for file in os.listdir(
            #     data_dir+folder) if "pairs_mpi_" in file]
            scores = [file for file in os.listdir(
                data_dir+folder) if '_'+args.target in file and ".csv" in file]

            # score folders
            if len(scores) >= 1:
                df_all = []
                for score_csv in tqdm(scores):
                    if os.stat(data_dir+folder+'/' + score_csv).st_size != 0:
                        df = proc_df(data_dir+folder+'/'+score_csv, N=args.N)
                        df_all.append(df)

                df_all = pd.concat(df_all).drop_duplicates(
                    subset=['smiles'], keep='first')
                df_all = df_all.nlargest(
                    n=min(args.N, len(df_all)), columns='mean_score')
                df_all.to_csv(data_dir+folder+'/topN_' +
                              args.target+'.csv', index=False)
                logging.info('{} done!'.format(folder))
            else:
                logging.warning('{} has no scores!'.format(folder))
        else:
            logging.info('{} already searched!'.format(folder))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-fname', type=str, required=True,
                        help='Location of file containing names of EnamineREAL tranches to score.')
    parser.add_argument('-N', type=int, required=True,
                        help='Number of top-N candidates to return')
    parser.add_argument('-target', type=str, required=True,
                        help='Name of protein target to score.')
    args = parser.parse_args()

    main(args)
