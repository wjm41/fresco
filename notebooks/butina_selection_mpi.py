import argparse
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdDepictor, rdMolDescriptors, Lipinski
from rdkit.ML.Cluster.Butina import ClusterData

from frag_funcs import return_borders
from mpi4py import MPI

rdDepictor.SetPreferCoordGen(True)

logging.basicConfig(level=logging.INFO)


def remove_stereo(mol):
    Chem.rdmolops.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def main(args):
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    df = pd.read_csv(args.data_dir+'topN_taut_filtered_' +
                     args.target+'.csv')

    df.reset_index(drop=True, inplace=True)

    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df['nonchiral'] = df['mol'].apply(remove_stereo)
    df['fps'] = [rdMolDescriptors.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smi), 2, 2048, useFeatures=args.useFeatures) for smi in df["smiles"]]  # pharmacophore fingerprint
    fps = df['fps'].values

    n = len(df)
    d_mat = np.zeros(int(n*(n-1)/2))
    recv_dmat = np.zeros(int(n*(n-1)/2))

    my_border_low, my_border_high = return_borders(
        mpi_rank, len(df), mpi_size)
    if mpi_rank == 0:
        logging.info('Runing on {} MPI processes, each process handling {} molecules - len(d_mat) = {}'.format(
            mpi_size, my_border_high - my_border_low, len(d_mat)))
    n = 0
    my_len = 0
    start_ind = 0
    for i in tqdm(range(1, len(fps))):
        if i >= my_border_low and i < my_border_high:
            if my_len == 0:
                start_ind = n
            d_mat[n:n+i] = np.ones_like(fps[:i]) - \
                DataStructs.cDataStructs.BulkTanimotoSimilarity(
                    fps[i], fps[:i])
            my_len += i
        n += i

    d_mat = d_mat[start_ind:start_ind+my_len]

    if len(d_mat) != my_len:
        logging.warning('mpi_rank == {}, len(d_mat) = {}, my_len = {}'.format(
            mpi_rank, len(d_mat), my_len))
    assert len(d_mat) == my_len
    sendcounts = np.array(mpi_comm.gather(my_len, root=0))

    # vector gather
    mpi_comm.Gatherv(d_mat,
                     recvbuf=(recv_dmat, sendcounts), root=0)
    mpi_comm.Barrier()

    if mpi_rank == 0:
        if args.useFeatures:

            np.save('numpy/mpi_dmat_constrained_taut_' +
                    args.target+'.npy', d_mat)
        else:

            np.save('numpy/mpi_dmat_constrained_taut_' +
                    args.target+'_noFeatures.npy', d_mat)

        print(recv_dmat)
        clusters = ClusterData(recv_dmat, nPts=len(
            df), isDistData=True, distThresh=args.thresh, reordering=True)

        print('Number of '+args.target+' clusters: {} (from {} mols)'.format(
            len(clusters), len(df)))

        cluster_id_list = [0]*len(df)
        for idx, cluster in enumerate(clusters, 1):
            for member in cluster:
                cluster_id_list[member] = idx
        df.reset_index(drop=True, inplace=True)
        df['cluster'] = [x-1 for x in cluster_id_list]
        if args.useFeatures:
            df[['smiles', 'nonchiral', 'taut_smiles', 'cluster']].to_csv(
                args.data_dir+args.target+'_taut_clustered.csv', index=False)

        butina_df = df.iloc[[clusters[n][0]
                             for n in range(args.npicks)]].copy()
        butina_df['membership'] = [len(clusters[n])
                                   for n in range(args.npicks)]

        if args.useFeatures:
            butina_df[['smiles', 'nonchiral', 'taut_smiles', 'membership']].to_csv(
                args.data_dir+args.target+'_taut_picks_constrained_new.csv', index=False)
        else:
            butina_df[['smiles', 'nonchiral', 'taut_smiles', 'membership']].to_csv(
                args.data_dir+args.target+'_taut_noFeatures_picks_constrained_new.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains a Molecular Transformer on a reaction dataset')
    parser.add_argument('-data_dir', '--data_dir', type=str, default='/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/topN/',
                        help='Directory containing topN molecules.')
    parser.add_argument('-method', '--method', type=str,
                        help='Scoring method for the molecules.')
    parser.add_argument('-target', '--target', type=str,
                        help='Protein target.')
    parser.add_argument('--useFeatures', '--useFeatures', action='store_true', default=False,
                        help='Whether or not to use pharmacophore features in the fingerprint calculation.')
    parser.add_argument('-thresh', '--thresh', type=float, default=0.2,
                        help='Distance threshold = max distance from centroid in cluster.')
    parser.add_argument('-npicks', '--npicks', type=int, default=60,
                        help='Number of cluster centroids to pick.')
    args = parser.parse_args()

    main(args)
