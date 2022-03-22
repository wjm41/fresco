import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdDepictor, rdMolDescriptors, Lipinski
from rdkit.ML.Cluster.Butina import ClusterData

from frag_funcs import return_borders

rdDepictor.SetPreferCoordGen(True)


def remove_stereo(mol):
    Chem.rdmolops.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def main(args):

    # df = pd.read_csv(args.data_dir+'topN_'+args.method +
    #                  '_filtered_'+args.target+'.csv')
    df = pd.read_csv(args.data_dir+'topN_taut_filtered_'+args.target+'.csv')

    df.reset_index(drop=True, inplace=True)
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df['nonchiral'] = df['mol'].apply(remove_stereo)
    df['fps'] = [rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, 2, 2048, useFeatures=args.useFeatures) for mol in df["mol"]]  # pharmacophore fingerprint

    fps = df['fps'].values

    n = len(df)
    d_mat = np.zeros(int(n*(n-1)/2))

    n = 0
    for i in tqdm(range(1, len(fps))):
        d_mat[n:n+i] = np.ones_like(fps[:i]) - \
            DataStructs.cDataStructs.BulkTanimotoSimilarity(
                fps[i], fps[:i])
        n += i

    if args.useFeatures:
        # np.save('numpy/dmat_constrained_'+args.method +
        #         '_'+args.target+'.npy', d_mat)
        np.save('numpy/dmat_constrained_taut_'+args.target+'.npy', d_mat)
    else:
        # np.save('numpy/dmat_constrained_'+args.method+'_' +
        # args.target+'_noFeatures.npy', d_mat)
        np.save('numpy/dmat_constrained_taut_' +
                args.target+'_noFeatures.npy', d_mat)

    clusters = ClusterData(d_mat, nPts=len(
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

    butina_df = df.iloc[[clusters[n][0] for n in range(args.npicks)]]
    butina_df['membership'] = [len(clusters[n]) for n in range(args.npicks)]

    if args.useFeatures:
        butina_df[['smiles', 'nonchiral', 'membership']].to_csv(
            args.data_dir+args.target+'_taut_picks_constrained_new.csv', index=False)
    else:
        butina_df[['smiles', 'nonchiral', 'membership']].to_csv(
            args.data_dir+args.target+'_taut_noFeatures_picks_constrained_new.csv', index=False)
        # args.data_dir+args.target+'_'+args.method+'_noFeatures_picks_constrained.csv', index=False)

    # display(HTML('<b>'+args.target+' Butina</b>'))
    # mols2grid.display(butina_df, template="pages", smiles_col='smiles', mol_col='mol',
    #                   n_rows=15, n_cols=4, subset=["img"], tooltip=['smiles', 'membership'],
    #                   maxMols=60, size=(300, 150), selection=False)


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
