import random
import os
import math
from tqdm import tqdm

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, MolToSmiles, AllChem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from rdkit import Geometry
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
from rdkit.RDPaths import RDDataDir

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

fdefFile = os.path.join(RDDataDir, 'BaseFeatures.fdef')
featFactory = ChemicalFeatures.BuildFeatureFactory(fdefFile)


def return_borders(index, dat_len, mpi_size):
    mpi_borders = np.linspace(0, dat_len, mpi_size + 1).astype('int')

    border_low = mpi_borders[index]
    border_high = mpi_borders[index+1]
    return border_low, border_high


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)


def return_random_dataframe(mols, interesting_pcores, hit=False, jiggle=False, jiggle_std=1.2):
    """
    retun dictionary of numpy arrays containing (x,y,z) of pharmacophore coordinates (averaged over atoms)
    """
    columns = ['mol_id', 'pcore', 'coord_x',
               'coord_y', 'coord_z', 'frag', 'active']

    pcore_df = pd.DataFrame(columns=columns)

    pcore_dict = {}
    for pcore in interesting_pcores:
        pcore_dict[pcore] = 0

    for i, mol in tqdm(enumerate(mols)):
        frag = True
        activity = False
        if hit:
            frag = False
            IC50 = mol[-1]

            if not math.isnan(IC50) and IC50 < 10:
                #             if not math.isnan(IC50):
                activity = True

        mol_df = pd.DataFrame(columns=columns)
        feats = featFactory.GetFeaturesForMol(mol[0])

        num_atoms = len(mol[0].GetAtoms())
        all_atoms = list(range(num_atoms))

        counter = 0
        for feat in feats:
            feat_fam = feat.GetFamily()

            if feat_fam in interesting_pcores:
                pcore_dict[feat_fam] += 1

        # include all atoms
        for id in all_atoms:
            atom = (mol[1:][id])
            xyz = np.array(atom[1])
            feat_df = pd.DataFrame(
                {
                    'pcore': 'NaN',
                    'mol_id': [i],
                    'coord_x': [xyz[0]],
                    'coord_y': [xyz[1]],
                    'coord_z': [xyz[2]],
                    'frag': frag,
                    'active': activity
                })
            # loop over features and append to mol_df
            mol_df = pd.concat([mol_df, feat_df])

        # loop over molecules and append to pcore_df
        pcore_df = pd.concat([pcore_df, mol_df])

    pcore_df['frag'] = pcore_df['frag'].astype(bool)
    pcore_df['active'] = pcore_df['active'].astype(bool)
    pcore_df.reset_index(inplace=True, drop=True)

    # sample n pcores from all atoms
    partition_indices = list(pcore_dict.values())
    pcore_df = pcore_df.sample(n=sum(partition_indices), replace=False)
    pcore_df.reset_index(inplace=True, drop=True)
    counter = 0
    for i, pcore in enumerate(pcore_dict):
        pcore_df.iloc[counter:partition_indices[i]+counter]['pcore'] = pcore
        counter += partition_indices[i]

    return pcore_df


def return_pcore_dataframe(mols, interesting_pcores, hit=False, jiggle=False, jiggle_std=1.2):
    """
    retun dictionary of numpy arrays containing (x,y,z) of pharmacophore coordinates (averaged over atoms)
    """
    columns = ['smiles', 'pcore', 'coord_x',
               'coord_y', 'coord_z', 'frag', 'active', 'IC50']

    mol_dfs = [None]*len(mols)

    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        frag = True
        activity = False
        if hit:
            frag = False
            IC50 = mol[-1]

            if not math.isnan(IC50) and IC50 < 5:
                #             if not math.isnan(IC50):
                activity = True
        else:
            IC50 = np.nan

        feat_dfs = []  # empty dataframe
        feats = featFactory.GetFeaturesForMol(mol[0])

        for feat in feats:
            feat_fam = feat.GetFamily()

            if feat_fam in interesting_pcores:
                atom_ids = feat.GetAtomIds()
                xyz = np.empty((len(atom_ids), 3))

                for j, id in enumerate(atom_ids):
                    atom = (mol[1:][id])
                    xyz[j] = np.array(atom[1])
                    if jiggle:
                        # uncertainty from Xray crystallography
                        xyz[j] += np.random.normal(scale=jiggle_std)

                xyz = np.mean(xyz, axis=0)  # mean all aromatic rings!
                feat_df = pd.DataFrame(
                    {
                        'pcore': feat_fam,
                        'smiles': MolToSmiles(mol[0]),
                        'mol_id': [i],
                        'coord_x': [xyz[0]],
                        'coord_y': [xyz[1]],
                        'coord_z': [xyz[2]],
                        'frag': frag,
                        'active': activity,
                        'IC50': IC50
                    })

                # loop over features and append to mol_df
                feat_dfs.append(feat_df)
        if feat_dfs != []:
            mol_dfs[i] = pd.concat(feat_dfs)

    mol_dfs = [mol_df for mol_df in mol_dfs if mol_df is not None]

    if mol_dfs != []:
        # loop over molecules and append to pcore_df
        pcore_df = pd.concat(mol_dfs)

        pcore_df['frag'] = pcore_df['frag'].astype(bool)
        pcore_df['active'] = pcore_df['active'].astype(bool)
        pcore_df.reset_index(inplace=True, drop=True)
        return pcore_df
    else:
        return None


def get_pair_distances(pcore_df, pcore_a, pcore_b, frag=False, active=False):
    '''
    calculates the distribution of pair distances between pcore_a in either hits or frags with pcore_b

    frag argument is to specify calculation of inter-frag distributions which require avoidance of intra-frag counting
    '''

    df_a = pcore_df[pcore_df['pcore'] == pcore_a]
    id_a = df_a['mol_id']
    coords_a = df_a[['coord_x', 'coord_y', 'coord_z']].to_numpy()

    df_b = pcore_df[pcore_df['pcore'] == pcore_b]
    id_b = df_b['mol_id']
    coords_b = df_b[['coord_x', 'coord_y', 'coord_z']].to_numpy()

    all_distances = [None] * len(set(df_a['mol_id']))
    for j, i in enumerate(set(df_a['mol_id'])):
        try:
            xyz_i = coords_a[id_a == i]  # DOUBLE COUNTING

            if frag:
                xyz_j = coords_b[id_b != i]  # INTER-FRAG COUNTING
            else:
                xyz_j = coords_b[id_b == i]  # INTRA-MOLECULE COUNTING

            distance = xyz_i[:, np.newaxis] - xyz_j

            distance = np.linalg.norm(distance, axis=2)

            distance = distance.flatten()

            all_distances[j] = distance
        except Exception as ex:
            print(ex)
            pass
    all_distances = [x for x in all_distances if x is not None]
    if all_distances == []:
        all_distances = [np.array([])]
    return all_distances


def get_trip_distances(pcore_df, pcore_a, pcore_b, pcore_c, frag=False, active=False):

    df_a = pcore_df[pcore_df['pcore'] == pcore_a]
    df_b = pcore_df[pcore_df['pcore'] == pcore_b]
    df_c = pcore_df[pcore_df['pcore'] == pcore_c]

    all_distances = []

    counter = -1
    for i in set(df_a['mol_id']):
        xyz_i = df_a[df_a['mol_id'] == i][['coord_x',
                                           'coord_y', 'coord_z']].to_numpy()  # DOUBLE COUNTING
        nx = xyz_i.shape[0]

        i_values = df_a[df_a['mol_id'] == i]['mol_id'].values

        if frag:
            xyz_j = df_b[['coord_x', 'coord_y', 'coord_z']].to_numpy()

            ny = xyz_j.shape[0]

            j_values = df_b['mol_id'].values

            r12 = np.linalg.norm(
                xyz_i[:, np.newaxis] - xyz_j, axis=2).flatten()
            for j in set(df_b['mol_id']):
                xyz_k = df_c[['coord_x', 'coord_y', 'coord_z']].to_numpy()
                nz = xyz_k.shape[0]

                k_values = df_c['mol_id'].values

                # inter-frag counting
                ijk = np.array(
                    [[x, y, z] for x in i_values for y in j_values for z in k_values]).T
                mask = np.logical_or(np.logical_or(np.equal(ijk[0], ijk[1]), np.equal(
                    ijk[1], ijk[2])), np.equal(ijk[2], ijk[0]))

                r23 = np.linalg.norm(
                    xyz_j[:, np.newaxis] - xyz_k, axis=2).flatten()
                r31 = np.linalg.norm(
                    xyz_k[:, np.newaxis] - xyz_i, axis=2).T  # .flatten()

                distance_matrix = np.empty((3, nx*ny*nz))
                distance_matrix[0] = np.repeat(r12, repeats=nz)
                distance_matrix[1] = np.tile(r23, reps=nx)

                z_axis = 0
                for x in range(r31.shape[0]):
                    if x == 0:
                        z_axis = np.tile(r31[x], reps=ny)
                    else:
                        z_axis = np.hstack([z_axis, np.tile(r31[x], reps=ny)])

                distance_matrix[2] = z_axis

                assert ijk.shape == distance_matrix.shape
                all_distances.append(distance_matrix.T[~mask].T)

        else:
            xyz_j = df_b[df_b['mol_id'] == i][['coord_x', 'coord_y',
                                               'coord_z']].to_numpy()  # INTRA-MOLECULE COUNTING
            xyz_k = df_c[df_c['mol_id'] == i][['coord_x', 'coord_y',
                                               'coord_z']].to_numpy()  # INTRA-MOLECULE COUNTING

            ny = xyz_j.shape[0]
            nz = xyz_k.shape[0]

            r12 = np.linalg.norm(
                xyz_i[:, np.newaxis] - xyz_j, axis=2).flatten()
            r23 = np.linalg.norm(
                xyz_j[:, np.newaxis] - xyz_k, axis=2).flatten()
            r31 = np.linalg.norm(
                xyz_k[:, np.newaxis] - xyz_i, axis=2).T  # .flatten()

            distance_matrix = np.empty((3, nx*ny*nz))
            distance_matrix[0] = np.repeat(r12, repeats=nz)
            distance_matrix[1] = np.tile(r23, reps=nx)

            z_axis = 0
            for x in range(r31.shape[0]):
                if x == 0:
                    z_axis = np.tile(r31[x], reps=ny)
                else:
                    z_axis = np.hstack([z_axis, np.tile(r31[x], reps=ny)])

            distance_matrix[2] = z_axis
            all_distances.append(distance_matrix)

    return all_distances


def fit_pair_kde(data):
    params = {'bandwidth': np.logspace(-3, 3, 50)}

    grid = GridSearchCV(KernelDensity(kernel='gaussian', rtol=1e-4), params)
    grid.fit(data.reshape(-1, 1))

    kde = grid.best_estimator_
#     print(kde.get_params())
    return kde


def fit_trip_kde(data):
    #     params = {'bandwidth': np.logspace(-3, 2, 20)}

    #     grid = GridSearchCV(KernelDensity(kernel='gaussian', rtol=1e-4), params)
    #     grid.fit(data.T)

    #     kde = grid.best_estimator_
    kde = KernelDensity(kernel='gaussian', rtol=1e-4)
    kde.fit(data.T)
#     print(kde.get_params())
    return kde


def clean_smiles(smi_list, interesting_pcores=['Donor', 'Acceptor', 'Aromatic']):
    #     f = open(smi_file, 'r')
    #     smi_list = f.read().splitlines()
    #     f.close()
    df = pd.DataFrame(smi_list, columns=['smi'])
    df['keep'] = False
    # keep_list = [False]*len(smi_list)
    # mol_ids = []
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smi'])
        feats = featFactory.GetFeaturesForMol(mol)

        for feat in feats:
            if not row['keep']:
                feat_fam = feat.GetFamily()

                if feat_fam in interesting_pcores:
                    df.loc[i, 'keep'] = True
                    # keep_list[i] = True
                    # mol_ids.append(i)

    # mol_ids = list(set(mol_ids))
    # df['keep'] = keep_list
    df = df[df['keep']]
    smi_list = df['smi'].values
    # smi_list = [smi_list[i] for i in mol_ids]
    return smi_list


def score_dist(kde, dist):
    if len(dist):  # non-zero length

        # log-prob (larger = higher prob)
        # score using scipy spline!
        score = kde(dist.reshape(-1, 1))

        # score = kde.score_samples(dist.reshape(-1,1))
        score = np.mean(score)

        return score
    else:
        return np.nan


def check_mol_lead_like(mol: Chem.rdchem.Mol) -> bool:

    if mol.GetNumHeavyAtoms() > 25 or mol.GetNumHeavyAtoms() < 8:
        return False
    elif Descriptors.MolWt(mol) > 400 or Descriptors.MolWt(mol) < 109:
        return False
    elif Descriptors.MolLogP(mol) > 3.5 or Descriptors.MolLogP(mol) < -2.7:
        return False
    elif Descriptors.TPSA(mol) > 179 or Descriptors.TPSA(mol) < 3:
        return False
    elif Lipinski.NumHAcceptors(mol) > 8 or Lipinski.NumHAcceptors(mol) < 1:
        return False
    elif Lipinski.NumHDonors(mol) > 4:
        return False
    elif rdMolDescriptors.CalcNumRotatableBonds(mol) > 10:
        return False
    else:
        return True


def calculate_enrichment_for_df(score_df, n=10, score='score', index='active', log=False, ascending=True):

    df = score_df[score_df[score].notna()]
    orig_prop = len(df[df[index]])/len(df)
    if log:
        print('orig proportion of {}: {:.3f}%'.format(index, orig_prop*100))

    sorted_df = df.sort_values(by=score, ascending=ascending).iloc[:n]
    new_prop = len(sorted_df[sorted_df[index]])/len(sorted_df)
    if log:
        print('N = {}, n_hits = {}, new proportion of {}: {:.3f}%'.format(
            n, len(sorted_df[sorted_df[index]]), index, new_prop*100))
    EF = new_prop/orig_prop
    return EF
