import random
import os.path
import math

from tqdm import tqdm
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, MolToSmiles, AllChem
from rdkit import Geometry
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
from rdkit.RDPaths import RDDataDir

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

fdefFile = os.path.join(RDDataDir,'BaseFeatures.fdef')
featFactory = ChemicalFeatures.BuildFeatureFactory(fdefFile)


def return_borders(index, dat_len, mpi_size):
    mpi_borders = np.linspace(0, dat_len, mpi_size + 1).astype('int')

    border_low = mpi_borders[index]
    border_high = mpi_borders[index+1]
    return border_low, border_high

def return_random_dataframe(mols, interesting_pcores, weights=None, hit=False, jiggle=False, jiggle_std=1.2):
    """ 
    retun dictionary of numpy arrays containing (x,y,z) of pharmacophore coordinates (averaged over atoms)
    """
    columns = ['smiles', 'mol_id', 'pcore', 'coord_x', 'coord_y', 'coord_z', 'weight', 'frag', 'active']
    if weights is None:
        weights = [1.0]*len(mols)
    pcore_df = pd.DataFrame(columns=columns)
    
    pcore_dict = {}
    for pcore in interesting_pcores:
        pcore_dict[pcore] = 0
    
    for i, mol in enumerate(mols):
        frag = True
        activity = False
        if hit:
            frag = False
            IC50 = mol[-1]

            if not math.isnan(IC50) and IC50<10:
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
                pcore_dict[feat_fam]+=1

        # include all atoms 
        for id in all_atoms:
                atom = (mol[1:][id])
                xyz = np.array(atom[1])
                feat_df = pd.DataFrame(
                            {
                                'pcore': 'NaN',
                                'smiles': MolToSmiles(mol[0]),
                                'mol_id': [i],
                                'coord_x': [xyz[0]],
                                'coord_y': [xyz[1]],
                                'coord_z': [xyz[2]],
                                'weight': weights[i],
                                'frag': frag,
                                'active': activity
                            }) 
                mol_df = pd.concat([mol_df, feat_df]) # loop over features and append to mol_df
                
        pcore_df = pd.concat([pcore_df, mol_df]) # loop over molecules and append to pcore_df

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
        counter+=partition_indices[i]    
    
    return pcore_df

def return_pcore_dataframe(mols, interesting_pcores, weights=None, hit=False, jiggle=False, jiggle_std=1.2, threshold=5):
    """ 
    retun dictionary of numpy arrays containing (x,y,z) of pharmacophore coordinates (averaged over atoms)
    """
    columns = ['smiles', 'pcore', 'coord_x', 'coord_y', 'coord_z', 'weight', 'frag', 'active', 'IC50']

    if weights is None:
        weights = [1.0]*len(mols)
        
    mol_dfs = [None]*len(mols)

    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        frag = True
        activity = False
        if hit:
            frag = False
            IC50 = mol[-1]

            if not math.isnan(IC50) and IC50<threshold:
#             if not math.isnan(IC50): 
                activity = True
        else:
            IC50 = np.nan

        feat_dfs = [] #empty dataframe
        feats = featFactory.GetFeaturesForMol(mol[0])

        for feat in feats:
            feat_fam = feat.GetFamily()

            if feat_fam in interesting_pcores:
                atom_ids = feat.GetAtomIds()
                xyz = np.empty((len(atom_ids),3))

                for j,id in enumerate(atom_ids):
                    atom = (mol[1:][id])
                    xyz[j] = np.array(atom[1])
                    if jiggle:
                        xyz[j]+= np.random.normal(scale=jiggle_std) # uncertainty from Xray crystallography

                xyz = np.mean(xyz, axis=0) # mean all aromatic rings!
                feat_df = pd.DataFrame(
                            {
                                'pcore': feat_fam,
                                'smiles': MolToSmiles(mol[0]),
                                'mol_id': [i],
                                'coord_x': [xyz[0]],
                                'coord_y': [xyz[1]],
                                'coord_z': [xyz[2]],
                                'weight': weights[i],
                                'frag': frag,
                                'active': activity,
                                'IC50': IC50
                            })

                feat_dfs.append(feat_df) # loop over features and append to mol_df
        if feat_dfs!=[]:
            mol_dfs[i] = pd.concat(feat_dfs)
            
    mol_dfs = [mol_df for mol_df in mol_dfs if mol_df is not None]

    if mol_dfs!=[]:
        pcore_df = pd.concat(mol_dfs) # loop over molecules and append to pcore_df

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
    
    df_a = pcore_df[pcore_df['pcore']==pcore_a]
    df_b = pcore_df[pcore_df['pcore']==pcore_b]

    all_distances = []
    all_weights = []
    for smi in df_a['smiles']:
        try:
            xyz_i = df_a[df_a['smiles']==smi][['coord_x','coord_y','coord_z']].to_numpy() # DOUBLE COUNTING
            w_i = df_a[df_a['smiles']==smi][['weight']].to_numpy()
            if frag:
                xyz_j = df_b[df_b['smiles']!=smi][['coord_x','coord_y','coord_z']].to_numpy() # INTER-FRAG COUNTING
                w_j = df_b[df_b['smiles']!=smi][['weight']].to_numpy()
            else:
                xyz_j = df_b[df_b['smiles']==smi][['coord_x','coord_y','coord_z']].to_numpy() # INTRA-MOLECULE COUNTING
                w_j = df_b[df_b['smiles']==smi][['weight']].to_numpy()
            distance = xyz_i[:, np.newaxis] - xyz_j
            distance = np.linalg.norm(distance, axis=2)
            
            weights = w_i[:, np.newaxis] * w_j
            
            distance = distance.flatten()
            weights = weights.flatten()
            
            assert distance.shape == weights.shape
            
            all_distances.append(distance)
            all_weights.append(weights)
            
        except Exception as ex:
            print(ex)
            pass
        
    if all_distances==[]:
        all_distances = [np.array([])]
        all_weights = [np.array([])]
        
    return all_distances, all_weights

def get_trip_distances(pcore_df, pcore_a, pcore_b, pcore_c, frag=False, active=False):
    
    df_a = pcore_df[pcore_df['pcore']==pcore_a]
    df_b = pcore_df[pcore_df['pcore']==pcore_b]
    df_c = pcore_df[pcore_df['pcore']==pcore_c]
    
    all_distances = []
            
    counter=-1
    for i in set(df_a['mol_id']):
        xyz_i = df_a[df_a['mol_id']==i][['coord_x','coord_y','coord_z']].to_numpy() # DOUBLE COUNTING
        nx = xyz_i.shape[0]
        
        i_values = df_a[df_a['mol_id']==i]['mol_id'].values
        
        if frag:
            xyz_j = df_b[['coord_x','coord_y','coord_z']].to_numpy()
            
            ny = xyz_j.shape[0]
        
            j_values = df_b['mol_id'].values
            
            
            r12 = np.linalg.norm(xyz_i[:, np.newaxis] - xyz_j, axis=2).flatten()
            for j in set(df_b['mol_id']):
                xyz_k = df_c[['coord_x','coord_y','coord_z']].to_numpy()
                nz = xyz_k.shape[0]
    
                k_values = df_c['mol_id'].values
                
                # inter-frag counting
                ijk = np.array([[x, y, z] for x in i_values for y in j_values for z in k_values]).T
                mask = np.logical_or(np.logical_or(np.equal(ijk[0], ijk[1]), np.equal(ijk[1], ijk[2])), np.equal(ijk[2], ijk[0]))    
                
                r23 = np.linalg.norm(xyz_j[:, np.newaxis] - xyz_k, axis=2).flatten()
                r31 = np.linalg.norm(xyz_k[:, np.newaxis] - xyz_i, axis=2).T#.flatten()
                
                distance_matrix = np.empty((3, nx*ny*nz))
                distance_matrix[0] = np.repeat(r12, repeats=nz)
                distance_matrix[1] = np.tile(r23, reps=nx)
                
                z_axis = 0
                for x in range(r31.shape[0]):
                    if x==0:
                        z_axis = np.tile(r31[x], reps=ny)
                    else:
                        z_axis = np.hstack([z_axis, np.tile(r31[x], reps=ny)])
                
                distance_matrix[2] = z_axis
                
                assert ijk.shape == distance_matrix.shape
                all_distances.append(distance_matrix.T[~mask].T)

        else:
            xyz_j = df_b[df_b['mol_id']==i][['coord_x','coord_y','coord_z']].to_numpy() # INTRA-MOLECULE COUNTING        
            xyz_k = df_c[df_c['mol_id']==i][['coord_x','coord_y','coord_z']].to_numpy() # INTRA-MOLECULE COUNTING
            
            ny = xyz_j.shape[0]
            nz = xyz_k.shape[0]
            
            r12 = np.linalg.norm(xyz_i[:, np.newaxis] - xyz_j, axis=2).flatten()
            r23 = np.linalg.norm(xyz_j[:, np.newaxis] - xyz_k, axis=2).flatten()
            r31 = np.linalg.norm(xyz_k[:, np.newaxis] - xyz_i, axis=2).T#.flatten()
            
            distance_matrix = np.empty((3, nx*ny*nz))
            distance_matrix[0] = np.repeat(r12, repeats=nz)
            distance_matrix[1] = np.tile(r23, reps=nx)
            
            z_axis = 0
            for x in range(r31.shape[0]):
                if x==0:
                    z_axis = np.tile(r31[x], reps=ny)
                else:
                    z_axis = np.hstack([z_axis, np.tile(r31[x], reps=ny)])

            distance_matrix[2] = z_axis
            all_distances.append(distance_matrix)

    return all_distances

def fit_pair_kde(data, weight=None, top=3, bottom=-3,  n=500):
    params = {'bandwidth': np.logspace(bottom, top, n)}
    
    grid = GridSearchCV(KernelDensity(kernel='gaussian', rtol=1e-4), params, n_jobs=-1)
    
    if weight is None:
        grid.fit(data.reshape(-1,1))
    else:
        grid.fit(data.reshape(-1,1), sample_weight = weight) 
    
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

def clean_smiles(smi_list, interesting_pcores = ['Donor', 'Acceptor', 'Aromatic']):
#     f = open(smi_file, 'r')
#     smi_list = f.read().splitlines()
#     f.close()
    df = pd.DataFrame(smi_list, columns=['smi'])
    
    keep_list = [False]*len(smi_list)
    for i, smi in enumerate(smi_list):
        mol = Chem.MolFromSmiles(smi)
        feats = featFactory.GetFeaturesForMol(mol)

        for feat in feats:
            if not keep_list[i]:
                feat_fam = feat.GetFamily()

                if feat_fam in interesting_pcores:
                    keep_list[i] = True
    df['keep'] = keep_list
    df = df[df['keep']]
    smi_list = df['smi'].values
    return smi_list

