import os
import math
import logging
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, MolToSmiles, AllChem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from rdkit import Geometry
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
from rdkit.RDPaths import RDDataDir

def return_default_pharmacophores() -> List[str]:
    pcores_of_interest = ['Donor', 'Acceptor', 'Aromatic']
    return pcores_of_interest

def return_default_pharmacophore_pairs() -> List[str]:
    pairs = ['Donor-Aromatic',
        'Aromatic-Acceptor',
        'Aromatic-Aromatic',
        'Donor-Donor',
        'Donor-Acceptor',
        'Acceptor-Acceptor']
    return pairs

def return_rdkit_feat_factory():
    fdefFile = os.path.join(RDDataDir, 'BaseFeatures.fdef')
    featFactory = ChemicalFeatures.BuildFeatureFactory(fdefFile)
    return featFactory


def return_pcore_dataframe_from_single_rdkit_molecule(mol: Mol, mol_id:int = 0, pcores_of_interest: List=None, featFactory=None):
    
    if pcores_of_interest is None:
        pcores_of_interest = return_default_pharmacophores()
    
    if featFactory is None:
        featFactory = return_rdkit_feat_factory()
        

    atom_coordinates = mol.GetConformer().GetPositions()
    dictionary_of_pcores_and_atom_ids = featFactory.GetFeaturesForMol(mol)

    df_of_pcores_for_this_mol = []
    for pcore in dictionary_of_pcores_and_atom_ids:
        pharmacophore_name = pcore.GetFamily()

        if pharmacophore_name in pcores_of_interest:
            atom_ids = pcore.GetAtomIds()
            xyz = []

            for id in atom_ids:
                xyz.append(np.array(atom_coordinates[id]))

            xyz = np.vstack(xyz)
            assert xyz.shape == (len(atom_ids), 3)
            xyz = np.mean(xyz, axis=0)  # mean all aromatic rings!
            df_for_this_pcore = pd.DataFrame(
                {
                    'pcore': pharmacophore_name,
                    'smiles': MolToSmiles(mol),
                    'mol_id': mol_id, 
                    'coord_x': [xyz[0]],
                    'coord_y': [xyz[1]],
                    'coord_z': [xyz[2]],
                })
            df_of_pcores_for_this_mol.append(df_for_this_pcore)
            
    if len(df_of_pcores_for_this_mol) == 0:
        logging.warning(f'No pharmacophores found for molecule {MolToSmiles(mol)}!')
        return None
    else:
        return pd.concat(df_of_pcores_for_this_mol)

def return_pcore_dataframe_for_list_of_mols(mols: List[Mol], pcores_of_interest: List=None, featFactory=None) -> pd.DataFrame:
    """
    retun dictionary of numpy arrays containing (x,y,z) of pharmacophore coordinates (averaged over atoms)
    """
    
    if pcores_of_interest is None:
        pcores_of_interest = return_default_pharmacophores()
    
    if featFactory is None:
        featFactory = return_rdkit_feat_factory()

    dfs_for_this_list_of_mols = []

    for mol_id, mol in tqdm(enumerate(mols), total=len(mols)):
        
        df_for_this_mol = return_pcore_dataframe_from_single_rdkit_molecule(mol, mol_id = mol_id, pcores_of_interest=pcores_of_interest, featFactory=featFactory)

        if df_for_this_mol is not None:
            dfs_for_this_list_of_mols.append(df_for_this_mol)

    if len(dfs_for_this_list_of_mols) == 0:
        logging.warning(f'No pharmacophores found for this list of molecules!')
        return None
    else:
        return pd.concat(dfs_for_this_list_of_mols)        
    
def calculate_pairwise_distances_between_pharmacophores_for_fragment_ensemble(df_of_frag_ensemble_pcores, pcore_a, pcore_b):
    '''
    calculates the distribution of pair distances between pcore_a in either hits or frags with pcore_b

    frag argument is to specify calculation of inter-frag distributions which require avoidance of intra-frag counting
    '''

    df_pcore_a = df_of_frag_ensemble_pcores.query('pcore == @pcore_a')
    id_a = df_pcore_a['mol_id']
    coords_a = df_pcore_a[['coord_x', 'coord_y', 'coord_z']].to_numpy()

    df_pcore_b = df_of_frag_ensemble_pcores.query('pcore == @pcore_b')
    id_b = df_pcore_b['mol_id']
    coords_b = df_pcore_b[['coord_x', 'coord_y', 'coord_z']].to_numpy()


    if len(df_pcore_a) == 0:
        raise ValueError(f'No pharmacophores found for pharmacophore {pcore_a} !')
    if len(df_pcore_b) == 0:
        raise ValueError(f'No pharmacophores found for pharmacophore {pcore_b} !')
    
    distances_for_all_pairs = []
    
    for id_of_frag_with_pcore_a in id_a.unique().astype(int):
        xyz_i = coords_a[id_a == id_of_frag_with_pcore_a, :]  # DOUBLE COUNTING

        xyz_j = coords_b[id_b != id_of_frag_with_pcore_a, :]  # Don't count distances within the same fragment

        if len(xyz_j) > 0:
            delta_coordinates_between_pcores = xyz_i[:, np.newaxis] - xyz_j

            distances_for_this_pair = np.linalg.norm(delta_coordinates_between_pcores, axis=2)

            distances_for_this_pair = distances_for_this_pair.flatten()
        else: 
            logging.warning(f'Only intra-fragment distance found for {id_of_frag_with_pcore_a} with pharmacophore {pcore_a}!')
            
        distances_for_all_pairs.append(distances_for_this_pair)
        
    if len(distances_for_all_pairs) == 0:
        distances_for_all_pairs = [np.array([])]
    return np.hstack(distances_for_all_pairs)


def calculate_pairwise_distances_between_pharmacophores_for_single_molecule(pcore_df, pcore_a, pcore_b, frag=False):
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