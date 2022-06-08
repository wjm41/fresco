import os
import math
import logging
from collections import Counter
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


def return_pcore_dataframe_from_single_rdkit_molecule(mol: Mol, mol_id: int = 0, pcores_of_interest: List = None, featFactory=None):

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
        logging.warning(
            f'No pharmacophores found for molecule {MolToSmiles(mol)}!')
        return None
    else:
        return pd.concat(df_of_pcores_for_this_mol)


def calculate_frequency_weights_for_duplicate_fragments(smiles_list: List[str]):
    
    frequency_counts_per_smiles = Counter(smiles_list)

    weights = [1/frequency_counts_per_smiles[smi] for smi in smiles_list]
    
    return weights
    

def return_pcore_dataframe_for_list_of_mols(mols: List[Mol], smiles_list: List[str] = None, pcores_of_interest: List = None, featFactory=None) -> pd.DataFrame:
    """
    retun dictionary of numpy arrays containing (x,y,z) of pharmacophore coordinates (averaged over atoms)
    """

    if smiles_list is None:
        smiles_list = [MolToSmiles(mol) for mol in mols]
    else:
        assert len(smiles_list) == len(mols)

    if pcores_of_interest is None:
        pcores_of_interest = return_default_pharmacophores()

    if featFactory is None:
        featFactory = return_rdkit_feat_factory()

    dfs_for_this_list_of_mols = []
    weights = calculate_frequency_weights_for_duplicate_fragments(smiles_list)
    
    for mol_id, mol in tqdm(enumerate(mols), total=len(mols)):

        df_for_this_mol = return_pcore_dataframe_from_single_rdkit_molecule(
            mol, mol_id=mol_id, pcores_of_interest=pcores_of_interest, featFactory=featFactory)
        df_for_this_mol['weight'] = weights[mol_id]
        
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
    smiles_a = df_pcore_a['smiles']
    weights_a = df_pcore_a['weight'].to_numpy().reshape(-1,1)
    coords_a = df_pcore_a[['coord_x', 'coord_y', 'coord_z']].to_numpy()

    df_pcore_b = df_of_frag_ensemble_pcores.query('pcore == @pcore_b')
    smiles_b = df_pcore_b['smiles']
    weights_b = df_pcore_b['weight'].to_numpy().reshape(-1,1)
    coords_b = df_pcore_b[['coord_x', 'coord_y', 'coord_z']].to_numpy()

    if len(df_pcore_a) == 0:
        raise ValueError(
            f'No pharmacophores found for pharmacophore {pcore_a} !')
    if len(df_pcore_b) == 0:
        raise ValueError(
            f'No pharmacophores found for pharmacophore {pcore_b} !')

    distances_for_all_pairs = []
    weights_for_all_pairs = []
    
    for smiles_of_frag_with_pcore_a in smiles_a.unique():
        from_the_same_fragment = smiles_a == smiles_of_frag_with_pcore_a
        xyz_i = coords_a[from_the_same_fragment, :]  # DOUBLE COUNTING
        w_i = weights_a[from_the_same_fragment, :]
        
        # Don't count distances within the same fragment
        from_a_different_fragment = smiles_b != smiles_of_frag_with_pcore_a
        xyz_j = coords_b[from_a_different_fragment, :]
        w_j = weights_b[from_a_different_fragment, :]

        if len(xyz_j) > 0:
            delta_coordinates_between_pcores = xyz_i[:, np.newaxis] - xyz_j
            
            distances_for_this_pair = np.linalg.norm(
                delta_coordinates_between_pcores, axis=2)

            distances_for_this_pair = distances_for_this_pair.flatten()
            
            weights_for_this_pair = w_i[:, np.newaxis] * w_j
            weights_for_this_pair = weights_for_this_pair.flatten()
            
            assert distances_for_this_pair.shape == weights_for_this_pair.shape
            
            distances_for_all_pairs.append(distances_for_this_pair)
            weights_for_all_pairs.append(weights_for_this_pair)
            
        else:
            logging.warning(
                f'Only intra-fragment distance found for {smiles_of_frag_with_pcore_a} with pharmacophore {pcore_a}!')
        
    if len(distances_for_all_pairs) == 0:
        distances_for_all_pairs = [np.array([])]
    return np.hstack(distances_for_all_pairs)


def calculate_pairwise_distances_between_pharmacophores_for_a_single_ligand(df_of_pcores_for_single_ligand, pcore_a, pcore_b):
    '''
    calculates the pairwise distance between pcore_a and pcore_b in the same ligand for a list of ligands
    '''

    assert len(df_of_pcores_for_single_ligand.mol_id.unique()
               ) == 1, 'This function is designed to work with a single ligand'

    df_pcore_a = df_of_pcores_for_single_ligand.query('pcore == @pcore_a')
    coords_a = df_pcore_a[['coord_x', 'coord_y', 'coord_z']].to_numpy()

    df_pcore_b = df_of_pcores_for_single_ligand.query('pcore == @pcore_b')
    coords_b = df_pcore_b[['coord_x', 'coord_y', 'coord_z']].to_numpy()

    if len(df_pcore_a) == 0:
        logging.warning(
            f'No pharmacophores found for pharmacophore {pcore_a} !')
    if len(df_pcore_b) == 0:
        logging.warning(
            f'No pharmacophores found for pharmacophore {pcore_b} !')

    xyz_i = coords_a  # DOUBLE COUNTING

    xyz_j = coords_b  # Don't count distances within the same fragment

    if len(xyz_j) > 0:
        delta_coordinates_between_pcores = xyz_i[:, np.newaxis] - xyz_j

        distances_for_this_pair = np.linalg.norm(
            delta_coordinates_between_pcores, axis=2)

        distances_for_this_pair = distances_for_this_pair.flatten()

        return distances_for_this_pair
    else:
        logging.warning(
            f"No {pcore_a}-{pcore_b} distance found for {df_of_pcores_for_single_ligand['mol_id'].values[0]}!")

        return np.array([])
