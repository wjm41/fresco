from typing import Union, List
import logging

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski, Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.MolStandardize import rdMolStandardize


def return_pains_lib():
    df_pains = pd.read_csv('/rds-d2/user/wjm41/hpc-work/datasets/Ugis/datasets/PAINS.sieve', delim_whitespace=True, skiprows=10,
                           names=['family', 'regID', 'smarts', 'a', 'b'])[['regID', 'smarts']]
    df_pains['regID'] = df_pains['regID'].str.replace('regId=', '')

    pains_lib = [Chem.MolFromSmarts(x) for x in df_pains['smarts']]
    return pains_lib


def return_alert_substructs():
    furan = Chem.MolFromSmarts('[o]1[c][c][c][c]1')
    thiophene = Chem.MolFromSmarts('[s]1[c][c][c][c]1')
    thiol = Chem.MolFromSmarts('[S][#1]')
    phenol = Chem.MolFromSmarts('[c]1[c][c][c][c][c]1[O][#1]')
    nitro = Chem.MolFromSmarts('[c][$([NX3](=O)=O),$([NX3+](=O)[O-])]')

    # just remove boron-containing compounds
    boronic_acid = Chem.MolFromSmarts('B')
    sulphonic_acid = Chem.MolFromSmarts('S(-O)-O')  # TODO proper smarts

    alert_substructs = \
        [furan,
         thiophene,
         thiol,
         phenol,
         nitro,
         boronic_acid,
         sulphonic_acid
         ]
    return alert_substructs


def does_mol_have_structural_alert(mol: Union[str, Mol], alert_substructs: List = None):

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    if alert_substructs is None:
        alert_substructs = return_alert_substructs()

    alert_matches = [bool(Chem.AddHs(mol).GetSubstructMatches(alert_substruct))
                     for alert_substruct in alert_substructs]
    has_structural_alert = np.any(alert_matches)
    return has_structural_alert


def does_mol_have_too_many_donors_and_acceptors(mol: Union[str, Mol], n=8):

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    too_many_pharmacophores = Lipinski.NumHDonors(
        mol) + Lipinski.NumHAcceptors(mol) > n
    return too_many_pharmacophores


def does_mol_match_pains(mol: Union[str, Mol], pains_lib: List = None, log=False):

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    if pains_lib is None:
        pains_lib = return_pains_lib()

    pains_matches = [bool(mol.GetSubstructMatches(pain)) for pain in pains_lib]
    if log:
        logging.info(pains_matches)

    matches_a_pains = np.any(pains_matches)
    return matches_a_pains


def does_mol_have_more_than_one_chiral_center(mol: Union[str, Mol], log=False) -> bool:

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    atom_indices_of_chiral_centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False)

    more_than_one_chiral_center = len(atom_indices_of_chiral_centers) > 1
    return more_than_one_chiral_center


def return_taut_enumerator(set_max: int = None):
    enumerator = rdMolStandardize.TautomerEnumerator()

    if set_max is not None:
        enumerator.SetMaxTautomers(100)
    return enumerator


def return_canonicalised_smiles(mol: Union[str, Mol], enumerator: rdMolStandardize.TautomerEnumerator = None) -> str:
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    if enumerator is None:
        enumerator = return_taut_enumerator()

    canonicalized_smiles = Chem.MolToSmiles(enumerator.Canonicalize(mol))
    return canonicalized_smiles


def does_mol_satisfy_ro5(mol: Union[str, Mol]) -> bool:

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if Descriptors.MolWt(mol) > 500:
        return False
    elif Descriptors.MolLogP(mol) > 5:
        return False
    elif Lipinski.NumHAcceptors(mol) > 10:
        return False
    elif Lipinski.NumHDonors(mol) > 5:
        return False
    else:
        return True


def filter_physchem(mol: Union[str, Mol]) -> bool:

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
