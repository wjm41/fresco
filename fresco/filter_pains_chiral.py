import logging 
from frag_funcs import return_borders
from mpi4py import MPI
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles, MolToSmiles, MolFromSmarts, FindMolChiralCenters, Lipinski
from rdkit import DataStructs, Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.core.display import display, HTML

from rdkit import RDLogger

logging.basicConfig(level=logging.INFO)

RDLogger.DisableLog('rdApp.*')

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

df_pains = pd.read_csv('../data/PAINS.sieve', delim_whitespace=True, skiprows=10,
                       names=['family', 'regID', 'smarts', 'a', 'b'])[['regID', 'smarts']]
df_pains['regID'] = df_pains['regID'].str.replace('regId=', '')

# logging.info('Number of PAINS filters: {}'.format(len(df_pains)))

pains_lib = [MolFromSmarts(x) for x in df_pains['smarts']]

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


def search_alerts(smi, alert_substructs=alert_substructs):
    mol = MolFromSmiles(smi)

    for substruct in alert_substructs:
        if Chem.AddHs(mol).HasSubstructMatch(substruct):
            return True
    else:
        return False


def lipinski(smi, n=8):
    mol = MolFromSmiles(smi)
    if Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol) > n:
        return True
    else:
        return False

data_dir = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/data/'

def pains_match(smi, pains_lib=pains_lib, log=False):
    mol = MolFromSmiles(smi)
    matches = [bool(mol.GetSubstructMatches(pain)) for pain in pains_lib]
    if log:
        logging.info(matches)
    return np.any(matches)

# logging.info(pains_match(
#     '[H]O[C@H]1CN(C(=O)c2cnn3ccn([H])c23)C[C@H]1N([H])C(=O)c1cc(F)cn1[H]', pains_lib=pains_lib, log=True))

def chiral_match(smi, log=False):
    mol = MolFromSmiles(smi)
    matches = FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False)
    # matches = FindMolChiralCenters(mol, includeUnassigned=True)

    if len(matches) > 1:
        return True
    else:
        return False

enumerator = rdMolStandardize.TautomerEnumerator()
# enumerator.SetMaxTautomers(100)

def taut_canon(mol):
    # mol = MolFromSmiles(smi)
    return MolToSmiles(enumerator.Canonicalize(mol))


# MEAN
data_dir = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/data/'
target = 'mac1'

# df_topn = pd.read_csv(data_dir+'../topN/topN_mean_' +
#   target+'.csv')
df_topn = pd.read_csv(data_dir+'../topN/top15M_' +
                      target+'_sorted.csv', nrows=500000)
if mpi_rank == 0:
    logging.info('No. of mols before any filtering: {}'.format(len(df_topn)))

tqdm.pandas()
# df_topn['mol'] = df_topn['smiles'].apply(Chem.MolFromSmiles)

lipinski_n = 8
df_topn["lipinski"] = df_topn['smiles'].apply(lipinski, n=lipinski_n)
df_topn = df_topn[~df_topn["lipinski"]]
if mpi_rank == 0:
    logging.info('No. of mols after lipinski filtering: {}'.format(len(df_topn)))

df_topn['diastereomer'] = df_topn['smiles'].apply(chiral_match)
df_topn = df_topn[~df_topn["diastereomer"]]
if mpi_rank == 0:
    logging.info('No. of mols after chiral filtering: {}'.format(len(df_topn)))

df_topn["alerts"] = df_topn['smiles'].apply(search_alerts)
df_topn = df_topn[~df_topn["alerts"]]
if mpi_rank == 0:
    logging.info('No. of mols after alert filtering: {}'.format(len(df_topn)))

df_topn['pains'] = df_topn['smiles'].apply(pains_match)
df_topn = df_topn[~df_topn["pains"]]
if mpi_rank == 0:
    logging.info('No. of mols after pains filtering: {}'.format(len(df_topn)))

df_topn = df_topn.reset_index()
my_border_low, my_border_high = return_borders(
    mpi_rank, len(df_topn), mpi_size)
my_df = df_topn.iloc[my_border_low:my_border_high]
my_df['mol'] = my_df['smiles'].apply(Chem.MolFromSmiles)

if mpi_rank == 0:
    my_df['taut_smiles'] = my_df['mol'].progress_apply(taut_canon)
else:
    my_df['taut_smiles'] = my_df['mol'].apply(taut_canon)

taut_smiles = mpi_comm.gather(my_df['taut_smiles'].values, root=0)
if mpi_rank == 0:
    # logging.info('taut_smiles: {},{}'.format(taut_smiles))
    taut_smiles = [
        item for sublist in taut_smiles for item in sublist]  # flatten
    df_topn['taut_smiles'] = taut_smiles

    df_topn.drop_duplicates(subset='taut_smiles', inplace=True)
    logging.info('No. of mols after taut filtering: {}'.format(len(df_topn)))

    N = 50000
    df_topn = df_topn.nlargest(n=N, columns='mean_score').reset_index()
    df_topn.to_csv(data_dir+'../topN/topN_taut_filtered_' + target+'.csv')
