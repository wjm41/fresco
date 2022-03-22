import sys

from os import listdir
from itertools import product
import pickle 

import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolDescriptors, MolFromSmiles, MolToSmiles, Draw, MolFromMolFile

from rdkit.Chem import ChemicalFeatures
from rdkit import Geometry
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
from rdkit.RDPaths import RDDataDir
import os.path

fdefFile = os.path.join(RDDataDir,'BaseFeatures.fdef')
featFactory = ChemicalFeatures.BuildFeatureFactory(fdefFile)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # removes annoying RDKit warnings

import py3Dmol
import mols2grid

data_dir = '/home/wjm41/ml_physics/frag-pcore-screen/data/Mac1'

df_sites = pd.read_csv(data_dir + '/hits_ids.csv')

print('Length of df_sites :{}'.format(len(df_sites)))

# 15/06/2021
# A1 - Adenine site (XChem) = 15
# A2 - Adenine site (UCSF) = 0, 10

sites = [0, 10, 15]

df_sites = df_sites[df_sites['site_number'].isin(sites)]

print('Length of new df_sites :{}'.format(len(df_sites)))

frags = []

ids = []
for i,row in tqdm(df_sites.iterrows(), total=len(df_sites)):
    try:
        crystal_id = row['crystal_id'].split(':')[0]
        smiles_path =  data_dir + '/aligned/'+crystal_id+'/'+crystal_id+'_smiles.txt'
        f = open(smiles_path, "r")
        smiles = f.read()

        mol_path = data_dir + '/aligned/'+crystal_id+'/'+crystal_id+'.mol'
        mol = MolFromMolFile(mol_path)    
        conf = mol.GetConformer()

        ligand_data = [mol]
        for j,atom in enumerate(mol.GetAtoms()):
            ligand_data.append([atom.GetSymbol(),
                                conf.GetPositions()[j]])

        frags.append(ligand_data)
        ids.append(crystal_id)

    except Exception as ex:
        print("Couldn't generate conformers for {}".format(smiles))
        print(ex)

frags = [frag for i,frag in enumerate(frags) if i!=138]
ids = [id for i,id in enumerate(ids) if i!=138]
with open(data_dir + '/frags.pickle', 'wb') as pickle_file:
    pickle.dump(frags, pickle_file)

import frag_funcs

n_rand = 100

frag_pair_distance_dict = {} 
# real_pair_dicts = [{} for i in range(n_rand)]  
rand_pair_dicts = [{} for i in range(n_rand)]  

interesting_pcores = ['Donor', 'Acceptor', 'Aromatic']

fragpcore_df = frag_funcs.return_pcore_dataframe(frags, interesting_pcores)
# fragpcore_dfs = [return_pcore_dataframe(frags, interesting_pcores, jiggle=True) for i in range(n_rand)]
rand_dfs = [frag_funcs.return_random_dataframe(frags, interesting_pcores) for i in range(n_rand)]

for pcore_pair in tqdm(product(interesting_pcores,repeat=2)):
    core_a,core_b = pcore_pair
    combo = core_a+'-'+core_b
    
    frag_pair_distance_dict[combo] = np.hstack(frag_funcs.get_pair_distances(fragpcore_df, core_a, core_b, frag=True))
#     for i,fragpcore_df in enumerate(fragpcore_dfs):
#         real_pair_dicts[i][combo] = np.hstack(get_pair_distances(fragpcore_df, core_a, core_b, frag=True))
        
    for i,rand_df in enumerate(rand_dfs):
        rand_pair_dicts[i][combo] = np.hstack(frag_funcs.get_pair_distances(rand_df, core_a, core_b, frag=True))

pickle.dump(frag_pair_distance_dict, open('frag_pair_distance_dict.pickle', 'wb'))
pickle.dump(rand_pair_dicts, open('rand_pair_dicts.pickle', 'wb'))

combo_list = []

kde_dict = {}
# real_kde_dicts = [{} for i in range(n_rand)]
rand_kde_dicts = [{} for i in range(n_rand)]

pair_name = sys.argv[1]
pairs = [pair_name]
#pairs = ['Donor-Aromatic',
#        'Aromatic-Acceptor',
#        'Aromatic-Aromatic',
#        'Donor-Donor',
#        'Donor-Acceptor',
#        'Acceptor-Acceptor']

kde_dict_opt = {}
for combo in tqdm(pairs):
    kde_dict_opt[combo] = frag_funcs.fit_pair_kde(frag_pair_distance_dict[combo], top=0, bottom=-2, n=20)
    for i in range(n_rand):
#         real_kde_dicts[i][combo] = fit_pair_kde(real_pair_dicts[i][combo])
        rand_kde_dicts[i][combo] = frag_funcs.fit_pair_kde(rand_pair_dicts[i][combo], top=0, bottom=-2, n=20)    
    
    pickle.dump(kde_dict_opt, open('kde_dict_opt'+combo+'.pickle', 'wb'))    
    pickle.dump(rand_kde_dicts, open('rand_kde_dicts'+combo+'.pickle', 'wb'))  



