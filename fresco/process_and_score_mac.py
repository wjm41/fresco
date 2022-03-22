import os
import os.path
import pickle
import sys
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # removes annoying RDKit warnings

from frag_funcs import return_pcore_dataframe, get_pair_distances, file_size

logging.basicConfig(level=logging.INFO)

tranch_file = open(sys.argv[1],'r')
tranches = tranch_file.read().splitlines()
    
#enamine_dir = '/rds-d2/user/wjm41/hpc-work/datasets/ZINC/real/'
#enamine_dir = '/rds-d7/project/rds-ZNFRY9wKoeE/val_dir/data/'
enamine_dir = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/'

folder_names = [enamine_dir + x for x in tranches]
# folder_names.sort()

important = ['Donor-Aromatic',
            'Aromatic-Acceptor',
            'Aromatic-Aromatic']
unimportant = ['Donor-Donor',
               'Donor-Acceptor',
               'Acceptor-Acceptor']

pairs = important+unimportant

kde_dict = pickle.load(open('kde_dict_mac1.pickle', 'rb'))

for folder in tqdm(folder_names):
    logging.info(folder)
    logging.info(file_size(folder+'/mols.sdf'))
    if not os.path.isfile(folder+'/pairs.pickle'): # check file existence 
            suppl = Chem.SDMolSupplier(folder+'/mols.sdf', sanitize=True, removeHs=False)

            print('test')
            mols = [None]*len(suppl)
            for i,mol in enumerate(suppl):

                try:
                    conf = mol.GetConformer()

                    mol_data = [mol]
                    for j,atom in enumerate(mol.GetAtoms()):
                        mol_data.append([atom.GetSymbol(),
                                            conf.GetPositions()[j]
                                            ])
                    mols[i] = mol_data

                except Exception as ex:
                    continue

            mols = [mol for mol in mols if mol is not None]

            interesting_pcores = ['Donor', 'Acceptor', 'Aromatic']
            print('test2')

            pcore_df = return_pcore_dataframe(mols, interesting_pcores, hit=False)
            if pcore_df is None:
                print('{} molecules do not contain relevant pharmacophores. Skipping and Delete SDF'.format(folder))
                logging.info('{} molecules do not contain relevant pharmacophores. Skipping and Delete SDF'.format(folder))

                #try:
                #    os.remove(folder+'/mols.sdf') #delete mols.sdf afterwards
                #except FileNotFoundError:
                #    pass

                continue # skip this folder
            smis = pcore_df['smiles'].unique()

            f = open(folder+'/mols.smi', 'w')
            for smi in smis:
                f.write(smi+'\n')
            f.close()

            print('Number of valid molecules for tranch {} :{}'.format(folder.split('/')[-1],len(smis)))
            mols = pcore_df.groupby('smiles')
            histograms = {}

            for j,smi in enumerate(smis):
                mol_histogram = {}
                match = mols.get_group(smi)

                for combo in pairs:
                    core_a,core_b = combo.split('-')

                    mol_histogram[combo]= get_pair_distances(match, core_a, core_b, frag=False, active=None)
                histograms[smi] = mol_histogram

            assert len(smis) == len(histograms) 

            pickle.dump(histograms, open(folder+'/pairs.pickle', 'wb')) 
            #try:
            #    os.remove(folder+'/mols.sdf') #delete mols.sdf afterwards
            #except FileNotFoundError:
            #    pass
            scores = {}

            for smi in histograms:
                score = {}
                for n, combo in enumerate(pairs):
                    kde = kde_dict[combo]
                    try:
                        ith_score = np.abs(kde.score_samples(histograms[smi][combo][0].reshape(-1,1)))
                        score[combo] = np.mean(ith_score)
                    except:
                        score[combo] = np.nan

                scores[smi] = score

            df = pd.DataFrame.from_dict(scores, orient='index')
            df.index = df.index.rename('smiles')
            df = df.reset_index()
            df.to_csv(folder+'/scores_mac.csv', index=False)
            print('{} scored!'.format(folder))

