import os
import os.path
import pickle
import math
import random
import sys
import logging

from tqdm import tqdm

from rdkit import Chem
import os.path

from frag_funcs import return_pcore_dataframe, get_pair_distances, clean_smiles

tranch_file = open(sys.argv[1],'r')
tranches = tranch_file.read().splitlines()
# tranches = [x.split('/')[-1][:-4] for x in tranch_file.read().splitlines()]
    
#zinc_dir = '/rds-d2/user/wjm41/hpc-work/datasets/ZINC/real/'
zinc_dir = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/'

folder_names = [zinc_dir + x for x in tranches]
# folder_names.sort()

important = ['Donor-Aromatic',
            'Aromatic-Acceptor',
            'Aromatic-Aromatic']
unimportant = ['Donor-Donor',
               'Donor-Acceptor',
               'Acceptor-Acceptor']

pairs = important+unimportant

for folder in tqdm(folder_names):
    logging.warning(folder)
    if not os.path.isfile(folder+'/pairs.pickle'): # check file existence 
        try:
            try:
                if os.stat(folder+'/mols.pickle').st_size == 0:
                    os.remove(folder+'/mols.pickle')
                
                with open(folder+'/mols.pickle', 'rb') as handle:
                    zinc_mols = pickle.load(handle) 
                
                zinc_smi = [Chem.MolToSmiles(mol[0]) for mol in zinc_mols]
                f = open(folder+'/mols.smi', 'w')
                for smi in zinc_smi:
                    f.write(smi+'\n')
            except Exception as ex:
                print(ex)
                zinc_mols = []
                suppl = Chem.SDMolSupplier(folder+'/mols.sdf', sanitize=True, removeHs=False)

                tmp_mols = [None]*len(suppl)
                bad_ids = []
                for i,mol in enumerate(suppl):

                    try:
                        conf = mol.GetConformer()

                        mol_data = [mol]
                        for j,atom in enumerate(mol.GetAtoms()):
                            mol_data.append([atom.GetSymbol(),
                                                conf.GetPositions()[j]
                                                ])
                        tmp_mols[i] = mol_data

                    except Exception as ex:
                        bad_ids.append(i)

                tmp_mols = [mol for mol in tmp_mols if mol is not None]
                zinc_mols = zinc_mols + tmp_mols
                #with open(folder+'/mols.pickle', 'wb') as handle:
                #    pickle.dump(zinc_mols, handle)  
                zinc_smi = [Chem.MolToSmiles(mol[0]) for mol in zinc_mols]
                zinc_smi = clean_smiles(zinc_smi)
                f = open(folder+'/mols.smi', 'w')
                for smi in zinc_smi:
                    f.write(smi+'\n')
                #try:
                #    os.remove(folder+'/mols.sdf') #delete mols.sdf afterwards
                #except:
                #    pass

            print('Number of valid molecules for tranch {} :{}'.format(folder.split('/')[-1],len(zinc_mols)))

            interesting_pcores = ['Donor', 'Acceptor', 'Aromatic']

            zinccore_df = return_pcore_dataframe(zinc_mols, interesting_pcores, hit=False)

            zinc_pairs = [None]*len(set(zinccore_df['mol_id']))
            for j,i in enumerate(set(zinccore_df['mol_id'])):
                zinc_pair_individual = {}

                for combo in pairs:
                    core_a,core_b = combo.split('-')

                    zinc_pair_individual[combo]= get_pair_distances(zinccore_df[zinccore_df['mol_id']==i], core_a, core_b, frag=False, active=None)
                zinc_pairs[j] = zinc_pair_individual
            with open(folder+'/pairs.pickle', 'wb') as handle:
                pickle.dump(zinc_pairs, handle) 
            #try:
            #    os.remove(folder+'/mols.sdf') #delete mols.sdf afterwards
            #except:
            #    pass
            #try:
            #    os.remove(folder+'/mols.pickle') #delete mols.sdf afterwards
            #except:
            #    pass
        except Exception as ex:
            print(ex)
            pass

