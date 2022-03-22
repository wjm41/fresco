import os
import sys

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def filter_physchem(mol):

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

sdf_file = sys.argv[1]
print('mols.sdf filesize: {}'.format(file_size(sdf_file)))

suppl = Chem.SDMolSupplier(sdf_file, sanitize=True, removeHs=False)
print('Original mols.sdf length: {}'.format(len(suppl)))  
mols = [None]*len(suppl)

for i,mol in tqdm(enumerate(suppl), total=len(suppl)):

    if mol is not None:
        if filter_physchem(mol):
            mols[i] = mol

mols = [mol for mol in mols if mol is not None]

w = Chem.SDWriter(sdf_file)
for m in mols:
    w.write(m)
print('new mols.sdf filesize: {}'.format(file_size(sdf_file)))
print('New mols.sdf length: {}'.format(len(mols)))  
