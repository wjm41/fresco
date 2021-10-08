import os
import time
import pickle

from tqdm import tqdm

dir1 = '/rds-d2/user/wjm41/hpc-work/datasets/ZINC/real/'
dir2 = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/'
data_dir = dir2 + 'data/'

file = open(dir1+'all_tranches.txt', 'r')
all_tranches = file.read().splitlines()
all_tranches = [line.split('/')[-1].split('.')[0] for line in all_tranches]


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


existing_dirs = [
    data_dir+x for x in os.listdir(data_dir) if os.path.isdir(data_dir+x)]

print('{:.2f}% ({}/{}) Downloaded!'.format(
    100*(len(list(set(existing_dirs)))/len(all_tranches)), len(existing_dirs), len(all_tranches)))

time.sleep(1)
empty_dirs = []
pickle_dirs = []
smi_dirs = []
pickle_and_smi = []
scored_dirs = []
dirs_to_score = []
mpi_dirs_to_score = []

for folder in tqdm(existing_dirs, smoothing=0):
    if os.path.isfile(folder+'/mols.sdf'):
        if os.stat(folder+'/mols.sdf').st_size == 0:
            empty_dirs.append(folder)

    if os.path.isfile(folder+'/pairs.pickle'):  # check file existence
        pickle_dirs.append(folder)

    if os.path.isfile(folder+'/mols.smi'):
        smi_dirs.append(folder)

    if os.path.isfile(folder+'/pairs.pickle') and os.path.isfile(folder+'/mols.smi'):
        real_pairs = pickle.load(open(folder+'/pairs.pickle', 'rb'))
        file = open(folder+'/mols.smi', 'r')
        real_smi = file.read().splitlines()
        if len(real_smi) == len(real_pairs):
            pickle_and_smi.append(folder)

    pairs = [file for file in os.listdir(
        folder) if "pairs_mpi_" in file]
    scores = [file for file in os.listdir(
        folder) if "_mac" in file and ".csv" in file]

    not_scored = True

    if os.path.isfile(folder+'/scores_mac.csv'):
        not_scored = False
    # some dirs had MPI processes without pharmacophores
    elif len(pairs) <= len(scores) and len(pairs) != 0:
        not_scored = False
    elif len(pairs) > len(scores):
        mpi_dirs_to_score.append(folder)
    else:
        dirs_to_score.append(folder)

    if not not_scored:
        scored_dirs.append(folder)
nonempty_dirs = [x for x in existing_dirs if x not in empty_dirs]

print('{:.2f}% ({}/{}) NonEmpty!'.format(100 *
      (len(list(set(nonempty_dirs)))/len(existing_dirs)), len(nonempty_dirs), len(existing_dirs)))
print('{:.2f}% ({}/{}) Has Pickles!'.format(100 *
      (len(list(set(pickle_dirs)))/len(existing_dirs)), len(pickle_dirs), len(existing_dirs)))
print('{:.2f}% ({}/{}) Has Smiles!'.format(100 *
      (len(list(set(smi_dirs)))/len(existing_dirs)), len(smi_dirs), len(existing_dirs)))
print('{:.2f}% ({}/{}) Has Pickle lengths matching Smiles!'.format(100 *
      (len(list(set(pickle_and_smi)))/len(existing_dirs)), len(pickle_and_smi), len(existing_dirs)))
print('{:.2f}% ({}/{}) Scored!'.format(100 *
      (len(list(set(scored_dirs)))/len(existing_dirs)), len(scored_dirs), len(existing_dirs)))
print('len(dirs_to_score): {}'.format(len(dirs_to_score)))
print('len(mpi_dirs_to_score): {}'.format(len(mpi_dirs_to_score)))

f = open('dirs/nonempty.txt', 'w')
for line in nonempty_dirs:
    f.write(line+'\n')
f.close()
f = open('dirs/pickle.txt', 'w')
for line in pickle_dirs:
    f.write(line+'\n')
f.close()
f = open('dirs/smi.txt', 'w')
for line in smi_dirs:
    f.write(line+'\n')
f.close()
f = open('dirs/pickle_and_smi.txt', 'w')
for line in pickle_and_smi:
    f.write(line+'\n')
f.close()
f = open('dirs/dirs_to_score.txt', 'w')
for line in dirs_to_score:
    f.write(line+'\n')
f.close()
f = open('dirs/mpi_dirs_to_score.txt', 'w')
for line in mpi_dirs_to_score:
    f.write(line+'\n')
f.close()
