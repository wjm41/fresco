import numpy as np
import os
from tqdm import tqdm

dir2 = '/rds-d7/project/rds-ZNFRY9wKoeE/EnamineREAL/'
dirs2 = [x for x in os.listdir(dir2) if os.path.isdir(dir2+x)]

rescore_dirs2 = []
bad_dirs2 = []
for path in tqdm(dirs2):
        pairs = [file for file in os.listdir(dir2+path) if "pairs_mpi_" in file]
        try:
            if len(pairs)!=0:
                for n in range(len(pairs)):
                    smi_len = len(open(dir2+path+'/mols'+str(n)+'.smi','r').read().splitlines())
#                     pickle_len = len(pickle.load(open(dir+path+'/pairs_mpi_'+str(n)+'.pickle','rb')))
                    score_len = len(np.load(dir2+path+'/scores'+str(n)+'.npy'))

                    if smi_len != score_len:
                        rescore_dirs2.append(path)
                        break
            else:
                smi_len = len(open(dir2+path+'/mols.smi','r').read().splitlines())
#                 pickle_len = len(pickle.load(open(dir+path+'/pairs.pickle','rb')))
                score_len = len(np.load(dir2+path+'/scores.npy'))
    
                if smi_len != score_len:
                    rescore_dirs2.append(path)
        except FileNotFoundError as ex:
            bad_dirs2.append(path)
#             print(dir2+path)
#             print(ex)
            continue
print('Num bad dirs: {}'.format(len(rescore_dirs2 + bad_dirs2)))
print(len(rescore_dirs2))
print(len(bad_dirs2))
f = open(dir2+'bad_dirs.txt', 'w')
for line in rescore_dirs2 + bad_dirs2:
    f.write(line+'\n')
f.close()



