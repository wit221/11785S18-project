import os
import os.path
import numpy as np
import xport
import csv
from functools import reduce

data_dir = os.path.abspath('../../data/2013/Dietary')
main_files = ['DR1IFF_H', 'DR2IFF_H', 'DR1TOT_H', 'DR2TOT_H', 'DS1IDS_H', 'DS2IDS_H', 'DS1TOT_H', 'DS2TOT_H', 'DSQIDS_H', 'DSQTOT_H']

if __name__ == '__main__':
    seqn_all = []
    for fn in main_files:
        file_path = os.path.join(data_dir, "{}.XPT".format(fn))
        with open(file_path, 'rb') as f:
            a = xport.to_numpy(f)
            print(type(a[0,0]))
            try:
                seqn = np.unique(a[:,0].astype(np.float).astype(np.int))
            except Exception as e:
                print(fn, str(e))
            seqn_all.append(seqn)
    union = reduce(np.union1d, seqn_all)
    intersect = reduce(np.intersect1d, seqn_all)

    print('Dietary seqn info:')
    print('Unique seqn across all files:', union.size)
    print('Seqn appearing in all files', intersect.size)
    for i in range(len(main_files)):
        diff = np.setdiff1d(union, seqn_all[i])
        print("{}: missing {} seqn\t(start seqn: {}, end seqn: {})".format(
            main_files[i], diff.size, np.min(seqn_all[i]), np.max(seqn_all[i])
        ))
