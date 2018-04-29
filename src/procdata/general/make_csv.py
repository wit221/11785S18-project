"""
Converts .xpt files to .csv files for all files in target directory.
"""
import os
import os.path
import subprocess
import xport
import numpy as np

master_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(master_dir,'data/2013/Dietary')

if __name__ == '__main__':
    for (_, _, filenames) in os.walk(data_path):
        for fn in filenames:
            fnn = os.path.join(data_path, fn)
            cmd = 'python -m xport ' + fnn
            with open(fnn[:-4]+'.csv', 'w') as out_file:
                subprocess.Popen(cmd.split(),stdout=out_file)
