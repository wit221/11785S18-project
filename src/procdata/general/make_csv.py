"""
Converts .xpt files to .csv files for all files in target directory.
"""
import os
import os.path
import subprocess
import xport
import numpy as np

root = os.environ['NHANES_PROJECT_ROOT']
years = [1999,2001,2003,2005,2007,2009, 2011]

if __name__ == '__main__':
    for year in years:
        data_path = os.path.join(root,'data/diet/{}/Dietary'.format(year))
        csv_path = os.path.join(data_path, 'csv')
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        for (_, _, filenames) in os.walk(data_path):
            for fn in filenames:
                fnn = os.path.join(data_path, fn)
                cmd = 'python -m xport ' + fnn
                out_file_path = os.path.join(csv_path, '{}.csv'.format(fn.strip('.')[0]))
                print(cmd, out_file_path)
                with open(out_file_path, 'w') as out_file:
                    subprocess.call(cmd.split(),stdout=out_file)
            break
