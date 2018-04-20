import wget
import pandas as pd
import pdb, sys
import os, pickle
import json
import numpy as np
import math

def saveJson(jsonObj, fileName):
  """Write jsonObj to JSON file.
    @param  jsonObj   parsed JSON
    @param  fileName  the name of the output file
  """
  with open(fileName, 'w') as outFile:
    json.dump(jsonObj, outFile)

def readJson(fileName):
  """Read and parse JSON from file.
    @param fileName   a file to read
    @return a parsed JSON object.
  """
  with open(fileName, 'r') as inpFile:
    return json.loads(inpFile.read())

def get_meta(data_dir):
    '''
    Downloads or loads the NHANES metadata file
    :param data_dir:
    :return metadata: pandas data frame
    '''
    if os.path.exists(data_dir+'doc/metadata.pcl'):
        print('Loading metadata')
        metadata = pickle.load(open(data_dir+'doc/metadata.pcl', 'rb'))

    else:
        print('Downloading metadata')
        output_directory = 'data/doc'
        url = 'https://wwwn.cdc.gov/nchs/nhanes/search/variablelist.aspx'
        filename = wget.download(url, out=output_directory)
        df = pd.read_html(filename)
        metadata = df[1]
        metadata.to_pickle(data_dir+'doc/metadata.pcl')

    return metadata

def get_cheminfo(metadata,llod,cycle):
    '''
    Creates the chemicals dictionary file
    :param metadata: NHANES metadata
    :param llod: LOD dictionary
    :param begin_year: start of the survey cycle
    :return cheminfo: a dictionary file with chemical, units, year, LLOD or variable for LLOD
    '''

    begin_year_str, _ = cycle.split("-")
    begin_year = int(begin_year_str)

    lab_metadata = metadata.loc[(metadata['Component'] == 'Laboratory') & (metadata['Begin Year'] == begin_year)]
    chem_names = lab_metadata['Variable Name'].unique()
    cheminfo=dict()

    for ch in chem_names:

        df = lab_metadata.loc[metadata['Variable Name'] == ch]
        desc = df['Variable Description'].unique()[0]
        file = df['Data File Name'].unique()[0]
        file_desc = df['Data File Description'].unique()[0]

        if ch[0:3] == 'LBX' or ch[0:3] == 'URX':

            if ch not in cheminfo:
                cheminfo[ch] = {}
                cheminfo[ch][cycle] = {}

            cheminfo[ch][cycle]['desc'] = desc
            cheminfo[ch][cycle]['file'] = file
            cheminfo[ch][cycle]['file_desc'] = file_desc

            if ch in llod:
                cheminfo[ch][cycle]['llod_value'] = llod[ch]

            lcvar = ch[0:2]+'D'+ch[3:len(ch)]+'LC'

            if lcvar in chem_names:
                cheminfo[ch][cycle]['censored_flag'] = lcvar

    return cheminfo

def proc_labdata(labdata,cheminfo,cycle):
    '''
    Creates npy arrays for use in modeling
    :param labdata:
    :param cheminfo:
    :param cycle:
    :return dense_labdata: a dense numpy array with measurements coded using 2 variables
                            v1 is 0 if value is below LOD or missing and measurement otherwise
                            v2 is 0 if value is missing, LOD is value is non-detect, measurement otherwise
    :return sparse_labdata: a sparse numpy array with measurements coded using 3 variables
                            v0 is the index number of the chemical in chemlist
                            v1 is 0 if value is below LOD or missing and measurement otherwise
                            v2 is 0 if value is missing, LOD is value is non-detect, measurement otherwise
    :return chemlist: a numpy array with chemicals and their indices
    '''

    dense_labdata = []
    sparse_labdata = []
    chemlist=[]
    pers_num_chem = []

    # Generate list of chemicals
    for ch in cheminfo:
        if cycle in cheminfo[ch]:
            if 'censored_flag' in cheminfo[ch][cycle]:
                chemlist.append(np.array([ch,cheminfo[ch][cycle]['file']]))
                continue
            if 'llod_value' in cheminfo[ch][cycle]:
                chemlist.append(np.array([ch, cheminfo[ch][cycle]['file']]))
                continue

    num_chem = len(chemlist)

    # Generate dense matrix
    for i in labdata:

        dat = np.zeros((num_chem,2),dtype=float)
        qty = 0

        for j in range(num_chem):

            if chemlist[j][0] in labdata[i]:
                val = labdata[i][chemlist[j][0]]

                if math.isnan(val):
                    continue

                if 'censored_flag' in cheminfo[chemlist[j][0]][cycle] \
                        and cheminfo[chemlist[j][0]][cycle]['censored_flag'] in labdata[i]:
                    lc = labdata[i][cheminfo[chemlist[j][0]][cycle]['censored_flag']]

                    if math.isnan(lc):
                        continue

                    if int(lc) == 1:
                        dat[j,0] = 0
                        dat[j,1] = val * math.sqrt(2)
                    else:
                        dat[j,0] = val
                        dat[j,1] = val
                    qty = qty + 1
                    continue

                if 'llod_value' in cheminfo[chemlist[j][0]][cycle]:
                    llod = cheminfo[chemlist[j][0]][cycle]['llod_value']

                    if math.isnan(llod):
                        continue

                    if val<llod:
                        dat[j, 0] = 0
                        dat[j, 1] = llod
                    else:
                        dat[j, 0] = val
                        dat[j, 1] = val
                    qty = qty + 1
                    continue

        dense_labdata.append(dat)
        pers_num_chem.append(qty)

    # Generate sparse matrix
    s = 0
    for i in labdata:

        if pers_num_chem[s] == 0:
            s = s + 1
            continue

        dat = np.zeros((pers_num_chem[s], 3), dtype=float)
        s = s + 1

        qty = 0
        for j in range(num_chem):

            if chemlist[j][0] in labdata[i]:
                val = labdata[i][chemlist[j][0]]

                if math.isnan(val):
                     continue

                if 'censored_flag' in cheminfo[chemlist[j][0]][cycle] \
                        and cheminfo[chemlist[j][0]][cycle]['censored_flag'] in labdata[i]:
                    lc = labdata[i][cheminfo[chemlist[j][0]][cycle]['censored_flag']]

                    if math.isnan(lc):
                        continue

                    if int(lc) == 1:
                        dat[qty, 0] = j
                        dat[qty, 1] = 0
                        dat[qty, 2] = val * math.sqrt(2)
                    else:
                        dat[qty, 0] = j
                        dat[qty, 1] = val
                        dat[qty, 2] = val

                    qty = qty + 1
                    continue

                if 'llod_value' in cheminfo[chemlist[j][0]][cycle]:
                    llod = cheminfo[chemlist[j][0]][cycle]['llod_value']

                    if math.isnan(llod):
                         continue

                    if val < llod:
                        dat[qty, 0] = j
                        dat[qty, 1] = 0
                        dat[qty, 2] = llod
                    else:
                        dat[qty, 0] = j
                        dat[qty, 1] = val
                        dat[qty, 2] = val

                    qty = qty + 1
                    continue

        sparse_labdata.append(dat)

    return np.array(dense_labdata), np.array(sparse_labdata), np.array(chemlist)

def main(argv):
    if len(argv) != 3:
        sys.stderr.write(
            'Usage: <data dir><cycle>\n')
        sys.exit(1)

    data_dir = argv[1] + '/'
    cycle = argv[2]

    try:

        metadata = get_meta(data_dir)
        labdata = readJson(data_dir+'json/Laboratory_'+cycle+'.json')
        llod =    readJson(data_dir+'json/llod_Laboratory_'+cycle+'.json')

        cheminfo = get_cheminfo(metadata,llod,cycle)
        saveJson(cheminfo,
                 data_dir + 'json/cheminfo.json')

        for ch in cheminfo:
            lv = 'llod_value'
            cf = 'censored_flag'
            if lv in cheminfo[ch][cycle]:
                continue
            if cf in cheminfo[ch][cycle]:
                continue

            print(ch,cheminfo[ch][cycle]['file'])
            print(cheminfo[ch][cycle]['file_desc'])

        dense_labdata, sparse_labdata, chemlist = proc_labdata(labdata,cheminfo,cycle)

        np.save(data_dir+'dense_labdata_'+cycle+'.npy',dense_labdata)
        np.save(data_dir+'sparse_labdata_'+cycle+'.npy',sparse_labdata)
        np.save(data_dir+'chemlist_labdata_'+cycle+'.npy',chemlist)

        print('Number of lab results = ', len(chemlist))
        print('Number of persons (dense):',len(dense_labdata))
        print('Number of persons (sparse):',len(sparse_labdata))

        num_lab_res = np.array([pers.shape[0] for pers in sparse_labdata])
        for p in [5, 10, 25, 50, 75, 90, 95, 99]:
            print("Number of lab results per person: %d-th percentile = %0.2f" % (p, np.percentile(num_lab_res, [p])))




    except:
        # tb is traceback
        exType, value, tb = sys.exc_info()
        print(value)
        print(tb)
        pdb.post_mortem(tb)

if __name__ == '__main__':
    main(sys.argv)
