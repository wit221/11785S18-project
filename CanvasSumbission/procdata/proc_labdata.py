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

def proc_labdata(labdata,cheminfo,cycle,demodata):
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
    :return seqn_dense: a numpy array with survey ID numbers for dense data
    :return seqn_sparse: a numpy array with survey ID numbers for sparse data
    '''

    dense_labdata = []
    sparse_labdata = []
    chemlist=[]
    seqn_list = []
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
    for i in demodata:

        seqn_list.append(i)
        dat = np.zeros((num_chem,2),dtype=float)
        qty = 0

        if i not in labdata.keys():
            dense_labdata.append(dat)
            pers_num_chem.append(qty)
            continue

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
    for i in demodata:

        dat = np.zeros((pers_num_chem[s], 3), dtype=float)

        if pers_num_chem[s] == 0:
            sparse_labdata.append(dat)
            s = s + 1
            continue

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
        s = s + 1

    return np.array(dense_labdata), np.array(sparse_labdata), np.array(chemlist), np.array(seqn_list)

def quantize_labdata(dense_labdata, chemlist):
    '''
    Takes dense 3-d numpy array (seqn,2 llod coding variants, 188 chemicals) and converts it into a
    2-d numpy array (seqn, 188 chemicals) containing the following categorical variables:
        9 - value is missing entirely
        0 - value is below llod
        1 - value in the 1st quartile of values above llod
        2 - value in the 2nd quartile of values above llod
        3 - value in the 3rd quartile of values above llod
        4 - value in the 4th quartile of values above llod
    :param dense_labdata: dense numpy array
    :param chemlist: numpy array with chemical names and indices
    :return: quantized labdata, metainformation
    '''
    qdat = np.full((len(dense_labdata),len(chemlist)),9,dtype=int)
    info = []


    for i in range(len(chemlist)):

        nonmiss_value_set = []
        for j in range(len(dense_labdata)):
            if dense_labdata[j][i][0]>0 and dense_labdata[j][i][1]>0:
                nonmiss_value_set.append(dense_labdata[j][i][0])

        #print(nonmiss_value_set)

        if len(nonmiss_value_set)>0 :
            nonmiss_value_set = np.array(nonmiss_value_set)
            q25 = np.percentile(nonmiss_value_set, [25])
            q50 = np.percentile(nonmiss_value_set, [50])
            q75 = np.percentile(nonmiss_value_set, [75])
            info.append({'name':chemlist[i,0], \
                         'index' : i, \
                         'num_nonmiss': len(nonmiss_value_set), \
                         'cat0':'value below LLOD', \
                         'cat1': 'value above LLOD and <='+str(q25), \
                         'cat2': 'value >'+str(q25)+' and  <=' + str(q50), \
                         'cat3': 'value >' + str(q50) + ' and  <=' + str(q75), \
                         'cat4': 'value >' + str(q75) , \
                         'cat9': 'value missing' })

            for j in range(len(dense_labdata)):
                if dense_labdata[j][i][0]==0 and dense_labdata[j][i][1]>0:
                    qdat[j,i] = 0
                if dense_labdata[j][i][0]>0 and dense_labdata[j][i][1]>0 and \
                        dense_labdata[j][i][1] <= q25:
                    qdat[j,i] = 1
                if dense_labdata[j][i][0]>0 and dense_labdata[j][i][1]>0 and \
                        dense_labdata[j][i][1] > q25 and dense_labdata[j][i][1] <= q50:
                    qdat[j,i] = 2
                if dense_labdata[j][i][0]>0 and dense_labdata[j][i][1]>0 and \
                        dense_labdata[j][i][1] > q50 and dense_labdata[j][i][1] <= q75:
                    qdat[j,i] = 3
                if dense_labdata[j][i][0]>0 and dense_labdata[j][i][1]>0 and \
                        dense_labdata[j][i][1] > q75:
                    qdat[j,i] = 4
        else:
            for j in range(len(dense_labdata)):
                if dense_labdata[j][i][0]==0 and dense_labdata[j][i][1]>0:
                    qdat[j,i] = 0
            info.append({'name':chemlist[i,0], \
                         'index' : i, \
                         'num_nonmiss': 0 ,\
                         'cat0':'value below LLOD', \
                         'cat9': 'value missing' })


    return np.array(qdat), np.array(info)

def main(argv):
    if len(argv) != 3:
        sys.stderr.write(
            'Usage: <data dir><cycle>\n')
        sys.exit(1)

    data_dir = argv[1] + '/'
    cycle = argv[2]

    try:

        metadata = get_meta(data_dir)
        demodata = readJson(data_dir+'json/Demographics_'+cycle+'.json')
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

        dense_labdata, sparse_labdata, chemlist, seqn_list = proc_labdata(labdata,cheminfo,cycle,demodata)
        quantized_dense_labdata, quantized_dense_labdata_info  = quantize_labdata(dense_labdata, chemlist)

        np.save(data_dir+'npy/dense_labdata_'+cycle+'.npy',dense_labdata)
        np.save(data_dir+'npy/sparse_labdata_'+cycle+'.npy',sparse_labdata)
        np.save(data_dir+'npy/chem_names_'+cycle+'.npy',chemlist)
        np.save(data_dir+'npy/seqn_labdata_'+cycle+'.npy',seqn_list)
        np.save(data_dir+'npy/quantized_dense_labdata_'+cycle+'.npy',quantized_dense_labdata)
        np.save(data_dir+'npy/quantized_dense_labdata_info_'+cycle+'.npy',quantized_dense_labdata_info)

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
