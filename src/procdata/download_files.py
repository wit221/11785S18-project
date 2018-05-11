import wget
import pandas as pd
import pdb, sys
import os, pickle
import json
import numpy as np

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

def get_file_dict(metadata,component,begin_year):
    '''
    Selects file name stubs from the relevant component and year
    Return a dictionary of file name stubs with description
    :param metadata:
    :param component:
    :param begin_year:
    :return:
    '''
    lab_metadata = metadata.loc[(metadata['Component'] == component) & (metadata['Begin Year'] == begin_year)]
    file_names = lab_metadata['Data File Name'].unique()

    file_dict=dict()

    for f in file_names:
        df = lab_metadata.loc[metadata['Data File Name'] == f]
        constr = df['Use Constraints'].unique()[0]
        if constr == 'RDC Only':
            continue
        else:
            file_dict[f] = df['Data File Description'].unique()[0]

    return file_dict


def get_file_dict_fixed(fileList, component, begin_year):
    '''
    Selects file name stubs from the relevant component and year
    Return a dictionary of file name stubs with description
    :param fileList:
    :param component:
    :param begin_year:
    :return:
    '''
    file_names = fileList

    file_dict = dict()

    for f in file_names:
        file_dict[f] = component + " " + str(begin_year)

    return file_dict


def download_files(file_dict,data_dir,cycle):
    '''
    Downloads XPT and DOC files from NHANES website
    Stores the files in the appropriate directory
    :param file_dict:
    :param data_dir:
    :return xpt_list, doc_list: the list of xpt and doc location
    '''

    output_directory_xpt = data_dir+'xpt/'+cycle
    output_directory_doc = data_dir+'doc/'+cycle
    url_stub = 'https://wwwn.cdc.gov/Nchs/Nhanes/'+cycle+'/'

    xpt_list = []
    doc_list = []

    for f in file_dict:
        try:
            xptfile = output_directory_xpt + '/'+f+'.XPT'
            docfile = output_directory_doc + '/'+f+'.htm'

            if os.path.exists(xptfile):
                xpt_list.append(xptfile)
            else:
                wget.download(url_stub + f + '.XPT', out=output_directory_xpt)
                xpt_list.append(xptfile)

            if os.path.exists(docfile):
                doc_list.append(docfile)
            else:
                docfile = wget.download(url_stub + f + '.htm', out=output_directory_doc)
                doc_list.append(docfile)
        except:
            print('Cannot access file',f)
            continue

    return xpt_list, doc_list

def convert_files(xpt_list):
    '''
    Imports XPT files and adds data from each file to
    a dictionary, with respondent sequence numbers as keys
    :param xpt_list:
    :return data_dict: The dictionary file with extracted data
    '''
    data_dict=dict()

    for f in xpt_list:
        df = pd.read_sas(f)

        if 'SEQN' not in df.columns:
            continue

        for r in range(df.shape[0]):
            id = int(df.loc[r,'SEQN'])
            colnames = list(set(df.columns) - set({'SEQN'}))

            if id not in data_dict:
                data_dict[id] = {}

            for c in colnames:
                data_dict[id][c] = df.loc[r,c]

    return data_dict

def get_llod(doc_list):
    '''
    Extracts LLOD data from the documentation files
    and adds this analyte-level data to the dictionary
    :param doc_list:
    :param data_dir:
    :return llod_dict: A dictionary with LLOD for each analyte
    '''
    llod_dict = dict()
    llod = 'LLOD'

    for doc in doc_list:

        try:
            struct = pd.read_html(doc)
        except:
            continue

        lodcol = None
        lodrow = -1

        for s in range(len(struct)):

            df = struct[s]

            #print(doc)
            #print(df)

            for col in range(df.shape[1]):

                if type(df.columns[col]) is str and llod == df.columns[col].strip():
                    lodcol = col

                for row in range(df.shape[0]):
                    if type(df.iloc[row,col]) is str and llod == df.iloc[row,col].strip():
                        lodcol = col
                        lodrow = row


            if lodcol is not None:

                for row in range(df.shape[0]):

                    if row == lodrow:
                        continue

                    else:
                        for name_col in range(len(df.iloc[row,:])):

                            name = df.iloc[row, name_col]

                            if type(name) is str and (name.strip()[0:3] == "LBX" or name.strip()[0:3] == "URX"):

                                lodval_raw = df.iloc[row,lodcol]

                                if type(lodval_raw) is str:
                                    lodval = float(lodval_raw.strip().split(" ")[0])
                                else:
                                    lodval = float(lodval_raw)

                                llod_dict[name.strip()] = lodval
                                print(doc,s,name,lodval)

    return llod_dict

def main(argv):
    if len(argv) != 4:
        sys.stderr.write(
            'Usage: <data dir><component><cycle>\n')
        sys.exit(1)

    data_dir = argv[1] + '/'
    component = argv[2]
    cycle = argv[3]
    begin_year_str, _ = cycle.split("-")
    begin_year = int(begin_year_str)

    try:

        #metadata = get_meta(data_dir)
        #file_dict = get_file_dict(metadata,component,begin_year)

        if begin_year==2007:
            fileList = ["UHG_E","UHM_E","PBCD_E","UAS_E"]
        if begin_year==2009:
            fileList = ["UHG_F","UHM_F","PBCD_F","UAS_F"]
        if begin_year==2011:
            fileList = ["UHG_G","UHM_G","IHGEM_G", "PBCD_G","UAS_G"]
        if begin_year==2013:
            fileList = ["UHG_H","UM_H","IHGEM_H", "PBCD_H","UTAS_H"]

        file_dict = get_file_dict_fixed(fileList,component,begin_year)

        saveJson(file_dict,
                 data_dir+'json/files_'+component+'_'+cycle+'.json')

        xpt_list, doc_list = download_files(file_dict,data_dir,cycle)
        data_dict = convert_files(xpt_list)

        saveJson(data_dict,
                 data_dir+'json/'+component+'_'+cycle+'.json')

        if component=='Laboratory':
            llod_dict = get_llod(doc_list)
            saveJson(llod_dict,
                     data_dir+'json/llod_'+component+'_'+cycle+'.json')


    except:
        # tb is traceback
        exType, value, tb = sys.exc_info()
        print(value)
        print(tb)
        pdb.post_mortem(tb)

if __name__ == '__main__':
    main(sys.argv)
