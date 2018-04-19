import os
import os.path
import numpy as np
import pandas as pd
import xport

#parameters and util vars
data_dir = os.path.abspath('../data/2013/Dietary')
ind_files = ['DR1IFF_H', 'DR2IFF_H', 'DS1IDS_H', 'DS2IDS_H', 'DSQIDS_H']
tot_files =  ['DR1TOT_H', 'DR2TOT_H', 'DS1TOT_H', 'DS2TOT_H', 'DSQTOT_H']
all_files = ind_files + tot_files

fill_val = float('nan')
seqn_start = 73557
seqn_end = 83732 #exclusive

#helpers
#from seqn number domain to index domain
def seqn2ind(seqn):
    assert(seqn>=seqn_start and seqn <=seqn_end)
    return seqn-seqn_start
#drop all columns from given column incl.
def dropColToEnd(df, col_label):
    col_labels = list(df)
    col_labels_to_drop = col_labels[col_labels.index(col_label):]
    return df.drop(col_labels_to_drop, axis=1)
#drop all columns between start column incl. and end column excl.
def dropBetween(df, col_label_start, col_label_end):
    col_labels = list(df)
    col_labels_to_drop = col_labels[col_labels.index(col_label_start):col_labels.index(col_label_end)]
    return df.drop(col_labels_to_drop, axis=1)
#fill missing sequence number rows with nans
def fillMissingSeqn(df):
    return df.set_index("SEQN").reindex(pd.Index(np.arange(seqn_start,seqn_end), name="SEQN")).reset_index()

def process_DR1IFF_H():
    """
    DR1IFF_H is an ind_files
    """
    fn = 'DR1IFF_H'
    file_path = os.path.join(data_dir, "{}.XPT".format(fn))
    with open(file_path, 'rb') as f:
        dfm = xport.to_dataframe(f)
    #convert columns of interest to int
    df=dfm
    df['SEQN'] = dfm['SEQN'].astype(int)
    df['DR1IFDCD'] = dfm['DR1IFDCD'].astype(int)

    #remove rendundant info past grams
    df = dropColToEnd(df, 'DR1IKCAL')
    #remove time of day info
    #TODO: include time of day, quantize into ~3-5 cats
    df = dropColToEnd(df, 'DR1_020')
    #create core frame
    core = df.drop(['DR1ILINE'], axis=1)
    core = dropColToEnd(core, 'DR1CCMNM')
    core = core.drop_duplicates(subset=['SEQN'], keep='first')
    num_seqs = core.shape[0]
    #determine food encodings
    food_codes = np.sort(df.DR1IFDCD.unique()).astype(np.int)
    f2i = {food_codes[i]:i for i in range(len(food_codes))}
    f2l = {food_codes[i]:'FOOD_{}'.format(i) for i in range(len(food_codes))}
    fc_cols = [str(fc) for fc in food_codes]
    num_fc = food_codes.size
    #expand core by food code labels
    dfadd = pd.DataFrame(np.zeros((num_seqs, num_fc)).astype(np.float),
                    columns=fc_cols)
    core = core.reset_index(drop=True)
    core = pd.concat([core, dfadd], axis=1)
    #add the ind.food entries to core
    df = dropBetween(df, 'WTDRD1', 'DR1IFDCD')
    for seqn in core['SEQN']:
        dfs = df.loc[df['SEQN'] == seqn]
        for i in dfs.index:
            #set appropriate value in core...
            try:
                grams = dfs.at[i, 'DR1IGRMS']
                fc = dfs.at[i, 'DR1IFDCD']
            except Exception as e:
                print(str(e), dfs)
            #sum all occurences of eating in the day
            core.loc[core.index[core['SEQN']==seqn], str(fc)] += grams
    core = fillMissingValues(core)
    core = core.as_matrix()
    return core

def aggregate():
    DR1IFF_H = process_DR1IFF_H()

if __name__ == '__main__':
    x = aggregate()
