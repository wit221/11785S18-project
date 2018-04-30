# """
# Script that aggregates all NHANES dietary data files into one.
# Each file has its own processing function,
# as no realistically feasible and flexible generalization is possible.
# Notes:
# -discretizes continuous data: 1st iteration simplification
# -discards WTDR1D, WTDR2D: weights are taken from exam. data
# -discards exam-specific logistical features: examiner ID, etc.
# """
import os
import os.path
import itertools
import numpy as np
import pandas as pd
import xport

#parameters and util vars
master_path = os.environ['NHANES_PROJECT_ROOT']
data_dir = os.path.join(master_path, 'data/diet/2013/Dietary')
out_path = os.path.join(master_path, 'data/diet')

ind_files = ['DR1IFF_H', 'DR2IFF_H', 'DS1IDS_H', 'DS2IDS_H', 'DSQIDS_H']
tot_files =  ['DR1TOT_H', 'DR2TOT_H', 'DS1TOT_H', 'DS2TOT_H', 'DSQTOT_H']
all_files = ind_files + tot_files

fill_val = 9
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
def dropAllBut(df, keep_cols):
    return df.drop(df.columns.difference(keep_cols), 1)
def fillMissingSeqnAndDropSeqn(df):
    df =  df.set_index("SEQN").reindex(pd.Index(np.arange(seqn_start,seqn_end), name="SEQN")).reset_index()
    df = df.drop('SEQN',1)
    return df

def process_DRIFF_H(day_num):
    # """
    # DR*IFF_H is an ind_files
    # note: drops DR1CCMNM DR1CCMTX DR1_020 DR1_030Z DR1FS DR1_040Z
    # #TODO: include time of day, quantize into ~3-5 cats
    # End cols:
    # food1, food2, food3
    # """
    #setup some var names based on day
    DRIFF_H = 'DR{}IFF_H'.format(day_num)
    DRIFDCD = 'DR{}IFDCD'.format(day_num)
    DRIKCAL = 'DR{}IKCAL'.format(day_num)
    DR_020 = 'DR{}_020'.format(day_num)
    DRILINE = 'DR{}ILINE'.format(day_num)
    DRCCMNM = 'DR{}CCMNM'.format(day_num)
    WTDRD = 'WTDRD1'
    DRIGRMS = 'DR{}IGRMS'.format(day_num)

    fn = DRIFF_H
    file_path = os.path.join(data_dir, "{}.XPT".format(fn))
    with open(file_path, 'rb') as f:
        dfm = xport.to_dataframe(f)

    df=dfm
    df['SEQN'] = dfm['SEQN'].astype(int)
    df[DRIFDCD] = dfm[DRIFDCD].astype(int)

    #drop redundant data
    df = dropAllBut(dfm, ['SEQN', DRIFDCD])

    #make core out of just seqn
    core = df['SEQN'].drop_duplicates(keep='first').to_frame()
    num_seqs = core.shape[0]

    # determine food encodings
    food_codes = np.sort(df[DRIFDCD].unique()).astype(np.int)
    fc_label_format = DRIFDCD+'-{}'
    fc_cols = [fc_label_format.format(fc) for fc in food_codes]
    num_fc = food_codes.size

    # #expand core by food code labels
    dfadd = pd.DataFrame(np.zeros((num_seqs, num_fc)).astype(np.float),
                    columns=fc_cols)
    core = core.reset_index(drop=True)
    core = pd.concat([core, dfadd], axis=1)
    # #add the ind.food entries to core
    for seqn in core['SEQN']:
        dfs = df.loc[df['SEQN'] == seqn]
        for i in dfs.index:
            #set appropriate value in core.o..
            fc = dfs.at[i, DRIFDCD]
            core.loc[core.index[core['SEQN']==seqn], fc_label_format.format(fc)] = 1

    #final readifiction
    core = fillMissingSeqnAndDropSeqn(core)

    return core

def process_DRTOT_H(day_num):
    #setup some var names based on day
    DRTOT_H = 'DR{}TOT_H'.format(day_num)
    WTDRD = 'WTDRD1'
    DRTKCAL = 'DR{}TKCAL'.format(day_num)

    fn = DRTOT_H
    file_path = os.path.join(data_dir, "{}.XPT".format(fn))
    with open(file_path, 'rb') as f:
        dfm = xport.to_dataframe(f)
    df = dfm
    df = dropBetween(df, WTDRD, DRTKCAL)
    df = fillMissingSeqnAndDropSeqn(df)

    return df

def process_DSIDS_H(day_num):
    """
    DR*IDS_H is an ind_files
    note: drops DR1CCMNM DR1CCMTX DR1_020 DR1_030Z DR1FS DR1_040Z
    """
    #setup some var names based on day
    DSIDS_H = 'DS{}IDS_H'.format(day_num)
    DSDSUPP = 'DSDSUPP'
    WTDRD = 'WTDRD1'

    fn = DSIDS_H
    file_path = os.path.join(data_dir, "{}.XPT".format(fn))
    with open(file_path, 'rb') as f:
        dfm = xport.to_dataframe(f)

    #convert columns of interest to int
    df=dfm
    df['SEQN'] = dfm['SEQN'].astype(int)
    df['DSDSUPID'] = dfm['DSDSUPID'].astype(int)

    #drop redundant data
    df = dropAllBut(df, ['SEQN', 'DSDSUPID'])

    #make core out of just seqn
    core = df['SEQN'].drop_duplicates(keep='first').to_frame()
    num_seqs = core.shape[0]

    # determine sppl encodings
    sppl_codes = np.sort(df.DSDSUPID.unique()).astype(np.int)
    sc_label_format = 'DSDSUPID_2D_{}-{}'
    sc_cols = [sc_label_format.format(day_num, sc) for sc in sppl_codes]
    num_sc = sppl_codes.size

    # #expand core by food code labels
    dfadd = pd.DataFrame(np.zeros((num_seqs, num_sc)).astype(np.float),
                    columns=sc_cols)
    core = core.reset_index(drop=True)
    core = pd.concat([core, dfadd], axis=1)
    # #add the ind.food entries to core
    for seqn in core['SEQN']:
        dfs = df.loc[df['SEQN'] == seqn]
        for i in dfs.index:
            #set appropriate value in core.o..
            sc = dfs.at[i, 'DSDSUPID']
            core.loc[core.index[core['SEQN']==seqn], sc_label_format.format(day_num, sc)] = 1
    core = fillMissingSeqnAndDropSeqn(core)

    return core

def process_DSTOT_H(day_num):
    DSTOT_H = 'DS{}TOT_H'.format(day_num)
    WTDRD = 'WTDRD1'
    DSTKCAL = 'DS{}TKCAL'.format(day_num)

    fn = DSTOT_H
    file_path = os.path.join(data_dir, "{}.XPT".format(fn))
    with open(file_path, 'rb') as f:
        dfm = xport.to_dataframe(f)
    df = dfm
    df = dropBetween(df, WTDRD, DSTKCAL)
    df = fillMissingSeqnAndDropSeqn(df)

    return df

def process_DSQIDS_H():
    #setup some var names based on day
    DSQIDS_H = 'DSQIDS_H'
    DSDSUPP = 'DSDSUPP'
    WTDRD = 'WTDRD1'

    fn = DSQIDS_H
    file_path = os.path.join(data_dir, "{}.XPT".format(fn))
    with open(file_path, 'rb') as f:
        dfm = xport.to_dataframe(f)

    #convert columns of interest to int
    df=dfm
    df['SEQN'] = dfm['SEQN'].astype(int)
    df['DSDSUPID'] = dfm['DSDSUPID'].astype(int)

    #drop redundant data
    df = dropAllBut(df, ['SEQN', 'DSDSUPID'])

    #make core out of just seqn
    core = df['SEQN'].drop_duplicates(keep='first').to_frame()
    num_seqs = core.shape[0]

    # determine sppl encodings
    sppl_codes = np.sort(df.DSDSUPID.unique()).astype(np.int)
    sc_label_format = 'DSDSUPID_30D-{}'
    sc_cols = [sc_label_format.format(sc) for sc in sppl_codes]
    num_sc = sppl_codes.size

    # #expand core by food code labels
    dfadd = pd.DataFrame(np.zeros((num_seqs, num_sc)).astype(np.float),
                    columns=sc_cols)
    core = core.reset_index(drop=True)
    core = pd.concat([core, dfadd], axis=1)
    # #add the ind.food entries to core
    for seqn in core['SEQN']:
        dfs = df.loc[df['SEQN'] == seqn]
        for i in dfs.index:
            #set appropriate value in core.o..
            sc = dfs.at[i, 'DSDSUPID']
            core.loc[core.index[core['SEQN']==seqn], sc_label_format.format(sc)] = 1

    core = fillMissingSeqnAndDropSeqn(core)

    return core

def process_DSQTOT_H():
    #setup some var names based on day
    DSQTOT_H = 'DSQTOT_H'

    fn = DSQTOT_H
    file_path = os.path.join(data_dir, "{}.XPT".format(fn))
    with open(file_path, 'rb') as f:
        dfm = xport.to_dataframe(f)
    df = dfm
        #convert columns of interest to int
    df=dfm
    df['SEQN'] = df['SEQN'].astype(int)

    df = dropBetween(df, 'DSDCOUNT', 'DSQTKCAL')
    df = fillMissingSeqnAndDropSeqn(df)

    return df
def aggregate():
    DR1IFF_H = process_DRIFF_H(1)
    DR2IFF_H = process_DRIFF_H(2)
    DR1TOT = process_DRTOT_H(1)
    DR2TOT = process_DRTOT_H(2)

    print('DR: done')

    DS1IDS_H = process_DSIDS_H(1)
    DS2IDS_H = process_DSIDS_H(2)
    DS1TOT_H = process_DSTOT_H(1)
    DS2TOT_H = process_DSTOT_H(2)

    print('DS: done')

    DSQIDS_H = process_DSQIDS_H()
    DSQTOT_H = process_DSQTOT_H()

    print('DSQ: done')

    files = [DR1IFF_H, DR2IFF_H, DR1TOT, DR2TOT, DS1IDS_H, DS2IDS_H, DS1TOT_H,
    DS2TOT_H, DSQIDS_H, DSQTOT_H]

    x = np.hstack([file.as_matrix() for file in files])
    labels = np.array(list(itertools.chain.from_iterable(files)))
    miss_mask = np.isnan(x)
    seqn = np.arange(seqn_start, seqn_end)
    return x.astype(np.int64), labels, miss_mask, seqn

if __name__ == '__main__':
    x, labels, miss_mask, seqn = aggregate()

    x[x!=0] = 1
    x[miss_mask] = fill_val

    np.save(os.path.join(out_path, 'dietary_data.npy'), x)
    np.save(os.path.join(out_path, 'dietary_labels.npy'), labels)
    np.save(os.path.join(out_path, 'dietary_miss_mask.npy'), miss_mask)
    np.save(os.path.join(out_path, 'dietary_seqn.npy'), seqn)
