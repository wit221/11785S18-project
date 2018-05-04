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
data_dir = os.path.join(master_path, 'data/diet/{}/Dietary')
out_path = os.path.join(master_path, 'data/diet')

year2letter = {2007: 'E', 2009: 'F', 2011: 'G', 2013: 'H'}

seqn_starts = {2007: 41475, 2009: 51624, 2011: 62161, 2013: 73557}
seqn_ends = {2007: 51624, 2009: 62161, 2011: 71917, 2013: 83732} #exclusive

fill_val = 9

#helpers
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
def fillMissingSeqn(df, year):
    df =  df.set_index("SEQN").reindex(pd.Index(np.arange(seqn_starts[year],seqn_ends[year]), name="SEQN"))
    return df
def process_DRIFF(day_num, year):
    # """
    # DR*IFF_H is an ind_files
    # note: drops DR1CCMNM DR1CCMTX DR1_020 DR1_030Z DR1FS DR1_040Z
    # #TODO: include time of day, quantize into ~3-5 cats
    # End cols:
    # food1, food2, food3
    # """
    #setup some var names based on day
    DRIFF = 'DR{}IFF_{}'.format(day_num, year2letter[year])

    DRIFDCD = 'DR{}IFDCD'.format(day_num)
    DRIKCAL = 'DR{}IKCAL'.format(day_num)
    DR_020 = 'DR{}_020'.format(day_num)
    DRILINE = 'DR{}ILINE'.format(day_num)
    DRCCMNM = 'DR{}CCMNM'.format(day_num)
    WTDRD = 'WTDRD1'
    DRIGRMS = 'DR{}IGRMS'.format(day_num)

    fn = DRIFF
    file_path = os.path.join(data_dir.format(year), "{}.XPT".format(fn))
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
    core_seqn = list(core.set_index("SEQN").index)
    core = fillMissingSeqn(core, year)

    return core, core_seqn

def process_DRTOT(day_num, year):
    #setup some var names based on day
    DRTOT = 'DR{}TOT_{}'.format(day_num, year2letter[year])
    WTDRD = 'WTDRD1'
    DRTKCAL = 'DR{}TKCAL'.format(day_num)

    fn = DRTOT
    file_path = os.path.join(data_dir.format(year), "{}.XPT".format(fn))
    with open(file_path, 'rb') as f:
        dfm = xport.to_dataframe(f)
    df = dfm
    df = dropBetween(df, WTDRD, DRTKCAL)

    df_seqn = list(df.set_index("SEQN").index)

    df = fillMissingSeqn(df, year), df_seqn

    return df

def process_DSIDS(day_num, year):
    """
    DR*IDS_H is an ind_files
    note: drops DR1CCMNM DR1CCMTX DR1_020 DR1_030Z DR1FS DR1_040Z
    """
    #setup some var names based on day
    DSIDS = 'DS{}IDS_{}'.format(day_num, year2letter[year])
    DSDSUPP = 'DSDSUPP'
    WTDRD = 'WTDRD1'

    fn = DSIDS
    file_path = os.path.join(data_dir.format(year), "{}.XPT".format(fn))
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
    core_seqn = list(core.set_index("SEQN").index)
    core = fillMissingSeqn(core, year)

    return core, core_seqn

def process_DSTOT(day_num, year):
    DSTOT = 'DS{}TOT_{}'.format(day_num, year2letter[year])
    WTDRD = 'WTDRD1'
    DSTKCAL = 'DS{}TKCAL'.format(day_num)

    fn = DSTOT
    file_path = os.path.join(data_dir.format(year), "{}.XPT".format(fn))
    with open(file_path, 'rb') as f:
        dfm = xport.to_dataframe(f)
    df = dfm
    df = dropBetween(df, WTDRD, DSTKCAL)

    df_seqn = list(df.set_index("SEQN").index)
    df = fillMissingSeqn(df, year)

    return df, df_seqn

def process_DSQIDS(year):
    #setup some var names based on day
    DSQIDS = 'DSQIDS_{}'.format(year2letter[year])
    DSDSUPP = 'DSDSUPP'
    WTDRD = 'WTDRD1'

    fn = DSQIDS
    file_path = os.path.join(data_dir.format(year), "{}.XPT".format(fn))
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

    core_seqn = list(core.set_index("SEQN").index)
    core = fillMissingSeqn(core, year)

    return core, core_seqn

def process_DSQTOT(year):
    #setup some var names based on day
    DSQTOT = 'DSQTOT_{}'.format(year2letter[year])

    fn = DSQTOT
    file_path = os.path.join(data_dir.format(year), "{}.XPT".format(fn))
    with open(file_path, 'rb') as f:
        dfm = xport.to_dataframe(f)
    df = dfm
        #convert columns of interest to int
    df=dfm
    df['SEQN'] = df['SEQN'].astype(int)

    df = dropBetween(df, 'DSDCOUNT', 'DSQTKCAL')

    df_seqn = list(df.set_index("SEQN").index)
    df = fillMissingSeqn(df, year)

    return df, df_seqn

def aggregate(years):

    dfs_all = []
    seqns = []

    for year in years:
        print('Starting year {}.'.format(year))

        DR1IFF, dr1iff_seqn = process_DRIFF(1, year)
        DR2IFF, dr2iff_seqn = process_DRIFF(2, year)
        DR1TOT, dr1tot_seqn= process_DRTOT(1, year)
        DR2TOT, dr2tot_seqn = process_DRTOT(2, year)
        print('DR: done')

        DS1IDS, ds1ids_seqn = process_DSIDS(1, year)
        DS2IDS, ds2ids_seqn = process_DSIDS(2, year)
        DS1TOT, ds1tot_seqn = process_DSTOT(1, year)
        DS2TOT, ds2tot_seqn = process_DSTOT(2, year)
        print('DS: done')

        DSQIDS, dsqids_seqn = process_DSQIDS(year)
        DSQTOT, dsqtot_seqn = process_DSQTOT(year)

        #fill dr1iff values
        #those that are in interview but are not in dr1iff ->set to 0
        indices = list(set(dr1tot_seqn).difference(set(dr1iff_seqn)))
        DR1IFF.loc[indices] = 0

        #fill dr2iff values
        #those that are in interview but are not in dr2iff ->set to 0
        indices = set(dr2tot_seqn).difference(set(dr2iff_seqn))
        DR2IFF.loc[indices] = 0

        #fill dr1tot values
        #those that are not in dr1iff but are in interview -> set to 0
        indices = set(dr1tot_seqn).difference(set(dr1iff_seqn))
        DR1TOT.loc[indices] = 0

        #fill dr2tot values
        #those that are not in dr2iff but are in interview -> set to 0
        indices = set(dr2tot_seqn).difference(set(dr2iff_seqn))
        DR2TOT.loc[indices] = 0

        #fill ds1ids values
        #those that are in interview but are not in ds1ids -> set to 0
        indices = set(ds1tot_seqn).difference(set(ds1ids_seqn))
        DS1IDS.loc[indices] = 0

        #fill ds2ids values
        #those that are in interview but are not in ds2ids -> set to 0
        indices = set(ds2tot_seqn).difference(set(ds2ids_seqn))
        DS2IDS.loc[indices] = 0

        #fill ds1tot values
        #those that are not in ds1ids but are in interview -> set to 0
        indices = set(ds1tot_seqn).difference(set(ds1ids_seqn))
        DS1IDS.loc[indices] = 0

        #fill ds2tot values
        #those that are not in ds2ids but are in interview -> set to 0
        indices = set(ds2tot_seqn).difference(set(ds2ids_seqn))
        DS2IDS.loc[indices] = 0

        #fill dsqids
        #those that are in questionnaire but are not in dsqids -> set to 0
        indices = set(dsqtot_seqn).difference(set(dsqids_seqn))
        DSQIDS.loc[indices] = 0

        #fill dsqtot
        #those that are in questionnaire but are not in ds1ids -> set to 0
        indices = set(dsqtot_seqn).difference(set(dsqids_seqn))
        DSQTOT.loc[indices] = 0

        print('DSQ: done')

        dfs_year = [DR1IFF, DR2IFF, DR1TOT, DR2TOT, DS1IDS, DS2IDS, DS1TOT, DS2TOT, DSQIDS, DSQTOT]
        df_year = pd.concat(dfs_year, axis=1)
        dfs_all.append(df_year)
        seqns.append(np.arange(seqn_starts[year], seqn_ends[year]))

    #discretize
    for i in range(len(dfs_all)):
        dfs_all[i].clip(0,1, inplace=True)
    #fill nans
    for i in range(len(dfs_all)):
        dfs_all[i].fillna(fill_val, inplace=True)
    #convert to uint to avoid memory errors
    for i in range(len(dfs_all)):
        dfs_all[i] = dfs_all[i].astype('uint8')
    #create final dataframe and remove seqn from data frame
    dfs_all = pd.concat(dfs_all, axis=0)
    dfs_all = dfs_all.reset_index()
    dfs_all = dfs_all.drop('SEQN',1)

    #convert to numpy
    labels = list(dfs_all)

    dfs_all = dfs_all.as_matrix()
    #get all labels
    #aggregate sequence numbers
    seqns = np.concatenate(seqns)

    return dfs_all.astypr('uint8'), labels, seqns

if __name__ == '__main__':
    years = [2007,2009]

    x, labels, seqn = aggregate(years)

    # save to disk
    suffix = '{}-{}'.format(years[0], years[-1])

    np.save(os.path.join(out_path, 'dietary_data.npy_{}'.format(suffix)), x)
    np.save(os.path.join(out_path, 'dietary_labels.npy_{}'.format(suffix)), labels)
    np.save(os.path.join(out_path, 'dietary_seqn.npy_{}'.format(suffix)), seqn)
