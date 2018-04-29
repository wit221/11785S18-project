import pandas as pd
import numpy as np

cat_cols = np.array([7])
cols_to_bucket = np.array([6,10,11,13,14,16,17,19,20])

fill_val = float('nan')
seqn_start = 73557
seqn_end = 83732 #exclusive

def fillMissingSeqnAndDropSeqn(df):
    df =  df.set_index("SEQN").reindex(pd.Index(np.arange(seqn_start,seqn_end), name="SEQN")).reset_index()
    df = df.drop('SEQN',1)
    return df

# Read in the CSV file
bp = pd.read_csv("BPX_H.csv",
                  na_values="nan")

bp_full = fillMissingSeqnAndDropSeqn(bp)

# Need to make all categorical columns of type category
col_names = list(bp_full)
for i in cat_cols:
    bp_full[col_names[i]] = bp_full[col_names[i]].astype('category')

# Keep just the categorical columns
bp_cat = bp_full.iloc[:, cat_cols]

# Create one-hot encoding of categorical variables
bp_cat_OH = pd.get_dummies(bp_cat, dummy_na=True)

# Convert continuous to categorical buckets (0-9, 10-19, 20-29, etc.)
bp_cont = bp_full.iloc[:, cols_to_bucket]
col_names = list(bp_cont)
bp_out = pd.DataFrame()
for i in col_names:
    cats = 'cats{}'.format(i)
    cats_OH = 'cats{}-OH'.format(i)
    cats = pd.cut(bp_cont[i], bins=10,
                  include_lowest=True)
    cats_OH = pd.get_dummies(cats, dummy_na=True, prefix=i)
    bp_out = pd.concat([bp_out, cats_OH], axis=1)

combined_bp = pd.concat([bp_out, bp_cat_OH], axis=1)

# Get columns headers of body measurment array
blood_pressure_headers = list(combined_bp)
# Convert to a numpy array
blood_pressure_headers = np.asarray(blood_pressure_headers)

# Convert combined_bp to numpy array
combined_bp = combined_bp.values

np.save("blood_pressure_data_2013-1014", combined_bp)
np.save("blood_pressure_column_headers_2013-2014", blood_pressure_headers)
