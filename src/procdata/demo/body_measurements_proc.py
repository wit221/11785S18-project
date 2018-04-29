import pandas as pd
import numpy as np

cols_to_keep = np.array([0,2,8,10,18,24])

fill_val = float('nan')
seqn_start = 73557
seqn_end = 83732 #exclusive

def fillMissingSeqnAndDropSeqn(df):
    df =  df.set_index("SEQN").reindex(pd.Index(np.arange(seqn_start,seqn_end), name="SEQN")).reset_index()
    df = df.drop('SEQN',1)
    return df

# Read in the CSV file
bm = pd.read_csv("BMX_H.csv",
                  na_values="nan")

# Keep just the important features
bm = bm.iloc[:, cols_to_keep]

# Add rows (fill with nan) for missing SEQN numbers
bm_full = fillMissingSeqnAndDropSeqn(bm)

# Convert survey participant age to categorical age-range buckets (0-9, 10-19, 20-29, etc.)
col_names = list(bm_full)
bm_out = pd.DataFrame()
for i in col_names:
    cats = 'cats{}'.format(i)
    cats_OH = 'cats{}-OH'.format(i)
    cats = pd.cut(bm_full[i], bins=10,
                  include_lowest=True)
    cats_OH = pd.get_dummies(cats, dummy_na=True, prefix=i)
    bm_out = pd.concat([bm_out, cats_OH], axis=1)

# Get columns headers of body measurment array
body_measurement_headers = list(bm_out)
# Convert to a numpy array
body_measurement_headers = np.asarray(body_measurement_headers)

# Convert bm_out to numpy array
bm_out = bm_out.values

np.save("body_measurement_data_2013-1014", bm_out)
np.save("body_measurement_column_headers_2013-2014", body_measurement_headers)
