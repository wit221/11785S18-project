import pandas as pd
import numpy as np

cols_to_keep = np.array([2,3,4,7,8,10,11,12,14,16,18,29,30,31,32,33,34,35,36,37,38,40,41,42,43,44,45])
cat_cols = np.array([2,3,7,8,10,11,12,14,16,18,29,30,31,32,33,34,36,37,38,44,45])
cols_to_bucket = np.array([4, 35, 46])

# Read in the CSV file
df = pd.read_csv("DEMO_H.csv",
                  na_values="nan")

# Get the list of SEQN numbers
seqn = df["SEQN"].astype('int')

#Convert from pandas DF to numpy array
seqn = seqn.values

# Need to make all categorical columns of type category
col_names = list(df)
for i in cat_cols:
    df[col_names[i]] = df[col_names[i]].astype('category')

# Keep just the categorical columns
df_cat = df.iloc[:, cat_cols]

# Create one-hot encoding of categorical variables
df_cat_OH = pd.get_dummies(df_cat, dummy_na=True)

# Convert survey participant age to categorical age-range buckets (0-9, 10-19, 20-29, etc.)
ages = df["RIDAGEYR"]
age_cats = pd.cut(ages, bins=[0,9,19,29,39,49,59,69,79,80],
                  labels=["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"],
                  include_lowest=True)
age_cats_OH = pd.get_dummies(age_cats, prefix="RIDAGEYR")

# Convert head of household age to categorical age-range buckets (0-9, 10-19, 20-29, etc.)
hh_ages = df["DMDHRAGE"]
hh_age_cats = pd.cut(hh_ages, bins=[0,9,19,29,39,49,59,69,79,80],
                  labels=["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"],
                  include_lowest=True)
hh_age_cats_OH = pd.get_dummies(hh_age_cats, prefix="DMDHRAGE")

# Convert poverty ratio to categorical buckets (0-1, 1.01 - 2, 2.01-3, 3.01-4, 4.01-5, 5+)
poverty = df["INDFMPIR"]
poverty_cats = pd.cut(poverty, bins=[0,.99,1.9,2.9,3.9,4.9,5],
                  labels=["0-.99", "1-1.9", "2-2.9", "3-3.9", "4-4.9", "5+"],
                  include_lowest=True)
poverty_cats_OH = pd.get_dummies(poverty_cats, prefix="INDFMPIR", dummy_na=True)


# Now combine all one-hot encoded data frames
combined_demo = pd.concat([df_cat_OH, age_cats_OH, hh_age_cats_OH, poverty_cats_OH], axis=1)

# Get columns headers of combined_demo
combined_demo_headers = list(combined_demo)

# Convert combined_demo to numpy array
combined_demo = combined_demo.values