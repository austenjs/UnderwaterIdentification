import os
import warnings

import pandas as pd

# Supress warnings
warnings.filterwarnings("ignore")

# SPECIFY PATHS HERE
PATH_TO_CLEAN_DATA_FOLDER = '../data/combined_labels/cleaned_data_full'
PATH_TO_SAVE = '../data/combined_labels/combined_full.csv'
filenames = os.listdir(PATH_TO_CLEAN_DATA_FOLDER)

df = pd.DataFrame()
for filename in filenames:
    new_df = pd.read_csv(os.path.join(PATH_TO_CLEAN_DATA_FOLDER, filename))
    print(new_df.shape)
    df = pd.concat([df, new_df])

df.to_csv(PATH_TO_SAVE, index = None)
