import os
import warnings

import pandas as pd

# Supress warnings
warnings.filterwarnings("ignore")

# SPECIFY PATHS HERE
PATH_TO_RAW_DATA_FOLDER = '../data/raw_data/1000Hz'
PATH_TO_CLEAN_DATA_FOLDER = '../data/combined_labels/cleaned_data_full'
filenames = os.listdir(PATH_TO_RAW_DATA_FOLDER)

# Change accordingly
shape_mapper = {
    'cube' : 0,
    'cylinder' : 1,
    'sphere' : 2
}

material_mapper = {
    'aluminum' : 0,
    'redoak' : 1,
    'steel' : 2
}

filename_mapper = {
    'cube_aluminum.': 0,
    'cube_redoak.': 1,
    'cube_steel.': 2,
    'cylinder_aluminum.': 3,
    'cylinder_redoak.': 4,
    'cylinder_steel.': 5,
    'sphere_aluminum.': 6,
    'sphere_redoak.': 7,
    'sphere_steel.': 8
}

for filename in filenames:
    # Read text file from COMSOL
    df = pd.read_csv(os.path.join(PATH_TO_RAW_DATA_FOLDER, filename), sep = '\s\s+', engine = 'python', skiprows = 8)

    # Zip coordinates
    locations = list(zip(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]))
    df = df.iloc[:, 3:]
    df['coordinates'] = locations

    columns = df.columns.tolist()
    columns = [columns[-1]] + columns[:-1]
    df = df[columns]

    # Select 4 receivers as feature
    # receivers_locations = {
    #     (-4, -4, 0),
    #     (-4, 4, 0),
    #     (4, -4, 0),
    #     (4, 4, 0)
    # }
    # df = df[df['coordinates'].isin(receivers_locations)]
    # df = df.reset_index(drop = True)

    # Convert string to complex
    new_df = pd.DataFrame()
    for col in df.columns[1:]:
        reals = []
        imags = []
        item = df.loc[:, col]
        for string in item:
            try:
                string = string.replace('i', 'j')
            except:
                string = '0+0j'
            complex_number = complex(string)
            real = complex_number.real
            imaginary = complex_number.imag
            reals.append(real)
            imags.append(imaginary)
        new_df[col] = reals + imags
    df = new_df.T.reset_index(drop = True)

    # Comment / Uncomment based on your need

    # Add target features (split shape and material)
    # shape, material = filename.split('_')
    # shape_class = shape_mapper[shape]
    # material_class = material_mapper[material.split('.')[0]]
    # N = len(df.columns)
    # df[N] = shape_class
    # df[N + 1] = material_class

    # Add target features (combined shape and material)
    shape_material_name = filename[:-3]
    N = len(df.columns)
    df[N] = filename_mapper[shape_material_name]

    # Fix column names for clarity
    column_names = []
    for i in range((len(reals))):
        column_names.append(f'real{i + 1}')
    for i in range(len(imags)):
        column_names.append(f'imag{i + 1}')

    # Comment / Uncomment based on your need
    # column_names.append('shape')
    # column_names.append('material')

    column_names.append('class_id')

    df = df.set_axis(column_names, axis = 1)

    # Save csv
    df.to_csv(os.path.join(PATH_TO_CLEAN_DATA_FOLDER, filename.split('.')[0] + '.csv'), index = None)
