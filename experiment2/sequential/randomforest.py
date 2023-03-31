import math
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Supress warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_NAME = '../../data/combined_labels/combined_full.csv'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED) # reproducibility

# Read data
df = pd.read_csv(os.path.join(ROOT, CSV_NAME))
X, y = df.iloc[:, :-1], df.iloc[:, -1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = RANDOM_SEED)
print(f'Train Size: {len(X_train)}')
print(f'Test Size: {len(X_test)}')
print("=" * 20)

# Preprocess data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Prepare model
model = RandomForestClassifier(random_state = RANDOM_SEED)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Get list of importance֫s
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = df.columns

# Find min number of features
solved = False
index = 1
features = [indices[0]] # start with 1 feature
while not solved and index < len(indices):
    CSV_NAME = '../../data/split_labels/combined_full.csv'
    df = pd.read_csv(os.path.join(ROOT, CSV_NAME))
    X, y = df.iloc[:, :-2], df.iloc[:, -2:]
    X = X.iloc[:, features]

    # Split data
    test_size = 0.05
    N = df.shape[0]
    k = math.ceil(N * test_size)

    permutation = np.random.permutation(N)
    X_rand = X.iloc[permutation, :]
    y_rand = y.iloc[permutation, :]
    df_train = pd.concat([X_rand[k:], y_rand[k:]], axis = 1)

    X_test, y_test = X_rand[:k], y_rand[:k]
    y_shape_test = y_test.iloc[:, 0]
    y_material_test = y_test.iloc[:, 1]

    X_train_material, y_train_material = df_train.iloc[:, :-2], df_train.iloc[:, -1]

    # Split into multiple dataframes based on material
    df_0 = df_train[df_train['material'] == 0]
    X_train_shape_0, y_train_shape_0 = df_0.iloc[:, :-2], df_0.iloc[:, -2]
    df_1 = df_train[df_train['material'] == 1]
    X_train_shape_1, y_train_shape_1 = df_1.iloc[:, :-2], df_1.iloc[:, -2]
    df_2 = df_train[df_train['material'] == 2]
    X_train_shape_2, y_train_shape_2 = df_2.iloc[:, :-2], df_2.iloc[:, -2]

    # Preprocess data
    scaler = StandardScaler().fit(X_train_material)
    X_train_material = scaler.transform(X_train_material)
    X_train_shape_0 = scaler.transform(X_train_shape_0)
    X_train_shape_1 = scaler.transform(X_train_shape_1)
    X_train_shape_2 = scaler.transform(X_train_shape_2)
    X_test = scaler.transform(X_test)

    X_train_shapes = [X_train_shape_0, X_train_shape_1, X_train_shape_2]
    y_train_shapes = [y_train_shape_0, y_train_shape_1, y_train_shape_2]

    # Prepare model
    shape_models = [RandomForestClassifier(random_state = RANDOM_SEED) for _ in range(3)]
    material_model = RandomForestClassifier(random_state = RANDOM_SEED)

    # Train model
    for i in range(3):
        shape_models[i].fit(X_train_shapes[i], y_train_shapes[i])
    material_model.fit(X_train_material, y_train_material)

    # Inference
    y_material_pred = material_model.predict(X_test)
    shape_preds = list(shape_models[i].predict(X_test) for i in range(3))

    y_shape_pred = []
    for i, num in enumerate(y_material_pred):
        y_shape_pred.append(shape_preds[num][i])

    if accuracy_score(y_shape_test, y_shape_pred) == 1 and accuracy_score(y_material_test, y_material_pred) == 1:
        solved = True
        break

    # Add next useful feature
    features.append(indices[index])
    index += 1

if not solved:
    print(f'Minimum Number of Measurements: Can\'t be determined')
else:
    print(f'Minimum Number of Measurements: {len(features)}')
