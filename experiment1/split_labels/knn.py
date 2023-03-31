import os
import warnings

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Supress warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_NAME = '../../data/split_labels/combined_full.csv'
RANDOM_SEED = 42

# Read data
df = pd.read_csv(os.path.join(ROOT, CSV_NAME))
X, y = df.iloc[:, :-2], df.iloc[:, -2:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = RANDOM_SEED)
print(f'Train Size: {len(X_train)}')
print(f'Test Size: {len(X_test)}')
print("=" * 20)
y_shape_train = y_train.iloc[:, 0]
y_shape_test = y_test.iloc[:, 0]
y_material_train = y_train.iloc[:, 1]
y_material_test = y_test.iloc[:, 1]

# Preprocess data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Prepare model
shape_model = KNeighborsClassifier(n_neighbors = 6)
material_model = KNeighborsClassifier(n_neighbors = 6)

# Train model
shape_model.fit(X_train, y_material_train)
material_model.fit(X_train, y_material_train)
y_shape_pred = shape_model.predict(X_test)
y_material_pred = material_model.predict(X_test)

# Shape prediction score
print(f'Shape Accuracy Score: {accuracy_score(y_shape_test, y_shape_pred)}')
print("=" * 20)
print(f'Material Accuracy Score: {accuracy_score(y_material_test, y_material_pred)}')
print("=" * 20)

# Confusion matrix
print(f'CM for shape: {confusion_matrix(y_shape_test, y_shape_pred)}')
print("=" * 20)
print(f'CM for material: {confusion_matrix(y_material_test, y_material_pred)}')
print("=" * 20)
