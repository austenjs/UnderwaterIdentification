import os
import warnings

import matplotlib.pyplot as plt
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
feature_names = df.columns

# Find min number of features
solved = False
index = 1
features = [indices[0]] # start with 1 feature
while not solved and index < len(indices):
    X = df.iloc[:, features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = RANDOM_SEED)

    # Preprocess data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    # Prepare model
    model = RandomForestClassifier(random_state = RANDOM_SEED)

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if accuracy_score(y_test, y_pred) == 1:
        solved = True
        break

    # Add next useful feature
    features.append(indices[index])
    index += 1

# Use the last trained model to get feature importances
print(f'Number of measurements: {len(features)}')
importances = model.feature_importances_
indices = np.argsort(importances)[-10:] # Top 10

def find_int(string):
    s = ''.join(x for x in string if x.isdigit())
    return int(s)

target_indices = list(map(lambda i: find_int(feature_names[i]) - 1, indices))

df = pd.read_csv(os.path.join('../../data/raw_data/1000Hz/cube_aluminum.txt'), sep = '\s\s+', engine = 'python', skiprows = 8)
locations = list(zip(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]))

tmp = []
for i in reversed(target_indices):
    coord = (round(locations[i][0], 3), round(locations[i][1], 3), round(locations[i][2], 3))
    tmp.append(coord[:-1])
    print(f'real{i + 1}', coord)

data = np.array(tmp)
x, y = data.T
ax = plt.gca()
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
plt.scatter(x, y)
plt.show()