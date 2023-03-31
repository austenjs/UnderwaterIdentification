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
model = KNeighborsClassifier(n_neighbors = 6)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Shape prediction score
print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
print("=" * 20)

# Confusion matrix
print(f'CM: {confusion_matrix(y_test, y_pred)}')
print("=" * 20)
