import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle

# ---------------------------
# Angle Function
# ---------------------------
def get_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))

# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features(row):
    row = np.array(row)

    # Normalize
    row = row - row[0]
    norm = np.linalg.norm(row)
    if norm != 0:
        row = row / norm

    points = row.reshape(21, 3)

    angles = []

    finger_sets = [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16],
        [17,18,19,20]
    ]

    for finger in finger_sets:
        angles.append(get_angle(points[finger[0]], points[finger[1]], points[finger[2]]))
        angles.append(get_angle(points[finger[1]], points[finger[2]], points[finger[3]]))

    return np.concatenate([row, angles])

# ---------------------------
# LOAD DATASET
# ---------------------------
data = []
labels = []
label_map = {}
current_label = 0

for file in os.listdir("dataset"):
    if file.endswith(".csv"):
        label_name = file.split(".")[0]   # includes UNKNOWN also
        label_map[current_label] = label_name

        df = pd.read_csv(f"dataset/{file}", header=None)

        for row in df.values:
            feat = extract_features(row)
            data.append(feat)
            labels.append(current_label)

        current_label += 1

X = np.array(data)
y = np.array(labels)

# ---------------------------
# TRAIN
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=700,
    alpha=0.005
)

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# ---------------------------
# SAVE MODEL
# ---------------------------
with open("model.pkl", "wb") as f:
    pickle.dump((model, label_map), f)