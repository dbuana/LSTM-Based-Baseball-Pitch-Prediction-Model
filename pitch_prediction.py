from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import torch.optim as optim
import torch.nn as nn
import torch

import os
import re

"""
Class: ISC353 Topics in Information Science
Group Members: Mateo Henriquez (251722), Jung Hyun Park (282806), Davian Buana (271706) 

Title: LSTM-Based Prediction of Baseball Pitch Type from Pre-Pitch Motion

Hypothesis: We can predict the type of pitch a pitcher will throw based on their pre-pitch motion.

Link to the manually created dataset: https://drive.google.com/drive/folders/1hXhuYaFO7BRnTka2cpRQDer6cmQ8bMcg?usp=sharing (Please refer to this link for dataset access)
"""

# Custom Dataset Preprocessing
class PitchDataset(Dataset):
    """
    Custom dataset class to load and preprocess from CSV files with the pitching data.
    
    Each file contains a frame-by-frame estimation of a pitcher. The dataset then converts pose coordinates into sequential tensors.
    """
    def __init__(self, folder_path):
        self.X_list = [] # Appends the motion sequences
        self.y_list = [] # Appends the corresponding labels
        self.label_map = {"FF": 0, "SL": 1, "CU": 2} # Label encoding for pitch types

        # Collects all unique landmark IDs accross all CSV files
        # Landmark - A key point that is detected on the pitcher's body
        all_landmark_ids = set()
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        for file in csv_files:
            df = pd.read_csv(os.path.join(folder_path, file))
            all_landmark_ids.update(df["landmark_id"].unique()) # Stores it into a section in the CSV file

        # Creates consistent mapping for the landmarks
        self.global_landmark_ids = sorted(all_landmark_ids)
        self.idx_convert = {lid: i for i, lid in enumerate(self.global_landmark_ids)}
        num_landmarks = len(self.global_landmark_ids)

        print(f"Global Landmark Count: {num_landmarks}")

        # Processes each CSV into tensors
        for file in csv_files:
            match = re.search(r"_(FF|SL|CU)_", file)
            if not match:
                continue

            label = match.group(1)
            label_idx = self.label_map[label]

            df = pd.read_csv(os.path.join(folder_path, file))
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0) # Handles invalid values such as Na/Inf

            frame_ids = sorted(df["frame"].unique())
            grouped = []

            for f_id in frame_ids:
                frame_df = df[df["frame"] == f_id]
                coords = np.zeros((num_landmarks, 3), dtype=float)
                for _, row in frame_df.iterrows():
                    lid = row.get("landmark_id")
                    if lid in self.idx_convert:
                        coords[self.idx_convert[lid]] = [row.get("x", 0.0), row.get("y", 0.0), row.get("z", 0.0)]
                grouped.append(coords.flatten())

            # Skips empty files
            if not grouped:
                continue

            X_np = np.stack(grouped, axis=0)
            X_np = np.nan_to_num(X_np, nan=0.0)

            self.X_list.append(X_np)
            self.y_list.append(label_idx)

        # Converts all datas into PyTorch tensors
        self.X = [torch.tensor(X, dtype=torch.float32) for X in self.X_list]
        self.y = torch.tensor(self.y_list, dtype=torch.long)

        print(f"Loaded {len(self.X)} samples from {folder_path}")

        if len(self.X) > 0:
            flat = torch.cat([t.reshape(-1) for t in self.X])
        else:
            flat = torch.tensor([], dtype=torch.float32)

        # Dataset diagnostics to determine whether it is of appropriate use
        if flat.numel() > 0:
            print("Dataset Integrity Check: ")
            print(f"Total Elements: {flat.numel()}")
            print(f"NaN Count: {torch.isnan(flat).sum().item()}")
            print(f"Inf Count: {torch.isinf(flat).sum().item()}")
            print(f"Min Value: {flat.min().item():.4f}")
            print(f"Max Value: {flat.max().item():.4f}")
            print(f"Mean Value: {flat.mean().item():.4f}")
        else:
            print("Dataset is empty.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Defining the LSTM model
class PitchLSTM(nn.Module):
    """
    Parameters:
    input_size: Number of features per step (landmark coordinates)
    hidden_size: Number of LSTM hidden units
    num_layers: Number of stacked LSTM layers
    num_classes: Number of pitch types to classify
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PitchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Model and Training Evaluation
path = input("Enter the path to the folder: ").strip()
dataset = PitchDataset(path)

# Initialize the model parameters
loader = DataLoader(dataset, batch_size=1, shuffle=True)
input_size = dataset.X[0].shape[1]
model = PitchLSTM(input_size, hidden_size=128, num_layers=2, num_classes=3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
for epoch in range(20):
    total_loss = 0
    for X, y in loader:
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/20], Loss: {total_loss/len(loader):.4f}")

# Model Evaluation
model.eval()
label_map_inv = {0: "Fastball", 1: "Slider", 2: "Curveball"}

pred_labels = []
true_labels = []

with torch.no_grad():
    for i, (X, y) in enumerate(loader):
        output = model(X)
        pred = torch.argmax(output, dim=1).item()

        pred_labels.append(label_map_inv[pred])
        true_labels.append(label_map_inv[y.item()])

        print(f"File {i+1}: Predicted = {label_map_inv[pred]}, True = {label_map_inv[y.item()]}")

labels = sorted(list(set(pred_labels + true_labels)))

# Model Evaluation through Confusion Matrix and Bar Graph Visualization
conf_mat = confusion_matrix(true_labels, pred_labels, labels=labels)
acc = accuracy_score(true_labels, pred_labels)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix: LSTM Pitch Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Plot accuracy per pitch type
type_accuracy = {}
for lbl in labels:
    true_index = [i for i, t in enumerate(true_labels) if t == lbl]
    correct = sum(1 for i in true_index if pred_labels[i] == lbl)
    acc_label = correct / len(true_index) if len(true_index) > 0 else 0
    type_accuracy[lbl] = acc_label * 100

plt.figure(figsize=(6, 4))
plt.bar(type_accuracy.keys(), type_accuracy.values())
plt.ylim(0, 100)
plt.title("Accuracy by Pitch Type")
plt.ylabel("Accuracy (%)")
plt.xlabel("Pitch Type")

for i, v in enumerate(type_accuracy.values()):
    plt.text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.show()
