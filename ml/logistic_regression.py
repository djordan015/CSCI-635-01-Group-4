import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load final dataset
data = pd.read_csv("../data/processed/final_dataset.csv")

print(data.shape)

# 2. Define features & target
feature_cols = [
    'danceability', 'energy', 'valence', 'acousticness', 'instrumentalness',
    'liveness', 'speechiness', 'tempo', 'loudness'
]
target_col = "macro_genre"

X = data[feature_cols].values
y = data[target_col].values  # keep as strings; encode after split

# Remove samples with NaN values in feature columns
df = data.dropna(subset=feature_cols)

X = df[feature_cols].values
y = df["macro_genre"].values

print("Cleaned dataset size:", len(df))


# Using simple slicing instead of sklearn splitter for speed
n = len(X)
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)

split = int(0.8 * n)
train_idx, test_idx = idx[:split], idx[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 4. Standard scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 5. Train fast model
model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_train_s, y_train)

# 6. Predictions
y_pred = model.predict(X_test_s)

# 7. Metrics
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test F1 Macro:", f1_score(y_test, y_pred, average='macro'))

print(classification_report(y_test, y_pred))

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 6))
plt.imshow(cm_norm)  # no color specification!
plt.xticks(range(len(model.classes_)), model.classes_, rotation=45, ha='right')
plt.yticks(range(len(model.classes_)), model.classes_)
plt.title("Macro-Genre Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.tight_layout()
plt.show()
