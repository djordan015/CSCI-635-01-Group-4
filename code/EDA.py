from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE


# Primitive analysis of the dataset to explore directions we can take with it.

# Load data
data = pd.read_csv("../../data/processed/spotify_features.csv")
X = data
top_n = 20
RANDOM_STATE = 42

feature_cols = [
    'danceability', 'energy', 'valence', 'acousticness', 'instrumentalness',
    'liveness', 'speechiness', 'tempo', 'loudness', 'duration_sec', 'popularity'
]

# Filter to top 20 genres
top_genres = X['genre'].value_counts().head(top_n).index.tolist()
X_filtered = X[X['genre'].isin(top_genres)].copy()

# Prepare features and labels
X_features = X_filtered[feature_cols].values
le = LabelEncoder()
y_labels = le.fit_transform(X_filtered['genre'])

print(f"\nFeatures shape: {X_features.shape}")
print(f"Classes: {len(le.classes_)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_labels,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_labels
)

print(f"\nTrain: {X_train.shape[0]:,} samples")
print(f"Test:  {X_test.shape[0]:,} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
print("\nTraining SVM...")
svm = SGDClassifier(
    loss='hinge',
    class_weight='balanced',
    max_iter=1000,
    tol=1e-3,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

svm.fit(X_train_scaled, y_train)
print("Training complete!")

# Predictions
y_pred_train = svm.predict(X_train_scaled)
y_pred_test = svm.predict(X_test_scaled)

# Evaluation
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
train_f1 = f1_score(y_train, y_pred_train, average='macro')
test_f1 = f1_score(y_test, y_pred_test, average='macro')

print("\nSVM RESULTS (Top 20 Genres)")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Train F1 Macro: {train_f1:.4f}")
print(f"Test F1 Macro:  {test_f1:.4f}")

# Classification report
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred_test, target_names=le.classes_))

# Confusion matrix
plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_test, y_pred_test)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_norm, annot=False, cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'SVM Confusion Matrix (Top {top_n} Genres)\nTest Accuracy: {test_acc:.2%}')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Per-class accuracy
print("\nPER-CLASS ACCURACY")
per_class_acc = cm.diagonal() / cm.sum(axis=1)
acc_df = pd.DataFrame({
    'genre': le.classes_,
    'accuracy': per_class_acc,
    'support': cm.sum(axis=1)
}).sort_values('accuracy', ascending=False)

print(acc_df.to_string(index=False))

# Test with different N
for n in [5, 10, 15, 20]:
    top_genres = X['genre'].value_counts().head(n).index.tolist()
    X_filt = X[X['genre'].isin(top_genres)]

    X_feat = X_filt[feature_cols].values
    y_lab = LabelEncoder().fit_transform(X_filt['genre'])

    X_tr, X_te, y_tr, y_te = train_test_split(X_feat, y_lab, test_size=0.2,
                                              random_state=42, stratify=y_lab)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    svm = SGDClassifier(loss='hinge', class_weight='balanced', max_iter=1000, random_state=42)
    svm.fit(X_tr_s, y_tr)

    acc = accuracy_score(y_te, svm.predict(X_te_s))
    print(f"Top {n:2d} genres: {acc:.2%} accuracy")

# Dendrogram analysis on top 50 genres
top_n_analysis = 50
top_genres_analysis = X['genre'].value_counts().head(top_n_analysis).index.tolist()
X_analysis = X[X['genre'].isin(top_genres_analysis)].copy()

print(f"Analyzing {len(top_genres_analysis)} genres")
print(f"Total samples: {len(X_analysis):,}")

# Genre centroids
genre_centroids = X_analysis.groupby('genre')[feature_cols].mean()
print(f"\nGenre centroids shape: {genre_centroids.shape}")

# Dendrogram
plt.figure(figsize=(20, 10))
linkage_matrix = linkage(genre_centroids, method='ward')

dendrogram(
    linkage_matrix,
    labels=genre_centroids.index.tolist(),
    leaf_rotation=90,
    leaf_font_size=10
)
plt.title("Genre Similarity Dendrogram (Ward Linkage)", fontsize=14)
plt.ylabel("Distance", fontsize=12)
plt.xlabel("Genre", fontsize=12)
plt.tight_layout()
plt.show()

# Auto-cluster into K groups
n_clusters = 6

cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
genre_to_cluster = pd.DataFrame({
    'genre': genre_centroids.index,
    'cluster': cluster_labels
}).sort_values('cluster')

print(f"\nGENRES GROUPED INTO {n_clusters} CLUSTERS")

for c in range(1, n_clusters + 1):
    genres_in_cluster = genre_to_cluster[genre_to_cluster['cluster'] == c]['genre'].tolist()
    print(f"\nCluster {c} ({len(genres_in_cluster)} genres):")
    print(f"  {', '.join(genres_in_cluster)}")

# Pairwise distances
dist_matrix = pd.DataFrame(
    squareform(pdist(genre_centroids)),
    index=genre_centroids.index,
    columns=genre_centroids.index
)

pairs = []
for g1 in dist_matrix.index:
    for g2 in dist_matrix.columns:
        if g1 < g2:
            pairs.append((g1, g2, dist_matrix.loc[g1, g2]))

closest_pairs = sorted(pairs, key=lambda x: x[2])[:30]

print("\nTOP 30 MOST SIMILAR GENRE PAIRS")
for g1, g2, dist in closest_pairs:
    print(f"  {g1:20s} <-> {g2:20s} : {dist:.4f}")

# Heatmap
plt.figure(figsize=(16, 14))
sns.clustermap(dist_matrix, cmap='viridis_r', figsize=(16, 14))
plt.title("Genre Distance Heatmap (clustered)")
plt.show()

# Manual macro-genre mapping
genre_mapping = {
    # Acoustic/Calm
    'ambient': 'Acoustic',
    'acoustic': 'Acoustic',
    'chill': 'Acoustic',
    'sleep': 'Acoustic',
    'new-age': 'Acoustic',
    'classical': 'Acoustic',
    'piano': 'Acoustic',
    'guitar': 'Acoustic',
    'opera': 'Acoustic',

    # Electronic/Dance
    'deep-house': 'Electronic',
    'minimal-techno': 'Electronic',
    'drum-and-bass': 'Electronic',
    'club': 'Electronic',
    'dance': 'Electronic',
    'disco': 'Electronic',
    'techno': 'Electronic',
    'house': 'Electronic',
    'edm': 'Electronic',
    'trance': 'Electronic',
    'dubstep': 'Electronic',
    'progressive-house': 'Electronic',
    'chicago-house': 'Electronic',
    'detroit-techno': 'Electronic',
    'electro': 'Electronic',
    'hardstyle': 'Electronic',
    'breakbeat': 'Electronic',
    'garage': 'Electronic',
    'trip-hop': 'Electronic',
    'electronic': 'Electronic',
    'dub': 'Electronic',

    # Hip-Hop/Funk
    'hip-hop': 'Hip-Hop',
    'funk': 'Hip-Hop',
    'groove': 'Hip-Hop',
    'soul': 'Hip-Hop',
    'afrobeat': 'Hip-Hop',
    'dancehall': 'Hip-Hop',

    # Rock
    'rock': 'Rock',
    'alt-rock': 'Rock',
    'hard-rock': 'Rock',
    'rock-n-roll': 'Rock',
    'punk': 'Rock',
    'punk-rock': 'Rock',
    'psych-rock': 'Rock',
    'power-pop': 'Rock',
    'ska': 'Rock',

    # Metal
    'metal': 'Metal',
    'black-metal': 'Metal',
    'death-metal': 'Metal',
    'heavy-metal': 'Metal',
    'grindcore': 'Metal',
    'hardcore': 'Metal',
    'industrial': 'Metal',
    'goth': 'Metal',
    'emo': 'Metal',
    'metalcore': 'Metal',

    # Latin
    'salsa': 'Latin',
    'samba': 'Latin',
    'sertanejo': 'Latin',
    'forro': 'Latin',
    'tango': 'Latin',
    'spanish': 'Latin',

    # Folk/Country
    'folk': 'Folk',
    'country': 'Folk',
    'blues': 'Folk',

    # Jazz
    'jazz': 'Jazz',
    'gospel': 'Jazz',

    # Pop/Other
    'pop': 'Pop',
    'pop-film': 'Pop',
    'party': 'Pop',
    'romance': 'Pop',
    'sad': 'Pop',
    'show-tunes': 'Pop',
    'comedy': 'Pop',
    'k-pop': 'Pop',
    'indie-pop': 'Pop',
    'cantopop': 'Pop',
}

# Apply mapping
X_mapped = X[X['genre'].isin(genre_mapping.keys())].copy()
X_mapped['macro_genre'] = X_mapped['genre'].map(genre_mapping)

print("MACRO-GENRE DISTRIBUTION:")
print(X_mapped['macro_genre'].value_counts())
print(f"\nTotal samples: {len(X_mapped):,}")
print(f"Genres mapped: {len(genre_mapping)}")
print(f"Macro-genres: {X_mapped['macro_genre'].nunique()}")

# Sample for visualization
sample_size = 3000

X_sample = (
    X_mapped
    .groupby('macro_genre', group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), sample_size), random_state=42))
    .reset_index(drop=True)
)

print(f"Sample size: {len(X_sample):,}")
print(X_sample['macro_genre'].value_counts())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample[feature_cols])

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 10))
for macro in X_sample['macro_genre'].unique():
    mask = X_sample['macro_genre'] == macro
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=macro, alpha=0.5, s=10)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Macro-Genre Separability (PCA)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# t-SNE visualization
print("\nRunning t-SNE")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(12, 10))
for macro in X_sample['macro_genre'].unique():
    mask = X_sample['macro_genre'] == macro
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=macro, alpha=0.5, s=10)

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Macro-Genre Separability (t-SNE)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Macro-genre centroid distances
macro_centroids = X_mapped.groupby('macro_genre')[feature_cols].mean()

dist_matrix = pd.DataFrame(
    squareform(pdist(macro_centroids)),
    index=macro_centroids.index,
    columns=macro_centroids.index
)

plt.figure(figsize=(10, 8))
sns.heatmap(dist_matrix, annot=True, fmt='.2f', cmap='viridis_r')
plt.title('Distance Between Macro-Genre Centroids\n(Higher = More Separable)')
plt.tight_layout()
plt.show()

# Separability analysis
print("\nMACRO-GENRE SEPARABILITY ANALYSIS")

avg_dist = dist_matrix.mean(axis=1).sort_values(ascending=False)
print("\nAverage distance to other genres (higher = more unique):")
for genre, dist in avg_dist.items():
    print(f"  {genre:15s}: {dist:.4f}")

pairs = []
for g1 in dist_matrix.index:
    for g2 in dist_matrix.columns:
        if g1 < g2:
            pairs.append((g1, g2, dist_matrix.loc[g1, g2]))

print("\nClosest macro-genre pairs (most likely to confuse):")
for g1, g2, d in sorted(pairs, key=lambda x: x[2])[:5]:
    print(f"  {g1:15s} <-> {g2:15s}: {d:.4f}")

print("\nFarthest macro-genre pairs (most separable):")
for g1, g2, d in sorted(pairs, key=lambda x: x[2], reverse=True)[:5]:
    print(f"  {g1:15s} <-> {g2:15s}: {d:.4f}")
