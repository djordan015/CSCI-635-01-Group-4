# HarmonAIze - AI Music Genre Classification System

HarmonAIze is a music genre classification system built as part of the CSCI-635: Introduction to Machine Learning course project (Fall 2025). We explore multiple ML paradigms - unsupervised clustering, supervised learning (decision trees, logistic regression and neural networks) - to predict music genres from audio features.

The system is trained on the **Spotify 1 Million Tracks Dataset**, which contains 11 audio features such as danceability, energy, valence, tempo, and acousticness for over a million songs across 82 genres.

## 🚀 Overview

### The Challenge
Classifying 82 fine-grained genres using only 11 numerical audio features is unrealistic - our initial experiments achieved only 47% accuracy at their best. Many genre labels (e.g., "Canadian Pop" vs "Australian Pop") are market-driven rather than acoustically distinct.

### Our Solution
We developed a **hybrid clustering approach** that combines:
1. **Hierarchical clustering** on genre feature centroids to identify acoustically similar genres
2. **Domain knowledge refinement** to refine the clusters into musically coherent groupings
3. **Feature-based validation** to confirm sub-genre membership

This reduced 82 genres → -> 10 macro-genres -> a selection of 5 maximally separable macro-genres**:
- **Acoustic** (high acousticness): classical, piano, ambient, sleep, etc.
- **Electronic** (low acousticness, high danceability): house, techno, EDM, etc.
- **Metal** (high energy, high loudness): metal, death-metal, hardcore, etc.
- **Latin** (high valence): salsa, samba, tango, etc.
- **Hip-Hop** (high speechiness): hip-hop

### Results

| Model | Accuracy | MCC |
|-------|----------|-----|
| Logistic Regression | 70.04% | ~0.59 |
| Decision Tree | ~74% | ~0.63 |
| MLP Neural Network | 74.50% | ~0.65 |
| **Gradient Boosted Tree** | **75.69%** | **~0.67** |

GBTree likely outperformed for a few key reasons:

1. Tabular data is GBT's sweet spot
Gradient Boosted Trees consistently dominate on structured/tabular data (like the available Spotify features: danceability, energy, tempo, etc.). Neural networks shine with images, text, and sequential data but for feature tables, tree-based methods usually still win unless there are enough features and enough observations. 
2. Captures non-linear feature interactions
Genre classification depends on combinations of features. Metal might be high energy + low acousticness + high loudness. GBT naturally captures these interactions through its tree splits without us having to engineer them explicitly.

Reference: https://arxiv.org/abs/2207.08815

## 🎯 Key Concepts Demonstrated

- **Data Preprocessing**: handling missing values, normalization, feature scaling
- **Unsupervised Learning**: hierarchical clustering for genre grouping
- **Supervised Learning**: Logistic Regression, Decision Tree, Gradient Boosted Tree, MLP
- **Class Imbalance**: SMOTE oversampling vs balanced class weights
- **Evaluation**: Accuracy, F1-Score, Precision-Recall, Matthews Correlation Coefficient, Confusion Matrices
- **Critical Analysis**: model comparison, overfitting/underfitting detection, trade-off analysis

## 📁 Project Structure

```
├── data/
│   ├── raw/                    # Original Spotify dataset
│   └── processed/              # Preprocessed features with macro-genre labels
├── notebooks/
│   ├── macro_genre_analysis.ipynb    # Clustering & genre mapping methodology
│   └── model_training.ipynb          # Model training & evaluation
├── src/
│   ├── preprocessing.py        # Data cleaning & feature engineering
│   ├── clustering.py           # Hierarchical clustering & genre mapping
│   └── models.py               # Model training & evaluation
├── results/                    # Confusion matrices, classification reports
├── requirements.txt
└── README.md
```

## 🛠️ Setup & Installation

### Clone the Repository
```bash
git clone https://github.com/djordan015/CSCI-635-01-Group-4.git
cd CSCI-635-01-Group-4
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Dataset Setup

This project uses the **Spotify 1 Million Tracks dataset** from Kaggle:
https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks

**Option 1: Manual Download**
```bash
# Visit the Kaggle dataset page, download and unzip
mkdir -p data/raw
mv ~/Downloads/spotify-1million-tracks/*.csv data/raw/
```

**Option 2: Kaggle API**
```bash
pip install kaggle
kaggle datasets download -d amitanshjoshi/spotify-1million-tracks -p data/raw --unzip
```

## 📊 Features Used

| Feature | Description |
|---------|-------------|
| danceability | How suitable for dancing (0-1) |
| energy | Intensity and activity (0-1) |
| loudness | Overall loudness in dB |
| speechiness | Presence of spoken words (0-1) |
| acousticness | Acoustic vs electronic (0-1) |
| instrumentalness | Lack of vocals (0-1) |
| liveness | Presence of audience (0-1) |
| valence | Musical positiveness (0-1) |
| tempo | Beats per minute |
| duration_ms | Track length |
| popularity | Spotify popularity score |

## 🔮 Future Improvements

- **Top-3 categorical accuracy**: Expected 90%+ (accounts for genre blending)
- **4-class model**: Dropping Hip-Hop due to severe class imbalance
- **Ensemble methods**: Combining GBT + MLP predictions
- **Additional features**: Lyrics analysis, spectral features
- **Interactive webapp**: Real-time genre prediction demo

## 👥 Team

Developed by **Group 4** — CSCI-635 Introduction to Machine Learning (Fall 2025)
- David Alexander Jordan
- Prionti Nasir
