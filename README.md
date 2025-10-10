# 🎧 HarmonAIze — AI Music Recommendation System

**HarmonAIze** is a **music recommendation system** built as part of the **CSCI-635: Introduction to Machine Learning** course project (Fall 2025).  
We explore multiple ML paradigms — supervised, unsupervised, and neural — and select the best-performing approach to predict user preferences and generate personalized song recommendations.

The system is trained on the **[Spotify 1 Million Tracks Dataset](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks)**, which contains rich audio features such as danceability, energy, valence, tempo, and acousticness for over a million songs across genres and years.

Our final goal is to build an **interactive webapp** that showcases the best-performing model and presents dynamic, visually elegant music recommendations — connecting sound, mood, and personalization.

---

## 🚀 Overview

This project demonstrates key machine-learning concepts from our syllabus:

- **Data Preprocessing:** handling missing values, normalization, feature selection, and scaling  
- **Modeling:** training multiple ML algorithms — both traditional (e.g., logistic regression, random forest) and neural (e.g., MLP)  
- **Evaluation:** comparing models with metrics such as ROC-AUC, F1-Score, Accuracy, and Precision-Recall  
- **Critical Analysis:** analyzing model performance and discussing strengths, limitations, and improvements  
- **Webapp Deployment (optional ambitious goal):** integrating the best model into a simple, interactive recommendation interface  

---
## Setup & Installation

### Clone the Repository
```bash
git clone https://github.com/djordan015/CSCI-635-01-Group-4.git
```

### Install Dependencies
```bash
pip install -r requirements.txt
```
### Dataset Setup
This project uses the Spotify 1 Million Tracks dataset from Kaggle:
 https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks
Due to Kaggle’s file-size and license restrictions, the dataset is not included in this repository.
Follow the steps below to add it locally.
#### Option 1: Manual Download
Visit the Kaggle dataset page.
Download and unzip the file.
Move the CSV into the project folder:
```bash
mkdir -p data/raw
mv ~/Downloads/spotify-1million-tracks/*.csv data/raw/
```
#### Option 2: Kaggle API (if configured)
```bash 
pip install kaggle
kaggle datasets download -d amitanshjoshi/spotify-1million-tracks -p data/raw --unzip
```


**Developed by Group 4 — CSCI-635 (Fall 2025)**  
Prionti Nasir, David Alexander Jordan

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-Academic-lightgrey)
