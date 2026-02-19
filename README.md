# MIPRE Project – Malaria Incidence and Prevention and Plasmodium Image Classification

## Project Overview

This project investigates malaria trends across African countries from 2007–2017 and applies machine learning techniques to:

1. Analyze malaria incidence patterns and prevention strategies.
2. Predict future malaria outbreaks using historical country-level data.
3. Classify Plasmodium species from images using deep learning.

The project combines epidemiological data analysis with computer vision to provide both macro-level (country trends) and micro-level (parasite identification) insights.

---

# Part 1: Malaria Incidence Analysis (2007–2017)

## Dataset Description

The dataset includes:

* ISO-3 country codes
* Country names
* Latitude and longitude
* Annual reported malaria cases
* Preventive measures (e.g., bed net coverage, spraying, treatment access)
* Time range: 2007–2017

Each observation represents a country-year record.

## Objectives

* Identify temporal and spatial trends in malaria cases.
* Evaluate the effectiveness of prevention measures.
* Cluster countries based on malaria burden.
* Forecast short-term malaria incidence.

## Methods Used

### 1. Data Preprocessing

* Missing value imputation
* Outlier detection
* Feature scaling
* Log transformation of malaria cases

### 2. Exploratory Data Analysis (EDA)

* Trend visualization
* Correlation analysis
* Regional comparisons
* Geographic mapping

### 3. Machine Learning Models

#### Regression Models

* Linear Regression
* Random Forest Regressor
* Gradient Boosting (XGBoost / LightGBM)

#### Clustering

* K-Means Clustering
* Hierarchical Clustering

#### Time-Series Forecasting

* Rolling window validation
* SARIMA or LSTM (for high-burden countries)

## Training and Validation

* Time-aware data split (Train: 2007–2014, Validation: 2015–2016, Test: 2017)
* Cross-validation (rolling window for forecasting)
* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)

## Evaluation Metrics

* RMSE
* MAE
* R² Score

---

# Part 2: Plasmodium Species Image Classification

## Problem Statement

Correct identification of Plasmodium species from blood smear images is critical for accurate malaria diagnosis and treatment planning.

## Target Classes

* Plasmodium falciparum
* Plasmodium vivax
* Plasmodium malariae
* Plasmodium ovale
* Plasmodium knowlesi (if available in dataset)

## Dataset

* Microscopic blood smear images
* Labeled by Plasmodium species
* Train / Validation / Test split

## Image Preprocessing

* Image resizing
* Normalization
* Data augmentation (rotation, flipping, zooming)
* Class balancing if necessary

## Deep Learning Models

* Convolutional Neural Networks (CNN)
* Transfer learning (e.g., ResNet, EfficientNet)
* Fine-tuning pretrained models

## Training Strategy

* Train-validation-test split (70/15/15 or 80/10/10)
* Early stopping
* Learning rate scheduling
* Regularization (dropout, weight decay)

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC (if multi-class approach supported)

---

# Project Structure

```
Malaria-ML-Project/
│
├── data/
│   ├── malaria_country_data.csv
│   └── plasmodium_images/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Regression_Modeling.ipynb
│   ├── Forecasting.ipynb
│   └── Image_Classification.ipynb
│
├── models/
│   ├── regression_models/
│   └── cnn_models/
│
├── results/
│   ├── figures/
│   └── reports/
│
└── README.md
```

---

# Tools and Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost / LightGBM
* TensorFlow / PyTorch
* Matplotlib / Seaborn

---

# Expected Outcomes

* Identification of key drivers of malaria incidence.
* Forecasting model capable of predicting short-term outbreaks.
* High-accuracy Plasmodium species classification model.
* Integrated analytical framework combining epidemiology and computer vision.

