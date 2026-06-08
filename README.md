# Credit Card Fraud Detection Using Machine Learning

## Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques on a highly imbalanced dataset. Since fraudulent transactions account for only a small fraction of all transactions, special preprocessing and resampling methods are required to build effective predictive models.

The project explores different approaches for handling class imbalance, compares multiple machine learning algorithms, and evaluates the impact of undersampling and oversampling techniques on model performance.

---

## Dataset

The dataset contains credit card transactions collected over a two-day period.

### Statistics

- Total Transactions: 284,807
- Fraudulent Transactions: 492
- Fraud Rate: 0.17%
- Missing Values: None

### Features

| Feature | Description |
|----------|------------|
| Time | Transaction timestamp |
| Amount | Transaction amount |
| V1 - V28 | PCA-transformed anonymized features |
| Class | Target variable (0 = Legitimate, 1 = Fraud) |

---

## Project Pipeline

### 1. Data Preprocessing

- Scaled `Time` and `Amount` features.
- Performed train-test split before any resampling to prevent data leakage.
- Addressed class imbalance using:
  - Random Undersampling
  - NearMiss Undersampling
  - SMOTE Oversampling

### 2. Exploratory Data Analysis

- Analyzed class distribution.
- Generated correlation matrices.
- Identified features most strongly associated with fraud.
- Visualized distributions using boxplots.

### 3. Outlier Removal

Applied the Interquartile Range (IQR) method to remove extreme outliers from highly correlated features.

Benefits:
- Reduces noise in the data.
- Improves model stability.
- Enhances classification performance.

### 4. Model Training

The following classifiers were trained and evaluated:

- Logistic Regression
- Support Vector Classifier (SVC)
- Additional baseline classification models

Hyperparameter tuning was performed using GridSearchCV.

### Evaluation Metrics

- ROC-AUC Score
- Precision
- Recall
- Confusion Matrix
- Learning Curves

---

## Handling Class Imbalance

### Random Undersampling

Balances the dataset by reducing the number of majority-class samples.

**Advantages**
- Faster training
- Balanced dataset

**Disadvantages**
- Significant information loss
- Reduced generalization capability

### NearMiss Undersampling

A distance-based undersampling technique that retains majority samples closest to minority samples.

**Advantages**
- Preserves more useful information
- Better decision boundaries than random undersampling

### SMOTE (Synthetic Minority Oversampling Technique)

Generates synthetic fraud samples instead of removing legitimate transactions.

**Advantages**
- No loss of original data
- Improved fraud detection performance
- Better class representation

**Disadvantages**
- Increased computational cost
- Risk of data leakage if applied incorrectly

---

## Neural Network Implementation

A simple neural network was developed using Keras to compare undersampling and oversampling strategies.

### Architecture

```text
Input Layer
      ↓
Hidden Layer (32 neurons, ReLU)
      ↓
Output Layer (Binary Classification)
```

### Training Configuration

| Parameter | Value |
|------------|--------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |

---

## Results

### Traditional Machine Learning Models

- Logistic Regression achieved the best overall performance.
- Support Vector Classifier produced competitive results.
- Logistic Regression showed the strongest balance between training and validation performance.

### Neural Network Results

The neural network trained on the SMOTE-balanced dataset significantly outperformed the network trained on the randomly undersampled dataset.

Reasons:

- SMOTE retains all original data.
- Synthetic fraud samples improve minority-class learning.
- Less information loss compared to undersampling.

---

## Key Findings

- Severe class imbalance negatively impacts model performance.
- Resampling techniques are essential for effective fraud detection.
- Logistic Regression was the strongest traditional classifier.
- SMOTE consistently outperformed random undersampling.
- Sampling methods should be applied within cross-validation to prevent data leakage.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

---

## Conclusion

This project demonstrates how proper handling of class imbalance can significantly improve credit card fraud detection performance. Through a combination of resampling techniques, feature analysis, traditional machine learning models, and neural networks, SMOTE-based oversampling emerged as the most effective approach for preserving information and improving fraud detection accuracy.
