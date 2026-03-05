## Inquiries & Code Access
**Due to privacy and intellectual property considerations related to academic coursework, the full source code (R scripts and .nb notebooks) and the comprehensive Spanish report (PDF) are hosted in a secure private repository.
Verified academic supervisors and recruiters may request access via the contact information provided in my GitHub Profile Bio or by opening a GitHub Issue in this repository.**

![Contact me](contact.png)

# High-Performance-Spam-Detection-via-Weighted-Stacking-Ensembles

## Executive Summary

This project focuses on the automated classification of email spam using the Spambase dataset (UCI Machine Learning Repository). The core objective was to develop a robust filtering system by applying advanced Exploratory Data Analysis (EDA), dimensionality reduction, and a Meta-learning (Stacking) approach to outperform individual classification models.

The final architecture achieved a 96.45% Accuracy and a 97% F1-Score, demonstrating the efficiency of weighted ensembles in complex binary classification tasks.

## Key Technical Highlights
- Advanced Statistical Profiling: Conducted exhaustive EDA focusing on Skewness and Kurtosis to identify non-normal distributions and data bias.
- Multivariate Outlier Management: Implemented Robust Mahalanobis Distance (MCD) for outlier detection, using median-based imputation to maintain dataset integrity (4,601 instances).
- Dimensionality Reduction: Utilized Principal Component Analysis (PCA) and correlation filtering (threshold > 0.9) to mitigate multicollinearity and optimize computational efficiency.
- Model Stacking Architecture: Designed and implemented a Weighted Stacking Ensemble combining:
  - Random Forest (40%): For robust bagging and non-linear feature capture.
  - SVM (30%): For balanced precision and high-dimensional boundary management.
  - Artificial Neural Networks (30%): For deep pattern recognition.

## Methodology & Experiments

I designed a KDD-based pipeline (Knowledge Discovery in Databases) including:
- Preprocessing: Normalization using scale() and feature selection through correlation matrices.
- Base Learners: Evaluated Naive Bayes, Logistic Regression, SVM, Random Forest, and ANN individually.
- Optimization: A systematic grid search for optimal weights resulted in the 40/30/30 distribution, which successfully reduced both False Positives and False Negatives compared to standalone models.

## Results Summary
```
Model                Accuracy     Kappa     Precision    Recall    F1-Score
SVM	                  93.6%	      0.86	     0.93	      0.95	    0.94
Random Forest	      95.7%	      0.91	     0.96	      0.96	    0.96
Neural Network	      91.9%	      0.83	     0.93	      0.92	    0.93
Weighted Ensemble	  96.4%	      0.92	     0.96	      0.97	    0.97
```

## Inquiries & Code Access
**Due to privacy and intellectual property considerations related to academic coursework, the full source code (R scripts and .nb notebooks) and the comprehensive Spanish report (PDF) are hosted in a secure private repository.
Verified academic supervisors and recruiters may request access via the contact information provided in my GitHub Profile Bio or by opening a GitHub Issue in this repository.**
