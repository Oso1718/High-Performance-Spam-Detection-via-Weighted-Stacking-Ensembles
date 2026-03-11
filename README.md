# Spam-Detection-via-Weighted-Stacking-Ensembles

## Executive Summary

This project develops a high-performance spam detection system using the **Spambase dataset (UCI ML Repository)**.  
The dataset contains **4,601 emails and 57 engineered features** describing word frequency, special characters, and capitalization patterns. The core objective was to develop a robust filtering system by applying advanced Exploratory Data Analysis (EDA), dimensionality reduction, and a Meta-learning (Stacking) approach to outperform individual classification models.

A complete **machine learning pipeline** was implemented including:

- Exploratory Data Analysis
- Outlier detection using **Robust Mahalanobis Distance (MCD)**
- Dimensionality reduction via **PCA and correlation filtering**
- Training multiple base classifiers

The final architecture achieved 96.45% Accuracy and a 97% F1-Score, demonstrating the effectiveness of weighted ensembles in complex binary classification tasks. The final model combines:

- Random Forest - 40%
- Support Vector Machine - 30%
- Artificial Neural Network - 30%

Outperforming all individual models and demonstrating the effectiveness of **ensemble learning for spam detection tasks**.

## Inquiries & Code Access

**Due to privacy and intellectual property considerations related to academic coursework, the full source code (R scripts and .nb notebooks) and the comprehensive Spanish report (PDF) are hosted in a secure private repository.
Verified academic supervisors and recruiters may request access via the contact information provided in my GitHub Profile Bio or by opening a GitHub Issue in this repository.**

![Contact me](contact.png)

## Key Technical Highlights
- Advanced Statistical Profiling: Conducted exhaustive EDA focusing on Skewness and Kurtosis to identify non-normal distributions and data bias.
- Multivariate Outlier Management: Implemented Robust Mahalanobis Distance (MCD) for outlier detection, using median-based imputation to maintain dataset integrity (4,601 instances).
- Dimensionality Reduction: Utilized Principal Component Analysis (PCA) and correlation filtering (threshold > 0.9) to mitigate multicollinearity and optimize computational efficiency.
- Model Stacking Architecture: Designed and implemented a Weighted Stacking Ensemble combining:
  - Random Forest (40%): For robust bagging and non-linear feature capture.
  - SVM (30%): For balanced precision and high-dimensional boundary management.
  - Artificial Neural Networks (30%): For deep pattern recognition.
- Ensemble learning significantly improved classification performance compared to individual models.
- Random Forest provided strong baseline performance due to its robustness to feature interactions.
- SVM contributed improved decision boundaries in high-dimensional space.
- The neural network captured additional non-linear patterns within the dataset.

The combination of these complementary models resulted in improved recall and reduced false positives.


## Dataset

Spambase (UCI ML Repository)

- 4601 emails
- 57 features:
  - Word frequency
  - Special characters
  - Capital letter usage


## Methodology & Experiments

I designed a KDD-based pipeline (Knowledge Discovery in Databases) including:
- Preprocessing: Normalization using scale() and feature selection through correlation matrices.
- Base Learners: Evaluated Naive Bayes, Logistic Regression, SVM, Random Forest, and ANN individually.
- Optimization: A systematic grid search for optimal weights resulted in the 40/30/30 distribution, which successfully reduced both False Positives and False Negatives compared to standalone models.

Following the next structure:

1. Exploratory Data Analysis
2. Data preprocessing and normalization
3. Outlier detection
4. Feature reduction
5. Base model training
6. Ensemble stacking
7. Performance evaluation


Key statistical techniques used:
- Skewness & Kurtosis analysis
- Mahalanobis Distance (MCD) for outlier detection
- Correlation filtering (>0.9)
- Principal Component Analysis (PCA)

## Models evaluated

- Naive Bayes
- Logistic Regression
- Support Vector Machine
- Random Forest
- Artificial Neural Network

## Weighted Stacking Ensemble

Random Forest - 40%
SVM - 30%
ANN - 30%

## Visualizations

- Feature distribution analysis
![Feature Distribution](1.png)

- PCA projection of the dataset
![PCA](2.png)

- Confusion matrices for each model
![Confusion matrices](3.png)

- ROC curves for model comparison
![ROC curve_NaiveBayes](naivebayes.jpeg)
![ROC curve_SMV](SVM.jpeg)
![ROC curve_RandomForest](randomforest.jpeg)
![ROC curve_LogisticRegresion](logisticr.jpeg)
![ROC curve_NeuralNetwork](neuralnetwork.jpeg)

## Results
```
Model                Accuracy     Kappa     Precision    Recall    F1-Score
SVM	                  93.6%	      0.86	     0.93	      0.95	    0.94
Random Forest	      95.7%	      0.91	     0.96	      0.96	    0.96
Neural Network	      91.9%	      0.83	     0.93	      0.92	    0.93
Weighted Ensemble	  96.4%	      0.92	     0.96	      0.97	    0.97
```
The weighted stacking ensemble improved the performance of all individual models, achieving the highest overall accuracy and F1-score. This result suggests that combining complementary learning algorithms can significantly improve robustness in spam classification tasks.
