# serum-adulteration-detection

# Serum Adulteration Detection Using Raman Spectroscopy and Machine Learning

This project investigates the adulteration of cosmetic facial serums with mineral oil using Raman spectroscopy and machine learning techniques. It aims to build a reliable, non-destructive system for classifying samples as pure serum, pure oil, or adulterated mixtures based on spectral data.

---

## Table of Contents

- [Objective](#objective)  
- [Experimental Setup](#experimental-setup)  
- [Raman Spectroscopy Overview](#raman-spectroscopy-overview)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
- [Technologies Used](#technologies-used)  
- [Model Evaluation](#model-evaluation)  
- [Future Work](#future-work)  
- [Author](#author)  
- [License](#license)

---

## Objective

- Detect mineral oil adulteration in facial serums using Raman spectroscopy.  
- Classify samples into three categories: Pure Serum, Pure Mineral Oil, and Mixtures.  
- Evaluate and compare multiple machine learning models based on performance metrics.  

---

## Experimental Setup

- **Instrument**: Portable Raman Spectrometer  
- **Laser Source**: 785 nm  
- **Integration Time**: 10 seconds  
- **Laser Power**: 80%  
- **Accumulations**: 5 times per scan  
- **Sampling**:  
  - 3 Raman spectra collected from different spots per sample  
  - Mixtures prepared at 4 concentrations: 10%, 15%, 30%, and 35% mineral oil in serum  

This setup ensures reliable and reproducible spectra for model training and evaluation.

---

## Raman Spectroscopy Overview

Raman spectroscopy captures molecular vibrations through inelastic scattering of laser light, making it suitable for detecting subtle changes in chemical composition. It is:

- Non-destructive  
- Requires minimal sample preparation  
- Sensitive to adulterants such as mineral oil in cosmetic formulations  

---

## Dataset

- **Classes**:
  - Pure Serum: 1 sample  
  - Pure Mineral Oil: 1 sample  
  - Mixtures: 4 samples (10%, 15%, 30%, 35% oil)  

- **Spectra per Mixture**: 3 manually acquired  
- **Data Format**: CSV files with `RamanShift` and `Intensity` columns

- **Augmentation Strategy**:
  - Data augmentation techniques were applied to increase sample diversity and improve model performance.
  - Methods included Gaussian noise injection, spectrum shifting, and intensity scaling.
  - Augmented data was shuffled and normalized to preserve distribution characteristics.

---

## Methodology

1. **Data Preprocessing**  
   - Outlier and NaN removal  
   - Median filtering to reduce noise  
   - Normalization using RobustScaler  

2. **Dimensionality Reduction**  
   - PCA (Principal Component Analysis) applied to reduce features to 2 principal components

3. **Model Training and Testing**  
   - Stratified train-test split (80:20)  
   - Training on reduced features  
   - Evaluation using multiple algorithms

4. **Performance Metrics**  
   - Accuracy  
   - F1 Score  
   - Precision  
   - Confusion Matrix

---

## Technologies Used

- Python 3  
- pandas, NumPy  
- matplotlib, seaborn  
- scikit-learn  
- Jupyter Notebook  

---

## Model Evaluation

Several supervised machine learning models were evaluated for classifying Raman spectra into three categories: Serum, Oil, and Mixture. Among all models, **Gradient Boosting** demonstrated the best overall performance.

### Best Model: Gradient Boosting

| Metric       | Value    |
|--------------|----------|
| Accuracy     | 93.56 %  |
| F1 Score     | 93.56 %  |
| Precision    | 93.76 %  |

Other models tested include Random Forest, SVM, K-Nearest Neighbors, Logistic Regression, Decision Tree, Naive Bayes, AdaBoost, and Extra Trees. Gradient Boosting consistently outperformed them across all metrics.

### Confusion Matrix

A confusion matrix was generated for the Gradient Boosting model to analyze classification behavior across the three target classes.

### Accuracy Comparison

A bar plot was created to visually compare the accuracy scores of all evaluated models.

---

## Future Work

- Expand dataset to include more brands, variants, and adulterants  
- Explore deep learning models such as 1D-CNNs directly on raw spectra  
- Develop a lightweight GUI or web application for real-time use  
- Validate the model with field-portable Raman devices

---

## Author

**Ch Preethi Krishna**  
BSc Computer Science (Honours with Research)  
SRMAP University

---
