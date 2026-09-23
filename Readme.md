# Cardiac Disease Prediction

A machine learning classification project that predicts the target class using patient-related clinical features.

The project covers exploratory data analysis, preprocessing, model training, hyperparameter tuning, cross-validation, model evaluation, feature importance analysis, threshold analysis, and deployment using Streamlit.

> **Disclaimer:** This project is intended for educational and demonstration purposes. The model output is not a medical diagnosis and should not be used as a substitute for professional medical advice.

---

## 📌 Project Overview

The objective of this project is to develop a machine learning classification system that uses clinical features to predict the target class in a cardiac disease dataset.

The project follows an end-to-end machine learning workflow:

- Exploratory Data Analysis
- Data preprocessing
- Feature engineering
- Train-test splitting
- Model pipeline creation
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- ROC-AUC and PR-AUC analysis
- Confusion matrix analysis
- Threshold analysis
- Feature importance
- Model serialization
- Streamlit deployment

---

## 📊 Dataset

The dataset contains:

- **303 observations**
- **14 columns**
- **13 input features**
- **1 target variable**

The features used by the model are:

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting electrocardiographic result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression |
| `slope` | Slope of the ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia-related feature |

The `target` column is used as the classification target.

---

## 🔍 Exploratory Data Analysis

The notebook performs exploratory analysis to understand the dataset before model training.

The analysis includes:

- Dataset structure
- Data types
- Missing-value inspection
- Target distribution
- Feature distributions
- Correlation analysis
- Visual exploration of important variables

The dataset contains 303 rows and 14 columns, with the target variable separated from the 13 input features.

---

## ⚙️ Data Preprocessing

A `ColumnTransformer` is used to apply different preprocessing strategies to numerical and categorical features.

### Numerical Features

The numerical features are:

```text
age
trestbps
chol
thalach
oldpeak