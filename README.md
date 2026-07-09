# immo-eliza-ml

## Project Overview

This project predicts Belgian real estate prices using Machine Learning techniques.

The project follows a complete machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, and model serialization.

Three regression models were implemented and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The Random Forest model achieved the best overall performance.

---

# Project Structure

```
immoeliza_ml/
│
├── data/
│   ├── SaleCleanForAnalysis.csv
│   └── RentCleanForAnalysis.csv
│
├── models/
│   ├── linear_regression_model.joblib
│   ├── decision_tree_model.joblib
│   └── random_forest_model.joblib
│
├── notebooks/
│   └── immoeliza_ml.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---
# Project Objectives

The objectives of this project were to:

- clean and prepare a real estate dataset
- build several regression models
- compare their performance
- identify overfitting
- improve the dataset by handling suspicious outliers
- save trained models for future predictions

---

# Technologies Used

See requirements.txt
---

# Installation

Clone the repository

```bash
git clone <git@github.com:Gunay-Bayramova/immoeliza_ml.git>
```

Install dependencies

```bash
pip install -r requirements.txt
```

After cloning the repository, run python src/train.py to generate the trained model files in the models/ directory.

---
# Dataset

The project uses the cleaned ImmoEliza sale dataset.

Target variable:

```
price
```

The dataset contains information about Belgian residential properties:

## Missing values	Decision
< 20%	Keep and impute (fill missing values)
20–60%	Keep only if the feature is important 
Above 60%	Drop the column

## kept: 
price (target)
longitude
latitude
province
property_type
property_subtype
property_condition
livable_surface
number_of_bedrooms
number_of_bathrooms
date_of_construction
elevator
land_surface

## dropped
seller_id
transaction_type
street
street_number
postal_code
availability
furnished
energy_consumption
balcony
swimming_pool
garage
garden
terrace

## Reasoning for the dropped values: Feature selection
The following columns were removed before training the model:

- seller_id: Identifier, not predictive.
- transaction_type: Only one unique value ("Sale").
- street: Too many unique values for the first model.
- street_number: House number is not predictive.
- postal_code: Redundant with latitude, longitude and province.
- balcony: More than 60% missing values.
- swimming_pool: More than 60% missing values.
- availability: High percentage of missing values and low predictive value.
- furnished: High percentage of missing values and limited influence on price.
- energy_consumption: Approximately 75% missing values, making imputation unreliable.

---

# Outlier Removal 

the following listing were removed before data preprossesing

- listings equal/above 8 millions euros (lots of missing values, very few information with etremly high price)
- listings prices below €10,000" (Six rows out of ~9,700 (0.06%) — cheap to remove, poisonous to keep, since a €2,500 "mansion" pulls your model's understanding of what mansions cost in absurd directions.)

# Data Preprocessing

The preprocessing pipeline performs:

- removal of unnecessary columns
- handling missing values
- median imputation for numerical variables
- most frequent imputation for categorical variables
- one-hot encoding of categorical features
- feature scaling for numerical features
- removal of suspicious properties priced above €8 million

---

# Machine Learning Workflow

1. Load dataset
2. Clean data
3. Remove suspicious outliers
4. Train/Test split
5. Preprocessing using ColumnTransformer
6. Train multiple models
7. Evaluate performance
8. Cross-validation
9. Save trained models

---

# Models

Three regression models were implemented.

## Linear Regression

A simple baseline model used for comparison.

---

## Decision Tree Regressor

A more flexible model capable of learning nonlinear relationships, but prone to overfitting.

---

## Random Forest Regressor

An ensemble model combining multiple decision trees.

This model achieved the best performance.

---

# Results

| Model | Performance |
|---------|------------|
| Linear Regression | Baseline model |
| Decision Tree | Strong overfitting |
| Random Forest | Best overall performance |

Final Random Forest results after removing suspicious outliers:

- MAE ≈ **87,523 €**
- RMSE ≈ **184,355 €**
- Test R² ≈ **0.69**
- Train R² ≈ **0.96**

Cross-validation (5-fold):

- Mean R² ≈ **0.53**
- Standard deviation ≈ **0.13**

These results indicate that the Random Forest model provides the best predictive performance while still showing some degree of overfitting.

Decision Tree: 0.9999 → 0.11 — catastrophic overfit. Memorized every training house, learned almost nothing general.
Random Forest: 0.95 → 0.73 — real gap, so yes, overfitting — but the test score is the best in the table. An imperfect model can still be the most useful one.
XGBoost: 0.97 → 0.71 — same story as RF, slightly worse test.

---

# Saved Models

The trained models are stored in the `models/` directory.

```
linear_regression_model.joblib

decision_tree_model.joblib

random_forest_model.joblib
```

These models can be loaded directly using Joblib for future predictions.

---

# Running the Project

Train the models

```bash
python src/train.py
```

Generate predictions

```bash
python src/predict.py
```

Evaluate the models

```bash
python src/evaluate.py
```

---

# The tools used

| Tool                  | What it does                                                    |
| --------------------- | --------------------------------------------------------------- |
| **Pipeline**          | Connects several preprocessing steps into one workflow.         |
| **ColumnTransformer** | Sends numerical and categorical columns to different pipelines. |
| **SimpleImputer**     | Fills in missing values.                                        |
| **StandardScaler**    | Scales numerical features to a similar range.                   |
| **OneHotEncoder**     | Converts categorical values into numerical columns.             |

---

# Future Improvements

Possible future improvements include:

- Hyperparameter tuning
- Additional feature engineering
- More advanced ensemble methods
- XGBoost or LightGBM
- Better handling of remaining outliers
- Deployment as a web application

---

# Author

**Gunay Bayramova**

BeCode AI Bootcamp – Machine Learning Project