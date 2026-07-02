# immo-eliza-ml/

# Git Repo structure

│
├── data/
│     SaleCleanForAnalysis.csv
│
├── notebooks/
│     exploration.ipynb
│
├── src/
│     preprocess.py
│     train.py
│     predict.py
│     evaluate.py
│
├── models/
│     random_forest.pkl
│
├── README.md
│
└── requirements.txt



# the first model structure: 

Let's agree on this for our first model:

## Missing values	Decision
< 20%	Keep and impute (fill missing values)
20–60%	Keep only if the feature is important 
> 60%	Drop the column

# kept: 
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

# dropped
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


# Machine Learnin Pipeline
✅ Load data
✅ Explore data
✅ Select features

⬜ Drop unnecessary columns   ← NOW
⬜ Separate X and y
⬜ Handle missing values
⬜ Encode categorical columns
⬜ Scale numerical columns
⬜ Split train/test
⬜ Train first model
⬜ Evaluate
⬜ Improve

# we define the target
X = house information
y = price we want to predict

# the worflow
1. Load data
2. Feature selection (drop columns)
3. X / y split
4. Train / Test split
5. Build preprocessing pipeline
   • Median for numeric columns
   • Most frequent (or chosen value) for categorical columns
   • One-Hot Encoding
6. Train Linear Regression
7. Evaluate
8. Try Decision Tree
9. Try Random Forest
10. Compare models

# separate the columns 

## Numerical Pipeline
------------------
longitude
latitude
livable_surface
number_of_bedrooms
...
## Categorical Pipeline
--------------------
property_type
property_subtype
province
property_condition
elevator

# | Tool                  | What it does                                                    |
| --------------------- | --------------------------------------------------------------- |
| **Pipeline**          | Connects several preprocessing steps into one workflow.         |
| **ColumnTransformer** | Sends numerical and categorical columns to different pipelines. |
| **SimpleImputer**     | Fills in missing values.                                        |
| **StandardScaler**    | Scales numerical features to a similar range.                   |
| **OneHotEncoder**     | Converts categorical values into numerical columns.             |
