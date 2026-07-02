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

Missing values	Decision
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
garden
terrace
elevator
land_surface

# dropped
seller_id
transaction_type
street
street_number
postal_code
balcony
swimming_pool
energy_consumption
furnished
availability


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