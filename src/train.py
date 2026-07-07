from preprocess import load_and_preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import joblib

# Load the data and preprocess it
X, y = load_and_preprocess_data("../data/SaleCleanForAnalysis.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate numerical and categorical columns
numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = X.select_dtypes(include=["object", "string"]).columns

# Pipeline for numerical features
numerical_pipeline = Pipeline([
    ("fill_nan", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Pipeline for categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ]
)

# Define pipelines for different models
linear_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

decision_tree_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(random_state=42))
])

random_forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# Train the models
linear_pipeline.fit(X_train, y_train)
decision_tree_pipeline.fit(X_train, y_train)
random_forest_pipeline.fit(X_train, y_train)    

# Save the trained models using joblib
joblib.dump(linear_pipeline, "../models/linear_regression_model.joblib")
joblib.dump(decision_tree_pipeline, "../models/decision_tree_model.joblib")
joblib.dump(random_forest_pipeline, "../models/random_forest_model.joblib") 

print("Models trained and saved successfully.")