import joblib
import pandas as pd

# Load the trained model
model = joblib.load("../models/random_forest_model.joblib")

# Load the dataset and preprocess it
from preprocess import load_and_preprocess_data
X, y = load_and_preprocess_data("../data/SaleCleanForAnalysis.csv")

# Prepare the input data for prediction (assuming you have a new dataset for prediction)
predictions = model.predict(X.head())

print(predictions)