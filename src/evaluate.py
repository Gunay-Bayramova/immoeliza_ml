import joblib
import numpy as np
import pandas as pd

from preprocess import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


X, y = load_and_preprocess_data("../data/SaleCleanForAnalysis.csv")

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2,random_state=42)

models = {
    "Linear Regression": joblib.load("../models/linear_regression_model.joblib"),
    "Decision Tree": joblib.load("../models/decision_tree_model.joblib"),
    "Random Forest": joblib.load("../models/random_forest_model.joblib"),
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2
    })

results_df = pd.DataFrame(results)

print(results_df)