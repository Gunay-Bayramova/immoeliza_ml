import pandas as pd


def load_and_preprocess_data(filepath):
    """
    Load the dataset, drop unnecessary columns,
    remove suspicious extreme outliers,
    and return X and y.
    """

    df = pd.read_csv(filepath)

    columns_to_drop = [
        "seller_id",
        "transaction_type",
        "street",
        "street_number",
        "postal_code",
        "balcony",
        "swimming_pool",
        "availability",
        "furnished",
        "energy_consumption",
        "garden",
        "terrace",
        "garage"
     ]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    df = df[df["price"] < 8_000_000].reset_index(drop=True)

    X = df.drop(columns="price")
    y = df["price"]

    return X, y
