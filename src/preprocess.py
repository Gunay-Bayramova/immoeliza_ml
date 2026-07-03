import pandas as pd


def load_and_preprocess_data(filepath):
    """
    Load the dataset, drop unnecessary columns,
    remove suspicious extreme outliers,
    and return X and y.
    """

    df = pd.read_csv(filepath)

    columns_to_drop = [
        "id",
        "locality",
        "subproperty_type",
        "equipped_kitchen",
        "open_fire",
        "swimming_pool",
        "flood_zone",
        "garden",
        "terrace",
        "surface_of_the_plot",
    ]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    df = df[df["price"] < 8_000_000].reset_index(drop=True)

    X = df.drop(columns="price")
    y = df["price"]

    return X, y
