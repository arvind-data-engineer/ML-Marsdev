import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "Processed.csv"
MODEL_PATH = PROJECT_ROOT / "Models" / "Model.pkl"
EXAMPLE_DATA_PATH = PROJECT_ROOT / "Data" / "Processed.example.csv"
REQUIRED_COLUMNS = {"user_id", "product_id", "is_purchased"}
RANDOM_STATE = 42


def load_interactions(data_path):
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("Data file not found at %s.", data_path)
        logger.error("Create it by running: python Models/EDA.py")
        logger.error("Schema example: %s", EXAMPLE_DATA_PATH)
        return None

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        logger.error("Missing required columns: %s", sorted(missing_columns))
        logger.error("Required training columns are: %s", sorted(REQUIRED_COLUMNS))
        return None

    interactions = data[["user_id", "product_id", "is_purchased"]].dropna().copy()
    interactions["is_purchased"] = pd.to_numeric(interactions["is_purchased"], errors="coerce").fillna(0)
    interactions["is_purchased"] = interactions["is_purchased"].clip(0, 1)
    interactions = interactions.groupby(["user_id", "product_id"], as_index=False)["is_purchased"].max()

    logger.info("Loaded %s interaction rows from %s.", len(interactions), data_path)
    return interactions


def build_interaction_matrix(interactions):
    matrix = interactions.pivot_table(
        index="user_id",
        columns="product_id",
        values="is_purchased",
        fill_value=0,
        aggfunc="max",
    )
    return matrix.astype(float)


def train_model(interaction_matrix):
    max_components = min(interaction_matrix.shape) - 1
    n_components = max(1, min(50, max_components))

    if max_components < 1:
        logger.warning("Not enough users/products for matrix factorization; using popularity-only fallback.")
        return None

    model = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    user_factors = model.fit_transform(interaction_matrix)
    product_factors = model.components_.T
    reconstructed = np.clip(user_factors @ product_factors.T, 0, 1)

    logger.info("Trained TruncatedSVD model with %s components.", n_components)
    return model, user_factors, product_factors, reconstructed


def evaluate_model(interactions, interaction_matrix, reconstructed):
    if reconstructed is None or len(interactions) < 2:
        logger.info("Evaluation skipped because there are too few interactions.")
        return None, None

    _, test_rows = train_test_split(interactions, test_size=0.2, random_state=RANDOM_STATE)
    user_positions = {user_id: index for index, user_id in enumerate(interaction_matrix.index)}
    product_positions = {product_id: index for index, product_id in enumerate(interaction_matrix.columns)}

    actual = []
    predicted = []
    for row in test_rows.itertuples(index=False):
        user_index = user_positions.get(row.user_id)
        product_index = product_positions.get(row.product_id)
        if user_index is None or product_index is None:
            continue
        actual.append(row.is_purchased)
        predicted.append(reconstructed[user_index, product_index])

    if not actual:
        logger.info("Evaluation skipped because no test rows matched the trained matrix.")
        return None, None

    rmse = mean_squared_error(actual, predicted) ** 0.5
    mae = mean_absolute_error(actual, predicted)
    logger.info("Model evaluation - RMSE: %.4f, MAE: %.4f", rmse, mae)
    return rmse, mae


def build_model_artifact(interactions):
    interaction_matrix = build_interaction_matrix(interactions)
    trained = train_model(interaction_matrix)

    if trained is None:
        reconstructed = None
        product_scores = interaction_matrix.mean(axis=0).to_dict()
        user_factors = None
        product_factors = None
    else:
        _, user_factors, product_factors, reconstructed = trained
        product_scores = interaction_matrix.mean(axis=0).to_dict()
        evaluate_model(interactions, interaction_matrix, reconstructed)

    return {
        "model_type": "sklearn_truncated_svd",
        "user_ids": list(interaction_matrix.index),
        "product_ids": list(interaction_matrix.columns),
        "user_id_to_index": {user_id: index for index, user_id in enumerate(interaction_matrix.index)},
        "product_id_to_index": {product_id: index for index, product_id in enumerate(interaction_matrix.columns)},
        "user_factors": user_factors,
        "product_factors": product_factors,
        "predicted_matrix": reconstructed,
        "product_popularity": product_scores,
        "global_mean": float(interaction_matrix.values.mean()),
    }


def save_model(artifact, model_path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(artifact, file)
    logger.info("Recommendation model saved to %s", model_path)


def main():
    interactions = load_interactions(DATA_PATH)
    if interactions is None or interactions.empty:
        raise SystemExit(
            "Training skipped. Expected Data/Processed.csv with columns: "
            "user_id, product_id, is_purchased. Run Models/EDA.py first or create the CSV manually."
        )

    artifact = build_model_artifact(interactions)
    save_model(artifact, MODEL_PATH)


if __name__ == "__main__":
    main()
