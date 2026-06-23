import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Data" / "Processed.csv"
MODEL_PATH = PROJECT_ROOT / "Models" / "Model.pkl"
EXAMPLE_DATA_PATH = PROJECT_ROOT / "Data" / "Processed.example.csv"
REQUIRED_COLUMNS = {"user_id", "product_id", "is_purchased"}
MODEL_VERSION = "v5_content_features"
RANDOM_STATE = 42
POPULARITY_SMOOTHING = 5
MAX_SIMILAR_ITEMS = 20
MAX_CONTENT_SIMILAR_ITEMS = 30
INTERACTION_SCORE_COLUMN = "interaction_score"
CONTENT_FEATURE_COLUMNS = [
    "category",
    "category_name",
    "category_id",
    "brand",
    "tags",
    "description",
    "availability",
    "seller",
    "seller_id",
    "seller_name",
    "seller_location",
    "location",
    "city",
    "state",
    "area",
]
BEHAVIOR_WEIGHTS = {
    "is_purchased": 1.0,
    "is_favorite": 0.8,
    "favorite": 0.8,
    "added_to_cart": 0.6,
    "is_cart": 0.6,
    "cart_count": 0.6,
    "viewed": 0.2,
    "view_count": 0.2,
}


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
    interactions[INTERACTION_SCORE_COLUMN] = build_interaction_score(data)
    interactions["is_purchased"] = pd.to_numeric(interactions["is_purchased"], errors="coerce").fillna(0).clip(0, 1)
    interactions = interactions.groupby(["user_id", "product_id"], as_index=False).agg(
        is_purchased=("is_purchased", "max"),
        interaction_score=(INTERACTION_SCORE_COLUMN, "max"),
    )

    logger.info("Loaded %s interaction rows from %s.", len(interactions), data_path)
    return interactions


def build_interaction_score(data):
    if INTERACTION_SCORE_COLUMN in data.columns:
        return pd.to_numeric(data[INTERACTION_SCORE_COLUMN], errors="coerce").fillna(0).clip(0, 1)

    score = pd.Series(0.0, index=data.index)
    used_columns = []
    for column, weight in BEHAVIOR_WEIGHTS.items():
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce").fillna(0)
        if values.max() > 1:
            values = values / values.max()
        score += values.clip(0, 1) * weight
        used_columns.append(column)

    if not used_columns:
        score = pd.to_numeric(data["is_purchased"], errors="coerce").fillna(0)
        used_columns = ["is_purchased"]

    logger.info("Built interaction scores from columns: %s", ", ".join(used_columns))
    return score.clip(0, 1)


def build_interaction_matrix(interactions):
    matrix = interactions.pivot_table(
        index="user_id",
        columns="product_id",
        values=INTERACTION_SCORE_COLUMN,
        fill_value=0,
        aggfunc="max",
    )
    return matrix.astype(float)


def build_item_similarity(interaction_matrix):
    product_ids = list(interaction_matrix.columns)
    if len(product_ids) < 2:
        return {}

    similarity_matrix = cosine_similarity(interaction_matrix.T)
    similar_items = {}

    for product_index, product_id in enumerate(product_ids):
        similarities = []
        for other_index, other_product_id in enumerate(product_ids):
            if product_index == other_index:
                continue
            score = float(similarity_matrix[product_index, other_index])
            if score > 0:
                similarities.append((other_product_id, score))

        similarities.sort(key=lambda item: item[1], reverse=True)
        similar_items[product_id] = similarities[:MAX_SIMILAR_ITEMS]

    return similar_items


def build_user_positive_items(interactions):
    positive_rows = interactions[interactions[INTERACTION_SCORE_COLUMN] > 0]
    return positive_rows.groupby("user_id")["product_id"].apply(list).to_dict()


def find_category_column(data):
    for column in ["category", "category_name", "category_id"]:
        if column in data.columns:
            return column
    return None


def build_category_affinity(data):
    category_column = find_category_column(data)
    if category_column is None or "is_purchased" not in data.columns:
        return {}, {}

    clean_data = data[["user_id", "product_id", "is_purchased", category_column]].dropna().copy()
    clean_data[INTERACTION_SCORE_COLUMN] = build_interaction_score(data).loc[clean_data.index]
    positive_rows = clean_data[clean_data[INTERACTION_SCORE_COLUMN] > 0]
    if positive_rows.empty:
        return {}, {}

    user_category_counts = positive_rows.groupby(["user_id", category_column]).size()
    category_affinity = {}
    for user_id, user_counts in user_category_counts.groupby(level=0):
        counts = user_counts.droplevel(0)
        total = counts.sum()
        if total:
            category_affinity[user_id] = {category: float(count / total) for category, count in counts.items()}

    product_categories = (
        clean_data.dropna(subset=[category_column])
        .drop_duplicates("product_id")
        .set_index("product_id")[category_column]
        .to_dict()
    )
    return category_affinity, product_categories


def build_product_content_profiles(data):
    available_columns = [column for column in CONTENT_FEATURE_COLUMNS if column in data.columns]
    if not available_columns:
        return {}, {}, []

    product_data = data[["product_id", *available_columns]].drop_duplicates("product_id").copy()
    content_parts = []
    for column in available_columns:
        values = product_data[column].fillna("").astype(str)
        content_parts.append(values.str.replace(r"[,|;/]", " ", regex=True))

    if "price" in data.columns:
        price_data = data[["product_id", "price"]].drop_duplicates("product_id").copy()
        price_data["price"] = pd.to_numeric(price_data["price"], errors="coerce")
        product_data = product_data.merge(price_data, on="product_id", how="left")
        price_count = int(product_data["price"].notna().sum())
        if price_count > 1:
            tier_count = min(4, price_count)
            product_data["price_tier"] = pd.qcut(
                product_data["price"].rank(method="first"),
                q=tier_count,
                labels=["budget", "value", "premium", "luxury"][:tier_count],
                duplicates="drop",
            )
            content_parts.append(product_data["price_tier"].astype(str).replace("nan", ""))
            available_columns.append("price_tier")

    product_data["content_text"] = pd.concat(content_parts, axis=1).agg(" ".join, axis=1).str.lower().str.strip()
    product_content = product_data.set_index("product_id")["content_text"].to_dict()

    if not product_data["content_text"].str.len().gt(0).any():
        return product_content, {}, available_columns

    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    content_matrix = vectorizer.fit_transform(product_data["content_text"])
    similarity_matrix = cosine_similarity(content_matrix)
    product_ids = list(product_data["product_id"])
    similar_content_items = {}

    for product_index, product_id in enumerate(product_ids):
        similarities = []
        for other_index, other_product_id in enumerate(product_ids):
            if product_index == other_index:
                continue
            score = float(similarity_matrix[product_index, other_index])
            if score > 0:
                similarities.append((other_product_id, score))

        similarities.sort(key=lambda item: item[1], reverse=True)
        similar_content_items[product_id] = similarities[:MAX_CONTENT_SIMILAR_ITEMS]

    return product_content, similar_content_items, available_columns


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
        actual.append(getattr(row, INTERACTION_SCORE_COLUMN))
        predicted.append(reconstructed[user_index, product_index])

    if not actual:
        logger.info("Evaluation skipped because no test rows matched the trained matrix.")
        return None, None

    rmse = mean_squared_error(actual, predicted) ** 0.5
    mae = mean_absolute_error(actual, predicted)
    logger.info("Model evaluation - RMSE: %.4f, MAE: %.4f", rmse, mae)
    return rmse, mae


def build_model_artifact(interactions, raw_data):
    interaction_matrix = build_interaction_matrix(interactions)
    trained = train_model(interaction_matrix)
    similar_items = build_item_similarity(interaction_matrix)
    user_positive_items = build_user_positive_items(interactions)
    category_affinity, product_categories = build_category_affinity(raw_data)
    product_content, similar_content_items, content_feature_columns = build_product_content_profiles(raw_data)
    product_stats = interactions.groupby("product_id")[INTERACTION_SCORE_COLUMN].agg(["sum", "count"])
    global_mean = float(interaction_matrix.values.mean())
    smoothed_popularity = (
        (product_stats["sum"] + POPULARITY_SMOOTHING * global_mean)
        / (product_stats["count"] + POPULARITY_SMOOTHING)
    ).to_dict()
    purchase_stats = interactions.groupby("product_id")["is_purchased"].agg(["sum", "count"])
    product_purchase_count = purchase_stats["sum"].astype(int).to_dict()
    product_behavior_score = product_stats["sum"].to_dict()
    product_interaction_count = product_stats["count"].astype(int).to_dict()
    user_activity = interactions.groupby("user_id")[INTERACTION_SCORE_COLUMN].agg(["sum", "count"])
    user_purchase_count = interactions.groupby("user_id")["is_purchased"].sum().astype(int).to_dict()

    if trained is None:
        reconstructed = None
        user_factors = None
        product_factors = None
    else:
        _, user_factors, product_factors, reconstructed = trained
        evaluate_model(interactions, interaction_matrix, reconstructed)

    return {
        "model_type": "sklearn_truncated_svd",
        "version": MODEL_VERSION,
        "user_ids": list(interaction_matrix.index),
        "product_ids": list(interaction_matrix.columns),
        "user_id_to_index": {user_id: index for index, user_id in enumerate(interaction_matrix.index)},
        "product_id_to_index": {product_id: index for index, product_id in enumerate(interaction_matrix.columns)},
        "user_factors": user_factors,
        "product_factors": product_factors,
        "predicted_matrix": reconstructed,
        "product_popularity": smoothed_popularity,
        "similar_items": similar_items,
        "user_positive_items": user_positive_items,
        "category_affinity": category_affinity,
        "product_categories": product_categories,
        "product_content": product_content,
        "similar_content_items": similar_content_items,
        "content_feature_columns": content_feature_columns,
        "product_purchase_count": product_purchase_count,
        "product_behavior_score": product_behavior_score,
        "product_interaction_count": product_interaction_count,
        "user_purchase_count": user_purchase_count,
        "user_behavior_score": user_activity["sum"].to_dict(),
        "user_interaction_count": user_activity["count"].astype(int).to_dict(),
        "global_mean": global_mean,
        "popularity_smoothing": POPULARITY_SMOOTHING,
        "interaction_score_column": INTERACTION_SCORE_COLUMN,
        "behavior_weights": BEHAVIOR_WEIGHTS,
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

    raw_data = pd.read_csv(DATA_PATH)
    artifact = build_model_artifact(interactions, raw_data)
    save_model(artifact, MODEL_PATH)


if __name__ == "__main__":
    main()
