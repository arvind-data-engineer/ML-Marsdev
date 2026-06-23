import logging
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from geopy.distance import geodesic

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "Models" / "Model.pkl"
CSV_PATH = PROJECT_ROOT / "Data" / "Processed.csv"
DEFAULT_RECOMMENDATIONS = 5
DISTANCE_WEIGHT = 0.2
RATING_WEIGHT = 1 - DISTANCE_WEIGHT

model_artifact = None
data = None
product_catalog = None


def load_model():
    global model_artifact
    try:
        with open(MODEL_PATH, "rb") as file:
            model_artifact = pickle.load(file)
        app.logger.info("Model loaded successfully from %s.", MODEL_PATH)
    except FileNotFoundError:
        app.logger.error("Model file not found at %s.", MODEL_PATH)


def load_data():
    global data, product_catalog
    try:
        data = pd.read_csv(CSV_PATH)
        product_catalog = build_product_catalog(data)
        app.logger.info("CSV file loaded successfully with %s products.", len(product_catalog))
    except FileNotFoundError:
        app.logger.error("CSV file not found at %s.", CSV_PATH)
    except ValueError as exc:
        app.logger.error("Invalid product CSV: %s", exc)


def build_product_catalog(raw_data):
    required_columns = {"product_id", "Product_name", "latitude", "longitude"}
    missing_columns = required_columns.difference(raw_data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    clean_data = raw_data.copy()
    clean_data["latitude"] = pd.to_numeric(clean_data["latitude"], errors="coerce")
    clean_data["longitude"] = pd.to_numeric(clean_data["longitude"], errors="coerce")
    clean_data = clean_data.dropna(subset=["product_id", "latitude", "longitude"])

    optional_columns = ["background_color", "image"]
    aggregation = {
        "Product_name": "first",
        "latitude": "first",
        "longitude": "first",
    }
    aggregation.update({column: "first" for column in optional_columns if column in clean_data.columns})
    return clean_data.groupby("product_id", as_index=False).agg(aggregation)


def parse_request_args():
    try:
        user_id = int(request.args.get("user_id", default=123))
        latitude = float(request.args.get("latitude", default=0.0))
        longitude = float(request.args.get("longitude", default=0.0))
        limit = int(request.args.get("limit", default=DEFAULT_RECOMMENDATIONS))
    except (TypeError, ValueError) as exc:
        raise ValueError("user_id, latitude, longitude, and limit must be numeric.") from exc

    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90.")
    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180.")
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50.")

    return user_id, latitude, longitude, limit


def to_json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def get_seen_product_ids(user_id):
    if data is None or "is_purchased" not in data.columns:
        return set()

    user_rows = data[data["user_id"] == user_id]
    purchased_flags = pd.to_numeric(user_rows["is_purchased"], errors="coerce").fillna(0) > 0
    purchased_rows = user_rows[purchased_flags]
    return set(purchased_rows["product_id"])


def estimate_rating(user_id, product_id):
    if model_artifact is None:
        return 0.0

    user_index = model_artifact.get("user_id_to_index", {}).get(user_id)
    product_index = model_artifact.get("product_id_to_index", {}).get(product_id)
    predicted_matrix = model_artifact.get("predicted_matrix")

    if user_index is not None and product_index is not None and predicted_matrix is not None:
        return float(predicted_matrix[user_index, product_index])

    popularity = model_artifact.get("product_popularity", {}).get(product_id)
    if popularity is not None:
        return float(popularity)

    return float(model_artifact.get("global_mean", 0.0))


def score_distance(latitude, longitude, product_latitude, product_longitude):
    distance_km = geodesic((latitude, longitude), (product_latitude, product_longitude)).kilometers
    proximity_score = 1 / (1 + distance_km / 10)
    return distance_km, proximity_score


def get_top_recommendations(user_id, latitude, longitude, limit=DEFAULT_RECOMMENDATIONS):
    if model_artifact is None or product_catalog is None:
        app.logger.error("Model or data not found.")
        return []

    seen_product_ids = get_seen_product_ids(user_id)
    recommendations = []

    for product in product_catalog.itertuples(index=False):
        product_id = getattr(product, "product_id")
        if product_id in seen_product_ids:
            continue

        estimated_rating = max(0, min(1, estimate_rating(user_id, product_id)))
        distance_km, proximity_score = score_distance(
            latitude,
            longitude,
            getattr(product, "latitude"),
            getattr(product, "longitude"),
        )
        final_score = (RATING_WEIGHT * estimated_rating) + (DISTANCE_WEIGHT * proximity_score)

        recommendation = {
            "user_id": user_id,
            "product_id": to_json_value(product_id),
            "product_name": to_json_value(getattr(product, "Product_name")),
            "estimated_rating": round(estimated_rating, 4),
            "distance_km": round(distance_km, 2),
            "score": round(final_score, 4),
        }
        if hasattr(product, "background_color"):
            recommendation["background_color"] = to_json_value(getattr(product, "background_color"))
        if hasattr(product, "image"):
            recommendation["image"] = to_json_value(getattr(product, "image"))
        recommendations.append(recommendation)

    recommendations.sort(key=lambda item: item["score"], reverse=True)
    return recommendations[:limit]


@app.before_request
def ensure_assets_loaded():
    if model_artifact is None:
        load_model()
    if product_catalog is None:
        load_data()


@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        user_id, latitude, longitude, limit = parse_request_args()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    recommendations = get_top_recommendations(user_id, latitude, longitude, limit)
    if not recommendations and (model_artifact is None or product_catalog is None):
        return jsonify({"error": "Recommendation assets are not available. Train the model first."}), 503

    return jsonify({"recommendations": recommendations})


if __name__ == "__main__":
    load_model()
    load_data()
    app.run(debug=True, use_reloader=False)
