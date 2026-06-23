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
MAX_LIMIT = 50
USER_CONFIDENCE_INTERACTIONS = 5
COLLABORATIVE_WEIGHT = 0.45
SIMILARITY_WEIGHT = 0.25
CONTENT_WEIGHT = 0.10
POPULARITY_WEIGHT = 0.10
RECENCY_WEIGHT = 0.10
DIVERSITY_PENALTY = 0.05

model_artifact = None
model_last_modified = None
data = None
product_catalog = None
data_last_modified = None


def load_model():
    global model_artifact, model_last_modified
    try:
        with open(MODEL_PATH, "rb") as file:
            model_artifact = pickle.load(file)
        model_last_modified = MODEL_PATH.stat().st_mtime
        app.logger.info("Model loaded successfully from %s.", MODEL_PATH)
    except FileNotFoundError:
        model_artifact = None
        model_last_modified = None
        app.logger.error("Model file not found at %s.", MODEL_PATH)


def load_data():
    global data, product_catalog, data_last_modified
    try:
        data = pd.read_csv(CSV_PATH)
        product_catalog = build_product_catalog(data)
        data_last_modified = CSV_PATH.stat().st_mtime
        app.logger.info("CSV file loaded successfully with %s products.", len(product_catalog))
    except FileNotFoundError:
        data = None
        product_catalog = None
        data_last_modified = None
        app.logger.error("CSV file not found at %s.", CSV_PATH)
    except ValueError as exc:
        data = None
        product_catalog = None
        data_last_modified = None
        app.logger.error("Invalid product CSV: %s", exc)


def reload_assets_if_changed():
    if MODEL_PATH.exists():
        current_model_mtime = MODEL_PATH.stat().st_mtime
        if model_artifact is None or model_last_modified != current_model_mtime:
            load_model()
    elif model_artifact is not None:
        load_model()

    if CSV_PATH.exists():
        current_data_mtime = CSV_PATH.stat().st_mtime
        if product_catalog is None or data_last_modified != current_data_mtime:
            load_data()
    elif product_catalog is not None:
        load_data()


def build_product_catalog(raw_data):
    required_columns = {"product_id", "Product_name", "latitude", "longitude"}
    missing_columns = required_columns.difference(raw_data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    clean_data = raw_data.copy()
    clean_data["latitude"] = pd.to_numeric(clean_data["latitude"], errors="coerce")
    clean_data["longitude"] = pd.to_numeric(clean_data["longitude"], errors="coerce")
    clean_data = clean_data.dropna(subset=["product_id", "latitude", "longitude"])

    optional_columns = [
        "background_color",
        "image",
        "category",
        "category_name",
        "category_id",
        "brand",
        "price",
        "unit",
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
    aggregation = {
        "Product_name": "first",
        "latitude": "first",
        "longitude": "first",
    }
    aggregation.update({column: "first" for column in optional_columns if column in clean_data.columns})
    return clean_data.groupby("product_id", as_index=False).agg(aggregation)


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_request_args():
    try:
        user_id = int(request.args.get("user_id", default=123))
        latitude = float(request.args.get("latitude", default=0.0))
        longitude = float(request.args.get("longitude", default=0.0))
        limit = int(request.args.get("limit", default=DEFAULT_RECOMMENDATIONS))
        distance_weight = float(request.args.get("distance_weight", default=DISTANCE_WEIGHT))
        diversity_weight = float(request.args.get("diversity_weight", default=DIVERSITY_PENALTY))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "user_id, latitude, longitude, limit, distance_weight, and diversity_weight must be numeric."
        ) from exc

    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90.")
    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180.")
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}.")
    if not 0 <= distance_weight <= 1:
        raise ValueError("distance_weight must be between 0 and 1.")
    if not 0 <= diversity_weight <= 0.5:
        raise ValueError("diversity_weight must be between 0 and 0.5.")

    exclude_seen = parse_bool(request.args.get("exclude_seen"), default=True)

    return user_id, latitude, longitude, limit, distance_weight, diversity_weight, exclude_seen


def to_json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def lookup(mapping, key, default=None):
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    clean_key = to_json_value(key)
    if clean_key in mapping:
        return mapping[clean_key]
    string_key = str(clean_key)
    if string_key in mapping:
        return mapping[string_key]
    return default


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

    user_index = lookup(model_artifact.get("user_id_to_index", {}), user_id)
    product_index = lookup(model_artifact.get("product_id_to_index", {}), product_id)
    predicted_matrix = model_artifact.get("predicted_matrix")

    if user_index is not None and product_index is not None and predicted_matrix is not None:
        return float(predicted_matrix[user_index, product_index])

    popularity = lookup(model_artifact.get("product_popularity", {}), product_id)
    if popularity is not None:
        return float(popularity)

    return float(model_artifact.get("global_mean", 0.0))


def get_popularity_score(product_id):
    if model_artifact is None:
        return 0.0
    popularity = lookup(model_artifact.get("product_popularity", {}), product_id)
    if popularity is not None:
        return float(popularity)
    return float(model_artifact.get("global_mean", 0.0))


def get_user_confidence(user_id):
    if model_artifact is None:
        return 0.0
    interaction_count = lookup(model_artifact.get("user_interaction_count", {}), user_id, 0)
    try:
        interaction_count = float(interaction_count)
    except (TypeError, ValueError):
        interaction_count = 0.0
    return min(1.0, interaction_count / USER_CONFIDENCE_INTERACTIONS)


def get_similarity_score(user_id, product_id):
    if model_artifact is None:
        return 0.0

    user_positive_items = lookup(model_artifact.get("user_positive_items", {}), user_id, [])
    if not user_positive_items:
        return 0.0

    positive_items = set(user_positive_items)
    similar_items = model_artifact.get("similar_items", {})
    best_score = 0.0

    for positive_product_id in positive_items:
        for similar_product_id, score in lookup(similar_items, positive_product_id, []):
            if similar_product_id == product_id:
                best_score = max(best_score, float(score))

    return max(0.0, min(1.0, best_score))


def get_recent_interest_score(user_id, product_id):
    if model_artifact is None:
        return 0.0

    recent_positive_items = lookup(model_artifact.get("user_recent_positive_items", {}), user_id, [])
    if not recent_positive_items:
        return 0.0

    recent_items = set(recent_positive_items)
    similar_items = model_artifact.get("similar_items", {})
    similar_content_items = model_artifact.get("similar_content_items", {})
    best_score = 0.0

    for recent_product_id in recent_items:
        for similar_product_id, score in lookup(similar_items, recent_product_id, []):
            if similar_product_id == product_id:
                best_score = max(best_score, float(score))
        for similar_product_id, score in lookup(similar_content_items, recent_product_id, []):
            if similar_product_id == product_id:
                best_score = max(best_score, float(score))

    return max(0.0, min(1.0, best_score))


def get_content_score(user_id, product_id):
    if model_artifact is None:
        return 0.0

    product_category = lookup(model_artifact.get("product_categories", {}), product_id)
    user_affinity = lookup(model_artifact.get("category_affinity", {}), user_id, {})
    category_score = 0.0
    if product_category is not None:
        category_score = float(lookup(user_affinity, product_category, 0.0))

    content_similarity_score = get_content_similarity_score(user_id, product_id)
    return max(category_score, content_similarity_score)


def get_content_similarity_score(user_id, product_id):
    if model_artifact is None:
        return 0.0

    user_positive_items = lookup(model_artifact.get("user_positive_items", {}), user_id, [])
    if not user_positive_items:
        return 0.0

    positive_items = set(user_positive_items)
    similar_content_items = model_artifact.get("similar_content_items", {})
    best_score = 0.0

    for positive_product_id in positive_items:
        for similar_product_id, score in lookup(similar_content_items, positive_product_id, []):
            if similar_product_id == product_id:
                best_score = max(best_score, float(score))

    return max(0.0, min(1.0, best_score))


def build_preference_score(user_id, product_id):
    estimated_rating = max(0, min(1, estimate_rating(user_id, product_id)))
    similarity_score = get_similarity_score(user_id, product_id)
    recent_interest_score = get_recent_interest_score(user_id, product_id)
    content_score = get_content_score(user_id, product_id)
    popularity_score = max(0, min(1, get_popularity_score(product_id)))
    user_confidence = get_user_confidence(user_id)

    personalized_score = (
        (COLLABORATIVE_WEIGHT * estimated_rating)
        + (SIMILARITY_WEIGHT * similarity_score)
        + (CONTENT_WEIGHT * content_score)
        + (POPULARITY_WEIGHT * popularity_score)
        + (RECENCY_WEIGHT * recent_interest_score)
    )
    cold_start_score = (0.65 * popularity_score) + (0.25 * content_score) + (0.10 * recent_interest_score)
    preference_score = (user_confidence * personalized_score) + ((1 - user_confidence) * cold_start_score)

    return {
        "estimated_rating": estimated_rating,
        "similarity_score": similarity_score,
        "recent_interest_score": recent_interest_score,
        "content_score": content_score,
        "popularity_score": popularity_score,
        "preference_score": max(0.0, min(1.0, preference_score)),
        "user_confidence": user_confidence,
    }


def get_recommendation_reason(
    user_confidence,
    similarity_score,
    recent_interest_score,
    content_score,
    popularity_score,
    proximity_score,
):
    if recent_interest_score >= 0.5:
        return "recent_interest"
    if user_confidence >= 0.8:
        if similarity_score >= 0.5:
            return "similar_to_liked"
        if content_score >= 0.5:
            return "category_match"
        return "personalized"
    if proximity_score >= 0.5 and popularity_score >= 0.5:
        return "popular_nearby"
    if popularity_score >= 0.5:
        return "popular"
    if proximity_score >= 0.5:
        return "nearby"
    return "exploration"


def score_distance(latitude, longitude, product_latitude, product_longitude):
    distance_km = geodesic((latitude, longitude), (product_latitude, product_longitude)).kilometers
    proximity_score = 1 / (1 + distance_km / 10)
    return distance_km, proximity_score


def get_top_recommendations(
    user_id,
    latitude,
    longitude,
    limit=DEFAULT_RECOMMENDATIONS,
    distance_weight=DISTANCE_WEIGHT,
    diversity_weight=DIVERSITY_PENALTY,
    exclude_seen=True,
):
    if model_artifact is None or product_catalog is None:
        app.logger.error("Model or data not found.")
        return []

    seen_product_ids = get_seen_product_ids(user_id)
    preference_weight = 1 - distance_weight
    recommendations = []

    for product in product_catalog.itertuples(index=False):
        product_id = getattr(product, "product_id")
        if exclude_seen and product_id in seen_product_ids:
            continue

        preference = build_preference_score(user_id, product_id)
        distance_km, proximity_score = score_distance(
            latitude,
            longitude,
            getattr(product, "latitude"),
            getattr(product, "longitude"),
        )
        final_score = (preference_weight * preference["preference_score"]) + (distance_weight * proximity_score)

        recommendation = {
            "user_id": user_id,
            "product_id": to_json_value(product_id),
            "product_name": to_json_value(getattr(product, "Product_name")),
            "estimated_rating": round(preference["estimated_rating"], 4),
            "similarity_score": round(preference["similarity_score"], 4),
            "recent_interest_score": round(preference["recent_interest_score"], 4),
            "content_score": round(preference["content_score"], 4),
            "popularity_score": round(preference["popularity_score"], 4),
            "preference_score": round(preference["preference_score"], 4),
            "user_confidence": round(preference["user_confidence"], 4),
            "distance_km": round(distance_km, 2),
            "score": round(final_score, 4),
            "reason": get_recommendation_reason(
                preference["user_confidence"],
                preference["similarity_score"],
                preference["recent_interest_score"],
                preference["content_score"],
                preference["popularity_score"],
                proximity_score,
            ),
        }
        for column in product_catalog.columns:
            if column not in {"product_id", "Product_name", "latitude", "longitude"}:
                recommendation[column] = to_json_value(getattr(product, column))
        recommendations.append(recommendation)

    return diversify_recommendations(recommendations, limit, diversity_weight)


def diversify_recommendations(recommendations, limit, diversity_weight):
    ranked = sorted(recommendations, key=lambda item: item["score"], reverse=True)
    selected = []
    used_categories = set()

    while ranked and len(selected) < limit:
        best_index = 0
        best_score = None

        for index, item in enumerate(ranked):
            category = item.get("category") or item.get("category_name")
            penalty = diversity_weight if category in used_categories else 0
            adjusted_score = item["score"] - penalty
            if best_score is None or adjusted_score > best_score:
                best_score = adjusted_score
                best_index = index

        selected_item = ranked.pop(best_index)
        category = selected_item.get("category") or selected_item.get("category_name")
        if category is not None:
            used_categories.add(category)
        selected.append(selected_item)

    return selected


@app.before_request
def ensure_assets_loaded():
    reload_assets_if_changed()


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "model_loaded": model_artifact is not None,
            "model_version": model_artifact.get("version") if model_artifact else None,
            "model_path": str(MODEL_PATH),
            "model_last_modified": model_last_modified,
            "data_loaded": product_catalog is not None,
            "data_path": str(CSV_PATH),
            "data_last_modified": data_last_modified,
            "product_count": int(len(product_catalog)) if product_catalog is not None else 0,
            "content_feature_columns": model_artifact.get("content_feature_columns", []) if model_artifact else [],
            "recency_half_life_days": model_artifact.get("recency_half_life_days") if model_artifact else None,
            "recent_interaction_days": model_artifact.get("recent_interaction_days") if model_artifact else None,
        }
    )


@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        user_id, latitude, longitude, limit, distance_weight, diversity_weight, exclude_seen = parse_request_args()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    recommendations = get_top_recommendations(
        user_id,
        latitude,
        longitude,
        limit,
        distance_weight,
        diversity_weight,
        exclude_seen,
    )
    if not recommendations and (model_artifact is None or product_catalog is None):
        return jsonify({"error": "Recommendation assets are not available. Train the model first."}), 503

    return jsonify(
        {
            "recommendations": recommendations,
            "meta": {
                "count": len(recommendations),
                "distance_weight": distance_weight,
                "diversity_weight": diversity_weight,
                "exclude_seen": exclude_seen,
                "model_version": model_artifact.get("version") if model_artifact else None,
                "model_last_modified": model_last_modified,
                "data_last_modified": data_last_modified,
                "content_feature_columns": model_artifact.get("content_feature_columns", []) if model_artifact else [],
                "recency_half_life_days": model_artifact.get("recency_half_life_days") if model_artifact else None,
                "recent_interaction_days": model_artifact.get("recent_interaction_days") if model_artifact else None,
            },
        }
    )


if __name__ == "__main__":
    load_model()
    load_data()
    app.run(debug=True, use_reloader=False)
