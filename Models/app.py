from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
from surprise import SVD
import logging
from geopy.distance import geodesic

app = Flask(__name__)

# Configure Flask logging
app.logger.setLevel(logging.INFO)

# Load the model and data lazily
MODEL_PATH = "Model.pkl"
CSV_PATH = os.path.join("Data", "Processed.csv")
model_svd = None
data = None
product_id_to_item_name = {}

def load_model():
    global model_svd
    try:
        with open(MODEL_PATH, "rb") as f:
            model_svd = pickle.load(f)
            app.logger.info("Model loaded successfully.")
    except FileNotFoundError:
        app.logger.error("Model file not found.")

def load_data():
    global data, product_id_to_item_name
    try:
        data = pd.read_csv(CSV_PATH)
        app.logger.info("CSV file loaded successfully.")
        product_id_to_item_name = data.set_index("product_id")["Product_name"].to_dict()
    except FileNotFoundError:
        app.logger.error("CSV file not found.")

def get_top_recommendations(user_id, latitude, longitude, num_recommendations=2):
    if model_svd is not None and data is not None:
        all_products = data["product_id"].unique()
        recommendations = []
        
        for item in all_products:
            product_lat = data[data["product_id"] == item]["latitude"].iloc[0]
            product_long = data[data["product_id"] == item]["longitude"].iloc[0]
            distance = geodesic((latitude, longitude), (product_lat, product_long)).kilometers
            recommendations.append({
                "user_id": user_id,
                "product_id": int(item),
                "product_name": product_id_to_item_name.get(item, "Unknown"),
                "background_color": data[data["product_id"] == item]["background_color"].iloc[0],
                "image": data[data["product_id"] == item]["image"].iloc[0],
            })
        
        # No need to sort recommendations by predicted rating
        top_recommendations = recommendations[:num_recommendations]
        
        return top_recommendations
    else:
        app.logger.error("Model or data not found.")
        return []

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = int(request.args.get("user_id", default=123))
    latitude = float(request.args.get("latitude", default=0.0))
    longitude = float(request.args.get("longitude", default=0.0))
    recommendations = get_top_recommendations(user_id, latitude, longitude)
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    load_model()
    load_data()
    app.run(debug=True, use_reloader=False)
