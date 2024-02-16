from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
from surprise import SVD

app = Flask(__name__)

# Load the model from the pickled file
model_path = 'Model.pkl'
try:
    with open(model_path, 'rb') as f:
        model_svd = pickle.load(f)
except FileNotFoundError:
    model_svd = None

# Load the processed product data
csv_path = 'Processed.csv'
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    data = None

# Get mapping of product IDs to item names
if data is not None:
    product_id_to_item_name = data.set_index('product_id')['Product_name'].to_dict()
else:
    product_id_to_item_name = {}

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id', default=123))
    if model_svd is not None and data is not None:
        # Get all unique product IDs from the original DataFrame
        all_products = data['product_id'].unique()

        # SVD recommendations
        predicted_ratings_svd = [(item, model_svd.predict(user_id, item).est) for item in all_products]
        sorted_ratings_svd = sorted(predicted_ratings_svd, key=lambda x: x[1], reverse=True)
        top_recommendations_svd = [product_id_to_item_name.get(item, "Unknown") for item, rating in sorted_ratings_svd[:2]]

        return jsonify({"recommendations": top_recommendations_svd})
    else:
        return jsonify({"error": "Model or data not found."})

if __name__ == '__main__':
    app.run(debug=True)
