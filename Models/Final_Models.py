import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
import os

# Load the dataset
data = pd.read_csv("Data/Processed.csv")

# Define the rating scale
reader = Reader(rating_scale=(0, 1))

# Load the dataset into the Surprise format
data_surprise = Dataset.load_from_df(data[['user_id', 'product_id', 'is_purchased']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data_surprise, test_size=0.2, random_state=42)

# Train the SVD model
model_svd = SVD()
model_svd.fit(trainset)

# Make predictions on the test set
predictions_svd = model_svd.test(testset)

# Evaluate the model
rmse_svd = accuracy.rmse(predictions_svd)
mae_svd = accuracy.mae(predictions_svd)
#print(f'SVD Model - RMSE: {rmse_svd}, MAE: {mae_svd}')

# Define the filename for the pickle file in the current directory
model_svd_filename = 'Model.pkl'

# Dump the trained SVD model into the pickle file
with open(model_svd_filename, 'wb') as f:
    pickle.dump(model_svd, f)

print("SVD model saved to", model_svd_filename)
