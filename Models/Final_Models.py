import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define file paths
DATA_PATH = "Data/Processed.csv"
MODEL_PATH = "Model.pkl"

def load_data(data_path):
    try:
        data = pd.read_csv(data_path)
        # Assuming your CSV file has columns: user_id, product_id, is_purchased, latitude, longitude
        reader = Reader(rating_scale=(0, 1))
        data_surprise = Dataset.load_from_df(data[["user_id", "product_id", "is_purchased"]], reader)
        
        # Adding latitude and longitude as additional features
        data_surprise.df = data_surprise.df.assign(latitude=data['latitude'], longitude=data['longitude'])

        trainset, testset = train_test_split(data_surprise, test_size=0.2, random_state=42)
        logger.info("Data loaded successfully.")
        return trainset, testset
    except FileNotFoundError:
        logger.error("Data file not found.")
        return None, None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None, None


def train_model(trainset):
    model_svd = SVD()
    model_svd.fit(trainset)
    logger.info("SVD model trained successfully.")
    return model_svd

def evaluate_model(model, testset):
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    logger.info(f"SVD Model - RMSE: {rmse}, MAE: {mae}")
    return rmse, mae

def save_model(model, model_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("SVD model saved to %s", model_path)

def main():
    trainset, testset = load_data(DATA_PATH)
    if trainset is not None and testset is not None:
        model = train_model(trainset)
        rmse, mae = evaluate_model(model, testset)
        save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()
