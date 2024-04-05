import pandas as pd
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
import os
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_from_db():
    # Retrieve database credentials from environment variables
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_PORT = os.getenv("DB_PORT")

    # Use the obtained IP address
    container_ip = "3.7.136.194"

    # Encode the password
    encoded_password = quote_plus(DB_PASSWORD)

    # Define the connection string using the IP address
    connection_string = f"postgresql://{DB_USER}:{encoded_password}@{container_ip}:{DB_PORT}/{DB_NAME}"

    # Create a connection to the PostgreSQL database
    engine = create_engine(connection_string)
    logger.info("Connected to the database.")

    # Define SQL queries for each table
    queries = {
        "auth_user": "SELECT * FROM auth_user",
        "list_product": "SELECT * FROM list_product",
        "product_items": "SELECT * FROM product_items",
        "product_category": "SELECT * FROM product_category",
        "users_favoriteproduct": "SELECT * FROM users_favoriteproduct",
        "users_favoritelocation": "SELECT *,ST_Y(ST_AsText(location)) AS latitude, ST_X(ST_AsText(location)) AS longitude FROM users_favoritelocation WHERE ST_IsValid(location)",
    }

    # Execute queries and store results in DataFrames
    dataframes = {}
    with engine.connect() as connection:
        for table, query in queries.items():
            result = connection.execute(text(query))
            dataframes[table] = pd.DataFrame(result.fetchall(), columns=result.keys())
            logger.info(f"Loaded data from {table} table.")

    return dataframes

def preprocess_data(dataframes):
    # Preprocessing User_df
    User_df = dataframes["auth_user"].copy()
    User_df = User_df.dropna(axis=1, how="all")
    User_df = User_df.rename(columns={"id": "user_id"})
    cols = ["password", "is_superuser", "last_login", "date_joined"]
    User_df = User_df.drop(cols, axis=1)
    logger.info("Preprocessed auth_user data.")

    # Preprocessing Purchase_df
    Purchase_df = dataframes["list_product"].copy()
    Purchase_df = Purchase_df.dropna(axis=1, how="all")
    Purchase_df["unit"] = pd.Categorical(Purchase_df["unit"])
    Purchase_df["unit"] = Purchase_df["unit"].cat.codes
    cols = ["id", "created_at", "updated_at"]
    Purchase_df = Purchase_df.drop(cols, axis=1)
    Purchase_df.rename(columns={"added_by_id": "user_id"}, inplace=True)
    logger.info("Preprocessed list_product data.")

    # Preprocessing Fav_location_df
    Fav_location_df = dataframes["users_favoritelocation"].copy()
    cols = ["id", "created_at", "updated_at", "label", "note"]
    Fav_location_df = Fav_location_df.drop(cols, axis=1)
    logger.info("Preprocessed users_favoritelocation data.")

    # Preprocessing Fav_product_df
    Fav_product_df = dataframes["users_favoriteproduct"].copy()
    cols = ["id", "created_at", "updated_at"]
    Fav_product_df = Fav_product_df.drop(cols, axis=1)
    logger.info("Preprocessed users_favoriteproduct data.")

    # Preprocessing Item_df
    Item_df = dataframes["product_items"].copy()
    cols = ["created_at", "updated_at", "category_id"]
    Item_df = Item_df.drop(cols, axis=1)
    Item_df = Item_df.rename(columns={"id": "product_id"})
    logger.info("Preprocessed product_items data.")

    return User_df, Purchase_df, Fav_location_df, Fav_product_df, Item_df

def merge_dataframes(User_df, Purchase_df, Fav_location_df, Fav_product_df, Item_df):
    # Merging DataFrames
    merged_df = pd.merge(User_df, Purchase_df, on="user_id", how="inner")
    merged_df = pd.merge(merged_df, Item_df, on="product_id", how="inner")
    merged_df = pd.merge(merged_df, Fav_location_df, on="user_id", how="inner")
    merged_df = pd.merge(merged_df, Fav_product_df, on="user_id", how="inner")
    logger.info("Merged all DataFrames.")

    return merged_df

def preprocess_and_save_data():
    try:
        # Load data from the database
        dataframes = load_data_from_db()

        # Preprocess data
        User_df, Purchase_df, Fav_location_df, Fav_product_df, Item_df = preprocess_data(dataframes)

        # Merge DataFrames
        merged_df = merge_dataframes(User_df, Purchase_df, Fav_location_df, Fav_product_df, Item_df)

        # Dropping duplicate location column and renaming columns
        merged_df.drop("location_y", axis=1, inplace=True)
        merged_df.rename(columns={"location_x": "favlocation", "name": "Product_name"}, inplace=True)

        # Normalize numerical variables
        scaler = StandardScaler()
        merged_df["quantity"] = scaler.fit_transform(merged_df["quantity"].values.reshape(-1, 1))

        # Encode target variable
        merged_df["is_purchased"] = merged_df["is_purchased"].astype(int)

        # Save processed data to CSV
        output_path = "Data/Processed.csv"
        merged_df.to_csv(output_path, index=False)
        logger.info("Processed data saved to %s", output_path)
    except Exception as e:
        logger.error("An error occurred while preprocessing and saving data: %s", e)

if __name__ == "__main__":
    preprocess_and_save_data()
