import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from database import create_database_engine, get_env, load_project_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "Data" / "Processed.csv"
REQUIRED_OUTPUT_COLUMNS = {
    "user_id",
    "product_id",
    "is_purchased",
    "Product_name",
    "latitude",
    "longitude",
}

load_project_env(PROJECT_ROOT)


def load_data_from_db():
    engine = create_database_engine()

    processed_data_query = get_env("PROCESSED_DATA_QUERY")
    if processed_data_query:
        with engine.connect() as connection:
            processed_data = pd.read_sql_query(text(processed_data_query), connection)
        logger.info("Loaded %s rows from PROCESSED_DATA_QUERY.", len(processed_data))
        return {"processed_data": processed_data}

    queries = {
        "auth_user": get_env("AUTH_USER_QUERY", "SELECT * FROM auth_user"),
        "list_product": get_env("LIST_PRODUCT_QUERY", "SELECT * FROM list_product"),
        "product_items": get_env("PRODUCT_ITEMS_QUERY", "SELECT * FROM product_items"),
        "product_category": get_env("PRODUCT_CATEGORY_QUERY", "SELECT * FROM product_category"),
        "users_favoriteproduct": get_env("USERS_FAVORITE_PRODUCT_QUERY", "SELECT * FROM users_favoriteproduct"),
        "users_favoritelocation": get_env(
            "USERS_FAVORITE_LOCATION_QUERY",
            "SELECT *, ST_Y(ST_AsText(location)) AS latitude, "
            "ST_X(ST_AsText(location)) AS longitude "
            "FROM users_favoritelocation WHERE ST_IsValid(location)",
        ),
    }

    dataframes = {}
    with engine.connect() as connection:
        for table, query in queries.items():
            dataframes[table] = pd.read_sql_query(text(query), connection)
            logger.info("Loaded %s rows from %s.", len(dataframes[table]), table)

    return dataframes


def preprocess_data(dataframes):
    user_df = dataframes["auth_user"].copy()
    user_df = user_df.dropna(axis=1, how="all")
    user_df = user_df.rename(columns={"id": "user_id"})
    user_df = user_df.drop(["password", "is_superuser", "last_login", "date_joined"], axis=1, errors="ignore")
    logger.info("Preprocessed auth_user data.")

    purchase_df = dataframes["list_product"].copy()
    purchase_df = purchase_df.dropna(axis=1, how="all")
    if "created_at" in purchase_df.columns:
        purchase_df = purchase_df.rename(columns={"created_at": "interaction_timestamp"})
    elif "updated_at" in purchase_df.columns:
        purchase_df = purchase_df.rename(columns={"updated_at": "interaction_timestamp"})
    if "unit" in purchase_df.columns:
        purchase_df["unit"] = pd.Categorical(purchase_df["unit"]).codes
    purchase_df = purchase_df.drop(["id", "created_at", "updated_at"], axis=1, errors="ignore")
    purchase_df = purchase_df.rename(columns={"added_by_id": "user_id"})
    logger.info("Preprocessed list_product data.")

    fav_location_df = dataframes["users_favoritelocation"].copy()
    fav_location_df = fav_location_df.drop(["id", "created_at", "updated_at", "label", "note"], axis=1, errors="ignore")
    logger.info("Preprocessed users_favoritelocation data.")

    fav_product_df = dataframes["users_favoriteproduct"].copy()
    fav_product_df = fav_product_df.drop(["id", "created_at", "updated_at"], axis=1, errors="ignore")
    logger.info("Preprocessed users_favoriteproduct data.")

    item_df = dataframes["product_items"].copy()
    item_df = item_df.drop(["created_at", "updated_at", "category_id"], axis=1, errors="ignore")
    item_df = item_df.rename(columns={"id": "product_id", "name": "Product_name"})
    logger.info("Preprocessed product_items data.")

    return user_df, purchase_df, fav_location_df, fav_product_df, item_df


def merge_dataframes(user_df, purchase_df, fav_location_df, fav_product_df, item_df):
    merged_df = pd.merge(user_df, purchase_df, on="user_id", how="inner")
    merged_df = pd.merge(merged_df, item_df, on="product_id", how="inner")
    merged_df = pd.merge(merged_df, fav_location_df, on="user_id", how="inner")
    merged_df = pd.merge(merged_df, fav_product_df, on="user_id", how="inner")
    logger.info("Merged all DataFrames into %s rows.", len(merged_df))
    return merged_df


def preprocess_and_save_data():
    try:
        dataframes = load_data_from_db()
        if "processed_data" in dataframes:
            merged_df = dataframes["processed_data"]
        else:
            user_df, purchase_df, fav_location_df, fav_product_df, item_df = preprocess_data(dataframes)
            merged_df = merge_dataframes(user_df, purchase_df, fav_location_df, fav_product_df, item_df)

            if "location_y" in merged_df.columns:
                merged_df = merged_df.drop("location_y", axis=1)
            merged_df = merged_df.rename(columns={"location_x": "favlocation", "name": "Product_name"})

        if "quantity" in merged_df.columns:
            scaler = StandardScaler()
            merged_df["quantity"] = scaler.fit_transform(merged_df[["quantity"]])

        missing_columns = REQUIRED_OUTPUT_COLUMNS.difference(merged_df.columns)
        if missing_columns:
            raise ValueError(f"Processed data is missing required columns: {sorted(missing_columns)}")

        merged_df["is_purchased"] = pd.to_numeric(merged_df["is_purchased"], errors="coerce").fillna(0).astype(int)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(OUTPUT_PATH, index=False)
        logger.info("Processed data saved to %s", OUTPUT_PATH)
    except Exception as exc:
        logger.error("An error occurred while preprocessing and saving data: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    preprocess_and_save_data()
