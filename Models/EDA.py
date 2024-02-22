# Import necessary libraries
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

try:
    # Try to connect to the database
    connection = engine.connect()
    
    # If the connection succeeds, print a success message
    print("Connection successful!")
    
    # Close the connection
    connection.close()
except Exception as e:
    # If an exception occurs, print the error message
    print("Connection failed:", e)


try:
    # Create a folder named "Data" if it doesn't exist
    data_folder = "Data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Define SQL queries for each table
    queries = {
        "auth_user": "SELECT * FROM auth_user",
        "list_product": "SELECT * FROM list_product",
        "product_items": "SELECT * FROM product_items",
        "product_category": "SELECT * FROM product_category",
        "users_favoriteproduct": "SELECT * FROM users_favoriteproduct",
        "users_favoritelocation": "SELECT * FROM users_favoritelocation"
    }

    # Execute queries and store results in DataFrames
    dataframes = {}
    with engine.connect() as connection:
        for table, query in queries.items():
            result = connection.execute(text(query))
            dataframes[table] = pd.DataFrame(result.fetchall(), columns=result.keys())

    # Store each DataFrame in a separate CSV file in the "Data" folder
    for table, df in dataframes.items():
        df.to_csv(os.path.join(data_folder, f"{table}_data.csv"), index=False)

except Exception as e:
    # If an exception occurs, print the error message
    print("Query failed:", e)


# Loading files
User_df = pd.read_csv("Data/auth_user_data.csv")
Purchase_df = pd.read_csv("Data/list_product_data.csv")
Fav_location_df = pd.read_csv("Data/users_favoritelocation_data.csv")
Fav_product_df = pd.read_csv("Data/users_favoriteproduct_data.csv")
Item_df = pd.read_csv("Data/product_items_data.csv")


# Preprocessing User_df
User_df = User_df.dropna(axis=1, how="all")
User_df = User_df.rename(columns={"id": "user_id"})
cols = ["password", "is_superuser", "last_login", "date_joined"]
User_df = User_df.drop(cols, axis=1)

# Preprocessing Purchase_df
Purchase_df = Purchase_df.dropna(axis=1, how="all")
Purchase_df["unit"] = pd.Categorical(Purchase_df["unit"])
Purchase_df["unit"] = Purchase_df["unit"].cat.codes
cols = ["id", "created_at", "updated_at"]
Purchase_df = Purchase_df.drop(cols, axis=1)
Purchase_df.rename(
    columns={"added_by_id": "user_id"},
    inplace=True,
)

# Preprocessing Fav_location_df
cols = ["id", "created_at", "updated_at", "label", "note"]
Fav_location_df = Fav_location_df.drop(cols, axis=1)

# Preprocessing Fav_product_df
cols = ["id", "created_at", "updated_at"]
Fav_product_df = Fav_product_df.drop(cols, axis=1)

# Preprocessing Item_df
cols = ["created_at", "updated_at", "category_id"]
Item_df = Item_df.drop(cols, axis=1)
Item_df = Item_df.rename(columns={"id": "product_id"})

merged_df = pd.merge(User_df, Purchase_df, on="user_id", how="inner")
merged_df = pd.merge(merged_df, Item_df, on="product_id", how="inner")
merged_df = pd.merge(merged_df, Fav_location_df, on="user_id", how="inner")
merged_df = pd.merge(merged_df, Fav_product_df, on="user_id", how="inner")

# Dropping duplicate location column and renaming columns
merged_df.drop("location_y", axis=1, inplace=True)
merged_df.rename(
    columns={"location_x": "favlocation", "name": "Product_name"},
    inplace=True,
)

# Normalize numerical variables
scaler = StandardScaler()
merged_df["quantity"] = scaler.fit_transform(
    merged_df["quantity"].values.reshape(-1, 1)
)

# Encode target variable
merged_df["is_purchased"] = merged_df["is_purchased"].astype(int)

# Define the path where you want to save the processed CSV file
output_path = "Data/Processed.csv"

# Save processed data to CSV
merged_df.to_csv(output_path, index=False)
