import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Loading files
User_df = pd.read_csv('Data/User.csv')
Purchase_df = pd.read_csv('Data/Product.csv')
Fav_location_df = pd.read_csv('Data/Favorite_Location.csv')
Fav_product_df = pd.read_csv('Data/Favorite_Product.csv')
Category_df = pd.read_csv('Data/Category.csv')
Item_df = pd.read_csv('Data/Items.csv')

# Preprocessing User_df
User_df = User_df.dropna(axis=1, how='all')
User_df = User_df.rename(columns={'id': 'user_id'})
cols = ['last_login', 'date_joined']
User_df = User_df.drop(cols, axis=1)

# Preprocessing Purchase_df
Purchase_df = Purchase_df.dropna(axis=1, how='all')
Purchase_df['unit'] = pd.Categorical(Purchase_df['unit'])
Purchase_df['unit'] = Purchase_df['unit'].cat.codes
cols = ['id','created_at', 'updated_at']
Purchase_df = Purchase_df.drop(cols, axis=1)
Purchase_df.rename(columns={'product': 'product_id', 'added_by': 'user_id'}, inplace=True)

# Preprocessing Fav_location_df
cols = ['id', 'created_at', 'updated_at', 'label', 'note']
Fav_location_df = Fav_location_df.drop(cols, axis=1)
Fav_location_df.rename(columns={'user': 'user_id'}, inplace=True)

# Preprocessing Fav_product_df
cols = ['id', 'created_at', 'updated_at']
Fav_product_df = Fav_product_df.drop(cols, axis=1)
Fav_product_df.rename(columns={'user': 'user_id'}, inplace=True)

# Preprocessing Category_df
cols = ['id', 'created_at', 'updated_at', 'background_color']
Category_df = Category_df.drop(cols, axis=1)
Category_df.rename(columns={'id': 'user_id'}, inplace=True)

# Preprocessing Item_df
cols = ['created_at', 'updated_at', 'background_color']
Item_df = Item_df.drop(cols, axis=1)
Item_df.rename(columns={'id': 'user_id'}, inplace=True)

# Merging DataFrames
merged_df = pd.merge(User_df, Purchase_df, on='user_id', how='inner')
merged_df = pd.merge(merged_df, Fav_product_df, on='user_id', how='inner')
merged_df = pd.merge(merged_df, Item_df, on='user_id', how='inner')
merged_df = pd.merge(merged_df, Fav_location_df, on='user_id', how='inner')

# Dropping duplicate location column and renaming columns
merged_df.drop('location_y', axis=1, inplace=True)
merged_df.rename(columns={'location_x': 'favlocation', 'name': 'Product_name'}, inplace=True)

# Normalize numerical variables
scaler = StandardScaler()
merged_df['quantity'] = scaler.fit_transform(merged_df['quantity'].values.reshape(-1, 1))

# Encode target variable
merged_df['is_purchased'] = merged_df['is_purchased'].astype(int)

# Define the path where you want to save the processed CSV file
output_path = 'Data/Processed.csv'

# Save processed data to CSV
merged_df.to_csv(output_path, index=False)
