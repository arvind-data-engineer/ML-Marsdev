# Product Recommendation Model

This project trains a scikit-learn matrix-factorization product recommendation model and exposes a Flask API for ranked product recommendations.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/carvind35/ML.git
   cd ML
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create your database config:
   ```bash
   copy .env.example .env
   ```

   Then edit `.env` with your real database values.

   The easiest way to connect your own database is to set `DATABASE_URL` and
   `PROCESSED_DATA_QUERY`. The query should return:
   `user_id, product_id, is_purchased, Product_name, latitude, longitude`.
   Optional product columns are `background_color`, `image`, `category`,
   `category_name`, `price`, `unit`, and `description`.

   Optional behavior columns improve model quality:
   - `interaction_score`: direct 0 to 1 user-product score
   - `is_favorite` or `favorite`
   - `added_to_cart`, `is_cart`, or `cart_count`
   - `viewed` or `view_count`

   Example:
   ```env
   DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/products
   PROCESSED_DATA_QUERY=SELECT user_id, product_id, purchased AS is_purchased, name AS "Product_name", latitude, longitude, image FROM recommendations_view
   ```

   You can also use MySQL, SQLite, SQL Server, or another SQLAlchemy-supported
   database by changing `DATABASE_URL` and installing the matching Python driver.
   If your database already has the original app tables, leave
   `PROCESSED_DATA_QUERY` empty and override the table query variables only when
   your table names or columns differ.

4. Exploratory data analysis and preprocessing:
   ```bash
   python Models/EDA.py
   ```

   This creates `Data/Processed.csv`. If you do not have database access, create that file manually using `Data/Processed.example.csv` as the schema reference.

5. Model training:
   ```bash
   python Models/Final_Models.py
   ```

6. Run the application:
   ```bash
   python Models/app.py
   ```

7. Request recommendations:
   ```bash
   curl "http://127.0.0.1:5000/recommend?user_id=123&latitude=12.97&longitude=77.59&limit=5"
   ```

   Optional query parameters:
   - `limit`: number of recommendations, from 1 to 50
   - `distance_weight`: location importance, from 0 to 1. Default is `0.2`
   - `diversity_weight`: reduces repeated categories in the top results. Default is `0.05`
   - `exclude_seen`: hide previously purchased products. Default is `true`

   Example with stronger nearby-product ranking:
   ```bash
   curl "http://127.0.0.1:5000/recommend?user_id=123&latitude=12.97&longitude=77.59&limit=10&distance_weight=0.4"
   ```

The API ranks products using a hybrid recommendation model:
- collaborative SVD score
- item-to-item similarity from previous purchases
- optional category/content affinity when your data has `category`, `category_name`, or `category_id`
- weighted behavior signals such as purchase, favorite, cart, and view events
- smoothed product popularity for new users
- location proximity
- diversity reranking to avoid repetitive results

New users automatically fall back to popularity and nearby products until they have enough interaction history.

## Model Versions

- `v4_weighted_hybrid`: current model. Uses weighted behavior scores, SVD,
  item similarity, content/category affinity, popularity, location, and
  diversity reranking.
- `v3_hybrid`: hybrid model with SVD, item similarity, category affinity,
  popularity, location, and diversity reranking.
- `v2_popularity_fallback`: SVD model with smoothed popularity fallback for
  cold-start users.
