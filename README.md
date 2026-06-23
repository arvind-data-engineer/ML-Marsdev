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
   Optional product content columns improve cold-start recommendations:
   `category`, `category_name`, `category_id`, `brand`, `price`, `unit`,
   `tags`, `description`, `availability`, `seller`, `seller_id`,
   `seller_name`, `seller_location`, `location`, `city`, `state`, `area`,
   `background_color`, and `image`.

   Optional behavior columns improve model quality:
   - `interaction_score`: direct 0 to 1 user-product score
   - `is_favorite` or `favorite`
   - `added_to_cart`, `is_cart`, or `cart_count`
   - `viewed` or `view_count`

   Optional time columns make recent behavior rank higher:
   `interaction_timestamp`, `event_time`, `created_at`, `updated_at`,
   `purchased_at`, `order_date`, `viewed_at`, `favorited_at`, or `carted_at`.

   Example:
   ```env
   DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/products
   PROCESSED_DATA_QUERY=SELECT user_id, product_id, purchased AS is_purchased, created_at AS interaction_timestamp, name AS "Product_name", latitude, longitude, category, brand, price, tags, description, availability, seller_name, seller_location, image FROM recommendations_view
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

   Confirm the active model and data:
   ```bash
   curl "http://127.0.0.1:5000/health"
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
- product content similarity from category, brand, price tier, tags,
  description, availability, seller, and location fields
- weighted behavior signals such as purchase, favorite, cart, and view events
- time-decayed ranking so recent behavior matters more than old behavior
- smoothed product popularity for new users
- location proximity
- diversity reranking to avoid repetitive results

New users automatically fall back to popularity and nearby products until they have enough interaction history.

If recommendations look unchanged after a model update:
- run `python Models/EDA.py` if your database data changed
- run `python Models/Final_Models.py` to rebuild `Models/Model.pkl`
- call `/health` and confirm `model_version` is `v6_time_decay`
- include behavior columns such as `interaction_score`, `is_favorite`,
  `added_to_cart`, or `view_count`; otherwise the model only has purchase data
  and may rank similarly

## Model Versions

- `v6_time_decay`: current model. Applies time decay to user interactions so
  recent purchases, views, favorites, or cart activity influence ranking more
  than older behavior.
- `v5_content_features`: adds product content profiles for
  cold-start products using category, brand, price, tags, description,
  availability, seller, and location fields.
- `v4_weighted_hybrid`: uses weighted behavior scores, SVD,
  item similarity, content/category affinity, popularity, location, and
  diversity reranking.
- `v3_hybrid`: hybrid model with SVD, item similarity, category affinity,
  popularity, location, and diversity reranking.
- `v2_popularity_fallback`: SVD model with smoothed popularity fallback for
  cold-start users.
