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
   Optional columns are `background_color` and `image`.

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

The API ranks products using the trained collaborative filtering model and a small location-proximity boost.
