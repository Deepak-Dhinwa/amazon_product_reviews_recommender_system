import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

# Loading the trained SVD model
with open(BASE_DIR / "svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

st.title("Product Recommendation System")
st.write("""
This app recommends products to users based on the trained Surprise SVD collaborative filtering model
""")

# Loading dataset to map user/item IDs if needed
@st.cache_data
def load_data():
    df = pd.read_csv(
        BASE_DIR / "office_products_filtered.csv",
        usecols=["reviewerID", "asin"],
    )
    return df

df = load_data()

# Input user ID
user_id = st.text_input("Enter your user ID:")

if st.button("Get Recommendations"):
    if user_id not in df['reviewerID'].unique():
        st.warning("User ID not found! Try another")
    else:
        # Get all items
        all_items = df['asin'].unique()
        user_items = df[df['reviewerID'] == user_id]['asin'].unique()
        items_to_predict = [item for item in all_items if item not in user_items]

        # Predict ratings for items not yet rated
        predictions = []
        for item_id in items_to_predict:
            pred = svd.predict(user_id, item_id)
            predictions.append((item_id, pred.est))

        # Sort by estimated rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Show top-N
        N = 10
        top_n = predictions[:N]
        st.subheader(f"Top {N} Recommendations for User {user_id}")
        for i, (item_id, rating) in enumerate(top_n, start=1):
            st.write(f"{i}. Item ID: {item_id}, Predicted Rating: {rating:.2f}")
