# Office Products Recommender System

One-page summary: open `poster.html` in a browser.
Start: `.\run.ps1`

## **Project Overview**

In this project, I built a **comprehensive recommender system pipeline** using Amazon Office Products review data. The goal was to **predict and recommend relevant products** to users based on their past interactions.

I explored **baseline models**, **collaborative filtering**, and **deep learning recommenders** to understand their strengths, limitations, and practical applicability.

---

## **Problem Statement**

Amazon has millions of products, and users cannot browse them all. The task is to:

- Recommend products to users based on what they (and similar users) have purchased or reviewed before.
- Evaluate these recommendations using **Precision@K** and **Recall@K** to ensure relevance and coverage.

---

## **Dataset**

- **Source:** Amazon Office Products Review Dataset (5-core subset)  
- **Size after filtering:** ~370,000 reviews  
- **Features used:**  
  - `reviewerID` (User ID)  
  - `asin` (Item ID)  
  - `overall` (Rating)  
  - Other metadata ignored for core recommendation task.

- **Key stats (k-core filtered with k=7):**
  - Users: 33,828
  - Items: 12,296
  - Average ratings per user: ~11
  - **Sparsity:** ~99.9%

---

## **Approach and Models**

### 1 **Baseline: Popularity Model**

- **Logic:** Recommend globally popular products to everyone.
- **Result:**  
  - High recall for universally popular items  
  - No personalization
  - for cold start users

---

### 2 **Collaborative Filtering (Matrix Factorization using Surprise SVD)**

- **Implemented using:** `surprise` library’s SVD 
- **RMSE:** ~0.84  
- **Precision@10:** ~0.88  
- **Recall@10:** ~0.94

**Interpretation:** Good coverage for seen users, performs well on offline metrics.

---

### 3 **Truncated SVD (Scikit-learn)**

- Used purely as dimensionality reduction on user-item matrix.
- **RMSE:** ~4.41 (high)
- Not an actual recommender model here, but used to understand decomposition behaviour

---

### 4️**Hybrid Model (Popularity + CF average)**

- Combined baseline popularity and SVD CF scores.
- **Precision@10:** ~0.10  
- **Recall@10:** ~0.09

**Interpretation:** Hybridisation benefits depend heavily on score scaling and dataset characteristics

---

### 5️**Neural Collaborative Filtering (NeuMF)**

- Implemented in PyTorch from scratch.
- Used user and item embeddings + MLP layers.
- **Training Loss:** Reduced to ~0.42 after 3 epochs
- **Validation Precision@10:** ~0.004  
- **Validation Recall@10:** ~0.024

**Observation:**  

- Needs tuning (more epochs, hyperparameters, proper negative sampling) to converge well 
- Current performance is low compared to traditional CF in this dataset setting

---

### 6️ **Two-Tower Model**

- Similar to NeuMF but using separate user and item towers with dot-product scoring.
- **Precision@10:** ~0.0002  
- **Recall@10:** ~0.0011

**Interpretation:** Underperformed in current configuration. Would need larger datasets, tuned embedding sizes, and longer training to perform competitively

---

## **Key Learnings**

- Importance of **data filtering (k-core)** for dense training
- Collaborative filtering (SVD) performs surprisingly well with sparse data
- **Deep learning recommenders require extensive tuning** and large-scale data for meaningful improvements
- Popularity models are strong baselines despite simplicity
- Precision and recall evaluation gives practical insight into user satisfaction vs. coverage

---

## **Next Steps**

- Hyperparameter tuning for NeuMF and Two-Tower models

---

## **Tech Stack**

- Python, Pandas, NumPy
- Scikit-learn
- Surprise
- PyTorch
- Matplotlib & Seaborn (for EDA)
- Streamlit 

---

## **Final Notes**

This project helped me deeply understand **how industrial-scale recommender systems are built and evaluated**. From baselines to hybrid to neural models, each step added practical ML skills and insights into **real-world personalization systems**

---

## Run

Use the existing environment in `amazon_rec_sys_env/`. The saved SVD pickle was loaded with the Surprise version in that environment.

```bash
amazon_rec_sys_env\Scripts\streamlit.exe run app.py
```

`app.py` reads `office_products_filtered.csv` and `svd_model.pkl` from this folder. A user id in the filtered file is `AN2FZ6SBNYDDQ`. The app scores items that user has not already reviewed and shows the top 10. For that user the saved model estimates 5.00 on the top items.