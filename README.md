# Customer Segmentation using Machine Learning

This repository contains a complete end-to-end Machine Learning pipeline for grouping mall customers into behavioral segments. By analyzing unlabeled customer data, the models identify distinct target demographics to optimize marketing strategies.

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)

## 🧠 Algorithms Implemented

The project tests and compares three fundamental unsupervised learning algorithms:

1. **K-Means Clustering:** Centroid-based partitioning (optimized using the Elbow Method).
2. **Hierarchical Clustering (Agglomerative):** Bottom-up clustering visualized via Dendrograms.
3. **DBSCAN:** Density-based spatial clustering capable of isolating noise/outliers.

## 📊 Results & Evaluation

The models were evaluated using the **Silhouette Score** to measure cluster separation and cohesion.

- K-Means successfully separated the data into **5 distinct customer groups** (e.g., High Income/High Spending, Low Income/Low Spending).
- Both K-Means and Hierarchical Clustering yielded a Silhouette Score of **~0.55**.
- The best-performing model (`K-Means`) and the `StandardScaler` have been exported as `.pkl` files for immediate use.

## 📂 Repository Structure

```text
.
├── Customer.csv                       # Dataset
├── Customer Segmentation Models.ipynb # Main Jupyter Notebook with code & plots
├── kmeans_segmentation_model.pkl      # Exported model for production
├── scaler.pkl                         # Exported standard scaler
└── README.md                          # Project documentation
```
