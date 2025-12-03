# Customer Segmentation Using PCA and K-Means

This project applies unsupervised machine learning to segment customers based on behavioral, financial, and engagement variables. The goal is to help organizations understand customer groups, target marketing actions, and support data-driven decision-making. The project demonstrates a full analytics workflow from raw data to insights, using PCA for dimensionality reduction and K-Means for clustering.


## Project Overview

Customer segmentation is essential for optimizing marketing strategies, improving customer retention, and identifying high-value or at-risk groups. This project uses a cleaned customer dataset to create meaningful customer segments through:

* Data preprocessing
* Feature engineering
* Dimensionality reduction (PCA)
* K-Means clustering
* Cluster profiling and interpretation
* Visual and business insights

This repository is designed to showcase industry-relevant skills for entry-level Data Science and Business Analytics roles.


## Skills Demonstrated

* Python (Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn)
* Data cleaning, missing value imputation, outlier handling
* Feature engineering and one-hot encoding
* PCA for dimensionality reduction
* K-Means clustering (Elbow Method, cluster analysis)
* Visualization and interpretability
* Business insight generation
* Reproducible, well-structured project organization



## Repository Structure

```
wafasalaf-customer-segmentation/
│
├── data/                    
├── src/                     
├── docs/                    
├── results/                 
│   ├── 01_summary_statistics_table.png
│   ├── 02_kmeans_elbow_method.png
│   ├── 03_pca_cluster_visualization.png
│   ├── 04_cluster_centroids_table.png
│   ├── 05_cluster_infographic_english.png
│   ├── 06_cluster_decision_tree.png
│   ├── 07_confusion_matrix_logistic_regression.png
│
└── README.md
```


## Methodology

### 1. Data Preprocessing

* Merged and cleaned datasets
* Treated missing values and outliers
* Encoded categorical variables
* Created engineered features such as:

  * credit_density
  * anticipation_ratio
  * part_affaire_payante
* Standardized numerical features

### 2. PCA (Principal Component Analysis)

* Reduced dimensionality
* Removed multicollinearity
* Highlighted key components driving customer variation
* Improved cluster separability

### 3. K-Means Clustering

* Elbow Method determined k = 6
* Trained K-Means on PCA-transformed data
* Computed and interpreted cluster centroids
* Validated cluster cohesion and separation

### 4. Cluster Profiling

Each cluster was analyzed using feature means, PCA trends, and business attributes to identify:

* High-value customers
* Low-engagement or inactive users
* Middle-tier stable customers
* Product-diverse but inconsistent groups


## Key Results

All visuals can be found in the `/results` folder.

1. Summary Statistics Table
2. Elbow Method Curve
3. PCA Cluster Visualization
4. Cluster Centroids Table
5. Customer Segment Infographic
6. Cluster Decision Tree
7. Confusion Matrix for Reactivation Prediction

These figures summarize segment differences and provide interpretable business insights.


## Business Insights

The segmentation supports:

* Identifying profitable customer groups
* Targeting at-risk or inactive customers
* Optimizing marketing campaigns
* Cross-selling and upselling strategies
* Enhancing customer relationship management
* Providing a foundation for reactivation and churn models


## How to Run

```bash
git clone https://github.com/ismaildouami21/wafasalaf-customer-segmentation.git
pip install -r requirements.txt
python src/pca_kmeans_pipeline.py
```

Python 3.10+ recommended.


## Author

Ismail Douami
Data Science & Finance

* LinkedIn:(https://www.linkedin.com/in/ismaildouami/)]
* GitHub: [https://github.com/ismaildouami21]

## License

This project is for educational and portfolio use only. No proprietary company data is included.


