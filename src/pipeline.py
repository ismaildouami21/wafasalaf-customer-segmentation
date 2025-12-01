"""
Customer Segmentation Pipeline (Synthetic Dataset Version)
----------------------------------------------------------

This script implements the full customer segmentation pipeline used
in the original Wafasalaf project, rewritten for public GitHub usage.

IMPORTANT:
- Real confidential data has been replaced with a synthetic dataset.
- The full pipeline is preserved to demonstrate technical skills.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from k_means_constrained import KMeansConstrained


# ============================================================
# 2. LOAD SYNTHETIC DATA (REPLACES CONFIDENTIAL REAL DATA)
# ============================================================

# The synthetic dataset is located in: ../data/synthetic_dataset.csv
df = pd.read_csv("../data/synthetic_dataset.csv")

print("Dataset loaded (synthetic):", df.shape)


# ============================================================
# 3. OUTLIER DETECTION & IMPUTATION
# ============================================================

def detect_and_impute_outliers(df, cols):
    df = df.copy()
    for col in cols:
        base = np.log(df[col]) if (df[col] > 0).all() else df[col]
        Q1, Q3 = base.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        mask = (base < lower) | (base > upper)
        df.loc[mask, col] = df.loc[~mask, col].median()
    return df


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):
    df = df.copy()

    # Fill missing values
    impute_cols = [
        'nbr_aff_payante', 'nbr_aff_gratuite', 'REVENU_MENSUEL',
        'NB_AFFAIRE_PP', 'NB_RPA', 'NB_AFFAIRE_AUTO', 'ANCIENNETE'
    ]

    for col in impute_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df['NB_CPL'] = df.get('NB_CPL', 0).fillna(0)
    df['NB_aff_digital'] = df.get('NB_aff_digital', 0).fillna(0)

    # Derived ratios (avoid division by zero)
    df['PART_AFFAIRES_GRATUITES'] = df['nbr_aff_gratuite'] / df['NB_AFFAIRE'].replace(0, pd.NA)
    df['nb_aff_sur_ancie'] = df['NB_AFFAIRE'] / df['ANCIENNETE'].replace(0, pd.NA)
    df['PART_AFFAIRES_RPA'] = df['NB_RPA'] / df['NB_AFFAIRE'].replace(0, pd.NA)
    df['PART_AFFAIRES_PP'] = df['NB_AFFAIRE_PP'] / df['NB_AFFAIRE'].replace(0, pd.NA)
    df['PART_AFFAIRES_Automobile'] = df['NB_AFFAIRE_AUTO'] / df['NB_AFFAIRE'].replace(0, pd.NA)
    df['PART_AFFAIRES_EDM'] = df['NB_AFFAIRE_EDM'] / df['NB_AFFAIRE'].replace(0, pd.NA)
    df['PART_AFFAIRES_cpl'] = df['NB_CPL'] / df['NB_AFFAIRE'].replace(0, pd.NA)

    # Final feature list
    features = [
        'ANCIENNETE', 'REVENU_MENSUEL', 'NB_AFFAIRE',
        'PART_AFFAIRES_GRATUITES', 'PART_AFFAIRES_PP',
        'PART_AFFAIRES_Automobile', 'nb_aff_sur_ancie',
        'PART_AFFAIRES_RPA', 'PART_AFFAIRES_EDM',
        'PART_AFFAIRES_cpl', 'NB_aff_digital'
    ]

    available = [f for f in features if f in df.columns]
    return df, df[available], available


# ============================================================
# 5. CLUSTERING
# ============================================================

def perform_clustering(df_features, df_original,
                       n_clusters=6,
                       size_min=200,
                       size_max=2000,
                       coef_revenu=2):

    df_clean = df_features.replace([np.inf, -np.inf], np.nan).dropna()

    # Standardize
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df_clean),
        columns=df_clean.columns,
        index=df_clean.index
    )

    # Weight income more heavily
    if "REVENU_MENSUEL" in X_scaled.columns:
        idx = X_scaled.columns.get_loc("REVENU_MENSUEL")
        X_scaled.iloc[:, idx] *= coef_revenu

    # Constrained KMeans
    kmc = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=42
    )

    labels = kmc.fit_predict(X_scaled)

    df_result = df_original.loc[df_clean.index].copy()
    df_result['cluster'] = labels

    return df_result, X_scaled, labels, scaler


# ============================================================
# 6. CLUSTER SUMMARY
# ============================================================

def describe_clusters(df_clustered, features):
    df_clean = df_clustered.dropna(subset=features + ['cluster'])
    summary = df_clean.groupby('cluster')[features].mean().round(2)
    summary['Count'] = df_clean['cluster'].value_counts().sort_index()
    summary['Percentage'] = (df_clean['cluster'].value_counts(normalize=True) * 100).round(2)
    return summary


# ============================================================
# 7. DECISION TREE EXPLAINABILITY
# ============================================================

def plot_decision_tree(df_clustered, features):
    df_clean = df_clustered.dropna(subset=features + ['cluster'])
    X = df_clean[features]
    y = df_clean['cluster']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    print("Decision Tree Test Accuracy:", accuracy_score(y_test, clf.predict(X_test)))

    plt.figure(figsize=(20, 8))
    plot_tree(clf, feature_names=features, filled=True, rounded=True, fontsize=8)
    plt.title("Decision Tree Explaining Cluster Assignment")
    plt.show()

    return clf


# ============================================================
# 8. PCA VISUALIZATION
# ============================================================

def plot_clusters_pca(df_clustered, features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clustered[features])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_clustered["cluster"], cmap="tab10", alpha=0.7)
    plt.title("Cluster Visualization (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()


# ============================================================
# 9. COMPLETE PIPELINE
# ============================================================

def run_pipeline(df):
    print("Running segmentation pipeline...")

    df_clean = detect_and_impute_outliers(df, [
        'nbr_aff_payante', 'nbr_aff_gratuite', 'ANCIENNETE', 'REVENU_MENSUEL'
    ])

    df_clean, df_features, feature_list = feature_engineering(df_clean)

    df_clusters, X_scaled, labels, scaler = perform_clustering(df_features, df_clean)

    summary = describe_clusters(df_clusters, feature_list)
    print(summary)

    tree_model = plot_decision_tree(df_clusters, feature_list)
    plot_clusters_pca(df_clusters, feature_list)

    return df_clusters, summary, tree_model


# ============================================================
# 10. EXECUTE
# ============================================================

if __name__ == "__main__":
    df_clusters, summary, tree_model = run_pipeline(df)
    print("\nPipeline completed successfully.")
