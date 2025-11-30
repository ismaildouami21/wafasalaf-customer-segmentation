Data Importation

import pandas as pd

# Importation des fichiers (assure-toi de les avoir uploadés via la barre latérale de Colab)
partie1 = pd.read_excel("Partie_1.xlsx")
partie2 = pd.read_excel("Partie_2.xlsx")
partie3 = pd.read_excel("Partie_3.xlsx")
partie4 = pd.read_excel("Partie_4.xlsx")

# Concaténation
df = pd.concat([partie1, partie2, partie3, partie4], ignore_index=True)

# Affichage des noms de colonnes
print("Noms des colonnes :", df.columns.tolist())


Encoding

colonnes_a_binariser = ['TYPE_CLIENT', 'CSP_MKT', 'PHASE_RELATION']
df = pd.get_dummies(df, columns=colonnes_a_binariser, prefix=colonnes_a_binariser)
print("Noms des colonnes :", df.columns.tolist())

Data Cleaning

!pip install k-means-constrained


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from k_means_constrained import KMeansConstrained  # Or switch to MiniBatchKMeans if needed

# ============ 0. Détection et imputation valeurs aberrantes ============
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

# ============ 1. Feature Engineering ============
def feature_engineering(df):
    df = df.copy()

    # Imputation avec median
    impute_cols = ['nbr_aff_payante', 'nbr_aff_gratuite', 'REVENU_MENSUEL',
                   'NB_AFFAIRE_PP', 'NB_RPA', 'NB_AFFAIRE_AUTO','ANCIENNETE']
    for col in impute_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing for new columns with 0
    df['NB_CPL'] = df['NB_CPL'].fillna(0)
    df['NB_aff_digital'] = df['NB_aff_digital'].fillna(0)


    # New and existing engineered features
    df['PART_AFFAIRES_GRATUITES'] = df['nbr_aff_gratuite'] / df['NB_AFFAIRE'].replace(0, pd.NA)
    df['nb_aff_sur_ancie'] = df['NB_AFFAIRE'] / df['ANCIENNETE'].replace(0, pd.NA)
    df['PART_AFFAIRES_RPA'] = df['NB_RPA'] / df['NB_AFFAIRE'].replace(0, pd.NA)
    df['PART_AFFAIRES_PP'] = df['NB_AFFAIRE_PP'] / df['NB_AFFAIRE'].replace(0, pd.NA)
    df['PART_AFFAIRES_Automobile'] = df['NB_AFFAIRE_AUTO'] / df['NB_AFFAIRE'].replace(0, pd.NA)
    df['PART_AFFAIRES_EDM'] = df['NB_AFFAIRE_EDM'] / df['NB_AFFAIRE'].replace(0, pd.NA)
    df['PART_AFFAIRES_cpl'] = df['NB_CPL'] / df['NB_AFFAIRE'].replace(0, pd.NA)

    # Feature list (extended with new features)
    features = [
        'nbr_affaires_place', 'ANCIENNETE', 'REVENU_MENSUEL',
        'PART_AFFAIRES_GRATUITES', 'PART_AFFAIRES_PP', 'PART_AFFAIRES_Automobile',
        'nb_aff_sur_ancie', 'NB_AFFAIRE', 'PART_AFFAIRES_RPA', 'PART_AFFAIRES_EDM',
        'PART_AFFAIRES_cpl', 'NB_aff_digital'
    ]

    return df, df[features], features

# ============ 2. Clustering Optimisé ============
def perform_clustering(df_features, df_original, n_clusters=6, size_min=20000, size_max=70000, coef_revenu=2):
    df_clean = df_features.replace([np.inf, -np.inf], np.nan).dropna()

    # Standardisation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns, index=df_clean.index)

    # Pondération de REVENU_MENSUEL
    index_revenu = list(X_scaled.columns).index('REVENU_MENSUEL')
    X_scaled.iloc[:, index_revenu] *= coef_revenu

    # Clustering
    kmc = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=42
    )
    labels = kmc.fit_predict(X_scaled)

    df_result = df_original.loc[df_clean.index].copy()
    df_result['cluster'] = labels
    return df_result, X_scaled, labels

# ============ 3. Analyse Cluster ============
def describe_clusters(df_clustered, features):
    df_clean = df_clustered.dropna(subset=features + ['cluster'])
    summary = df_clean.groupby('cluster')[features].mean().round(2)
    summary['Effectif'] = df_clean['cluster'].value_counts().sort_index()
    summary['Proportion'] = (df_clean['cluster'].value_counts(normalize=True) * 100).round(2).sort_index()
    summary = summary[['Effectif', 'Proportion'] + features]
    return summary.sort_index()

def plot_global_tree(df_clustered, features):
    df_clean = df_clustered.dropna(subset=features + ['cluster'])
    X = df_clean[features].astype(float)
    y = df_clean['cluster'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"✅ Précision arbre (test set) : {acc:.2%}")
    plt.figure(figsize=(20, 8))
    plot_tree(clf, feature_names=features, class_names=[str(c) for c in sorted(y.unique())],
              filled=True, rounded=True, fontsize=9)
    plt.title("Arbre de décision expliquant les clusters")
    plt.tight_layout()
    plt.show()
    return clf

def print_cluster_descriptions(summary_df):
    mean_global = summary_df.drop(['Effectif', 'Proportion'], axis=1).mean()
    for cluster_id, row in summary_df.iterrows():
        print(f"\n--- Cluster {cluster_id} ---")
        print(f"Effectif : {int(row['Effectif'])} ({row['Proportion']}%)")
        delta = row.drop(['Effectif', 'Proportion']) - mean_global
        main_features = delta.abs().sort_values(ascending=False).head(20)
        print("Caractéristiques distinctives :")
        for feat in main_features.index:
            val = row[feat]
            delta_val = delta[feat]
            sign = "↑" if delta_val > 0 else "↓"
            print(f"  - {feat}: {val:.2f} ({sign} {abs(delta_val):.2f} vs moy. globale)")

# ============ 4. Pipeline complet ============
def full_pipeline(df):
    # 🔍 Filtrage : clients actifs uniquement (duree_inactivite_activite > 0)
    df_active = df[df['duree_inactivite_activite'] > 0].copy()

    df_clean = detect_and_impute_outliers(df_active, [
        'nbr_affaires_place','nbr_aff_payante', 'nbr_aff_gratuite',
        'ANCIENNETE', 'REVENU_MENSUEL'
    ])
    df_clean, df_features, features_used = feature_engineering(df_clean)
    df_clusters, X_scaled, labels = perform_clustering(df_features, df_clean)

    summary = describe_clusters(df_clusters, features_used)
    tree_model = plot_global_tree(df_clusters, features_used)
    print_cluster_descriptions(summary)
    return df_clusters, summary, tree_model


# ============ 5. Exemple d'utilisation ============
# df = pd.read_excel("mon_fichier.xlsx")
# df_clusters, summary, tree_model = full_pipeline(df)


df_clusters, summary, tree_model = full_pipeline(df)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def plot_clusters_pca_from_df(df_clusters, feature_columns, cluster_column='cluster'):
    """
    Visualise les clusters en 2D via PCA à partir de df_clusters.
    """
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clusters[feature_columns])

    # PCA reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plotting
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_clusters[cluster_column], cmap='tab10', alpha=0.7)
    plt.title('Visualisation des clusters clients (PCA)')
    plt.xlabel('Composante principale 1')
    plt.ylabel('Composante principale 2')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


features = [
    'nbr_affaires_place', 'ANCIENNETE', 'REVENU_MENSUEL',
    'PART_AFFAIRES_GRATUITES', 'PART_AFFAIRES_PP', 'PART_AFFAIRES_Automobile',
    'nb_aff_sur_ancie', 'NB_AFFAIRE', 'PART_AFFAIRES_RPA', 'PART_AFFAIRES_EDM',
    'NB_CPL', 'NB_aff_digital'
]

plot_clusters_pca_from_df(df_clusters, features)


from sklearn.tree import _tree
import pandas as pd


def extract_decision_rules(tree_model, feature_names, df, target_name='cluster'):
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_path = path + [f"{name} ≤ {round(threshold, 2)}"]
            right_path = path + [f"{name} > {round(threshold, 2)}"]
            recurse(tree_.children_left[node], left_path)
            recurse(tree_.children_right[node], right_path)
        else:
            rules.append(path)

    recurse(0, [])

    all_infos = []

    for path in rules:
        mask = pd.Series(True, index=df.index)
        for cond in path:
            name, op, threshold = cond.split(' ', 2)
            threshold = float(threshold)
            if '≤' in op:
                mask &= df[name] <= threshold
            elif '>' in op:
                mask &= df[name] > threshold

        subset = df[mask]
        effectif = len(subset)
        if effectif == 0:
            continue
        proportion = round(100 * effectif / len(df), 2)
        cluster_value = subset[target_name].mode()[0]
        mean_values = subset[feature_names].mean().round(1).to_dict()

        all_infos.append({
            "Règle complète": "\n".join(path),  # retour à la ligne entre chaque condition
            "Effectif": effectif,
            "Proportion (%)": proportion,
            "Cluster prédominant": cluster_value,
            **mean_values
        })

    df_rules = pd.DataFrame(all_infos)
    df_rules = df_rules.sort_values(by="Effectif", ascending=False).reset_index(drop=True)
    return df_rules


features = [
        'nbr_affaires_place', 'ANCIENNETE', 'REVENU_MENSUEL',
        'PART_AFFAIRES_GRATUITES', 'PART_AFFAIRES_PP', 'PART_AFFAIRES_Automobile',
        'nb_aff_sur_ancie', 'NB_AFFAIRE', 'PART_AFFAIRES_RPA', 'PART_AFFAIRES_EDM',
        'NB_CPL', 'NB_aff_digital'
    ]

df_rules = extract_decision_rules(tree_model, features, df_clusters)
pd.set_option('display.max_colwidth', 150)  # Pour afficher bien la colonne règle
display(df_rules)

cluster_id = 1

# Filter for the target cluster
cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]

# Compute mean of MT_FINANCE_EDM and NB_AFFAIRE_EDM
mean_mt_finance = cluster_data['MT_FINANCE_EDM'].mean()
mean_nb_edm_affaire = cluster_data['NB_AFFAIRE_EDM'].mean()

# Avoid division by zero
if mean_nb_edm_affaire > 0:
    ratio = mean_mt_finance / mean_nb_edm_affaire
    print(f"Mean MT_FINANCE_EDM per EDM affair in cluster {cluster_id}: {ratio:.2f}")
else:
    print(f"Cluster {cluster_id} has 0 mean EDM affairs. Division not possible.")


# Filter cluster 1
cluster_1 = df_clusters[df_clusters['cluster'] == 1]

# Randomly sample 10 TIER numbers
sample_tiers = cluster_1['TIERS_CLIENT'].dropna().sample(n=10, random_state=42)

# Display the result
print("🔢 10 Random TIER numbers from Cluster 1:")
print(sample_tiers.tolist())


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clusters[features])

# Récupérer les moyennes et écarts-types pour SQL
scaler_means = pd.Series(scaler.mean_, index=features, name='mean')
scaler_stds = pd.Series(scaler.scale_, index=features, name='std')

# Affichage pour vérification
df_scaler_params = pd.concat([scaler_means, scaler_stds], axis=1)
display(df_scaler_params)

# Modèle de régression logistique multinomiale
logreg = LogisticRegression(solver='lbfgs', max_iter=5000)
logreg.fit(X_scaled, df_clusters['cluster'])

# Coefficients et intercepts
coefs = pd.DataFrame(logreg.coef_.T, index=features)
coefs.columns = [f"Cluster {i}" for i in range(logreg.coef_.shape[0])]
intercepts = pd.Series(logreg.intercept_, index=coefs.columns)

display(coefs)
print("\nIntercepts :\n", intercepts)

# Génération de la formule SQL pour chaque cluster
for cluster in coefs.columns:
    sql_expr = [f"{intercepts[cluster]:.8f}"]
    for feature in features:
        coef = coefs.loc[feature, cluster]
        mean = scaler_means[feature]
        std = scaler_stds[feature]
        sql_expr.append(f"({coef:.8f} * (([{feature}] - {mean:.8f}) / {std:.8f}))")

    sql_formula = " + ".join(sql_expr)
    print(f"-- Logit for {cluster}:\n{sql_formula}\n")

# ⚠️ Évaluation : Matrice de confusion
y_true = df_clusters['cluster']
y_pred = logreg.predict(X_scaled)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
disp.plot(cmap='Blues')
plt.title("Matrice de confusion - Régression Logistique")
plt.show()


# Génération du SQL depuis coefficients + scaler
def generate_sql_logreg_softmax(features, coefs, intercepts, scaler_means, scaler_stds, table_name='MA_TABLE'):

    sql_lines = []
    exp_logits = []

    for cluster in coefs.columns:
        formula_terms = [f"{intercepts[cluster]:.8f}"]
        for feature in features:
            coef = coefs.loc[feature, cluster]
            mean = scaler_means[feature]
            std = scaler_stds[feature]
            term = f"(({feature} - {mean:.8f}) / {std:.8f}) * {coef:.8f}"
            formula_terms.append(term)
        formula = " + ".join(formula_terms)
        logit = f"LOGIT_{cluster}"
        sql_lines.append(f"{logit} = {formula}")
        exp_logits.append(f"EXP({logit})")

    sql_code = "-- Calcul des logits\n" + "\n".join(sql_lines) + "\n\n"

    # Softmax
    sum_exp = " + ".join(exp_logits)
    probas = [f"EXP(LOGIT_{col}) / ({sum_exp}) AS PROBA_{col}" for col in coefs.columns]
    sql_code += "-- Calcul des probabilités\nSELECT *,\n" + ",\n".join(probas)

    # CASE to pick the max proba
    greatest = f"GREATEST({', '.join([f'PROBA_{col}' for col in coefs.columns])})"
    case_lines = [f"WHEN PROBA_{col} = {greatest} THEN {col}" for col in coefs.columns]
    sql_code += ",\nCASE\n" + "\n".join(case_lines) + "\nEND AS PREDICTED_CLUSTER\n"
    sql_code += f"FROM {table_name};"

    return sql_code


# Appel de la fonction (en supposant que tu as déjà coefs, intercepts, scaler_means, scaler_stds)
sql_output = generate_sql_logreg_softmax(features, coefs, intercepts, scaler_means, scaler_stds)

# Affiche le SQL prêt à l'emploi
print(sql_output)


!pip install k-means-constrained

!pip install numpy==1.24.4 --force-reinstall --no-cache-dir

