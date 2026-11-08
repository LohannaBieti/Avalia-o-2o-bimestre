import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# ==========================================
# 1. MÓDULO DE TREINAMENTO E PRÉ-PROCESSAMENTO
# ==========================================

# Carregamento da base
df = pd.read_csv("HousingData (1).csv")

# Separando variáveis categóricas (CHAS) e quantitativas
categorical_cols = ['CHAS']
numerical_cols = [col for col in df.columns if col not in categorical_cols]

# Pipelines de pré-processamento (imputação de nulos + padronização)
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ]
)

# Processando a base completa
X_processed = preprocessor.fit_transform(df)

# ==========================================
# 2. MÉTODO DO COTOVELO (DETERMINAÇÃO DE K)
# ==========================================

print("--- Método do Cotovelo ---")
inertias = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_processed)
    inertias.append(kmeans.inertia_)
    print(f"K = {k:2d} | Inércia: {kmeans.inertia_:.2f}")

# Definindo K ótimo = 4 com base na curva do cotovelo
optimal_k = 4

# ==========================================
# 3. TREINAMENTO E SALVAMENTO DO MODELO
# ==========================================

kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_model.fit(X_processed)

# Salvando os artefatos
joblib.dump(preprocessor, 'housing_preprocessor.pkl')
joblib.dump(kmeans_model, 'housing_kmeans.pkl')
print("\nModelo e pré-processador salvos com sucesso!\n")

# ==========================================
# 4. MÓDULO DE INFERÊNCIA
# ==========================================

# Dados de exemplo de uma nova residência
new_residence = {
    'CRIM': 0.05,
    'ZN': 12.5,
    'INDUS': 2.5,
    'CHAS': 0.0,
    'NOX': 0.45,
    'RM': 6.8,
    'AGE': 45.0,
    'DIS': 5.0,
    'RAD': 2,
    'TAX': 250,
    'PTRATIO': 16.0,
    'B': 390.0,
    'LSTAT': 5.5,
    'MEDV': 30.0
}

df_new = pd.DataFrame([new_residence])

# Carregando os objetos salvos
loaded_preprocessor = joblib.load('housing_preprocessor.pkl')
loaded_kmeans = joblib.load('housing_kmeans.pkl')

# Normalizando os dados da nova residência
X_new_processed = loaded_preprocessor.transform(df_new)

# Inferindo o grupo (cluster)
cluster_id = loaded_kmeans.predict(X_new_processed)[0]
print(f"-> A nova residência pertence ao Grupo (Cluster): {cluster_id}")

# Recuperando os dados desnormalizados do centroide do cluster inferido
centroid_scaled = loaded_kmeans.cluster_centers_[cluster_id]

num_scaler = loaded_preprocessor.named_transformers_['num'].named_steps['scaler']
cat_scaler = loaded_preprocessor.named_transformers_['cat'].named_steps['scaler']

num_len = len(numerical_cols)
centroid_num_orig = num_scaler.inverse_transform(centroid_scaled[:num_len].reshape(1, -1))[0]
centroid_cat_orig = cat_scaler.inverse_transform(centroid_scaled[num_len:].reshape(1, -1))[0]

all_cols = numerical_cols + categorical_cols
centroid_orig_values = np.concatenate([centroid_num_orig, centroid_cat_orig])
centroid_df = pd.Series(centroid_orig_values, index=all_cols)

print("\n-> Dados desnormalizados do centroide do grupo inferido:")
print(centroid_df.to_string())