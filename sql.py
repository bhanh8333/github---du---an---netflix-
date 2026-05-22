# =========================================================
# K-MEANS CLUSTERING
# READ DATA DIRECTLY FROM SQL SERVER
# =========================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pyodbc

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

import warnings
warnings.filterwarnings('ignore')

print('✅ Libraries loaded.')

# =========================================================
# 2. CONNECT TO SQL SERVER
# =========================================================

conn = pyodbc.connect(
    r'DRIVER={SQL Server};'
    r'SERVER=ADMIN-PC\SQLEXPRESS;'
    r'DATABASE=NetflixDB;'
    r'Trusted_Connection=yes;'
)

print('✅ Connected to SQL Server.')

# =========================================================
# 3. LOAD DATA FROM SQL
# =========================================================

query = """
SELECT *
FROM netflix_users
"""

df = pd.read_sql(query, conn)

print('\n===== DATA OVERVIEW =====')
print(df.shape)
print(df.head())

# =========================================================
# 4. ENCODING CATEGORICAL VARIABLES
# =========================================================

cat_cols = [
    'gender',
    'country',
    'subscription_type',
    'payment_method',
    'primary_device',
    'devices_used',
    'favorite_genre',
    'churned'
]

encoders = {}

for col in cat_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

    encoders[col] = le

print('\n✅ Encoding completed.')

# =========================================================
# 5. SELECT FEATURES
# =========================================================

drop_cols = ['user_id']

X = df.drop(columns=drop_cols)

print('\n===== FEATURES =====')
print(X.columns.tolist())

# =========================================================
# 6. STANDARDIZATION
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print('\n✅ Standard scaling completed.')

# =========================================================
# 7. PCA
# =========================================================

pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)

print('\n===== PCA RESULT =====')
print(f'Original shape : {X_scaled.shape}')
print(f'PCA shape      : {X_pca.shape}')

# =========================================================
# 8. CREATE PCA DATAFRAME
# =========================================================

pc_cols = [
    f'PC{i+1}'
    for i in range(X_pca.shape[1])
]

pca_df = pd.DataFrame(
    X_pca,
    columns=pc_cols
)

# giữ user_id để join lại SQL
pca_df['user_id'] = df['user_id']

# churn để phân tích
pca_df['churned'] = df['churned']

print('\n✅ PCA dataframe created.')

# =========================================================
# 9. FIND BEST K
# =========================================================

K_RANGE = range(2, 11)

silhouette_scores = []
db_scores = []
ch_scores = []

print('\n🔍 Finding optimal K...')

for k in K_RANGE:

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = km.fit_predict(
        pca_df[pc_cols]
    )

    sil = silhouette_score(
        pca_df[pc_cols],
        labels
    )

    db = davies_bouldin_score(
        pca_df[pc_cols],
        labels
    )

    ch = calinski_harabasz_score(
        pca_df[pc_cols],
        labels
    )

    silhouette_scores.append(sil)
    db_scores.append(db)
    ch_scores.append(ch)

    print(
        f'k={k} | '
        f'Silhouette={sil:.4f} | '
        f'DB={db:.4f} | '
        f'CH={ch:.2f}'
    )

# =========================================================
# 10. VISUALIZE K SELECTION
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(K_RANGE),
    silhouette_scores,
    marker='o'
)

plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.title('K Selection')

plt.grid(True)
plt.show()

# =========================================================
# 11. FINAL KMEANS
# =========================================================

BEST_K = 4

kmeans = KMeans(
    n_clusters=BEST_K,
    random_state=42,
    n_init=20
)

pca_df['cluster'] = kmeans.fit_predict(
    pca_df[pc_cols]
)

print('\n===== CLUSTER DISTRIBUTION =====')

cluster_counts = (
    pca_df['cluster']
    .value_counts()
    .sort_index()
)

print(cluster_counts)

# =========================================================
# 12. CLUSTER LABELS
# =========================================================

cluster_map = {
    0: 'Champions',
    1: 'At Risk',
    2: 'Binge Watchers',
    3: 'Casual Users'
}

pca_df['cluster_label'] = (
    pca_df['cluster']
    .map(cluster_map)
)

# =========================================================
# 13. CLUSTER ANALYSIS
# =========================================================

analysis_df = pd.concat(
    [
        df,
        pca_df[['cluster', 'cluster_label']]
    ],
    axis=1
)

cluster_analysis = (
    analysis_df
    .groupby(
        ['cluster', 'cluster_label']
    )
    .agg({
        'engagement_rate': 'mean',
        'total_revenue_to_date': 'mean',
        'avg_watch_time_minutes': 'mean',
        'days_since_last_login': 'mean'
    })
    .round(2)
)

print('\n===== CLUSTER ANALYSIS =====')
print(cluster_analysis)

# =========================================================
# 14. PUSH RESULTS BACK TO SQL SERVER
# =========================================================

cluster_output = pca_df[
    [
        'user_id',
        'cluster',
        'cluster_label'
    ]
]

cursor = conn.cursor()

# xóa bảng cũ nếu tồn tại
cursor.execute("""
IF OBJECT_ID('netflix_clusters', 'U') IS NOT NULL
DROP TABLE netflix_clusters
""")

conn.commit()

# tạo bảng mới
cursor.execute("""
CREATE TABLE netflix_clusters (
    user_id VARCHAR(20),
    cluster INT,
    cluster_label VARCHAR(100)
)
""")

conn.commit()

# insert từng dòng
for _, row in cluster_output.iterrows():

    cursor.execute("""
    INSERT INTO netflix_clusters
    VALUES (?, ?, ?)
    """,
    row['user_id'],
    int(row['cluster']),
    row['cluster_label']
    )

conn.commit()

print('\n✅ Cluster results pushed to SQL Server.')

# =========================================================
# 15. CLOSE CONNECTION
# =========================================================

conn.close()

print('\n✅ Pipeline completed successfully.')