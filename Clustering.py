# =========================================================
# 3_CLUSTERING_K-MEANS
# Phân nhóm khách hàng dựa trên PCA output
# =========================================================
 
# =========================
# 1. IMPORT LIBRARIES
# =========================
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
 
# =========================
# 2. LOAD PCA OUTPUT
# =========================
 
pca_df = pd.read_csv("pca_output_full.csv")
 
# fix tên cột bị thừa khoảng trắng
pca_df.columns = pca_df.columns.str.strip()
 
print("Columns:", pca_df.columns.tolist())
print("Shape:", pca_df.shape)
 
# =========================
# 3. TÁCH FEATURES VÀ LABEL
# =========================
 
X_pca = pca_df.drop(columns=['churned'])
y_churn = pca_df['churned']
 
print("\nFeatures used for clustering:")
print(X_pca.columns.tolist())
 
# =========================
# 4. TÌM K TỐI ƯU
#    Elbow + Silhouette
# =========================
 
inertia = []
silhouette_scores = []
K_range = range(2, 11)
 
for k in K_range:
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    km.fit(X_pca)
    inertia.append(km.inertia_)
    
    # Tính silhouette score với sample_size để không bị treo máy
    score = silhouette_score(X_pca, km.labels_, sample_size=5000)
    silhouette_scores.append(score)
    

 
# plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
axes[0].plot(K_range, inertia, marker='o', color='steelblue')
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].grid(True)
 
axes[1].plot(K_range, silhouette_scores, marker='o', color='orange')
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Score")
axes[1].grid(True)
 
plt.suptitle("Tìm K tối ưu cho K-Means", fontsize=13)
plt.tight_layout()
plt.show()
 
best_k = K_range[np.argmax(silhouette_scores)]
print(f"\nK tối ưu theo Silhouette: {best_k}")
 
# =========================
# 5. FIT K-MEANS VỚI K TỐI ƯU
# =========================
 
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

pca_df['cluster'] = kmeans.fit_predict(X_pca)
 
print("\nPhân bố cluster:")
print(pca_df['cluster'].value_counts().sort_index())
 
# =========================
# 6. VISUALIZE CLUSTERS
#    PC1 vs PC2
# =========================
 
plt.figure(figsize=(10, 6))
 
sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='cluster',
    palette='tab10',
    alpha=0.5,
    s=15
)
 
plt.title("K-Means Clusters trên PCA Space (PC1 vs PC2)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.grid(True, alpha=0.3)
plt.show()
 
# =========================
# 7. PHÂN TÍCH CHURN THEO CỤM
# =========================
 
cluster_analysis = pca_df.groupby('cluster').agg(
    so_khach=('churned', 'count'),
    so_churn=('churned', 'sum'),
    churn_rate=('churned', 'mean')
).reset_index()
 
cluster_analysis['churn_rate_%'] = (
    cluster_analysis['churn_rate'] * 100
).round(2)
 
cluster_analysis = cluster_analysis.drop(columns=['churn_rate'])
 
print("\n===== Churn Rate theo Cluster =====")
print(cluster_analysis.to_string(index=False))
 
# plot churn rate theo cluster
plt.figure(figsize=(8, 5))
 
bars = plt.bar(
    cluster_analysis['cluster'].astype(str),
    cluster_analysis['churn_rate_%'],
    color='salmon',
    edgecolor='black',
    width=0.5
)
 
for bar, val in zip(bars, cluster_analysis['churn_rate_%']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        f"{val}%",
        ha='center',
        va='bottom',
        fontsize=10
    )
 
plt.title("Churn Rate theo Cluster")
plt.xlabel("Cluster")
plt.ylabel("Churn Rate (%)")
plt.ylim(0, cluster_analysis['churn_rate_%'].max() + 5)
plt.grid(axis='y', alpha=0.3)
plt.show()
 
# =========================
# 8. THỐNG KÊ PC1 PC2 THEO CỤM
# =========================
 
print("\n===== Trung bình PC1, PC2 theo Cluster =====")
print(
    pca_df.groupby('cluster')[['PC1', 'PC2']]
    .mean()
    .round(3)
    .to_string()
)
 
# =========================
# 9. EXPORT
# =========================
 
pca_df.to_csv("clustering_output.csv", index=False)
 
print("\nclustering_output.csv saved!")
print("Shape:", pca_df.shape)
print("Columns:", pca_df.columns.tolist())
