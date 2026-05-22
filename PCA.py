import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'figure.facecolor': '#F8F9FA',
    'axes.facecolor': '#FFFFFF',
})

print('Thư viện sẵn sàng.')

df = pd.read_csv('netflix_step1_enriched.csv')

df.columns = df.columns.str.strip()
for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.strip()

print(f'Shape gốc: {df.shape}  ({df.shape[1]} cột)')
print(f'\nKiểu dữ liệu từng cột:')
print(df.dtypes)

# Encode churned để dùng cho visualization (KHÔNG đưa vào PCA)
le = LabelEncoder()
df['churned_enc'] = le.fit_transform(df['churned'])

# Lấy TẤT CẢ cột số — loại target
numeric_features = (
    df.select_dtypes(include=[np.number])
      .columns
      .drop('churned_enc')   # loại target
      .tolist()
)

# Tóm tắt phân loại
col_categorical = df.select_dtypes('object').columns.drop('churned').tolist()
print('=' * 55)
print(f'Tổng cột gốc             : 23')
print(f'Loại ID (user_id)        : 1 cột')
print(f'Loại categorical         : {len(col_categorical)} cột → {col_categorical}')
print(f'Loại target (churned)    : 1 cột')
print('─' * 55)
print(f'Features đưa vào PCA     : {len(numeric_features)} cột')
for i, c in enumerate(numeric_features, 1):
    print(f'   {i:2d}. {c}')
print('=' * 55)

X = df[numeric_features].copy()

# Kiểm tra missing
missing = X.isnull().sum()
print('Missing values:')
print(missing[missing > 0] if missing.sum() > 0 else '  Không có missing value')

X = X.fillna(X.mean())

# Chuẩn hóa — bắt buộc với PCA để các biến có scale khác nhau không át nhau
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f'\nShape sau chuẩn hóa: {X_scaled.shape}')
print(f'Mean ≈ {X_scaled.mean():.4f},  Std ≈ {X_scaled.std():.4f}  (sau StandardScaler)')
pca_full = PCA()
pca_full.fit(X_scaled)

ev  = pca_full.explained_variance_ratio_
cum = np.cumsum(ev)
eig = pca_full.explained_variance_
n   = len(ev)

summary = pd.DataFrame({
    'PC'             : [f'PC{i+1}' for i in range(n)],
    'Eigenvalue'     : eig.round(3),
    'Variance (%)'   : (ev * 100).round(2),
    'Cumulative (%)' : (cum * 100).round(2),
    'Kaiser (>1)'    : ['Yes' if e > 1 else 'No' for e in eig],
})
display(summary)

print()
for pct in [0.70, 0.80, 0.85, 0.90, 0.95]:
    n_pc = np.argmax(cum >= pct) + 1
    print(f'  {int(pct*100)}% variance  →  {n_pc} PCs')

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('PCA – Phân tích chọn số Components tối ưu', fontsize=14, fontweight='bold')

pcs = range(1, n + 1)

# Scree Plot
ax = axes[0]
bar_colors = ['#185FA5' if i < 3 else '#BA7517' if i < 8 else '#d0dff5' for i in range(n)]
ax.bar(pcs, ev * 100, color=bar_colors, edgecolor='white', width=0.7)
ax.plot(pcs, ev * 100, 'o-', color='#2C2C2A', ms=4, lw=1.2)
ax.set_title('Scree Plot – Variance mỗi PC')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis='y', alpha=0.3)
ax.annotate('Elbow (PC3)', xy=(3, ev[2]*100), xytext=(5, ev[2]*100+2),
            arrowprops=dict(arrowstyle='->', color='#A32D2D'), color='#A32D2D', fontsize=9)

# Cumulative Variance
ax = axes[1]
ax.plot(pcs, cum * 100, 'o-', color='#3B6D11', lw=2, ms=5)
ax.fill_between(pcs, cum * 100, alpha=0.1, color='#3B6D11')
for pct, color, ls in [(80,'#BA7517','--'),(85,'#185FA5','-.'),(95,'#533AB7','--')]:
    ax.axhline(pct, color=color, ls=ls, lw=1.2, alpha=0.9)
    n_pc = np.argmax(cum >= pct/100) + 1
    ax.text(n + 0.1, pct, f'{pct}% → {n_pc}PCs', fontsize=8.5, va='center', color=color)
ax.set_title('Cumulative Explained Variance')
ax.set_xlabel('Số Principal Components')
ax.set_ylabel('Cumulative Variance (%)')
ax.set_xlim(1, n + 2.8)
ax.set_ylim(0, 108)
ax.grid(alpha=0.3)

# Eigenvalue (Kaiser)
ax = axes[2]
eig_colors = ['#185FA5' if e > 1 else '#E24B4A' for e in eig]
ax.bar(pcs, eig, color=eig_colors, edgecolor='white', width=0.7)
ax.axhline(1, color='#A32D2D', lw=1.8, ls='--', label='Kaiser = 1')
n_kaiser = int(sum(eig > 1))
ax.set_title('Eigenvalue – Kaiser Criterion')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Eigenvalue')
ax.legend(fontsize=9)
ax.text(n_kaiser + 0.4, 1.12, f'Kaiser: {n_kaiser} PCs', color='#A32D2D', fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('pca_selection.png', bbox_inches='tight', dpi=120)
plt.show()

print(f'Kết luận:')
print(f'  Elbow Point      → PC3   ({cum[2]*100:.2f}% variance) — tốt cho clustering/viz')
print(f'  Kaiser Criterion → {n_kaiser} PCs  ({cum[n_kaiser-1]*100:.2f}% variance)')
print(f'  95% Variance     → {np.argmax(cum>=0.95)+1} PCs  — phù hợp cho ML pipeline')

pca = PCA(n_components=0.85)
X_pca = pca.fit_transform(X_scaled)

n_selected = pca.n_components_
pca_cols   = [f'PC{i+1}' for i in range(n_selected)]

pca_df = pd.DataFrame(X_pca, columns=pca_cols)
pca_df['churned'] = df['churned_enc'].values

print('=' * 55)
print(f'Shape gốc (features số) : {X_scaled.shape}')
print(f'Shape sau PCA (95%)     : {X_pca.shape}')
print(f'Số PCs được chọn        : {n_selected}')
print(f'Variance giữ được       : {pca.explained_variance_ratio_.sum()*100:.2f}%')
print(f'Mức giảm chiều          : {len(numeric_features)} → {n_selected} '
      f'({(1-n_selected/len(numeric_features))*100:.0f}% reduction)')
print('=' * 55)
print()
print('PCA DataFrame (5 dòng đầu):')
display(pca_df.head())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('PCA Visualization – PC1 vs PC2', fontsize=14, fontweight='bold')

colors = {0: '#2ECC71', 1: '#E74C3C'}
labels_map = {0: 'Retained (No Churn)', 1: 'Churned'}

# Scatter
ax = axes[0]
for churn_val in [0, 1]:
    mask = pca_df['churned'] == churn_val
    ax.scatter(
        pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
        c=colors[churn_val], label=labels_map[churn_val],
        alpha=0.25, s=8, edgecolors='none'
    )
ax.set_title('Phân bố theo Churn Status')
ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% variance)')
ax.legend(markerscale=3, fontsize=9)
ax.grid(alpha=0.2)

# Density
ax = axes[1]
for churn_val, color, label in [(0,'#2ECC71','Retained'),(1,'#E74C3C','Churned')]:
    mask = pca_df['churned'] == churn_val
    sns.kdeplot(
        x=pca_df.loc[mask,'PC1'], y=pca_df.loc[mask,'PC2'],
        ax=ax, color=color, label=label, levels=5, alpha=0.7
    )
ax.set_title('Density Plot – PC1 vs PC2')
ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% variance)')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('pca_scatter.png', bbox_inches='tight', dpi=120)
plt.show()

loading_matrix = pd.DataFrame(
    pca.components_,
    index=pca_cols,
    columns=numeric_features
)

print('PCA Loading Matrix (giá trị lớn = feature đóng góp nhiều vào PC đó):')
display(loading_matrix.round(3))

plt.figure(figsize=(14, max(5, n_selected * 0.75)))
sns.heatmap(
    loading_matrix,
    annot=True, fmt='.2f',
    cmap='coolwarm', center=0,
    linewidths=0.4, linecolor='white',
    annot_kws={'size': 8}
)
plt.title('PCA Component Loadings', fontsize=12, fontweight='bold')
plt.xlabel('Features gốc', fontsize=10)
plt.ylabel('Principal Components', fontsize=10)
plt.xticks(rotation=35, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig('pca_loadings.png', bbox_inches='tight', dpi=120)
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))

np.random.seed(42)
sample_idx = np.random.choice(len(pca_df), size=min(2000, len(pca_df)), replace=False)
sample = pca_df.iloc[sample_idx]

for churn_val in [0, 1]:
    mask = sample['churned'] == churn_val
    ax.scatter(
        sample.loc[mask, 'PC1'], sample.loc[mask, 'PC2'],
        c=colors[churn_val], alpha=0.3, s=12, edgecolors='none'
    )

scale = max(pca_df['PC1'].abs().max(), pca_df['PC2'].abs().max()) * 0.6
for feat in numeric_features:
    vx = loading_matrix.loc['PC1', feat] * scale
    vy = loading_matrix.loc['PC2', feat] * scale
    ax.annotate('', xy=(vx, vy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#185FA5', lw=1.5))
    ax.text(vx * 1.1, vy * 1.1, feat, fontsize=8, color='#185FA5', ha='center')

ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.axvline(0, color='gray', lw=0.8, ls='--')
ax.set_title('Biplot – PC1 vs PC2 với Loading Vectors', fontsize=13, fontweight='bold')
ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% variance)')
ax.legend(handles=[Patch(facecolor='#2ECC71', label='Retained'),
                   Patch(facecolor='#E74C3C', label='Churned')], fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('pca_biplot.png', bbox_inches='tight', dpi=120)
plt.show()

print('===== Ý NGHĨA CÁC PRINCIPAL COMPONENTS =====')
print()
for i, pc in enumerate(pca_cols):
    loadings = loading_matrix.loc[pc].abs().sort_values(ascending=False)
    top3 = loadings.head(3)
    print(f'{pc}  ({ev[i]*100:.2f}% variance):')
    for feat, val in top3.items():
        sign = '+' if loading_matrix.loc[pc, feat] > 0 else '-'
        print(f'   {sign} {feat:35s} loading = {val:.3f}')
    print()

pca_df.to_csv('pca_output_full.csv', index=False)
print('Da luu pca_output_full.csv')

print()
print('=' * 55)
print('            TOM TAT KET QUA PCA')
print('=' * 55)
print(f'Dataset goc               : 50,000 x 23 cot')
print(f'Cot dua vao PCA           : {len(numeric_features)} cot so')
print(f'  Loai categorical        : 6 cot')
print(f'  Loai target (churned)   : 1 cot')
print(f'  Loai ID (user_id)       : 1 cot (trong categorical)')
print()
print(f'Nguong giu lai            : 95% variance')
print(f'So PCs duoc chon          : {n_selected}')
print(f'Variance giu duoc         : {pca.explained_variance_ratio_.sum()*100:.2f}%')
print(f'Muc giam chieu            : {len(numeric_features)} -> {n_selected} '
      f'({(1-n_selected/len(numeric_features))*100:.0f}% reduction)')
print()
print('Y nghia PC chinh:')
print('  PC1 = Revenue & Account Tenure')
print('  PC2 = Engagement & Content Interaction')
print('  PC3 = Watch Activity')
print()
print('Loi ich sau PCA:')
print('  Loai bo multicollinearity giua cac features')
print('  Giam nhieu, tang toc Machine Learning')
print('  Ho tro clustering va visualization')
print('=' * 55)

# Export PCA output ra CSV
pca_df.to_csv('pca_output_full.csv', index=False)

print(f'Da luu pca_output_full.csv')
print(f'Shape: {pca_df.shape}')
print(f'Columns: {pca_df.columns.tolist()}')
pca_df.head()
