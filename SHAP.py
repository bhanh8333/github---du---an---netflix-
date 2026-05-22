import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 120, 'axes.titlesize': 13,
    'axes.titleweight': 'bold', 'axes.labelsize': 11,
    'figure.facecolor': '#F8F9FA', 'axes.facecolor': '#FFFFFF',
})
print('Thu vien san sang.')

df      = pd.read_csv('netflix_step1_enriched.csv')
clus_df = pd.read_csv('clustering_output.csv')

df.columns      = df.columns.str.strip()
clus_df.columns = clus_df.columns.str.strip()

for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.strip()

# Gan cluster
df['cluster'] = clus_df['cluster'].values.astype(str)
df = df.dropna(subset=['total_revenue_to_date'])

# Encode devices_used
if df['devices_used'].dtype == object:
    df['devices_used_enc'] = df['devices_used'].apply(
        lambda x: len(str(x).split(','))
    )
else:
    df['devices_used_enc'] = df['devices_used']

# One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=['country', 'subscription_type', 'cluster'],
    drop_first=True
)
df.columns = df.columns.str.strip()

country_cols = [c for c in df.columns if c.startswith('country_')]
sub_cols     = [c for c in df.columns if c.startswith('subscription_type_')]
cluster_cols = [c for c in df.columns if c.startswith('cluster_')]

FEATURES = (
    ['age', 'monthly_fee', 'devices_used_enc', 'avg_watch_time_minutes',
     'watch_sessions_per_week', 'binge_watch_sessions', 'completion_rate',
     'rating_given', 'content_interactions', 'recommendation_click_rate',
     'days_since_last_login']
    + country_cols + sub_cols + cluster_cols
)
FEATURES = [c for c in FEATURES if c in df.columns]

X = df[FEATURES].fillna(0)
y = df['total_revenue_to_date']

print(f'Shape: {df.shape}')
print(f'Features: {len(FEATURES)}')
print(f'Target mean: ${y.mean():.2f}  std: ${y.std():.2f}')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# GridSearchCV tim alpha toi uu
cv = KFold(5, shuffle=True, random_state=42)
ridge_gs = GridSearchCV(
    Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100, 500]},
    scoring='r2', cv=cv, n_jobs=-1
)
ridge_gs.fit(X_train_s, y_train)
model      = ridge_gs.best_estimator_
best_alpha = ridge_gs.best_params_['alpha']

y_pred = model.predict(X_test_s)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f'Best alpha : {best_alpha}')
print(f'R2         : {r2:.4f}')
print(f'RMSE       : ${rmse:.2f}')
print(f'MAE        : ${mae:.2f}')

# Tinh SHAP values
shap_matrix = X_test_s * model.coef_   # (n_samples, n_features)
shap_df     = pd.DataFrame(shap_matrix, columns=FEATURES)

# Tong hop
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
mean_shap     = shap_df.mean().sort_values(ascending=False)
base_value    = float(y_train.mean())

print(f'Base value (mean revenue train): ${base_value:.2f}')
print(f'\nTop 10 features theo Mean |SHAP|:')
print(mean_abs_shap.head(10).round(3).to_string())
print(f'\nKiem tra: base + sum(SHAP) vs prediction')
check = base_value + shap_df.sum(axis=1).mean()
print(f'  base + mean(sum SHAP) = {check:.2f}')
print(f'  mean(y_pred)          = {y_pred.mean():.2f}')

top1_feat = mean_abs_shap.index[0]

print('\nGiai thich:')
print(f'  - {top1_feat}: feature anh huong lon nhat') 
print('  - SHAP value the hien dong gop ($) vao du doan revenue')

top10_feats = mean_abs_shap.head(10).index.tolist()
np.random.seed(42)
sample_idx  = np.random.choice(len(shap_df), size=min(1000, len(shap_df)), replace=False)
shap_sample = shap_df.iloc[sample_idx][top10_feats]
X_std_df    = pd.DataFrame(X_test_s, columns=FEATURES)
X_sample    = X_std_df.iloc[sample_idx][top10_feats]

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('#F8F9FA')

for i, feat in enumerate(reversed(top10_feats)):
    sv        = shap_sample[feat].values
    fv        = X_sample[feat].values
    fmin, fmax = fv.min(), fv.max()
    norm_fv   = (fv - fmin) / (fmax - fmin + 1e-9)
    colors_dot = plt.cm.RdYlBu_r(norm_fv)
    jitter     = np.random.uniform(-0.25, 0.25, size=len(sv))
    ax.scatter(sv, i + jitter, c=colors_dot, alpha=0.4, s=10, edgecolors='none')

ax.set_yticks(range(len(top10_feats)))
ax.set_yticklabels(list(reversed(top10_feats)), fontsize=10)
ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.7)
ax.set_xlabel('SHAP Value ($) — dong gop vao du doan revenue', fontsize=10)
ax.set_title(
    'SHAP Beeswarm Plot — Top 10 Features\n'
    'Moi cham = 1 khach hang  |  Mau do = gia tri cao  |  Mau xanh = gia tri thap',
    fontsize=12, fontweight='bold'
)
ax.grid(axis='x', alpha=0.2)
ax.set_facecolor('#FAFAFA')

sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(0,1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
cbar.set_label('Feature value (low → high)', fontsize=9)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(['Thap', 'TB', 'Cao'])

plt.tight_layout()
plt.savefig('shap_beeswarm.png', bbox_inches='tight', dpi=120)
plt.show()

# Chon khach hang co tong |SHAP| cao nhat
idx_example  = shap_df.abs().sum(axis=1).idxmax()
shap_one     = shap_df.loc[idx_example].sort_values(key=abs, ascending=False).head(10)
actual_val   = float(y_test.loc[idx_example]) if idx_example in y_test.index else float(y_test.iloc[0])

# Tinh cumulative cho waterfall
cumulative  = base_value
bar_starts  = []
bar_heights = []
for v in shap_one.values:
    bar_starts.append(cumulative)
    bar_heights.append(v)
    cumulative += v

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#F8F9FA')

colors_wf = ['#2ECC71' if v > 0 else '#E74C3C' for v in shap_one.values]
ax.barh(range(len(shap_one)), bar_heights, left=bar_starts,
        color=colors_wf, edgecolor='white', height=0.6)

for i, (start, height, val) in enumerate(zip(bar_starts, bar_heights, shap_one.values)):
    label = f'+${val:.1f}' if val > 0 else f'-${abs(val):.1f}'
    offset = max(np.abs(bar_heights)) * 0.02
    x_pos  = start + height + (offset if val > 0 else -offset)
    ax.text(x_pos, i, label, va='center', fontsize=9, fontweight='bold',
            color='#2ECC71' if val > 0 else '#E74C3C',
            ha='left' if val > 0 else 'right')

ax.axvline(base_value, color='gray', lw=1.5, ls='--',
           label=f'Base value = ${base_value:.0f} (trung binh revenue)')
ax.axvline(cumulative, color='#185FA5', lw=2,
           label=f'Prediction = ${cumulative:.0f}')

ax.set_yticks(range(len(shap_one)))
ax.set_yticklabels(shap_one.index, fontsize=10)
ax.set_xlabel('Revenue ($)', fontsize=10)
ax.set_title(
    f'SHAP Waterfall — Giai thich du doan 1 khach hang\n'
    f'Base=${base_value:.0f}  →  Prediction=${cumulative:.0f}  |  '
    f'Actual=${actual_val:.0f}',
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.2)
ax.set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig('shap_waterfall.png', bbox_inches='tight', dpi=120)
plt.show()

print('Giai thich waterfall:')
print(f'  Bat dau tu base value = ${base_value:.0f}')
print(f'  Tung feature cong/tru vao du doan')
print(f'  Ket qua cuoi: ${cumulative:.0f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'SHAP Dependence Plot — Moi quan he Feature <-> SHAP Value\n'
    '(Mau = completion_rate: do = cao, xanh = thap)',
    fontsize=13, fontweight='bold'
)

X_std_df    = pd.DataFrame(X_test_s, columns=FEATURES)
inter_feat  = 'completion_rate'
inter_vals  = X_std_df[inter_feat].values
norm_inter  = (inter_vals - inter_vals.min()) / (inter_vals.max() - inter_vals.min() + 1e-9)

for ax, feat in zip(axes, ['monthly_fee', 'avg_watch_time_minutes']):
    if feat not in shap_df.columns:
        continue
    x_vals = X_std_df[feat].values
    sv     = shap_df[feat].values

    sc = ax.scatter(x_vals, sv, c=norm_inter, cmap='RdYlBu_r',
                    alpha=0.3, s=8, edgecolors='none')
    ax.axhline(0, color='black', lw=1, ls='--', alpha=0.5)

    # Trend line
    z = np.polyfit(x_vals, sv, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'k-', lw=1.5, alpha=0.7, label='Trend')

    ax.set_xlabel(f'{feat} (chuan hoa)', fontsize=10)
    ax.set_ylabel(f'SHAP value of {feat} ($)', fontsize=10)
    ax.set_title(f'Dependence: {feat}', fontsize=11)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label(f'{inter_feat}\n(thap → cao)', fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Thap', 'TB', 'Cao'])

plt.tight_layout()
plt.savefig('shap_dependence.png', bbox_inches='tight', dpi=120)
plt.show()

print('=' * 62)
print('       TOM TAT KET QUA SHAP — EXPLAINABLE AI')
print('=' * 62)
print(f'Model     : Ridge Regression (alpha={best_alpha})')
print(f'Target    : total_revenue_to_date')
print(f'R2        : {r2:.4f}')
print(f'Base value: ${base_value:.2f} (trung binh revenue train set)')
print()
print('Top 5 features anh huong nhat (Mean |SHAP|):')
for feat, val in mean_abs_shap.head(5).items():
    direction = '(tang revenue)' if model.coef_[FEATURES.index(feat)] > 0 else '(giam revenue)'
    print(f'  {feat:<35s} ${val:.2f}  {direction}')
print()
print('Y nghia 3 bieu do SHAP:')
print('  1. Beeswarm      : Gia tri feature cao/thap anh huong the nao')
print('  2. Waterfall     : Giai thich du doan 1 khach hang cu the')
print('  3. Dependence    : Moi quan he phi tuyen giua feature va SHAP')
print('=' * 62)
