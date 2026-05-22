# =========================================================
# LINEAR REGRESSION – DỰ ĐOÁN TOTAL REVENUE TO DATE
# Target : total_revenue_to_date
# Input  : netflix_step1_enriched.csv + clustering_output.csv
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.rcParams.update({
    'figure.dpi': 120, 'axes.titlesize': 13,
    'axes.titleweight': 'bold', 'figure.facecolor': '#F8F9FA',
    'axes.facecolor': '#FFFFFF',
})

print('✅ Thư viện sẵn sàng.')

# =========================
# 1. LOAD DATA
# =========================

df      = pd.read_csv('netflix_step1_enriched.csv')
clus_df = pd.read_csv('clustering_output.csv')

df.columns      = df.columns.str.strip()
clus_df.columns = clus_df.columns.str.strip()

# Gán cluster vào df
df['cluster'] = clus_df['cluster'].values.astype(str)

# Bỏ dòng thiếu target
df = df.dropna(subset=['total_revenue_to_date'])

print(f'Shape: {df.shape}')
print(f"Target mean={df['total_revenue_to_date'].mean():.2f}  "
      f"std={df['total_revenue_to_date'].std():.2f}")

# =========================
# 2. ENCODE
# =========================

# devices_used → đếm số thiết bị
if df['devices_used'].dtype == object:
    df['devices_used_enc'] = df['devices_used'].apply(
        lambda x: len(str(x).split(','))
    )
else:
    df['devices_used_enc'] = df['devices_used']

# One-Hot encode: country, subscription_type, cluster
df = pd.get_dummies(df, columns=['country', 'subscription_type', 'cluster'],
                    drop_first=True)

country_cols  = [c for c in df.columns if c.startswith('country_')]
sub_cols      = [c for c in df.columns if c.startswith('subscription_type_')]
cluster_cols  = [c for c in df.columns if c.startswith('cluster_')]

print(f'\nEncode xong:')
print(f'  country    : {len(country_cols)} cột')
print(f'  sub_type   : {len(sub_cols)} cột')
print(f'  cluster    : {len(cluster_cols)} cột')

# =========================
# 3. FEATURES & TARGET
# =========================

TARGET = 'total_revenue_to_date'

FEATURES = (
    [
        'age',
        'monthly_fee',
        'devices_used_enc',
        'avg_watch_time_minutes',
        'watch_sessions_per_week',
        'binge_watch_sessions',
        'completion_rate',
        'rating_given',
        'content_interactions',
        'recommendation_click_rate',
        'days_since_last_login',
    ]
    + country_cols
    + sub_cols
    + cluster_cols
)

FEATURES = [c for c in FEATURES if c in df.columns]

X = df[FEATURES].fillna(0)
y = df[TARGET]

print(f'\nTổng features : {len(FEATURES)} cột')
print(f'Target        : {TARGET}')

# =========================
# 4. TRAIN/TEST SPLIT & SCALE
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f'Train: {len(X_train):,}  |  Test: {len(X_test):,}')

# =========================
# 5. FIT 3 MODELS
# =========================

cv = KFold(3, shuffle=True, random_state=42)

# OLS
ols = LinearRegression()
ols.fit(X_train_s, y_train)
ols_pred = ols.predict(X_test_s)
ols_r2   = r2_score(y_test, ols_pred)
ols_rmse = np.sqrt(mean_squared_error(y_test, ols_pred))
ols_mae  = mean_absolute_error(y_test, ols_pred)
ols_cv   = cross_val_score(ols, X_train_s, y_train, scoring='r2', cv=cv)

# Ridge – tìm alpha tối ưu
ridge_gs = GridSearchCV(
    Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100, 500]},
    scoring='r2', cv=cv, n_jobs=-1
)
ridge_gs.fit(X_train_s, y_train)
ridge      = ridge_gs.best_estimator_
ridge_pred = ridge.predict(X_test_s)
ridge_r2   = r2_score(y_test, ridge_pred)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_cv   = cross_val_score(ridge, X_train_s, y_train, scoring='r2', cv=cv)
best_alpha_ridge = ridge_gs.best_params_['alpha']

# Lasso – tìm alpha tối ưu
lasso_gs = GridSearchCV(
    Lasso(max_iter=5000), {'alpha': [ 0.01, 0.1, 1]},
    scoring='r2', cv=cv, n_jobs=-1
)
lasso_gs.fit(X_train_s, y_train)
lasso      = lasso_gs.best_estimator_
lasso_pred = lasso.predict(X_test_s)
lasso_r2   = r2_score(y_test, lasso_pred)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_cv   = cross_val_score(lasso, X_train_s, y_train, scoring='r2', cv=cv)
best_alpha_lasso = lasso_gs.best_params_['alpha']

# Chọn best model
results = [
    ('OLS',   ols,   ols_r2,   ols_rmse, ols_mae, ols_cv,   ols_pred),
    ('Ridge', ridge, ridge_r2, ridge_rmse, ridge_mae, ridge_cv, ridge_pred),
    ('Lasso', lasso, lasso_r2, lasso_rmse,lasso_mae, lasso_cv, lasso_pred),
]
best_name, best_model, best_r2, best_rmse, best_mae,best_cv, best_pred = max(
    results, key=lambda x: x[2]
)

def rate_r2(r):
    if r < 0.20:   return 'Yếu'
    elif r < 0.40: return 'Chấp nhận'
    elif r < 0.60: return 'Tốt ✅'
    elif r < 0.80: return 'Rất tốt ✅'
    else:          return 'Xuất sắc ✅'

# =========================
# 6. IN KẾT QUẢ
# =========================

print()
print('=' * 60)
print('        📊 KẾT QUẢ LINEAR REGRESSION')
print('=' * 60)
print(f'  {"Model":<15} {"R²":>8}  {"RMSE":>10}  {"MAE":>10} {"CV R² (mean)":>14}')
print(f'  {"-"*52}')
print(f'  {"OLS":<15} {ols_r2:>8.4f}  {ols_rmse:>10.2f}   {ols_mae:>10.2f}   {ols_cv.mean():>14.4f}')
print(f'  {"Ridge (α="+str(best_alpha_ridge)+")":<15} {ridge_r2:>8.4f}  {ridge_rmse:>10.2f}   {ridge_mae:>10.2f}   {ridge_cv.mean():>14.4f}')
print(f'  {"Lasso (α="+str(best_alpha_lasso)+")":<15} {lasso_r2:>8.4f}  {lasso_rmse:>10.2f}   {lasso_mae:>10.2f}   {lasso_cv.mean():>14.4f}')
print(f'  {"-"*52}')
print(f'  ✅ Best: {best_name}')
print(f'     R²             : {best_r2:.4f}  → {rate_r2(best_r2)}')
print(f'     RMSE           : {best_rmse:.2f}')
print(f'     MAE            : {best_mae:.2f}')
print(f'     CV R² (5-fold) : {best_cv.mean():.4f} ± {best_cv.std():.4f}')
print('=' * 60)
print(f'\n  📌 R² = {best_r2:.4f}')
print(f'  → Model giải thích được {best_r2*100:.1f}% sự biến động')
print(f'    của {TARGET}')
print(f'  → Đánh giá: {rate_r2(best_r2)}')

# =========================
# 7. BIỂU ĐỒ
# =========================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Linear Regression – Dự đoán {TARGET}', fontsize=14, fontweight='bold')

# So sánh R² 3 models
ax = axes[0]
model_names = ['OLS', f'Ridge\n(α={best_alpha_ridge})', f'Lasso\n(α={best_alpha_lasso})']
r2_vals     = [ols_r2, ridge_r2, lasso_r2]
bar_colors  = ['#2ECC71' if n.split('\n')[0] == best_name else '#95A5A6'
               for n in model_names]
bars = ax.bar(model_names, r2_vals, color=bar_colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.set_title('So sánh R² – 3 Models')
ax.set_ylabel('R² Score')
ax.set_ylim(0, max(r2_vals) + 0.1)
ax.grid(axis='y', alpha=0.3)

# Actual vs Predicted
ax = axes[1]
sample = np.random.choice(len(y_test), size=min(3000, len(y_test)), replace=False)
ax.scatter(y_test.iloc[sample], best_pred[sample],
           alpha=0.3, s=8, color='#3498DB', edgecolors='none')
mn, mx = y_test.min(), y_test.max()
ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect fit')
ax.set_title(f'Actual vs Predicted ({best_name})\nR² = {best_r2:.4f}')
ax.set_xlabel(f'Actual {TARGET}')
ax.set_ylabel(f'Predicted {TARGET}')
ax.legend(); ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('linear_regression_final.png', bbox_inches='tight', dpi=120)
plt.show()
print('✅ Hình đã lưu: linear_regression_final.png')


coef_sorted = pd.DataFrame({
    'Feature'    : FEATURES,
    'Coefficient': best_model.coef_,
    'Abs'        : np.abs(best_model.coef_),
    'Direction'  : ['↑ Tăng revenue' if c > 0 else '↓ Giảm revenue'
                    for c in best_model.coef_],
}).sort_values('Abs', ascending=False).reset_index(drop=True)

print('\n===== TOP 15 FEATURES ẢNH HƯỞNG NHẤT =====')
print(coef_sorted[['Feature', 'Coefficient', 'Direction']].head(15).to_string(index=False))

# =========================
# 9. EXPORT
# =========================

result_df = X_test.copy()
result_df[f'actual_{TARGET}']    = y_test.values
result_df[f'predicted_{TARGET}'] = best_pred
result_df['residual']            = y_test.values - best_pred
result_df.to_csv('linear_regression_output.csv', index=False)

print(f'\n✅ Đã lưu linear_regression_output.csv  |  Shape: {result_df.shape}')
print(f'\n{"="*60}')
print(f'  TÓM TẮT – {best_name} (Best Model)')
print(f'{"="*60}')
print(f'  Features   : {len(FEATURES)} cột')
print(f'  R²         : {best_r2:.4f}  → {rate_r2(best_r2)}')
print(f'  RMSE       : {best_rmse:.2f}')
print(f'  MAE        : {mean_absolute_error(y_test, best_pred):.2f}')
print(f'  CV R²      : {best_cv.mean():.4f} ± {best_cv.std():.4f}')
print(f'{"="*60}')