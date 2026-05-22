# =========================================================================
# LINEAR REGRESSION (CẢI TIẾN) – DỰ ĐOÁN SỐ THÁNG GẮN BÓ DỰA TRÊN HÀNH VI
# Target : account_age_months (Xóa bỏ hoàn toàn rò rỉ dữ liệu tài chính)
# =========================================================================

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

# 1. LOAD DATA
df = pd.read_csv('netflix_step1_enriched.csv')
clus_df = pd.read_csv('clustering_output.csv')

df.columns = df.columns.str.strip()
clus_df.columns = clus_df.columns.str.strip()
df['cluster'] = clus_df['cluster'].values.astype(str)

for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.strip()

# 2. ONE-HOT ENCODING (Giữ lại cả gói cước vì dữ liệu không bị đa cộng tuyến)
df = pd.get_dummies(df, columns=['country', 'subscription_type', 'cluster', 'gender', 'favorite_genre'], drop_first=True)

country_cols = [c for c in df.columns if c.startswith('country_')]
sub_cols     = [c for c in df.columns if c.startswith('subscription_type_')]
cluster_cols = [c for c in df.columns if c.startswith('cluster_')]
gender_cols  = [c for c in df.columns if c.startswith('gender_')]
genre_cols   = [c for c in df.columns if c.startswith('favorite_genre_')]

# 3. THIẾT LẬP FEATURES & TARGET MỚI (Chuẩn học máy thực thụ)
TARGET = 'account_age_months' # Dự đoán thời gian gắn bó

FEATURES = [
    'age',
    'monthly_fee',
    'devices_used',
    'avg_watch_time_minutes',
    'watch_sessions_per_week',
    'binge_watch_sessions',
    'completion_rate',
    'rating_given',
    'content_interactions',
    'recommendation_click_rate',
    'days_since_last_login',
    'watch_efficiency',
    'engagement_rate'
] + country_cols + sub_cols + cluster_cols + gender_cols + genre_cols

# Loại bỏ tuyệt đối biến tổng doanh thu để tránh lặp logic đại số
X = df[FEATURES].fillna(0)
y = df[TARGET]

print(f'📦 Tổng features đưa vào mô hình học máy thực thụ: {len(FEATURES)} cột')

# 4. TRAIN/TEST SPLIT & SCALE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 5. GRIDSEARCH CHẠY 3 MODELS (OLS, RIDGE, LASSO)
cv = KFold(5, shuffle=True, random_state=42)
results = {}

# OLS
ols = LinearRegression()
ols.fit(X_train_s, y_train)
results['OLS'] = {
    'model': ols, 'pred': ols.predict(X_test_s),
    'r2': r2_score(y_test, ols.predict(X_test_s)),
    'rmse': np.sqrt(mean_squared_error(y_test, ols.predict(X_test_s)))
}

# Ridge
ridge_gs = GridSearchCV(Ridge(), {'alpha': [0.1, 1, 10, 100, 500]}, scoring='r2', cv=cv)
ridge_gs.fit(X_train_s, y_train)
best_ridge = ridge_gs.best_estimator_
results['Ridge'] = {
    'model': best_ridge, 'pred': best_ridge.predict(X_test_s),
    'r2': r2_score(y_test, best_ridge.predict(X_test_s)),
    'rmse': np.sqrt(mean_squared_error(y_test, best_ridge.predict(X_test_s)))
}

# Lasso
lasso_gs = GridSearchCV(Lasso(max_iter=5000), {'alpha': [0.001, 0.01, 0.1, 1]}, scoring='r2', cv=cv)
lasso_gs.fit(X_train_s, y_train)
best_lasso = lasso_gs.best_estimator_
results['Lasso'] = {
    'model': best_lasso, 'pred': best_lasso.predict(X_test_s),
    'r2': r2_score(y_test, best_lasso.predict(X_test_s)),
    'rmse': np.sqrt(mean_squared_error(y_test, best_lasso.predict(X_test_s)))
}

print("\n" + "="*50)
print(f"{'Model':<12} | {'Test R2':<10} | {'RMSE (Tháng)':<10}")
print("-"*50)
for name, m_res in results.items():
    print(f"{name:<12} | {m_res['r2']:<10.4f} | {m_res['rmse']:<10.2f}")
print("="*50)