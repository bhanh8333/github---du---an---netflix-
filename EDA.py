import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
from IPython.display import display
warnings.filterwarnings('ignore')

# ─── Style ───────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'figure.dpi': 130,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.facecolor': '#F8F9FA',
    'axes.facecolor': '#FFFFFF',
})

# Color palette
CLR_CHURN   = '#E74C3C'   # red   – churned
CLR_RETAIN  = '#2ECC71'   # green – retained
CLR_ACCENT  = '#3498DB'   # blue  – neutral accent
CLR_GOLD    = '#F39C12'   # gold  – highlight

FEE_COLORS = {'7.99': '#E74C3C', '12.99': '#F39C12', '15.99': '#2ECC71'}

print('✅ Thư viện đã sẵn sàng.')

df = pd.read_csv('netflix_step1_enriched.csv')

# Làm sạch tên cột & giá trị chuỗi
df.columns = df.columns.str.strip()
for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.strip()

# Biến mục tiêu nhị phân
df['churned_bin'] = (df['churned'] == 'Yes').astype(int)

# CLV ước tính
df['total_revenue_to_date'] = df['monthly_fee'] * df['account_age_months']

# Phân khúc CLV (tứ phân vị)
df['clv_segment'] = pd.qcut(
    df['total_revenue_to_date'], 4,
    labels=['Thấp (Low)', 'Trung Bình (Mid)', 'Cao (High)', 'Cao Cấp (Premium)']
)

# Label monthly_fee
fee_map = {7.99: 'Basic ($7.99)', 12.99: 'Standard ($12.99)', 15.99: 'Premium ($15.99)'}
df['fee_label'] = df['monthly_fee'].map(fee_map)

print(f'📦 Shape: {df.shape}')
print(f"👥 Tỷ lệ Churn: {df['churned_bin'].mean():.1%}")
df.head(3)

print('=== Thống kê mô tả các cột số ===\n')
display(df[['monthly_fee','account_age_months','total_revenue_to_date',
            'avg_watch_time_minutes','engagement_rate','days_since_last_login']].describe().round(2))

print('\n=== Phân phối Churn ===\n')
churn_dist = df['churned'].value_counts()
display(churn_dist.to_frame())

print('\n=== Phân phối Monthly Fee ===\n')
display(df['monthly_fee'].value_counts().sort_index().to_frame())

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle('Tổng quan CLV & Churn Rate', fontsize=15, fontweight='bold', y=1.01)

# ── 3.1 Phân phối CLV theo nhóm Churn ─────────────────────────────────────
ax = axes[0]
for status, color, label in [('No', CLR_RETAIN, 'Giữ lại'), ('Yes', CLR_CHURN, 'Rời bỏ')]:
    subset = df[df['churned'] == status]['total_revenue_to_date']
    ax.hist(subset, bins=40, alpha=0.6, color=color, label=label, edgecolor='none')
ax.axvline(df['total_revenue_to_date'].median(), color='gray', ls='--', lw=1.2, label='Median')
ax.set_title('Phân phối CLV ước tính')
ax.set_xlabel('CLV ($)')
ax.set_ylabel('Số lượng khách hàng')
ax.legend()

# ── 3.2 Churn Rate theo gói Phí ───────────────────────────────────────────
ax = axes[1]
fee_churn = df.groupby('fee_label')['churned_bin'].mean().reset_index()
fee_churn = fee_churn.sort_values('monthly_fee' if 'monthly_fee' in fee_churn.columns else 'churned_bin')
bars = ax.bar(fee_churn['fee_label'], fee_churn['churned_bin'],
              color=[CLR_CHURN, CLR_GOLD, CLR_RETAIN], width=0.5, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, fee_churn['churned_bin']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_title('Churn Rate theo Gói Cước')
ax.set_xlabel('')
ax.set_ylabel('Tỷ lệ Churn')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0, 0.30)
ax.tick_params(axis='x', rotation=10)

# ── 3.3 CLV trung bình theo phân khúc ─────────────────────────────────────
ax = axes[2]
seg_clv = df.groupby('clv_segment', observed=True)['churned_bin'].mean()
colors_seg = [CLR_CHURN, CLR_GOLD, CLR_ACCENT, CLR_RETAIN]
bars2 = ax.bar(seg_clv.index, seg_clv.values, color=colors_seg, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, seg_clv.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_title('Churn Rate theo Phân Khúc CLV')
ax.set_xlabel('Phân khúc CLV')
ax.set_ylabel('Tỷ lệ Churn')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0, 0.28)
ax.tick_params(axis='x', rotation=10)

plt.tight_layout()
plt.savefig('fig1_clv_overview.png', bbox_inches='tight', dpi=130)
plt.show()
print('✅ Hình 1 đã lưu.')

# Tính churn rate theo quốc gia × gói cước
country_fee_churn = (
    df.groupby(['country', 'fee_label'])['churned_bin']
    .agg(['mean', 'count'])
    .reset_index()
    .rename(columns={'mean': 'churn_rate', 'count': 'n_customers'})
)
country_fee_churn['monthly_fee_num'] = country_fee_churn['fee_label'].map(
    {'Basic ($7.99)': 7.99, 'Standard ($12.99)': 12.99, 'Premium ($15.99)': 15.99}
)
country_fee_churn = country_fee_churn.sort_values('monthly_fee_num')

display(country_fee_churn.pivot(index='country', columns='fee_label', values='churn_rate').round(3))

countries = df['country'].unique()
n_cols = 5
n_rows = int(np.ceil(len(countries) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7), sharey=False)
fig.suptitle('Churn Rate theo Gói Phí Hàng Tháng – Theo Từng Quốc Gia',
             fontsize=15, fontweight='bold', y=1.02)

fee_colors_list = [CLR_CHURN, CLR_GOLD, CLR_RETAIN]
fee_labels_order = ['Basic ($7.99)', 'Standard ($12.99)', 'Premium ($15.99)']

for idx, country in enumerate(sorted(countries)):
    ax = axes[idx // n_cols][idx % n_cols]
    subset = country_fee_churn[country_fee_churn['country'] == country].set_index('fee_label')
    subset = subset.reindex(fee_labels_order)

    bars = ax.bar(range(3), subset['churn_rate'].values,
                  color=fee_colors_list, width=0.6, edgecolor='white', linewidth=1.2)

    for i, (bar, val) in enumerate(zip(bars, subset['churn_rate'].values)):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_title(f'🌏 {country}', fontsize=11, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['$7.99', '$12.99', '$15.99'], fontsize=8)
    ax.set_ylabel('Churn Rate', fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, 0.32)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#FAFAFA')

# Tắt subplot dư
for extra in range(len(countries), n_rows * n_cols):
    fig.delaxes(axes[extra // n_cols][extra % n_cols])

# Legend chung
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=CLR_CHURN,   label='Basic ($7.99)'),
    Patch(facecolor=CLR_GOLD,    label='Standard ($12.99)'),
    Patch(facecolor=CLR_RETAIN,  label='Premium ($15.99)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
plt.savefig('fig2_churn_by_country_fee.png', bbox_inches='tight', dpi=130)
plt.show()
print('✅ Hình 2 đã lưu.')

pivot_heat = country_fee_churn.pivot(index='country', columns='fee_label', values='churn_rate')
pivot_heat = pivot_heat[fee_labels_order]

# Sắp xếp theo churn rate trung bình
pivot_heat['avg'] = pivot_heat.mean(axis=1)
pivot_heat = pivot_heat.sort_values('avg', ascending=False).drop(columns='avg')

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor('#F8F9FA')

sns.heatmap(
    pivot_heat,
    annot=True,
    fmt='.1%',
    cmap='RdYlGn_r',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Churn Rate', 'format': mticker.FuncFormatter(lambda x, _: f'{x:.0%}')},
    annot_kws={'size': 10, 'weight': 'bold'},
    ax=ax,
    vmin=0.17, vmax=0.25
)

ax.set_title('🌡️ Heatmap – Churn Rate theo Quốc Gia & Gói Cước',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Gói Cước', fontsize=11)
ax.set_ylabel('Quốc Gia', fontsize=11)
ax.tick_params(axis='both', labelsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('fig3_heatmap_churn.png', bbox_inches='tight', dpi=130)
plt.show()
print('✅ Hình 3 đã lưu.')

clv_country = (
    df.groupby(['country', 'churned'])['total_revenue_to_date']
    .mean()
    .unstack()
    .sort_values('No', ascending=True)
)

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor('#F8F9FA')

x = np.arange(len(clv_country))
width = 0.38

bars1 = ax.barh(x - width/2, clv_country['No'],   width, label='Giữ lại (No Churn)',  color=CLR_RETAIN, alpha=0.85)
bars2 = ax.barh(x + width/2, clv_country['Yes'],  width, label='Rời bỏ (Churned)',   color=CLR_CHURN,  alpha=0.85)

for bar in bars1:
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            f'${bar.get_width():.0f}', va='center', fontsize=8)
for bar in bars2:
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            f'${bar.get_width():.0f}', va='center', fontsize=8)

ax.set_yticks(x)
ax.set_yticklabels(clv_country.index, fontsize=10)
ax.set_xlabel('CLV Trung Bình ($)', fontsize=11)
ax.set_title('💰 CLV Trung Bình theo Quốc Gia – Giữ lại vs Rời bỏ',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig('fig4_clv_by_country.png', bbox_inches='tight', dpi=130)
plt.show()
print('✅ Hình 4 đã lưu.')

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Phân tích Monthly Fee × Account Age → CLV', fontsize=14, fontweight='bold')

# ── 7.1 Boxplot CLV theo Fee × Churn ──────────────────────────────────────
ax = axes[0]
data_box = [
    df[(df['fee_label'] == fl) & (df['churned'] == ch)]['total_revenue_to_date'].values
    for fl in fee_labels_order
    for ch in ['No', 'Yes']
]
positions = [1, 1.6, 3, 3.6, 5, 5.6]
colors_box = [CLR_RETAIN, CLR_CHURN] * 3
bp = ax.boxplot(data_box, positions=positions, widths=0.45, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                showfliers=False)
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticks([1.3, 3.3, 5.3])
ax.set_xticklabels(['Basic\n$7.99', 'Standard\n$12.99', 'Premium\n$15.99'], fontsize=10)
ax.set_ylabel('CLV ước tính ($)', fontsize=10)
ax.set_title('Phân phối CLV theo Gói Phí & Churn Status')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=CLR_RETAIN, label='Giữ lại'),
                   Patch(facecolor=CLR_CHURN,  label='Rời bỏ')], fontsize=9)
ax.grid(axis='y', alpha=0.3)

# ── 7.2 Scatter: Account Age vs CLV, tô màu theo Churn ───────────────────
ax = axes[1]
for status, color, label, alpha in [('No', CLR_RETAIN, 'Giữ lại', 0.15), ('Yes', CLR_CHURN, 'Rời bỏ', 0.35)]:
    sample = df[df['churned'] == status].sample(min(1500, len(df[df['churned'] == status])), random_state=42)
    ax.scatter(sample['account_age_months'], sample['total_revenue_to_date'],
               c=color, alpha=alpha, s=15, label=label)

# Trend lines
for status, color in [('No', CLR_RETAIN), ('Yes', CLR_CHURN)]:
    subset = df[df['churned'] == status]
    z = np.polyfit(subset['account_age_months'], subset['total_revenue_to_date'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(1, 59, 100)
    ax.plot(x_line, p(x_line), color=color, linewidth=2.5, linestyle='--')

ax.set_xlabel('Account Age (tháng)', fontsize=10)
ax.set_ylabel('CLV ước tính ($)', fontsize=10)
ax.set_title('Account Age vs CLV (theo Churn Status)')
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig('fig5_clv_scatter.png', bbox_inches='tight', dpi=130)
plt.show()
print('✅ Hình 5 đã lưu.')

# Tính chênh lệch churn rate: Basic – Premium theo quốc gia
pivot_diff = country_fee_churn.pivot(index='country', columns='fee_label', values='churn_rate')
pivot_diff = pivot_diff[fee_labels_order]
pivot_diff['diff_basic_premium'] = pivot_diff['Basic ($7.99)'] - pivot_diff['Premium ($15.99)']
pivot_diff = pivot_diff.sort_values('diff_basic_premium', ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#F8F9FA')

colors_diff = [CLR_CHURN if v > 0 else CLR_RETAIN for v in pivot_diff['diff_basic_premium']]
bars = ax.barh(pivot_diff.index, pivot_diff['diff_basic_premium'],
               color=colors_diff, edgecolor='white', height=0.6)

for bar, val in zip(bars, pivot_diff['diff_basic_premium']):
    ha = 'left' if val < 0 else 'right'
    offset = 0.001 if val < 0 else -0.001
    ax.text(val + offset * np.sign(val) * 5, bar.get_y() + bar.get_height()/2,
            f'{val:+.1%}', va='center', ha=ha, fontsize=9.5, fontweight='bold')

ax.axvline(0, color='black', linewidth=1.2)
ax.set_xlabel('Chênh lệch Churn Rate (Basic − Premium)', fontsize=11)
ax.set_title('📊 Chênh lệch Churn Rate: Gói Basic ($7.99) vs Premium ($15.99) theo Quốc Gia',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor('#FAFAFA')

# Annotation
ax.text(0.98, 0.02, '🔴 Đỏ: Basic có churn cao hơn Premium\n🟢 Xanh: Premium có churn cao hơn Basic',
        transform=ax.transAxes, fontsize=8, va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('fig6_churn_diff_by_country.png', bbox_inches='tight', dpi=130)
plt.show()
print('✅ Hình 6 đã lưu.')

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle('Yếu Tố Hành Vi & Nhân Khẩu Học → Churn', fontsize=14, fontweight='bold')

metrics = [
    ('avg_watch_time_minutes', 'Thời gian xem TB (phút)'),
    ('engagement_rate',        'Engagement Rate'),
    ('days_since_last_login',  'Số ngày từ lần đăng nhập cuối'),
]

for ax, (col, label) in zip(axes, metrics):
    data_no  = df[df['churned'] == 'No'][col]
    data_yes = df[df['churned'] == 'Yes'][col]

    ax.hist(data_no,  bins=40, alpha=0.6, color=CLR_RETAIN, label='Giữ lại', density=True)
    ax.hist(data_yes, bins=40, alpha=0.6, color=CLR_CHURN,  label='Rời bỏ',  density=True)

    ax.axvline(data_no.median(),  color=CLR_RETAIN, ls='--', lw=1.5)
    ax.axvline(data_yes.median(), color=CLR_CHURN,  ls='--', lw=1.5)

    ax.set_title(label, fontsize=11)
    ax.set_ylabel('Mật độ', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig('fig7_behavior_churn.png', bbox_inches='tight', dpi=130)
plt.show()
print('✅ Hình 7 đã lưu.')

cols_corr = [
    'monthly_fee', 'account_age_months', 'total_revenue_to_date',
    'avg_watch_time_minutes', 'engagement_rate', 'days_since_last_login',
    'completion_rate', 'binge_watch_sessions', 'churned_bin'
]
labels_corr = [
    'Monthly Fee', 'Account Age', 'CLV ước tính',
    'Watch Time TB', 'Engagement Rate', 'Days Since Login',
    'Completion Rate', 'Binge Sessions', 'Churned'
]

corr = df[cols_corr].corr()
corr.index = labels_corr
corr.columns = labels_corr

mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#F8F9FA')

sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='coolwarm', center=0,
    square=True, linewidths=0.5, linecolor='white',
    cbar_kws={'shrink': 0.8},
    annot_kws={'size': 9},
    ax=ax
)

ax.set_title('🔗 Ma Trận Tương Quan – Các Biến Liên Quan đến CLV & Churn',
             fontsize=13, fontweight='bold', pad=15)
ax.tick_params(axis='x', rotation=30, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)

plt.tight_layout()
plt.savefig('fig8_correlation.png', bbox_inches='tight', dpi=130)
plt.show()
print('✅ Hình 8 đã lưu.')

# Tính các KPI tổng hợp
overall_churn   = df['churned_bin'].mean()
avg_clv         = df['total_revenue_to_date'].mean()
avg_clv_churn   = df[df['churned']=='Yes']['total_revenue_to_date'].mean()
avg_clv_retain  = df[df['churned']=='No']['total_revenue_to_date'].mean()
clv_loss        = avg_clv_retain - avg_clv_churn

# Country với churn cao nhất & thấp nhất
country_churn = df.groupby('country')['churned_bin'].mean()
top_churn_country = country_churn.idxmax()
low_churn_country = country_churn.idxmin()

# Fee với churn thấp nhất
fee_churn = df.groupby('monthly_fee')['churned_bin'].mean()
best_fee   = fee_churn.idxmin()

print('=' * 60)
print('          📋 TÓM TẮT KẾT QUẢ EDA CLV')
print('=' * 60)
print(f'📦 Tổng khách hàng        : {len(df):,}')
print(f'📉 Tỷ lệ Churn tổng thể   : {overall_churn:.1%}')
print(f'💰 CLV trung bình tổng thể : ${avg_clv:.1f}')
print(f'   └─ Khách giữ lại        : ${avg_clv_retain:.1f}')
print(f'   └─ Khách rời bỏ         : ${avg_clv_churn:.1f}')
print(f'   └─ Chênh lệch CLV       : ${clv_loss:.1f}')
print()
print(f'🌍 Quốc gia churn cao nhất : {top_churn_country} ({country_churn[top_churn_country]:.1%})')
print(f'🌍 Quốc gia churn thấp nhất: {low_churn_country} ({country_churn[low_churn_country]:.1%})')
print()
print(f'💳 Gói phí có churn thấp nhất: ${best_fee} ({fee_churn[best_fee]:.1%})')
print()
print('🔍 Xu hướng nổi bật theo quốc gia:')
print('   ✅ Australia, India: gói rẻ hơn → churn cao hơn (nhạy cảm giá)')
print('   ⚡ Japan, France, UK: gói đắt hơn → churn cao hơn (khác biệt về nhu cầu)')
print('   ➡️  USA, Germany: churn tương đối đồng đều theo mức giá')
print('=' * 60)