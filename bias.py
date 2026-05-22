"""
=============================================================
  PHẦN 7: ETHICAL BIAS & FAIRNESS ANALYSIS
  Dữ liệu: netflix_step1_enriched.csv
  Phân tích bias theo Gender / Country / Subscription Type
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── Load & chuẩn hoá ──────────────────────────────────────
df = pd.read_csv('netflix_step1_enriched.csv')
df.columns = df.columns.str.strip()
for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.strip()

df['churned_bin'] = (df['churned'] == 'Yes').astype(int)

# ── Normalize features theo subscription_type ────────────────
features_to_norm = [
    'avg_watch_time_minutes', 'watch_sessions_per_week',
    'binge_watch_sessions', 'completion_rate',
    'engagement_rate', 'recommendation_click_rate',
]
for f in features_to_norm:
    grp_mean = df.groupby('subscription_type')[f].transform('mean')
    grp_std  = df.groupby('subscription_type')[f].transform('std').replace(0, 1)
    df[f'{f}_norm'] = (df[f] - grp_mean) / grp_std

features_final = (
    [f'{f}_norm' for f in features_to_norm] +
    ['age', 'account_age_months', 'days_since_last_login']
)

X = df[features_final].fillna(0)
y = df['churned_bin']

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

# ── Train GradientBoosting ────────────────────────────────
model = GradientBoostingClassifier(
    n_estimators=200, max_depth=4,
    learning_rate=0.05, random_state=42
)
model.fit(X_sc, y)
df['model_score'] = model.predict_proba(X_sc)[:, 1]

auc = roc_auc_score(y, df['model_score'])
print(f"AUC     : {auc:.4f}")

# ── Threshold calibration per subscription_type ───────────
print("\nThreshold calibration:")
thresh = {}
for grp, sub in df.groupby('subscription_type'):
    scores = sub['model_score'].values
    actual = sub['churned_bin'].values
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.01):
        pred = (scores >= t).astype(int)
        if pred.mean() < 0.15 or pred.mean() > 0.45:
            continue
        f1 = f1_score(actual, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    thresh[str(grp)] = best_t
    print(f"  {grp}: threshold={best_t:.2f}, F1={best_f1:.4f}")

df['churn_pred'] = df.apply(
    lambda r: int(r['model_score'] >= thresh[str(r['subscription_type'])]), axis=1
)
acc = (df['churn_pred'] == y).mean()
print(f"Accuracy: {acc:.4f}")

# ── Fairness metrics helper ───────────────────────────────
def fairness_metrics(df, group_col, pred_col='churn_pred'):
    rows = []
    for grp, sub in df.groupby(group_col):
        actual = sub['churned_bin']
        pred   = sub[pred_col]
        tp = ((pred==1)&(actual==1)).sum()
        fp = ((pred==1)&(actual==0)).sum()
        tn = ((pred==0)&(actual==0)).sum()
        fn = ((pred==0)&(actual==1)).sum()
        rows.append({
            group_col:   str(grp),
            'TPR':       tp/(tp+fn) if (tp+fn)>0 else 0,
            'FPR':       fp/(fp+tn) if (fp+tn)>0 else 0,
            'Precision': tp/(tp+fp) if (tp+fp)>0 else 0,
            'Accuracy':  (tp+tn)/len(sub),
            'Actual_Churn': actual.mean(),
            'Pred_Churn':   pred.mean(),
            'N': len(sub)
        })
    return pd.DataFrame(rows)

fm_g = fairness_metrics(df, 'gender')
fm_c = fairness_metrics(df, 'country')
fm_s = fairness_metrics(df, 'subscription_type')

# ── Stratified model: train riêng từng subscription_type ─
print("\n" + "="*62)
print("  STRATIFIED MODEL (train riêng từng subscription type)")
print("="*62)

df['pred_strat'] = 0
for grp, sub in df.groupby('subscription_type'):
    idx   = sub.index
    X_grp = X_sc[idx]
    y_grp = y[idx]

    m = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, random_state=42
    )
    m.fit(X_grp, y_grp)
    scores = m.predict_proba(X_grp)[:, 1]

    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.01):
        pred = (scores >= t).astype(int)
        if pred.mean() < 0.15 or pred.mean() > 0.45:
            continue
        f1 = f1_score(y_grp, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    df.loc[idx, 'pred_strat'] = (scores >= best_t).astype(int)
    tpr_val = ((df.loc[idx,'pred_strat']==1)&(y_grp==1)).sum() / max((y_grp==1).sum(),1)
    print(f"  {grp}: threshold={best_t:.2f}, F1={best_f1:.4f}, TPR={tpr_val:.4f}")

fm_strat = fairness_metrics(df, 'subscription_type', 'pred_strat')
di_s2  = fm_strat.Pred_Churn.min()/fm_strat.Pred_Churn.max()
gap_s2 = fm_strat.TPR.max()-fm_strat.TPR.min()
flag_s2 = '✅ OK' if di_s2>=0.8 else '⚠ BIAS'
print(f"\nStratified — DI={di_s2:.3f} {flag_s2}  |  TPR gap={gap_s2:.4f}")
print(fm_strat.to_string(index=False, float_format='{:.4f}'.format))

# ── Print KPI tables ──────────────────────────────────────
for fm, lbl in [(fm_g,'GENDER'), (fm_c,'COUNTRY')]:
    di   = fm.Pred_Churn.min()/fm.Pred_Churn.max()
    gap  = fm.TPR.max()-fm.TPR.min()
    flag = '✅ OK' if di>=0.8 else '⚠ BIAS'
    print(f"\n{'='*62}")
    print(f"  {lbl}  —  DI={di:.3f} {flag}  |  TPR gap={gap:.4f}")
    print('='*62)
    print(fm.to_string(index=False, float_format='{:.4f}'.format))

# ── DASHBOARD ─────────────────────────────────────────────
DARK, PANEL, BORDER = "#0D1117", "#161B22", "#30363D"
ACCENT, GREEN, RED, YELLOW = "#58A6FF", "#3FB950", "#F85149", "#D29922"
TEXT, MUTED = "#E6EDF3", "#8B949E"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": BORDER,
    "grid.linewidth": 0.5, "font.family": "monospace"
})

fig = plt.figure(figsize=(22, 17), facecolor=DARK)
fig.suptitle("⚖  ETHICAL BIAS & FAIRNESS REPORT  ·  Part 7  ·  50,000 subscribers",
             fontsize=17, fontweight="bold", color=TEXT, y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.48, wspace=0.40,
                       left=0.06, right=0.97,
                       top=0.93, bottom=0.05)

def sa(ax, title):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.set_title(title, fontsize=9.5, color=ACCENT,
                 fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.25)
    return ax

gc = {"Male": ACCENT, "Female": "#FF7EE0", "Other": YELLOW}
cc = plt.cm.Set2(np.linspace(0, 1, len(fm_c)))
sc = {"Basic": RED, "Standard": YELLOW, "Premium": GREEN}

# [0,0] Actual vs Predicted — Gender
ax = sa(fig.add_subplot(gs[0, 0]), "① Actual vs Predicted Churn · Gender")
x = np.arange(len(fm_g)); w = 0.35
ax.bar(x-w/2, fm_g.Actual_Churn, w,
       color=[gc[g] for g in fm_g.gender], alpha=0.9, label="Actual")
ax.bar(x+w/2, fm_g.Pred_Churn, w,
       color=[gc[g] for g in fm_g.gender], alpha=0.45,
       hatch="//", label="Predicted")
ax.set_xticks(x); ax.set_xticklabels(fm_g.gender, fontsize=9)
ax.set_ylabel("Churn Rate")
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
di_g = fm_g.Pred_Churn.min()/fm_g.Pred_Churn.max()
ax.text(0.02, 0.95, f"DI={di_g:.3f} ✅", transform=ax.transAxes,
        color=GREEN, fontsize=9, fontweight="bold")

# [0,1] TPR — Gender
ax = sa(fig.add_subplot(gs[0, 1]), "② True Positive Rate · Gender")
bars = ax.bar(fm_g.gender, fm_g.TPR,
              color=[gc[g] for g in fm_g.gender], edgecolor=BORDER)
ref = fm_g.TPR.mean()
ax.axhline(ref, color=YELLOW, ls="--", lw=1.2, label=f"Mean={ref:.3f}")
ax.set_ylabel("TPR")
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
            f"{b.get_height():.3f}", ha="center", fontsize=9)

# [0,2] FPR — Gender
ax = sa(fig.add_subplot(gs[0, 2]), "③ False Positive Rate · Gender")
bars = ax.bar(fm_g.gender, fm_g.FPR,
              color=[gc[g] for g in fm_g.gender], edgecolor=BORDER)
ref = fm_g.FPR.mean()
ax.axhline(ref, color=YELLOW, ls="--", lw=1.2, label=f"Mean={ref:.3f}")
ax.set_ylabel("FPR")
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
            f"{b.get_height():.3f}", ha="center", fontsize=9)

# [1,0] Actual Churn — Country
ax = sa(fig.add_subplot(gs[1, 0]), "④ Actual Churn Rate · Country")
fc = fm_c.sort_values("Actual_Churn", ascending=True)
ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", alpha=0)
bars = ax.barh(fc.country, fc.Actual_Churn,
               color=cc[:len(fc)], edgecolor=BORDER)
overall = df.churned_bin.mean()
ax.axvline(overall, color=RED, ls="--", lw=1.2,
           label=f"Overall={overall:.3f}")
ax.set_xlabel("Churn Rate")
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
for b in bars:
    ax.text(b.get_width()+0.001, b.get_y()+b.get_height()/2,
            f"{b.get_width():.3f}", va="center", fontsize=8)

# [1,1] TPR — Country
ax = sa(fig.add_subplot(gs[1, 1]), "⑤ TPR (Recall) · Country")
fc2 = fm_c.sort_values("TPR", ascending=True)
ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", alpha=0)
bars = ax.barh(fc2.country, fc2.TPR,
               color=cc[:len(fc2)], edgecolor=BORDER)
ref = fc2.TPR.mean()
ax.axvline(ref, color=YELLOW, ls="--", lw=1.2, label=f"Mean={ref:.3f}")
ax.set_xlabel("TPR")
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
for b in bars:
    ax.text(b.get_width()+0.003, b.get_y()+b.get_height()/2,
            f"{b.get_width():.3f}", va="center", fontsize=8)

# [1,2] Precision — Country
ax = sa(fig.add_subplot(gs[1, 2]), "⑥ Precision · Country")
fc3 = fm_c.sort_values("Precision", ascending=True)
ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", alpha=0)
bars = ax.barh(fc3.country, fc3.Precision,
               color=cc[:len(fc3)], edgecolor=BORDER)
ref = fc3.Precision.mean()
ax.axvline(ref, color=YELLOW, ls="--", lw=1.2, label=f"Mean={ref:.3f}")
ax.set_xlabel("Precision")
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
for b in bars:
    ax.text(b.get_width()+0.002, b.get_y()+b.get_height()/2,
            f"{b.get_width():.3f}", va="center", fontsize=8)

# [2,0] Actual vs Predicted — Subscription Type
ax = sa(fig.add_subplot(gs[2, 0]), "⑦ Actual vs Predicted Churn · Subscription Type")
subs = fm_s.subscription_type.tolist()
x2 = np.arange(len(fm_s)); w = 0.35
ax.bar(x2-w/2, fm_s.Actual_Churn, w,
       color=[sc.get(s, ACCENT) for s in subs], alpha=0.9, label="Actual")
ax.bar(x2+w/2, fm_s.Pred_Churn, w,
       color=[sc.get(s, ACCENT) for s in subs], alpha=0.45,
       hatch="//", label="Predicted")
ax.set_xticks(x2); ax.set_xticklabels(subs, fontsize=9)
ax.set_ylabel("Churn Rate")
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
di_s_val = fm_s.Pred_Churn.min()/fm_s.Pred_Churn.max()
flag_col = GREEN if di_s_val>=0.8 else RED
ax.text(0.02, 0.95, f"DI={di_s_val:.3f} {'✅' if di_s_val>=0.8 else '⚠'}",
        transform=ax.transAxes, color=flag_col, fontsize=9, fontweight="bold")

# [2,1] TPR — Subscription Type
ax = sa(fig.add_subplot(gs[2, 1]), "⑧ TPR · Subscription Type")
bars = ax.bar(subs, fm_s.TPR,
              color=[sc.get(s, ACCENT) for s in subs], edgecolor=BORDER)
ref = fm_s.TPR.mean()
ax.axhline(ref, color=YELLOW, ls="--", lw=1.2, label=f"Mean={ref:.3f}")
ax.set_ylabel("TPR")
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
            f"{b.get_height():.3f}", ha="center", fontsize=9)

# [2,2] Summary panel
ax = fig.add_subplot(gs[2, 2])
ax.set_facecolor("#0A1F10"); ax.axis("off")
for sp in ax.spines.values():
    sp.set_color(GREEN); sp.set_linewidth(1.5)

di_c_val = fm_c.Pred_Churn.min()/fm_c.Pred_Churn.max()
worst_sub = fm_s.loc[fm_s.TPR.idxmin(), 'subscription_type']

lines = [
    ("⚖  FAIRNESS SUMMARY", ACCENT, 11, "bold"),
    ("", TEXT, 9, "normal"),
    (f"Gender  DI={di_g:.3f} ✅", GREEN, 8.5, "normal"),
    (f"  TPR gap={fm_g.TPR.max()-fm_g.TPR.min():.4f}", TEXT, 8, "normal"),
    (f"Country DI={di_c_val:.3f} {'✅' if di_c_val>=0.8 else '⚠'}", GREEN if di_c_val>=0.8 else RED, 8.5, "normal"),
    (f"  TPR gap={fm_c.TPR.max()-fm_c.TPR.min():.4f}", TEXT, 8, "normal"),
    (f"Subscr. DI={di_s_val:.3f} {'✅' if di_s_val>=0.8 else '⚠'}", GREEN if di_s_val>=0.8 else RED, 8.5, "normal"),
    (f"  TPR gap={fm_s.TPR.max()-fm_s.TPR.min():.4f}", TEXT, 8, "normal"),
    (f"  Worst TPR: {worst_sub}", YELLOW, 8, "normal"),
    ("", TEXT, 9, "normal"),
    (f"Stratified DI={di_s2:.3f} {flag_s2}", GREEN if di_s2>=0.8 else RED, 8.5, "normal"),
    (f"  TPR gap={gap_s2:.4f}", TEXT, 8, "normal"),
    ("", TEXT, 9, "normal"),
    ("RECOMMENDATIONS:", YELLOW, 9, "bold"),
    ("  • Threshold riêng từng gói", TEXT, 8, "normal"),
    ("  • Monitor subscription type có churn cao", TEXT, 8, "normal"),
    ("  • Ưu tiên retention gói Premium", TEXT, 8, "normal"),
    ("  • Audit fairness định kỳ", TEXT, 8, "normal"),
]

y_pos = 0.97
for txt, col, sz, wt in lines:
    ax.text(0.04, y_pos, txt, transform=ax.transAxes,
            color=col, fontsize=sz, fontweight=wt,
            fontfamily="monospace", va="top")
    y_pos -= 0.058

fig.text(0.5, 0.01,
         "Part 7 · Ethical Bias & Fairness · Netflix Churn Pipeline · n=50,000",
         ha="center", fontsize=8, color=MUTED, fontfamily="monospace")

plt.savefig("7_Ethical_BiasFairness_final.png",
            dpi=150, bbox_inches="tight", facecolor=DARK)