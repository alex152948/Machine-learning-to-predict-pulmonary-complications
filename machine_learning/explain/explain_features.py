

import sys
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from process_data.dm05_get_data import prepare_data

# ==================== 1. 数据准备 ====================
selected_features = [
    '切除肺叶_5', '胸膜黏连', 'APTT', 'FEV1实/预', 'FEV1/FVC预计值',
    'DLCO实/预', 'Smoking', 'CCI', 'bleeding', 'surgerytime',
    '袖切类型1（1-简单袖切，2=扩大袖切，3=袖+部分叶）',
    'FVC实测值', 'FVC实/预', 'FEV1/FVC实测值', 'DLCO实测值'
]

X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
    file_path="./data/肺科多模态数据-处理后.xlsx",
    sheet_name="Sheet1",
    selected_features=selected_features,
    use_lobe_features=True,
    test_size=0.2,
    random_state=42
)

X_train = np.asarray(X_train, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)
y_train = np.asarray(y_train).astype(int).ravel()
y_test = np.asarray(y_test).astype(int).ravel()

print("原始训练集标签分布：")
print(pd.Series(y_train).value_counts())
print("测试集标签分布：")
print(pd.Series(y_test).value_counts())

# ==================== 2. 依赖 ====================
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTETomek
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)
from scipy.stats import rankdata
from scipy.interpolate import interp1d

# ==================== 3. 统一评估函数 ====================
results = {}

def evaluate(name, y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        auc, pr_auc = np.nan, np.nan

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 62}")
    print(f"模型：{name}")
    print(f"{'=' * 62}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"MCC      : {mcc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print("混淆矩阵：")
    print(cm)

    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1': f1, 'MCC': mcc, 'ROC-AUC': auc, 'PR-AUC': pr_auc
    }

# ==================== 4. 模型工厂 ====================
def build_lgb():
    return LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=-1, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )

def build_xgb():
    return XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric='logloss',
        base_score=0.5
    )

def build_lr():
    return LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs', max_iter=2000,
        random_state=42, n_jobs=-1
    )

# ==================== 5. 重复交叉验证收集 OOF 概率 ====================
N_SPLITS, N_REPEATS = 5, 5
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

oof_lgb = np.zeros(len(y_train), dtype=float)
oof_xgb = np.zeros(len(y_train), dtype=float)
oof_lr  = np.zeros(len(y_train), dtype=float)
cnt     = np.zeros(len(y_train), dtype=float)

print(f"\n>>> 开始 RepeatedStratifiedKFold（{N_SPLITS}折 × {N_REPEATS}次 = {N_SPLITS*N_REPEATS} 次训练，每次 3 个模型）")

for i, (tr_idx, va_idx) in enumerate(rskf.split(X_train, y_train), 1):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    # 只对训练折做 SMOTETomek，验证折保持原始分布
    X_tr_res, y_tr_res = SMOTETomek(random_state=42).fit_resample(X_tr, y_tr)

    m_l = build_lgb(); m_l.fit(X_tr_res, y_tr_res)
    oof_lgb[va_idx] += m_l.predict_proba(X_va)[:, 1]

    m_x = build_xgb(); m_x.fit(X_tr_res, y_tr_res)
    oof_xgb[va_idx] += m_x.predict_proba(X_va)[:, 1]

    m_r = build_lr(); m_r.fit(X_tr_res, y_tr_res)
    oof_lr[va_idx] += m_r.predict_proba(X_va)[:, 1]

    cnt[va_idx] += 1

    if i % 5 == 0:
        print(f"  已完成 {i}/{N_SPLITS*N_REPEATS} 次")

oof_lgb /= cnt
oof_xgb /= cnt
oof_lr  /= cnt

print(f"\nOOF 完成。")
print(f"  LGB OOF PR-AUC = {average_precision_score(y_train, oof_lgb):.4f}")
print(f"  XGB OOF PR-AUC = {average_precision_score(y_train, oof_xgb):.4f}")
print(f"  LR  OOF PR-AUC = {average_precision_score(y_train, oof_lr):.4f}")

# ==================== 6. 在 OOF 上搜权重 + 阈值（三模型 Rank 融合）====================
# 6.1 OOF rank（0~1）
oof_lgb_rank = rankdata(oof_lgb) / len(oof_lgb)
oof_xgb_rank = rankdata(oof_xgb) / len(oof_xgb)
oof_lr_rank  = rankdata(oof_lr)  / len(oof_lr)

# 6.2 三维权重网格搜索
# 步长 0.05，三个权重和=1，每个限制在 [0.10, 0.70] 避免极端
step = 0.05
w_range = np.arange(0.10, 0.701, step)
candidates = []
for w_l in w_range:
    for w_x in w_range:
        w_r = 1 - w_l - w_x
        if 0.10 <= w_r <= 0.70:
            candidates.append((round(w_l, 2), round(w_x, 2), round(w_r, 2)))

print(f"\n有效权重组合数：{len(candidates)}")

scores = []
for w_l, w_x, w_r in candidates:
    p = w_l * oof_lgb_rank + w_x * oof_xgb_rank + w_r * oof_lr_rank
    pr = average_precision_score(y_train, p)
    scores.append(((w_l, w_x, w_r), pr))

scores.sort(key=lambda x: -x[1])

print("\n三模型权重扫描（按 OOF PR-AUC 降序，前 10 名）：")
for (w_l, w_x, w_r), pr in scores[:10]:
    print(f"  w_lgb={w_l:.2f}, w_xgb={w_x:.2f}, w_lr={w_r:.2f} -> PR-AUC={pr:.4f}")

# top-5 平滑（对三个权重分别取平均，然后重新归一化）
top_k = 5
top_weights = [w for w, _ in scores[:top_k]]
w_lgb_best = float(np.mean([w[0] for w in top_weights]))
w_xgb_best = float(np.mean([w[1] for w in top_weights]))
w_lr_best  = float(np.mean([w[2] for w in top_weights]))
# 重新归一化，保证和为 1
s = w_lgb_best + w_xgb_best + w_lr_best
w_lgb_best /= s
w_xgb_best /= s
w_lr_best  /= s

print(f"\n>>> top-{top_k} 平滑 + 归一化后：")
print(f"  w_lgb={w_lgb_best:.3f}, w_xgb={w_xgb_best:.3f}, w_lr={w_lr_best:.3f}")

p_oof_rank = w_lgb_best * oof_lgb_rank + w_xgb_best * oof_xgb_rank + w_lr_best * oof_lr_rank
print(f"  OOF PR-AUC = {average_precision_score(y_train, p_oof_rank):.4f}")

# 6.3 阈值搜索
thr_grid = np.arange(0.10, 0.91, 0.02)

def find_best_threshold(y_true, y_prob, grid=thr_grid):
    best_thr, best_f1 = 0.5, -1
    for thr in grid:
        f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)

thr_lgb, _ = find_best_threshold(y_train, oof_lgb)
thr_xgb, _ = find_best_threshold(y_train, oof_xgb)
thr_lr,  _ = find_best_threshold(y_train, oof_lr)
thr_ens, f1_ens_oof = find_best_threshold(y_train, p_oof_rank)

print(f"\n>>> OOF 最优阈值：")
print(f"  LGB={thr_lgb:.2f}, XGB={thr_xgb:.2f}, LR={thr_lr:.2f}, 融合={thr_ens:.2f}")

# 6.4 打印融合阈值扫描表
print("\n融合阈值扫描（OOF，rank 空间）：")
scan_rows = []
for thr in thr_grid:
    pred = (p_oof_rank >= thr).astype(int)
    scan_rows.append({
        'thr': round(thr, 2),
        'Precision': precision_score(y_train, pred, zero_division=0),
        'Recall':    recall_score(y_train, pred, zero_division=0),
        'F1':        f1_score(y_train, pred, zero_division=0),
        'MCC':       matthews_corrcoef(y_train, pred),
    })
print(pd.DataFrame(scan_rows).round(4).to_string(index=False))

# 6.5 构建 (原始概率 -> rank) 映射
f_lgb = interp1d(np.sort(oof_lgb), np.linspace(0, 1, len(oof_lgb)),
                 bounds_error=False, fill_value=(0, 1))
f_xgb = interp1d(np.sort(oof_xgb), np.linspace(0, 1, len(oof_xgb)),
                 bounds_error=False, fill_value=(0, 1))
f_lr  = interp1d(np.sort(oof_lr),  np.linspace(0, 1, len(oof_lr)),
                 bounds_error=False, fill_value=(0, 1))

# ==================== 7. 全训练集重训 + 测试集评估 ====================
print("\n>>> 用全训练集 + SMOTETomek 重新训练最终模型")
X_train_res, y_train_res = SMOTETomek(random_state=42).fit_resample(X_train, y_train)
print("SMOTETomek 后训练集分布：")
print(pd.Series(y_train_res).value_counts())

# ---- 单模型 ----
lgb_final = build_lgb(); lgb_final.fit(X_train_res, y_train_res)
proba_lgb = lgb_final.predict_proba(X_test)[:, 1]
evaluate(f"SMOTETomek + LightGBM (thr={thr_lgb:.2f})",
         y_test, (proba_lgb >= thr_lgb).astype(int), proba_lgb)

xgb_final = build_xgb(); xgb_final.fit(X_train_res, y_train_res)
proba_xgb = xgb_final.predict_proba(X_test)[:, 1]
evaluate(f"SMOTETomek + XGBoost (thr={thr_xgb:.2f})",
         y_test, (proba_xgb >= thr_xgb).astype(int), proba_xgb)

lr_final = build_lr(); lr_final.fit(X_train_res, y_train_res)
proba_lr = lr_final.predict_proba(X_test)[:, 1]
evaluate(f"SMOTETomek + LogisticRegression (thr={thr_lr:.2f})",
         y_test, (proba_lr >= thr_lr).astype(int), proba_lr)

# ---- 三模型 Rank 融合 ----
proba_lgb_rank = f_lgb(proba_lgb)
proba_xgb_rank = f_xgb(proba_xgb)
proba_lr_rank  = f_lr(proba_lr)

proba_ens_rank = (w_lgb_best * proba_lgb_rank +
                  w_xgb_best * proba_xgb_rank +
                  w_lr_best  * proba_lr_rank)

evaluate(
    f"SMOTETomek + Rank融合 LGB+XGB+LR (LGB={w_lgb_best:.2f}, XGB={w_xgb_best:.2f}, LR={w_lr_best:.2f}, thr={thr_ens:.2f})",
    y_test,
    (proba_ens_rank >= thr_ens).astype(int),
    proba_ens_rank
)

# ==================== 8. 汇总 ====================
print("\n\n" + "=" * 62)
print("SMOTETomek + LGB / XGB / LR / Rank加权融合 汇总（按 PR-AUC 降序）")
print("=" * 62)

summary = pd.DataFrame(results).T.sort_values('PR-AUC', ascending=False)
print(summary.round(4))

# summary.to_csv("./smotetomek_lgb_xgb_lr_rank_fusion_results.csv", encoding="utf-8-sig")
# print("\n结果已保存到 ./smotetomek_lgb_xgb_lr_rank_fusion_results.csv")
# ==================== 9. 可解释性分析 ====================
import shap
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 统一输出目录
FIG_DIR = "./figures"
RES_DIR = "./results_explain"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# 用于 SHAP 背景：用原始训练集（不要用 SMOTETomek 后的合成集）
X_bg = X_train.copy()
X_exp = X_test.copy()

# 特征名转成 list，防止 numpy array
feature_names = list(feature_names)


# ---------------------------------------------------------
# 9.1 TreeSHAP 解释 LightGBM（只保留 summary）
# ---------------------------------------------------------
print("\n" + "=" * 62)
print("9.1 TreeSHAP - LightGBM")
print("=" * 62)

explainer_lgb = shap.TreeExplainer(lgb_final)
sv_lgb = explainer_lgb.shap_values(X_exp)

if isinstance(sv_lgb, list):
    sv_lgb_pos = sv_lgb[1]
else:
    sv_lgb_pos = sv_lgb

plt.figure()
shap.summary_plot(sv_lgb_pos, X_exp, feature_names=feature_names,
                  show=False, max_display=15)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_shap_lgb_summary.png", dpi=300, bbox_inches='tight')
plt.close()

lgb_imp = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': np.abs(sv_lgb_pos).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

lgb_imp.to_csv(f"{RES_DIR}/shap_importance_lgb.csv",
               index=False, encoding='utf-8-sig')
print(lgb_imp.round(4).to_string(index=False))


# ---------------------------------------------------------
# 9.2 TreeSHAP 解释 XGBoost（只保留 summary）
# ---------------------------------------------------------
print("\n" + "=" * 62)
print("9.2 TreeSHAP - XGBoost")
print("=" * 62)

explainer_xgb = shap.TreeExplainer(xgb_final.get_booster())
sv_xgb = explainer_xgb.shap_values(X_exp)

if isinstance(sv_xgb, list):
    sv_xgb_pos = sv_xgb[1]
else:
    sv_xgb_pos = sv_xgb

plt.figure()
shap.summary_plot(sv_xgb_pos, X_exp, feature_names=feature_names,
                  show=False, max_display=15)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_shap_xgb_summary.png", dpi=300, bbox_inches='tight')
plt.close()

xgb_imp = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': np.abs(sv_xgb_pos).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

xgb_imp.to_csv(f"{RES_DIR}/shap_importance_xgb.csv",
               index=False, encoding='utf-8-sig')
print(xgb_imp.round(4).to_string(index=False))


# ---------------------------------------------------------
# 9.3 SHAP dependence（只保留 Top 3 特征）
# ---------------------------------------------------------
print("\n" + "=" * 62)
print("9.3 SHAP dependence - Top 3 特征")
print("=" * 62)

merged_imp = lgb_imp.merge(xgb_imp, on='feature', suffixes=('_lgb', '_xgb'))
merged_imp['mean_shap'] = (merged_imp['mean_abs_shap_lgb'] +
                           merged_imp['mean_abs_shap_xgb']) / 2
top_features = merged_imp.sort_values('mean_shap', ascending=False)\
                         .head(3)['feature'].tolist()
print("Top 3 特征：", top_features)

for rank, feat in enumerate(top_features, 1):
    idx = feature_names.index(feat)
    plt.figure()
    shap.dependence_plot(idx, sv_lgb_pos, X_exp,
                         feature_names=feature_names, show=False)
    plt.tight_layout()
    safe = feat.replace('/', '_').replace('\\', '_').replace(' ', '_')
    plt.savefig(f"{FIG_DIR}/fig_shap_dep_top{rank}_{safe}.png",
                dpi=300, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------
# 9.4 单样本 SHAP waterfall（只保留 TP 和 TN）
# ---------------------------------------------------------
print("\n" + "=" * 62)
print("9.4 单样本 SHAP waterfall（TP / TN）")
print("=" * 62)

proba_test_lgb = lgb_final.predict_proba(X_exp)[:, 1]
pred_test_lgb = (proba_test_lgb >= thr_lgb).astype(int)

case_tp = np.where((y_test == 1) & (pred_test_lgb == 1))[0]
case_tn = np.where((y_test == 0) & (pred_test_lgb == 0))[0]

def pick_top(cands, proba, mode='high'):
    if len(cands) == 0:
        return None
    if mode == 'high':
        return cands[np.argmax(proba[cands])]
    else:
        return cands[np.argmin(proba[cands])]

cases = {
    'TP_high': pick_top(case_tp, proba_test_lgb, 'high'),
    'TN_low':  pick_top(case_tn, proba_test_lgb, 'low'),
}

expected_value_lgb = explainer_lgb.expected_value
if isinstance(expected_value_lgb, (list, np.ndarray)):
    expected_value_lgb = float(np.ravel(expected_value_lgb)[-1])

for tag, idx in cases.items():
    if idx is None:
        print(f"  {tag}: 无样本，跳过")
        continue
    plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        expected_value_lgb,
        sv_lgb_pos[idx],
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_shap_waterfall_{tag}.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  {tag}: 样本索引={idx}, LGB 预测概率={proba_test_lgb[idx]:.4f}")


# ---------------------------------------------------------
# 9.5 Permutation Importance 解释融合模型
# ---------------------------------------------------------
print("\n" + "=" * 62)
print("9.5 Permutation Importance - 融合模型")
print("=" * 62)

def ensemble_predict_proba(X):
    X = np.asarray(X, dtype=np.float32)
    p_l = lgb_final.predict_proba(X)[:, 1]
    p_x = xgb_final.predict_proba(X)[:, 1]
    p_r = lr_final.predict_proba(X)[:, 1]
    r_l = f_lgb(p_l)
    r_x = f_xgb(p_x)
    r_r = f_lr(p_r)
    return (w_lgb_best * r_l +
            w_xgb_best * r_x +
            w_lr_best  * r_r)

base_pr  = average_precision_score(y_test, ensemble_predict_proba(X_exp))
base_auc = roc_auc_score(y_test, ensemble_predict_proba(X_exp))
base_f1  = f1_score(y_test,
                    (ensemble_predict_proba(X_exp) >= thr_ens).astype(int),
                    zero_division=0)

print(f"融合模型基线：PR-AUC={base_pr:.4f}, ROC-AUC={base_auc:.4f}, F1={base_f1:.4f}")

N_REPEATS_PI = 30
imp_pr  = np.zeros((N_REPEATS_PI, len(feature_names)))
imp_auc = np.zeros((N_REPEATS_PI, len(feature_names)))
imp_f1  = np.zeros((N_REPEATS_PI, len(feature_names)))

for r in range(N_REPEATS_PI):
    rng = np.random.RandomState(42 + r)
    for j in range(len(feature_names)):
        X_perm = X_exp.copy()
        X_perm[:, j] = rng.permutation(X_perm[:, j])
        p_perm = ensemble_predict_proba(X_perm)
        imp_pr[r, j]  = base_pr  - average_precision_score(y_test, p_perm)
        imp_auc[r, j] = base_auc - roc_auc_score(y_test, p_perm)
        imp_f1[r, j]  = base_f1  - f1_score(
            y_test, (p_perm >= thr_ens).astype(int), zero_division=0
        )

pi_df = pd.DataFrame({
    'feature': feature_names,
    'PR_AUC_drop_mean': imp_pr.mean(axis=0),
    'PR_AUC_drop_std':  imp_pr.std(axis=0),
    'ROC_AUC_drop_mean': imp_auc.mean(axis=0),
    'ROC_AUC_drop_std':  imp_auc.std(axis=0),
    'F1_drop_mean': imp_f1.mean(axis=0),
    'F1_drop_std':  imp_f1.std(axis=0),
}).sort_values('PR_AUC_drop_mean', ascending=False).reset_index(drop=True)

pi_df.to_csv(f"{RES_DIR}/permutation_importance_ensemble.csv",
             index=False, encoding='utf-8-sig')
print(pi_df.round(4).to_string(index=False))

plt.figure(figsize=(8, 6))
plot_df = pi_df.sort_values('PR_AUC_drop_mean', ascending=True)
plt.barh(plot_df['feature'], plot_df['PR_AUC_drop_mean'],
         xerr=plot_df['PR_AUC_drop_std'], color='steelblue')
plt.xlabel('Drop in PR-AUC when feature is permuted')
plt.title('Permutation Importance (Ensemble, PR-AUC)')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_permutation_importance_ensemble.png",
            dpi=300, bbox_inches='tight')
plt.close()


# ---------------------------------------------------------
# 9.6 Logistic Regression 系数 / OR（保留图和表）
# ---------------------------------------------------------
print("\n" + "=" * 62)
print("9.6 Logistic Regression 系数 / OR")
print("=" * 62)

coef = lr_final.coef_.ravel()

lr_df = pd.DataFrame({
    'feature': feature_names,
    'coef': coef,
    'OR': np.exp(coef),
    'abs_coef': np.abs(coef)
}).sort_values('abs_coef', ascending=False).reset_index(drop=True)

lr_df.to_csv(f"{RES_DIR}/lr_coefficients.csv",
             index=False, encoding='utf-8-sig')
print(lr_df.drop(columns=['abs_coef']).round(4).to_string(index=False))

plt.figure(figsize=(8, 6))
plot_lr = lr_df.sort_values('coef', ascending=True)
colors = ['crimson' if v > 0 else 'steelblue' for v in plot_lr['coef']]
plt.barh(plot_lr['feature'], plot_lr['coef'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('Standardized coefficient')
plt.title('Logistic Regression Coefficients')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_lr_coefficients.png",
            dpi=300, bbox_inches='tight')
plt.close()


# ---------------------------------------------------------
# 9.7 三种方法重要性对比表
# ---------------------------------------------------------
print("\n" + "=" * 62)
print("9.7 三种方法重要性对比（按平均排名）")
print("=" * 62)

lgb_rank = lgb_imp.copy()
lgb_rank['rank_lgb'] = np.arange(1, len(lgb_rank) + 1)

xgb_rank = xgb_imp.copy()
xgb_rank['rank_xgb'] = np.arange(1, len(xgb_rank) + 1)

pi_rank = pi_df.copy()
pi_rank['rank_pi'] = np.arange(1, len(pi_rank) + 1)

lr_rank = lr_df.copy()
lr_rank['rank_lr'] = np.arange(1, len(lr_rank) + 1)

compare = lgb_rank[['feature', 'mean_abs_shap', 'rank_lgb']]\
    .merge(xgb_rank[['feature', 'mean_abs_shap', 'rank_xgb']],
           on='feature', suffixes=('_lgb', '_xgb'))\
    .merge(pi_rank[['feature', 'PR_AUC_drop_mean', 'rank_pi']], on='feature')\
    .merge(lr_rank[['feature', 'coef', 'OR', 'rank_lr']], on='feature')

compare['rank_mean'] = compare[['rank_lgb', 'rank_xgb',
                                'rank_pi', 'rank_lr']].mean(axis=1)
compare = compare.sort_values('rank_mean').reset_index(drop=True)

compare.to_csv(f"{RES_DIR}/feature_importance_compare.csv",
               index=False, encoding='utf-8-sig')
print(compare.round(4).to_string(index=False))

print("\n可解释性分析完成。")
print("图保存到：", FIG_DIR)
print("结果表保存到：", RES_DIR)