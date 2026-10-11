
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
def build_xgb():
    return XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, eval_metric='logloss'
    )

def build_lr():
    return LogisticRegression(
        penalty='l2',              # 想稀疏化可改 'l1' + solver='liblinear'
        C=1.0,
        solver='lbfgs',
        max_iter=2000,
        random_state=42,
        n_jobs=-1
    )

# ==================== 5. 重复交叉验证收集 OOF 概率 ====================
N_SPLITS, N_REPEATS = 5, 5
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

oof_xgb = np.zeros(len(y_train), dtype=float)
oof_lr  = np.zeros(len(y_train), dtype=float)
cnt     = np.zeros(len(y_train), dtype=float)

print(f"\n>>> 开始 RepeatedStratifiedKFold（{N_SPLITS}折 × {N_REPEATS}次 = {N_SPLITS*N_REPEATS} 次训练）")

for i, (tr_idx, va_idx) in enumerate(rskf.split(X_train, y_train), 1):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    # 只对训练折做 SMOTETomek，验证折保持原始分布
    X_tr_res, y_tr_res = SMOTETomek(random_state=42).fit_resample(X_tr, y_tr)

    # XGBoost
    m_x = build_xgb(); m_x.fit(X_tr_res, y_tr_res)
    oof_xgb[va_idx] += m_x.predict_proba(X_va)[:, 1]

    # Logistic Regression
    # SMOTETomek 已经平衡，不再加 class_weight，避免双重平衡
    m_r = build_lr(); m_r.fit(X_tr_res, y_tr_res)
    oof_lr[va_idx] += m_r.predict_proba(X_va)[:, 1]

    cnt[va_idx] += 1

    if i % 5 == 0:
        print(f"  已完成 {i}/{N_SPLITS*N_REPEATS} 次")

oof_xgb /= cnt
oof_lr  /= cnt

print(f"\nOOF 完成。")
print(f"  XGB OOF PR-AUC = {average_precision_score(y_train, oof_xgb):.4f}")
print(f"  LR  OOF PR-AUC = {average_precision_score(y_train, oof_lr):.4f}")

# ==================== 6. 在 OOF 上搜权重 + 阈值（Rank 归一化融合）====================
# 6.1 OOF rank（0~1）
oof_xgb_rank = rankdata(oof_xgb) / len(oof_xgb)
oof_lr_rank  = rankdata(oof_lr)  / len(oof_lr)

# 6.2 权重搜索（rank 空间，限制在 [0.3, 0.7]）
weight_grid = np.arange(0.30, 0.701, 0.05)
scores = [(w, average_precision_score(y_train, w * oof_xgb_rank + (1 - w) * oof_lr_rank))
          for w in weight_grid]
scores.sort(key=lambda x: -x[1])

print("\nRank 空间权重扫描（按 OOF PR-AUC 降序，前 5 名）：")
for w, s in scores[:5]:
    print(f"  w_xgb={w:.2f}, w_lr={1-w:.2f} -> PR-AUC={s:.4f}")

w_xgb_best = float(np.mean([w for w, _ in scores[:3]]))
w_lr_best  = 1 - w_xgb_best
print(f"\n>>> top-3 平滑后：w_xgb={w_xgb_best:.2f}, w_lr={w_lr_best:.2f}")

p_oof_rank = w_xgb_best * oof_xgb_rank + w_lr_best * oof_lr_rank

# 6.3 阈值搜索
thr_grid = np.arange(0.10, 0.91, 0.02)

def find_best_threshold(y_true, y_prob, grid=thr_grid):
    best_thr, best_f1 = 0.5, -1
    for thr in grid:
        f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)

thr_xgb, f1_xgb_oof = find_best_threshold(y_train, oof_xgb)
thr_lr,  f1_lr_oof  = find_best_threshold(y_train, oof_lr)
thr_ens, f1_ens_oof = find_best_threshold(y_train, p_oof_rank)

print("\n>>> OOF 上选出的最优阈值：")
print(f"  XGB  : thr={thr_xgb:.2f}, OOF F1={f1_xgb_oof:.4f}")
print(f"  LR   : thr={thr_lr:.2f}, OOF F1={f1_lr_oof:.4f}")
print(f"  融合 : thr={thr_ens:.2f}, OOF F1={f1_ens_oof:.4f}（rank 空间）")

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

# 6.5 构建 (原始概率 -> rank) 映射，供测试集使用
f_xgb = interp1d(np.sort(oof_xgb), np.linspace(0, 1, len(oof_xgb)),
                 bounds_error=False, fill_value=(0, 1))
f_lr  = interp1d(np.sort(oof_lr),  np.linspace(0, 1, len(oof_lr)),
                 bounds_error=False, fill_value=(0, 1))

# 可选：保存 OOF 概率
# np.savez("./oof_probs_xgb_lr.npz",
#          oof_xgb=oof_xgb, oof_lr=oof_lr, p_oof_rank=p_oof_rank, y_train=y_train)

# ==================== 7. 全训练集重训 + 测试集评估 ====================
print("\n>>> 用全训练集 + SMOTETomek 重新训练最终模型")
X_train_res, y_train_res = SMOTETomek(random_state=42).fit_resample(X_train, y_train)
print("SMOTETomek 后训练集分布：")
print(pd.Series(y_train_res).value_counts())

# ---- 单模型 ----
xgb_final = build_xgb(); xgb_final.fit(X_train_res, y_train_res)
proba_xgb = xgb_final.predict_proba(X_test)[:, 1]
evaluate(f"SMOTETomek + XGBoost (thr={thr_xgb:.2f})",
         y_test, (proba_xgb >= thr_xgb).astype(int), proba_xgb)

lr_final = build_lr(); lr_final.fit(X_train_res, y_train_res)
proba_lr = lr_final.predict_proba(X_test)[:, 1]
evaluate(f"SMOTETomek + LogisticRegression (thr={thr_lr:.2f})",
         y_test, (proba_lr >= thr_lr).astype(int), proba_lr)

# ---- Rank 融合 ----
proba_xgb_rank = f_xgb(proba_xgb)
proba_lr_rank  = f_lr(proba_lr)
proba_ens_rank = w_xgb_best * proba_xgb_rank + w_lr_best * proba_lr_rank

evaluate(
    f"SMOTETomek + Rank加权融合 XGB+LR (XGB={w_xgb_best:.2f}, LR={w_lr_best:.2f}, thr={thr_ens:.2f})",
    y_test,
    (proba_ens_rank >= thr_ens).astype(int),
    proba_ens_rank
)

# ==================== 8. 汇总 ====================
print("\n\n" + "=" * 62)
print("SMOTETomek + XGB / LR / Rank加权融合 汇总（按 PR-AUC 降序）")
print("=" * 62)

summary = pd.DataFrame(results).T.sort_values('PR-AUC', ascending=False)
print(summary.round(4))

# summary.to_csv("./smotetomek_xgb_lr_rank_fusion_results.csv", encoding="utf-8-sig")
# print("\n结果已保存到 ./smotetomek_xgb_lr_rank_fusion_results.csv")