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
# dm03挑选的
# selected_features = ['切除肺叶_5', '胸膜黏连', 'APTT', 'FEV1实/预', 'FEV1/FVC预计值', 'DLCO实/预']
selected_features = ['切除肺叶_5', '胸膜黏连', 'APTT', 'FEV1实/预', 'FEV1/FVC预计值', 'DLCO实/预', 'Smoking', 'CCI', 'bleeding', 'surgerytime', '袖切类型1（1-简单袖切，2=扩大袖切，3=袖+部分叶）', 'FVC实测值', 'FVC实/预', 'FEV1/FVC实测值', 'DLCO实测值']
# # dm04挑选的
# selected_features = ['APTT', 'FEV1实/预', 'surgerytime', 'FEV1/FVC预计值']
#selected_features = ['APTT', 'FEV1实/预', 'surgerytime', 'FEV1/FVC预计值', 'Dimer', 'bleeding', 'FEV1/FVC实/预', '切除肺叶_5', 'BMI', 'DLCO实/预', '胸膜黏连', 'CCI', 'DLCO预计值', 'FVC实/预', 'fibrinogen', 'FEV1/FVC实测值', 'DLCO实测值', '白蛋白', 'LYM']

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

print("训练集标签分布：")
print(pd.Series(y_train).value_counts())
print("测试集标签分布：")
print(pd.Series(y_test).value_counts())

# ==================== 2. 依赖 ====================
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)
from imblearn.combine import SMOTEENN, SMOTETomek

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


# ==================== 4. 方法一：scale_pos_weight ====================
print("\n>>> 方法 1：XGBoost + scale_pos_weight")
# 计算正负样本比例，用于 scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight = {scale_pos_weight:.4f}")

xgb_bal = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_bal.fit(X_train, y_train)
proba = xgb_bal.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
evaluate("XGBoost + scale_pos_weight", y_test, pred, proba)


# ==================== 5. 方法二：SMOTEENN + XGBoost ====================
print("\n>>> 方法 2：SMOTEENN + XGBoost")
smoteenn = SMOTEENN(random_state=42)
X_se, y_se = smoteenn.fit_resample(X_train, y_train)
print("SMOTEENN 后训练集分布：")
print(pd.Series(y_se).value_counts())

xgb_se = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_se.fit(X_se, y_se)
proba = xgb_se.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
evaluate("SMOTEENN + XGBoost", y_test, pred, proba)


# ==================== 6. 方法三：SMOTETomek + XGBoost ====================
print("\n>>> 方法 3：SMOTETomek + XGBoost")
smote_tomek = SMOTETomek(random_state=42)
X_st, y_st = smote_tomek.fit_resample(X_train, y_train)
print("SMOTETomek 后训练集分布：")
print(pd.Series(y_st).value_counts())

xgb_st = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_st.fit(X_st, y_st)
proba = xgb_st.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
evaluate("SMOTETomek + XGBoost", y_test, pred, proba)


# ==================== 7. 汇总对比 ====================
print("\n\n" + "=" * 62)
print("XGBoost 相关方法对比汇总（按 PR-AUC 降序）")
print("=" * 62)
summary = pd.DataFrame(results).T
summary = summary.sort_values('PR-AUC', ascending=False)
print(summary.round(4))

# summary.to_csv("./xgb_methods_results.csv", encoding="utf-8-sig")
# print("\n结果已保存到 ./xgb_methods_results.csv")