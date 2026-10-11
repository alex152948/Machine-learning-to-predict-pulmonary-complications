
import sys, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from process_data.dm05_get_data import prepare_data

selected_features = ['切除肺叶_5', '胸膜黏连', 'APTT', 'FEV1实/预', 'FEV1/FVC预计值', 'DLCO实/预', 'Smoking', 'CCI', 'bleeding', 'surgerytime', '袖切类型1（1-简单袖切，2=扩大袖切，3=袖+部分叶）', 'FVC实测值', 'FVC实/预', 'FEV1/FVC实测值', 'DLCO实测值']

X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
    file_path="./data/肺科多模态数据-处理后.xlsx", sheet_name="Sheet1",
    selected_features=selected_features, use_lobe_features=True,
    test_size=0.2, random_state=42)

X_train = np.asarray(X_train, dtype=np.float32)
X_test  = np.asarray(X_test,  dtype=np.float32)
y_train = np.asarray(y_train).astype(int).ravel()
y_test  = np.asarray(y_test).astype(int).ravel()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix)
from imblearn.combine import SMOTEENN, SMOTETomek

results = {}
def evaluate(name, y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred); prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0); f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score); pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        auc, pr_auc = np.nan, np.nan
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{'='*62}\n模型：{name}\n{'='*62}")
    print(f"Accuracy : {acc:.4f}\nPrecision: {prec:.4f}\nRecall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}\nMCC      : {mcc:.4f}\nROC-AUC  : {auc:.4f}\nPR-AUC   : {pr_auc:.4f}")
    print("混淆矩阵：\n", cm)
    results[name] = {'Accuracy':acc,'Precision':prec,'Recall':rec,'F1':f1,'MCC':mcc,'ROC-AUC':auc,'PR-AUC':pr_auc}

# 方法1：GBDT + sample_weight 处理不平衡
print("\n>>> 方法 1：GBDT + sample_weight")
sample_w = np.where(y_train == 1, (y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
gbdt_bal = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                      max_depth=3, subsample=0.8, random_state=42)
gbdt_bal.fit(X_train, y_train, sample_weight=sample_w)
proba = gbdt_bal.predict_proba(X_test)[:, 1]; pred = (proba >= 0.5).astype(int)
evaluate("GBDT + sample_weight", y_test, pred, proba)

# 方法2：SMOTEENN + GBDT
print("\n>>> 方法 2：SMOTEENN + GBDT")
X_se, y_se = SMOTEENN(random_state=42).fit_resample(X_train, y_train)
print(pd.Series(y_se).value_counts())
gbdt_se = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                     max_depth=3, subsample=0.8, random_state=42)
gbdt_se.fit(X_se, y_se)
proba = gbdt_se.predict_proba(X_test)[:, 1]; pred = (proba >= 0.5).astype(int)
evaluate("SMOTEENN + GBDT", y_test, pred, proba)

# 方法3：SMOTETomek + GBDT
print("\n>>> 方法 3：SMOTETomek + GBDT")
X_st, y_st = SMOTETomek(random_state=42).fit_resample(X_train, y_train)
print(pd.Series(y_st).value_counts())
gbdt_st = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                     max_depth=3, subsample=0.8, random_state=42)
gbdt_st.fit(X_st, y_st)
proba = gbdt_st.predict_proba(X_test)[:, 1]; pred = (proba >= 0.5).astype(int)
evaluate("SMOTETomek + GBDT", y_test, pred, proba)

print("\n\n" + "="*62 + "\nGBDT 相关方法对比汇总（按 PR-AUC 降序）\n" + "="*62)
summary = pd.DataFrame(results).T.sort_values('PR-AUC', ascending=False)
print(summary.round(4))
# summary.to_csv("./gbdt_methods_results.csv", encoding="utf-8-sig")