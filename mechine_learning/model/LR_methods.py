
import sys, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from process_data.dm05_get_data import prepare_data

# ==================== 1. 数据准备 ====================
selected_features = ['切除肺叶_5', '胸膜黏连', 'APTT', 'FEV1实/预', 'FEV1/FVC预计值', 'DLCO实/预', 'Smoking', 'CCI', 'bleeding', 'surgerytime', '袖切类型1（1-简单袖切，2=扩大袖切，3=袖+部分叶）', 'FVC实测值', 'FVC实/预', 'FEV1/FVC实测值', 'DLCO实测值']

X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
    file_path="./data/肺科多模态数据-处理后.xlsx", sheet_name="Sheet1",
    selected_features=selected_features, use_lobe_features=True,
    test_size=0.2, random_state=42)

X_train = np.asarray(X_train, dtype=np.float32)
X_test  = np.asarray(X_test,  dtype=np.float32)
y_train = np.asarray(y_train).astype(int).ravel()
y_test  = np.asarray(y_test).astype(int).ravel()

print("训练集标签分布：")
print(pd.Series(y_train).value_counts())
print("测试集标签分布：")
print(pd.Series(y_test).value_counts())

# ==================== 2. 依赖 ====================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix)
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


# ==================== 4. 方法一：class_weight ====================
print("\n>>> 方法 1：LogisticRegression + class_weight='balanced'")
lr_bal = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='liblinear',       # 小样本/不平衡下更稳
    class_weight='balanced',
    max_iter=2000,
    random_state=42
)
lr_bal.fit(X_train, y_train)
proba = lr_bal.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
evaluate("LR + class_weight", y_test, pred, proba)


# ==================== 5. 方法二：SMOTEENN + LR ====================
print("\n>>> 方法 2：SMOTEENN + LogisticRegression")
smoteenn = SMOTEENN(random_state=42)
X_se, y_se = smoteenn.fit_resample(X_train, y_train)
print("SMOTEENN 后训练集分布：")
print(pd.Series(y_se).value_counts())

lr_se = LogisticRegression(C=1.0, penalty='l2', solver='liblinear',
                           max_iter=2000, random_state=42)
lr_se.fit(X_se, y_se)
proba = lr_se.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
evaluate("SMOTEENN + LR", y_test, pred, proba)


# ==================== 6. 方法三：SMOTETomek + LR ====================
print("\n>>> 方法 3：SMOTETomek + LogisticRegression")
smote_tomek = SMOTETomek(random_state=42)
X_st, y_st = smote_tomek.fit_resample(X_train, y_train)
print("SMOTETomek 后训练集分布：")
print(pd.Series(y_st).value_counts())

lr_st = LogisticRegression(C=1.0, penalty='l2', solver='liblinear',
                           max_iter=2000, random_state=42)
lr_st.fit(X_st, y_st)
proba = lr_st.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
evaluate("SMOTETomek + LR", y_test, pred, proba)


# ==================== 7. 汇总对比 ====================
print("\n\n" + "=" * 62)
print("LogisticRegression 相关方法对比汇总（按 PR-AUC 降序）")
print("=" * 62)
summary = pd.DataFrame(results).T
summary = summary.sort_values('PR-AUC', ascending=False)
print(summary.round(4))

# summary.to_csv("./lr_methods_results.csv", encoding="utf-8-sig")
# print("\n结果已保存到 ./lr_methods_results.csv")