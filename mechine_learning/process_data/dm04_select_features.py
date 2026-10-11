import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)

# ===================== 0. 统一参数 =====================
K = 20   # 每个方法都取 Top-K 个特征

# ===================== 1. 读取数据 =====================
data_path = './data/肺科多模态数据-处理后.xlsx'
df = pd.read_excel(data_path)

target_col = df.columns[-1]
df = df.dropna(subset=[target_col]).copy()

X_raw = df.iloc[:, 3:-1].copy()
y_raw = df[target_col].copy()

# ===================== 1.1 特殊处理：切除肺叶 =====================
# 说明：切除肺叶取值有 1、2、3、4、5 共 5 种，代表 5 个肺叶。
# 例如 34 表示切除 3 和 4，不是数字三十四；345 表示切除 3、4、5。
# 因此把“切除肺叶”拆成 5 个二值列：切除肺叶_1 ~ 切除肺叶_5。
X_raw.columns = X_raw.columns.astype(str).str.strip()

if '切除肺叶' in X_raw.columns:
    LOBE_CODES = [1, 2, 3, 4, 5]   # ← 5 个肺叶

    def split_lung_lobes(x):
        if pd.isna(x):
            return [0] * len(LOBE_CODES)
        s = str(x).strip()
        s = re.sub(r'\.0+$', '', s)  # 34.0 -> 34, 3.0 -> 3
        if s in ['', 'nan', 'None', '0']:
            return [0] * len(LOBE_CODES)
        # 提取所有 1~5 的数字字符，支持 "34"、"3,4"、"3、4"、"3 4"、"345" 等写法
        lobes = set(re.findall(r'[1-5]', s))
        return [1 if str(i) in lobes else 0 for i in LOBE_CODES]

    lobe_df = X_raw['切除肺叶'].apply(split_lung_lobes).apply(pd.Series)
    lobe_df.columns = [f'切除肺叶_{i}' for i in LOBE_CODES]

    X_raw = pd.concat([X_raw.drop(columns=['切除肺叶']), lobe_df], axis=1)
    print('切除肺叶已拆分为：', lobe_df.columns.tolist())

# ===================== 2. 自动判断任务类型 =====================
if (y_raw.dtype == 'object') or (str(y_raw.dtype).startswith('category')) or (y_raw.nunique() <= 20):
    task = 'classification'
else:
    task = 'regression'

print(f'任务类型: {task}，目标唯一值数: {y_raw.nunique()}')

if task == 'classification':
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    print('类别标签:', list(le.classes_))
else:
    y = y_raw.values.astype(float)

# ===================== 3. 特征预处理 =====================
num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_raw.columns if c not in num_cols]

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocess = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
], remainder='drop')

X_arr = preprocess.fit_transform(X_raw)

encoded_feature_names = []
if num_cols:
    encoded_feature_names.extend(num_cols)
if cat_cols:
    ohe = preprocess.named_transformers_['cat'].named_steps['onehot']
    encoded_feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())

X = pd.DataFrame(X_arr, columns=encoded_feature_names)
print(f'原始特征数: {X_raw.shape[1]}，编码后特征数: {X.shape[1]}')

K_use = min(K, X.shape[1])
if K_use != K:
    print(f'警告：K={K} 超过特征总数 {X.shape[1]}，自动改为 K={K_use}')


# ===================== 4. 三个统一“重要性打分 + Top-K”方法 =====================

def topk_from_scores(scores, feature_names, k):
    """按分数从大到小取 Top-K，返回 (选中特征列表, 排名Series)"""
    s = pd.Series(np.abs(scores), index=feature_names).sort_values(ascending=False)
    return s.head(k).index.tolist(), s


# ---------- 方法 1：Lasso / L1 逻辑回归 ----------
print('\n===== 1. Lasso / L1 —— |coef| Top-K =====')

if task == 'classification':
    model_l1 = LogisticRegressionCV(
        Cs=10, cv=5, penalty='l1', solver='liblinear',
        scoring='roc_auc' if len(np.unique(y)) == 2 else 'accuracy',
        max_iter=5000, class_weight='balanced', random_state=0
    )
    model_l1.fit(X, y)
    coef = model_l1.coef_
    l1_scores = np.abs(coef) if coef.ndim == 1 else np.max(np.abs(coef), axis=0)
else:
    model_l1 = LassoCV(cv=5, random_state=0, max_iter=10000)
    model_l1.fit(X, y)
    l1_scores = np.abs(model_l1.coef_)

selected_l1, rank_l1 = topk_from_scores(l1_scores, X.columns, K_use)
print(f'Lasso/L1 选中 Top-{K_use}:')
print(selected_l1)


# ---------- 方法 2：随机森林 ----------
print('\n===== 2. Random Forest —— feature_importances_ Top-K =====')

if task == 'classification':
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None,
        class_weight='balanced', n_jobs=-1, random_state=0
    )
else:
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=None,
        n_jobs=-1, random_state=0
    )

rf.fit(X, y)
rf_scores = rf.feature_importances_
selected_rf, rank_rf = topk_from_scores(rf_scores, X.columns, K_use)
print(f'RF 选中 Top-{K_use}:')
print(selected_rf)


# ---------- 方法 3：梯度提升树 ----------
print('\n===== 3. Gradient Boosting —— feature_importances_ Top-K =====')

if task == 'classification':
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=3, random_state=0
    )
else:
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=3, random_state=0
    )

gb.fit(X, y)
gb_scores = gb.feature_importances_
selected_gb, rank_gb = topk_from_scores(gb_scores, X.columns, K_use)
print(f'GB 选中 Top-{K_use}:')
print(selected_gb)


# ===================== 5. 汇总 =====================
result = pd.DataFrame({
    'feature': X.columns,
    'L1': [1 if c in selected_l1 else 0 for c in X.columns],
    'RF': [1 if c in selected_rf else 0 for c in X.columns],
    'GB': [1 if c in selected_gb else 0 for c in X.columns],
})
result['投票数'] = result[['L1', 'RF', 'GB']].sum(axis=1)

# 每个方法的重要性排名（越小越重要）
result['L1_rank'] = result['feature'].map({f: i + 1 for i, f in enumerate(rank_l1.index)})
result['RF_rank'] = result['feature'].map({f: i + 1 for i, f in enumerate(rank_rf.index)})
result['GB_rank'] = result['feature'].map({f: i + 1 for i, f in enumerate(rank_gb.index)})
result['平均排名'] = result[['L1_rank', 'RF_rank', 'GB_rank']].mean(axis=1)

result = result.sort_values(['投票数', '平均排名'], ascending=[False, True])

out_path = './data/特征筛选结果dm04.xlsx'
result.to_excel(out_path, index=False)
print(f'\n汇总结果已保存到: {out_path}')

# 三种方法共同选中
common = result[
    (result['L1'] == 1) & (result['RF'] == 1) & (result['GB'] == 1)
]['feature'].tolist()
print(f'\n三种方法共同选中特征数: {len(common)}')
print(common)

# 至少被 2 种方法选中
vote2 = result[result['投票数'] >= 2]['feature'].tolist()
print(f'\n至少被 2 种方法选中特征数: {len(vote2)}')
print(vote2)

"""
三种方法共同选中特征数: 4
['APTT', 'FEV1实/预', 'surgerytime', 'FEV1/FVC预计值']

至少被 2 种方法选中特征数: 19
['APTT', 'FEV1实/预', 'surgerytime', 'FEV1/FVC预计值', 'Dimer', 'bleeding', 'FEV1/FVC实/预', '切除肺叶_5', 'BMI', 'DLCO实/预', '胸膜黏连', 'CCI', 'DLCO预计值', 'FVC实/预', 'fibrinogen', 'FEV1/FVC实测值', 'DLCO实测值', '白蛋白', 'LYM']
"""