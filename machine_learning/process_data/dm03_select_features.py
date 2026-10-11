import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from boruta import BorutaPy

# ===================== 0. 关键参数 =====================
K = 20

# ===================== 1. 读取数据 =====================
data_path = './data/肺科多模态数据-处理后.xlsx'
df = pd.read_excel(data_path)

target_col = df.columns[-1]
df = df.dropna(subset=[target_col]).copy()

X_raw = df.iloc[:, 3:-1].copy()
y_raw = df[target_col].copy()

# ===================== 1.1 特殊处理：切除肺叶 =====================
# 切除肺叶取值有 1、2、3、4、5 共 5 种，代表 5 个肺叶。
# 例如 34 表示切除 3 和 4，不是数字三十四。
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
        lobes = set(re.findall(r'[1-5]', s))
        return [1 if str(i) in lobes else 0 for i in LOBE_CODES]

    lobe_df = X_raw['切除肺叶'].apply(split_lung_lobes).apply(pd.Series)
    lobe_df.columns = [f'切除肺叶_{i}' for i in LOBE_CODES]

    X_raw = pd.concat([X_raw.drop(columns=['切除肺叶']), lobe_df], axis=1)
    print('切除肺叶已拆分为：', lobe_df.columns.tolist())

# ===================== 2. 自动判断分类/回归 =====================
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

# ===================== 4. Lasso / L1 =====================
print('\n===== 1. Lasso / L1 Top-K =====')

if task == 'classification':
    model_lasso = LogisticRegressionCV(
        Cs=10, cv=5, penalty='l1', solver='liblinear',
        scoring='roc_auc' if len(np.unique(y)) == 2 else 'accuracy',
        max_iter=5000, class_weight='balanced', random_state=0
    )
    model_lasso.fit(X, y)
    coef = model_lasso.coef_
    importance = np.abs(coef) if coef.ndim == 1 else np.max(np.abs(coef), axis=0)
else:
    model_lasso = LassoCV(cv=5, random_state=0, max_iter=10000)
    model_lasso.fit(X, y)
    importance = np.abs(model_lasso.coef_)

lasso_rank = pd.Series(importance, index=X.columns).sort_values(ascending=False)
selected_lasso = lasso_rank.head(K_use).index.tolist()

print(f'Lasso 选中特征数: {len(selected_lasso)}')
print(selected_lasso)

# ===================== 5. RFECV =====================
print('\n===== 2. RFECV Top-K =====')

if task == 'classification':
    estimator = LogisticRegression(max_iter=5000, solver='liblinear',
                                   class_weight='balanced', random_state=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scoring = 'roc_auc' if len(np.unique(y)) == 2 else 'accuracy'
else:
    estimator = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scoring = 'neg_mean_squared_error'

rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    min_features_to_select=K_use,
)
rfecv.fit(X, y)

rfecv_rank = pd.Series(rfecv.ranking_, index=X.columns).sort_values()
selected_rfecv = rfecv_rank.head(K_use).index.tolist()

print(f'RFECV 选中特征数: {len(selected_rfecv)}（固定为 Top-{K_use}）')
print(selected_rfecv)

# ===================== 6. Boruta =====================
print('\n===== 3. Boruta Top-K =====')

if task == 'classification':
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced',
                                max_depth=5, random_state=0)
else:
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=0)

boruta = BorutaPy(
    estimator=rf,
    n_estimators='auto',
    verbose=2,
    random_state=0,
    max_iter=100
)
boruta.fit(X.values, y)

boruta_rank = pd.Series(boruta.ranking_, index=X.columns).sort_values()
selected_boruta = boruta_rank.head(K_use).index.tolist()

print(f'Boruta 选中特征数: {len(selected_boruta)}（固定为 Top-{K_use}）')
print(selected_boruta)

# ===================== 7. 汇总 =====================
result = pd.DataFrame({
    'feature': X.columns,
    'Lasso': [1 if c in selected_lasso else 0 for c in X.columns],
    'RFECV': [1 if c in selected_rfecv else 0 for c in X.columns],
    'Boruta': [1 if c in selected_boruta else 0 for c in X.columns],
})
result['投票数'] = result[['Lasso', 'RFECV', 'Boruta']].sum(axis=1)

result['Lasso_rank'] = result['feature'].map(
    {f: i + 1 for i, f in enumerate(lasso_rank.index)})
result['RFECV_rank'] = result['feature'].map(
    {f: i + 1 for i, f in enumerate(rfecv_rank.index)})
result['Boruta_rank'] = result['feature'].map(
    {f: i + 1 for i, f in enumerate(boruta_rank.index)})

result = result.sort_values(['投票数', 'Lasso_rank'], ascending=[False, True])

out_path = './data/特征筛选结果dm03.xlsx'
result.to_excel(out_path, index=False)
print(f'\n汇总结果已保存到: {out_path}')

common = result[
    (result['Lasso'] == 1) & (result['RFECV'] == 1) & (result['Boruta'] == 1)
]['feature'].tolist()
print(f'\n三种方法共同选中特征数: {len(common)}')
print(common)

vote2 = result[result['投票数'] >= 2]['feature'].tolist()
print(f'\n至少被 2 种方法选中特征数: {len(vote2)}')
print(vote2)

"""
三种方法共同选中特征数: 6
['切除肺叶_5', '胸膜黏连', 'APTT', 'FEV1实/预', 'FEV1/FVC预计值', 'DLCO实/预']

至少被 2 种方法选中特征数: 15
['切除肺叶_5', '胸膜黏连', 'APTT', 'FEV1实/预', 'FEV1/FVC预计值', 'DLCO实/预', 'Smoking', 'CCI', 'bleeding', 'surgerytime', '袖切类型1（1-简单袖切，2=扩大袖切，3=袖+部分叶）', 'FVC实测值', 'FVC实/预', 'FEV1/FVC实测值', 'DLCO实测值']
"""