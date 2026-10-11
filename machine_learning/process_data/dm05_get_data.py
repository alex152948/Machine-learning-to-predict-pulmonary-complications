# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def prepare_data(
    file_path,
    sheet_name="Sheet1",
    selected_features=None,
    target_col=None,
    use_lobe_features=False,
    test_size=0.2,
    random_state=42,
    scale=True
):
    """
    读取肺科数据并做预处理。

    改动说明：
    1. 先拆分“切除肺叶”为二值列，例如 切除肺叶_1、切除肺叶_2。
    2. 然后最终特征完全根据 selected_features 挑选。
       如果想用肺叶拆分列，请在 selected_features 中显式写：
       ['APTT', 'FEV1实/预', '切除肺叶_1', '切除肺叶_2']
    3. use_lobe_features 仅在 selected_features=None 时生效：
       如果为 True，则默认特征后追加所有肺叶拆分列；
       如果为 False，则默认特征不包含肺叶拆分列。
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = df.columns.astype(str).str.strip()

    # 前三列不要：住院号、手术日期、患者ID
    df = df.iloc[:, 3:].copy()

    # 确定标签列
    if target_col is None:
        target_col = df.columns[-1]
    else:
        target_col = target_col.strip()

    # 1) 先处理“切除肺叶”，拆成二值列
    lobe_cols = []
    if '切除肺叶' in df.columns:
        df['切除肺叶'] = df['切除肺叶'].astype(str).str.replace(r'\.0$', '', regex=True)

        lobe_numbers = set()
        for val in df['切除肺叶']:
            for ch in str(val):
                if ch.isdigit():
                    lobe_numbers.add(ch)

        lobe_numbers = sorted(lobe_numbers)

        for lobe in lobe_numbers:
            col_name = f'切除肺叶_{lobe}'
            df[col_name] = df['切除肺叶'].apply(lambda x: 1 if lobe in str(x) else 0)
            lobe_cols.append(col_name)

    # 2) 再确定最终要用的特征
    if selected_features is None:
        selected_features = [
            'APTT',
            'FEV1实/预',
            'surgerytime',
            'FEV1/FVC预计值'
        ]

        # 只有没显式传 selected_features 时，use_lobe_features 才决定是否加入肺叶列
        if use_lobe_features:
            selected_features = selected_features + lobe_cols
    else:
        # 显式传入 selected_features 时，完全按传入列表选特征
        # 注意：如果要使用肺叶拆分列，请写 '切除肺叶_1' 这类列名
        selected_features = [c.strip() for c in selected_features]

    # 统一 strip，防止默认特征里有空格
    selected_features = [c.strip() for c in selected_features]

    # 检查列是否存在
    missing_cols = [c for c in selected_features + [target_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"以下列不存在，请检查列名：{missing_cols}")

    X = df[selected_features].copy()
    y = df[target_col].copy()

    # 缺失值处理
    data = pd.concat([X, y], axis=1).dropna()
    X = data[selected_features]
    y = data[target_col]

    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )

    scaler = None
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=selected_features, index=X_test.index)

    return X_train, X_test, y_train, y_test, scaler, selected_features