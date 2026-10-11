
import pandas as pd
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

file_path = ".\\data\\肺科多模态数据-纯享版.xlsx"
df = pd.read_excel(file_path, sheet_name=0, engine="openpyxl")

# 清理列名
df.columns = df.columns.astype(str).str.strip()

# 先备份原始列名
print("数据形状:", df.shape)

# 每列缺失情况
missing_col = pd.DataFrame({
    "列名": df.columns,
    "非空数": df.notna().sum().values,
    "空值数": df.isna().sum().values,
    "空值率": df.isna().mean().values
}).sort_values("空值数", ascending=False)

print("=== 空值较多的列（按空值数降序）===")
print(missing_col[missing_col["空值数"] > 0].head(50))  # 只看有缺失的列，前50

# 如果只想看空值率超过20%的列
print("=== 空值率 > 20% 的列 ===")
print(missing_col[missing_col["空值率"] > 0.2])

# 行缺失情况
row_missing_count = df.isna().sum(axis=1)
rows_with_missing = df[row_missing_count > 0].copy()

# 给出行号、缺失数量、缺失列名
rows_with_missing_info = pd.DataFrame({
    "行号(Excel从2开始)": rows_with_missing.index + 2,  # 如果第一行是表头
    "pandas索引": rows_with_missing.index,
    "缺失数量": row_missing_count[rows_with_missing.index],
    "缺失列名": rows_with_missing.apply(
        lambda row: "、".join(row.index[row.isna()].astype(str)), axis=1
    )
})

# 如果保留有住院号，可以加入住院号
if "住院号" in df.columns:
    rows_with_missing_info.insert(1, "住院号", df.loc[rows_with_missing.index, "住院号"].values)

print("=== 有空值的行 ===")
print(rows_with_missing_info.head(100))  # 先看前100行

# 保存结果
missing_col.to_excel(".\\data\\缺失列统计.xlsx", index=False)
rows_with_missing_info.to_excel(".\\data\\缺失行统计.xlsx", index=False)