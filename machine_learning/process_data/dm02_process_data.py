import pandas as pd
import os

# 1. 设置文件路径（根据你的截图，数据在上一级的data文件夹中）
data_path = './data/肺科多模态数据-纯享版.xlsx'
output_path = './data/肺科多模态数据-处理后.xlsx'

# 检查文件是否存在
if not os.path.exists(data_path):
    print(f"找不到文件，请确认路径是否正确: {data_path}")
else:
    # 2. 读取数据
    print("正在读取数据...")
    df = pd.read_excel(data_path)
    print(f"✅ 原始数据集形状: {df.shape[0]} 行, {df.shape[1]} 列\n")

    # 3. 计算每一列的缺失值比例
    missing_ratio = df.isnull().mean()

    # 4. 找出缺失比例大于 15% 的列名
    threshold = 0.15
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

    print(f"⚠️ 缺失率大于 {threshold*100}% 的列共有 {len(cols_to_drop)} 列:")
    for col in cols_to_drop:
        print(f"   - {col} (缺失率: {missing_ratio[col]*100:.2f}%)")
    print("-" * 50)

    # 5. 删除这些列
    df_cleaned = df.drop(columns=cols_to_drop)
    print(f"✅ 处理后数据集形状: {df_cleaned.shape[0]} 行, {df_cleaned.shape[1]} 列")
    print(f"✅ 共删除了 {df.shape[1] - df_cleaned.shape[1]} 列\n")
    """
    ⚠️ 缺失率大于 15.0% 的列共有 5 列:
   - CEA (缺失率: 17.61%)
   - CYFRA211 (缺失率: 17.93%)
   - SCCA (缺失率: 18.04%)
   - 并发症等级 (缺失率: 88.15%)
   - Unnamed: 68 (缺失率: 99.57%)
    """
    # ================= 统计有缺失值的行 =================
    print("🔍 开始统计行缺失情况...")
    
    # isnull().any(axis=1) 表示只要该行有任意一个缺失值就为 True
    rows_with_missing = df_cleaned.isnull().any(axis=1).sum()
    total_rows = df_cleaned.shape[0]
    
    print(f"⚠️ 处理后数据中，存在缺失值的行共有: {rows_with_missing} 行 (占比 {rows_with_missing / total_rows * 100:.2f}%)")
    print(f"✅ 处理后数据中，无任何缺失值的行共有: {total_rows - rows_with_missing} 行")
    """
    🔍 开始统计行缺失情况...
⚠️ 处理后数据中，存在缺失值的行共有: 10 行 (占比 1.09%)
✅ 处理后数据中，无任何缺失值的行共有: 910 行
    """
    
    # 比原始数据的行缺失情况
    # raw_rows_with_missing = df.isnull().any(axis=1).sum()
    # print(f"📊 对比：原始数据中，存在缺失值的行共有: {raw_rows_with_missing} 行 (占比 {raw_rows_with_missing / df.shape[0] * 100:.2f}%)")
    
    print("-" * 50)
 # ================= 新增：直接删除有缺失值的行 =================
    print("🔍 开始处理行缺失情况...")
    
    # 记录删除前的行数
    rows_before = df_cleaned.shape[0]
    
    # dropna(axis=0, how='any') 表示：只要该行有任意一个缺失值，就删除该行
    df_final = df_cleaned.dropna(axis=0, how='any')
    
    # 计算删除了多少行
    rows_deleted = rows_before - df_final.shape[0]
    
    print(f"⚠️ 发现并删除了 {rows_deleted} 行包含缺失值的数据。")
    print(f"✅ 最终数据集形状: {df_final.shape[0]} 行, {df_final.shape[1]} 列\n")


    # ================= 复查是否还有缺失值 =================
    print("🕵️‍♂️ 正在复查最终数据集中是否还有缺失值...")
    
    # 计算每一列的缺失值数量
    final_missing_counts = df_final.isnull().sum()
    # 计算整个数据集的缺失值总数
    total_missing = final_missing_counts.sum()

    if total_missing == 0:
        print("🎉 完美！最终数据集中没有任何缺失值了 (总缺失值: 0)。")
    else:
        print(f"❗ 警告：最终数据集中仍然存在 {total_missing} 个缺失值！具体分布如下：")
        # 只打印有缺失值的列名和数量
        print(final_missing_counts[final_missing_counts > 0])
    print("-" * 50)


    #  保存最终处理后的数据
    df_final.to_excel(output_path, index=False)
    print(f"\n💾 最终处理后的数据已保存至: {output_path}")

    """
    正在读取数据...
✅ 原始数据集形状: 920 行, 69 列

⚠️ 缺失率大于 15.0% 的列共有 5 列:
   - CEA (缺失率: 17.61%)
   - CYFRA211 (缺失率: 17.93%)
   - SCCA (缺失率: 18.04%)
   - 并发症等级 (缺失率: 88.15%)
   - Unnamed: 68 (缺失率: 99.57%)
--------------------------------------------------
✅ 处理后数据集形状: 920 行, 64 列
✅ 共删除了 5 列

🔍 开始统计行缺失情况...
⚠️ 处理后数据中，存在缺失值的行共有: 10 行 (占比 1.09%)
✅ 处理后数据中，无任何缺失值的行共有: 910 行
--------------------------------------------------
🔍 开始处理行缺失情况...
⚠️ 发现并删除了 10 行包含缺失值的数据。
✅ 最终数据集形状: 910 行, 64 列

🕵️‍♂️ 正在复查最终数据集中是否还有缺失值...
🎉 完美！最终数据集中没有任何缺失值了 (总缺失值: 0)。
--------------------------------------------------
    """

