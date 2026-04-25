import pandas as pd
import os

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.font_manager as fm

# 设置中文字体，保证图中中文标签正常显示
# 如果系统中没有 "SimHei"，可改为其他常用中文字体如 "Microsoft YaHei"、"STSong" 等
chinese_font = 'SimHei'
plt.rcParams['font.sans-serif'] = [chinese_font]
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示


def inspect_trade_data(file_path):
    print(f"========== 正在诊断文件: {os.path.basename(file_path)} ==========")
    
    # 1. 动态判断文件类型并读取
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == '.csv':
            # 尝试常见编码，防止报错
            try:
                df = pd.read_csv(file_path, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='gbk', low_memory=False)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            print(f"不支持的文件格式: {ext}")
            return
    except Exception as e:
        print(f"读取文件出错: {e}")
        return

    # 2. 打印基础信息
    print(f"\n[1] 数据框维度: 行数 {df.shape[0]}, 列数 {df.shape[1]}")
    print("\n[2] 列名及数据类型:")
    print(df.dtypes)
    
    # 3. 缺失值检查
    print("\n[3] 缺失值统计 (仅显示存在缺失的列):")
    missing = df.isnull().sum()
    print(missing[missing > 0] if not missing[missing > 0].empty else "无缺失值！")

    # 4. 核心维度探查 (通过名称模式动态匹配，先分类后分析)
    import re
    print("\n[4] 核心分析维度内容总览:")

    # 模式定义，可根据需要调整或扩展
    year_pattern     = re.compile(r"(year|time|年)", re.I)
    country_pattern  = re.compile(r"(cou|rep|par|国)", re.I)
    sector_pattern   = re.compile(r"(ind|sec|ser|ebops|行)", re.I)

    year_cols, country_cols, sector_cols = [], [], []
    for col in df.columns:
        col_str = str(col)
        if year_pattern.search(col_str):
            year_cols.append(col)
        elif country_pattern.search(col_str):
            country_cols.append(col)
        elif sector_pattern.search(col_str):
            sector_cols.append(col)

    # 处理年份列
    if not year_cols:
        print("  -> 未识别到任何可能的年份列")
    for col in year_cols:
        values = df[col].dropna().unique()
        if values.size == 0:
            print(f"  -> 年份列 [{col}] 被识别但没有任何有效数据")
            continue
        years = sorted(values)
        lo, hi = years[0], years[-1]
        print(f"  -> 发现年份列 [{col}]: 共 {len(years)} 年，范围 {lo} 到 {hi}")

    # 处理国家/地区列
    if not country_cols:
        print("  -> 未识别到任何可能的国家/地区列")
    for col in country_cols:
        unique_countries = df[col].nunique(dropna=True)
        print(f"  -> 发现国家/地区列 [{col}]: 共 {unique_countries} 个唯一节点")

    # 处理行业/服务分类列
    if not sector_cols:
        print("  -> 未识别到任何可能的行业/服务分类列")
    for col in sector_cols:
        sectors = df[col].dropna().unique()
        unique_sectors = sorted([str(x) for x in sectors])
        print(f"  -> 发现分类列 [{col}]: 包含 {len(unique_sectors)} 种分类。示例前5项: {unique_sectors[:5]}")
        # 输出全部不重复的行业/服务分类代码
        print(f"     -> 全部不重复分类代码 ({len(unique_sectors)}): {unique_sectors}")

    print("====================== 诊断结束 ======================\n")


def clean_trade_data(file_path, ref_area_col='REF_AREA', counterpart_col='COUNTERPART_AREA', time_col='TIME_PERIOD'):
    """
    数据清洗函数：读取贸易数据，过滤非国家宏观聚合体，返回清洁数据框。
    
    参数:
        file_path: 数据文件路径
        ref_area_col: 出口国列名（默认 'REF_AREA'）
        counterpart_col: 进口国列名（默认 'COUNTERPART_AREA'）
        time_col: 年份列名（默认 'TIME_PERIOD'）
    
    返回:
        清洁后的 DataFrame（已过滤非国家聚合体）
    """
    
    print(f"\n========== 开始数据清洗: {os.path.basename(file_path)} ==========\n")
    
    # 读取数据
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == '.csv':
            try:
                df = pd.read_csv(file_path, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='gbk', low_memory=False)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            print(f"❌ 不支持的文件格式: {ext}")
            return None
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return None
    
    print(f"✓ 原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
    
    # ========== 【非国家宏观聚合体过滤模块】==========
    # 定义常见的区域聚合、超国家组织或非独立国家代码
    NON_COUNTRY_AGGREGATES = {
        # 欧盟及相关聚合体
        'EU27', 'EU28', 'EU25', 'EU15', 'EU12', 'EU',
        # 区域组织与经济体
        'OECD', 'G7', 'G20', 'BRICS',
        # 世界与大陆聚合
        'WLD', 'WORLD',
        'ASI', 'ASIA', 'AFR', 'AFRICA', 'EUR', 'EUROPE', 
        'AMR', 'AMERICAS', 'NAM', 'NORTH AMERICA', 'SAM', 'SOUTH AMERICA',
        # 其他常见聚合代码
        'XA', 'XB', 'XC', 'X9', # OECD 特殊代码
        'OAS',  # 其他亚洲
        'USA+CAN',  # 北美组合（如果出现）
    }
    
    print(f"\n[过滤] 定义的非国家聚合体列表 ({len(NON_COUNTRY_AGGREGATES)} 项):")
    print(f"       {sorted(NON_COUNTRY_AGGREGATES)}\n")
    
    # 检查数据框中是否存在必要列
    if ref_area_col not in df.columns:
        print(f"❌ 错误: 列 '{ref_area_col}' 不存在于数据框中")
        print(f"   可用列: {list(df.columns)}")
        return None
    
    if counterpart_col not in df.columns:
        print(f"❌ 错误: 列 '{counterpart_col}' 不存在于数据框中")
        print(f"   可用列: {list(df.columns)}")
        return None
    
    # 记录过滤前的行数
    original_rows = len(df)
    
    # 过滤第一步: 删除 REF_AREA（出口国）中包含聚合体的行
    df_filtered = df[~df[ref_area_col].isin(NON_COUNTRY_AGGREGATES)].copy()
    ref_area_removed = original_rows - len(df_filtered)
    
    print(f"[过滤-步骤1] 删除出口国({ref_area_col})为聚合体的行")
    print(f"             移除 {ref_area_removed} 行，剩余 {len(df_filtered)} 行")
    
    # 过滤第二步: 删除 COUNTERPART_AREA（进口国）中包含聚合体的行
    df_filtered = df_filtered[~df_filtered[counterpart_col].isin(NON_COUNTRY_AGGREGATES)].copy()
    counterpart_removed = len(df) - ref_area_removed - len(df_filtered)
    
    print(f"[过滤-步骤2] 删除进口国({counterpart_col})为聚合体的行")
    print(f"             移除 {counterpart_removed} 行，剩余 {len(df_filtered)} 行")
    
    total_removed = original_rows - len(df_filtered)
    removal_rate = (total_removed / original_rows * 100) if original_rows > 0 else 0
    
    print(f"\n[过滤完成] 总移除 {total_removed} 行 ({removal_rate:.2f}%)")
    print(f"           清洁后数据: {len(df_filtered)} 行, {df_filtered.shape[1]} 列\n")
    
    # 验证清洁结果
    print("[验证] 清洁数据中的国家/地区节点:")
    ref_area_unique = df_filtered[ref_area_col].unique()
    counterpart_unique = df_filtered[counterpart_col].unique()
    all_nodes = set(ref_area_unique) | set(counterpart_unique)
    
    print(f"       出口国({ref_area_col}): {len(ref_area_unique)} 个节点")
    print(f"       进口国({counterpart_col}): {len(counterpart_unique)} 个节点")
    print(f"       总计独立经济体: {len(all_nodes)} 个节点")
    
    # 二次验证: 检查是否还有聚合体残留
    remaining_aggregates_ref = set(ref_area_unique) & NON_COUNTRY_AGGREGATES
    remaining_aggregates_cp = set(counterpart_unique) & NON_COUNTRY_AGGREGATES
    
    if remaining_aggregates_ref or remaining_aggregates_cp:
        print(f"\n⚠️  警告: 发现残留的聚合体代码:")
        if remaining_aggregates_ref:
            print(f"   出口国({ref_area_col}): {remaining_aggregates_ref}")
        if remaining_aggregates_cp:
            print(f"   进口国({counterpart_col}): {remaining_aggregates_cp}")
    else:
        print(f"\n✓ 验证通过: 清洁数据中无聚合体代码残留\n")
    
    print("====================== 清洗完成 ======================\n")
    
    return df_filtered


def build_batis_static_weights(clean_df, ref_area_col='REF_AREA', counterpart_col='COUNTERPART_AREA', 
                              value_col='OBS_VALUE', time_col='TIME_PERIOD', base_years=[2005, 2006, 2007]):
    """
    基于清洁数据生成静态权重矩阵。
    
    参数:
        clean_df: 清洁后的数据框（已过滤非国家聚合体）
        ref_area_col: 出口国列名
        counterpart_col: 进口国列名
        value_col: 贸易额列名
        time_col: 年份列名
        base_years: 基期年份列表
    
    返回:
        权重矩阵 DataFrame，包含列: ref_area_col, counterpart_col, Base_Trade_ij, Base_Trade_i_Total, w_ij
    """
    
    print(f"\n========== 生成静态权重矩阵 ==========\n")
    
    # 验证必要的列是否存在
    required_cols = [ref_area_col, counterpart_col, value_col, time_col]
    missing_cols = [col for col in required_cols if col not in clean_df.columns]
    if missing_cols:
        print(f"❌ 错误: 以下列不存在于数据框中: {missing_cols}")
        print(f"   可用列: {list(clean_df.columns)}")
        return None
    
    print(f"✓ 输入清洁数据: {clean_df.shape[0]} 行")
    
    # 1. 截取基期年份数据
    df_base = clean_df[clean_df[time_col].isin(base_years)].copy()
    print(f"✓ 截取基期 {base_years}: {df_base.shape[0]} 行")
    
    if df_base.empty:
        print(f"❌ 错误: 基期年份 {base_years} 中没有数据")
        return None
    
    # 2. 聚合：计算基期内 i国(出口国) 到 j国(进口国) 的贸易额年均值
    # 注意：如果数据中有多个 SERVICE 等维度，会自动合并
    print(f"\n[步骤1] 按双边对 ({ref_area_col}-{counterpart_col}) 聚合贸易额...")
    trade_ij = df_base.groupby([ref_area_col, counterpart_col])[value_col].mean().reset_index()
    trade_ij.rename(columns={value_col: 'Base_Trade_ij'}, inplace=True)
    print(f"        生成 {len(trade_ij)} 条双边记录")
    
    # 3. 剔除自循环（本国到本国）
    before_self_loop = len(trade_ij)
    trade_ij = trade_ij[trade_ij[ref_area_col] != trade_ij[counterpart_col]].copy()
    self_loop_removed = before_self_loop - len(trade_ij)
    
    print(f"\n[步骤2] 剔除自循环 ({ref_area_col} == {counterpart_col}):")
    print(f"        移除 {self_loop_removed} 条记录，剩余 {len(trade_ij)} 条")
    
    # 4. 计算分母：i国在基期向全球出口的贸易总额
    print(f"\n[步骤3] 计算各出口国向全球的总出口额...")
    trade_i_total = trade_ij.groupby(ref_area_col)['Base_Trade_ij'].sum().reset_index()
    trade_i_total.rename(columns={'Base_Trade_ij': 'Base_Trade_i_Total'}, inplace=True)
    print(f"        {len(trade_i_total)} 个出口国")
    
    # 5. 合并并计算权重 w_ij = (i→j贸易额) / (i→全球总出口额)
    print(f"\n[步骤4] 计算权重矩阵 w_ij = Base_Trade_ij / Base_Trade_i_Total...")
    weight_matrix = pd.merge(trade_ij, trade_i_total, on=ref_area_col, how='left')
    weight_matrix['w_ij'] = weight_matrix['Base_Trade_ij'] / weight_matrix['Base_Trade_i_Total']
    
    # 处理可能的 NaN (分母为0)
    nan_count = weight_matrix['w_ij'].isna().sum()
    weight_matrix['w_ij'] = weight_matrix['w_ij'].fillna(0)
    
    if nan_count > 0:
        print(f"⚠️  发现 {nan_count} 个 NaN 权重值（分母为0），已填充为 0")
    
    print(f"\n✓ 权重矩阵生成完成:")
    print(f"   形状: {weight_matrix.shape}")
    print(f"   权重范围: [{weight_matrix['w_ij'].min():.6f}, {weight_matrix['w_ij'].max():.6f}]")
    print(f"   权重和校验 (每个出口国应为1): min={weight_matrix.groupby(ref_area_col)['w_ij'].sum().min():.6f}, "
          f"max={weight_matrix.groupby(ref_area_col)['w_ij'].sum().max():.6f}")
    
    print(f"\n====================== 权重矩阵生成完成 ======================\n")
    
    return weight_matrix


# ====== plotting utilities ======

def plot_weight_distribution(weight_matrix):
    """绘制 w_ij 分布的直方图 + KDE（对数 X 轴）。"""
    if 'w_ij' not in weight_matrix.columns:
        print("⚠️ 数据框中缺少 'w_ij' 列，无法绘图")
        return
    data = weight_matrix[['w_ij']].copy()
    data = data[data['w_ij'] > 0]  # 去除零值

    if data.empty:
        print("⚠️ 所有权重均为0，无法绘制分布图")
        return

    median_val = data['w_ij'].median()

    plt.figure(figsize=(8, 5))
    ax = sns.histplot(data=data, x='w_ij', bins=50, kde=True, color='steelblue', edgecolor='black')
    ax.set_xscale('log')
    ax.axvline(median_val, color='red', linestyle='--', label=f'Median = {median_val:.2e}')

    ax.set_title('weigt (w_ij) Distribution (Log Scale)')
    ax.set_xlabel('双边贸易权重 w_ij (对数刻度)')
    ax.set_ylabel('频数 / 密度')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_core_edge_heatmap(weight_matrix, core_n=15, edge_n=5, edge_fixed=None, random_state=42):
    """绘制核心-边缘国家热力图。"""
    # 1. 找到基期总出口额排名前 core_n 的国家
    if 'Base_Trade_i_Total' not in weight_matrix.columns:
        print("⚠️ 数据框中缺少 'Base_Trade_i_Total' 列，无法绘图")
        return
    trade_totals = weight_matrix[['REF_AREA', 'Base_Trade_i_Total']].drop_duplicates()
    trade_totals = trade_totals.sort_values('Base_Trade_i_Total', ascending=False)
    cores = trade_totals['REF_AREA'].head(core_n).tolist()

    # 2. 选取边缘国家
    remaining = trade_totals[~trade_totals['REF_AREA'].isin(cores)]['REF_AREA'].tolist()
    if edge_fixed:
        # 保证固定小国存在
        for code in edge_fixed:
            if code in remaining and code not in cores:
                remaining.remove(code)
        sampled = []
        # 添加指定的固定小国
        for code in edge_fixed:
            if code not in sampled:
                sampled.append(code)
        # 还需随机选择的数量
        remaining_n = edge_n - len(sampled)
        if remaining_n > 0:
            np.random.seed(random_state)
            sampled += list(np.random.choice(remaining, size=min(remaining_n, len(remaining)), replace=False))
        edges = sampled
    else:
        np.random.seed(random_state)
        edges = list(np.random.choice(remaining, size=min(edge_n, len(remaining)), replace=False))

    selected = cores + edges

    # 3. 构建子集
    sub = weight_matrix[weight_matrix['REF_AREA'].isin(selected) & weight_matrix['COUNTERPART_AREA'].isin(selected)]

    # 4. 数据透视
    pivot = sub.pivot(index='REF_AREA', columns='COUNTERPART_AREA', values='w_ij').fillna(0)

    # 5. 绘制热力图
    plt.figure(figsize=(12, 10))
    # 将 0 替换为 NaN 以便 LogNorm 处理
    plot_matrix = pivot.replace(0, np.nan)
    sns.heatmap(plot_matrix, norm=LogNorm(vmin=1e-5, vmax=1), cmap='YlOrRd', 
                cbar_kws={'label': 'w_ij (log scale)'},
                square=True, linewidths=0.5, linecolor='gray',
                mask=plot_matrix.isna())
    plt.title('Core-Periphery Digital Trade Weight Submatrix Heatmap')
    plt.ylabel('Exporter (REF_AREA)')
    plt.xlabel('Importer (COUNTERPART_AREA)')

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# global placeholder for cleaned dataset
cleaned_batis_df = None

# default file path used when no path is supplied to loader
DEFAULT_DATA_FILE = r"C:\Users\29106\Downloads\OECD_BaTIS_Data3.csv"


def get_cleaned_batis_df(data_file=None, **kwargs):
    """返回过滤掉聚合体后的数据框。

    如果已经加载过同一个数据，会直接返回缓存的版本。
    参数"kwargs"会直接传给 clean_trade_data，例如可指定列名。
    """
    global cleaned_batis_df
    if cleaned_batis_df is None:
        if data_file is None:
            data_file = DEFAULT_DATA_FILE
        cleaned_batis_df = clean_trade_data(data_file, **kwargs)
    return cleaned_batis_df


# ====== 主执行流程 ======
if __name__ == "__main__":
    # 配置文件路径
    data_file = r"C:\Users\29106\Downloads\OECD_BaTIS_Data3.csv"
    
    print("\n" + "="*70)
    print("贸易网络数据处理流程（包含聚合体过滤）")
    print("="*70)
    
    # 步骤1：诊断原始数据
    print("\n【步骤1】诊断原始数据结构")
    print("-" * 70)
    inspect_trade_data(data_file)
    
    # 步骤2：清洗并过滤非国家聚合体
    print("\n【步骤2】数据清洗与聚合体过滤")
    print("-" * 70)
    df_clean = clean_trade_data(data_file)
    
    if df_clean is None:
        print("❌ 数据清洗失败，程序终止")
    else:
        print(f"✓ 清洁数据已生成，形状: {df_clean.shape}")
        
        # 步骤3：生成静态权重矩阵
        print("\n【步骤3】生成静态权重矩阵")
        print("-" * 70)
        weight_matrix = build_batis_static_weights(
            df_clean,
            ref_area_col='REF_AREA',
            counterpart_col='COUNTERPART_AREA',
            value_col='OBS_VALUE',
            time_col='TIME_PERIOD',
            base_years=[2005, 2006, 2007]
        )
        
        # 步骤4：导出清洁数据
        print("\n【步骤4】导出清洁数据")
        print("-" * 70)
        output_dir = os.path.dirname(data_file)
        cleaned_output_file = os.path.join(output_dir, "cleaned_batis_df.csv")
        df_clean.to_csv(cleaned_output_file, index=False, encoding='utf-8')
        print(f"✓ 清洁数据已保存至: {cleaned_output_file}")
        print(f"  (共 {df_clean.shape[0]} 行, {df_clean.shape[1]} 列)")
        
        if weight_matrix is not None:
            print("\n✓ 完整流程执行成功！")
            print(f"\n权重矩阵预览（前10行）:")
            print(weight_matrix.head(10))
            
             # 微调1：只提取需要的3列，并把 'w_ij' 改名为 'weight'
            export_df = weight_matrix[['REF_AREA', 'COUNTERPART_AREA', 'w_ij']].copy()
            export_df.rename(columns={'w_ij': 'weight'}, inplace=True)
            
            # 微调2：更改输出文件名为 static_weights.csv
            output_file = "static_weights.csv" 
            export_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\n✓ 权重矩阵已保存至: {output_file}，可以发给Copilot使用了！")
        else:
            print("❌ 权重矩阵生成失败")

        # 假设你已经调用函数生成了 weight_matrix

