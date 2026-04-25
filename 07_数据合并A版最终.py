import pandas as pd
import numpy as np
import os

# 设置工作目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, '输入文件')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '输出文件')

# 1. 构建核心样本池（Inner Join）
print("步骤1: 构建核心样本池")

# 读取 exposure_panel.csv
exposure_file = os.path.join(OUTPUT_DIR, "output_exposure_panel.csv")
exposure_df = pd.read_csv(exposure_file)
print(f"exposure_panel.csv 维度: {exposure_df.shape}")
print(exposure_df.head())

# 读取 final_resilience_panel.csv（已有表头）
resilience_file = os.path.join(OUTPUT_DIR, "output_final_resilience_panel.csv")
resilience_df = pd.read_csv(resilience_file)
# 只保留需要的列：TIME_PERIOD, REF_AREA, True_Resilience
resilience_df = resilience_df[['TIME_PERIOD', 'REF_AREA', 'True_Resilience']]
# 确保数据类型一致：TIME_PERIOD 转为 int64
resilience_df['TIME_PERIOD'] = resilience_df['TIME_PERIOD'].astype(int)
print(f"final_resilience_panel.csv 维度: {resilience_df.shape}")
print(resilience_df.head())

# 内连接构建 base_panel
base_panel = pd.merge(exposure_df, resilience_df, on=['REF_AREA', 'TIME_PERIOD'], how='inner')
print(f"base_panel 维度: {base_panel.shape}")
print(base_panel.head())

# 2. 清洗并合并常规控制变量（Left Join）
print("\n步骤2: 清洗并合并常规控制变量")

def load_wb_wide(file_path, value_name):
    """加载世界银行宽表数据，跳过元数据行，melt为长表"""
    # 世界银行标准格式：前4行是元数据，从第5行开始是表头和数据
    # 跳过前4行，读取表头
    df = pd.read_csv(file_path, skiprows=4)
    # 提取年份列（2005-2024）
    year_cols = [col for col in df.columns if col.isdigit() and 2005 <= int(col) <= 2024]
    # melt 操作
    melted = df.melt(id_vars=['Country Code'], value_vars=year_cols, var_name='TIME_PERIOD', value_name=value_name)
    melted = melted.rename(columns={'Country Code': 'REF_AREA'})
    melted['TIME_PERIOD'] = melted['TIME_PERIOD'].astype(int)
    # 清理缺失值和空字符
    melted[value_name] = pd.to_numeric(melted[value_name], errors='coerce')
    melted = melted.dropna(subset=[value_name])
    return melted

# 读取并处理三个控制变量文件
fdi_file = os.path.join(INPUT_DIR, "input_WB_FDI.csv")
fdi_df = load_wb_wide(fdi_file, 'FDI')
print(f"FDI 数据维度: {fdi_df.shape}")

gdp_file = os.path.join(INPUT_DIR, "input_WB_GDPer.csv")
gdp_df = load_wb_wide(gdp_file, 'GDP_PC')
print(f"GDP_PC 数据维度: {gdp_df.shape}")

internet_file = os.path.join(INPUT_DIR, "input_WDI_individuals_using_the_Internet.csv")
internet_df = load_wb_wide(internet_file, 'Internet_Use')
print(f"Internet_Use 数据维度: {internet_df.shape}")

# Left Join 到 base_panel
base_panel = pd.merge(base_panel, fdi_df, on=['REF_AREA', 'TIME_PERIOD'], how='left')
base_panel = pd.merge(base_panel, gdp_df, on=['REF_AREA', 'TIME_PERIOD'], how='left')
base_panel = pd.merge(base_panel, internet_df, on=['REF_AREA', 'TIME_PERIOD'], how='left')
print(f"合并控制变量后 base_panel 维度: {base_panel.shape}")
print(base_panel.head())

# 3. 处理复杂的 WGI 制度环境变量
print("\n步骤3: 处理 WGI 制度环境变量")

wgi_file = os.path.join(INPUT_DIR, "input_WB_WGI_WIDEF.csv")
wgi_df = pd.read_csv(wgi_file)

# WGI 数据结构：REF_AREA 为国家码，INDICATOR 为指标码
print(f"WGI 原始数据维度: {wgi_df.shape}")
print(f"WGI 列名：{wgi_df.columns.tolist()[:10]}")

# 过滤仅保留 _EST 指标（估计值）
wgi_df = wgi_df[wgi_df['INDICATOR'].str.contains('_EST', na=False, case=False)]
print(f"WGI 过滤 _EST 指标后维度: {wgi_df.shape}")

# 提取年份列（2005-2024）
year_cols = [col for col in wgi_df.columns if col.isdigit() and 2005 <= int(col) <= 2024]
print(f"提取的年份列: {year_cols}")

# Melt 操作：转为长表
melted_wgi = wgi_df.melt(id_vars=['REF_AREA', 'INDICATOR'], value_vars=year_cols, var_name='TIME_PERIOD', value_name='Value')
melted_wgi['TIME_PERIOD'] = melted_wgi['TIME_PERIOD'].astype(int)
melted_wgi = melted_wgi.dropna(subset=['Value'])
# 转换 Value 为 float
melted_wgi['Value'] = pd.to_numeric(melted_wgi['Value'], errors='coerce')
melted_wgi = melted_wgi.dropna(subset=['Value'])
print(f"melt 后 WGI 数据维度: {melted_wgi.shape}")

# Pivot 操作：将 Indicator 横向展开
pivoted_wgi = melted_wgi.pivot_table(index=['REF_AREA', 'TIME_PERIOD'], columns='INDICATOR', values='Value', aggfunc='first').reset_index()
# 重命名列，去掉前缀 WB_WGI_，去掉后缀 _EST
pivoted_wgi.columns.name = None
old_cols = pivoted_wgi.columns.tolist()
new_cols = []
for col in old_cols:
    if col in ['REF_AREA', 'TIME_PERIOD']:
        new_cols.append(col)
    else:
        # 去掉 WB_WGI_ 前缀和 _EST 后缀
        col_name = col.replace('WB_WGI_', '').replace('_EST', '')
        new_cols.append(col_name)
pivoted_wgi.columns = new_cols
print(f"WGI pivot 后维度: {pivoted_wgi.shape}")
print(f"WGI 指标列: {[col for col in pivoted_wgi.columns if col not in ['REF_AREA', 'TIME_PERIOD']]}")

# Left Join 到 base_panel
base_panel = pd.merge(base_panel, pivoted_wgi, on=['REF_AREA', 'TIME_PERIOD'], how='left')
print(f"合并 WGI 后 base_panel 维度: {base_panel.shape}")
print(base_panel.head())

# 4. 导出最终数据
print("\n步骤4: 导出最终数据")

print(f"最终面板维度: {base_panel.shape}")

# 检查重复的 (REF_AREA, TIME_PERIOD) 组合
duplicates = base_panel.duplicated(subset=['REF_AREA', 'TIME_PERIOD'], keep=False)
if duplicates.any():
    raise ValueError("检测到重复的 (REF_AREA, TIME_PERIOD) 组合，请检查数据")

# 导出为 Master_Panel.csv
output_file = os.path.join(OUTPUT_DIR, "output_Master_Panel.csv")
base_panel.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"最终面板已导出到: {output_file}")
