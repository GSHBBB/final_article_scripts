import pandas as pd
import numpy as np
import re
import os

# 设置文件路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, '输入文件')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '输出文件')

weights_file = os.path.join(INPUT_DIR, "input_static_weights.csv")
taped_file = os.path.join(INPUT_DIR, "input_taped_bilateral.xlsx")
output_file = os.path.join(OUTPUT_DIR, "output_exposure_panel.csv")

# 1. 读取基期贸易权重表
print("正在读取基期贸易权重表...")
weights_df = pd.read_csv(weights_file)
# 修复：将weight除以100，确保量纲正确
weights_df['weight'] = weights_df['weight'] / 100
print(f"权重表维度: {weights_df.shape}")
print(weights_df.head())

# 2. 读取TAPED双边化数据库
print("正在读取TAPED数据库...")
taped_df = pd.read_excel(taped_file)
taped_df = taped_df.copy()  # 避免碎片化问题
print(f"TAPED表维度: {taped_df.shape}")
print(taped_df.head())

# 解析parties为Country1和Country2（假设双边，取前两个国家）
def parse_parties(p):
    if pd.isna(p) or not isinstance(p, str):
        return None, None
    p = p.replace(' ', '')
    if ';' in p:
        countries = p.split(';')
    elif ',' in p:
        countries = p.split(',')
    else:
        countries = [p]
    if len(countries) >= 2:
        return countries[0], countries[1]
    else:
        return None, None

taped_df['Country1'], taped_df['Country2'] = zip(*taped_df['parties'].apply(parse_parties))
taped_df = taped_df.dropna(subset=['Country1', 'Country2'])
print("解析缔约国完成")
print(taped_df[['Country1', 'Country2']].head())

# 修复：EU映射展开
eu_countries = ['AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE']
def expand_eu(country):
    if country in ['EU', 'EC']:
        return eu_countries
    else:
        return [country]

taped_df['Country1_expanded'] = taped_df['Country1'].apply(expand_eu)
taped_df['Country2_expanded'] = taped_df['Country2'].apply(expand_eu)
taped_df = taped_df.explode('Country1_expanded').explode('Country2_expanded')
taped_df = taped_df.reset_index(drop=True)
taped_df = taped_df.drop(columns=['Country1', 'Country2'])
taped_df = taped_df.rename(columns={'Country1_expanded': 'Country1', 'Country2_expanded': 'Country2'})
print("EU映射展开完成")
print(taped_df[['Country1', 'Country2']].head())

# 3. 计算规则深度 (Depth Score)
# 定义元数据列（标识列），其余为条款列
metadata_cols = ['taped_number', 'long_title', 'short_title', 'type ', 'type_memb', 'parties', 'status_parties', 'date_signed', 'year_signed', 'date_into_force', 'Country1', 'Country2']
clause_cols = [col for col in taped_df.columns if col not in metadata_cols and pd.api.types.is_numeric_dtype(taped_df[col])]
print(f"条款列数量: {len(clause_cols)}")
print(f"条款列示例: {clause_cols[:10]}")

# 计算Depth_Score
taped_df['Depth_Score'] = taped_df[clause_cols].sum(axis=1)
print("Depth_Score计算完成")
print(taped_df[['Country1', 'Country2', 'Depth_Score']].head())

# 4. 提取生效年份 (Start_Year)
def extract_year(date_str):
    if pd.isna(date_str):
        return np.nan
    match = re.search(r'(\d{4})', str(date_str))
    return int(match.group(1)) if match else np.nan

# 优先从date_into_force提取
taped_df = taped_df.assign(Start_Year = taped_df['date_into_force'].apply(extract_year))
# 如果失败，回退到date_signed
mask = taped_df['Start_Year'].isna()
taped_df = taped_df.assign(Start_Year = taped_df['Start_Year'].where(~mask, taped_df.loc[mask, 'date_signed'].apply(extract_year)))
# 去除无法提取年份的记录
taped_df = taped_df.dropna(subset=['Start_Year'])
taped_df['Start_Year'] = taped_df['Start_Year'].astype(int)
print("Start_Year提取完成")
print(taped_df[['Country1', 'Country2', 'Start_Year']].head())

# 5. 双向展开无向数据
# 保留必要列
taped_bidirectional = taped_df[['Country1', 'Country2', 'Start_Year', 'Depth_Score']].copy()
# 创建反向记录
taped_rev = taped_bidirectional.rename(columns={'Country1': 'Country2', 'Country2': 'Country1'})
taped_full = pd.concat([taped_bidirectional, taped_rev], ignore_index=True)
# 重命名列
taped_full = taped_full.rename(columns={'Country1': 'REF_AREA', 'Country2': 'COUNTERPART_AREA', 'Start_Year': 'year'})
# 修复：当同一年同一对有多个协定，使用max Depth_Score
taped_full = taped_full.groupby(['REF_AREA', 'COUNTERPART_AREA', 'year'])['Depth_Score'].max().reset_index()
print("双向展开完成")
print(taped_full.head())

# 6. 构建连续累积的面板数据
years = list(range(2005, 2025))  # 2005-2024
# 获取所有国家对
pairs = weights_df[['REF_AREA', 'COUNTERPART_AREA']].drop_duplicates()
print(f"国家对数量: {len(pairs)}")

# 创建面板框架
panel = []
for year in years:
    df_year = pairs.copy()
    df_year['TIME_PERIOD'] = year
    # 合并历史协定：t <= year 的记录，取cumulative max Depth_Score
    historical = taped_full[taped_full['year'] <= year].copy()
    if not historical.empty:
        # 对每对国家，取历史最大Depth_Score
        max_depth = historical.groupby(['REF_AREA', 'COUNTERPART_AREA'])['Depth_Score'].max().reset_index()
        df_year = df_year.merge(max_depth, on=['REF_AREA', 'COUNTERPART_AREA'], how='left')
    else:
        df_year['Depth_Score'] = 0
    df_year['Depth_Score'] = df_year['Depth_Score'].fillna(0)
    panel.append(df_year)

panel_df = pd.concat(panel, ignore_index=True)
print("面板数据构建完成")
print(panel_df.head())

# 7. 合并权重并计算最终敞口指数
# 合并权重
panel_df = panel_df.merge(weights_df, on=['REF_AREA', 'COUNTERPART_AREA'], how='left')
# 计算weighted_depth
panel_df['weighted_depth'] = panel_df['weight'] * panel_df['Depth_Score']
# 按REF_AREA和TIME_PERIOD汇总Exposure
exposure_df = panel_df.groupby(['REF_AREA', 'TIME_PERIOD'])['weighted_depth'].sum().reset_index()
exposure_df = exposure_df.rename(columns={'weighted_depth': 'Exposure'})
print("Exposure指数计算完成")
print(exposure_df.head())

# 8. 输出结果
os.makedirs(os.path.dirname(output_file), exist_ok=True)
exposure_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"结果已保存到: {output_file}")

# 9. 可视化: 提取过去25年中暴露指数变化最大的前20名国家
# 这里数据时间从2005到2024，如需25年则假定为这段时间
start_year = min(exposure_df['TIME_PERIOD'])
end_year = max(exposure_df['TIME_PERIOD'])
print(f"分析期间: {start_year} - {end_year}")

# 计算每个国家在期末与期初的Exposure差值
pivot = exposure_df.pivot(index='REF_AREA', columns='TIME_PERIOD', values='Exposure')
# 清理缺失值
pivot = pivot.fillna(0)
# 计算变化
pivot['change'] = pivot[end_year] - pivot[start_year]
# 选取前20名（按增量降序）
top20 = pivot['change'].abs().sort_values(ascending=False).head(20)
print("变化幅度前20的国家:")
print(top20)

# 获取这些国家的时序数据用于绘图
top_countries = top20.index.tolist()
plot_df = exposure_df[exposure_df['REF_AREA'].isin(top_countries)].copy()

# 绘图
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
for country in top_countries:
    subset = plot_df[plot_df['REF_AREA'] == country]
    plt.plot(subset['TIME_PERIOD'], subset['Exposure'], label=country)

plt.xlabel('Year')
plt.ylabel('Exposure Index')
plt.title(f'Top 20 Countries by Exposure Change ({start_year}-{end_year})')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout()

# 将图像保存到文件
fig_path = os.path.join(OUTPUT_DIR, 'output_exposure_top20_trends.png')
plt.savefig(fig_path, dpi=300)
print(f"趋势图已保存到: {fig_path}")
plt.show()