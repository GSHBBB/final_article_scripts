import pandas as pd
import numpy as np
import os

# 设置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, '输入文件')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '输出文件')
output_file = os.path.join(OUTPUT_DIR, "output_controls_panel.csv")

# 清洗WB和WDI数据：melt宽转长
def clean_wb_data(file_path, value_name):
    df = pd.read_csv(file_path, header=2)  # header在第三行
    # 提取年份列：数字年份，2005-2024
    year_cols = [col for col in df.columns if col.isdigit() and 2005 <= int(col) <= 2024]
    # melt
    df_long = df.melt(id_vars=['Country Code'], value_vars=year_cols, var_name='Year', value_name=value_name)
    # 转换Year为int
    df_long['Year'] = df_long['Year'].astype(int)
    # 剔除无效值
    df_long = df_long.dropna(subset=[value_name])
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce')
    df_long = df_long.dropna(subset=[value_name])
    return df_long

# 读取并清洗WB_FDI
fdi_file = os.path.join(INPUT_DIR, "input_WB_FDI.csv")
fdi_df = clean_wb_data(fdi_file, 'FDI')
print("FDI数据清洗完成")
print(fdi_df.head())

# WB_GDP_GROWTH
gdp_file = os.path.join(INPUT_DIR, "input_WB_GDP_GROWTH.csv")
gdp_df = clean_wb_data(gdp_file, 'GDP_Growth')
print("GDP Growth数据清洗完成")
print(gdp_df.head())

# WDI_individuals_using_the_Internet
internet_file = os.path.join(INPUT_DIR, "input_WDI_individuals_using_the_Internet.csv")
internet_df = clean_wb_data(internet_file, 'Internet_Usage')
print("Internet Usage数据清洗完成")
print(internet_df.head())

# 清洗WGI数据
wgi_file = os.path.join(INPUT_DIR, "input_WB_WGI_WIDEF.csv")
wgi_df = pd.read_csv(wgi_file)
# 只保留标识列（INDICATOR）包含_EST的行
wgi_df = wgi_df[wgi_df['INDICATOR'].str.contains('_EST', na=False)]
print("WGI数据过滤完成")
print(wgi_df.head())

# melt
year_cols_wgi = [col for col in wgi_df.columns if col.isdigit() and 2005 <= int(col) <= 2024]
wgi_long = wgi_df.melt(id_vars=['REF_AREA'], value_vars=year_cols_wgi, var_name='Year', value_name='WGI')
wgi_long = wgi_long.rename(columns={'REF_AREA': 'Country Code'})
wgi_long['Year'] = wgi_long['Year'].astype(int)
wgi_long = wgi_long.dropna(subset=['WGI'])
wgi_long['WGI'] = pd.to_numeric(wgi_long['WGI'], errors='coerce')
wgi_long = wgi_long.dropna(subset=['WGI'])
print("WGI数据清洗完成")
print(wgi_long.head())

# 合并控制变量
controls_df = fdi_df.merge(gdp_df, on=['Country Code', 'Year'], how='outer') \
                    .merge(internet_df, on=['Country Code', 'Year'], how='outer') \
                    .merge(wgi_long, on=['Country Code', 'Year'], how='outer')

print("合并完成")
print(controls_df.head())

# 保存
os.makedirs(os.path.dirname(output_file), exist_ok=True)
controls_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"控制变量面板数据已保存到: {output_file}")