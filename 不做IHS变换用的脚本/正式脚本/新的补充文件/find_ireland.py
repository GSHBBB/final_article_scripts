"""
直接找出爱尔兰1.44T的来源
"""
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(r"c:\Users\29106\OneDrive\文档\毕业论文\不做IHS变换用的脚本\正式脚本\输入文件")
OECD_FILE = INPUT_DIR / 'input_OECD_BaTIS_Data3.csv'

# 读取指定列，避免行数过多导致内存溢出
print("读取 CSV 文件...")
usecols = ['Reference area', 'TRADE_FLOW', 'SERVICE', 'TIME_PERIOD', 'OBS_VALUE']
df = pd.read_csv(OECD_FILE, usecols=usecols)
print(f"总行数: {len(df)}")

# 只保留出口（X）流向
df_export = df[df['TRADE_FLOW'] == 'X'].copy()
print(f"导出行数: {len(df_export)}")

# 按最新年份分组
print("\n数据年份范围:")
print(f"最小年份: {df_export['TIME_PERIOD'].min()}")
print(f"最大年份: {df_export['TIME_PERIOD'].max()}")

# 查找爱尔兰的所有记录
ireland_all = df_export[df_export['Reference area'] == 'Ireland'].copy()
print(f"\n爱尔兰的总记录数: {len(ireland_all)}")
print(f"爱尔兰涉及的年份: {sorted(ireland_all['TIME_PERIOD'].unique())}")

# 查看最新年份
max_year = int(df_export['TIME_PERIOD'].max())
ireland_latest = ireland_all[ireland_all['TIME_PERIOD'] == max_year]
print(f"\n爱尔兰在 {max_year} 年的记录数: {len(ireland_latest)}")

if len(ireland_latest) == 0:
    print(f"爱尔兰在 {max_year} 年没有数据！")
    # 查看最新年份爱尔兰数据
    latest_ireland_year = ireland_all['TIME_PERIOD'].max()
    print(f"爱尔兰的最新数据年份是: {latest_ireland_year}")
    ireland_latest = ireland_all[ireland_all['TIME_PERIOD'] == latest_ireland_year]
    print(f"爱尔兰在 {latest_ireland_year} 年的数据行数: {len(ireland_latest)}")
    
# 按 SERVICE 分类
print(f"\n按 SERVICE 分类（年份={max_year if len(ireland_latest)>0 else latest_ireland_year}）:")
service_grouped = ireland_latest.groupby('SERVICE')['OBS_VALUE'].agg(['count', 'sum'])
print(service_grouped)

# 总和
total = ireland_latest['OBS_VALUE'].sum()
print(f"\n爱尔兰总计（所有 SERVICE）: {total:,.2f} 百万美元 = {total/1000000:,.2f} 万亿美元")

# 查看具体的 SERVICE 代码
print("\n具体的 SERVICE 代码和数值:")
print(ireland_latest[['SERVICE', 'OBS_VALUE']].sort_values('OBS_VALUE', ascending=False))

# 查看是否有 SI（信息通信服务）
ireland_si = ireland_latest[ireland_latest['SERVICE'] == 'SI']
print(f"\nSI（信息通信服务）记录数: {len(ireland_si)}")
if len(ireland_si) > 0:
    print(f"SI 总额: {ireland_si['OBS_VALUE'].sum():,.2f} 百万美元")

# 查看 2023 年爱尔兰的数据
ireland_2023 = ireland_all[ireland_all['TIME_PERIOD'] == 2023]
print(f"\n爱尔兰在 2023 年的记录数: {len(ireland_2023)}")
print(f"爱尔兰在 2023 年的总出口额: {ireland_2023['OBS_VALUE'].sum():,.2f} 百万美元")
print("2023 年按 SERVICE:")
print(ireland_2023.groupby('SERVICE')['OBS_VALUE'].agg(['count', 'sum']))

# 查看全球 SI 前 10
print("\n全球 SI 前 10（最新年份）:")
si_latest = df_export[
    (df_export['SERVICE'] == 'SI') & 
    (df_export['TIME_PERIOD'] == max_year)
]
si_grouped = si_latest.groupby('Reference area')['OBS_VALUE'].sum().sort_values(ascending=False)
print(si_grouped.head(10))

# 查看全球所有 SERVICE 前 10
print("\n全球所有 SERVICE 前 10（最新年份）:")
all_latest = df_export[df_export['TIME_PERIOD'] == max_year]
all_grouped = all_latest.groupby('Reference area')['OBS_VALUE'].sum().sort_values(ascending=False)
print(all_grouped.head(10))
