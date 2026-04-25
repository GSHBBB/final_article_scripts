"""
改进诊断：查看最新年份的数据
"""
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(r"c:\Users\29106\OneDrive\文档\毕业论文\不做IHS变换用的脚本\正式脚本\输入文件")
OECD_FILE = INPUT_DIR / 'input_OECD_BaTIS_Data3.csv'

df = pd.read_csv(OECD_FILE)  # 读取全部数据

# 定义六大数字服务行业
six_sectors = ['SF', 'SG', 'SH', 'SI', 'SJ', 'SK']

print("=" * 80)
print("1. 数据中的最新年份")
print("=" * 80)
max_year = df['TIME_PERIOD'].max()
print(f"最新年份: {max_year}")

print("\n" + "=" * 80)
print(f"2. 爱尔兰在 {max_year} 年的数据（按 Reference area 搜索）")
print("=" * 80)
ireland_latest = df[(df['Reference area'] == 'Ireland') & 
                     (df['TIME_PERIOD'] == max_year)]
print(f"数据行数: {len(ireland_latest)}")

if len(ireland_latest) > 0:
    print("\n按 SERVICE 分类的统计：")
    by_service = ireland_latest.groupby('SERVICE')['OBS_VALUE'].agg(['count', 'sum'])
    print(by_service)
    
    print(f"\n爱尔兰 {max_year} 年总出口（全部 SERVICE）: {ireland_latest['OBS_VALUE'].sum():,.2f} 百万美元")
    
    # 检查六大行业
    ireland_six = ireland_latest[ireland_latest['SERVICE'].isin(six_sectors)]
    print(f"爱尔兰 {max_year} 年六大行业出口: {ireland_six['OBS_VALUE'].sum():,.2f} 百万美元")
    
    # 检查是否有其他 SERVICE 代码
    other_services = ireland_latest[~ireland_latest['SERVICE'].isin(six_sectors)]
    if len(other_services) > 0:
        print(f"\n存在其他 SERVICE 代码（非六大行业）:")
        print(other_services[['SERVICE', 'OBS_VALUE']].drop_duplicates())

print("\n" + "=" * 80)
print("3. 全球 Top 10 国家在最新年份（仅 SI）")
print("=" * 80)
df_export = df[df['TRADE_FLOW'] == 'X']
si_latest = df_export[(df_export['SERVICE'] == 'SI') & 
                       (df_export['TIME_PERIOD'] == max_year)]
si_by_country = si_latest.groupby('Reference area')['OBS_VALUE'].sum().sort_values(ascending=False)
print(si_by_country.head(10))

print("\n" + "=" * 80)
print("4. 全球 Top 10 国家在最新年份（六大行业）")
print("=" * 80)
six_latest = df_export[(df_export['SERVICE'].isin(six_sectors)) & 
                        (df_export['TIME_PERIOD'] == max_year)]
six_by_country = six_latest.groupby('Reference area')['OBS_VALUE'].sum().sort_values(ascending=False)
print(six_by_country.head(10))

print("\n" + "=" * 80)
print("5. 爱尔兰在各年份的出口额变化（仅 SI）")
print("=" * 80)
ireland_si = df_export[(df_export['Reference area'] == 'Ireland') & 
                        (df_export['SERVICE'] == 'SI')]
ireland_si_by_year = ireland_si.groupby('TIME_PERIOD')['OBS_VALUE'].sum().sort_index()
print(ireland_si_by_year)
