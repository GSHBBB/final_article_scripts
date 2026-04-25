"""
诊断脚本：检查 BaTIS 中的 SERVICE 代码、重复计算情况
"""
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(r"c:\Users\29106\OneDrive\文档\毕业论文\不做IHS变换用的脚本\正式脚本\输入文件")
OECD_FILE = INPUT_DIR / 'input_OECD_BaTIS_Data3.csv'

# 读取一小部分数据查看结构
df = pd.read_csv(OECD_FILE, nrows=50000)

print("=" * 80)
print("1. SERVICE 代码分布统计")
print("=" * 80)
service_counts = df['SERVICE'].value_counts()
print(service_counts)
print(f"\n总共 {len(service_counts)} 种 SERVICE 代码\n")

print("=" * 80)
print("2. 检查是否有母类代码（如'S'、'SXX'等总类）")
print("=" * 80)
all_services = df['SERVICE'].dropna().unique()
print("所有 SERVICE 代码:")
print(sorted(all_services))

print("\n" + "=" * 80)
print("3. 对于爱尔兰（IRL）在 2024 年的数据，按 SERVICE 分类统计")
print("=" * 80)
ireland_2024 = df[(df['Reference area'] == 'Ireland') & (df['TIME_PERIOD'] == 2024)]
print(f"爱尔兰 2024 年数据行数: {len(ireland_2024)}")
ireland_by_service = ireland_2024.groupby('SERVICE')['OBS_VALUE'].agg(['count', 'sum'])
print("\n按 SERVICE 分类的统计：")
print(ireland_by_service)

print("\n" + "=" * 80)
print("4. 检查是否有多个方法论代码（Methodology）")
print("=" * 80)
if 'METHODOLOGY_TYPE' in df.columns or 'Methodology type' in df.columns:
    meth_col = 'METHODOLOGY_TYPE' if 'METHODOLOGY_TYPE' in df.columns else 'Methodology type'
    ireland_methods = ireland_2024.groupby(meth_col).size()
    print(f"爱尔兰 2024 年的方法论类型：")
    print(ireland_methods)
else:
    print("未找到 METHODOLOGY_TYPE 列")

print("\n" + "=" * 80)
print("5. 检查爱尔兰 2024 年是否有重复的（REF_AREA, COUNTERPART_AREA, SERVICE）组合")
print("=" * 80)
drop_cols = [c for c in ['COUNTERPART_AREA', 'Counterpart area'] if c in ireland_2024.columns]
if drop_cols:
    cp_col = drop_cols[0]
    ireland_dedup = ireland_2024.groupby(['Reference area', cp_col, 'SERVICE']).size()
    duplicates = ireland_dedup[ireland_dedup > 1]
    if len(duplicates) > 0:
        print(f"发现 {len(duplicates)} 组重复记录：")
        print(duplicates)
    else:
        print("未发现重复的 (REF_AREA, COUNTERPART_AREA, SERVICE) 组合")

print("\n" + "=" * 80)
print("6. 计算包含六大细分行业（SF/SG/SH/SI/SJ/SK）的爱尔兰 2024 年总出口")
print("=" * 80)
six_sectors = ['SF', 'SG', 'SH', 'SI', 'SJ', 'SK']
ireland_six = df[(df['Reference area'] == 'Ireland') & 
                  (df['TIME_PERIOD'] == 2024) & 
                  (df['SERVICE'].isin(six_sectors)) &
                  (df['TRADE_FLOW'] == 'X')]
ireland_six_total = ireland_six['OBS_VALUE'].sum()
print(f"爱尔兰 2024 年六大数字服务行业出口总额: {ireland_six_total:,.2f} 百万美元")

print("\n按行业分类：")
by_sector = ireland_six.groupby('SERVICE')['OBS_VALUE'].sum().sort_values(ascending=False)
print(by_sector)

print("\n" + "=" * 80)
print("7. 对比：仅 SI 与包含六行业的差异")
print("=" * 80)
ireland_si = df[(df['Reference area'] == 'Ireland') & 
                (df['TIME_PERIOD'] == 2024) & 
                (df['SERVICE'] == 'SI') &
                (df['TRADE_FLOW'] == 'X')]
ireland_si_total = ireland_si['OBS_VALUE'].sum()
print(f"爱尔兰 2024 年仅 SI: {ireland_si_total:,.2f} 百万美元")
print(f"爱尔兰 2024 年六行业: {ireland_six_total:,.2f} 百万美元")
print(f"两倍关系: {ireland_six_total / ireland_si_total:.2f}x")
