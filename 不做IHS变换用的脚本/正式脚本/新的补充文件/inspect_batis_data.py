
"""
探查 OECD BaTIS 输入文件的结构和内容
目的：搞清楚列名、聚合体情况、counterpart 编码方式等
"""
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(__file__).resolve().parents[1] / '输入文件' / 'input_OECD_BaTIS_Data3.csv'

print("=" * 70)
print("1. 基本信息")
print("=" * 70)
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"行数: {len(df)}")
print(f"列数: {len(df.columns)}")
print(f"\n所有列名:\n{list(df.columns)}")
print(f"\n前5行:")
print(df.head().to_string())

print("\n" + "=" * 70)
print("2. 各列数据类型和唯一值数量")
print("=" * 70)
for col in df.columns:
    print(f"  {col:30s} | dtype={str(df[col].dtype):10s} | nunique={df[col].nunique()}")

print("\n" + "=" * 70)
print("3. REF_AREA（出口国代码）唯一值")
print("=" * 70)
ref_areas = sorted(df['REF_AREA'].dropna().astype(str).str.strip().unique())
print(f"共 {len(ref_areas)} 个: {ref_areas}")

print("\n" + "=" * 70)
print("4. Counterpart area 相关列")
print("=" * 70)
# 检查是否有 counterpart 代码列
counter_cols = [c for c in df.columns if 'counterpart' in c.lower() or 'partner' in c.lower()]
print(f"Counterpart 相关列: {counter_cols}")
for col in counter_cols:
    vals = sorted(df[col].dropna().astype(str).str.strip().unique())
    print(f"\n  {col} 唯一值 ({len(vals)} 个):")
    for v in vals:
        print(f"    {v}")

print("\n" + "=" * 70)
print("5. TRADE_FLOW 唯一值")
print("=" * 70)
print(df['TRADE_FLOW'].value_counts())

print("\n" + "=" * 70)
print("6. SERVICE 唯一值")
print("=" * 70)
print(df['SERVICE'].value_counts().to_string())

print("\n" + "=" * 70)
print("7. METHODOLOGY_TYPE 唯一值")
print("=" * 70)
print(df['METHODOLOGY_TYPE'].value_counts(dropna=False).to_string())

print("\n" + "=" * 70)
print("8. TIME_PERIOD 唯一值")
print("=" * 70)
print(sorted(df['TIME_PERIOD'].dropna().unique()))

print("\n" + "=" * 70)
print("9. 检查重复：同一 (REF_AREA, Counterpart, SERVICE, TIME_PERIOD) 有多少条记录")
print("=" * 70)
# 只看出口
df_x = df[df['TRADE_FLOW'] == 'X'].copy()
dup_counts = df_x.groupby(['REF_AREA', 'Counterpart area', 'SERVICE', 'TIME_PERIOD']).size()
print(f"总组合数: {len(dup_counts)}")
print(f"有重复的组合数: {(dup_counts > 1).sum()}")
print(f"重复分布:\n{dup_counts.value_counts().sort_index()}")

print("\n" + "=" * 70)
print("10. 检查聚合体：REF_AREA 中长度不为3或不像国家代码的值")
print("=" * 70)
ref_str = df['REF_AREA'].astype(str).str.strip()
non_3 = ref_str[ref_str.str.len() != 3].unique()
print(f"长度不为3的 REF_AREA: {sorted(non_3)}")

# 常见聚合体代码
KNOWN_AGGREGATES = {
    'WLD', 'OECD', 'EU27_2020', 'EU28', 'EUU', 'G20', 'G7',
    'ARB', 'CSS', 'CEB', 'EAR', 'EAS', 'EAP', 'EMU', 'ECA', 'ECS',
    'FCS', 'HIC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX',
    'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'MEA', 'MIC', 'MNA',
    'NAC', 'OED', 'OSS', 'PST', 'PRE', 'SAS', 'SSA', 'SSF',
    'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS', 'UMC', 'WXD'
}
found_agg = set(ref_areas) & KNOWN_AGGREGATES
print(f"\nREF_AREA 中发现的已知聚合体: {sorted(found_agg)}")

# 同样检查 counterpart
if 'COUNTERPART_AREA' in df.columns:
    cp_str = df['COUNTERPART_AREA'].astype(str).str.strip()
    cp_agg = set(cp_str.unique()) & KNOWN_AGGREGATES
    print(f"COUNTERPART_AREA 中发现的已知聚合体: {sorted(cp_agg)}")

print("\n" + "=" * 70)
print("11. 抽样：展示几条典型数据")
print("=" * 70)
sample = df_x.head(20)
print(sample.to_string())

print("\n完成！")
