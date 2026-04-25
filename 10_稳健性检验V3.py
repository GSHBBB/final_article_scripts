import pandas as pd
import numpy as np
import os
import warnings
from linearmodels.panel import PanelOLS

warnings.filterwarnings('ignore')

print("=" * 90)
print("顶刊标准：稳健性检验 (Robustness Checks)")
print("=" * 90)

# ============================================================================
# 0. 数据加载与基础准备 (与 08_最终回归模型V4 完全一致的数据源和分组)
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, '输出文件')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '输出文件_不强制InternetUse')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def resolve_input(filename):
    preferred = os.path.join(OUTPUT_DIR, filename)
    fallback = os.path.join(SOURCE_DIR, filename)
    return preferred if os.path.exists(preferred) else fallback

data_file = resolve_input("output_Master_Panel.csv")
resilience_file = resolve_input("output_final_resilience_panel.csv")

df_model = pd.read_csv(data_file)
df_resilience = pd.read_csv(resilience_file)

if 'Out_Degree_Centrality' in df_resilience.columns:
    centrality_df = df_resilience[['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']].copy()
    df_model = df_model.merge(centrality_df, on=['REF_AREA', 'TIME_PERIOD'], how='left')

# 与 V4_不强制InternetUse 一致：主回归不强制 Internet_Use 完整
required_cols = ['REF_AREA', 'TIME_PERIOD', 'True_Resilience', 'Exposure', 'GDP_PC', 'FDI', 'Out_Degree_Centrality']
model_cols = required_cols + (['Internet_Use'] if 'Internet_Use' in df_model.columns else [])
df = df_model[model_cols].dropna(subset=required_cols).copy()
df['ln_Resilience_Inv'] = -np.log(df['True_Resilience'])
df['log_GDP_PC'] = np.log(df['GDP_PC'])

df_indexed = df.set_index(['REF_AREA', 'TIME_PERIOD'])
df_indexed['L1_Exposure'] = df_indexed.groupby(level='REF_AREA')['Exposure'].shift(1)
# 与 V4 一致：使用 L1_Centrality（滞后1期）进行分组
df_indexed['L1_Centrality'] = df_indexed.groupby(level='REF_AREA')['Out_Degree_Centrality'].shift(1)
df_clean = df_indexed.dropna(subset=['L1_Exposure', 'L1_Centrality', 'ln_Resilience_Inv']).copy()
df_clean['Constant'] = 1.0

# 与 V4 一致：按 L1_Centrality 的 33% 分位数分组
q33 = df_clean['L1_Centrality'].quantile(0.33)
df_peri = df_clean[df_clean['L1_Centrality'] <= q33].copy()
print(f"数据源: {data_file}")
print(f"边缘组样本量 (Bottom 33% by L1_Centrality): {len(df_peri)}")

# ============================================================================
# 稳健性检验 1：剔除重大外部冲击年份 (2008-2010, 2020-2021)
# ============================================================================
print("\n>>> 稳健性检验 1：剔除金融危机与新冠疫情年份样本 (Excluding Crisis Years)")
crisis_years = (2008, 2009, 2010, 2020, 2021)
df_robust_1 = df_peri[~df_peri.index.get_level_values('TIME_PERIOD').isin(crisis_years)]

X_rob1 = df_robust_1[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
y_rob1 = df_robust_1['ln_Resilience_Inv']
res_rob1 = PanelOLS(y_rob1, X_rob1, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(f"L1_Exposure 系数: {res_rob1.params['L1_Exposure']:.6f} | P值: {res_rob1.pvalues['L1_Exposure']:.6f}")
if res_rob1.pvalues['L1_Exposure'] < 0.10: print("[PASS] 检验 1 通过: 剔除极端年份后结果依然稳健!")

# ============================================================================
# 稳健性检验 2：对核心连续变量进行 1% 缩尾处理 (Winsorization)
# ============================================================================
print("\n>>> 稳健性检验 2：双侧 1% 缩尾处理 (Winsorizing at 1% and 99%)")
df_robust_2 = df_peri.copy()
# 对 L1_Exposure 和 ln_Resilience_Inv 进行缩尾
for col in ['L1_Exposure', 'ln_Resilience_Inv']:
    p01 = df_robust_2[col].quantile(0.01)
    p99 = df_robust_2[col].quantile(0.99)
    df_robust_2[col] = df_robust_2[col].clip(lower=p01, upper=p99)

X_rob2 = df_robust_2[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
y_rob2 = df_robust_2['ln_Resilience_Inv']
res_rob2 = PanelOLS(y_rob2, X_rob2, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(f"L1_Exposure 系数: {res_rob2.params['L1_Exposure']:.6f} | P值: {res_rob2.pvalues['L1_Exposure']:.6f}")
if res_rob2.pvalues['L1_Exposure'] < 0.10: print("[PASS] 检验 2 通过: 排除极端值干扰后结果依然稳健!")

# ============================================================================
# 稳健性检验 3：采用更严苛的“国家-年份”双向聚类标准误
# ============================================================================
print("\n>>> 稳健性检验 3：使用国家和年份的双向聚类标准误 (Two-way Clustered SE)")
X_rob3 = df_peri[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
y_rob3 = df_peri['ln_Resilience_Inv']
# 同时在 Entity 和 Time 层面聚类
res_rob3 = PanelOLS(y_rob3, X_rob3, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
print(f"L1_Exposure 系数: {res_rob3.params['L1_Exposure']:.6f} | P值: {res_rob3.pvalues['L1_Exposure']:.6f}")
if res_rob3.pvalues['L1_Exposure'] < 0.10: print("[PASS] 检验 3 通过: 提高标准误门槛后显著性依然成立!")

print("\n" + "=" * 90)
print("学术建议：直接将这三个模型的结果合并成一张《稳健性检验表》放入论文。")
print("并在正文中引用刘采斌(2025)等文献，说明连续DID模型的稳健性检验范式。")
print("=" * 90)