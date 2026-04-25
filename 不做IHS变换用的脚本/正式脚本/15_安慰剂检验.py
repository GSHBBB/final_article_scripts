import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

warnings.filterwarnings('ignore')


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, '输入文件')
SOURCE_DIR = os.path.join(SCRIPT_DIR, '输出文件')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '输出文件_不强制InternetUse')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASTER_CANDIDATES = [
    os.path.join(OUTPUT_DIR, 'output_Master_Panel.csv'),
    os.path.join(SOURCE_DIR, 'output_Master_Panel.csv'),
    os.path.join(INPUT_DIR, 'input_Master_Panel.csv'),
]

master_path = next((p for p in MASTER_CANDIDATES if os.path.exists(p)), None)
if master_path is None:
    raise FileNotFoundError('未找到 Master_Panel 数据，请先运行正式脚本 07 生成 output_Master_Panel.csv')

print('=' * 80)
print('安慰剂检验（Placebo Test）')
print('=' * 80)
print(f'数据来源: {master_path}')

df = pd.read_csv(master_path)

if 'Out_Degree_Centrality' not in df.columns:
    resilience_path = os.path.join(OUTPUT_DIR, 'output_final_resilience_panel.csv')
    if not os.path.exists(resilience_path):
        resilience_path = os.path.join(SOURCE_DIR, 'output_final_resilience_panel.csv')
    if os.path.exists(resilience_path):
        df_res = pd.read_csv(resilience_path)
        if {'TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality'}.issubset(df_res.columns):
            df = df.merge(
                df_res[['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']].drop_duplicates(),
                on=['TIME_PERIOD', 'REF_AREA'],
                how='left'
            )

required = ['REF_AREA', 'TIME_PERIOD', 'True_Resilience', 'Exposure', 'GDP_PC', 'FDI', 'Out_Degree_Centrality']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f'缺失必要列: {missing}')

df = df[required].dropna().copy()
df['ln_Resilience_Inv'] = -np.log(df['True_Resilience'])
df['log_GDP_PC'] = np.log(df['GDP_PC'])

df = df.set_index(['REF_AREA', 'TIME_PERIOD']).sort_index()
df['L1_Exposure'] = df.groupby(level='REF_AREA')['Exposure'].shift(1)
df['L1_Centrality'] = df.groupby(level='REF_AREA')['Out_Degree_Centrality'].shift(1)

df = df.dropna(subset=['ln_Resilience_Inv', 'L1_Exposure', 'L1_Centrality']).copy()
q33 = df['L1_Centrality'].quantile(0.33)
df_bottom = df[df['L1_Centrality'] <= q33].copy()
df_bottom['Constant'] = 1.0

print(f'Bottom 33% 子样本量: {len(df_bottom)}')
print(f'国家数: {df_bottom.index.get_level_values(0).nunique()}')
print(f'年份范围: {df_bottom.index.get_level_values(1).min()} - {df_bottom.index.get_level_values(1).max()}')

# 真实系数（边缘组）
y = df_bottom['ln_Resilience_Inv']
X_true = df_bottom[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
res_true = PanelOLS(y, X_true, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
true_coef = float(res_true.params['L1_Exposure'])
true_p = float(res_true.pvalues['L1_Exposure'])
print(f'真实 L1_Exposure 系数: {true_coef:.6f} | p值: {true_p:.6f}')

# 安慰剂检验：随机打乱 L1_Exposure
n_iter = 500
rng = np.random.default_rng(20260413)
placebo_coefs = []

for _ in range(n_iter):
    shuffled = rng.permutation(df_bottom['L1_Exposure'].values)
    X_pl = df_bottom[['Constant', 'log_GDP_PC', 'FDI']].copy()
    X_pl['Placebo_Exposure'] = shuffled
    X_pl = X_pl[['Constant', 'Placebo_Exposure', 'log_GDP_PC', 'FDI']]

    try:
        # 安慰剂只关心系数分布，使用 unadjusted 提升速度
        res_pl = PanelOLS(y, X_pl, entity_effects=True, time_effects=True).fit(cov_type='unadjusted')
        coef = float(res_pl.params['Placebo_Exposure'])
        if np.isfinite(coef):
            placebo_coefs.append(coef)
    except Exception:
        continue

if len(placebo_coefs) < 100:
    raise RuntimeError(f'安慰剂有效回归次数过少: {len(placebo_coefs)}')

coefs = np.array(placebo_coefs)
empirical_p = float(np.mean(np.abs(coefs) >= abs(true_coef)))

print(f'安慰剂有效回归次数: {len(coefs)}')
print(f'安慰剂均值: {coefs.mean():.6f}, 标准差: {coefs.std():.6f}')
print(f'经验p值 P(|beta_placebo|>=|beta_true|): {empirical_p:.6f}')

summary_df = pd.DataFrame([
    {
        'n_obs_bottom33': int(len(df_bottom)),
        'n_iter_requested': int(n_iter),
        'n_iter_valid': int(len(coefs)),
        'true_coef_L1_Exposure': true_coef,
        'true_pvalue_L1_Exposure': true_p,
        'placebo_coef_mean': float(coefs.mean()),
        'placebo_coef_std': float(coefs.std()),
        'empirical_pvalue': empirical_p,
        'placebo_pass_empirical_p_lt_0_10': bool(empirical_p < 0.10),
    }
])

summary_path = os.path.join(OUTPUT_DIR, 'output_安慰剂检验结果.csv')
detail_path = os.path.join(OUTPUT_DIR, 'output_安慰剂系数分布.csv')
plot_path = os.path.join(OUTPUT_DIR, 'output_安慰剂检验分布图.png')

summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
pd.DataFrame({'placebo_coef': coefs}).to_csv(detail_path, index=False, encoding='utf-8-sig')

plt.figure(figsize=(18.96, 10), dpi=216)
plt.hist(coefs, bins=40, edgecolor='black', alpha=0.75)
plt.axvline(true_coef, color='red', linestyle='--', linewidth=2, label=f'True beta={true_coef:.4f}')
plt.title('安慰剂检验：随机敞口系数分布', fontsize=16, fontweight='bold')
plt.xlabel('Placebo Coefficient', fontsize=13)
plt.ylabel('Frequency', fontsize=13)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(plot_path, dpi=216, bbox_inches='tight', facecolor='white')

print(f'结果文件: {summary_path}')
print(f'系数分布: {detail_path}')
print(f'图表文件: {plot_path}')
print('=' * 80)
