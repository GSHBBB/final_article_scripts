"""
========================================================================
Shift-Share (Bartik) 连续敞口指数 — 事前趋势检验（Pre-trend Test）
========================================================================

研究问题：
  基于 2005-2007 年静态权重构建的连续敞口指数是否严格外生？
  即：未来（2015年）的敞口是否能预测过去（2005-2008）的韧性变化？

核心检验逻辑（外生性验证）：
  如果 Exposure_{2015} 显著预测 Δ ln_Resilience_{08-12}，
  则违反 Bartik 假设 → 权重可能本身就受到被解释变量的影响
  
  反之（p > 0.10），则支持外生性假设

模型设定：
  Δ ln_Resilience_{05-08, i} = α + β * Exposure_{2015, i} 
                                 + γ1 * log_GDP_PC_{2005, i} 
                                 + γ2 * FDI_{2005, i} 
                                 + γ3 * ln_Resilience_{2005, i}
                                 + ε_i

关键参数：
  • 因变量：2005-2008 年间网络韧性的变化
  • 核心自变量：2015 年连续敞口指数（未来敞口）
  • 控制变量：2005 年人均GDP对数、FDI、2005年韧性初始水平
  • 标准误：HC1 稳健标准误（交叉截面）
  
期望结果：β 不显著（p > 0.10）→ 支持 Bartik 外生性

========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和高分辨率显示
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用SimHei显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['figure.dpi'] = 100  # 设置figure基础dpi

# 导入统计包
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("=" * 80)
print("Bartik 事前趋势检验（Pre-trend Test for Bartik Exogeneity）")
print("=" * 80)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, '输出文件')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '输出文件_不强制InternetUse')
os.makedirs(OUTPUT_DIR, exist_ok=True)

master_candidates = [
    os.path.join(OUTPUT_DIR, 'output_Master_Panel.csv'),
    os.path.join(SOURCE_DIR, 'output_Master_Panel.csv'),
    os.path.join(SCRIPT_DIR, 'Master_Panel.csv'),
]
master_path = next((p for p in master_candidates if os.path.exists(p)), None)
if master_path is None:
    raise FileNotFoundError('未找到 Master_Panel 数据（已检查输出文件_不强制InternetUse、输出文件、脚本目录）')

# ========== 第一部分：加载和预处理数据 ==========
print("\n【第一步】加载主面板数据...")
print("-" * 80)

# 加载数据
df_panel = pd.read_csv(master_path)

# 标准化列名
df_panel = df_panel.rename(columns={
    'REF_AREA': 'country_id',
    'TIME_PERIOD': 'year'
})

# 检查必需列
print(f"\n原始数据列名：{df_panel.columns.tolist()}")
print(f"原始数据形状：{df_panel.shape}")
print(f"年份范围：{df_panel['year'].min()} - {df_panel['year'].max()}")
print(f"国家/地区数：{df_panel['country_id'].nunique()}")

# 检查关键变量
key_vars = ['Exposure', 'True_Resilience', 'GDP_PC', 'FDI']
missing_vars = [var for var in key_vars if var not in df_panel.columns]
if missing_vars:
    print(f"\n⚠️  警告：缺失以下变量 {missing_vars}")
else:
    print(f"\n✓ 所有关键变量已齐备")

# 创建log变量
df_panel['ln_Resilience_Inv'] = np.log(df_panel['True_Resilience'] + 1e-6)
df_panel['log_GDP_PC'] = np.log(df_panel['GDP_PC'] + 1e-6)

# ========== 第二部分：构建截面数据集 ==========
print("\n【第二步】构建截面回归数据集...")
print("-" * 80)

# 定义关键年份
year_2005 = 2005
year_2008 = 2008
year_2015 = 2015
pre_period = [2005, 2006, 2007, 2008]

print(f"\n数据定义：")
print(f"  • Pre-period: {year_2005} - {year_2008}")
print(f"  • 因变量年份：计算 {year_2005} 到 {year_2008} 的韧性变化")
print(f"  • Exposure 来源年份：{year_2015}")
print(f"  • 基期年份（控制变量）：{year_2005}")

# 提取 2005 年数据（基期）
df_2005 = df_panel[df_panel['year'] == year_2005][['country_id', 'log_GDP_PC', 'FDI']].copy()
df_2005.rename(columns={
    'log_GDP_PC': 'log_GDP_PC_2005',
    'FDI': 'FDI_2005'
}, inplace=True)

print(f"\n2005 年基期样本：{len(df_2005)} 个国家/地区")

# 提取 2005 年韧性
df_resilience_2005 = df_panel[df_panel['year'] == year_2005][['country_id', 'ln_Resilience_Inv']].copy()
df_resilience_2005.rename(columns={'ln_Resilience_Inv': 'ln_Resilience_2005'}, inplace=True)

# 提取 2008 年韧性
df_resilience_2008 = df_panel[df_panel['year'] == year_2008][['country_id', 'ln_Resilience_Inv']].copy()
df_resilience_2008.rename(columns={'ln_Resilience_Inv': 'ln_Resilience_2008'}, inplace=True)

# 计算韧性变化（因变量）
df_resilience_change = df_resilience_2005.merge(df_resilience_2008, on='country_id', how='inner')
df_resilience_change['Delta_ln_Resilience_05_08'] = (
    df_resilience_change['ln_Resilience_2008'] - df_resilience_change['ln_Resilience_2005']
)

print(f"韧性变化样本：{len(df_resilience_change)} 个国家/地区")
print(f"  • 2005-2008 年平均韧性变化：{df_resilience_change['Delta_ln_Resilience_05_08'].mean():.6f}")
print(f"  • 标准差：{df_resilience_change['Delta_ln_Resilience_05_08'].std():.6f}")
print(f"  • 最小值：{df_resilience_change['Delta_ln_Resilience_05_08'].min():.6f}")
print(f"  • 最大值：{df_resilience_change['Delta_ln_Resilience_05_08'].max():.6f}")

# 提取 2015 年敞口指数（核心自变量）
df_exposure_2015 = df_panel[df_panel['year'] == year_2015][['country_id', 'Exposure']].copy()
df_exposure_2015.rename(columns={'Exposure': 'Exposure_2015'}, inplace=True)

print(f"\n2015 年敞口指数样本：{len(df_exposure_2015)} 个国家/地区")
print(f"  • 平均敞口：{df_exposure_2015['Exposure_2015'].mean():.6f}")
print(f"  • 标准差：{df_exposure_2015['Exposure_2015'].std():.6f}")
print(f"  • 最小值：{df_exposure_2015['Exposure_2015'].min():.6f}")
print(f"  • 最大值：{df_exposure_2015['Exposure_2015'].max():.6f}")

# 合并截面数据集
df_cs = df_resilience_change.merge(df_2005, on='country_id', how='inner')
df_cs = df_cs.merge(df_exposure_2015, on='country_id', how='inner')

print(f"\n✓ 合并后的截面数据集：{len(df_cs)} 个国家/地区")

# 删除缺失值
df_cs_clean = df_cs.dropna(subset=[
    'Delta_ln_Resilience_05_08',
    'Exposure_2015',
    'log_GDP_PC_2005',
    'FDI_2005',
    'ln_Resilience_2005'
])

print(f"✓ 清洗后的有效样本：{len(df_cs_clean)} 个国家/地区")

# 显示样本信息
print(f"\n【截面数据样本统计】")
print(df_cs_clean[['country_id', 'Delta_ln_Resilience_05_08', 'Exposure_2015', 'log_GDP_PC_2005', 'FDI_2005', 'ln_Resilience_2005']].head(15))

# ========== 第三部分：描述性统计 ==========
print("\n【第三步】描述性统计分析...")
print("-" * 80)

print("\n【因变量统计】")
print(f"Δ ln_Resilience_{{{year_2005}-{year_2008}}}:")
print(df_cs_clean['Delta_ln_Resilience_05_08'].describe())

print("\n【核心自变量统计】")
print(f"Exposure_{{{year_2015}}}:")
print(df_cs_clean['Exposure_2015'].describe())

print("\n【控制变量统计】")
print(f"log_GDP_PC_{{{year_2005}}}:")
print(df_cs_clean['log_GDP_PC_2005'].describe())
print(f"\nFDI_{{{year_2005}}}:")
print(df_cs_clean['FDI_2005'].describe())
print(f"\nln_Resilience_{{{year_2005}}} (初始水平):")
print(df_cs_clean['ln_Resilience_2005'].describe())

# ========== 第四部分：相关性分析 ==========
print("\n【第四步】相关性矩阵...")
print("-" * 80)

corr_matrix = df_cs_clean[[
    'Delta_ln_Resilience_05_08',
    'Exposure_2015',
    'log_GDP_PC_2005',
    'FDI_2005',
    'ln_Resilience_2005'
]].corr()

print("\n相关性矩阵：")
print(corr_matrix)

# 特别关注核心关系
bartik_corr = corr_matrix.loc['Delta_ln_Resilience_05_08', 'Exposure_2015']
print(f"\n核心关系（简单相关）：")
print(f"  corr(Δ ln_Resilience, Exposure_2015) = {bartik_corr:.6f}")

# ========== 第五部分：OLS 截面回归 ==========
print("\n" + "=" * 80)
print("【第五步】运行 OLS 截面回归（HC1 稳健标准误）")
print("=" * 80)

# 准备回归变量
y = df_cs_clean['Delta_ln_Resilience_05_08'].values
X = df_cs_clean[[
    'Exposure_2015',
    'log_GDP_PC_2005',
    'FDI_2005',
    'ln_Resilience_2005'
]].values

# 添加常数项
X_with_const = sm.add_constant(X)

# OLS 回归
model = sm.OLS(y, X_with_const)
results = model.fit(cov_type='HC1')  # HC1 稳健标准误

print("\n【完整回归结果摘要】")
print(results.summary())

# ========== 第六部分：关键诊断 ==========
print("\n" + "=" * 80)
print("【第六步】Bartik 外生性诊断")
print("=" * 80)

# 提取核心系数
exposure_coef = results.params[1]  # Exposure_2015 的系数
exposure_pval = results.pvalues[1]
exposure_se = results.bse[1]
exposure_tstat = results.tvalues[1]
ci_array = results.conf_int()
exposure_ci_lower = ci_array[1, 0]
exposure_ci_upper = ci_array[1, 1]

print(f"\n【核心自变量：Exposure_2015】")
print(f"  • 系数 β = {exposure_coef:.8f}")
print(f"  • 标准误 SE = {exposure_se:.8f}")
print(f"  • t 统计量 = {exposure_tstat:.6f}")
print(f"  • P 值 = {exposure_pval:.6f}")
print(f"  • 95% 置信区间：[{exposure_ci_lower:.8f}, {exposure_ci_upper:.8f}]")

# 诊断：Bartik 假设检验
print(f"\n【Bartik 外生性假设检验】")
print(f"  原假设（H0）：Exposure_2015 对 Δ ln_Resilience 无预测力 → β = 0")
print(f"  备选假设（H1）：Exposure_2015 显著预测 Δ ln_Resilience → β ≠ 0")
print()

# 不同显著性水平
alpha_values = [0.10, 0.05, 0.01]
for alpha in alpha_values:
    if exposure_pval < alpha:
        print(f"  ✗ 在 {alpha*100:.0f}% 显著性水平拒绝 H0（p = {exposure_pval:.6f} < {alpha}）")
        print(f"    → 结论：未来敞口 DOES 显著预测过去增长 → 违反 Bartik 假设")
    else:
        print(f"  ✓ 在 {alpha*100:.0f}% 显著性水平接受 H0（p = {exposure_pval:.6f} ≥ {alpha}）")
        print(f"    → 结论：未来敞口无法预测过去增长 → 支持 Bartik 外生性")

print()
if exposure_pval > 0.10:
    print("  🎯 总体评价：✓✓✓ 强烈支持 Bartik 外生性假设")
    print("     基于 2005-2007 年权重的敞口指数是严格外生的")
elif exposure_pval > 0.05:
    print("  🎯 总体评价：✓✓ 中等支持 Bartik 外生性假设")
    print("     虽然在 10% 水平不显著，但在 5% 水平接近显著")
else:
    print("  🎯 总体评价：⚠️  不支持 Bartik 外生性假设")
    print("     需要进一步检查权重构造的合理性")

# ========== 第七部分：其他控制变量的诊断 ==========
print("\n" + "=" * 80)
print("【第七步】其他控制变量诊断")
print("=" * 80)

# GDP 控制
gdp_coef = results.params[2]
gdp_pval = results.pvalues[2]
print(f"\n【log_GDP_PC_2005 的效应】")
print(f"  • 系数：{gdp_coef:.8f}")
print(f"  • P 值：{gdp_pval:.6f}")
if gdp_pval < 0.05:
    print(f"  • 显著性：✓ 显著（p < 0.05）")
else:
    print(f"  • 显著性：× 不显著（p ≥ 0.05）")

# FDI 控制
fdi_coef = results.params[3]
fdi_pval = results.pvalues[3]
print(f"\n【FDI_2005 的效应】")
print(f"  • 系数：{fdi_coef:.8f}")
print(f"  • P 值：{fdi_pval:.6f}")
if fdi_pval < 0.05:
    print(f"  • 显著性：✓ 显著（p < 0.05）")
else:
    print(f"  • 显著性：× 不显著（p ≥ 0.05）")

# 初始水平控制
resilience_coef = results.params[4]
resilience_pval = results.pvalues[4]
print(f"\n【ln_Resilience_2005 的效应 (初始水平控制)】")
print(f"  • 系数：{resilience_coef:.8f}")
print(f"  • P 值：{resilience_pval:.6f}")
if resilience_pval < 0.05:
    print(f"  • 显著性：✓ 显著（p < 0.05）")
else:
    print(f"  • 显著性：× 不显著（p ≥ 0.05）")

# ========== 第八部分：多重共线性检验（VIF）==========
print("\n【第八步】多重共线性检验（VIF）...")
print("-" * 80)

X_for_vif = df_cs_clean[[
    'Exposure_2015',
    'log_GDP_PC_2005',
    'FDI_2005',
    'ln_Resilience_2005'
]].values

print("\n方差膨胀因子（VIF）：")
for i, col in enumerate(['Exposure_2015', 'log_GDP_PC_2005', 'FDI_2005', 'ln_Resilience_2005']):
    vif = variance_inflation_factor(X_for_vif, i)
    print(f"  • {col}: VIF = {vif:.4f}", end="")
    if vif < 5:
        print(" ✓（低共线性）")
    elif vif < 10:
        print(" ⚠️ （中等共线性）")
    else:
        print(" ✗✗（高共线性）")

# ========== 第九部分：模型拟合度 ==========
print("\n【第九步】模型拟合度诊断...")
print("-" * 80)

r_squared = results.rsquared
adj_r_squared = results.rsquared_adj
f_stat = results.fvalue
f_pval = results.f_pvalue
n_obs = results.nobs
k_vars = results.df_model

print(f"\n样本量：{n_obs} 个国家/地区")
print(f"回归变量数（含常数）：{k_vars + 1}")
print(f"R-squared：{r_squared:.6f}")
print(f"Adjusted R-squared：{adj_r_squared:.6f}")
print(f"F-statistic：{f_stat:.4f}（df1={k_vars}, df2={n_obs-k_vars-1}）")
print(f"P-value (F-test)：{f_pval:.6f}")

if f_pval < 0.05:
    print(f"✓ 模型整体显著（p < 0.05）")
else:
    print(f"× 模型整体不显著（p ≥ 0.05）")

# ========== 第十部分：残差诊断 ==========
print("\n【第十步】残差诊断...")
print("-" * 80)

residuals = results.resid
fitted = results.fittedvalues

print(f"\n残差统计：")
print(f"  • 平均值：{residuals.mean():.8f}（应接近0）")
print(f"  • 标准差：{residuals.std():.6f}")
print(f"  • 斜度：{stats.skew(residuals):.6f}")
print(f"  • 峰度：{stats.kurtosis(residuals):.6f}")

# Jarque-Bera 正态性检验
from scipy.stats import jarque_bera as jb_test
jb_stat, jb_pval = jb_test(residuals)
print(f"\nJarque-Bera 正态性检验：")
print(f"  • JB 统计量：{jb_stat:.4f}")
print(f"  • P 值：{jb_pval:.6f}")
if jb_pval > 0.05:
    print(f"  ✓ 残差正态分布（p > 0.05）")
else:
    print(f"  ⚠️  残差可能非正态（p ≤ 0.05）")

# ========== 第十一部分：绘制诊断图 ==========
print("\n【第十一步】绘制诊断图...")
print("-" * 80)

# 创建高分辨率图表（4096x2160分辨率）
# 计算figsize以达到目标分辨率：4096÷216=18.96, 2160÷216=10
fig, axes = plt.subplots(2, 2, figsize=(18.96, 10), dpi=216)

# 残差 vs 拟合值
axes[0, 0].scatter(fitted, residuals, alpha=0.6, edgecolors='k', s=50)
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('拟合值 (Fitted Values)', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('残差 (Residuals)', fontsize=14, fontweight='bold')
axes[0, 0].set_title('残差 vs 拟合值', fontsize=16, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Q-Q 图
from scipy.stats import probplot
probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q 图', fontsize=16, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 残差分布
axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7, density=True)
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(residuals.min(), residuals.max(), 100)
axes[1, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=3, label='正态分布')
axes[1, 0].set_xlabel('残差 (Residuals)', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('密度 (Density)', fontsize=14, fontweight='bold')
axes[1, 0].set_title('残差分布', fontsize=16, fontweight='bold')
axes[1, 0].legend(fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# 核心关系:未来敞口 vs 过去增长（带回归线）
axes[1, 1].scatter(df_cs_clean['Exposure_2015'], df_cs_clean['Delta_ln_Resilience_05_08'], 
                   alpha=0.6, edgecolors='k', s=80)
# 添加回归线
z = np.polyfit(df_cs_clean['Exposure_2015'], df_cs_clean['Delta_ln_Resilience_05_08'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_cs_clean['Exposure_2015'].min(), df_cs_clean['Exposure_2015'].max(), 100)
axes[1, 1].plot(x_line, p(x_line), "r-", linewidth=3, label=f'β = {exposure_coef:.6f}, p = {exposure_pval:.4f}')
axes[1, 1].set_xlabel('敞口指数 (Exposure_{2015})', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('韧性变化 (Δ ln_Resilience_{{05-08}})', fontsize=14, fontweight='bold')
axes[1, 1].set_title('Bartik 外生性检验：未来敞口 vs 过去增长', fontsize=16, fontweight='bold')
axes[1, 1].legend(fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

# 调整整体布局以确保标签完整显示
plt.tight_layout()

# 保存为高分辨率图片（4096x2160）
bartik_plot = os.path.join(OUTPUT_DIR, 'Bartik_事前趋势检验_诊断图.png')
plt.savefig(bartik_plot, dpi=216, bbox_inches='tight', facecolor='white')
print(f"\n✓ 诊断图已保存：{bartik_plot}")
print(f"  • 输出分辨率：4096 × 2160 像素")
print(f"  • 中文字体：已启用 (SimHei)")
plt.show()

# ========== 第十二部分：总结报告 ==========
print("\n" + "=" * 80)
print("【总结报告】Bartik 外生性诊断")
print("=" * 80)

print(f"""
【数据规格】
  样本规模：{n_obs} 个国家/地区
  时间跨度：2005-2008 年（3 年变化）
  估计方法：OLS 截面回归，HC1 稳健标准误

【模型规格】
  因变量：Δ ln_Resilience_{{05-08}}（2005-2008 年网络韧性对数变化）
  核心自变量：Exposure_2015（2015 年连续敞口指数）
  控制变量：log_GDP_PC_2005, FDI_2005, ln_Resilience_2005（初始水平）

【关键结果】
  Exposure_2015 系数：β = {exposure_coef:.8f}
  P 值：{exposure_pval:.6f}
  
【Bartik 外生性诊断】
""")

if exposure_pval > 0.10:
    print(f"""
  ✓✓✓ 强烈支持 Bartik 外生性
  
  结论：
    • 未来（2015）的敞口指数无法显著预测过去（2005-2008）的韧性变化
    • 基于 2005-2007 年权重的连续敞口指数是严格外生的
    • 未被历史前置趋势污染
    • 可以合理用于差分法（Difference-in-Differences）估计
""")
elif exposure_pval > 0.05:
    print(f"""
  ✓✓ 中等支持 Bartik 外生性
  
  结论：
    • 虽然在 10% 显著性水平不显著，但在 5% 水平接近显著临界值
    • 建议进行敏感性分析以确保 DID 结果的稳健性
    • 考虑使用 Callaway-Sant'Anna 等不依赖平行趋势的方法作为补充检验
""")
else:
    print(f"""
  ⚠️  ⚠️  不支持 Bartik 外生性
  
  结论：
    • 未来敞口指数显著预测过去的韧性变化（p < 0.05）
    • 存在潜在的反向因果或遗漏变量偏差
    • 需要重新审视权重构造方法
    • Bartik 敞口可能被结果变量本身污染
    • 不建议直接用于 DID 估计
""")

print("\n" + "=" * 80)
print("检验完成！")
print("=" * 80)
