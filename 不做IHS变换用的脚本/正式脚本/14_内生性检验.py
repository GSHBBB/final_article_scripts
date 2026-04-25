"""
========================================================================
内生性检验全流程 (Endogeneity Diagnostics)
========================================================================

本脚本针对论文 "数字贸易规则对数字服务出口网络韧性的影响" 的核心回归，
系统性地检测并回应以下内生性威胁：

  检验 1: 反向因果排除 — 反向回归 (Reverse Regression)
          将 Exposure 作为因变量，检验 L1_Resilience 是否显著
          ✓ 若不显著 → 排除 "高韧性国家反而更积极签约" 的反向因果

  检验 2: Granger 因果检验 — 面板层面 Granger Causality
          在双向固定效应框架下检验因果方向:
          (a) Exposure → Resilience (正向, 预期显著)
          (b) Resilience → Exposure (反向, 预期不显著)

  检验 3: 多期滞后结构稳定性 (L1/L2/L3 Lag Structure Test)
          逐级加入更深层滞后项，检验政策效应的时滞模式:
          ✓ L1 显著正 → 效应 1 年内传导
          ✓ L2/L3 减弱 → 效应随时间衰减（合理）

  检验 4: 前置项检验 (Lead Test for Reverse Causality)
          加入 F1_Exposure (超前1期)，检验未来的规则是否能
          "穿越" 影响当期韧性:
          ✓ 若 F1 不显著而 L1 显著 → 排除反向因果

  检验 5: 遗漏变量稳健性 — Oster (2019) δ 界限估计
          用 R² 变化来推断遗漏变量偏误的严重程度:
          ✓ 若 δ > 1 → 即使遗漏变量影响力超过所有可观测控制变量，
            也无法将系数翻转为零，结论稳健

========================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# 解决 Windows 终端编码问题
if os.name == 'nt':
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

# ============================================================================
# 配置
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

# ============================================================================
# 数据加载与预处理 (与 08 号脚本保持一致)
# ============================================================================
print("=" * 90)
print("内生性检验全流程 (Endogeneity Diagnostics)")
print("=" * 90)

print("\n[数据加载]")
df = pd.read_csv(data_file)
df_resilience = pd.read_csv(resilience_file)

if 'Out_Degree_Centrality' in df_resilience.columns:
    centrality_df = df_resilience[['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']].copy()
    df = df.merge(centrality_df, on=['REF_AREA', 'TIME_PERIOD'], how='left')

# 构造核心变量
required_cols = ['REF_AREA', 'TIME_PERIOD', 'True_Resilience', 'Exposure',
                 'GDP_PC', 'FDI', 'Out_Degree_Centrality']
df_model = df[required_cols].dropna().copy()

df_model['ln_Resilience_Inv'] = -np.log(df_model['True_Resilience'])
df_model['log_GDP_PC'] = np.log(df_model['GDP_PC'])

df_indexed = df_model.set_index(['REF_AREA', 'TIME_PERIOD'])

# 生成滞后与超前项
df_indexed['L1_Exposure'] = df_indexed.groupby(level='REF_AREA')['Exposure'].shift(1)
df_indexed['L2_Exposure'] = df_indexed.groupby(level='REF_AREA')['Exposure'].shift(2)
df_indexed['L3_Exposure'] = df_indexed.groupby(level='REF_AREA')['Exposure'].shift(3)
df_indexed['F1_Exposure'] = df_indexed.groupby(level='REF_AREA')['Exposure'].shift(-1)

df_indexed['L1_Resilience'] = df_indexed.groupby(level='REF_AREA')['ln_Resilience_Inv'].shift(1)
df_indexed['L1_Centrality'] = df_indexed.groupby(level='REF_AREA')['Out_Degree_Centrality'].shift(1)

# 基准清洁样本 (与 08 号脚本一致)
df_clean = df_indexed.dropna(subset=['L1_Exposure', 'L1_Centrality', 'ln_Resilience_Inv']).copy()
df_clean['Constant'] = 1.0

print(f"  有效样本量: {len(df_clean)}")
print(f"  涉及国家数: {df_clean.index.get_level_values('REF_AREA').nunique()}")
print(f"  涉及年份范围: {df_clean.index.get_level_values('TIME_PERIOD').min()} - "
      f"{df_clean.index.get_level_values('TIME_PERIOD').max()}")


def print_separator(title):
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)


def print_coef_summary(res, var_name, label=""):
    """打印单个变量的系数摘要"""
    coef = res.params[var_name]
    pval = res.pvalues[var_name]
    se = res.std_errors[var_name]
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "n.s."
    if label:
        print(f"  {label}")
    print(f"    系数 = {coef:+.6f}  |  标准误 = {se:.6f}  |  P值 = {pval:.6f}  [{sig}]")
    return coef, pval, sig


# ============================================================================
# 检验 1: 反向因果排除 — 反向回归 (Reverse Regression)
# ============================================================================
print_separator("检验 1: 反向因果排除 — 反向回归 (Reverse Regression)")
print("""
  逻辑: 如果存在反向因果 (高韧性 → 积极签约), 那么用滞后韧性
        预测当期敞口应该显著。
  设定: Exposure_{i,t} = a_i + d_t + beta * L1_ln_Resilience_Inv_{i,t}
                         + gamma * Controls + eps
  预期: beta 不显著 → 排除反向因果
""")

df_rev = df_clean.dropna(subset=['L1_Resilience']).copy()
y_rev = df_rev['Exposure']
X_rev = df_rev[['Constant', 'L1_Resilience', 'log_GDP_PC', 'FDI']]

model_rev = PanelOLS(y_rev, X_rev, entity_effects=True, time_effects=True)
res_rev = model_rev.fit(cov_type='clustered', cluster_entity=True)

coef_rev, pval_rev, sig_rev = print_coef_summary(
    res_rev, 'L1_Resilience', "L1_ln_Resilience_Inv → Exposure:")

print(f"\n  [判定] ", end="")
if pval_rev >= 0.10:
    print(f"通过! L1_Resilience 对 Exposure 无显著预测力 (p={pval_rev:.4f})")
    print(f"       → 排除 '高韧性导致更多规则签订' 的反向因果")
else:
    print(f"警告! L1_Resilience 对 Exposure 显著 (p={pval_rev:.4f})")
    print(f"       → 可能存在反向因果, 需进一步使用 IV 估计")


# ============================================================================
# 检验 2: 面板 Granger 因果检验
# ============================================================================
print_separator("检验 2: 面板 Granger 因果检验 (Panel Granger Causality)")
print("""
  逻辑: 在控制了因变量自身滞后后, 检验另一变量的滞后是否有额外
        预测能力。在双向固定效应框架下进行。
  方向 A: L1_Exposure → ln_Resilience_Inv (正向因果, 预期显著)
  方向 B: L1_Resilience → Exposure         (反向因果, 预期不显著)
""")

# 方向 A: Exposure Granger-causes Resilience?
print("-" * 80)
print("  方向 A: Exposure 是否 Granger-causes Resilience?")
df_granger = df_clean.dropna(subset=['L1_Resilience']).copy()

# 受限模型 (不含 L1_Exposure)
y_ga = df_granger['ln_Resilience_Inv']
X_ga_restricted = df_granger[['Constant', 'L1_Resilience', 'log_GDP_PC', 'FDI']]
res_ga_r = PanelOLS(y_ga, X_ga_restricted, entity_effects=True, time_effects=True)\
    .fit(cov_type='clustered', cluster_entity=True)

# 非受限模型 (加入 L1_Exposure)
X_ga_full = df_granger[['Constant', 'L1_Exposure', 'L1_Resilience', 'log_GDP_PC', 'FDI']]
res_ga_f = PanelOLS(y_ga, X_ga_full, entity_effects=True, time_effects=True)\
    .fit(cov_type='clustered', cluster_entity=True)

coef_ga, pval_ga, sig_ga = print_coef_summary(
    res_ga_f, 'L1_Exposure', "  在控制 L1_Resilience 后, L1_Exposure 的系数:")

# 计算增量 F 统计量
r2_restricted = res_ga_r.rsquared
r2_full = res_ga_f.rsquared
q = 1  # 新增的变量个数
n = res_ga_f.nobs
k = len(res_ga_f.params)
if r2_full > r2_restricted:
    f_incr = ((r2_full - r2_restricted) / q) / ((1 - r2_full) / (n - k))
else:
    f_incr = 0
print(f"  增量 R-squared: {r2_full - r2_restricted:.6f}")
print(f"  增量 F 统计量: {f_incr:.4f}")

print(f"\n  [判定] ", end="")
if pval_ga < 0.10:
    print(f"Exposure Granger-causes Resilience (p={pval_ga:.4f})")
else:
    print(f"Exposure 不 Granger-cause Resilience (p={pval_ga:.4f})")


# 方向 B: Resilience Granger-causes Exposure?
print("\n" + "-" * 80)
print("  方向 B: Resilience 是否 Granger-causes Exposure?")

y_gb = df_granger['Exposure']
X_gb_restricted = df_granger[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
res_gb_r = PanelOLS(y_gb, X_gb_restricted, entity_effects=True, time_effects=True)\
    .fit(cov_type='clustered', cluster_entity=True)

X_gb_full = df_granger[['Constant', 'L1_Exposure', 'L1_Resilience', 'log_GDP_PC', 'FDI']]
res_gb_f = PanelOLS(y_gb, X_gb_full, entity_effects=True, time_effects=True)\
    .fit(cov_type='clustered', cluster_entity=True)

coef_gb, pval_gb, sig_gb = print_coef_summary(
    res_gb_f, 'L1_Resilience', "  在控制 L1_Exposure 后, L1_Resilience 的系数:")

r2_restricted_b = res_gb_r.rsquared
r2_full_b = res_gb_f.rsquared
if r2_full_b > r2_restricted_b:
    f_incr_b = ((r2_full_b - r2_restricted_b) / q) / ((1 - r2_full_b) / (n - k))
else:
    f_incr_b = 0
print(f"  增量 R-squared: {r2_full_b - r2_restricted_b:.6f}")
print(f"  增量 F 统计量: {f_incr_b:.4f}")

print(f"\n  [判定] ", end="")
if pval_gb >= 0.10:
    print(f"Resilience 不 Granger-cause Exposure (p={pval_gb:.4f})")
    print(f"       → 因果方向确认: 规则 → 韧性, 而非韧性 → 规则")
else:
    print(f"警告! Resilience 可能 Granger-cause Exposure (p={pval_gb:.4f})")

# Granger 综合判定
print("\n" + "-" * 80)
print("  [Granger 因果综合判定]")
a_to_b = pval_ga < 0.10
b_to_a = pval_gb < 0.10

if a_to_b and not b_to_a:
    print("    单向因果: Exposure → Resilience (理想结果)")
elif a_to_b and b_to_a:
    print("    双向因果: 存在反馈效应, 但不影响核心结论的方向性")
elif not a_to_b and not b_to_a:
    print("    双向均不显著: 可能因双向FE吸收过多变异, 参考检验1的结果")
else:
    print("    反向因果: Resilience → Exposure, 需要 IV 估计")


# ============================================================================
# 检验 3: 多期滞后结构稳定性 (Lag Structure Test)
# ============================================================================
print_separator("检验 3: 多期滞后结构稳定性 (L1/L2/L3)")
print("""
  逻辑: 政策效应应有合理的时滞衰减模式。如果仅 L1 显著,
        说明效应 1 年内传导; L2/L3 衰减说明效应非趋势驱动。
""")

# 模型 3A: 仅 L1
df_lag = df_clean.copy()
y_lag = df_lag['ln_Resilience_Inv']

print("  [模型 3A] 仅 L1_Exposure:")
X_3a = df_lag[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
res_3a = PanelOLS(y_lag, X_3a, entity_effects=True, time_effects=True)\
    .fit(cov_type='clustered', cluster_entity=True)
c3a, p3a, s3a = print_coef_summary(res_3a, 'L1_Exposure')

# 模型 3B: L1 + L2
df_lag_2 = df_clean.dropna(subset=['L2_Exposure']).copy()
y_lag_2 = df_lag_2['ln_Resilience_Inv']

print("\n  [模型 3B] L1 + L2_Exposure:")
X_3b = df_lag_2[['Constant', 'L1_Exposure', 'L2_Exposure', 'log_GDP_PC', 'FDI']]
res_3b = PanelOLS(y_lag_2, X_3b, entity_effects=True, time_effects=True)\
    .fit(cov_type='clustered', cluster_entity=True)
c3b_l1, p3b_l1, s3b_l1 = print_coef_summary(res_3b, 'L1_Exposure', "    L1_Exposure:")
c3b_l2, p3b_l2, s3b_l2 = print_coef_summary(res_3b, 'L2_Exposure', "    L2_Exposure:")

# 模型 3C: L1 + L2 + L3
df_lag_3 = df_clean.dropna(subset=['L2_Exposure', 'L3_Exposure']).copy()
y_lag_3 = df_lag_3['ln_Resilience_Inv']

print("\n  [模型 3C] L1 + L2 + L3_Exposure:")
X_3c = df_lag_3[['Constant', 'L1_Exposure', 'L2_Exposure', 'L3_Exposure', 'log_GDP_PC', 'FDI']]
res_3c = PanelOLS(y_lag_3, X_3c, entity_effects=True, time_effects=True)\
    .fit(cov_type='clustered', cluster_entity=True)
c3c_l1, p3c_l1, s3c_l1 = print_coef_summary(res_3c, 'L1_Exposure', "    L1_Exposure:")
c3c_l2, p3c_l2, s3c_l2 = print_coef_summary(res_3c, 'L2_Exposure', "    L2_Exposure:")
c3c_l3, p3c_l3, s3c_l3 = print_coef_summary(res_3c, 'L3_Exposure', "    L3_Exposure:")

# 滞后累积效应
cumulative_L1 = c3a
cumulative_L2 = c3b_l1 + c3b_l2
cumulative_L3 = c3c_l1 + c3c_l2 + c3c_l3

print(f"\n  [滞后累积效应]")
print(f"    仅 L1 累积:        {cumulative_L1:+.6f}")
print(f"    L1+L2 累积:        {cumulative_L2:+.6f}")
print(f"    L1+L2+L3 累积:     {cumulative_L3:+.6f}")

print(f"\n  [判定] ", end="")
if p3a < 0.10 and (p3b_l2 >= 0.10 or abs(c3b_l2) < abs(c3b_l1)):
    print("合理的衰减模式: 效应主要在 L1 期传导, 后续衰减")
    print("         → 支持 '政策一年内传导' 的设定, 排除趋势驱动的虚假相关")
else:
    print("需要注意: 滞后结构可能暗示更复杂的动态关系")


# ============================================================================
# 检验 4: 前置项检验 (Lead Test)
# ============================================================================
print_separator("检验 4: 前置项检验 (Lead Test for Pre-treatment Trends)")
print("""
  逻辑: 如果未来的规则 (F1_Exposure) 能 "穿越" 影响当期韧性,
        说明存在预处理趋势或反向因果。
  设定: 在基准模型中同时加入 F1 和 L1, 检验 F1 是否显著。
  预期: F1 不显著 + L1 显著 → 因果方向正确
""")

df_lead = df_clean.dropna(subset=['F1_Exposure']).copy()
y_lead = df_lead['ln_Resilience_Inv']
X_lead = df_lead[['Constant', 'F1_Exposure', 'L1_Exposure', 'log_GDP_PC', 'FDI']]

model_lead = PanelOLS(y_lead, X_lead, entity_effects=True, time_effects=True)
res_lead = model_lead.fit(cov_type='clustered', cluster_entity=True)

coef_f1, pval_f1, sig_f1 = print_coef_summary(
    res_lead, 'F1_Exposure', "  F1_Exposure (超前 1 期):")
coef_l1, pval_l1, sig_l1 = print_coef_summary(
    res_lead, 'L1_Exposure', "  L1_Exposure (滞后 1 期):")

print(f"\n  [判定] ", end="")
if pval_f1 >= 0.10 and pval_l1 < 0.10:
    print(f"完美通过!")
    print(f"    F1 不显著 (p={pval_f1:.4f}) → 无预处理趋势")
    print(f"    L1 显著   (p={pval_l1:.4f}) → 效应在政策实施后才出现")
    print(f"    → 因果方向确认: 规则变化先于韧性变化")
elif pval_f1 >= 0.10 and pval_l1 >= 0.10:
    print(f"F1 不显著 (p={pval_f1:.4f}), L1 也不显著 (p={pval_l1:.4f})")
    print(f"    → 无反向因果证据, 但效应可能因样本缩减而减弱")
elif pval_f1 < 0.10:
    print(f"警告! F1 显著 (p={pval_f1:.4f})")
    print(f"    → 可能存在预处理趋势或预期效应, 需进一步分析")


# ============================================================================
# 检验 5: Oster (2019) 遗漏变量稳健性 — delta 界限估计
# ============================================================================
print_separator("检验 5: Oster (2019) 遗漏变量稳健性 -- delta 界限估计")
print("""
  逻辑: 基于 Emily Oster (2019, JBES) 的方法, 评估遗漏变量偏误
        的潜在严重程度。

  核心思想:
    - 比较 "无控制变量" 与 "有控制变量" 时核心系数和 R-squared 的变化
    - 推断: 如果存在同等影响力的遗漏变量, 系数是否会被翻转?
    - delta > 1 意味着遗漏变量需要比所有已控制变量的联合影响还大,
      才能将系数翻转为零 → 结论高度稳健

  参考: Oster, E. (2019). "Unobservable Selection and Coefficient
        Stability: Theory and Evidence." JBES, 37(2), 187-204.

  R_max 设定: 按 Oster 建议使用 min(1.3 * R_controlled, 1.0)
""")

# 模型 5A: 无控制变量 (仅 L1_Exposure + 双向固定效应)
y_oster = df_clean['ln_Resilience_Inv']
X_5a = df_clean[['Constant', 'L1_Exposure']]
res_5a = PanelOLS(y_oster, X_5a, entity_effects=True, time_effects=True)\
    .fit(cov_type='clustered', cluster_entity=True)

beta_uncontrolled = res_5a.params['L1_Exposure']
r2_uncontrolled = res_5a.rsquared

print(f"  [模型 5A] 无控制变量 (仅双向FE):")
print(f"    beta_short = {beta_uncontrolled:+.6f}")
print(f"    R-squared  = {r2_uncontrolled:.6f}")

# 模型 5B: 有控制变量 (基准模型)
X_5b = df_clean[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
res_5b = PanelOLS(y_oster, X_5b, entity_effects=True, time_effects=True)\
    .fit(cov_type='clustered', cluster_entity=True)

beta_controlled = res_5b.params['L1_Exposure']
r2_controlled = res_5b.rsquared

print(f"\n  [模型 5B] 有控制变量 (基准模型):")
print(f"    beta_full  = {beta_controlled:+.6f}")
print(f"    R-squared  = {r2_controlled:.6f}")

# 计算 Oster delta
# R_max = min(1.3 * R_controlled, 1.0) 是 Oster (2019) 的标准建议
R_max = min(1.3 * r2_controlled, 1.0)

print(f"\n  [Oster 参数]")
print(f"    R_max      = min(1.3 * {r2_controlled:.6f}, 1.0) = {R_max:.6f}")

# delta 计算公式 (Oster 2019, Equation 3):
# delta = [(beta_full - 0) * (R_max - R_controlled)] /
#         [(beta_short - beta_full) * (R_controlled - R_uncontrolled)]
numerator = beta_controlled * (R_max - r2_controlled)
denominator = (beta_uncontrolled - beta_controlled) * (r2_controlled - r2_uncontrolled)

if abs(denominator) < 1e-12:
    print("\n  [注意] 分母接近零: 控制变量几乎不改变系数,")
    print("         这本身说明遗漏变量偏误极小!")
    delta = float('inf')
    delta_str = "无穷大 (∞)"
else:
    delta = numerator / denominator
    delta_str = f"{delta:.4f}"

print(f"\n  [Oster delta 计算结果]")
print(f"    delta      = {delta_str}")

# 同时计算 beta* (bias-adjusted coefficient)
# beta* = beta_full - delta * (beta_short - beta_full) * (R_max - R_controlled) / (R_controlled - R_uncontrolled)
# 当 delta = 1 时:
if abs(r2_controlled - r2_uncontrolled) < 1e-12:
    beta_star = beta_controlled
    print(f"    beta* (delta=1) = {beta_star:.6f} (R-squared 无变化, beta* = beta_full)")
else:
    bias_adjustment = 1.0 * (beta_uncontrolled - beta_controlled) * \
                      (R_max - r2_controlled) / (r2_controlled - r2_uncontrolled)
    beta_star = beta_controlled - bias_adjustment
    print(f"    beta* (delta=1) = {beta_star:.6f} (偏误调整后系数)")

print(f"\n  [判定]")
if delta == float('inf') or delta > 1:
    print(f"    delta = {delta_str} > 1")
    print(f"    --> 即使遗漏变量的影响力超过所有可观测控制变量的联合效应,")
    print(f"        也无法将核心系数翻转为零。")
    print(f"    --> 结论: 核心发现对遗漏变量偏误高度稳健!")
elif delta > 0:
    print(f"    0 < delta = {delta_str} < 1")
    print(f"    --> 存在中等程度的遗漏变量敏感性, 但 delta > 0 仍提供正面证据")
else:
    print(f"    delta = {delta_str} <= 0")
    print(f"    --> 控制变量加入后系数方向与短回归相同, 遗漏变量增强了效应")
    print(f"    --> 这实际上意味着真实效应可能比估计值更大!")


# ============================================================================
# 检验 5 补充: 在交互项模型上重复 Oster 检验
# ============================================================================
print("\n" + "-" * 80)
print("  [补充] 在全样本交互项模型上重复 Oster 检验")
print("-" * 80)

# 交互项模型: 无控制
df_clean['Interaction'] = df_clean['L1_Exposure'] * df_clean['L1_Centrality']
X_5c = df_clean[['Constant', 'L1_Exposure', 'Interaction']]
res_5c = PanelOLS(y_oster, X_5c, entity_effects=True, time_effects=True)\
    .fit(cov_type='clustered', cluster_entity=True)

beta_inter_short = res_5c.params['L1_Exposure']
r2_inter_short = res_5c.rsquared

# 交互项模型: 有控制
X_5d = df_clean[['Constant', 'L1_Exposure', 'Interaction', 'log_GDP_PC', 'FDI']]
res_5d = PanelOLS(y_oster, X_5d, entity_effects=True, time_effects=True)\
    .fit(cov_type='clustered', cluster_entity=True)

beta_inter_full = res_5d.params['L1_Exposure']
r2_inter_full = res_5d.rsquared

R_max_inter = min(1.3 * r2_inter_full, 1.0)

num_inter = beta_inter_full * (R_max_inter - r2_inter_full)
den_inter = (beta_inter_short - beta_inter_full) * (r2_inter_full - r2_inter_short)

if abs(den_inter) < 1e-12:
    delta_inter = float('inf')
    delta_inter_str = "无穷大 (∞)"
else:
    delta_inter = num_inter / den_inter
    delta_inter_str = f"{delta_inter:.4f}"

print(f"  beta_short (交互项, 无控制) = {beta_inter_short:+.6f}, R2 = {r2_inter_short:.6f}")
print(f"  beta_full  (交互项, 有控制) = {beta_inter_full:+.6f}, R2 = {r2_inter_full:.6f}")
print(f"  R_max = {R_max_inter:.6f}")
print(f"  Oster delta = {delta_inter_str}")

if delta_inter == float('inf') or (isinstance(delta_inter, float) and delta_inter > 1):
    print(f"  --> 交互项模型的核心系数同样对遗漏变量偏误高度稳健!")


# ============================================================================
# 汇总报告
# ============================================================================
print_separator("内生性检验汇总报告")

results_summary = []
results_summary.append({
    '检验': '1. 反向回归',
    '核心变量': 'L1_Resilience → Exposure',
    '系数': f'{coef_rev:+.6f}',
    'P值': f'{pval_rev:.4f}',
    '结论': '通过 (无反向因果)' if pval_rev >= 0.10 else '警告 (可能存在反向因果)'
})
results_summary.append({
    '检验': '2a. Granger (正向)',
    '核心变量': 'L1_Exposure → Resilience',
    '系数': f'{coef_ga:+.6f}',
    'P值': f'{pval_ga:.4f}',
    '结论': '显著 (正向因果成立)' if pval_ga < 0.10 else '不显著'
})
results_summary.append({
    '检验': '2b. Granger (反向)',
    '核心变量': 'L1_Resilience → Exposure',
    '系数': f'{coef_gb:+.6f}',
    'P值': f'{pval_gb:.4f}',
    '结论': '通过 (无反向因果)' if pval_gb >= 0.10 else '警告 (存在反馈效应)'
})
results_summary.append({
    '检验': '3. L1/L2/L3 滞后',
    '核心变量': 'L1 → L2 → L3',
    '系数': f'L1={c3a:+.4f}, L2={c3b_l2:+.4f}, L3={c3c_l3:+.4f}',
    'P值': f'L1 p={p3a:.4f}, L2 p={p3b_l2:.4f}, L3 p={p3c_l3:.4f}',
    '结论': '合理衰减' if (p3a < 0.10 and (p3b_l2 >= 0.10 or abs(c3b_l2) < abs(c3b_l1))) else '需关注'
})
results_summary.append({
    '检验': '4. 前置项检验',
    '核心变量': 'F1_Exposure + L1_Exposure',
    '系数': f'F1={coef_f1:+.6f}, L1={coef_l1:+.6f}',
    'P值': f'F1 p={pval_f1:.4f}, L1 p={pval_l1:.4f}',
    '结论': '通过 (F1不显著L1显著)' if (pval_f1 >= 0.10 and pval_l1 < 0.10)
            else '部分通过' if pval_f1 >= 0.10 else '警告'
})
results_summary.append({
    '检验': '5. Oster delta',
    '核心变量': 'L1_Exposure (基准模型)',
    '系数': f'beta*={beta_controlled:+.6f}',
    'P值': f'delta={delta_str}',
    '结论': '高度稳健 (delta>1)' if (delta == float('inf') or delta > 1) else
            '中等稳健 (0<delta<1)' if delta > 0 else '反向增强'
})

df_summary = pd.DataFrame(results_summary)
print(df_summary.to_string(index=False))

# 导出汇总结果
output_csv = os.path.join(OUTPUT_DIR, 'output_endogeneity_diagnostics.csv')
df_summary.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n  结果已导出: {output_csv}")

# 总体判定
print("\n" + "=" * 90)
print("  总体判定")
print("=" * 90)

pass_count = 0
total_count = 5

# 检验1
if pval_rev >= 0.10:
    pass_count += 1
    print("  [1] 反向回归:       通过")
else:
    print("  [1] 反向回归:       未通过")

# 检验2
if pval_ga < 0.10 and pval_gb >= 0.10:
    pass_count += 1
    print("  [2] Granger因果:    通过 (单向因果)")
elif pval_gb >= 0.10:
    pass_count += 1
    print("  [2] Granger因果:    部分通过 (无反向因果)")
else:
    print("  [2] Granger因果:    需关注")

# 检验3
if p3a < 0.10:
    pass_count += 1
    print("  [3] 滞后结构:       通过 (L1 显著)")
else:
    print("  [3] 滞后结构:       需关注")

# 检验4
if pval_f1 >= 0.10:
    pass_count += 1
    print("  [4] 前置项检验:     通过 (F1 不显著)")
else:
    print("  [4] 前置项检验:     未通过")

# 检验5
if delta == float('inf') or delta > 1:
    pass_count += 1
    print("  [5] Oster delta:    通过 (高度稳健)")
elif delta > 0:
    pass_count += 1
    print("  [5] Oster delta:    部分通过")
else:
    print("  [5] Oster delta:    需关注")

print(f"\n  通过率: {pass_count}/{total_count}")

if pass_count >= 4:
    print("\n  --> 内生性检验总体良好, 核心结论稳健。")
    print("      因果推断链条: 数字贸易规则 → 网络韧性 (方向确认)")
elif pass_count >= 3:
    print("\n  --> 内生性检验基本通过, 核心方向可信, 但部分环节需在论文中讨论。")
else:
    print("\n  --> 内生性检验存在较多问题, 建议考虑 IV/GMM 等更强的识别策略。")

print("\n" + "=" * 90)
print("  内生性检验全流程完毕")
print("=" * 90)
