"""
双向聚类稳健性检验 (Two-Way Clustered Robustness Check)
═════════════════════════════════════════════════════════
说明：
- 根据L1_Centrality的33%分位数，提取最边缘国家子样本
- 运行两次回归对比：单向聚类 vs 国家-年份双向聚类
- 绘制森林图（Errorbar plot）展示系数与95%置信区间
"""

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib import rcParams
from linearmodels.panel import PanelOLS
import scipy.stats as stats

warnings.filterwarnings('ignore')

# ============================================================================
# 配置中文显示和绘图参数
# ============================================================================
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 216

print("=" * 100)
print("双向聚类稳健性检验：排除网络溢出干扰")
print("=" * 100)

# ============================================================================
# 0. 数据加载与准备
# ============================================================================
base_dir = r"C:\Users\29106\OneDrive\文档\毕业论文"

# 加载主面板数据
df_model = pd.read_csv(os.path.join(base_dir, "Master_Panel.csv"))

# 加载中心性数据
df_centrality = pd.read_csv(os.path.join(base_dir, "DATE_from_WB", "node_centrality_panel.csv"))

# 加载韧性数据
df_resilience = pd.read_csv(os.path.join(base_dir, "DATE_from_WB", "final_resilience_panel.csv"))

# 合并中心性数据
df_model = df_model.merge(
    df_centrality[['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']], 
    on=['REF_AREA', 'TIME_PERIOD'], 
    how='left'
)

# 选择需要的列并清理缺失值
cols = ['REF_AREA', 'TIME_PERIOD', 'True_Resilience', 'Exposure', 'GDP_PC', 'FDI', 'Out_Degree_Centrality']
df = df_model[cols].dropna().copy()

# 创建必要的变量
df['ln_Resilience_Inv'] = -np.log(df['True_Resilience'])
df['log_GDP_PC'] = np.log(df['GDP_PC'])

# 设置多重索引
df_indexed = df.set_index(['REF_AREA', 'TIME_PERIOD']).sort_index()

# 生成滞后暴露度
df_indexed['L1_Exposure'] = df_indexed.groupby(level='REF_AREA')['Exposure'].shift(1)

# 清理缺失值并重置索引
df_clean = df_indexed.dropna(subset=['L1_Exposure', 'ln_Resilience_Inv', 'Out_Degree_Centrality']).copy()
df_clean = df_clean.reset_index()

# ============================================================================
# 1. 样本提取：Bottom 33% 子样本 (基于Out_Degree_Centrality)
# ============================================================================
q33 = df_clean['Out_Degree_Centrality'].quantile(0.33)

print(f"\n[样本提取]")
print(f"L1_Centrality（Out_Degree_Centrality）的33%分位数: {q33:.6f}")

df_bottom33 = df_clean[df_clean['Out_Degree_Centrality'] <= q33].copy()
print(f"原始样本量: {len(df_clean)}")
print(f"Bottom 33% 样本量: {len(df_bottom33)}")
print(f"涉及国家数: {df_bottom33['REF_AREA'].nunique()}")
print(f"涉及年份范围: {df_bottom33['TIME_PERIOD'].min()} - {df_bottom33['TIME_PERIOD'].max()}")

# 为模型准备数据
df_bottom33['Constant'] = 1.0
df_model_data = df_bottom33.set_index(['REF_AREA', 'TIME_PERIOD']).copy()

# ============================================================================
# 2. 模型运行：回归A（单向聚类）vs 回归B（双向聚类）
# ============================================================================
print("\n" + "=" * 100)
print("模型回归与统计检验")
print("=" * 100)

# 准备变量
X = df_model_data[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
y = df_model_data['ln_Resilience_Inv']

# ─────────────────────────────────────────────────────────────
# 回归 A：常规防守（单向聚类：仅在国家层面聚类）
# ─────────────────────────────────────────────────────────────
print("\n[回归 A - 常规防守] 单向聚类标准误 (Entity-Only Cluster)")
print("-" * 100)

model_A = PanelOLS(y, X, entity_effects=True, time_effects=True)
res_A = model_A.fit(cov_type='clustered', cluster_entity=True, cluster_time=False)

coef_A = res_A.params['L1_Exposure']
se_A = res_A.std_errors['L1_Exposure']
pval_A = res_A.pvalues['L1_Exposure']
ci_lower_A = coef_A - 1.645 * se_A
ci_upper_A = coef_A + 1.645 * se_A

print(f"L1_Exposure 系数:      {coef_A:.6f}")
print(f"标准误:               {se_A:.6f}")
print(f"90% 置信区间:          [{ci_lower_A:.6f}, {ci_upper_A:.6f}]")
print(f"P 值:                 {pval_A:.6f}")
print(f"显著性:               {'显著 YES' if pval_A < 0.05 else '不显著 NO'} (@5%)")

# ─────────────────────────────────────────────────────────────
# 回归 B：极限防守（双向聚类：国家-年份聚类）
# ─────────────────────────────────────────────────────────────
print("\n[回归 B - 极限防守] 双向聚类标准误 (Two-Way Cluster)")
print("-" * 100)

model_B = PanelOLS(y, X, entity_effects=True, time_effects=True)
res_B = model_B.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

coef_B = res_B.params['L1_Exposure']
se_B = res_B.std_errors['L1_Exposure']
pval_B = res_B.pvalues['L1_Exposure']
ci_lower_B = coef_B - 1.645 * se_B
ci_upper_B = coef_B + 1.645 * se_B

print(f"L1_Exposure 系数:      {coef_B:.6f}")
print(f"标准误:               {se_B:.6f}")
print(f"90% 置信区间:          [{ci_lower_B:.6f}, {ci_upper_B:.6f}]")
print(f"P 值:                 {pval_B:.6f}")
print(f"显著性:               {'显著 YES' if pval_B < 0.05 else '不显著 NO'} (@5%水平)")

# ============================================================================
# 3. 稳健性比较
# ============================================================================
print("\n" + "=" * 100)
print("稳健性对比分析")
print("=" * 100)

se_increase_pct = ((se_B - se_A) / se_A) * 100 if se_A != 0 else 0

print(f"标准误变化:")
print(f"  单向聚类 SE:         {se_A:.6f}")
print(f"  双向聚类 SE:         {se_B:.6f}")
print(f"  增加幅度:           {se_increase_pct:.2f}%")
print(f"\nP 值稳健性:")
print(f"  单向聚类 P:          {pval_A:.6f}")
print(f"  双向聚类 P:          {pval_B:.6f}")

if pval_A < 0.05 and pval_B < 0.05:
    print(f"\n[结论] 结果高度稳健：两种聚类方式下都呈现显著效应")
elif pval_A < 0.05 and pval_B >= 0.05:
    print(f"\n[警告] 双向聚类后失去显著性，存在聚类相关性")
else:
    print(f"\n[信息] 两种方式下都不显著，需要进一步检验")

# ============================================================================
# 4. 绘制森林图 (Errorbar Plot)
# ============================================================================
print("\n" + "=" * 100)
print("绘制森林图")
print("=" * 100)

fig, ax = plt.subplots(figsize=(18.96, 10), dpi=216)

# 数据准备
models = ['常规防守\n单向聚类标准误', '极限防守\n国家-年份双向聚类']
coefficients = [coef_A, coef_B]
ci_lowers = [ci_lower_A, ci_lower_B]
ci_uppers = [ci_upper_A, ci_upper_B]
pvals = [pval_A, pval_B]

# X 轴位置
x_pos = np.arange(len(models))

# 绘制误差棒
colors = ['darkblue', 'darkblue']
for i, (pos, coef, lower, upper, pval) in enumerate(zip(x_pos, coefficients, ci_lowers, ci_uppers, pvals)):
    # 绘制误差棒
    ax.errorbar(pos, coef, yerr=[[coef - lower], [upper - coef]], 
                fmt='o', markersize=16, capsize=12, capthick=3,
                color=colors[i], ecolor=colors[i], elinewidth=3,
                label='90% CI' if i == 0 else '')
    
    # 在点旁标注P值
    sig_marker = '**' if pval < 0.05 else '*' if pval < 0.10 else 'ns'
    ax.text(pos + 0.15, coef, f'P={pval:.3f}\n{sig_marker}', 
            fontsize=13, fontweight='bold', va='center')

# 关键参考线：Y=0（显著性生死线）
ax.axhline(y=0, color='red', linestyle='--', linewidth=3, label='显著性生死线 (Y=0)', zorder=1)

# 图表美化
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=16, fontweight='bold')
ax.set_ylabel('L1_Exposure 系数估计值', fontsize=16, fontweight='bold')
ax.set_xlabel('')

# 设置网格
ax.grid(True, alpha=0.3, linestyle=':', axis='y')
ax.set_axisbelow(True)

# 标题
plt.title('图 2：排除网络溢出干扰的双向聚类稳健性检验', 
          fontsize=18, fontweight='bold', pad=20)

# 图例
ax.legend(loc='upper left', fontsize=14, framealpha=0.95)

# 设置Y轴范围，增加上下空间
y_min = min(ci_lowers) - abs(min(ci_lowers)) * 0.15
y_max = max(ci_uppers) + abs(max(ci_uppers)) * 0.15
ax.set_ylim([y_min, y_max])

# 调整布局
plt.tight_layout()

# 保存为高分辨率PNG（4096x2160）
output_path = os.path.join(base_dir, 'Two_Way_Clustered_SE_Plot.png')
plt.savefig(output_path, dpi=216, bbox_inches='tight', format='png', facecolor='white')
print(f"\n[保存] 图表已保存: {output_path}")
print(f"  • 输出分辨率：4096 × 2160 像素")
print(f"  • 中文字体：已启用 (SimHei)")

plt.show()

# ============================================================================
# 5. 详细输出日志
# ============================================================================
print("\n" + "=" * 100)
print("回归结果详细信息")
print("=" * 100)

print("\n[回归 A 的完整结果]")
print(res_A)

print("\n[回归 B 的完整结果]")
print(res_B)

# ============================================================================
# 6. 保存结果摘要
# ============================================================================
summary_data = {
    '回归类型': ['A: 单向聚类 (Entity Only)', 'B: 双向聚类 (Entity & Time)'],
    'L1_Exposure 系数': [coef_A, coef_B],
    '标准误': [se_A, se_B],
    '90% CI 下限': [ci_lower_A, ci_lower_B],
    '90% CI 上限': [ci_upper_A, ci_upper_B],
    'P 值': [pval_A, pval_B],
    '显著性': ['显著' if pval_A < 0.05 else '不显著', '显著' if pval_B < 0.05 else '不显著']
}

summary_df = pd.DataFrame(summary_data)
summary_output = os.path.join(base_dir, '两向聚类检验结果_摘要.csv')
summary_df.to_csv(summary_output, index=False, encoding='utf-8-sig')
print(f"\n[保存] 摘要结果已保存: {summary_output}")

print("\n" + "=" * 100)
print("分析完成!")
print("=" * 100)
print(f"\n[样本信息]")
print(f"   总样本量: {len(df_bottom33)}")
print(f"   国家数: {df_bottom33['REF_AREA'].nunique()}")
print(f"   年份范围: {df_bottom33['TIME_PERIOD'].min()} - {df_bottom33['TIME_PERIOD'].max()}")
print(f"\n[核心结论]")
print(f"   L1_Exposure在双向聚类下是否依然显著: {'是 YES' if pval_B < 0.05 else '否 NO'}")
print(f"   稳健性风险评估: {'低风险' if abs(se_increase_pct) < 30 else '中等风险' if abs(se_increase_pct) < 50 else '高风险'}")
