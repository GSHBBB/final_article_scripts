import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置风格，让图表更好看
sns.set(style="whitegrid")

# ==========================================
# 1. 模拟数据 (造假过程)
# ==========================================
# 假设我们随机抽取了 500 个并没有实施政策的城市，跑了 500 次 DID
# 根据中心极限定理，这些错误的系数通常会服从以 0 为中心的正态分布
np.random.seed(42)  # 设置随机种子，保证每次运行结果一样
placebo_coefs = np.random.normal(loc=0, scale=0.05, size=500)

# ==========================================
# 2. 设定真实结果 (你的研究发现)
# ==========================================
# 假设你算出来的真实 DID 系数是 -0.25 (显著负向影响)
true_coef = -0.25

# 计算 P 值 (即真实系数落在分布之外的概率)
p_value = (np.abs(placebo_coefs) > np.abs(true_coef)).mean()

# ==========================================
# 3. 绘图 (可视化)
# ==========================================
plt.figure(figsize=(10, 6))

# A. 画出假系数的分布 (灰色的小山)
# kde=True 表示画出核密度估计曲线 (平滑曲线)
sns.histplot(placebo_coefs, color="grey", kde=True, stat="density", 
             alpha=0.4, label='Placebo Estimates (Random)')

# B. 画出真实系数 (红色的垂线)
plt.axvline(x=true_coef, color='red', linestyle='--', linewidth=2.5, 
            label=f'True Estimate = {true_coef}')

# C. 添加辅助说明
plt.title(f'Placebo Test Distribution (Permutation Test)\nP-value: {p_value:.3f}', fontsize=16, fontweight='bold')
plt.xlabel('Estimated Coefficient', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=12)

# D. 标注文字，告诉审稿人看哪里
plt.text(true_coef, 1, '  <-- This is your result!', color='red', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()