import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats.mstats import winsorize
from pathlib import Path
import os

# ============================================================================
# 描述性统计脚本：基于最终确定的变量组合
# ============================================================================

# 使用脚本所在目录，避免硬编码绝对路径
script_dir = Path(__file__).resolve().parent
base_dir = script_dir
source_data_dir = script_dir / "输出文件"
output_data_dir = script_dir / "输出文件_不强制InternetUse"
output_data_dir.mkdir(parents=True, exist_ok=True)


def resolve_existing_file(*candidates: Path) -> Path:
    """在多个候选路径中返回第一个存在的文件，避免硬编码路径导致读取失败。"""
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "未找到目标文件，请检查以下候选路径：\n" + "\n".join(str(p) for p in candidates)
    )

# 1. 读取主面板数据
print("读取主面板数据...")
master_panel_path = resolve_existing_file(
    output_data_dir / "output_Master_Panel.csv",
    source_data_dir / "output_Master_Panel.csv",
    base_dir / "output_Master_Panel.csv"
)
df = pd.read_csv(master_panel_path)

# 2. 读取并合并节点中心性数据
print("读取并合并节点中心性数据...")
centrality_path = resolve_existing_file(
    output_data_dir / "output_node_centrality_panel.csv",
    source_data_dir / "output_node_centrality_panel.csv",
    source_data_dir / "output_final_resilience_panel.csv",
    output_data_dir / "output_final_resilience_panel.csv"
)
centrality_df = pd.read_csv(centrality_path)
df = df.merge(centrality_df, on=['REF_AREA', 'TIME_PERIOD'], how='left')

# 3. 计算新变量
print("计算新变量...")
df['ln_Resilience_Inv'] = -np.log(df['True_Resilience'])  # 网络韧性对数（负对数）
df['log_GDP_PC'] = np.log(df['GDP_PC'])  # 人均GDP对数

# 4. 生成滞后项
print("生成滞后项...")
df = df.sort_values(['REF_AREA', 'TIME_PERIOD'])
df['L1_Exposure'] = df.groupby('REF_AREA')['Exposure'].shift(1)  # 敞口指数滞后1期

# 5. 选择最终变量组合（剔除Internet_Use）
variables = [
    'ln_Resilience_Inv',    # 被解释变量：网络韧性对数
    'Exposure',             # 核心自变量：敞口指数

    'L1_Exposure',          # 敞口指数滞后项
    'log_GDP_PC',           # 控制变量：人均GDP对数
    'FDI',                  # 控制变量：FDI
    'RQ',                   # 机制变量：监管质量
    'GE',                   # 机制变量：政府效能
    'Out_Degree_Centrality' # 异质性变量：出度中心性
]

# 6. 数据清理：删除缺失值
print("数据清理...")
df_clean = df[variables].dropna()
print(f"最终样本量：{len(df_clean)} 观测值")

# 7. 执行描述性统计
print("\n执行描述性统计...")
desc = df_clean.describe()

# 8. 提取关键统计量
result = desc.loc[['count', 'mean', 'std', 'min', 'max']]

# 9. 格式化输出
print("\n" + "="*80)
print("描述性统计结果")
print("="*80)

# 定义变量中文名称映射
var_names = {
    'ln_Resilience_Inv': '网络韧性对数',
    'Exposure': '敞口指数',
    'L1_Exposure': '敞口指数滞后项',
    'log_GDP_PC': '人均GDP对数',
    'FDI': 'FDI',
    'RQ': '监管质量',
    'GE': '政府效能',
    'Out_Degree_Centrality': '出度中心性'
}

# 打印表格
print(f"{'变量':<12} {'观测值':<8} {'均值':<10} {'标准差':<8} {'最小值':<10} {'最大值':<10}")
print("-" * 70)

for var in variables:
    if var in result.columns:
        count = int(result.loc['count', var])
        mean = f"{result.loc['mean', var]:.3f}"

        std = f"{result.loc['std', var]:.3f}"
        min_val = f"{result.loc['min', var]:.3f}"
        max_val = f"{result.loc['max', var]:.3f}"
        print(f"{var_names[var]:<12} {count:<8} {mean:<10} {std:<8} {min_val:<10} {max_val:<10}")

print("\n" + "="*80)

# 可选：保存结果到CSV文件
output_file = output_data_dir / "output_descriptive_stats.csv"
result.to_csv(output_file, encoding='utf-8-sig')
print(f"描述性统计结果已保存到：{output_file}")

print("\n脚本执行完成！")

# ============================================================================
# Jointplot 对比绘图：原始未处理 vs 科学处理后
# ============================================================================

# 10. 从 Master_Panel.csv 提取绘图所需变量
print("\n准备 Jointplot 对比数据...")
plot_df = pd.read_csv(master_panel_path, usecols=['Exposure', 'True_Resilience']).dropna()

# 按论文设定构造校正后的韧性变量 ln_Resilience_Inv = -ln(True_Resilience)
# 仅对 True_Resilience > 0 的观测做对数变换，避免无定义值。
plot_df = plot_df[plot_df['True_Resilience'] > 0].copy()
plot_df['ln_Resilience_Inv'] = -np.log(plot_df['True_Resilience'])

# ---- 设置画图风格 ----
sns.set_theme(style="white")
# 优先使用SimHei显示中文，否则使用Times New Roman
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

output_dir = output_data_dir
os.makedirs(output_dir, exist_ok=True)

# 11. 准备图 B 所需数据：对数反转 + 1% 双侧缩尾处理
plot_w = plot_df[['Exposure', 'ln_Resilience_Inv']].copy()
plot_w['Exposure'] = np.asarray(winsorize(plot_w['Exposure'], limits=[0.01, 0.01]))
plot_w['ln_Resilience_Inv'] = np.asarray(winsorize(plot_w['ln_Resilience_Inv'], limits=[0.01, 0.01]))

# 12. 整合为一张完整对比图（左右并排，每侧含边缘分布）
print("绘制整合对比图（原始 vs 缩尾）...")
fig = plt.figure(figsize=(18.96, 10), constrained_layout=True, dpi=216)
outer = fig.add_gridspec(1, 2, wspace=0.18)


def draw_joint_like_panel(gs_slot, data, x_col, y_col, title, x_label, y_label):
    """在同一画布中绘制 jointplot 风格子图（主图+上/右边缘分布）。"""
    sub = gs_slot.subgridspec(
        2,
        2,
        height_ratios=[1, 4],
        width_ratios=[4, 1],
        hspace=0.05,
        wspace=0.05,
    )
    ax_top = fig.add_subplot(sub[0, 0])
    ax_main = fig.add_subplot(sub[1, 0])
    ax_right = fig.add_subplot(sub[1, 1])

    # 主图：散点 + 回归拟合线
    sns.regplot(
        data=data,
        x=x_col,
        y=y_col,
        ax=ax_main,
        scatter_kws={'alpha': 0.3, 's': 20, 'color': '#4C72B0'},
        line_kws={'color': 'red', 'linewidth': 2},
        ci=95,
    )
    ax_main.set_title(title, fontsize=12, pad=8)
    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)
    ax_main.grid(False)

    # 上边缘分布：X 直方图 + KDE
    sns.histplot(data=data, x=x_col, bins=30, stat='density', alpha=0.4, color='#4C72B0', ax=ax_top)
    sns.kdeplot(data=data, x=x_col, color='red', linewidth=1.5, ax=ax_top)
    ax_top.set_xlabel("")
    ax_top.set_ylabel("Density")
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.grid(False)

    # 右边缘分布：Y 直方图 + KDE
    sns.histplot(data=data, y=y_col, bins=30, stat='density', alpha=0.4, color='#4C72B0', ax=ax_right)
    sns.kdeplot(data=data, y=y_col, color='red', linewidth=1.5, ax=ax_right)
    ax_right.set_xlabel("Density")
    ax_right.set_ylabel("")
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.grid(False)


draw_joint_like_panel(
    outer[0],
    plot_df,
    'Exposure',
    'True_Resilience',
    'Raw Unprocessed Volatility Distribution',
    'Exposure',
    'True Resilience',
)

draw_joint_like_panel(
    outer[1],
    plot_w,
    'Exposure',
    'ln_Resilience_Inv',
    'Log-Inverted and 1% Winsorized Distribution',
    'Exposure (1% Winsorized)',
    'ln_Resilience_Inv (1% Winsorized)',
)

fig.suptitle('Jointplot Comparison: Raw vs Winsorized', fontsize=14, y=0.98)

compare_file = output_dir / "Jointplot_Comparison.png"
fig.savefig(compare_file, dpi=216, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"整合对比图已保存：{compare_file}")
print(f"  • 输出分辨率：4096 × 2160 像素")
print(f"  • 中文字体：已启用 (SimHei)")
print("\nJointplot 对比图绘制完成！")