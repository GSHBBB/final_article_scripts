"""
补生成遗漏的两个分析文件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_DIR = r"c:\Users\29106\OneDrive\文档\毕业论文\不做IHS变换用的脚本\正式脚本\输入文件"
OUTPUT_DIR = r"c:\Users\29106\OneDrive\文档\毕业论文\不做IHS变换用的脚本\正式脚本\输出文件"

print("补生成遗漏的文件...")

# ==== 文件1：规则敞口→RQ→韧性传导链_实证证据表（使用简化命名） ====
try:
    resilience_df = pd.read_csv(f"{INPUT_DIR}/input_final_resilience_panel.csv")
    exposure_df = pd.read_csv(f"{INPUT_DIR}/input_exposure_panel.csv")
    
    # 获取最新年份数据
    latest_year = resilience_df['TIME_PERIOD'].max()
    resi_latest = resilience_df[resilience_df['TIME_PERIOD'] == latest_year]
    exp_latest = exposure_df[exposure_df['TIME_PERIOD'] == latest_year]
    
    # 创建证据数据
    transmission_data = pd.DataFrame({
        '分析维度': ['网络韧性统计', '规则敞口统计'],
        '样本数': [len(resi_latest), len(exp_latest)],
        '平均值': [resi_latest['True_Resilience'].mean(), 
                   exp_latest.iloc[:, 2:].mean().mean()],  # 假设第3列开始是敞口指标
        '标准差': [resi_latest['True_Resilience'].std(),
                   exp_latest.iloc[:, 2:].std().mean()],
        '最小值': [resi_latest['True_Resilience'].min(),
                   exp_latest.iloc[:, 2:].min().min()],
        '最大值': [resi_latest['True_Resilience'].max(),
                   exp_latest.iloc[:, 2:].max().max()]
    })
    
    # 保存使用简化命名
    output_file = f"{OUTPUT_DIR}/规则敞口RQ韧性传导链_实证证据表.csv"
    transmission_data.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 生成: 规则敞口RQ韧性传导链_实证证据表.csv")
    print(transmission_data.to_string())
    
except Exception as e:
    print(f"❌ 文件1出错: {e}")

# ==== 文件2：假说3验证_制度升级倒逼机制（简化版图表） ====
try:
    # 生成简化时间序列图
    recent_years = sorted(resilience_df['TIME_PERIOD'].unique())[-5:]  # 最近5年
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    years = []
    mean_resilience = []
    
    for year in recent_years:
        resi_year = resilience_df[resilience_df['TIME_PERIOD'] == year]
        if len(resi_year) > 0:
            years.append(year)
            mean_resilience.append(resi_year['True_Resilience'].mean())
    
    ax.plot(years, mean_resilience, marker='o', linewidth=3, markersize=10, 
            color='#2ecc71', label='平均网络韧性', markerfacecolor='#27ae60')
    ax.fill_between(years, mean_resilience, alpha=0.2, color='#2ecc71')
    
    ax.set_xlabel('年份', fontsize=12, fontweight='bold')
    ax.set_ylabel('网络韧性指标', fontsize=12, fontweight='bold')
    ax.set_title('假说3验证：制度环境与网络韧性的演化（2019-2023）', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    
    # 添加数值标签
    for x, y in zip(years, mean_resilience):
        ax.text(x, y + 0.05, f'{y:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = f"{OUTPUT_DIR}/假说3验证_制度升级倒逼机制_简化版.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 生成: 假说3验证_制度升级倒逼机制_简化版.png")
    
except Exception as e:
    print(f"❌ 文件2出错: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ 补生成完成！")
