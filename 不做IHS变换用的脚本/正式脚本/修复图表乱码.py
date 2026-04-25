"""
修复假说3验证图表中的中文乱码问题
确保标题、坐标轴标签等中文字体正确显示
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

INPUT_DIR = r"c:\Users\29106\OneDrive\文档\毕业论文\不做IHS变换用的脚本\正式脚本\输入文件"
OUTPUT_DIR = r"c:\Users\29106\OneDrive\文档\毕业论文\不做IHS变换用的脚本\正式脚本\观测的辅助数据\3_假说验证核心表格"

print("="*60)
print("中文乱码修复: 假说3验证图表")
print("="*60)

try:
    # 设置中文字体（多种字体备选）
    font_candidates = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
    
    # 尝试找到可用的中文字体
    available_fonts = []
    for font_name in font_candidates:
        try:
            font = fm.findfont(fm.FontProperties(family=font_name))
            available_fonts.append(font_name)
            print(f"✓ 找到字体: {font_name}")
            break
        except:
            continue
    
    # 使用第一个可用的字体
    if available_fonts:
        plt.rcParams['font.sans-serif'] = available_fonts[0]
        print(f"使用字体: {available_fonts[0]}")
    else:
        # 如果没有中文字体，使用系统默认
        plt.rcParams['font.sans-serif'] = ['SimHei']
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 读取韧性数据
    resilience_df = pd.read_csv(f"{INPUT_DIR}/input_final_resilience_panel.csv")
    
    # 获取最近5年数据
    recent_years = sorted(resilience_df['TIME_PERIOD'].unique())[-5:]
    
    years = []
    mean_resilience = []
    std_resilience = []
    
    for year in recent_years:
        resi_year = resilience_df[resilience_df['TIME_PERIOD'] == year]
        if len(resi_year) > 0:
            years.append(year)
            mean_resilience.append(resi_year['True_Resilience'].mean())
            std_resilience.append(resi_year['True_Resilience'].std())
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制扩展带
    ax.fill_between(years, 
                     np.array(mean_resilience) - np.array(std_resilience),
                     np.array(mean_resilience) + np.array(std_resilience),
                     alpha=0.25, color='#2ecc71', label='±标准差区间')
    
    # 绘制主线
    ax.plot(years, mean_resilience, marker='o', linewidth=3.5, markersize=11, 
            color='#27ae60', label='平均网络韧性', markerfacecolor='#1abc9c')
    
    # 设置坐标轴标签（确保中文）
    ax.set_xlabel('年份', fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('网络韧性指标', fontsize=13, fontweight='bold', color='#2c3e50')
    
    # 设置标题
    title_text = '假说3验证：制度环境与网络韧性的演化（2019-2023）'
    ax.set_title(title_text, fontsize=15, fontweight='bold', color='#34495e', pad=20)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 添加数值标签
    for x, y, std in zip(years, mean_resilience, std_resilience):
        ax.text(x, y + std + 0.08, f'{y:.3f}', 
               ha='center', fontsize=10, fontweight='bold', color='#27ae60')
    
    # 图例
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95, edgecolor='#95a5a6')
    
    # 设置Y轴范围
    ax.set_ylim(min(mean_resilience) - 0.5, max(mean_resilience) + 0.8)
    
    # 美化背景
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # 调整边距
    plt.tight_layout()
    
    # 保存图表
    output_path = f"{OUTPUT_DIR}/假说3验证_制度升级倒逼机制_简化版.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ 图表已修复并保存: {output_path}")
    print(f"   分辨率: 300 DPI")
    print(f"   格式: PNG（已优化）")
    
    plt.close()
    
    # 验证文件
    import os
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"   文件大小: {file_size:.1f} KB")
        print(f"\n✨ 中文乱码修复完成！")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
