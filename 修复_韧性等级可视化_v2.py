# -*- coding: utf-8 -*-
"""
修复韧性等级可视化
- 过滤掉国家聚合体（只在画图部分，不修改原始文件）
- 香港改为"香港（中国）"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import io

# 设置标准输出编码
if sys.platform == 'win32':
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

INPUT_DIR = Path(r"c:\Users\29106\OneDrive\文档\毕业论文\不做IHS变换用的脚本\正式脚本\输入文件")
OUTPUT_DIR = Path(r"c:\Users\29106\OneDrive\文档\毕业论文\不做IHS变换用的脚本\正式脚本\观测的辅助数据\2_核心权力变迁与异质性")

# 世界银行宏观区域代码（需要过滤的国家聚合体）
WB_MACRO_CODES = {
    'ARB', 'CSS', 'CEB', 'EAR', 'EAS', 'EAP', 'EMU', 'ECA', 'ECS', 'EUU',
    'FCS', 'HIC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN',
    'LDC', 'LIC', 'LMC', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PST',
    'PRE', 'SAS', 'SSA', 'SSF', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS',
    'UMC', 'WLD', 'WXD'
}

OTHER_AGGREGATES = {
    'EU27_2020', 'OECD', 'G20', 'BRICS'
}

print("="*70)
print("[TASK] 韧性等级可视化修复脚本")
print("="*70)

try:
    # 1. 加载原始韧性面板数据
    print("\n[STEP 1] 加载韧性数据...")
    resilience_df = pd.read_csv(str(INPUT_DIR / "input_final_resilience_panel.csv"))
    centrality_df = pd.read_csv(str(INPUT_DIR / "input_node_centrality_panel.csv"))
    
    print("  - 原始韧性数据: {} 行，{} 个国家".format(len(resilience_df), len(resilience_df['REF_AREA'].unique())))
    print("  - 原始中心度数据: {} 行，{} 个国家".format(len(centrality_df), len(centrality_df['REF_AREA'].unique())))
    
    # 2. 过滤主权国家（仅在画图时）
    print("\n[STEP 2] 过滤国家聚合体...")
    
    resi_filtered = resilience_df[
        ~resilience_df['REF_AREA'].isin(WB_MACRO_CODES) &
        ~resilience_df['REF_AREA'].isin(OTHER_AGGREGATES)
    ].copy()
    
    cent_filtered = centrality_df[
        ~centrality_df['REF_AREA'].isin(WB_MACRO_CODES) &
        ~centrality_df['REF_AREA'].isin(OTHER_AGGREGATES)
    ].copy()
    
    print("  - 过滤后韧性数据: {} 行，{} 个主权国家".format(len(resi_filtered), len(resi_filtered['REF_AREA'].unique())))
    print("  - 过滤后中心度数据: {} 行，{} 个主权国家".format(len(cent_filtered), len(cent_filtered['REF_AREA'].unique())))
    
    # 3. 准备合并数据 - 时间序列
    print("\n[STEP 3] 准备时间序列数据...")
    
    combined_all = resi_filtered.merge(
        cent_filtered,
        on=['REF_AREA', 'TIME_PERIOD'],
        how='inner',
        suffixes=('', '_cent')
    )
    
    print("  - 合并后数据: {} 行".format(len(combined_all)))
    print("  - 时间跨度: {} - {}".format(combined_all['TIME_PERIOD'].min(), combined_all['TIME_PERIOD'].max()))
    
    # 4. 分层（每年单独分层）
    print("\n[STEP 4] 计算韧性等级分层...")
    
    def assign_tier(group):
        return pd.qcut(
            group['Out_Degree_Centrality'],
            q=4,
            labels=['Tier_1_Edge', 'Tier_2_Main', 'Tier_3_Core', 'Tier_4_Super'],
            duplicates='drop'
        )
    
    combined_all['Resilience_Tier'] = combined_all.groupby('TIME_PERIOD', group_keys=False).apply(assign_tier)
    
    print("  - 分层完成")
    
    # 5. 生成可视化
    print("\n[STEP 5] 生成可视化（4面板）...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Global Digital Trade Network Resilience Level Evolution (Sovereign Nations Only)\n全球数字贸易网络韧性等级演化（仅主权国家，已修正）', 
                 fontsize=13, fontweight='bold', y=0.995)
    
    # 中英文标签
    tier_label_map = {
        'Tier_1_Edge': 'Edge (边缘节点)',
        'Tier_2_Main': 'Main (主要节点)',
        'Tier_3_Core': 'Core (核心枢纽)',
        'Tier_4_Super': 'Super (超级枢纽)'
    }
    
    colors_tier = {
        'Tier_1_Edge': '#e74c3c',
        'Tier_2_Main': '#f39c12',
        'Tier_3_Core': '#3498db',
        'Tier_4_Super': '#2ecc71'
    }
    
    # Panel 1: 各层平均韧性演化
    ax1 = axes[0, 0]
    for tier in ['Tier_1_Edge', 'Tier_2_Main', 'Tier_3_Core', 'Tier_4_Super']:
        tier_data = combined_all[combined_all['Resilience_Tier'] == tier].groupby('TIME_PERIOD')['True_Resilience'].mean()
        ax1.plot(tier_data.index, tier_data.values, marker='o', linewidth=2, 
                label=tier_label_map[tier], color=colors_tier[tier], markersize=6)
    
    ax1.set_xlabel('Year', fontweight='bold')
    ax1.set_ylabel('Avg Resilience', fontweight='bold')
    ax1.set_title('Resilience Level Evolution', fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: 各层国家数量演化
    ax2 = axes[0, 1]
    for tier in ['Tier_1_Edge', 'Tier_2_Main', 'Tier_3_Core', 'Tier_4_Super']:
        tier_count = combined_all[combined_all['Resilience_Tier'] == tier].groupby('TIME_PERIOD')['REF_AREA'].nunique()
        ax2.plot(tier_count.index, tier_count.values, marker='s', linewidth=2,
                label=tier_label_map[tier], color=colors_tier[tier], markersize=6)
    
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel('Number of Countries', fontweight='bold')
    ax2.set_title('Evolution of Countries per Tier', fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: 最新年份Tier分布（箱线图）
    ax3 = axes[1, 0]
    latest_year = combined_all['TIME_PERIOD'].max()
    latest_data = combined_all[combined_all['TIME_PERIOD'] == latest_year]
    
    tier_order = ['Tier_1_Edge', 'Tier_2_Main', 'Tier_3_Core', 'Tier_4_Super']
    tier_labels = [tier_label_map[t] for t in tier_order]
    
    plot_data = [latest_data[latest_data['Resilience_Tier'] == tier]['True_Resilience'].values 
                 for tier in tier_order]
    
    bp = ax3.boxplot(plot_data, labels=tier_labels, patch_artist=True)
    for patch, tier in zip(bp['boxes'], tier_order):
        patch.set_facecolor(colors_tier[tier])
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Resilience Score', fontweight='bold')
    ax3.set_title('Resilience Distribution in {}'.format(latest_year), fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: 统计表
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    summary_data = []
    for tier in tier_order:
        tier_subset = latest_data[latest_data['Resilience_Tier'] == tier]
        summary_data.append([
            tier_label_map[tier].split('(')[1].rstrip(')') if '(' in tier_label_map[tier] else tier_label_map[tier],
            len(tier_subset),
            '{:.3f}'.format(tier_subset['True_Resilience'].mean()),
            '{:.3f}'.format(tier_subset['Out_Degree_Centrality'].mean()),
            '{:.3f}'.format(tier_subset['True_Resilience'].std())
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Tier', 'N', 'Avg Resilience', 'Avg Centrality', 'Std Dev'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.12, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 着色表头
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 着色数据行
    for i, tier in enumerate(tier_order):
        table[(i+1, 0)].set_facecolor(colors_tier[tier])
        table[(i+1, 0)].set_alpha(0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = OUTPUT_DIR / "2005_2023年韧性等级国家趋势_可视化_修复版.png"
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
    print("  - SAVED: {}".format(output_path.name))
    
    plt.close()
    
    # 6. 生成验证数据表
    print("\n[STEP 6] 生成验证数据表...")
    
    validation_table = latest_data[[
        'REF_AREA', 'Out_Degree_Centrality', 'Resilience_Tier', 'True_Resilience'
    ]].sort_values('Out_Degree_Centrality', ascending=False)
    
    validation_table.columns = ['Country_Code', 'Centrality', 'Tier', 'Resilience']
    validation_table['Tier_CN'] = validation_table['Tier'].map({
        'Tier_1_Edge': '边缘节点',
        'Tier_2_Main': '主要节点',
        'Tier_3_Core': '核心枢纽',
        'Tier_4_Super': '超级枢纽'
    })
    
    # 应用香港标签 - 注意香港代码是HKG或HK
    validation_table['Country_Label'] = validation_table['Country_Code']
    validation_table.loc[validation_table['Country_Code'] == 'HK', 'Country_Label'] = 'Hong Kong (China)'
    validation_table.loc[validation_table['Country_Code'] == 'HKG', 'Country_Label'] = 'Hong Kong (China)'
    
    verify_path = OUTPUT_DIR / "韧性等级_修复验证表.csv"
    validation_table[['Country_Label', 'Tier_CN', 'Centrality', 'Resilience']].to_csv(str(verify_path), index=False, encoding='utf-8-sig')
    print("  - SAVED: {}".format(verify_path.name))
    
    # 统计信息
    print("\n" + "="*70)
    print("[SUCCESS] 修复完成！")
    print("="*70)
    print("\nSUMMARY:")
    print("  - Removed aggregates: {} aggregate regions".format(
        len(resilience_df['REF_AREA'].unique()) - len(resi_filtered['REF_AREA'].unique())))
    print("  - Retained sovereign nations: {}".format(len(resi_filtered['REF_AREA'].unique())))
    print("  - Hong Kong label: Changed to 'Hong Kong (China)'")
    print("  - Time span: {} - {}".format(combined_all['TIME_PERIOD'].min(), combined_all['TIME_PERIOD'].max()))
    print("  - Output directory: {}".format(OUTPUT_DIR.name))
    print("\nOUTPUT FILES:")
    print("  - 2005_2023年韧性等级国家趋势_可视化_修复版.png")
    print("  - 韧性等级_修复验证表.csv")
    print("\n" + "="*70)
    
except Exception as e:
    print("\n[ERROR] {}".format(str(e)))
    import traceback
    traceback.print_exc()
