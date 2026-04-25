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
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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

# 其他非主权国家聚合体代码
OTHER_AGGREGATES = {
    'EU27_2020', 'OECD', 'G20', 'BRICS'
}

print("="*70)
print("韧性等级可视化修复脚本")
print("="*70)

try:
    # 1. 加载原始韧性面板数据
    print("\n📊 加载韧性数据...")
    resilience_df = pd.read_csv(f"{INPUT_DIR}/input_final_resilience_panel.csv")
    centrality_df = pd.read_csv(f"{INPUT_DIR}/input_node_centrality_panel.csv")
    
    print(f"   原始韧性数据: {len(resilience_df)} 行，{len(resilience_df['REF_AREA'].unique())} 个国家")
    print(f"   原始中心度数据: {len(centrality_df)} 行，{len(centrality_df['REF_AREA'].unique())} 个国家")
    
    # 2. 过滤主权国家（仅在画图时）
    print("\n🔍 过滤国家聚合体...")
    
    # 获取并过滤韧性数据
    resi_filtered = resilience_df[
        ~resilience_df['REF_AREA'].isin(WB_MACRO_CODES) &
        ~resilience_df['REF_AREA'].isin(OTHER_AGGREGATES)
    ].copy()
    
    # 获取并过滤中心度数据
    cent_filtered = centrality_df[
        ~centrality_df['REF_AREA'].isin(WB_MACRO_CODES) &
        ~centrality_df['REF_AREA'].isin(OTHER_AGGREGATES)
    ].copy()
    
    print(f"   过滤后韧性数据: {len(resi_filtered)} 行，{len(resi_filtered['REF_AREA'].unique())} 个主权国家")
    print(f"   过滤后中心度数据: {len(cent_filtered)} 行，{len(cent_filtered['REF_AREA'].unique())} 个主权国家")
    
    # 3. 香港标签修改（仅显示）
    print("\n🌏 修改香港标签...")
    
    # 创建国家名称映射
    country_label_map = {}
    
    # 读取中文名称映射（从原始数据中获取）
    unique_areas = resi_filtered[['REF_AREA']].drop_duplicates()
    print(f"   检查到 {len(unique_areas)} 个唯一国家代码")
    
    # 查找香港
    hk_rows = resi_filtered[resi_filtered['REF_AREA'].str.contains('HK', case=False, na=False)]
    if len(hk_rows) > 0:
        print(f"   ✓ 找到香港数据")
        country_label_map['HK'] = '香港（中国）'
    
    # 4. 合并数据并计算韧性等级
    print("\n📈 计算韧性等级...")
    
    resi_latest = resi_filtered[resi_filtered['TIME_PERIOD'] == resi_filtered['TIME_PERIOD'].max()].copy()
    cent_latest = cent_filtered[cent_filtered['TIME_PERIOD'] == cent_filtered['TIME_PERIOD'].max()].copy()
    
    print(f"   韧性latest列: {resi_latest.columns.tolist()}")
    print(f"   中心度latest列: {cent_latest.columns.tolist()}")
    
    # 合并
    combined = resi_latest.merge(
        cent_latest[['REF_AREA', 'Out_Degree_Centrality']], 
        on='REF_AREA', 
        how='inner'
    )
    
    print(f"   合并后列: {combined.columns.tolist()}")
    print(f"   合并后行数: {len(combined)}")
    
    # 按中心度分层
    combined['Resilience_Tier'] = pd.qcut(
        combined['Out_Degree_Centrality'],
        q=4,
        labels=['Tier_1_Edge', 'Tier_2_Main', 'Tier_3_Core', 'Tier_4_Super'],
        duplicates='drop'
    )
    
    # 中文标签
    tier_chinese_map = {
        'Tier_1_Edge': '边缘节点',
        'Tier_2_Main': '主要节点',
        'Tier_3_Core': '核心枢纽',
        'Tier_4_Super': '超级枢纽'
    }
    
    print(f"   完成：{len(combined)} 个主权国家分层")
    
    # 5. 生成时间序列数据用于可视化
    print("\n📊 生成时间序列数据...")
    
    # 重新获取所有年份数据，应用国家过滤
    resi_all_years = resi_filtered.copy()
    cent_all_years = cent_filtered.copy()
    
    # 合并所有年份数据
    combined_all = resi_all_years.merge(
        cent_all_years[['REF_AREA', 'TIME_PERIOD', 'Out_Degree_Centrality']],
        on=['REF_AREA', 'TIME_PERIOD'],
        how='inner'
    )
    
    # 分层
    combined_all['Resilience_Tier'] = combined_all.groupby('TIME_PERIOD').apply(
        lambda group: pd.qcut(
            group['Out_Degree_Centrality'],
            q=4,
            labels=['Tier_1_Edge', 'Tier_2_Main', 'Tier_3_Core', 'Tier_4_Super'],
            duplicates='drop'
        )
    ).reset_index(level=0, drop=True)
    
    # 计算每个tier的平均韧性
    tier_resilience = combined_all.groupby(['TIME_PERIOD', 'Resilience_Tier']).agg({
        'True_Resilience': ['mean', 'std', 'count'],
        'REF_AREA': lambda x: len(x)
    }).round(4)
    
    print(f"   生成跨度: {combined_all['TIME_PERIOD'].min()}-{combined_all['TIME_PERIOD'].max()} 年")
    
    # 6. 生成可视化
    print("\n🎨 生成修复后的可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('全球数字贸易网络韧性等级演化（仅主权国家，已修正）', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    # Panel 1: 各Tier平均韧性演化
    ax1 = axes[0, 0]
    years_unique = sorted(combined_all['TIME_PERIOD'].unique())
    
    colors_tier = {
        'Tier_1_Edge': '#e74c3c',
        'Tier_2_Main': '#f39c12',
        'Tier_3_Core': '#3498db',
        'Tier_4_Super': '#2ecc71'
    }
    
    for tier in ['Tier_1_Edge', 'Tier_2_Main', 'Tier_3_Core', 'Tier_4_Super']:
        tier_data = combined_all[combined_all['Resilience_Tier'] == tier].groupby('TIME_PERIOD')['True_Resilience'].mean()
        ax1.plot(tier_data.index, tier_data.values, marker='o', linewidth=2, 
                label=tier_chinese_map[tier], color=colors_tier[tier], markersize=6)
    
    ax1.set_xlabel('年份', fontweight='bold')
    ax1.set_ylabel('平均韧性指标', fontweight='bold')
    ax1.set_title('韧性等级演化趋势', fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: 各Tier国家数量演化
    ax2 = axes[0, 1]
    for tier in ['Tier_1_Edge', 'Tier_2_Main', 'Tier_3_Core', 'Tier_4_Super']:
        tier_count = combined_all[combined_all['Resilience_Tier'] == tier].groupby('TIME_PERIOD')['REF_AREA'].nunique()
        ax2.plot(tier_count.index, tier_count.values, marker='s', linewidth=2,
                label=tier_chinese_map[tier], color=colors_tier[tier], markersize=6)
    
    ax2.set_xlabel('年份', fontweight='bold')
    ax2.set_ylabel('国家数量', fontweight='bold')
    ax2.set_title('各等级国家数量演化', fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: 最新年份Tier分布（箱线图）
    ax3 = axes[1, 0]
    latest_year = combined_all['TIME_PERIOD'].max()
    latest_data = combined_all[combined_all['TIME_PERIOD'] == latest_year]
    
    tier_order = ['Tier_1_Edge', 'Tier_2_Main', 'Tier_3_Core', 'Tier_4_Super']
    tier_labels = [tier_chinese_map[t] for t in tier_order]
    
    plot_data = [latest_data[latest_data['Resilience_Tier'] == tier]['True_Resilience'].values 
                 for tier in tier_order]
    
    bp = ax3.boxplot(plot_data, labels=tier_labels, patch_artist=True)
    for patch, tier in zip(bp['boxes'], tier_order):
        patch.set_facecolor(colors_tier[tier])
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('韧性指标', fontweight='bold')
    ax3.set_title(f'{latest_year}年韧性等级分布', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: 统计表
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    summary_data = []
    for tier in tier_order:
        tier_subset = latest_data[latest_data['Resilience_Tier'] == tier]
        summary_data.append([
            tier_chinese_map[tier],
            len(tier_subset),
            f"{tier_subset['True_Resilience'].mean():.3f}",
            f"{tier_subset['Out_Degree_Centrality'].mean():.3f}",
            f"{tier_subset['True_Resilience'].std():.3f}"
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['等级', '国家数', '平均韧性', '平均中心度', '韧性std'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.15, 0.2, 0.2, 0.2])
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ 保存: {output_path.name}")
    
    plt.close()
    
    # 7. 生成数据表（用于验证）
    print("\n📋 生成验证数据表...")
    
    validation_table = combined_all[combined_all['TIME_PERIOD'] == latest_year][
        ['REF_AREA', 'Out_Degree_Centrality', 'Resilience_Tier', 'True_Resilience']
    ].sort_values('Out_Degree_Centrality', ascending=False)
    
    # 应用标签映射
    validation_table['Resilience_Tier_CN'] = validation_table['Resilience_Tier'].map(tier_chinese_map)
    validation_table['Country_Label'] = validation_table['REF_AREA'].map(country_label_map).fillna(validation_table['REF_AREA'])
    
    # 保存验证表
    verify_path = OUTPUT_DIR / "韧性等级_修复验证表.csv"
    validation_table[[
        'Country_Label', 'Resilience_Tier_CN', 'Out_Degree_Centrality', 'True_Resilience'
    ]].to_csv(verify_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 保存: {verify_path.name}")
    
    # 统计信息
    print("\n✨ 修复完成！统计信息：")
    print(f"   ✓ 过滤国家聚合体: {len(resilience_df['REF_AREA'].unique()) - len(resi_filtered['REF_AREA'].unique())} 个聚合体已移除")
    print(f"   ✓ 保留主权国家: {len(resi_filtered['REF_AREA'].unique())} 个")
    print(f"   ✓ 香港改为: '香港（中国）'")
    print(f"   ✓ 时间跨度: {combined_all['TIME_PERIOD'].min()}-{combined_all['TIME_PERIOD'].max()}")
    print(f"   ✓ 新图表位置: {OUTPUT_DIR.name}/")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
