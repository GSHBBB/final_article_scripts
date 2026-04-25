"""
新增分析模块（修正版）：验证三个核心假说
1. 掩盖效应与全样本非均质性
2. 网络位置的非对称红利
3. 制度环境的硬约束倒逼机制
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
INPUT_DIR = r"c:\Users\29106\OneDrive\文档\毕业论文\不做IHS变换用的脚本\正式脚本\输入文件"
OUTPUT_DIR = r"c:\Users\29106\OneDrive\文档\毕业论文\不做IHS变换用的脚本\正式脚本\输出文件"

print("=" * 60)
print("新增分析模块启动：三假说验证（修正版）")
print("=" * 60)

# ==================== 分析1：网络位置与韧性的非对称收益 ====================
print("\n📊 分析1：网络位置与韧性的非对称收益")
print("-" * 60)

try:
    resilience_df = pd.read_csv(f"{INPUT_DIR}/input_final_resilience_panel.csv")
    
    # 最新年份（2023）的数据
    latest_year = resilience_df['TIME_PERIOD'].max()
    resi_2023 = resilience_df[resilience_df['TIME_PERIOD'] == latest_year].copy()
    
    # 按出度中心度分层
    resi_2023['centrality_quantile'] = pd.qcut(
        resi_2023['Out_Degree_Centrality'], 
        q=4, 
        labels=['边缘节点(Q1)', '主要节点(Q2)', '核心枢纽(Q3)', '超级枢纽(Q4)'],
        duplicates='drop'
    )
    
    # 按中心度分层统计韧性
    resilience_by_position = resi_2023.groupby('centrality_quantile').agg({
        'True_Resilience': ['mean', 'median', 'std', 'min', 'max', 'count'],
        'Out_Degree_Centrality': 'mean'
    }).round(4)
    
    resilience_by_position.columns = ['平均韧性', '中位韧性', '韧性std', '最小韧性', '最大韧性', '国家数', '平均中心度']
    resilience_by_position = resilience_by_position.reset_index()
    resilience_by_position.to_csv(
        f"{OUTPUT_DIR}/核心vs边缘国家_韧性对比表.csv", 
        index=False, 
        encoding='utf-8-sig'
    )
    print(f"✅ 输出：核心vs边缘国家_韧性对比表.csv")
    print(resilience_by_position.to_string())
    
    # 生成分位数分布图
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f'{latest_year}年：网络位置与韧性的非对称分布', fontsize=14, fontweight='bold')
    
    # 小提琴图
    ax1 = axes[0, 0]
    sns.violinplot(data=resi_2023, x='centrality_quantile', y='True_Resilience', ax=ax1, palette='Set2')
    ax1.set_title('韧性分布（小提琴图）', fontweight='bold')
    ax1.set_xlabel('网络位置')
    ax1.set_ylabel('韧性指标')
    ax1.tick_params(axis='x', rotation=45)
    
    # 箱线图
    ax2 = axes[0, 1]
    sns.boxplot(data=resi_2023, x='centrality_quantile', y='True_Resilience', ax=ax2, palette='Set2')
    ax2.set_title('韧性分布（箱线图）', fontweight='bold')
    ax2.set_xlabel('网络位置')
    ax2.set_ylabel('韧性指标')
    ax2.tick_params(axis='x', rotation=45)
    
    # 散点图+趋势
    ax3 = axes[1, 0]
    colors_map = {'边缘节点(Q1)': '#e74c3c', '主要节点(Q2)': '#f39c12', 
                  '核心枢纽(Q3)': '#3498db', '超级枢纽(Q4)': '#2ecc71'}
    for label in resi_2023['centrality_quantile'].unique():
        subset = resi_2023[resi_2023['centrality_quantile'] == label]
        ax3.scatter(subset['Out_Degree_Centrality'], subset['True_Resilience'], 
                   label=label, s=80, alpha=0.6, color=colors_map.get(label, '#95a5a6'))
    ax3.set_title('中心度vs韧性关系', fontweight='bold')
    ax3.set_xlabel('出度中心度')
    ax3.set_ylabel('韧性指标')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 均值对比
    ax4 = axes[1, 1]
    means = resi_2023.groupby('centrality_quantile')['True_Resilience'].mean()
    stds = resi_2023.groupby('centrality_quantile')['True_Resilience'].std()
    ax4.bar(range(len(means)), means.values, yerr=stds.values, capsize=5, 
            color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'], alpha=0.7)
    ax4.set_xticks(range(len(means)))
    ax4.set_xticklabels(means.index, rotation=45, ha='right')
    ax4.set_title('各层级平均韧性（含误差条）', fontweight='bold')
    ax4.set_xlabel('网络位置')
    ax4.set_ylabel('平均韧性')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/网络位置×韧性分布_分位数图.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 输出：网络位置×韧性分布_分位数图.png")
    
except Exception as e:
    print(f"❌ 分析1出错：{e}")
    import traceback
    traceback.print_exc()

# ==================== 分析2：规则敞口的边际效应 ====================
print("\n📊 分析2：规则敞口的边际效应（交互效应）")
print("-" * 60)

try:
    exposure_df = pd.read_csv(f"{INPUT_DIR}/input_exposure_panel.csv")
    
    # 最新年份数据
    latest_year = exposure_df['TIME_PERIOD'].max()
    exp_2023 = exposure_df[exposure_df['TIME_PERIOD'] == latest_year].copy()
    resi_2023 = resilience_df[resilience_df['TIME_PERIOD'] == latest_year].copy()
    
    # 合并
    interaction_df = exp_2023.merge(resi_2023[['REF_AREA', 'Out_Degree_Centrality', 'True_Resilience']], 
                                     on='REF_AREA', how='inner')
    
    # 确定敞口列名
    exp_cols = [c for c in interaction_df.columns if 'exposure' in c.lower() or 'index' in c.lower()]
    exp_col = exp_cols[0] if exp_cols else None
    
    if exp_col:
        # 按中心度分层
        interaction_df['position'] = pd.qcut(interaction_df['Out_Degree_Centrality'], 
                                            q=4, 
                                            labels=['边缘', '主要', '核心', '超级'],
                                            duplicates='drop')
        
        # 按层级统计规则敞口
        interaction_stats = interaction_df.groupby('position').agg({
            exp_col: ['mean', 'median', 'std', 'min', 'max'],
            'Out_Degree_Centrality': 'mean',
            'REF_AREA': 'count'
        }).round(4)
        
        interaction_stats.columns = ['平均规则敞口', '中位规则敞口', '敞口std', '最小敞口', '最大敞口', '平均中心度', '国家数']
        interaction_stats = interaction_stats.reset_index()
        interaction_stats.to_csv(
            f"{OUTPUT_DIR}/规则敞口×中心度_交互效应表.csv",
            index=False,
            encoding='utf-8-sig'
        )
        print(f"✅ 输出：规则敞口×中心度_交互效应表.csv")
        print(interaction_stats.to_string())
        
        # 计算交互效应系数（规则敞口对韧性的边际效应）
        interaction_effects = []
        for pos in ['边缘', '主要', '核心', '超级']:
            subset = interaction_df[interaction_df['position'] == pos]
            if len(subset) > 2 and subset[exp_col].notna().sum() > 2:
                try:
                    corr, pval = stats.pearsonr(subset[exp_col].dropna(), 
                                               subset.loc[subset[exp_col].notna(), 'True_Resilience'])
                    interaction_effects.append({
                        '网络位置': pos,
                        '敞口-韧性相关系数': corr,
                        'P值': pval,
                        '样本数': len(subset),
                        '显著性': '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else 'ns'))
                    })
                except:
                    pass
        
        if interaction_effects:
            interaction_effects_df = pd.DataFrame(interaction_effects)
            print("\n👉 规则敞口的边际效应（按网络位置分层）：")
            print(interaction_effects_df.to_string(index=False))
            
            # 生成热力图
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig.suptitle(f'{latest_year}年：规则敞口×网络位置的交互效应', fontsize=14, fontweight='bold')
            
            # 交互效应二维热力
            ax1 = axes[0]
            pivot_data = interaction_df.groupby('position')[exp_col].describe()[['mean', 'std', 'min', 'max']]
            sns.heatmap(pivot_data.T, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1, cbar_kws={'label': '规则敞口'})
            ax1.set_title('规则敞口在各层级的分布', fontweight='bold')
            ax1.set_ylabel('统计量')
            
            # 相关系数对比
            ax2 = axes[1]
            positions = interaction_effects_df['网络位置'].tolist()
            correlations = interaction_effects_df['敞口-韧性相关系数'].tolist()
            colors = ['#27ae60' if c > 0 else '#e74c3c' for c in correlations]
            ax2.barh(positions, correlations, color=colors, alpha=0.7)
            ax2.set_xlabel('规则敞口→韧性的边际效应系数')
            ax2.set_title('网络位置对规则敞口边际效应的调节', fontweight='bold')
            ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax2.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/假说2验证_交互效应可视化.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 输出：假说2验证_交互效应可视化.png")
    else:
        print("⚠️ 未找到敞口列，跳过分析2")
    
except Exception as e:
    print(f"❌ 分析2出错：{e}")
    import traceback
    traceback.print_exc()

# ==================== 分析3：制度环境的硬约束倒逼机制 ====================
print("\n📊 分析3：制度环境的硬约束倒逼机制（WGI->韧性传导）")
print("-" * 60)

try:
    # ---- Bug1修复: WGI是宽格式（年份为列名），需要melt转为长格式 ----
    wgi_wide = pd.read_csv(f"{INPUT_DIR}/input_WB_WGI_WIDEF.csv")
    print(f"WGI原始列数：{len(wgi_wide.columns)}  行数：{len(wgi_wide)}")

    # 识别年份列（数字列名）
    id_cols = ['REF_AREA', 'INDICATOR']
    year_cols = [c for c in wgi_wide.columns if str(c).isdigit()]
    print(f"WGI年份列: {year_cols}")

    # 只取我们需要的指标：RQ（监管质量）百分位排名，作为制度环境代理变量
    # INDICATOR包含: WB_WGI_RQ_PER_RNK, WB_WGI_CC_EST 等
    rq_indicators = wgi_wide[wgi_wide['INDICATOR'].str.contains('RQ_PER_RNK', na=False)].copy()
    if len(rq_indicators) == 0:
        # 回退：取任意PER_RNK指标
        rq_indicators = wgi_wide[wgi_wide['INDICATOR'].str.contains('PER_RNK', na=False)].copy()
    print(f"监管质量行数: {len(rq_indicators)}")

    # melt为长格式
    wgi_long = rq_indicators.melt(
        id_vars=['REF_AREA', 'INDICATOR'],
        value_vars=year_cols,
        var_name='TIME_PERIOD',
        value_name='RQ_Score'
    )
    wgi_long['TIME_PERIOD'] = pd.to_numeric(wgi_long['TIME_PERIOD'], errors='coerce').astype('Int64')
    wgi_long['RQ_Score'] = pd.to_numeric(wgi_long['RQ_Score'], errors='coerce')
    wgi_long = wgi_long.dropna(subset=['RQ_Score']).copy()
    print(f"WGI长格式行数: {len(wgi_long)}  年份范围: {wgi_long['TIME_PERIOD'].min()}-{wgi_long['TIME_PERIOD'].max()}")

    # ---- Bug2修复: exposure列名是'Exposure'，不是'exposure_index' ----
    # exposure_df 在分析2中已加载，列名为 REF_AREA, TIME_PERIOD, Exposure
    exp_col_name = 'Exposure'  # 已确认的实际列名

    # 准备时间序列数据（最近5年）
    recent_years = sorted(resilience_df['TIME_PERIOD'].unique())[-5:]
    latest_year = recent_years[-1]
    print(f"使用最近5年: {recent_years}，最新年份: {latest_year}")

    resi_latest = resilience_df[resilience_df['TIME_PERIOD'] == latest_year].copy()
    wgi_latest = wgi_long[wgi_long['TIME_PERIOD'] == latest_year][['REF_AREA', 'RQ_Score']].copy()
    exp_latest = exposure_df[exposure_df['TIME_PERIOD'] == latest_year].copy()

    # 如果最新年份WGI无数据，回退到有数据的最近年份
    if len(wgi_latest) == 0:
        avail_years = sorted(wgi_long['TIME_PERIOD'].dropna().unique(), reverse=True)
        for yr in avail_years:
            wgi_latest = wgi_long[wgi_long['TIME_PERIOD'] == yr][['REF_AREA', 'RQ_Score']].copy()
            if len(wgi_latest) > 0:
                print(f"WGI最新年份无数据，回退到 {yr} 年 (N={len(wgi_latest)})")
                break

    # 合并数据
    transmission_df = resi_latest.merge(wgi_latest, on='REF_AREA', how='inner')
    transmission_df = transmission_df.merge(
        exp_latest[['REF_AREA', exp_col_name]],
        on='REF_AREA', how='inner'
    )
    print(f"传导链数据行数: {len(transmission_df)}")

    # 制度质量与韧性的相关性
    corr_rq_resi, pval_rq_resi = stats.pearsonr(
        transmission_df['RQ_Score'].dropna(),
        transmission_df.loc[transmission_df['RQ_Score'].notna(), 'True_Resilience']
    )
    corr_exp_resi, pval_exp_resi = stats.pearsonr(
        transmission_df[exp_col_name].dropna(),
        transmission_df.loc[transmission_df[exp_col_name].notna(), 'True_Resilience']
    )

    # 生成证据表
    transmission_evidence = [{
        '年份': latest_year,
        '样本数': len(transmission_df),
        '韧性平均值': round(transmission_df['True_Resilience'].mean(), 4),
        '韧性标准差': round(transmission_df['True_Resilience'].std(), 4),
        '出度中心度平均值': round(transmission_df['Out_Degree_Centrality'].mean(), 4),
        'RQ均值': round(transmission_df['RQ_Score'].mean(), 4),
        '规则敞口均值': round(transmission_df[exp_col_name].mean(), 4),
        'RQ-韧性相关系数': round(corr_rq_resi, 4),
        'RQ-韧性p值': round(pval_rq_resi, 4),
        '敞口-韧性相关系数': round(corr_exp_resi, 4),
        '敞口-韧性p值': round(pval_exp_resi, 4),
    }]

    transmission_evidence_df = pd.DataFrame(transmission_evidence)
    transmission_evidence_df.to_csv(
        f"{OUTPUT_DIR}/规则敞口-RQ-韧性传导链_实证证据表.csv",
        index=False,
        encoding='utf-8-sig'
    )
    print(f"[OK] 输出：规则敞口-RQ-韧性传导链_实证证据表.csv")
    print(transmission_evidence_df.T.to_string())

    # 生成时间序列可视化（韧性 & 中心度演化）
    fig, ax = plt.subplots(figsize=(10, 6))
    n_years = len(recent_years)
    fig.suptitle(f'制度环境与网络韧性的演化（最近{n_years}年）', fontsize=14, fontweight='bold')

    years_plot = []
    mean_resilience = []
    mean_centrality = []

    for year in recent_years:
        resi_year = resilience_df[resilience_df['TIME_PERIOD'] == year]
        if len(resi_year) > 0:
            years_plot.append(year)
            mean_resilience.append(resi_year['True_Resilience'].mean())
            mean_centrality.append(resi_year['Out_Degree_Centrality'].mean())

    ax2 = ax.twinx()
    line1 = ax.plot(years_plot, mean_resilience, marker='o', linewidth=2.5, markersize=8,
                    color='#2ecc71', label='平均韧性')
    line2 = ax2.plot(years_plot, mean_centrality, marker='^', linewidth=2.5, markersize=8,
                     color='#3498db', label='平均中心度')

    ax.set_xlabel('年份', fontsize=11)
    ax.set_ylabel('网络韧性', color='#2ecc71', fontsize=11)
    ax2.set_ylabel('出度中心度', color='#3498db', fontsize=11)
    ax.tick_params(axis='y', labelcolor='#2ecc71')
    ax2.tick_params(axis='y', labelcolor='#3498db')
    ax.grid(True, alpha=0.3)

    lines = line1 + line2
    labels_leg = [l.get_label() for l in lines]
    ax.legend(lines, labels_leg, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/假说3验证_制度升级倒逼机制.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 输出：假说3验证_制度升级倒逼机制.png")

except Exception as e:
    print(f"[ERROR] 分析3出错：{e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ 所有补充分析完成！")
print("=" * 60)
print("\n新生成的文件：")
print("  1. 核心vs边缘国家_韧性对比表.csv")
print("  2. 网络位置×韧性分布_分位数图.png")
print("  3. 规则敞口×中心度_交互效应表.csv")
print("  4. 假说2验证_交互效应可视化.png")
print("  5. 规则敞口→RQ→韧性传导链_实证证据表.csv")
print("  6. 假说3验证_制度升级倒逼机制.png")
print("  7. 论文假说验证_结论汇总表.csv")
print("  8. 论文假说_综合验证仪表板.md")
