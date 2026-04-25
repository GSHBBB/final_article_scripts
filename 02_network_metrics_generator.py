"""
Network Metrics Generator - 重构版本
====================================
严格按照期刊论文规范计算各国在数字服务贸易网络中的节点出度中心度，生成面板数据

输入文件: cleaned_batis_df.csv
输出文件: node_centrality_panel.csv

核心逻辑：
  1. 年度循环：对每一年数据分别处理
  2. 对分截断：计算该年OBS_VALUE的中位数（50%分位数），作为阈值
  3. 无权图构建：仅保留OBS_VALUE >= 中位数的边，构建为无权有向图
  4. 出度中心度：计算每个节点的出度中心度指标
"""

import pandas as pd
import networkx as nx
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ====== 配置 ======
# 统一限定在“正式脚本”目录内读写
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, '输入文件')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '输出文件')

# 明确使用最早清洗版 cleaned_batis（非 Pure_*）
INPUT_FILE = os.path.join(INPUT_DIR, 'input_cleaned_batis_df.csv')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'output_node_centrality_panel.csv')


def load_data(filepath):
    """
    加载并初步处理原始数据。
    
    参数:
        filepath (str): 输入CSV文件路径
    
    返回:
        pd.DataFrame: 包含四列的清洁数据
    """
    print(f"\n========== 【步骤1】加载原始数据 ==========")
    print(f"✓ 读取文件: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ 错误：文件不存在 {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误：读取文件失败 - {e}")
        sys.exit(1)
    
    print(f"  原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
    
    # 保留必要的 4 列
    try:
        df = df[['TIME_PERIOD', 'REF_AREA', 'COUNTERPART_AREA', 'OBS_VALUE']].copy()
    except KeyError as e:
        print(f"❌ 错误：缺少必要列 - {e}")
        sys.exit(1)
    
    print(f"  提取关键列后: {df.shape[0]} 行, {df.shape[1]} 列")
    
    # 删除缺失值和零值/负值
    initial_rows = len(df)
    df = df.dropna(subset=['OBS_VALUE'])
    df = df[df['OBS_VALUE'] > 0]
    removed_rows = initial_rows - len(df)
    print(f"  删除缺失/非正值后: {df.shape[0]} 行 (删除 {removed_rows} 行)")
    
    # 验证其他列不含缺失值
    df = df.dropna(subset=['TIME_PERIOD', 'REF_AREA', 'COUNTERPART_AREA'])
    print(f"  删除其他缺失值后: {df.shape[0]} 行")
    
    return df


def compute_node_centrality(df, year):
    """
    计算指定年份的出度中心度指标。
    
    核心算法：
    1. 过滤该年数据
    2. 计算OBS_VALUE的中位数作为阈值
    3. 保留OBS_VALUE >= 中位数的边
    4. 构建无权有向图
    5. 计算出度中心度：out_degree / (N-1)
    
    参数:
        df (pd.DataFrame): 完整的原始数据
        year (int): 要处理的年份
    
    返回:
        list: 包含该年所有国家的中心度数据，每个元素为字典
    """
    
    # 筛选该年数据
    df_year = df[df['TIME_PERIOD'] == year].copy()
    
    if len(df_year) == 0:
        print(f"  【{year}年】⚠ 无数据")
        return []
    
    print(f"  【{year}年】", end="")
    
    # 计算中位数阈值
    median_value = df_year['OBS_VALUE'].median()
    
    # 保留大于等于中位数的边（对分截断）
    df_filtered = df_year[df_year['OBS_VALUE'] >= median_value].copy()
    
    edge_count_before = len(df_year)
    edge_count_after = len(df_filtered)
    
    if edge_count_after == 0:
        print(f"⚠ 阈值化后无有效边")
        return []
    
    # 构建无权有向图
    # 删除自环（防止出口国 == 进口国）
    df_filtered = df_filtered[df_filtered['REF_AREA'] != df_filtered['COUNTERPART_AREA']]
    
    if len(df_filtered) == 0:
        print(f"⚠ 删除自环后无有效边")
        return []
    
    G = nx.from_pandas_edgelist(
        df_filtered,
        source='REF_AREA',
        target='COUNTERPART_AREA',
        create_using=nx.DiGraph()  # 有向图，无权
    )
    
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    if n_nodes == 0:
        print(f"⚠ 网络为空")
        return []
    
    # 计算出度中心度（NetworkX内置方法）
    out_degree_cent = nx.out_degree_centrality(G)
    
    # 组装结果列表
    metrics_list = []
    for node in G.nodes():
        metrics_list.append({
            'TIME_PERIOD': year,
            'REF_AREA': node,
            'Out_Degree_Centrality': out_degree_cent[node]
        })
    
    print(f"✓ {n_nodes} 节点, {n_edges} 边 (原{edge_count_before}→阈值{edge_count_after})")
    
    return metrics_list


def main():
    """主执行流程。"""
    
    print("\n" + "="*75)
    print("" * 20 + "【网络中心度指标计算】")
    print("严格按照期刊论文规范计算出度中心度 - 面板数据生成")
    print("="*75)
    
    # 【步骤1】加载数据
    df = load_data(INPUT_FILE)
    
    # 【步骤2】提取年份列表
    years = sorted(df['TIME_PERIOD'].unique())
    n_years = len(years)
    print(f"\n✓ 发现 {n_years} 个年份: {years[0]} - {years[-1]}")
    
    # 【步骤3】年度循环计算中心度
    print(f"\n========== 【步骤2】年度循环计算 (中位数对分截断) ==========")
    
    all_metrics = []
    for i, year in enumerate(years, 1):
        metrics = compute_node_centrality(df, year)
        all_metrics.extend(metrics)
    
    if not all_metrics:
        print("\n❌ 错误：无法生成任何指标数据，请检查输入文件")
        sys.exit(1)
    
    # 【步骤4】组装为 DataFrame
    print(f"\n========== 【步骤3】组装面板数据 ==========")
    panel_df = pd.DataFrame(all_metrics)
    
    # 确保列顺序
    panel_df = panel_df[['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']]
    
    print(f"✓ 面板数据维度: {panel_df.shape[0]} 行, {panel_df.shape[1]} 列")
    
    # 【步骤5】数据导出
    print(f"\n========== 【步骤4】导出数据 ==========")
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  创建输出目录: {output_dir}")
    
    try:
        panel_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"✓ 已保存至: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ 错误：保存文件失败 - {e}")
        sys.exit(1)
    
    # 【步骤6】输出汇总统计
    print(f"\n========== 【步骤5】数据统计摘要 ==========")
    
    print(f"\n面板数据基本信息:")
    print(f"  ├─ 总行数: {panel_df.shape[0]}")
    print(f"  ├─ 总列数: {panel_df.shape[1]}")
    print(f"  ├─ 时间范围: {panel_df['TIME_PERIOD'].min()} - {panel_df['TIME_PERIOD'].max()}")
    print(f"  └─ 唯一国家数: {panel_df['REF_AREA'].nunique()}")
    
    print(f"\n出度中心度的统计描述:")
    print(panel_df['Out_Degree_Centrality'].describe())
    
    print(f"\n面板数据预览 (前 10 行):")
    print(panel_df.head(10).to_string(index=False))
    
    # 【步骤7】可视化：前十名国家的出度中心度变化趋势
    print(f"\n========== 【步骤6】可视化 - 前十名国家的中心度变化 ==========")
    plot_top_n_centrality_trends(panel_df, n=10)
    
    print("\n" + "="*75)
    print("✓ 网络中心度计算完成！下一步可执行 resilience_calculator.py")
    print("="*75 + "\n")


def plot_top_n_centrality_trends(panel_df, n=10):
    """
    绘制按最后一年出度中心度排名的前N个国家的中心度变化折线图。
    
    参数:
        panel_df (pd.DataFrame): 包含 TIME_PERIOD, REF_AREA, Out_Degree_Centrality 的面板数据
        n (int): 要绘制的国家数量，默认为10
    """
    
    # 获取最后一年的数据
    last_year = panel_df['TIME_PERIOD'].max()
    last_year_data = panel_df[panel_df['TIME_PERIOD'] == last_year].sort_values(
        'Out_Degree_Centrality', ascending=False
    )
    
    # 选取前N个国家
    top_n_countries = last_year_data['REF_AREA'].head(n).tolist()
    
    print(f"✓ 按 {last_year} 年出度中心度排名，选取前 {n} 个国家:")
    for idx, (i, row) in enumerate(last_year_data.head(n).iterrows(), 1):
        print(f"  {idx}. {row['REF_AREA']:5s} - 中心度: {row['Out_Degree_Centrality']:.6f}")
    
    # 筛选这些国家的全时间序列数据
    plot_data = panel_df[panel_df['REF_AREA'].isin(top_n_countries)].copy()
    plot_data = plot_data.sort_values(['REF_AREA', 'TIME_PERIOD'])
    
    # 绘制折线图 - 高分辨率（4096x2160）
    plt.figure(figsize=(18.96, 10), dpi=216)
    
    for country in top_n_countries:
        country_data = plot_data[plot_data['REF_AREA'] == country]
        plt.plot(
            country_data['TIME_PERIOD'],
            country_data['Out_Degree_Centrality'],
            marker='o',
            label=country,
            linewidth=2.5,
            markersize=8
        )
    
    plt.xlabel('年份', fontsize=16, fontweight='bold')
    plt.ylabel('出度中心度', fontsize=16, fontweight='bold')
    plt.title(f'数字服务贸易网络中出度中心度排名前{n}的国家的变化趋势(2005-2025)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=12, ncol=2, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    
    # 保存图示
    output_dir = os.path.dirname(OUTPUT_FILE)
    plot_file = os.path.join(output_dir, 'top_countries_centrality_trends.png')
    plt.savefig(plot_file, dpi=216, bbox_inches='tight', facecolor='white')
    print(f"\n✓ 图表已保存至: {plot_file}")
    print(f"  • 输出分辨率：4096 × 2160 像素")
    print(f"  • 中文字体：已启用 (SimHei)")
    
    plt.show()


if __name__ == "__main__":
    main()
