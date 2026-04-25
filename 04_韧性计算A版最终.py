import pandas as pd
import networkx as nx
import os


def _load_cleaned_loader():
    """按需加载 get_cleaned_batis_df，避免路径不是包时报错。"""
    import importlib.util

    module_path = os.path.join(os.path.dirname(__file__), '01_数据清洗A版.py')
    if not os.path.exists(module_path):
        return None

    spec = importlib.util.spec_from_file_location('data_cleaning_module', module_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, 'get_cleaned_batis_df', None)

# ========== 加载清洁数据 ==========
# 方式1: 从导出的CSV文件加载（推荐，性能更优）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, '输入文件')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '输出文件')

CSV_CLEANED_FILE = os.path.join(INPUT_DIR, 'input_cleaned_batis_df.csv')

if os.path.exists(CSV_CLEANED_FILE):
    print(f"✓ 从 CSV 加载清洁数据: {CSV_CLEANED_FILE}")
    cleaned_batis_df = pd.read_csv(CSV_CLEANED_FILE)
else:
    # 方式2: 若CSV不存在，从函数动态生成（用于first-run或数据更新）
    print(f"⚠ CSV 文件不存在: {CSV_CLEANED_FILE}")
    print("  正在从原始数据动态生成清洁数据...")
    loader = _load_cleaned_loader()
    if loader is None:
        print("❌ 无法加载 get_cleaned_batis_df，且CSV缓存不存在")
        cleaned_batis_df = None
    else:
        cleaned_batis_df = loader()
    if cleaned_batis_df is not None:
        cleaned_batis_df.to_csv(CSV_CLEANED_FILE, index=False, encoding='utf-8')
        print(f"✓ 已缓存至: {CSV_CLEANED_FILE}")

if cleaned_batis_df is None or cleaned_batis_df.empty:
    print("❌ 无法加载清洁数据，程序终止")
    exit(1)

print(f"✓ 清洁数据加载成功: {cleaned_batis_df.shape[0]} 行, {cleaned_batis_df.shape[1]} 列\n")

def calculate_network_resilience(df, year_col='TIME_PERIOD', source_col='REF_AREA', target_col='COUNTERPART_AREA', weight_col='OBS_VALUE', threshold_percentile=0.99):
    """
    计算历年网络韧性 (LCC 缩减比例)。
    threshold_percentile: 删边阈值。设为0.05意味着删掉每年全网贸易额最低的 5% 的边 (去除引力模型的噪音)。
    """
    years = sorted(df[year_col].dropna().unique())
    resilience_results = []

    for y in years:
        # 1. 提取当年的横截面数据
        df_year = df[df[year_col] == y].copy()
        
        # 2. 阈值化删边 (扫除微弱的贸易尘埃)
        # 找到当年贸易额的 5% 分位数作为切刀
        threshold_value = df_year[weight_col].quantile(threshold_percentile) 
        # 只保留大于阈值的真实贸易骨干
        df_edges = df_year[df_year[weight_col] > threshold_value]
        
        # 3. 构建有向网络图
        G = nx.from_pandas_edgelist(df_edges, source=source_col, target=target_col, edge_attr=weight_col, create_using=nx.DiGraph())
        
        # 4. 计算初始的最大连通分量 (LCC_initial)
        # 有向图通常使用弱连通分量 (Weakly Connected Components)
        initial_components = list(nx.weakly_connected_components(G))
        if not initial_components:
            continue
        LCC_initial_nodes = max(initial_components, key=len)
        LCC_initial_size = len(LCC_initial_nodes)
        
        # 5. 模拟攻击：蓄意移除核心节点 (此处以移除当年出度中心度最高的前 5% 节点为例)
        # 计算出度中心度
        out_degrees = dict(G.out_degree())
        # 找出前 5% 核心节点
        num_to_remove = max(1, int(len(G.nodes()) * 0.05))
        nodes_to_remove = sorted(out_degrees, key=out_degrees.get, reverse=True)[:num_to_remove]
        
        # 从图中移除这些核心枢纽
        G.remove_nodes_from(nodes_to_remove)
        
        # 6. 计算攻击后的最大连通分量 (LCC_after)
        after_components = list(nx.weakly_connected_components(G))
        LCC_after_size = len(max(after_components, key=len)) if after_components else 0
        
        # 7. 计算网络韧性 R (LCC 的保留率)
        # R 值越接近 1，说明韧性越强；越接近 0，说明网络崩溃越彻底。
        resilience_score = LCC_after_size / LCC_initial_size
        
        resilience_results.append({
            'Year': y,
            'LCC_Initial': LCC_initial_size,
            'LCC_After': LCC_after_size,
            'Resilience': resilience_score
        })
        
        print(f"{y}年: 初始LCC={LCC_initial_size}, 攻击后LCC={LCC_after_size}, 网络韧性={resilience_score:.4f}")

    return pd.DataFrame(resilience_results)

# 计算网络韧性
print("\n" + "="*70)
print("计算网络韧性")
print("="*70 + "\n")
resilience_df = calculate_network_resilience(cleaned_batis_df)

if resilience_df is not None and not resilience_df.empty:
    print("\n" + "="*70)
    print("网络韧性结果汇总")
    print("="*70)
    print(resilience_df)
    
    # 导出网络韧性结果
    output_dir = OUTPUT_DIR
    resilience_output_file = os.path.join(output_dir, "output_network_resilience_results.csv")
    resilience_df.to_csv(resilience_output_file, index=False, encoding='utf-8')
    print(f"\n✓ 网络韧性结果已保存至: {resilience_output_file}")
else:
    print("\n❌ 网络韧性计算失败或无结果")