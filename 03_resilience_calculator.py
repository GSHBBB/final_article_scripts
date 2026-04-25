"""
Network Resilience Calculator - 重构版本
=========================================
对网络中心度面板数据进行HP滤波，计算网络韧性的被解释变量(Inexpress)

输入文件: node_centrality_panel.csv
输出文件: final_resilience_panel.csv

核心处理流程：
  1. 加载node_centrality_panel.csv
  2. 按国家分组，进行HP滤波(lambda=6.25，年度数据标准参数)
  3. 计算波动率：volatility = |cycle|
  4. 关键保护机制：safe_trend = np.clip(trend, 0.01, None)
     防止极小值或负值导致比值爆炸
  5. 计算Express = volatility / safe_trend
  6. 最终被解释变量：Inexpress = log(Express + 1e-6)
  7. 过滤地区数据年份 < 4年的条目
  8. 输出最终面板数据并进行极值验证
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
import os
import sys
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
# 抑制不必要的警告
warnings.filterwarnings('ignore')


def setup_console_encoding():
    """在 Windows 终端下尽量启用 UTF-8，避免中文输出乱码。"""
    if os.name != 'nt':
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            continue

# 中文字体设置函数（健壮版本）
def setup_chinese_font():
    """设置可用的中文字体，自动检测系统环境"""
    # 尝试的中文字体列表（优先级从高到低）
    chinese_fonts = [
        'SimHei',           # Windows 内置
        'Microsoft YaHei',  # Windows 内置
        'STSong',          # Mac 内置（宋体）
        'Heiti SC',        # Mac 内置（黑体）
        'SimSun',          # Windows 备选（仿宋）
        'WenQuanYi Zen Hei',  # Linux 开源
    ]
    
    # 获取系统中已安装的字体名称
    try:
        available_font_names = {font.name.lower() for font in fm.fontManager.ttflist}
    except Exception:
        # 如果上述方法失败，使用备选方法
        try:
            available_font_names = set()
        except Exception:
            available_font_names = set()
    
    # 从列表中选择第一个可用的字体
    selected_font = None
    for font in chinese_fonts:
        if font.lower() in available_font_names:
            selected_font = font
            break
    
    # 如果都不可用，使用通用备选
    if selected_font is None:
        selected_font = 'DejaVu Sans'
    
    plt.rcParams['font.sans-serif'] = [selected_font, 'Arial', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    
    return selected_font

# 执行字体设置
setup_console_encoding()

try:
    CHINESE_FONT = setup_chinese_font()
except Exception as e:
    # 如果字体设置失败，使用最小化配置
    CHINESE_FONT = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    pass


# ====== 配置 ======
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '输出文件')

INPUT_FILE = os.path.join(OUTPUT_DIR, 'output_node_centrality_panel.csv')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'output_final_resilience_panel.csv')
LAMBDA = 6.25  # HP滤波平滑参数（年度数据的标准值）
MIN_YEARS = 5   # 最少年份要求（用于有效滤波）
TREND_MIN = 0.001  # 分母保护下界

# 常见 ISO3 国家代码到中文名映射（用于排名输出与图例显示）
COUNTRY_NAME_ZH = {
    'CHN': '中国', 'USA': '美国', 'JPN': '日本', 'KOR': '韩国', 'DEU': '德国',
    'FRA': '法国', 'GBR': '英国', 'ITA': '意大利', 'ESP': '西班牙', 'NLD': '荷兰',
    'BEL': '比利时', 'CHE': '瑞士', 'AUT': '奥地利', 'SWE': '瑞典', 'NOR': '挪威',
    'DNK': '丹麦', 'FIN': '芬兰', 'IRL': '爱尔兰', 'POL': '波兰', 'CZE': '捷克',
    'SVK': '斯洛伐克', 'HUN': '匈牙利', 'PRT': '葡萄牙', 'GRC': '希腊', 'ROU': '罗马尼亚',
    'BGR': '保加利亚', 'HRV': '克罗地亚', 'SVN': '斯洛文尼亚', 'EST': '爱沙尼亚', 'LVA': '拉脱维亚',
    'LTU': '立陶宛', 'LUX': '卢森堡', 'CYP': '塞浦路斯', 'MLT': '马耳他', 'ISL': '冰岛',
    'RUS': '俄罗斯', 'UKR': '乌克兰', 'BLR': '白俄罗斯', 'TUR': '土耳其', 'SAU': '沙特阿拉伯',
    'ARE': '阿联酋', 'QAT': '卡塔尔', 'KWT': '科威特', 'ISR': '以色列', 'EGY': '埃及',
    'IND': '印度', 'IDN': '印度尼西亚', 'MYS': '马来西亚', 'THA': '泰国', 'VNM': '越南',
    'PHL': '菲律宾', 'SGP': '新加坡', 'PAK': '巴基斯坦', 'BGD': '孟加拉国', 'LKA': '斯里兰卡',
    'AUS': '澳大利亚', 'NZL': '新西兰', 'CAN': '加拿大', 'MEX': '墨西哥', 'BRA': '巴西',
    'ARG': '阿根廷', 'CHL': '智利', 'COL': '哥伦比亚', 'PER': '秘鲁', 'URY': '乌拉圭',
    'ZAF': '南非', 'NGA': '尼日利亚', 'KEN': '肯尼亚', 'ETH': '埃塞俄比亚', 'MAR': '摩洛哥'
}


def country_display_name(iso3_code):
    """将 ISO3 代码格式化为中文显示名（不存在映射时回退为原代码）。"""
    code = str(iso3_code)
    zh_name = COUNTRY_NAME_ZH.get(code)
    if zh_name:
        return f"{zh_name}({code})"
    return code


def load_and_validate_data(filepath):
    """
    加载并验证输入数据。
    
    参数:
        filepath (str): 输入CSV文件路径
    
    返回:
        pd.DataFrame: 经过验证的数据框
    """
    print(f"\n========== 【步骤1】加载输入数据 ==========")
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
    print(f"  列名: {list(df.columns)}")
    
    # 验证必要列存在
    required_cols = ['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 错误：缺少必要列 {missing_cols}")
        sys.exit(1)
    
    # 删除缺失值
    df = df.dropna(subset=['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality'])
    print(f"  删除缺失值后: {df.shape[0]} 行")
    
    # 排序
    df = df.sort_values(['REF_AREA', 'TIME_PERIOD']).reset_index(drop=True)
    print(f"  ├─ 时间范围: {df['TIME_PERIOD'].min()} - {df['TIME_PERIOD'].max()}")
    print(f"  └─ 国家总数: {df['REF_AREA'].nunique()}")
    
    return df


def apply_hp_filter_to_country(sub_df, country, lamb=6.25):
    """
    对单个国家的时间序列应用HP滤波。
    
    参数:
        sub_df (pd.DataFrame): 该国家的子数据框
        country (str): 国家代码
        lamb (float): HP滤波参数
    
    返回:
        pd.DataFrame: 包含滤波结果的数据框，若失败返回None
    """
    n_years = len(sub_df)
    
    if n_years < MIN_YEARS:
        return None
    
    try:
        # 获取时间序列
        series = sub_df['Out_Degree_Centrality'].values
        
        # 执行HP滤波（注意：正式API返回顺序是(cycle, trend)）
        cycle, trend = hpfilter(series, lamb=lamb)
        
        # 计算波动率（绝对波动项）
        volatility = np.abs(cycle)
        
        # 【关键】分母保护：使用clip防止极小值导致的数值爆炸
        safe_trend = np.clip(trend, a_min=TREND_MIN, a_max=None)
        
        # 计算比值（网络表达能力Express）
        express = volatility / safe_trend
        
        # 取对数得到不导米韧性指标Inexpress
        inexpress = np.log(express + 1e-6)
        
        # 组装结果数据框
        result = sub_df[['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']].copy()
        result['Trend'] = trend
        result['Volatility'] = volatility
        result['Safe_Trend'] = safe_trend
        result['Express'] = express
        result['Inexpress'] = inexpress
        
        return result
        
    except Exception as e:
        # 单个国家失败不影响整体，仅记录警告
        print(f"  ⚠ 国家 {country} 滤波失败: {e}")
        return None


def main():
    """主执行流程。"""
    
    print("\n" + "="*75)
    print("" * 20 + "【网络韧性HP滤波计算】")
    print("生成双重差分回归的被解释变量 Inexpress")
    print("="*75)
    
    # 【步骤1】加载并验证数据
    df = load_and_validate_data(INPUT_FILE)
    
    # 【步骤2】按国家分组，执行HP滤波
    print(f"\n========== 【步骤2】按国家进行HP滤波处理 ==========")
    print(f"HP参数 lambda={LAMBDA}, 最低年份数={MIN_YEARS}, 分母保护下界={TREND_MIN}")
    
    all_results = []
    grouped = df.groupby('REF_AREA')
    total_countries = len(grouped)
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for i, (country, sub_df) in enumerate(grouped, 1):
        n_years = len(sub_df)
        
        # 检查年份数是否充足
        if n_years < MIN_YEARS:
            print(f"  [{i}/{total_countries}] {country}: ⊘ 跳过 ({n_years} 年 < {MIN_YEARS} 年)")
            skip_count += 1
            continue
        
        # 执行HP滤波
        result = apply_hp_filter_to_country(sub_df, country, lamb=LAMBDA)
        
        if result is not None:
            all_results.append(result)
            success_count += 1
            print(f"  [{i}/{total_countries}] {country}: ✓ 成功 ({n_years} 年)")
        else:
            fail_count += 1
    
    print(f"\n滤波统计：成功={success_count}, 跳过={skip_count}, 失败={fail_count}")
    
    if not all_results:
        print("\n❌ 错误：未成功处理任何国家，无法继续")
        sys.exit(1)
    
    # 【步骤3】组装最终面板数据
    print(f"\n========== 【步骤3】组装最终面板数据 ==========")
    
    final_df = pd.concat(all_results, ignore_index=True)
    
    # 清理任何仍可能存在的缺失值或无穷值
    final_df = final_df.dropna(subset=['Inexpress'])
    final_df = final_df[~np.isinf(final_df['Inexpress'])]
    
    # 计算真实网络韧性（取负Inexpress）
    final_df['True_Resilience'] = -final_df['Inexpress']
    
    print(f"✓ 最终数据维度: {final_df.shape[0]} 行, {final_df.shape[1]} 列")
    print(f"  ├─ 覆盖国家数: {final_df['REF_AREA'].nunique()}")
    print(f"  └─ 覆盖年份数: {final_df['TIME_PERIOD'].nunique()}")
    
    # 【步骤4】数据导出
    print(f"\n========== 【步骤4】导出最终数据 ==========")
    
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  创建输出目录: {output_dir}")
    
    # 选择要导出的列（仅保留必要的列）
    export_cols = ['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality', 
                   'Trend', 'Volatility', 'Express', 'Inexpress', 'True_Resilience']
    final_df_export = final_df[export_cols].copy()
    
    try:
        final_df_export.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"✓ 已保存至: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ 错误：保存文件失败 - {e}")
        sys.exit(1)
    
    # 【步骤5】极值验证与统计输出
    print(f"\n========== 【步骤5】极值验证与统计摘要 ==========")
    
    print(f"\n被解释变量 Inexpress 的统计描述:")
    print(final_df_export['Inexpress'].describe())
    
    print(f"\n关键变量统计:")
    stats_cols = ['Out_Degree_Centrality', 'Trend', 'Volatility', 'Express', 'Inexpress']
    print(final_df_export[stats_cols].describe().round(6))
    
    print(f"\n极值检查 (Inexpress):")
    print(f"  ├─ 最小值: {final_df_export['Inexpress'].min():.6f}")
    print(f"  ├─ 最大值: {final_df_export['Inexpress'].max():.6f}")
    print(f"  ├─ 平均值: {final_df_export['Inexpress'].mean():.6f}")
    print(f"  ├─ 标准差: {final_df_export['Inexpress'].std():.6f}")
    print(f"  └─ 异常值数: {(final_df_export['Inexpress'].abs() > 5).sum()} (|值| > 5)")
    
    # 检查是否存在极端离群值（可能表示分母保护机制未生效）
    extreme_outliers = (final_df_export['Express'] > 1000).sum()
    if extreme_outliers > 0:
        print(f"\n⚠ 警告：存在 {extreme_outliers} 个极端离群值 (Express > 1000)")
    else:
        print(f"\n✓ 表达能力指标 Express 已被有效控制，无极端离群值")
    
    print(f"\n数据预览 (前 15 行):")
    print(final_df_export.head(15).to_string(index=False))
    
    # 【步骤6】可视化：前十名国家的韧性变化趋势
    print(f"\n========== 【步骤6】可视化 - 前十名国家的韧性变化 ==========")
    plot_top_n_resilience_trends(final_df_export, n=10)
    
    print("\n" + "="*75)
    print("✓ HP滤波与韧性计算完成！")
    print(f"✓ 最终数据已保存至: {OUTPUT_FILE}")
    print("="*75 + "\n")


def plot_top_n_resilience_trends(final_df, n=10):
    """
    绘制按最后一年真实韧性从大到小排名的前N个国家的韧性变化折线图。
    
    使用seaborn风格，为10个国家分配易于区分的颜色和不同的标记。
    
    参数:
        final_df (pd.DataFrame): 包含 TIME_PERIOD, REF_AREA, True_Resilience 的最终面板数据
        n (int): 要绘制的国家数量，默认为10
    """
    
    # 设置seaborn主题
    sns.set_theme(
        style="whitegrid",
        rc={
            'font.sans-serif': [CHINESE_FONT, 'Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif'],
            'axes.unicode_minus': False,
            'font.family': 'sans-serif'
        }
    )
    
    # 获取最后一年的数据，按韧性从大到小排序
    last_year = final_df['TIME_PERIOD'].max()
    last_year_data = final_df[final_df['TIME_PERIOD'] == last_year].sort_values(
        'True_Resilience', ascending=False
    )
    
    # 选取前N个国家，保持排名顺序（从大到小）
    top_n_countries = last_year_data['REF_AREA'].head(n).tolist()
    
    print(f"\n✓ 按 {last_year} 年韧性指标排名（从大到小），前 {n} 个国家：")
    print("="*65)
    ranking_data = []
    for idx, (i, row) in enumerate(last_year_data.head(n).iterrows(), 1):
        country_label = country_display_name(row['REF_AREA'])
        print(f"  第{idx:2d}名：{country_label}   真实韧性值 = {row['True_Resilience']:10.6f}")
        ranking_data.append((idx, country_label, row['True_Resilience']))
    print("="*65)
    
    # 筛选这些国家的全时间序列数据
    plot_data = final_df[final_df['REF_AREA'].isin(top_n_countries)].copy()
    plot_data = plot_data.sort_values(['REF_AREA', 'TIME_PERIOD'])
    
    # 定义颜色和标记
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers_list = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # 绘制折线图 - 高分辨率（4096x2160）
    fig, ax = plt.subplots(figsize=(18.96, 10), dpi=216)
    
    for rank_idx, country in enumerate(top_n_countries):
        country_data = plot_data[plot_data['REF_AREA'] == country]
        
        # 图例中显示排名和国家代码
        label = f"#{rank_idx+1} {country_display_name(country)}"
        
        ax.plot(
            country_data['TIME_PERIOD'],
            country_data['True_Resilience'],
            marker=markers_list[rank_idx % len(markers_list)],
            color=colors_list[rank_idx % len(colors_list)],
            label=label,
            linewidth=2.5,
            markersize=7,
            alpha=0.8
        )
    
    # 设置X轴刻度为整数年份
    years = sorted(plot_data['TIME_PERIOD'].unique())
    ax.set_xticks(years)
    ax.set_xticklabels([int(y) for y in years], rotation=45, ha='right')
    
    # 设置标签和标题
    ax.set_xlabel('年份', fontsize=12, fontweight='bold')
    ax.set_ylabel('网络韧性 (-Inexpress)', fontsize=12, fontweight='bold')
    ax.set_title('数字服务出口网络中心度排名前十国家的网络韧性趋势 (2005-2024)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 将图例放置在图表外侧（右侧）
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10, 
              frameon=True, fancybox=True, shadow=True, ncol=1)
    
    # 网格设置
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 保存图示
    output_dir = os.path.dirname(OUTPUT_FILE)
    plot_file = os.path.join(output_dir, 'top_countries_resilience_trends.png')
    plt.savefig(plot_file, dpi=216, bbox_inches='tight', facecolor='white')
    print(f"\n✓ 折线图已保存至: {plot_file}")
    print(f"  • 输出分辨率：4096 × 2160 像素")
    print(f"  • 中文字体：已启用 (SimHei)")
    
    plt.show()


if __name__ == "__main__":
    main()
