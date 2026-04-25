"""
数字贸易数据清洗与分析脚本 - 修复版本 v2

修复说明：
1. 行业分类重复累加：使用完整的DDS六个细分行业(SH,SF,SK,SG,SI,SJ)
2. METHODOLOGY_TYPE去重：优先选择DE(Derivation)数据，确保同一双边贸易流只保留一条记录
3. 国家聚合体过滤：使用pycountry严格过滤，确保只包含主权国家
4. 年份聚合：确保只使用最新年份的数据

数据源限制说明：
- 输入文件仅包含DDS六个服务，不包含全部服务贸易数据
- 因此无法计算DDS占总服务出口的比重（占比功能已移除）

Digital-Deliverable Services (DDS) 分类：
- SF: Insurance services (保险服务)
- SG: Financial services (金融服务)
- SH: Charges for the use of intellectual property (知识产权使用费)
- SI: Telecommunications, computer and information services (电信、计算机和信息服务)
- SJ: Other business services (其他商业服务)
- SK: Personal, cultural and recreational services (个人、文化和娱乐服务)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pycountry

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / '输入文件'
OUTPUT_DIR = BASE_DIR / '输出文件'

OECD_FILE = INPUT_DIR / 'input_OECD_BaTIS_Data3.csv'
EXPORT_FLOW_CODE = 'X'

# 数字交付服务(DDS)的六个细分行业代码
DDS_SERVICE_CODES = ['SF', 'SG', 'SH', 'SI', 'SJ', 'SK']

# METHODOLOGY_TYPE 优先级（从高到低）
# DE: Derivation (直接推导，最可靠)
# IE: Interpolation/extrapolation (插值)
# ME: Model-based estimation (模型估算)
# AG: Aggregation (聚合)
# AD: Adjustment (调整)
METHODOLOGY_PRIORITY_MAP = {
    'DE': 5,  # 最高优先级
    'IE': 4,
    'ME': 3,
    'AG': 2,
    'AD': 1,
}

WB_MACRO_REGION_CODES = {
    'ARB', 'CSS', 'CEB', 'EAR', 'EAS', 'EAP', 'EMU', 'ECA', 'ECS', 'EUU',
    'FCS', 'HIC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN',
    'LDC', 'LIC', 'LMC', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PST',
    'PRE', 'SAS', 'SSA', 'SSF', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS',
    'UMC', 'WLD', 'WXD'
}

# COUNTERPART_AREA 中出现的聚合体代码（需要排除，否则 W=World 会导致重复计算）
COUNTERPART_AGGREGATE_CODES = {
    'W', 'WXD', 'WXEU27_2020', 'WXOECD', 'EU27_2020', 'OECD',
}

# 常见国家英文名到中文名映射（用于图表图例与坐标轴显示）
COUNTRY_NAME_ZH_MAP = {
    'United States': '美国',
    'China': '中国',
    'Germany': '德国',
    'Japan': '日本',
    'United Kingdom': '英国',
    'France': '法国',
    'India': '印度',
    'Ireland': '爱尔兰',
    'Netherlands': '荷兰',
    'Singapore': '新加坡',
    'South Korea': '韩国',
    'Korea': '韩国',
    'Switzerland': '瑞士',
    'Belgium': '比利时',
    'Italy': '意大利',
    'Spain': '西班牙',
    'Canada': '加拿大',
    'Australia': '澳大利亚',
    'Sweden': '瑞典',
    'Luxembourg': '卢森堡',
    'Hong Kong, China': '中国香港',
    'Hong Kong': '中国香港',
    'Taipei, Chinese': '中国台北',
    'Taiwan': '中国台湾',
    'Russian Federation': '俄罗斯',
    'Russia': '俄罗斯',
    'Brazil': '巴西',
    'Mexico': '墨西哥',
    'Türkiye': '土耳其',
    'Turkey': '土耳其',
    'Israel': '以色列',
    'Norway': '挪威',
    'Denmark': '丹麦',
    'Finland': '芬兰',
    'Austria': '奥地利',
    'Poland': '波兰',
    'Czechia': '捷克',
    'Portugal': '葡萄牙',
    'New Zealand': '新西兰',
    'Saudi Arabia': '沙特阿拉伯',
    'United Arab Emirates': '阿联酋',
    'South Africa': '南非',
}


def to_chinese_country_name(country_name):
    """将国家英文名转换为中文名，未命中时返回原值。"""
    return COUNTRY_NAME_ZH_MAP.get(country_name, country_name)


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_oecd_data():
    """
    加载OECD BaTIS数据，并进行初步清洗
    """
    usecols = [
        'REF_AREA', 'Reference area', 'COUNTERPART_AREA', 'Counterpart area',
        'TRADE_FLOW', 'SERVICE', 'TIME_PERIOD',
        'OBS_VALUE', 'METHODOLOGY_TYPE', 'Methodology type'
    ]
    # 处理混合类型列
    df = pd.read_csv(OECD_FILE, usecols=usecols, low_memory=False)
    df = df[df['TRADE_FLOW'] == EXPORT_FLOW_CODE].copy()
    df['REF_AREA'] = df['REF_AREA'].astype(str).str.strip().str.upper()
    df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce')
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')

    # 过滤无效数据
    df = df[df['TIME_PERIOD'].notna() & df['OBS_VALUE'].notna() & (df['OBS_VALUE'] > 0)].copy()
    df['TIME_PERIOD'] = df['TIME_PERIOD'].astype(int)

    # 只保留DDS相关的六个细分行业（修复问题1：行业分类重复累加）
    df = df[df['SERVICE'].isin(DDS_SERVICE_CODES)].copy()

    return df


def load_iso_alpha3_whitelist():
    return {
        c.alpha_3.strip().upper()
        for c in pycountry.countries
        if getattr(c, 'alpha_3', None)
    }


def keep_strict_whitelist_countries(df, iso_whitelist):
    """过滤只保留主权国家（同时过滤出口国和贸易伙伴方向的聚合体）"""
    # 过滤 REF_AREA（出口国）
    is_iso3 = df['REF_AREA'].str.len() == 3
    in_white = df['REF_AREA'].isin(iso_whitelist)
    not_macro = ~df['REF_AREA'].isin(WB_MACRO_REGION_CODES)
    # 过滤 COUNTERPART_AREA（贸易伙伴），排除 W(World)、WXD 等聚合体
    not_agg_cp = ~df['COUNTERPART_AREA'].astype(str).str.strip().isin(COUNTERPART_AGGREGATE_CODES)
    filtered = df[is_iso3 & in_white & not_macro & not_agg_cp].copy()
    return filtered


def fix_duplicate_entries_by_methodology(df):
    """
    修复数据中的重复记录问题（修复问题2：METHODOLOGY_TYPE去重）

    问题描述：
    - 同一个双边贸易流(REF_AREA, Counterpart area, SERVICE, TIME_PERIOD)有多条记录
    - 这些记录的区别在于METHODOLOGY_TYPE（DE, IE, ME, AG, AD, NaN）
    - 直接累加会导致重复计算

    解决方案：
    - 对于同一(REF_AREA, Counterpart area, SERVICE, TIME_PERIOD)组合
    - 根据METHODOLOGY_TYPE优先级选择最可靠的一条记录
    - 优先级：DE > IE > ME > AG > AD > NaN
    """
    # 转换METHODOLOGY_TYPE为字符串处理，处理NaN
    df['METHODOLOGY_TYPE_clean'] = df['METHODOLOGY_TYPE'].fillna('NA')

    # 添加优先级列
    df['priority'] = df['METHODOLOGY_TYPE_clean'].map(METHODOLOGY_PRIORITY_MAP).fillna(0)

    # 按优先级排序，保留优先级最高的记录
    df_sorted = df.sort_values(
        by=['REF_AREA', 'Counterpart area', 'SERVICE', 'TIME_PERIOD', 'priority'],
        ascending=[True, True, True, True, False]
    )

    # 去重，保留每组的第一条（优先级最高的）
    df_dedup = df_sorted.drop_duplicates(
        subset=['REF_AREA', 'Counterpart area', 'SERVICE', 'TIME_PERIOD'],
        keep='first'
    ).copy()

    # 删除临时列
    df_dedup = df_dedup.drop(columns=['METHODOLOGY_TYPE_clean', 'priority'])

    return df_dedup


def latest_year_digital_export_ranking(df, top_n=20):
    """
    生成最新年份的数字贸易出口排行榜，并计算占比统计
    """
    # 应用去重修复
    df_dedup = fix_duplicate_entries_by_methodology(df)

    latest_year = int(df_dedup['TIME_PERIOD'].max())

    # 严格筛选最新年份（确保没有跨期求和）
    df_latest = df_dedup[df_dedup['TIME_PERIOD'] == latest_year].copy()

    # 按出口国分组求和（全部国家）
    all_ranking = (
        df_latest.groupby('Reference area', as_index=False)['OBS_VALUE']
        .sum()
        .sort_values('OBS_VALUE', ascending=False)
        .reset_index(drop=True)
    )

    global_total = all_ranking['OBS_VALUE'].sum()
    total_countries = len(all_ranking)

    # Top N 排行
    ranking = all_ranking.head(top_n).copy()
    ranking.index = range(1, len(ranking) + 1)
    ranking.index.name = 'Rank'
    ranking.rename(
        columns={
            'Reference area': 'Country',
            'OBS_VALUE': 'Digital_Export_Value_MillionUSD'
        },
        inplace=True,
    )
    ranking['Digital_Export_Value_BillionUSD'] = ranking['Digital_Export_Value_MillionUSD'] / 1000
    ranking['Share_of_Global_Pct'] = (ranking['Digital_Export_Value_MillionUSD'] / global_total) * 100

    # 前33%国家的占比
    top_33pct_count = max(1, int(total_countries * 0.33))
    top_33pct_export = all_ranking.head(top_33pct_count)['OBS_VALUE'].sum()
    top_33pct_share = (top_33pct_export / global_total) * 100

    # Top 10 占比
    top10_export = all_ranking.head(10)['OBS_VALUE'].sum()
    top10_share = (top10_export / global_total) * 100

    # 打印占比统计
    print(f"\n{'='*60}")
    print(f"  {latest_year}年 全球数字贸易出口占比统计")
    print(f"{'='*60}")
    print(f"  全球总出口额: {global_total/1000:,.2f} 十亿美元")
    print(f"  参与国家数:   {total_countries}")
    print(f"  Top 10 国家占全球: {top10_share:.2f}%")
    print(f"  前33%国家({top_33pct_count}国)占全球: {top_33pct_share:.2f}%")
    print(f"{'='*60}")
    print(f"\n  Top 10 国家明细:")
    for idx, row in ranking.head(10).iterrows():
        print(f"    {idx:>2}. {row['Country']:<35s} "
              f"{row['Digital_Export_Value_BillionUSD']:>8.2f} 十亿美元  "
              f"占比 {row['Share_of_Global_Pct']:.2f}%")

    output_csv = OUTPUT_DIR / 'output_最新一年数字贸易出口总额排行榜_完整DDS版_FINAL.csv'
    ranking.to_csv(output_csv, encoding='utf-8-sig')

    # 保存占比统计到单独的CSV
    share_stats = pd.DataFrame({
        'Metric': [
            f'Global Total ({latest_year}, Million USD)',
            'Total Countries',
            'Top 10 Share (%)',
            f'Top 33% Countries Count ({top_33pct_count})',
            'Top 33% Share (%)',
        ],
        'Value': [
            f'{global_total:,.2f}',
            str(total_countries),
            f'{top10_share:.2f}',
            str(top_33pct_count),
            f'{top_33pct_share:.2f}',
        ]
    })
    share_stats.to_csv(
        OUTPUT_DIR / 'output_数字贸易出口占比统计_FINAL.csv',
        index=False, encoding='utf-8-sig'
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_df = ranking.head(10).sort_values('Digital_Export_Value_BillionUSD', ascending=True).copy()
    plot_df['Country_CN'] = plot_df['Country'].map(to_chinese_country_name)
    bars = ax.barh(plot_df['Country_CN'], plot_df['Digital_Export_Value_BillionUSD'], color='#1f77b4')
    # 在柱状图上标注占比
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{row["Share_of_Global_Pct"]:.1f}%', va='center', fontsize=9)
    ax.set_title(f'{latest_year}年 数字贸易出口Top10排行榜（已修复聚合体过滤）')
    ax.set_xlabel('出口额（十亿美元）')
    ax.set_ylabel('国家/地区')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'output_最新一年数字贸易出口总额排行榜_完整DDS版_FINAL.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return latest_year, ranking, global_total, top10_share, top_33pct_count, top_33pct_share


def digital_export_by_service(df, country=None, year=None):
    """
    按SERVICE分类统计数字贸易出口额
    """
    df_dedup = fix_duplicate_entries_by_methodology(df)

    if year is None:
        year = int(df_dedup['TIME_PERIOD'].max())

    df_year = df_dedup[df_dedup['TIME_PERIOD'] == year].copy()

    if country is not None:
        df_year = df_year[df_year['Reference area'] == country]

    by_service = df_year.groupby('SERVICE', as_index=False)['OBS_VALUE'].sum()

    # 添加SERVICE名称映射
    service_names = {
        'SF': 'Insurance services (保险服务)',
        'SG': 'Financial services (金融服务)',
        'SH': 'Charges for use of IP (知识产权)',
        'SI': 'Telecommunications, computer & info (电信计算机)',
        'SJ': 'Other business services (其他商业服务)',
        'SK': 'Personal, cultural & recreation (文化娱乐)'
    }
    by_service['SERVICE_Name'] = by_service['SERVICE'].map(service_names)
    by_service = by_service.sort_values('OBS_VALUE', ascending=False)

    return by_service


def digital_export_trend(df):
    """
    数字贸易出口额趋势（不计算占比，因为输入文件仅包含DDS服务）
    """
    df_dedup = fix_duplicate_entries_by_methodology(df)

    yearly_export = df_dedup.groupby('TIME_PERIOD', as_index=False)['OBS_VALUE'].sum()
    yearly_export.rename(columns={'OBS_VALUE': 'Digital_Export_Value_MillionUSD'}, inplace=True)
    yearly_export = yearly_export.sort_values('TIME_PERIOD')
    yearly_export['Digital_Export_Value_BillionUSD'] = yearly_export['Digital_Export_Value_MillionUSD'] / 1000

    # 计算年增长率
    yearly_export['YoY_Growth_Pct'] = yearly_export['Digital_Export_Value_MillionUSD'].pct_change() * 100

    output_csv = OUTPUT_DIR / 'output_数字贸易出口额趋势_完整DDS版_FINAL.csv'
    yearly_export.to_csv(output_csv, index=False, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly_export['TIME_PERIOD'], yearly_export['Digital_Export_Value_BillionUSD'],
            marker='o', linewidth=2, markersize=6, color='#1f77b4')
    ax.set_xlabel('年份')
    ax.set_ylabel('出口额（十亿美元）')
    ax.set_title('全球数字贸易出口额趋势（完整DDS，已去重）')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'output_数字贸易出口额趋势_完整DDS版_FINAL.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return yearly_export


def top10_share_trend(df):
    """
    计算最新年份Top10国家在过去所有年份中的出口占比趋势
    以最新年份的Top10为基准，回溯历年占比
    """
    df_dedup = fix_duplicate_entries_by_methodology(df)

    latest_year = int(df_dedup['TIME_PERIOD'].max())

    # 确定最新年份的 Top 10 国家
    df_latest = df_dedup[df_dedup['TIME_PERIOD'] == latest_year]
    top10_countries = (
        df_latest.groupby('Reference area', as_index=False)['OBS_VALUE']
        .sum()
        .nlargest(10, 'OBS_VALUE')['Reference area']
        .tolist()
    )

    # 计算每年每个国家的出口额
    yearly_country = (
        df_dedup.groupby(['TIME_PERIOD', 'Reference area'], as_index=False)['OBS_VALUE']
        .sum()
    )
    # 每年全球总额
    yearly_total = df_dedup.groupby('TIME_PERIOD', as_index=False)['OBS_VALUE'].sum()
    yearly_total.rename(columns={'OBS_VALUE': 'Global_Total'}, inplace=True)

    # 合并
    yearly_country = yearly_country.merge(yearly_total, on='TIME_PERIOD')
    yearly_country['Share_Pct'] = (yearly_country['OBS_VALUE'] / yearly_country['Global_Total']) * 100

    # 筛选 Top 10 国家
    top10_trend = yearly_country[yearly_country['Reference area'].isin(top10_countries)].copy()

    # 透视表：行=年份，列=国家
    pivot = top10_trend.pivot_table(
        index='TIME_PERIOD', columns='Reference area', values='Share_Pct'
    ).reindex(columns=top10_countries)

    # 保存 CSV
    pivot_out = pivot.copy()
    pivot_out.index.name = 'Year'
    pivot_out.to_csv(
        OUTPUT_DIR / 'output_Top10国家历年占比趋势_FINAL.csv',
        encoding='utf-8-sig', float_format='%.2f'
    )

    # 绘图：折线图
    fig, ax = plt.subplots(figsize=(14, 7))
    for country in top10_countries:
        country_label = to_chinese_country_name(country)
        ax.plot(pivot.index, pivot[country], marker='o', markersize=4, linewidth=1.5, label=country_label)
    ax.set_xlabel('年份')
    ax.set_ylabel('占全球DDS出口比重 (%)')
    ax.set_title(f'Top 10 国家数字贸易出口占比趋势（{int(pivot.index.min())}-{latest_year}）')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'output_Top10国家历年占比趋势_FINAL.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return pivot, top10_countries


def write_trend_analysis(latest_year, ranking, trend, by_service_ireland=None,
                         global_total=0, top10_share=0, top_33pct_count=0, top_33pct_share=0,
                         top10_share_pivot=None, top10_countries=None):
    """生成分析报告"""
    # 计算CAGR
    first_year = trend['TIME_PERIOD'].iloc[0]
    first_value = trend['Digital_Export_Value_MillionUSD'].iloc[0]
    latest_value = trend['Digital_Export_Value_MillionUSD'].iloc[-1]
    years = latest_year - first_year
    if years > 0 and first_value > 0:
        cagr = ((latest_value / first_value) ** (1 / years) - 1) * 100
    else:
        cagr = float('nan')

    markdown_content = f"""# 全球数字贸易格局趋势分析（完整DDS版本 v2）

**数据修正说明**：
- **修复问题1：行业分类重复累加** - 使用完整的DDS六个细分行业(SF,SF,SH,SI,SJ,SK)，而非仅使用SI
- **修复问题2：METHODOLOGY_TYPE去重** - 对于同一双边贸易流，根据METHODOLOGY_TYPE优先级选择最可靠的数据
- **修复问题3：年份聚合** - 严格筛选最新年份，确保无跨期求和
- **修复问题4：国家聚合体过滤** - 使用pycountry严格过滤，确保只包含主权国家

**DDS行业定义**：
- SF: Insurance services (保险服务)
- SG: Financial services (金融服务)
- SH: Charges for the use of intellectual property (知识产权使用费)
- SI: Telecommunications, computer and information services (电信、计算机和信息服务)
- SJ: Other business services (其他商业服务)
- SK: Personal, cultural and recreational services (个人、文化和娱乐服务)

**METHODOLOGY_TYPE优先级**：
- DE (Derivation): 直接推导数据（最可靠）
- IE (Interpolation/extrapolation): 插值/外推
- ME (Model-based estimation): 模型估算
- AG (Aggregation): 聚合数据
- AD (Adjustment): 调整数据

## 最新年份（{latest_year}）全球数字贸易出口排行榜（Top 10）

全球DDS出口总额：{global_total/1000:,.2f} 十亿美元

| 排名 | 国家 | 出口额（十亿美元） | 占全球比重 |
|:----:|:-----:|----------------:|----------:|
"""

    for idx, row in ranking.head(10).iterrows():
        markdown_content += f"| {idx} | {row['Country']} | {row['Digital_Export_Value_BillionUSD']:,.2f} | {row['Share_of_Global_Pct']:.2f}% |\n"

    markdown_content += f"""
**集中度统计**：
- Top 10 国家合计占全球DDS出口的 **{top10_share:.2f}%**
- 出口总量排名前33%的国家（{top_33pct_count}国）合计占全球DDS出口的 **{top_33pct_share:.2f}%**
"""

    markdown_content += f"""
## 爱尔兰按行业分类详细分析（{latest_year}年）
"""

    if by_service_ireland is not None:
        total = by_service_ireland['OBS_VALUE'].sum()
        markdown_content += f"\n| 行业代码 | 行业名称 | 出口额（百万美元） | 占比 |\n"
        markdown_content += f"|:---:|:---|---:|---:|\n"
        for _, row in by_service_ireland.iterrows():
            share = (row['OBS_VALUE'] / total) * 100
            markdown_content += f"| {row['SERVICE']} | {row['SERVICE_Name']} | {row['OBS_VALUE']:,.2f} | {share:.2f}% |\n"
        markdown_content += f"| **总计** | | **{total:,.2f}** | **100%** |\n"

    markdown_content += f"""

## 全球数字贸易出口额趋势

- **趋势周期**：{first_year}-{latest_year} 年
- **出口额变化**：从 {first_value:,.2f} 百万美元 增至 {latest_value:,.2f} 百万美元
- **复合年增长率(CAGR)**：{cagr:.2f}%

| 年份 | 出口额（十亿美元） | 年增长率(%) |
|:----:|----------------:|------------:|
"""
    for _, row in trend.iterrows():
        yoy = row['YoY_Growth_Pct']
        yoy_str = f"{yoy:.2f}" if pd.notna(yoy) else "-"
        markdown_content += f"| {int(row['TIME_PERIOD'])} | {row['Digital_Export_Value_BillionUSD']:,.2f} | {yoy_str} |\n"

    # Top 10 国家历年占比趋势
    if top10_share_pivot is not None and top10_countries is not None:
        markdown_content += f"""

## Top 10 国家数字贸易出口占比趋势（{int(top10_share_pivot.index.min())}-{latest_year}）

| 年份 | {'| '.join(c for c in top10_countries)} |
|:----:|{'|'.join(':---:' for _ in top10_countries)}|
"""
        for year_val, row in top10_share_pivot.iterrows():
            vals = ' | '.join(f"{row[c]:.2f}%" if pd.notna(row[c]) else "-" for c in top10_countries)
            markdown_content += f"| {int(year_val)} | {vals} |\n"

    markdown_content += f"""

## 数据质量声明

本分析数据来自OECD Balance of Trade Information System (BaTIS)。在处理过程中已修正以下问题：

1. **行业分类重复累加问题**：确保只使用DDS六个底层细分行业，避免大类与子类重复计算
2. **METHODOLOGY_TYPE重复问题**：对于同一双边贸易流有多条估算值的情况，根据优先级(DE>IE>ME>AG>AD)选择最可靠的数据
3. **年份聚合问题**：严格筛选最新年份，确保无跨期求和
4. **国家聚合体问题**：使用pycountry严格过滤，确保只包含主权国家，排除OECD、EU、World等聚合体
5. **贸易伙伴聚合体问题**：排除COUNTERPART_AREA中的W(World)、WXD等聚合体，避免双边贸易与汇总值重复累加

**数据限制说明**：
- 输入文件仅包含DDS六个服务的数据，不包含全部服务贸易数据
- 因此无法计算DDS占总服务出口的比重
- 本报告展示的是DDS出口额的绝对值趋势

**注意事项**：
- 爱尔兰等国因跨国公司总部聚集，数字贸易出口额显著高于其他国家
- 建议在引用时说明已进行上述数据清洗处理
"""

    output_md = OUTPUT_DIR / 'output_世界数字贸易格局趋势分析_完整DDS版_FINAL.md'
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(markdown_content)


def main():
    ensure_output_dir()
    print("加载数据...")
    df = load_oecd_data()

    print(f"原始DDS数据行数: {len(df)}")

    iso_whitelist = load_iso_alpha3_whitelist()
    df = keep_strict_whitelist_countries(df, iso_whitelist)
    print(f"过滤后数据行数: {len(df)}")

    print("\n生成最新年份排行...")
    latest_year, ranking, global_total, top10_share, top_33pct_count, top_33pct_share = \
        latest_year_digital_export_ranking(df, top_n=20)
    print(f"最新年份: {latest_year}")
    print("\nTop 10国家:")
    print(ranking.head(10))

    print("\n生成爱尔兰按行业分类分析...")
    by_service_ireland = digital_export_by_service(df, country='Ireland', year=latest_year)
    print("\n爱尔兰各行业出口额:")
    print(by_service_ireland)

    print("\n生成趋势分析...")
    trend = digital_export_trend(df)
    print("\n最近5年趋势:")
    print(trend.tail(5))

    print("\n生成Top10国家历年占比趋势...")
    top10_share_pivot, top10_countries = top10_share_trend(df)
    print(top10_share_pivot.tail(5))

    print("\n生成markdown报告...")
    write_trend_analysis(latest_year, ranking, trend, by_service_ireland,
                         global_total, top10_share, top_33pct_count, top_33pct_share,
                         top10_share_pivot, top10_countries)

    print("\n完成！输出文件:")
    print(f"  - output_最新一年数字贸易出口总额排行榜_完整DDS版_FINAL.csv")
    print(f"  - output_最新一年数字贸易出口总额排行榜_完整DDS版_FINAL.png")
    print(f"  - output_数字贸易出口占比统计_FINAL.csv")
    print(f"  - output_数字贸易出口额趋势_完整DDS版_FINAL.csv")
    print(f"  - output_数字贸易出口额趋势_完整DDS版_FINAL.png")
    print(f"  - output_Top10国家历年占比趋势_FINAL.csv")
    print(f"  - output_Top10国家历年占比趋势_FINAL.png")
    print(f"  - output_世界数字贸易格局趋势分析_完整DDS版_FINAL.md")


if __name__ == '__main__':
    main()
