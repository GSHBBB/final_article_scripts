from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pycountry

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / '输入文件'
OUTPUT_DIR = BASE_DIR / '输出文件'

OECD_FILE = INPUT_DIR / 'input_OECD_BaTIS_Data3.csv'
DIGITAL_SERVICE_CODE = 'SI'
EXPORT_FLOW_CODE = 'X'

WB_MACRO_REGION_CODES = {
    'ARB', 'CSS', 'CEB', 'EAR', 'EAS', 'EAP', 'EMU', 'ECA', 'ECS', 'EUU',
    'FCS', 'HIC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX', 'INX', 'LAC', 'LCN',
    'LDC', 'LIC', 'LMC', 'MEA', 'MIC', 'MNA', 'NAC', 'OED', 'OSS', 'PST',
    'PRE', 'SAS', 'SSA', 'SSF', 'TEA', 'TEC', 'TLA', 'TMN', 'TSA', 'TSS',
    'UMC', 'WLD', 'WXD'
}


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_oecd_data():
    usecols = ['REF_AREA', 'Reference area', 'TRADE_FLOW', 'SERVICE', 'TIME_PERIOD', 'OBS_VALUE']
    df = pd.read_csv(OECD_FILE, usecols=usecols)
    df = df[df['TRADE_FLOW'] == EXPORT_FLOW_CODE].copy()
    df['REF_AREA'] = df['REF_AREA'].astype(str).str.strip().str.upper()
    df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce')
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df[df['TIME_PERIOD'].notna() & df['OBS_VALUE'].notna() & (df['OBS_VALUE'] > 0)].copy()
    df['TIME_PERIOD'] = df['TIME_PERIOD'].astype(int)
    return df


def load_iso_alpha3_whitelist():
    return {
        c.alpha_3.strip().upper()
        for c in pycountry.countries
        if getattr(c, 'alpha_3', None)
    }


def keep_strict_whitelist_countries(df, iso_whitelist):
    is_iso3 = df['REF_AREA'].str.len() == 3
    in_white = df['REF_AREA'].isin(iso_whitelist)
    not_macro = ~df['REF_AREA'].isin(WB_MACRO_REGION_CODES)
    filtered = df[is_iso3 & in_white & not_macro].copy()
    return filtered


def latest_year_digital_export_ranking(df, top_n=20):
    digital_df = df[df['SERVICE'] == DIGITAL_SERVICE_CODE]
    latest_year = int(digital_df['TIME_PERIOD'].max())
    ranking = (
        digital_df[digital_df['TIME_PERIOD'] == latest_year]
        .groupby('Reference area', as_index=False)['OBS_VALUE']
        .sum()
        .sort_values('OBS_VALUE', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    ranking.index = ranking.index + 1
    ranking.index.name = 'Rank'
    ranking.rename(
        columns={
            'Reference area': 'Country',
            'OBS_VALUE': 'Digital_Export_Value_MillionUSD'
        },
        inplace=True,
    )

    output_csv = OUTPUT_DIR / 'output_最新一年数字贸易出口总额排行榜_主权国家版.csv'
    ranking.to_csv(output_csv, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_df = ranking.sort_values('Digital_Export_Value_MillionUSD', ascending=True)
    ax.barh(plot_df['Country'], plot_df['Digital_Export_Value_MillionUSD'], color='#1f77b4')
    ax.set_title(f'最新年份（{latest_year}）数字贸易出口总额排行榜（主权国家版）')
    ax.set_xlabel('出口额（百万美元）')
    ax.set_ylabel('国家/地区')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'output_最新一年数字贸易出口总额排行榜_主权国家版.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return latest_year, ranking


def digital_share_trend(df):
    yearly_total = df.groupby('TIME_PERIOD', as_index=False)['OBS_VALUE'].sum()
    yearly_total.rename(columns={'OBS_VALUE': 'Total_Export_Value_MillionUSD'}, inplace=True)

    digital_df = df[df['SERVICE'] == DIGITAL_SERVICE_CODE]
    yearly_digital = digital_df.groupby('TIME_PERIOD', as_index=False)['OBS_VALUE'].sum()
    yearly_digital.rename(columns={'OBS_VALUE': 'Digital_Export_Value_MillionUSD'}, inplace=True)

    trend = yearly_total.merge(yearly_digital, on='TIME_PERIOD', how='left')
    trend['Digital_Export_Value_MillionUSD'] = trend['Digital_Export_Value_MillionUSD'].fillna(0)
    trend['Digital_Share_Pct'] = (
        trend['Digital_Export_Value_MillionUSD'] / trend['Total_Export_Value_MillionUSD'] * 100
    )
    trend = trend.sort_values('TIME_PERIOD').reset_index(drop=True)

    trend.to_csv(OUTPUT_DIR / 'output_数字服务出口占总出口比重趋势_主权国家版.csv', index=False, encoding='utf-8-sig')

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(
        trend['TIME_PERIOD'],
        trend['Digital_Export_Value_MillionUSD'],
        marker='o',
        color='#1f77b4',
        label='数字服务出口额'
    )
    ax1.set_xlabel('年份')
    ax1.set_ylabel('数字服务出口额（百万美元）', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(axis='both', linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        trend['TIME_PERIOD'],
        trend['Digital_Share_Pct'],
        marker='s',
        color='#d62728',
        label='数字服务出口占比'
    )
    ax2.set_ylabel('占总出口比重（%）', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    ax1.set_title('数字服务出口额与占总出口比重趋势（主权国家版，OECD BaTIS）')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'output_数字服务出口占总出口比重趋势_主权国家版.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return trend


def write_trend_analysis(trend, latest_year, ranking):
    first_year = int(trend['TIME_PERIOD'].min())
    latest_row = trend[trend['TIME_PERIOD'] == latest_year].iloc[0]
    first_row = trend[trend['TIME_PERIOD'] == first_year].iloc[0]

    years = latest_year - first_year
    if years > 0 and first_row['Digital_Export_Value_MillionUSD'] > 0:
        cagr = (
            (latest_row['Digital_Export_Value_MillionUSD'] / first_row['Digital_Export_Value_MillionUSD'])
            ** (1 / years)
            - 1
        ) * 100
    else:
        cagr = float('nan')

    trend = trend.copy()
    trend['Share_Change_Pct_Point'] = trend['Digital_Share_Pct'].diff()
    max_up = trend.loc[trend['Share_Change_Pct_Point'].idxmax()]
    max_down = trend.loc[trend['Share_Change_Pct_Point'].idxmin()]

    top3 = ranking.head(3)
    top3_text = '\n'.join(
        [
            f"- {idx}. {row['Country']}：{row['Digital_Export_Value_MillionUSD']:.2f} 百万美元"
            for idx, row in top3.iterrows()
        ]
    )

    analysis = f"""# 世界数字贸易格局趋势分析（主权国家版，基于 OECD BaTIS）

## 1) 最新一年数字贸易出口总额排行榜（Top 3）
{top3_text}

## 2) 数字服务出口占总出口比重变化
- 样本期：{first_year}-{latest_year}
- 基期（{first_year}）数字服务出口占比：{first_row['Digital_Share_Pct']:.2f}%
- 最新期（{latest_year}）数字服务出口占比：{latest_row['Digital_Share_Pct']:.2f}%
- 占比变动：{latest_row['Digital_Share_Pct'] - first_row['Digital_Share_Pct']:.2f} 个百分点

## 3) 总体发展趋势解读
- 数字服务出口额从 {first_row['Digital_Export_Value_MillionUSD']:.2f} 增至 {latest_row['Digital_Export_Value_MillionUSD']:.2f}（百万美元）。
- 样本期数字服务出口额 CAGR：{cagr:.2f}%。
- 占比年度最大上升出现在 {int(max_up['TIME_PERIOD'])} 年，提升 {max_up['Share_Change_Pct_Point']:.2f} 个百分点。
- 占比年度最大回落出现在 {int(max_down['TIME_PERIOD'])} 年，变动 {max_down['Share_Change_Pct_Point']:.2f} 个百分点。

注：数字服务定义采用 SERVICE=SI；总出口定义为 OECD BaTIS 中 TRADE_FLOW=X 的全部服务出口。统计对象按 REF_AREA 与 ISO 3166-1 alpha-3 严格白名单过滤。
"""

    report_path = OUTPUT_DIR / 'output_世界数字贸易格局趋势分析_主权国家版.md'
    report_path.write_text(analysis, encoding='utf-8')


def main():
    ensure_output_dir()
    iso_whitelist = load_iso_alpha3_whitelist()
    print(f'ISO3白名单国家数: {len(iso_whitelist)}')

    print('正在读取 OECD BaTIS 数据...')
    df = load_oecd_data()
    print(f'有效出口样本行数: {len(df)}')

    df_sovereign = keep_strict_whitelist_countries(df, iso_whitelist)
    print(f'严格白名单国家口径样本行数: {len(df_sovereign)}')

    print('\n1. 计算最新一年数字贸易出口总额排行榜...')
    latest_year, ranking = latest_year_digital_export_ranking(df_sovereign, top_n=20)
    print(f'最新年份: {latest_year}')
    print(ranking.head(10).to_string())

    print('\n2. 计算数字服务出口占总出口比重变化趋势...')
    trend = digital_share_trend(df_sovereign)
    print(trend[['TIME_PERIOD', 'Digital_Share_Pct']].tail(10).to_string(index=False))

    print('\n3. 生成趋势分析报告...')
    write_trend_analysis(trend, latest_year, ranking)

    print('\n分析完成，输出文件位于 输出文件 目录：')
    print('- output_最新一年数字贸易出口总额排行榜_主权国家版.csv / .png')
    print('- output_数字服务出口占总出口比重趋势_主权国家版.csv / .png')
    print('- output_世界数字贸易格局趋势分析_主权国家版.md')


if __name__ == '__main__':
    main()
