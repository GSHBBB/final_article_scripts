import pandas as pd
import numpy as np
import os
import warnings
from linearmodels.panel import PanelOLS

# 抑制不必要的警告
warnings.filterwarnings('ignore')

# ============================================================================
# 第一部分：数据读取与预处理
# ============================================================================
print("=" * 90)
print("双向固定效应面板回归：终极基准、异质性与机制检验全流程")
print("=" * 90)

script_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(script_dir, '输出文件')
output_dir = os.path.join(script_dir, '输出文件_不强制InternetUse')
os.makedirs(output_dir, exist_ok=True)

def resolve_input(filename):
    preferred = os.path.join(output_dir, filename)
    fallback = os.path.join(source_dir, filename)
    return preferred if os.path.exists(preferred) else fallback

data_file = resolve_input("output_Master_Panel.csv")
resilience_file = resolve_input("output_final_resilience_panel.csv")

# 1. 读取数据
df = pd.read_csv(data_file)
df_resilience = pd.read_csv(resilience_file)

# 如果 Master_Panel 中没有中心度，则从 resilience 表合并
if 'Out_Degree_Centrality' in df_resilience.columns and 'Out_Degree_Centrality' not in df.columns:
    centrality_df = df_resilience[['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']].copy()
    df = df.merge(centrality_df, on=['REF_AREA', 'TIME_PERIOD'], how='left')

# 2. 提取变量（坚决剔除 Internet_Use 以释放边缘国家样本）
required_cols = ['REF_AREA', 'TIME_PERIOD', 'True_Resilience', 'Exposure', 
                 'GDP_PC', 'FDI', 'Out_Degree_Centrality', 'GE', 'RQ']

# 仅提取存在于数据表中的列并清理空值
available_cols = [col for col in required_cols if col in df.columns]
df_model = df[available_cols].dropna().copy()

# 3. 变量转换
# 因变量校正：取对数的相反数，使得数值越大代表韧性越强
df_model['ln_Resilience_Inv'] = -np.log(df_model['True_Resilience'])
# 人均 GDP 取对数
df_model['log_GDP_PC'] = np.log(df_model['GDP_PC'])

# 4. 设置多重面板索引
df_indexed = df_model.set_index(['REF_AREA', 'TIME_PERIOD'])

# 5. 生成核心滞后项（处理政策发酵时滞）
df_indexed['L1_Exposure'] = df_indexed.groupby(level='REF_AREA')['Exposure'].shift(1)
df_indexed['L1_Centrality'] = df_indexed.groupby(level='REF_AREA')['Out_Degree_Centrality'].shift(1)

# 清理因 shift 产生的缺失值
df_indexed_clean = df_indexed.dropna(subset=['L1_Exposure', 'L1_Centrality', 'ln_Resilience_Inv'])

# 强行添加截距项
df_indexed_clean['Constant'] = 1.0

# 6. 构造交乘项 (用于验证网络过饱和与内耗效应)
df_indexed_clean['Interaction'] = df_indexed_clean['L1_Exposure'] * df_indexed_clean['L1_Centrality']

# ============================================================================
# 第二部分：全样本交乘项回归（证明边际效用递减的核心表格）
# ============================================================================
print("\n" + "=" * 90)
print("【实证第一步】全样本交乘项模型：验证“核心国过饱和”与“边缘国雪中送炭”")
print("=" * 90)

y_inter = df_indexed_clean['ln_Resilience_Inv']
X_inter = df_indexed_clean[['Constant', 'L1_Exposure', 'Interaction', 'log_GDP_PC', 'FDI']]

model_inter = PanelOLS(y_inter, X_inter, entity_effects=True, time_effects=True)
results_inter = model_inter.fit(cov_type='clustered', cluster_entity=True)
print(results_inter.summary)

# ============================================================================
# 第三部分：异质性分组回归（证明普惠赋能机制）
# ============================================================================
print("\n" + "=" * 90)
print("【实证第二步】异质性分组检验：三分位 (Top 33% vs Bottom 33%) 切割")
print("=" * 90)

# 计算中心度的 33% 和 66% 分位数阈值
q33 = df_indexed_clean['L1_Centrality'].quantile(0.33)
q66 = df_indexed_clean['L1_Centrality'].quantile(0.66)

df_peri = df_indexed_clean[df_indexed_clean['L1_Centrality'] <= q33]
df_core = df_indexed_clean[df_indexed_clean['L1_Centrality'] >= q66]

print(f"最边缘组 (Bottom 33%) 样本量: {len(df_peri)}")
print(f"最核心组 (Top 33%) 样本量: {len(df_core)}")

print("\n>>> 分组回归 A: 最边缘组 (期待：强烈的雪中送炭效应)")
y_p = df_peri['ln_Resilience_Inv']
X_p = df_peri[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
res_p = PanelOLS(y_p, X_p, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
print("分组回归 A（最边缘组）完整回归结果：")
print(res_p.summary)

print("\n>>> 分组回归 B: 最核心组 (期待：过饱和甚至负效应)")
y_c = df_core['ln_Resilience_Inv']
X_c = df_core[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
res_c = PanelOLS(y_c, X_c, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
print("分组回归 B（最核心组）完整回归结果：")
print(res_c.summary)

print("\n>>> 全样本基准回归（与基准模型口径一致）")
y_base = df_indexed_clean['ln_Resilience_Inv']
X_base = df_indexed_clean[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
res_base = PanelOLS(y_base, X_base, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
print("全样本基准回归完整结果：")
print(res_base.summary)

# ============================================================================
# 第四部分：机制检验（释放个体固定效应以避免慢变量“绞杀”）
# ============================================================================
print("\n" + "=" * 90)
print("【实证第三步】传导机制检验：释放慢变量的组间差异 (Relaxing Entity Effects)")
print("=" * 90)

if 'GE' in df_indexed_clean.columns and 'RQ' in df_indexed_clean.columns:
    # 提取自变量
    X_mech_final = df_indexed_clean[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
    
    # ----------------------------------------------------------------------
    # 机制 1：政府效率 (GE)
    # ----------------------------------------------------------------------
    print("\n>>> 机制回归 1: L1_Exposure -> 政府效率 (GE)")
    y_ge = df_indexed_clean['GE']
    
    # 【核心改动】：entity_effects=False 释放慢变量方差；仅控制年份固定效应
    model_ge_final = PanelOLS(y_ge, X_mech_final, entity_effects=False, time_effects=True)
    res_ge_final = model_ge_final.fit(cov_type='clustered', cluster_entity=True)
    print("机制回归 1（GE）完整回归结果：")
    print(res_ge_final.summary)
    if res_ge_final.pvalues['L1_Exposure'] < 0.10:
        print("🎉 恭喜！机制 1 显著成立！数字规则敞口的扩大显著提升了缔约国的政府效率！")

    # ----------------------------------------------------------------------
    # 机制 2：监管质量 (RQ)
    # ----------------------------------------------------------------------
    print("\n>>> 机制回归 2: L1_Exposure -> 监管质量 (RQ)")
    y_rq = df_indexed_clean['RQ']
    
    # 【核心改动】：同样释放个体固定效应
    model_rq_final = PanelOLS(y_rq, X_mech_final, entity_effects=False, time_effects=True)
    res_rq_final = model_rq_final.fit(cov_type='clustered', cluster_entity=True)
    print("机制回归 2（RQ）完整回归结果：")
    print(res_rq_final.summary)
    if res_rq_final.pvalues['L1_Exposure'] < 0.10:
        print("🎉 恭喜！机制 2 显著成立！倒逼国内监管质量优化的理论闭环达成！")

    # ----------------------------------------------------------------------
    # 机制 3/4：按国家中心度排名后33%分组回归
    # ----------------------------------------------------------------------
    print("\n>>> 机制分组回归：国家中心度排名后33%样本")
    country_rank = (
        df_indexed_clean
        .reset_index()
        .groupby('REF_AREA', as_index=False)['L1_Centrality']
        .mean()
        .rename(columns={'L1_Centrality': 'avg_L1_Centrality'})
    )
    rank_cutoff = country_rank['avg_L1_Centrality'].quantile(0.33)
    bottom33_countries = set(country_rank.loc[country_rank['avg_L1_Centrality'] <= rank_cutoff, 'REF_AREA'])

    df_bottom33_rank = df_indexed_clean[
        df_indexed_clean.index.get_level_values('REF_AREA').isin(bottom33_countries)
    ].copy()
    print(f"后33%国家数: {len(bottom33_countries)}, 样本量: {len(df_bottom33_rank)}")

    if len(df_bottom33_rank) > 0:
        X_mech_bottom = df_bottom33_rank[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]

        print("\n>>> 机制回归 3: Bottom33 国家组 L1_Exposure -> GE")
        y_ge_bottom = df_bottom33_rank['GE']
        model_ge_bottom = PanelOLS(y_ge_bottom, X_mech_bottom, entity_effects=False, time_effects=True)
        res_ge_bottom33 = model_ge_bottom.fit(cov_type='clustered', cluster_entity=True)
        print("机制回归 3（Bottom33-GE）完整回归结果：")
        print(res_ge_bottom33.summary)

        print("\n>>> 机制回归 4: Bottom33 国家组 L1_Exposure -> RQ")
        y_rq_bottom = df_bottom33_rank['RQ']
        model_rq_bottom = PanelOLS(y_rq_bottom, X_mech_bottom, entity_effects=False, time_effects=True)
        res_rq_bottom33 = model_rq_bottom.fit(cov_type='clustered', cluster_entity=True)
        print("机制回归 4（Bottom33-RQ）完整回归结果：")
        print(res_rq_bottom33.summary)
    else:
        print("Bottom33 国家组样本为空，跳过机制分组回归。")
else:
    print("未检测到 GE 或 RQ 列，请检查 output_Master_Panel.csv 中是否成功包含了这些变量。")

print("\n" + "=" * 90)
print("恭喜！这篇论文所有的实证代码运行完毕，请直接复制结果填入论文表格中。")
print("=" * 90)

# ============================================================================
# 导出回归结果表与关键系数汇总表
# ============================================================================
def _stars(p):
    if pd.isna(p):
        return ''
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.1:
        return '*'
    return ''


def _coef_cell(res, var):
    if var not in res.params.index:
        return '—', '—'
    coef = res.params[var]
    tval = res.tstats[var] if var in res.tstats.index else np.nan
    p = res.pvalues[var] if var in res.pvalues.index else np.nan
    coef_txt = f"{coef:.4f}{_stars(p)}"
    t_txt = f"({tval:.4f})" if not pd.isna(tval) else '—'
    return coef_txt, t_txt


def _within_r2(res):
    for attr in ('rsquared_within', 'rsquared'):
        val = getattr(res, attr, np.nan)
        if not pd.isna(val):
            return float(val)
    return np.nan


def _single_model_table_lines(model_title, dep_var, res, controls_flag, countries_count, entity_fe, row_spec):
    period_txt = '2005-2023'
    lines = []
    lines.append(f'### 表：{model_title}')
    lines.append(f'被解释变量：{dep_var}')
    lines.append('')
    lines.append(f'| 变量 | {model_title} |')
    lines.append('|---|---:|')

    for row_label, var in row_spec:
        coef_txt, se_txt = _coef_cell(res, var)
        lines.append(f'| {row_label} | {coef_txt} |')
        lines.append(f'|  | {se_txt} |')

    lines.append(f'| Country FE | {entity_fe} |')
    lines.append('| Year FE | Yes |')
    lines.append(f'| Controls | {controls_flag} |')
    lines.append('| SE Cluster | Country |')
    lines.append(f'| Observations (N) | {int(res.nobs)} |')
    lines.append(f'| Countries | {int(countries_count)} |')
    lines.append(f'| Period | {period_txt} |')
    lines.append(f'| Within R² | {_within_r2(res):.3f} |')
    lines.append('')
    lines.append('注：括号内为t统计量（基于国家层面聚类稳健协方差计算）。')
    lines.append('机制模型仅包含年份固定效应，不包含个体固定效应，以释放慢变量的组间差异。')
    lines.append('***p<0.01，**p<0.05，*p<0.1。')
    return lines


def export_mechanism_three_line_tables(out_dir):
    row_spec = [
        ('L1_Exposure', 'L1_Exposure'),
        ('log_GDP_PC', 'log_GDP_PC'),
        ('FDI', 'FDI'),
    ]

    model_defs = []
    if 'res_ge_final' in globals():
        model_defs.append({
            'title': '机制模型1：L1_Exposure -> GE',
            'col': '(1) 全样本 GE',
            'dep': 'GE',
            'res': res_ge_final,
            'controls': 'Yes',
            'entity_fe': 'No',
            'countries': df_indexed_clean.index.get_level_values(0).nunique(),
            'base_name': 'output_机制检验三线表_模型1_GE_全样本',
        })
    if 'res_rq_final' in globals():
        model_defs.append({
            'title': '机制模型2：L1_Exposure -> RQ',
            'col': '(2) 全样本 RQ',
            'dep': 'RQ',
            'res': res_rq_final,
            'controls': 'Yes',
            'entity_fe': 'No',
            'countries': df_indexed_clean.index.get_level_values(0).nunique(),
            'base_name': 'output_机制检验三线表_模型2_RQ_全样本',
        })
    if 'res_ge_bottom33' in globals() and 'df_bottom33_rank' in globals():
        model_defs.append({
            'title': '机制模型3：L1_Exposure -> GE（Bottom33国家组）',
            'col': '(3) Bottom33 GE',
            'dep': 'GE',
            'res': res_ge_bottom33,
            'controls': 'Yes',
            'entity_fe': 'No',
            'countries': df_bottom33_rank.index.get_level_values(0).nunique(),
            'base_name': 'output_机制检验三线表_模型3_GE_Bottom33',
        })
    if 'res_rq_bottom33' in globals() and 'df_bottom33_rank' in globals():
        model_defs.append({
            'title': '机制模型4：L1_Exposure -> RQ（Bottom33国家组）',
            'col': '(4) Bottom33 RQ',
            'dep': 'RQ',
            'res': res_rq_bottom33,
            'controls': 'Yes',
            'entity_fe': 'No',
            'countries': df_bottom33_rank.index.get_level_values(0).nunique(),
            'base_name': 'output_机制检验三线表_模型4_RQ_Bottom33',
        })

    out_files = []
    for m in model_defs:
        lines = _single_model_table_lines(
            model_title=m['title'],
            dep_var=m['dep'],
            res=m['res'],
            controls_flag=m['controls'],
            countries_count=m['countries'],
            entity_fe=m['entity_fe'],
            row_spec=row_spec,
        )
        path_md = os.path.join(out_dir, m['base_name'] + '.md')
        with open(path_md, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        out_files.append(path_md)

    if len(model_defs) == 0:
        return out_files, None

    period_txt = '2005-2023'
    lines = []
    lines.append('# 机制检验结果三线表（表X-X）')
    lines.append('')
    lines.append('| 变量 | ' + ' | '.join([m['col'] for m in model_defs]) + ' |')
    lines.append('|---|' + '|'.join(['---:' for _ in model_defs]) + '|')

    for row_label, var in row_spec:
        coefs, ses = [], []
        for m in model_defs:
            coef_txt, se_txt = _coef_cell(m['res'], var)
            coefs.append(coef_txt)
            ses.append(se_txt)
        lines.append(f'| {row_label} | ' + ' | '.join(coefs) + ' |')
        lines.append('|  | ' + ' | '.join(ses) + ' |')

    lines.append('| Country FE | ' + ' | '.join([m['entity_fe'] for m in model_defs]) + ' |')
    lines.append('| Year FE | ' + ' | '.join(['Yes' for _ in model_defs]) + ' |')
    lines.append('| Controls | ' + ' | '.join([m['controls'] for m in model_defs]) + ' |')
    lines.append('| SE Cluster | ' + ' | '.join(['Country' for _ in model_defs]) + ' |')
    lines.append('| Observations (N) | ' + ' | '.join([str(int(m['res'].nobs)) for m in model_defs]) + ' |')
    lines.append('| Countries | ' + ' | '.join([str(int(m['countries'])) for m in model_defs]) + ' |')
    lines.append('| Period | ' + ' | '.join([period_txt for _ in model_defs]) + ' |')
    lines.append('| Within R² | ' + ' | '.join([f"{_within_r2(m['res']):.3f}" for m in model_defs]) + ' |')
    lines.append('')
    lines.append('注：括号内为t统计量（基于国家层面聚类稳健协方差计算）。')
    lines.append('机制模型仅包含年份固定效应，不包含个体固定效应，以释放慢变量的组间差异。')
    lines.append('***p<0.01，**p<0.05，*p<0.1。')

    combined_md = os.path.join(out_dir, 'output_机制检验三线表_合并版.md')
    with open(combined_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return out_files, combined_md


try:
    out_dir = output_dir
    excel_path = os.path.join(out_dir, "mechanism_regression_results.xlsx")
    summary_csv = os.path.join(out_dir, "regression_summary.csv")

    result_list = [
        ("full_sample", res_base),
        ("peri", res_p),
        ("core", res_c),
        ("GE", res_ge_final) if 'res_ge_final' in globals() else ("GE", None),
        ("RQ", res_rq_final) if 'res_rq_final' in globals() else ("RQ", None),
        ("GE_bottom33_rank", res_ge_bottom33) if 'res_ge_bottom33' in globals() else ("GE_bottom33_rank", None),
        ("RQ_bottom33_rank", res_rq_bottom33) if 'res_rq_bottom33' in globals() else ("RQ_bottom33_rank", None),
    ]

    summary_rows = []
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for name, res in result_list:
            if res is None:
                continue
            # 尝试安全读取常用统计量
            params = getattr(res, 'params', None)
            pvals = getattr(res, 'pvalues', None)
            std_err = getattr(res, 'std_errors', None)
            if std_err is None:
                std_err = getattr(res, 'std_err', None)
            # 置信区间
            try:
                ci = res.conf_int()
            except Exception:
                ci = None

            df_out = pd.DataFrame({
                'coef': params,
                'std_err': std_err,
                'pvalue': pvals,
            })
            if ci is not None and isinstance(ci, pd.DataFrame) and ci.shape[1] >= 2:
                df_out['ci_lower'] = ci.iloc[:, 0]
                df_out['ci_upper'] = ci.iloc[:, 1]

            # 写入 Excel 每个 sheet
            sheet_name = name[:31]
            df_out.to_excel(writer, sheet_name=sheet_name)

            # 抽取关键系数：统一按基准模型口径仅保留 L1_Exposure
            for var in ['L1_Exposure']:
                if params is not None and var in params.index:
                    summary_rows.append({
                        'model': name,
                        'variable': var,
                        'coef': float(params.loc[var]),
                        'pvalue': float(pvals.loc[var]) if pvals is not None else np.nan,
                    })

    # 保存汇总表
    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_csv, index=False)
        print(f"已导出回归结果文件: {excel_path}")
        print(f"已导出关键系数汇总: {summary_csv}")
    else:
        print("未能抽取到关键系数，未生成汇总表。")

    model_table_files, combined_table_file = export_mechanism_three_line_tables(out_dir)
    for p in model_table_files:
        print(f"已导出机制检验三线表: {p}")
    if combined_table_file is not None:
        print(f"已导出机制检验三线表(合并版): {combined_table_file}")
except Exception as e:
    print("导出回归结果时发生错误:", e)