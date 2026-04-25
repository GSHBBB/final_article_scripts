import pandas as pd
import numpy as np
import os
import warnings
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
warnings.filterwarnings('ignore')

# ============================================================================
# 第一部分：数据读取与预处理
# ============================================================================
print("=" * 90)
print("双向固定效应面板回归：样本释放与交乘项检验（V4 不强制Internet_Use版）")
print("=" * 90)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, '输出文件')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '输出文件_不强制InternetUse')
os.makedirs(OUTPUT_DIR, exist_ok=True)

data_file = os.path.join(INPUT_DIR, "output_Master_Panel.csv")
resilience_file = os.path.join(INPUT_DIR, "output_final_resilience_panel.csv")

df = pd.read_csv(data_file)
df_resilience = pd.read_csv(resilience_file)
if 'Out_Degree_Centrality' in df_resilience.columns:
    centrality_df = df_resilience[['TIME_PERIOD', 'REF_AREA', 'Out_Degree_Centrality']].copy()
    df = df.merge(centrality_df, on=['REF_AREA', 'TIME_PERIOD'], how='left')

# 【关键修改 1】：主回归不再强制 Internet_Use 完整，仅在稳健性R2中使用。
required_cols = ['REF_AREA', 'TIME_PERIOD', 'True_Resilience', 'Exposure', 'GDP_PC', 'FDI', 'Out_Degree_Centrality']
model_cols = required_cols + (['Internet_Use'] if 'Internet_Use' in df.columns else [])
df_model = df[model_cols].dropna(subset=required_cols).copy()

# 因变量与对数化
df_model['ln_Resilience_Inv'] = -np.log(df_model['True_Resilience'])
df_model['log_GDP_PC'] = np.log(df_model['GDP_PC'])

df_indexed = df_model.set_index(['REF_AREA', 'TIME_PERIOD'])

# 生成滞后1期
df_indexed['L1_Exposure'] = df_indexed.groupby(level='REF_AREA')['Exposure'].shift(1)
df_indexed['L1_Centrality'] = df_indexed.groupby(level='REF_AREA')['Out_Degree_Centrality'].shift(1)
if 'Internet_Use' in df_indexed.columns:
    df_indexed['L1_Internet_Use'] = df_indexed.groupby(level='REF_AREA')['Internet_Use'].shift(1)
else:
    df_indexed['L1_Internet_Use'] = np.nan

df_indexed_clean = df_indexed.dropna(subset=['L1_Exposure', 'L1_Centrality', 'ln_Resilience_Inv'])
df_indexed_clean['Constant'] = 1.0

# 【关键修改 2】：构造核心交乘项 (Exposure * Centrality)
df_indexed_clean['Interaction'] = df_indexed_clean['L1_Exposure'] * df_indexed_clean['L1_Centrality']

# ============================================================================
# 第二部分：全样本交乘项回归（用最严谨的方式证明异质性）
# ============================================================================
print("\n" + "-" * 80)
print(">>> 模型 1: 全样本交互项模型 (验证边际效用递减)")
y_inter = df_indexed_clean['ln_Resilience_Inv']
X_inter = df_indexed_clean[['Constant', 'L1_Exposure', 'Interaction', 'log_GDP_PC', 'FDI']]

model_inter = PanelOLS(y_inter, X_inter, entity_effects=True, time_effects=True)
results_inter = model_inter.fit(cov_type='clustered', cluster_entity=True)
print(results_inter.summary)

print("\n【交乘项经济学含义解析】")
print("如果 L1_Exposure 系数为正(***)，Interaction 系数为负(***)：")
print("这证明：在网络最边缘(Centrality接近0)，规则对韧性有极强的正向拉动（雪中送炭）；")
print("随着网络中心度升高，正向作用被 Interaction 的负值抵消，直至在核心国家转为负效应（竞争内耗）！")

# ============================================================================
# 第三部分：重新运行 33% 宽口径分组回归 (验证机制)
# ============================================================================
print("\n" + "=" * 90)
print("【放宽口径的三分位分组检验】")
q33 = df_indexed_clean['L1_Centrality'].quantile(0.33)
q66 = df_indexed_clean['L1_Centrality'].quantile(0.66)

df_peri = df_indexed_clean[df_indexed_clean['L1_Centrality'] <= q33]
df_core = df_indexed_clean[df_indexed_clean['L1_Centrality'] >= q66]

print(f"释放样本后，边缘组(Bottom 33%)样本量: {len(df_peri)}")

print("\n>>> 模型 2: 真正的边缘组回归 (剔除缺失值干扰后)")
y_p = df_peri['ln_Resilience_Inv']
X_p = df_peri[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
res_p = PanelOLS(y_p, X_p, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(f"L1_Exposure 系数: {res_p.params['L1_Exposure']:.6f} | P值: {res_p.pvalues['L1_Exposure']:.6f}")

print("\n>>> 模型 3: 真正的核心组回归")
y_c = df_core['ln_Resilience_Inv']
X_c = df_core[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
res_c = PanelOLS(y_c, X_c, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(f"L1_Exposure 系数: {res_c.params['L1_Exposure']:.6f} | P值: {res_c.pvalues['L1_Exposure']:.6f}")


print("\n>>> 模型 4: 全样本基准回归 (与分组回归保持同一形式)")
y_base = df_indexed_clean['ln_Resilience_Inv']
X_base = df_indexed_clean[['Constant', 'L1_Exposure', 'log_GDP_PC', 'FDI']]
res_base = PanelOLS(y_base, X_base, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(f"L1_Exposure 系数: {res_base.params['L1_Exposure']:.6f} | P值: {res_base.pvalues['L1_Exposure']:.6f}")


print("\n>>> 模型 0: 全样本简约模型 (仅核心变量, 无控制变量)")
y_m1 = df_indexed_clean['ln_Resilience_Inv']
X_m1 = df_indexed_clean[['Constant', 'L1_Exposure']]
res_m1 = PanelOLS(y_m1, X_m1, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
print(f"L1_Exposure 系数: {res_m1.params['L1_Exposure']:.6f} | P值: {res_m1.pvalues['L1_Exposure']:.6f}")


# ============================================================================
# 第四部分：稳健性检验
# ============================================================================
print("\n" + "=" * 90)
print("【稳健性检验】")


def fit_fe_model(data, y_col, x_cols, cluster_entity=True, cluster_time=False):
    y = data[y_col]
    x = data[x_cols].copy()
    if 'Constant' not in x.columns:
        x['Constant'] = 1.0
        x = x[['Constant'] + [c for c in x.columns if c != 'Constant']]
    model = PanelOLS(y, x, entity_effects=True, time_effects=True)
    return model.fit(cov_type='clustered', cluster_entity=cluster_entity, cluster_time=cluster_time)


robustness_results = []

# R1: 对敞口做1%/99%缩尾
df_r1 = df_indexed_clean.copy()
q1 = df_r1['L1_Exposure'].quantile(0.01)
q99 = df_r1['L1_Exposure'].quantile(0.99)
df_r1['L1_Exposure_W'] = df_r1['L1_Exposure'].clip(lower=q1, upper=q99)
df_r1['Interaction_W'] = df_r1['L1_Exposure_W'] * df_r1['L1_Centrality']
res_r1 = fit_fe_model(df_r1, 'ln_Resilience_Inv', ['Constant', 'L1_Exposure_W', 'Interaction_W', 'log_GDP_PC', 'FDI'])
robustness_results.append(('R1_Exposure缩尾(1%-99%)', res_r1, 'L1_Exposure_W'))
print(f"R1 完成: L1_Exposure_W={res_r1.params.get('L1_Exposure_W', np.nan):.6f}, p={res_r1.pvalues.get('L1_Exposure_W', np.nan):.6f}")

# R2: 加入互联网使用率控制项（滞后一期）
df_r2 = df_indexed_clean.dropna(subset=['L1_Internet_Use']).copy()
if len(df_r2) > 0:
    res_r2 = fit_fe_model(df_r2, 'ln_Resilience_Inv', ['Constant', 'L1_Exposure', 'Interaction', 'log_GDP_PC', 'FDI', 'L1_Internet_Use'])
    robustness_results.append(('R2_加入互联网使用率', res_r2, 'L1_Exposure'))
    print(f"R2 完成: L1_Exposure={res_r2.params.get('L1_Exposure', np.nan):.6f}, p={res_r2.pvalues.get('L1_Exposure', np.nan):.6f}")
else:
    print("R2 跳过：L1_Internet_Use 无可用样本。")

# R3: 双向聚类标准误（国家+年份）
res_r3 = fit_fe_model(df_indexed_clean, 'ln_Resilience_Inv', ['Constant', 'L1_Exposure', 'Interaction', 'log_GDP_PC', 'FDI'], cluster_entity=True, cluster_time=True)
robustness_results.append(('R3_双向聚类标准误', res_r3, 'L1_Exposure'))
print(f"R3 完成: L1_Exposure={res_r3.params.get('L1_Exposure', np.nan):.6f}, p={res_r3.pvalues.get('L1_Exposure', np.nan):.6f}")

# R4: 排除疫情冲击年
mask_pre_covid = ~df_indexed_clean.index.get_level_values('TIME_PERIOD').isin([2020, 2021])
df_r4 = df_indexed_clean[mask_pre_covid].copy()
res_r4 = fit_fe_model(df_r4, 'ln_Resilience_Inv', ['Constant', 'L1_Exposure', 'Interaction', 'log_GDP_PC', 'FDI'])
robustness_results.append(('R4_剔除2020-2021', res_r4, 'L1_Exposure'))
print(f"R4 完成: L1_Exposure={res_r4.params.get('L1_Exposure', np.nan):.6f}, p={res_r4.pvalues.get('L1_Exposure', np.nan):.6f}")


def build_robustness_summary(results):
    rows = []
    for name, res, key_var in results:
        rows.append({
            '检验项': name,
            '核心变量': key_var,
            '系数': res.params.get(key_var, np.nan),
            'P值': res.pvalues.get(key_var, np.nan),
            '样本量': getattr(res, 'nobs', np.nan),
            'R平方': getattr(res, 'rsquared', np.nan),
        })
    return pd.DataFrame(rows)


robustness_df = build_robustness_summary(robustness_results)


# ============================================================================
# 第五部分：将所有回归结果导出为 Excel
# ============================================================================
def results_to_dataframe(res):
    try:
        params = res.params
    except Exception:
        return None, {}

    # 预先提取置信区间表，避免在循环中重复计算。
    try:
        ci_table = res.conf_int()
    except Exception:
        ci_table = None

    # 常见变量中文映射，未覆盖的变量保留原名。
    param_name_map = {
        'Constant': '常数项',
        'L1_Exposure': '滞后一期敞口指数',
        'L1_Dummy_Rules': '滞后一期规则虚拟变量',
        'L1_Interaction': '滞后一期交互项',
        'log_GDP_PC': '人均GDP对数',
        'FDI': '外商直接投资(FDI)',
        'Internet_Use': '互联网使用率',
        'L1_Centrality': '滞后一期网络中心度',
    }

    rows = []
    for param in params.index:
        coef = params[param]

        se = np.nan
        for attr in ('std_errors', 'std_err', 'bse'):
            try:
                se = getattr(res, attr)[param]
                break
            except Exception:
                se = np.nan

        tstat = np.nan
        for attr in ('tstats', 'tvalues'):
            try:
                tstat = getattr(res, attr)[param]
                break
            except Exception:
                tstat = np.nan

        pval = np.nan
        for attr in ('pvalues', 'pvals'):
            try:
                pval = getattr(res, attr)[param]
                break
            except Exception:
                pval = np.nan

        ci_lower, ci_upper = np.nan, np.nan
        if ci_table is not None and param in ci_table.index:
            try:
                # 优先使用列名定位（不同版本列名可能不同），失败后回退到位置索引。
                lower_col = next((c for c in ci_table.columns if str(c).lower() in ('lower', 'lower ci', 'ci_lower', '2.5%', 'lower_ci')), None)
                upper_col = next((c for c in ci_table.columns if str(c).lower() in ('upper', 'upper ci', 'ci_upper', '97.5%', 'upper_ci')), None)

                if lower_col is not None and upper_col is not None:
                    ci_lower = ci_table.loc[param, lower_col]
                    ci_upper = ci_table.loc[param, upper_col]
                else:
                    ci_lower = ci_table.loc[param].iloc[0]
                    ci_upper = ci_table.loc[param].iloc[1]
            except Exception:
                ci_lower, ci_upper = np.nan, np.nan

        rows.append({
            '变量': param_name_map.get(param, param),
            '系数': coef,
            '标准误': se,
            't值': tstat,
            'p值': pval,
            '置信区间下界': ci_lower,
            '置信区间上界': ci_upper,
        })

    df_coef = pd.DataFrame(rows).set_index('变量')
    df_coef.index.name = '变量'
    meta = {
        '样本量': getattr(res, 'nobs', np.nan),
        'R平方': getattr(res, 'rsquared', np.nan),
        '参数个数': len(params),
    }
    return df_coef, meta


def export_all_results_to_excel(path):
    writer = pd.ExcelWriter(path, engine='openpyxl')
    try:
        models = [
            ("全样本无控制基准", res_m1),
            ("全样本基准", res_base),
            ("全样本交互项", results_inter),
            ("边缘组", res_p),
            ("核心组", res_c),
        ]

        for name, res in models:
            df_coef, meta = results_to_dataframe(res)
            if df_coef is None:
                continue
            sheet_name = name[:31]
            df_coef.to_excel(writer, sheet_name=sheet_name)
            meta_df = pd.DataFrame.from_dict(meta, orient='index', columns=['数值'])
            meta_df.index.name = '指标'
            meta_sheet_name = (sheet_name + '_元信息')[:31]
            meta_df.to_excel(writer, sheet_name=meta_sheet_name)

        robustness_df.to_excel(writer, sheet_name='稳健性检验汇总', index=False)
        # 对于 openpyxl 的 ExcelWriter，不需要显式调用 save()
    finally:
        writer.close()


def write_validation_report(path):
    # 文档中的目标值（来自已提取表格）
    targets = {
        '全样本_交互项': {
            'Constant': -1.08122,
            'L1_Exposure': 0.10477,
            'Interaction': -0.11938,
            'log_GDP_PC': -0.05217,
            'FDI': -0.00017,
            'nobs': 3339,
        },
        '边缘组': {
            'L1_Exposure': 0.02378,
        },
        '核心组': {
            'L1_Exposure': -0.00506,
        }
    }

    def line_for_param(model_name, res, param, target, tol=1e-3):
        val = res.params.get(param, np.nan)
        diff = val - target
        ok = abs(diff) <= tol
        return f"| {model_name} | {param} | {target:.6f} | {val:.6f} | {diff:.6f} | {'是' if ok else '否'} |"

    lines = []
    lines.append('# 正式脚本结果校对报告')
    lines.append('')
    lines.append('说明：本报告用于核验重跑结果与论文文档导出表格的一致性，并给出是否“完美符合”的结论。')
    lines.append('')
    lines.append('## 1. 关键系数逐项校对（容差=0.001）')
    lines.append('')
    lines.append('| 模型 | 指标 | 文档值 | 重跑值 | 差值(重跑-文档) | 是否在容差内 |')
    lines.append('|---|---:|---:|---:|---:|---|')

    for p in ['Constant', 'L1_Exposure', 'Interaction', 'log_GDP_PC', 'FDI']:
        lines.append(line_for_param('全样本交互项', results_inter, p, targets['全样本_交互项'][p]))
    lines.append(line_for_param('边缘组', res_p, 'L1_Exposure', targets['边缘组']['L1_Exposure']))
    lines.append(line_for_param('核心组', res_c, 'L1_Exposure', targets['核心组']['L1_Exposure']))

    lines.append('')
    lines.append('## 2. 样本量校对')
    lines.append('')
    lines.append('| 模型 | 文档样本量 | 重跑样本量 | 差值 |')
    lines.append('|---|---:|---:|---:|')
    nobs_doc = targets['全样本_交互项']['nobs']
    nobs_now = int(getattr(results_inter, 'nobs', np.nan))
    lines.append(f"| 全样本交互项 | {nobs_doc} | {nobs_now} | {nobs_now - nobs_doc} |")

    # 描述性统计校对
    desc_targets = {
        'ln_Resilience_Inv': {'mean': -1.53036, 'std': 0.39149, 'min': -2.62205, 'max': 0.48573},
        'Exposure': {'mean': 0.79027, 'std': 2.00414, 'min': 0.0, 'max': 16.57581},
        'L1_Exposure': {'mean': 0.74199, 'std': 1.92979, 'min': 0.0, 'max': 16.57581},
        'log_GDP_PC': {'mean': 8.63001, 'std': 1.44535, 'min': 5.53174, 'max': 11.79506},
        'FDI': {'mean': 8.20725, 'std': 51.20423, 'min': -391.55512, 'max': 1709.82722},
        'RQ': {'mean': -0.01784, 'std': 0.97196, 'min': -2.54773, 'max': 2.30859},
        'GE': {'mean': -0.03738, 'std': 0.97853, 'min': -2.44023, 'max': 2.46966},
        'Out_Degree_Centrality': {'mean': 0.77715, 'std': 0.19871, 'min': 0.07426, 'max': 0.99010},
    }

    lines.append('')
    lines.append('## 3. 描述性统计校对（容差=0.01）')
    lines.append('')
    lines.append('| 变量 | 指标 | 文档值 | 重跑值 | 差值 | 是否在容差内 |')
    lines.append('|---|---|---:|---:|---:|---|')

    df_desc = df_indexed_clean.copy()
    # RQ/GE在Master_Panel里，先拼回去用于统计
    if 'RQ' not in df_desc.columns or 'GE' not in df_desc.columns:
        panel_cols = ['REF_AREA', 'TIME_PERIOD', 'RQ', 'GE']
        panel_tmp = df[panel_cols].drop_duplicates().set_index(['REF_AREA', 'TIME_PERIOD'])
        df_desc = df_desc.join(panel_tmp, how='left')

    for var, metrics in desc_targets.items():
        if var not in df_desc.columns:
            for metric, target in metrics.items():
                lines.append(f"| {var} | {metric} | {target:.5f} | NA | NA | 否 |")
            continue
        series = pd.to_numeric(df_desc[var], errors='coerce').dropna()
        current_vals = {
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
        }
        for metric, target in metrics.items():
            cur = current_vals[metric]
            diff = cur - target
            ok = abs(diff) <= 0.01
            lines.append(f"| {var} | {metric} | {target:.5f} | {cur:.5f} | {diff:.5f} | {'是' if ok else '否'} |")

    # 出度排名校对
    lines.append('')
    lines.append('## 4. 出度中心度排名表校对')
    lines.append('')
    lines.append('| 国家 | 文档2023值 | 重跑2023值 | 差值 | 文档2005值 | 重跑2005值 | 差值 |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')

    rank_targets = {
        'SGP': (1.0, 0.954545),
        'IRL': (0.994949, 0.969697),
        'USA': (0.994949, 0.974747),
        'IND': (0.994949, 0.969697),
        'CHN': (0.989899, 0.949495),
        'BRA': (0.984848, 0.919192),
        'CHE': (0.984848, 0.964646),
        'JPN': (0.979798, 0.969697),
        'CAN': (0.974747, 0.954545),
    }
    node_path = os.path.join(OUTPUT_DIR, 'output_node_centrality_panel.csv')
    if os.path.exists(node_path):
        node_df = pd.read_csv(node_path)
        for c, (target_2023, target_2005) in rank_targets.items():
            v2023 = node_df.loc[(node_df['REF_AREA'] == c) & (node_df['TIME_PERIOD'] == 2023), 'Out_Degree_Centrality']
            v2005 = node_df.loc[(node_df['REF_AREA'] == c) & (node_df['TIME_PERIOD'] == 2005), 'Out_Degree_Centrality']
            cur2023 = float(v2023.iloc[0]) if not v2023.empty else np.nan
            cur2005 = float(v2005.iloc[0]) if not v2005.empty else np.nan
            d2023 = cur2023 - target_2023 if not np.isnan(cur2023) else np.nan
            d2005 = cur2005 - target_2005 if not np.isnan(cur2005) else np.nan
            lines.append(f"| {c} | {target_2023:.6f} | {cur2023:.6f} | {d2023:.6f} | {target_2005:.6f} | {cur2005:.6f} | {d2005:.6f} |")
    else:
        lines.append('| NA | NA | NA | NA | NA | NA | NA |')

    lines.append('')
    lines.append('## 5. 结论')
    perfect = True
    for p in ['Constant', 'L1_Exposure', 'Interaction', 'log_GDP_PC', 'FDI']:
        if abs(results_inter.params.get(p, np.nan) - targets['全样本_交互项'][p]) > 1e-3:
            perfect = False
            break
    if abs(res_p.params.get('L1_Exposure', np.nan) - targets['边缘组']['L1_Exposure']) > 1e-3:
        perfect = False
    if abs(res_c.params.get('L1_Exposure', np.nan) - targets['核心组']['L1_Exposure']) > 1e-3:
        perfect = False
    if nobs_now != nobs_doc:
        perfect = False

    if perfect:
        lines.append('重跑结果与文档导出表格完美符合。')
    else:
        lines.append('重跑结果与文档导出表格未达到完美一致。主要原因通常是样本口径、预处理版本、变量定义或截面年份处理存在差异。')

    lines.append('')
    lines.append('## 6. 稳健性检验汇总')
    lines.append('')
    lines.append(robustness_df.to_markdown(index=False))

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


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
    # linearmodels 中优先使用 within R²，不可用时回退到 rsquared
    for attr in ('rsquared_within', 'rsquared'):
        val = getattr(res, attr, np.nan)
        if not pd.isna(val):
            return float(val)
    return np.nan


def _single_model_table_lines(model_title, res, controls_flag, countries_count, row_spec):
    period_txt = '2005-2023'
    lines = []
    lines.append(f'### 表：{model_title}')
    lines.append('被解释变量：ln_Resilience_Inv（网络韧性反转对数指数）')
    lines.append('')
    lines.append(f'| 变量 | {model_title} |')
    lines.append('|---|---:|')

    for row_label, var in row_spec:
        coef_txt, se_txt = _coef_cell(res, var)
        lines.append(f'| {row_label} | {coef_txt} |')
        lines.append(f'|  | {se_txt} |')

    lines.append('| Country FE | Yes |')
    lines.append('| Year FE | Yes |')
    lines.append(f'| Controls | {controls_flag} |')
    lines.append('| SE Cluster | Country |')
    lines.append(f'| Observations (N) | {int(res.nobs)} |')
    lines.append(f'| Countries | {int(countries_count)} |')
    lines.append(f'| Period | {period_txt} |')
    lines.append(f'| Within R² | {_within_r2(res):.3f} |')
    lines.append('')
    lines.append('注：括号内为t统计量（基于国家层面聚类稳健协方差计算）。所有模型均包含个体固定效应与年份固定效应，常数项已被固定效应吸收，不单独报告。')
    lines.append('L1_Exposure 为数字贸易规则敞口指数滞后一期；L1_Centrality 为出度中心度滞后一期。边缘组与核心组分别为出度中心度后验分布的后33%与前33%分位数组。')
    lines.append('***p<0.01，**p<0.05，*p<0.1。')
    return lines


def export_combined_markdown_table(path_md):
    """导出四模型合并版Markdown表格（纯md）。"""
    models = [
        {'title': '(1)', 'res': res_m1, 'controls': 'No', 'countries': df_indexed_clean.index.get_level_values(0).nunique()},
        {'title': '(2)', 'res': results_inter, 'controls': 'Yes', 'countries': df_indexed_clean.index.get_level_values(0).nunique()},
        {'title': '(3)', 'res': res_p, 'controls': 'Yes', 'countries': df_peri.index.get_level_values(0).nunique()},
        {'title': '(4)', 'res': res_c, 'controls': 'Yes', 'countries': df_core.index.get_level_values(0).nunique()},
        {'title': '(5)', 'res': res_base, 'controls': 'Yes', 'countries': df_indexed_clean.index.get_level_values(0).nunique()},
    ]
    period_txt = '2005-2023'
    row_spec = [
        ('L1_Exposure', 'L1_Exposure'),
        ('L1_Exposure × L1_Centrality', 'Interaction'),
        ('L1_Centrality', 'L1_Centrality'),
        ('log_GDP_PC', 'log_GDP_PC'),
        ('FDI', 'FDI'),
    ]

    lines = []
    lines.append('# 回归结果三线表（表X-X）')
    lines.append('')
    lines.append('被解释变量：ln_Resilience_Inv（网络韧性反转对数指数）')
    lines.append('')
    lines.append('| 变量 | (1) 全样本无控制 | (2) 全样本交互项 | (3) 边缘组 Bottom 33% | (4) 核心组 Top 33% | (5) 全样本基准 |')
    lines.append('|---|---:|---:|---:|---:|---:|')

    for row_label, var in row_spec:
        coefs, ses = [], []
        for m in models:
            coef_txt, se_txt = _coef_cell(m['res'], var)
            coefs.append(coef_txt)
            ses.append(se_txt)
        lines.append(f"| {row_label} | " + " | ".join(coefs) + ' |')
        lines.append('|  | ' + ' | '.join(ses) + ' |')

    lines.append('| Country FE | Yes | Yes | Yes | Yes | Yes |')
    lines.append('| Year FE | Yes | Yes | Yes | Yes | Yes |')
    lines.append('| Controls | No | Yes | Yes | Yes | Yes |')
    lines.append('| SE Cluster | Country | Country | Country | Country | Country |')
    lines.append('| Observations (N) | ' + ' | '.join([str(int(m['res'].nobs)) for m in models]) + ' |')
    lines.append('| Countries | ' + ' | '.join([str(int(m['countries'])) for m in models]) + ' |')
    lines.append('| Period | ' + ' | '.join([period_txt for _ in models]) + ' |')
    lines.append('| Within R² | ' + ' | '.join([f"{_within_r2(m['res']):.3f}" for m in models]) + ' |')
    lines.append('')
    lines.append('注：括号内为t统计量（基于国家层面聚类稳健协方差计算）。所有模型均包含个体固定效应与年份固定效应，常数项已被固定效应吸收，不单独报告。')
    lines.append('L1_Exposure 为数字贸易规则敞口指数滞后一期；L1_Centrality 为出度中心度滞后一期。边缘组与核心组分别为出度中心度后验分布的后33%与前33%分位数组。')
    lines.append('***p<0.01，**p<0.05，*p<0.1。')

    with open(path_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_model_wise_tables(output_dir):
    """按原始脚本模型逻辑导出：每个模型一张三线表。"""
    model_defs = [
        {
            'title': '模型0：全样本无控制基准回归',
            'label': 'model0_full_nocontrol',
            'res': res_m1,
            'controls': 'No',
            'countries': df_indexed_clean.index.get_level_values(0).nunique(),
            'rows': [
                ('L1_Exposure', 'L1_Exposure'),
            ],
            'base_name': 'output_回归结果三线表_模型0_全样本无控制基准',
        },
        {
            'title': '模型1：全样本交互项模型',
            'label': 'model1_full_interaction',
            'res': results_inter,
            'controls': 'Yes',
            'countries': df_indexed_clean.index.get_level_values(0).nunique(),
            'rows': [
                ('L1_Exposure', 'L1_Exposure'),
                ('L1\\_Exposure × L1\\_Centrality', 'Interaction'),
                ('log_GDP_PC', 'log_GDP_PC'),
                ('FDI', 'FDI'),
            ],
            'base_name': 'output_回归结果三线表_模型1_全样本交互项',
        },
        {
            'title': '模型2：边缘组 Bottom 33%',
            'label': 'model2_bottom33',
            'res': res_p,
            'controls': 'Yes',
            'countries': df_peri.index.get_level_values(0).nunique(),
            'rows': [
                ('L1_Exposure', 'L1_Exposure'),
                ('log_GDP_PC', 'log_GDP_PC'),
                ('FDI', 'FDI'),
            ],
            'base_name': 'output_回归结果三线表_模型2_边缘组Bottom33',
        },
        {
            'title': '模型3：核心组 Top 33%',
            'label': 'model3_top33',
            'res': res_c,
            'controls': 'Yes',
            'countries': df_core.index.get_level_values(0).nunique(),
            'rows': [
                ('L1_Exposure', 'L1_Exposure'),
                ('log_GDP_PC', 'log_GDP_PC'),
                ('FDI', 'FDI'),
            ],
            'base_name': 'output_回归结果三线表_模型3_核心组Top33',
        },
        {
            'title': '模型4：全样本基准模型',
            'label': 'model4_full_baseline',
            'res': res_base,
            'controls': 'Yes',
            'countries': df_indexed_clean.index.get_level_values(0).nunique(),
            'rows': [
                ('L1_Exposure', 'L1_Exposure'),
                ('log_GDP_PC', 'log_GDP_PC'),
                ('FDI', 'FDI'),
            ],
            'base_name': 'output_回归结果三线表_模型4_全样本基准',
        },
    ]

    out_files = []
    for m in model_defs:
        lines = _single_model_table_lines(
            model_title=m['title'],
            res=m['res'],
            controls_flag=m['controls'],
            countries_count=m['countries'],
            row_spec=m['rows'],
        )
        path_md = os.path.join(output_dir, m['base_name'] + '.md')
        path_tex = os.path.join(output_dir, m['base_name'] + '.tex')

        with open(path_md, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        tex_content = '\n'.join(lines[4:-1]).replace('```latex\n', '').replace('\n```', '')
        with open(path_tex, 'w', encoding='utf-8') as f:
            f.write(tex_content)

        out_files.append((path_md, path_tex))

    return out_files


output_path = os.path.join(OUTPUT_DIR, 'output_regression_results.xlsx')
export_all_results_to_excel(output_path)
print(f"\n已将回归结果导出到: {output_path}")

robustness_path = os.path.join(OUTPUT_DIR, 'output_robustness_results_summary.csv')
robustness_df.to_csv(robustness_path, index=False, encoding='utf-8-sig')
print(f"已将稳健性汇总导出到: {robustness_path}")

validation_report_path = os.path.join(OUTPUT_DIR, 'output_结果校对报告.md')
write_validation_report(validation_report_path)
print(f"已将校对报告导出到: {validation_report_path}")

model_table_files = export_model_wise_tables(OUTPUT_DIR)
for md_path, tex_path in model_table_files:
    print(f"已将模型三线表导出到: {md_path}")
    print(f"已将模型三线表(LaTeX)导出到: {tex_path}")

combined_md = os.path.join(OUTPUT_DIR, 'output_回归结果三线表.md')
export_combined_markdown_table(combined_md)
print(f"已将合并版Markdown三线表导出到: {combined_md}")

# ============================================================================
# 第六部分：导出 Stata .dta 文件
# ============================================================================
print("\n" + "=" * 90)
print("【Stata 数据文件导出】")
STATA_OUT_DIR = os.path.join(OUTPUT_DIR, 'Stata_Export')
os.makedirs(STATA_OUT_DIR, exist_ok=True)

# 1. 准备主模型与稳健性检验所需的宽表数据
stata_df = df_indexed_clean.copy()

# 重置多级索引使其扁平化
stata_df = stata_df.reset_index()

# 重命名以符合一般习惯
stata_df.rename(columns={'REF_AREA': 'Country', 'TIME_PERIOD': 'Year'}, inplace=True)

# 处理潜在的缺失列（例如 L1_Internet_Use 可能全为空），以及缩尾变量 L1_Exposure_W
if 'L1_Internet_Use' not in stata_df.columns:
    stata_df['L1_Internet_Use'] = np.nan

# 补充稳健性检验1 的缩尾变量
q1 = stata_df['L1_Exposure'].quantile(0.01)
q99 = stata_df['L1_Exposure'].quantile(0.99)
stata_df['L1_Exposure_W'] = stata_df['L1_Exposure'].clip(lower=q1, upper=q99)
stata_df['Interaction_W'] = stata_df['L1_Exposure_W'] * stata_df['L1_Centrality']

# 选择我们需要导出的列
cols_to_export = [
    'Country', 'Year', 'True_Resilience', 'ln_Resilience_Inv', 
    'Exposure', 'L1_Exposure', 'L1_Centrality',  'log_GDP_PC', 'FDI', 
    'Interaction', 'L1_Internet_Use', 'L1_Exposure_W', 'Interaction_W'
]

# 保留存在的列
final_stata_df = stata_df[[c for c in cols_to_export if c in stata_df.columns]]

# 2. 导出为 .dta
stata_out_file = os.path.join(STATA_OUT_DIR, "GeSonghao_Thesis_Data.dta")
try:
    # version=118 兼容现代 Stata
    final_stata_df.to_stata(stata_out_file, version=118, write_index=False)
    print(f"✅ Stata .dta 文件成功导出至: {stata_out_file}")
    print(f"导出变量: {list(final_stata_df.columns)}")
    print(f"导出记录数: {len(final_stata_df)}")
except Exception as e:
    print(f"❌ Stata 导出失败: {e}")

print("=" * 90)




# ============================================================================
# 第六部分：导出 Stata .dta 文件
# ============================================================================
print("\n" + "=" * 90)
print("【Stata 数据文件导出】")
STATA_OUT_DIR = os.path.join(OUTPUT_DIR, 'Stata_Export')
os.makedirs(STATA_OUT_DIR, exist_ok=True)

# 1. 准备主模型与稳健性检验所需的宽表数据
stata_df = df_indexed_clean.copy()

# 重置多级索引使其扁平化
stata_df = stata_df.reset_index()

# 重命名以符合一般习惯
stata_df.rename(columns={'REF_AREA': 'Country', 'TIME_PERIOD': 'Year'}, inplace=True)

# 处理潜在的缺失列（例如 L1_Internet_Use 可能全为空），以及缩尾变量 L1_Exposure_W
if 'L1_Internet_Use' not in stata_df.columns:
    stata_df['L1_Internet_Use'] = np.nan

# 补充稳健性检验1 的缩尾变量
q1 = stata_df['L1_Exposure'].quantile(0.01)
q99 = stata_df['L1_Exposure'].quantile(0.99)
stata_df['L1_Exposure_W'] = stata_df['L1_Exposure'].clip(lower=q1, upper=q99)
stata_df['Interaction_W'] = stata_df['L1_Exposure_W'] * stata_df['L1_Centrality']

# 选择我们需要导出的列
cols_to_export = [
    'Country', 'Year', 'True_Resilience', 'ln_Resilience_Inv', 
    'Exposure', 'L1_Exposure', 'L1_Centrality',  'log_GDP_PC', 'FDI', 
    'Interaction', 'L1_Internet_Use', 'L1_Exposure_W', 'Interaction_W'
]

# 保留存在的列
final_stata_df = stata_df[[c for c in cols_to_export if c in stata_df.columns]]

# 2. 导出为 .dta
stata_out_file = os.path.join(STATA_OUT_DIR, "GeSonghao_Thesis_Data.dta")
try:
    # version=118 兼容现代 Stata
    final_stata_df.to_stata(stata_out_file, version=118, write_index=False)
    print(f"✅ Stata .dta 文件成功导出至: {stata_out_file}")
    print(f"导出变量: {list(final_stata_df.columns)}")
    print(f"导出记录数: {len(final_stata_df)}")
except Exception as e:
    print(f"❌ Stata 导出失败: {e}")

print("=" * 90)