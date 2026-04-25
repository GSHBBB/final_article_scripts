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

base_dir = r"C:\Users\29106\OneDrive\文档\毕业论文"
data_file = os.path.join(base_dir, "Master_Panel.csv")
resilience_file = os.path.join(base_dir, "DATE_from_WB", "final_resilience_panel.csv")

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
else:
    print("未检测到 GE 或 RQ 列，请检查 Master_Panel.csv 中是否成功包含了这些变量。")

print("\n" + "=" * 90)
print("恭喜！这篇论文所有的实证代码运行完毕，请直接复制结果填入论文表格中。")
print("=" * 90)

# ============================================================================
# 导出回归结果表与关键系数汇总表
# ============================================================================
try:
    out_dir = base_dir
    excel_path = os.path.join(out_dir, "mechanism_regression_results.xlsx")
    summary_csv = os.path.join(out_dir, "regression_summary.csv")

    result_list = [
        ("full_sample", results_inter),
        ("peri", res_p),
        ("core", res_c),
        ("GE", res_ge_final) if 'res_ge_final' in globals() else ("GE", None),
        ("RQ", res_rq_final) if 'res_rq_final' in globals() else ("RQ", None),
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

            # 抽取关键系数（优先 L1_Exposure，再考虑 Interaction）
            for var in ['L1_Exposure', 'Interaction']:
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
except Exception as e:
    print("导出回归结果时发生错误:", e)