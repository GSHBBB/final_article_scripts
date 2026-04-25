/* ===========================================================================
Project: 数字贸易规则敞口对网络韧性的影响
Author: 葛松昊
Date: 2026-04-25
Description: 双向固定效应回归（主模型 + 机制 + 稳健性检验），基于 Python 导出的清洗后数据。
             不强制 Internet_Use，强制包含 L1_Exposure 滞后控制。
             【核心原则】：使用 reghdfe 处理 TWFE；标准误聚类在国家层面 (vce(cluster Country))。
=========================================================================== */

clear all
set more off
* 开启 estout 支持（如果没有请运行 ssc install estout）
* ssc install reghdfe
* ssc install estout

* --- 1. 工作目录与数据导入 ---
* (假设在 .dta 同一目录下执行)
use "GeSonghao_Thesis_Data.dta", clear

* 将文字 Country 转为数值编码，适配面板宣告
encode Country, gen(Country_ID)
xtset Country_ID Year

* --- 2. 基于 Python 逻辑创建分组与临时变量 ---
* 在 Python 脚本中：
* 阈值为 33% 宽分位 (q33 与 q66 基于清理后的 L1_Centrality 后验分布)
tempvar central
gen `central' = L1_Centrality
centile `central', centiles(33 66)
local q33 = r(c_1)
local q66 = r(c_2)

gen byte group_peri = (L1_Centrality <= `q33') // 边缘组
gen byte group_core = (L1_Centrality >= `q66') // 核心组

gen pre_covid = (Year != 2020 & Year != 2021)

* ============================================================================
* 第一部分：基准与机制回归（主表）
* ============================================================================
estimates clear

* 模型 0: 全样本基准无控制
* Python: PanelOLS(y, c+X, entity_effects=True, time_effects=True) cluster_entity=True
reghdfe ln_Resilience_Inv L1_Exposure, absorb(Country_ID Year) vce(cluster Country_ID)
estadd local fe_c "Yes"
estadd local fe_t "Yes"
estadd local fe_controls "No"
estimates store m0_base_nocontrol

* 模型 1: 全样本交互项 (主逻辑测试边际效应递减)
* L1_Exposure, Interaction (=L1_Exposure * L1_Centrality)
reghdfe ln_Resilience_Inv L1_Exposure Interaction log_GDP_PC FDI, absorb(Country_ID Year) vce(cluster Country_ID)
estadd local fe_c "Yes"
estadd local fe_t "Yes"
estadd local fe_controls "Yes"
estimates store m1_inter

* 模型 2: 真正的边缘组 (Bottom 33%)
reghdfe ln_Resilience_Inv L1_Exposure log_GDP_PC FDI if group_peri == 1, absorb(Country_ID Year) vce(cluster Country_ID)
estadd local fe_c "Yes"
estadd local fe_t "Yes"
estadd local fe_controls "Yes"
estimates store m2_peri

* 模型 3: 真正的核心组 (Top 33%)
reghdfe ln_Resilience_Inv L1_Exposure log_GDP_PC FDI if group_core == 1, absorb(Country_ID Year) vce(cluster Country_ID)
estadd local fe_c "Yes"
estadd local fe_t "Yes"
estadd local fe_controls "Yes"
estimates store m3_core

* 模型 4: 全样本基准控制回归
reghdfe ln_Resilience_Inv L1_Exposure log_GDP_PC FDI, absorb(Country_ID Year) vce(cluster Country_ID)
estadd local fe_c "Yes"
estadd local fe_t "Yes"
estadd local fe_controls "Yes"
estimates store m4_base_control

* 导出主回归三线表 (.rtf 即 Word 格式，和 .tex 格式)
local out_base "Main_Regression_Results"

esttab m0_base_nocontrol m1_inter m2_peri m3_core m4_base_control ///
    using "`out_base'.rtf", replace ///
    b(4) t(4) star(* 0.1 ** 0.05 *** 0.01) ///
    r2 within ///
    label nomtitle ///
    title("Table: Main Regression Results (TWFE)") ///
    mtitle("(1) No Controls" "(2) Interaction" "(3) Peri (Bot 33%)" "(4) Core (Top 33%)" "(5) Baseline") ///
    stats(fe_c fe_t fe_controls N, labels("Country FE" "Year FE" "Controls" "Observations"))

esttab m0_base_nocontrol m1_inter m2_peri m3_core m4_base_control ///
    using "`out_base'.tex", replace ///
    b(4) t(4) star(* 0.1 ** 0.05 *** 0.01) ///
    r2 within ///
    label nomtitle ///
    title("Main Regression Results (TWFE)") ///
    mtitle("(1) No Controls" "(2) Interaction" "(3) Peri (Bot 33%)" "(4) Core (Top 33%)" "(5) Baseline") ///
    stats(fe_c fe_t fe_controls N, labels("Country FE" "Year FE" "Controls" "Observations")) ///
    booktabs

* ============================================================================
* 第二部分：稳健性检验
* ============================================================================
estimates clear

* R1: L1_Exposure 缩尾 1%-99%
* Python 预先计算了 Interaction_W
reghdfe ln_Resilience_Inv L1_Exposure_W Interaction_W log_GDP_PC FDI, absorb(Country_ID Year) vce(cluster Country_ID)
estadd local fe_c "Yes"
estadd local fe_t "Yes"
estimates store r1_winsorize

* R2: 加入 Internet_Use (过滤掉无法观测的)
* L1_Internet_Use
reghdfe ln_Resilience_Inv L1_Exposure Interaction log_GDP_PC FDI L1_Internet_Use if L1_Internet_Use != ., absorb(Country_ID Year) vce(cluster Country_ID)
estadd local fe_c "Yes"
estadd local fe_t "Yes"
estimates store r2_internet

* R3: 双向聚类标准误 (Country + Year)
* 注意: reghdfe 吸收集群: vce(cluster Country_ID Year)
reghdfe ln_Resilience_Inv L1_Exposure Interaction log_GDP_PC FDI, absorb(Country_ID Year) vce(cluster Country_ID Year)
estadd local fe_c "Yes"
estadd local fe_t "Yes"
estimates store r3_twoway_cluster

* R4: 剔除 2020-2021 新冠肺炎年份
reghdfe ln_Resilience_Inv L1_Exposure Interaction log_GDP_PC FDI if pre_covid == 1, absorb(Country_ID Year) vce(cluster Country_ID)
estadd local fe_c "Yes"
estadd local fe_t "Yes"
estimates store r4_no_covid

* 导出稳健性检验三线表
local out_robust "Robustness_Results"

esttab r1_winsorize r2_internet r3_twoway_cluster r4_no_covid ///
    using "`out_robust'.rtf", replace ///
    b(4) t(4) star(* 0.1 ** 0.05 *** 0.01) ///
    r2 within ///
    label nomtitle ///
    title("Table: Robustness Checks") ///
    mtitle("(1) 1% Winsorize" "(2) Add Internet" "(3) Two-way Cluster" "(4) Excl. COVID") ///
    stats(fe_c fe_t N, labels("Country FE" "Year FE" "Observations"))

esttab r1_winsorize r2_internet r3_twoway_cluster r4_no_covid ///
    using "`out_robust'.tex", replace ///
    b(4) t(4) star(* 0.1 ** 0.05 *** 0.01) ///
    r2 within ///
    label nomtitle ///
    title("Robustness Checks") ///
    mtitle("(1) 1% Winsorize" "(2) Add Internet" "(3) Two-way Cluster" "(4) Excl. COVID") ///
    stats(fe_c fe_t N, labels("Country FE" "Year FE" "Observations")) ///
    booktabs

disp "✅ All Done. Stata do-file executed successfully."
