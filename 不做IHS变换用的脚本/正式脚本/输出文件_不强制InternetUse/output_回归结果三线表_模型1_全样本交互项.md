### 表：模型1：全样本交互项模型
被解释变量：ln_Resilience_Inv（网络韧性反转对数指数）

| 变量 | 模型1：全样本交互项模型 |
|---|---:|
| L1_Exposure | 0.1048*** |
|  | (3.8497) |
| L1\_Exposure × L1\_Centrality | -0.1194*** |
|  | (-3.6635) |
| log_GDP_PC | -0.0522 |
|  | (-1.2893) |
| FDI | -0.0002*** |
|  | (-3.1496) |
| Country FE | Yes |
| Year FE | Yes |
| Controls | Yes |
| SE Cluster | Country |
| Observations (N) | 3551 |
| Countries | 189 |
| Period | 2005-2023 |
| Within R² | 0.007 |

注：括号内为t统计量（基于国家层面聚类稳健协方差计算）。所有模型均包含个体固定效应与年份固定效应，常数项已被固定效应吸收，不单独报告。
L1_Exposure 为数字贸易规则敞口指数滞后一期；L1_Centrality 为出度中心度滞后一期。边缘组与核心组分别为出度中心度后验分布的后33%与前33%分位数组。
***p<0.01，**p<0.05，*p<0.1。