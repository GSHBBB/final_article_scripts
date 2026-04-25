### 表：模型1：全样本交互项模型
被解释变量：ln_Resilience_Inv（网络韧性反转对数指数）

| 变量 | 模型1：全样本交互项模型 |
|---|---:|
| L1_Exposure | 0.1034*** |
|  | (0.0274) |
| L1\_Exposure × L1\_Centrality | -0.1175*** |
|  | (0.0329) |
| log_GDP_PC | -0.0704 |
|  | (0.0515) |
| FDI | -0.0001* |
|  | (0.0000) |
| Country FE | Yes |
| Year FE | Yes |
| Controls | Yes |
| SE Cluster | Country |
| Observations (N) | 3293 |
| Countries | 188 |
| Period | 2005-2023 |
| Within R² | 0.006 |

注：括号内为国家层面聚类稳健标准误（Clustered SE at Country Level）。所有模型均包含个体固定效应与年份固定效应，常数项已被固定效应吸收，不单独报告。
L1_Exposure 为数字贸易规则敞口指数滞后一期；L1_Centrality 为出度中心度滞后一期。边缘组与核心组分别为出度中心度后验分布的后33%与前33%分位数组。
***p<0.01，**p<0.05，*p<0.1。