# ESWA 2025：VMD–DCNN-attention–BiGRU 智能投顾

对应论文：Zhu, Q., Han, C., Liu, S., Li, Y., Che, J. (2025). Artificial intelligence-driven financial innovation: A robo-advisor system for robust returns across diversified markets. *Expert Systems with Applications*, 274, 126881. https://doi.org/10.1016/j.eswa.2025.126881

本目录对应正文第 3–5 节和表 1–表 4：六个资产的 VMD、去掉 IMF1 后的 10×20 矩阵、空间注意力 DCNN、BiGRU、Algorithm 1 回测、描述统计，以及 ARIMA 与 Markov switching 对照。资产为 DXU4、CLU4、SSE50ETF、GCM4、SSEC、NDX。

## 数据列

指数文件包含 `date`、`open`、`high`、`low`、`close`、`volume` 以及 15 个技术指标，共 20 个数值列。基线脚本仍按存档方式读取第 5 列作为收盘价。Wind 原始数据不随仓库分发。

## 运行顺序

标签、VMD、矩阵重构、回测和统计可在当前 Python 环境运行。DCNN 与 BiGRU 需要 Python 3.6/3.7 和 TensorFlow 1.15。

```text
python -m eswa2025_robo_advisor.baselines.arima_baseline --data-dir data/eswa
python -m eswa2025_robo_advisor.baselines.markov_switching_baseline --data-dir data/eswa
```

神经网络阶段在对应模块中调用 `decompose_file`、`save_images`、`extract_features`、`rolling_predict` 和 `simulate_trading`。随机种子 111、222、369、149、800 的资金曲线按图 10 的宽表格式交给 `random_seed_metrics.metrics_from_wide_table`。
