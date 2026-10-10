# ESWA 2025：VMD–DCNN-attention–BiGRU 智能投顾

对应论文：Zhu, Q., Han, C., Liu, S., Li, Y., Che, J. (2025). Artificial intelligence-driven financial innovation: A robo-advisor system for robust returns across diversified markets. *Expert Systems with Applications*, 274, 126881. https://doi.org/10.1016/j.eswa.2025.126881

本目录对应正文第 3–5 节和表 1–表 4：六个资产的 VMD、去掉 IMF1 后的 10×20 矩阵、空间注意力 DCNN、BiGRU、Algorithm 1 回测、描述统计，以及 ARIMA 与 Markov switching 对照。资产为 DXU4、CLU4、SSE50ETF、GCM4、SSEC、NDX。

## 与论文表述需要同时阅读的实验细节

- 特征提取在全样本上训练，再把 200 维特征交给 BiGRU 做滚动切分。这是存档笔记本的顺序。
- DCNN 在交叉熵低于 0.01 的前一步导出特征。BiGRU 在交叉熵低于 0.05 时停止。BiGRU 图中没有接入 dropout；论文写的 dropout 为 0.3。
- 卷积后只使用空间注意力。笔记本里另有通道注意力函数，但没有进入实际计算图，因此不包含在这里。
- 卖出手续费的 5 个货币单位下限发生在盈利判断之后。论文 Algorithm 1 把下限放在判断之前。
- 原油文件在本地实验中名为 CLQ4，论文合约名为 CLU4。上证 50ETF 本地文件名为 sz50etf。配置同时接受这两个文件名。
- 各资产训练比例来自对应 BiGRU 笔记本：DXU4 0.8878，CLU4 0.889，SSE50ETF 0.888，GCM4 0.8872，SSEC 与 NDX 0.8881。滚动步长为 90。
- 表 1 的峰度使用 SciPy 的超额峰度。BDS 取 `max_dim=5` 的第一维统计量，与存档统计脚本一致。

## 数据列

指数文件包含 `date`、`open`、`high`、`low`、`close`、`volume` 以及 15 个技术指标，共 20 个数值列。基线脚本仍按存档方式读取第 5 列作为收盘价。Wind 原始数据不随仓库分发。

## 运行顺序

标签、VMD、矩阵重构、回测和统计可在当前 Python 环境运行。DCNN 与 BiGRU 需要 Python 3.6/3.7 和 TensorFlow 1.15。

```text
python -m eswa2025_robo_advisor.baselines.arima_baseline --data-dir data/eswa
python -m eswa2025_robo_advisor.baselines.markov_switching_baseline --data-dir data/eswa
```

神经网络阶段在对应模块中调用 `decompose_file`、`save_images`、`extract_features`、`rolling_predict` 和 `simulate_trading`。随机种子 111、222、369、149、800 的资金曲线按图 10 的宽表格式交给 `random_seed_metrics.metrics_from_wide_table`。
