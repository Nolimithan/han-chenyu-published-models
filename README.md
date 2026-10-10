# 已发表论文的实验代码

## 论文

1. Zhu, Q., Han, C., Li, Y. (2025). Dual-market quantitative trading: the dynamics of liquidity and turnover in financial markets. *Data Science and Management*, 8, 48–58. https://doi.org/10.1016/j.dsm.2024.07.003  
   代码目录：`dsm2025_dual_market`

2. Zhu, Q., Han, C., Liu, S., Li, Y., Che, J. (2025). Artificial intelligence-driven financial innovation: A robo-advisor system for robust returns across diversified markets. *Expert Systems with Applications*, 274, 126881. https://doi.org/10.1016/j.eswa.2025.126881  
   代码目录：`eswa2025_robo_advisor`

引用格式见 `CITATION.cff`。代码以 MIT 许可发布。论文文本和 Wind 行情数据不在本仓库中。

## 代码与论文的对应关系

| 论文内容 | 保留的实现 | 原笔记本或脚本 |
| --- | --- | --- |
| DSM：K=11 的 VMD，输入宽度 231 | `dsm2025_dual_market/vmd_preprocess.py` | `SSECt0数据预处理+VMD.ipynb`、`IXICT0数据预处理+VMD.ipynb`、`DJI数据预处理+VMD.ipynb`；SZI 的分解写在名为 `NDX数据预处理+VMD.ipynb` 的文件中 |
| DSM：开盘价与收盘价的二分类，交易成本筛除 | `labeling.py`，成本因子 0.9955 | `label.ipynb` 中保存 IXIC T+0 标签的单元 |
| DSM：序列长度 3、训练比例 0.65、周期 120、损失阈值 0.05 | `train_lstm.py` | `SSECt0训练模型.ipynb`、`IXICT0训练模型.ipynb`、`DJI训练模型.ipynb`、`NDX训练模型.ipynb`（数据为 SZI） |
| DSM：单市场 T+0，买入/卖出 2‰，印花税 0.5‰ | `single_market.py` | 上述训练笔记本的回测单元。初始资金 SSEC 210000 CNY，IXIC 30000 USD，SZI 350000 CNY，DJI 50000 USD |
| DSM：SSEC–IXIC 与 SZI–DJI，经 CNY/CNH/USD | `dual_market.py` | `两市场量化策略.ipynb` |
| ESWA：20 列指标的 VMD | `eswa2025_robo_advisor/vmd_preprocess.py` | 六个资产的 `Data preprocessing & vmd` 笔记本 |
| ESWA：去掉 IMF1，构成 10×20 矩阵 | `reconstruct_2d.py` | `DXU4 1d to 2d.ipynb` 等实际被 CNN 读取的 10×20 版本 |
| ESWA：两层卷积、7×7 空间注意力、池化、200 维特征 | `dcnn_spatial_attention.py` | `GCM4 cnn.ipynb`、`DXU4 cnn.ipynb`、`SSEC cnn.ipynb`、`NDX cnn.ipynb` |
| ESWA：BiGRU，隐藏层 200，序列长度 3，周期 90 | `train_bigru.py` | `DXU4 bigru.ipynb`、`CLQ4 bigru.ipynb`、`sz50etf bigru.ipynb`、`GCM4 cnn-bigru.ipynb`、`SSEC bigru.ipynb`、`NDX cnn-bigru.ipynb` |
| ESWA：Algorithm 1 | `trading.py` | 上述 BiGRU 笔记本的交易循环 |
| ESWA：表 1 | `metrics.py` | `ESWA/STATISTICS.PY` |
| ESWA：表 4 的风险指标与图 10 | `metrics.py`、`random_seed_metrics.py` | `ESWA/Risk indicators.py`、`ESWA/random seeds.py` |
| ESWA：ARIMA 与 Markov switching | `baselines/` | `ESWA/ARIMA.py`、`ESWA/Regime switching model.py` |


## 复现边界

神经网络脚本保持 TensorFlow 1.15 的计算图，不能在 TensorFlow 2 中直接运行。标签规则、矩阵形状、单市场持有和风险指标有不依赖行情数据的检查，见 `tests/test_released_logic.py`。没有原始 Wind 文件时，不能复现论文中的数值结果。各子目录的说明写明了存档脚本和论文措辞不一致的地方，公开代码按存档实验保留，并在说明中标出差异。
