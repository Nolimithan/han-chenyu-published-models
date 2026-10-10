# DSM 2025：双市场 VMD 交易实验

对应论文：Zhu, Q., Han, C., Li, Y. (2025). Dual-market quantitative trading: the dynamics of liquidity and turnover in financial markets. *Data Science and Management*, 8, 48–58. https://doi.org/10.1016/j.dsm.2024.07.003

本目录只保留与该文实验设定对应的流程：21 列技术指标的 VMD（K = 11）、二分类 T+0 标签、滚动预测、单市场回测，以及经 CNY/CNH/USD 连接的双市场回测。样本为 SSEC、IXIC，稳健性样本为 SZI、DJI。

## 数据

Wind 原始行情不随代码分发。单市场文件需含 `open` 与 `close`，以及送入 VMD 的数值列。双市场对齐文件需含 `open_1`、`close_1`、`open_2`、`close_2`、`CNYCNH_open`、`CNYCNH_close`、`USDCNH_open`、`USDCNH_close`，以及 `prediction_1`、`prediction_2` 或 `observation_1`、`observation_2`。

## 运行

预测脚本需要 Python 3.6/3.7 与 TensorFlow 1.15。标签、VMD 和回测只依赖 NumPy、pandas 与 vmdpy。在仓库根目录调用：

```python
from dsm2025_dual_market.vmd_preprocess import decompose_file
from dsm2025_dual_market.labeling import binary_action_labels
from dsm2025_dual_market.train_lstm import rolling_predict, save_predictions
from dsm2025_dual_market.single_market import simulate_single_market
from dsm2025_dual_market.dual_market import load_aligned_markets, simulate_dual_market
```

`rolling_predict` 的 `learning_rate` 与 `train_ratio` 使用 `config.ASSETS` 中对应市场的值。双市场预测路径使用 `variant="prediction"`，真实标签路径使用 `variant="realized"`。
