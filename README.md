# LSTM-MODGWO：有向加权动态多层网络竞争影响力最大化

私有研究仓库（学位论文第三部分工程）。

主工程目录：[`LSTM_MODGWO_Project/`](LSTM_MODGWO_Project/)

## 克隆（两边同步）

```bash
git clone https://github.com/wyufeng986-commits/LSTM-MODGWO-dyn-multilayer-CIM.git
cd LSTM-MODGWO-dyn-multilayer-CIM/LSTM_MODGWO_Project
```

日常：

```bash
git pull
# ... 改代码或跑实验 ...
git add -A
git commit -m "..."
git push
```

## 异机正式实验

见 `LSTM_MODGWO_Project/docs/阶段五正式实验_异机运行说明.md`。

入口：`LSTM_MODGWO_Project/run_formal_modgwo.m`（建议先 `PRESET='medium'`）。

## 仓库内容约定

- **纳入**：代码、文档、`processed` 网络 mat、残差 LSTM 模型、关键结果
- **忽略**：`01_Data/raw/` 原始 CNS 下载（体积大，可重建）
