# CNS 数据字典（阶段一）

## 数据来源

- 官方 Figshare：https://doi.org/10.6084/m9.figshare.7267433
- 论文：https://doi.org/10.1038/s41597-019-0325-x
- 本阶段实际下载：Netzschleuder `copenhagen` CSV zip（Figshare 在当前网络返回 403）
- Netzschleuder 保留时间戳与层语义，节点 `id` 为原始参与者编号

## 确认决策

- 主动态层：calls + sms + bluetooth
- facebook：静态 / 消融
- 时间窗：按天（86400 秒）
- 主传播模型（后续阶段）：任务书 IC

## 各层原始属性 vs 预处理后属性

| 层 | 原始方向 | 原始权重/强度 | 预处理后方向 | 日快照权重 | 角色 |
|---|---|---|---|---|---|
| calls | 有向（主叫→被叫） | duration（秒；-1/0 为未接通等） | 有向 | 当日通话次数 | 主层 |
| sms | 有向（发送→接收） | 事件本身 | 有向 | 当日短信条数 | 主层 |
| bt | 无向近距离接触 | timestamp + rssi | 双向等权弧 | 当日扫描次数 | 主层 |
| fb_friends | 无向好友 | 无时间戳 | 双向弧 | 静态（复制到各天便于对齐） | 静态/消融 |

## 字段说明（Netzschleuder CSV）

### nodes.csv
- `index`：该层局部节点下标（edges 的 source/target）
- `id`：原始参与者 ID（跨层对齐主键）
- `female`：性别标记（1/0/-1）

### edges.csv
- calls: `source,target,timestamp,duration`
- sms: `source,target,timestamp`
- bt: `source,target,timestamp,rssi`
- fb_friends: `source,target`

## 检查摘要

```json
{
  "layers": {
    "calls": {
      "path": "D:\\学术论文写作\\有向加权动态多层网络竞争影响力最大化研究\\LSTM_MODGWO_Project\\01_Data\\raw\\cns\\calls",
      "n_edges_raw": 3600,
      "n_nodes_raw": 536,
      "edge_columns": [
        "source",
        "target",
        "timestamp",
        "duration"
      ],
      "node_columns": [
        "index",
        "id",
        "female",
        "_pos"
      ],
      "original_id_min": 0,
      "original_id_max": 846,
      "timestamp_min": 184,
      "timestamp_max": 2416399,
      "span_days": 27.965451388888887,
      "unique_days": 28,
      "duration_positive": 2654,
      "duration_nonpositive": 946
    },
    "sms": {
      "path": "D:\\学术论文写作\\有向加权动态多层网络竞争影响力最大化研究\\LSTM_MODGWO_Project\\01_Data\\raw\\cns\\sms",
      "n_edges_raw": 24333,
      "n_nodes_raw": 568,
      "edge_columns": [
        "source",
        "target",
        "timestamp"
      ],
      "node_columns": [
        "index",
        "id",
        "female",
        "_pos"
      ],
      "original_id_min": 0,
      "original_id_max": 846,
      "timestamp_min": 18,
      "timestamp_max": 2418982,
      "span_days": 27.997268518518517,
      "unique_days": 28
    },
    "bt": {
      "path": "D:\\学术论文写作\\有向加权动态多层网络竞争影响力最大化研究\\LSTM_MODGWO_Project\\01_Data\\raw\\cns\\bt",
      "n_edges_raw": 2426279,
      "n_nodes_raw": 692,
      "edge_columns": [
        "source",
        "target",
        "timestamp",
        "rssi"
      ],
      "node_columns": [
        "index",
        "id",
        "female",
        "_pos"
      ],
      "original_id_min": 0,
      "original_id_max": 845,
      "timestamp_min": 0,
      "timestamp_max": 2418900,
      "span_days": 27.99652777777778,
      "unique_days": 28,
      "rssi_min": -100,
      "rssi_max": 20
    },
    "fb_friends": {
      "path": "D:\\学术论文写作\\有向加权动态多层网络竞争影响力最大化研究\\LSTM_MODGWO_Project\\01_Data\\raw\\cns\\fb_friends",
      "n_edges_raw": 6429,
      "n_nodes_raw": 800,
      "edge_columns": [
        "source",
        "target"
      ],
      "node_columns": [
        "index",
        "id",
        "female",
        "_pos"
      ],
      "original_id_min": 0,
      "original_id_max": 850
    }
  },
  "notes": [
    "Bluetooth/Facebook are originally undirected; preprocessing writes bidirectional arcs.",
    "Calls/SMS are originally directed (initiator -> recipient).",
    "Weights for dynamic layers are interaction counts per day (calls: count; missed calls kept with weight 1)."
  ]
}
```

## 输出

- `01_Data/processed/CNS_daily_multilayer.mat` → 变量 `Network`
- `Network.layer(l).A{t}` / `W{t}` / `edgeList{t}`
- 全局节点为 MATLAB 1-based；`Network.V` 存原始参与者 ID
