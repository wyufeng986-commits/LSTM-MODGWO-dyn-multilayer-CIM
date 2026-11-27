#!/usr/bin/env python3
"""Stage-1 CNS preprocessor: daily directed weighted dynamic multilayer Network.

Source: Netzschleuder Copenhagen CSV zips (derived from official Figshare CNS).
Official: https://doi.org/10.6084/m9.figshare.7267433
Paper:    https://doi.org/10.1038/s41597-019-0325-x

Confirmed decisions:
- Main dynamic layers: calls, sms, bluetooth
- Facebook: static / ablation
- Time window: 1 day (86400 s)
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import sparse
    from scipy.io import savemat
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required: pip install scipy") from exc


DAY_SECONDS = 86400
PROJECT_DATA = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DATA / "raw" / "cns"
PROC_DIR = PROJECT_DATA / "processed"
DOCS_DIR = PROJECT_DATA.parent / "docs"


@dataclass
class LayerSpec:
    name: str
    directed_raw: bool
    dynamic: bool
    weight_mode: str  # count | duration_sum | scan_count
    role: str         # main | static_ablation


LAYER_SPECS = {
    "calls": LayerSpec("calls", True, True, "count", "main"),
    "sms": LayerSpec("sms", True, True, "count", "main"),
    "bt": LayerSpec("bt", False, True, "scan_count", "main"),
    "fb_friends": LayerSpec("fb_friends", False, False, "count", "static_ablation"),
}


def _read_netz_csv(path: Path) -> pd.DataFrame:
    """Read Netzschleuder CSV whose header line starts with '# '."""
    return pd.read_csv(path, comment=None, skipinitialspace=True).rename(
        columns=lambda c: str(c).lstrip("# ").strip()
    )


def ensure_extracted(layer: str) -> Path:
    layer_dir = RAW_DIR / layer
    edges = layer_dir / "edges.csv"
    if edges.is_file():
        return layer_dir
    zpath = RAW_DIR / f"{layer}.csv.zip"
    if not zpath.is_file():
        raise FileNotFoundError(f"Missing {zpath}; download first.")
    layer_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(layer_dir)
    if not edges.is_file():
        # some zips nest one folder
        nested = list(layer_dir.rglob("edges.csv"))
        if not nested:
            raise FileNotFoundError(f"edges.csv not found after unzip: {layer}")
        return nested[0].parent
    return layer_dir


def load_layer_tables(layer: str) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    layer_dir = ensure_extracted(layer)
    edges = _read_netz_csv(layer_dir / "edges.csv")
    nodes = _read_netz_csv(layer_dir / "nodes.csv")
    # normalize column names
    edges.columns = [c.strip() for c in edges.columns]
    nodes.columns = [c.strip() for c in nodes.columns]
    return edges, nodes, layer_dir


def build_global_id_map(available_layers: Iterable[str]) -> Tuple[Dict[int, int], np.ndarray]:
    """Map original participant id -> 1..N (MATLAB-friendly)."""
    ids = set()
    for layer in available_layers:
        _, nodes, _ = load_layer_tables(layer)
        ids.update(int(x) for x in nodes["id"].tolist())
    sorted_ids = np.array(sorted(ids), dtype=np.int32)
    mapping = {int(pid): i + 1 for i, pid in enumerate(sorted_ids)}
    return mapping, sorted_ids


def edges_to_global(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    id_map: Dict[int, int],
    directed_raw: bool,
) -> pd.DataFrame:
    """Convert layer-local indices to global MATLAB node ids."""
    index_to_pid = {int(r["index"]): int(r["id"]) for _, r in nodes.iterrows()}
    # Netzschleuder nodes use '# index' values matching source/target
    if "index" not in nodes.columns:
        # fallback: first column
        idx_col = nodes.columns[0]
        index_to_pid = {int(r[idx_col]): int(r["id"]) for _, r in nodes.iterrows()}

    src_local = edges["source"].astype(int).to_numpy()
    tgt_local = edges["target"].astype(int).to_numpy()
    src_pid = np.array([index_to_pid[i] for i in src_local], dtype=np.int32)
    tgt_pid = np.array([index_to_pid[i] for i in tgt_local], dtype=np.int32)
    src_g = np.array([id_map[i] for i in src_pid], dtype=np.int32)
    tgt_g = np.array([id_map[i] for i in tgt_pid], dtype=np.int32)

    out = pd.DataFrame({"src": src_g, "tgt": tgt_g})
    if "timestamp" in edges.columns:
        out["timestamp"] = edges["timestamp"].astype(np.int64).to_numpy()
    if "duration" in edges.columns:
        out["duration"] = edges["duration"].astype(np.int64).to_numpy()
    if "rssi" in edges.columns:
        out["rssi"] = edges["rssi"].astype(np.int64).to_numpy()

    # drop self-loops
    out = out[out["src"] != out["tgt"]].copy()

    if not directed_raw:
        # undirected original -> store both directions later when aggregating
        out["_undirected"] = True
    else:
        out["_undirected"] = False
    return out


def aggregate_day_weights(
    events: pd.DataFrame,
    n_nodes: int,
    undirected: bool,
    weight_mode: str,
) -> Tuple[List[sparse.csr_matrix], List[sparse.csr_matrix], List[np.ndarray], Dict]:
    """Build per-day A/W and edge lists (vectorized)."""
    events = events.copy()
    if "timestamp" not in events.columns:
        events["day"] = np.zeros(len(events), dtype=np.int32)
    else:
        events["day"] = (events["timestamp"] // DAY_SECONDS).astype(np.int32)
    day_ids = sorted(events["day"].unique().tolist())

    if weight_mode == "duration_sum" and "duration" in events.columns:
        events["w"] = np.where(events["duration"].to_numpy() > 0, events["duration"].to_numpy(), 0.0)
    else:
        events["w"] = 1.0

    A_list: List[sparse.csr_matrix] = []
    W_list: List[sparse.csr_matrix] = []
    edge_lists: List[np.ndarray] = []

    for d in day_ids:
        sub = events[events["day"] == d]
        src0 = sub["src"].to_numpy(dtype=np.int32)
        tgt0 = sub["tgt"].to_numpy(dtype=np.int32)
        w0 = sub["w"].to_numpy(dtype=np.float64)
        if undirected and src0.size:
            src = np.concatenate([src0, tgt0])
            tgt = np.concatenate([tgt0, src0])
            w = np.concatenate([w0, w0])
        else:
            src, tgt, w = src0, tgt0, w0

        if src.size == 0:
            A = sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float64)
            W = sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float64)
            el = np.zeros((0, 3), dtype=np.float64)
        else:
            W = sparse.coo_matrix(
                (w, (src - 1, tgt - 1)), shape=(n_nodes, n_nodes), dtype=np.float64
            ).tocsr()
            W.sum_duplicates()
            A = (W > 0).astype(np.float64)
            coo = W.tocoo()
            el = np.column_stack([coo.row + 1, coo.col + 1, coo.data]).astype(np.float64)

        A_list.append(A)
        W_list.append(W)
        edge_lists.append(el)

    stats = {"days": day_ids, "numSnapshots": len(day_ids)}
    return A_list, W_list, edge_lists, stats


def inspect_report(available: List[str]) -> dict:
    report = {"layers": {}, "notes": []}
    for layer in available:
        edges, nodes, layer_dir = load_layer_tables(layer)
        info = {
            "path": str(layer_dir),
            "n_edges_raw": int(len(edges)),
            "n_nodes_raw": int(len(nodes)),
            "edge_columns": list(edges.columns),
            "node_columns": list(nodes.columns),
            "original_id_min": int(nodes["id"].min()),
            "original_id_max": int(nodes["id"].max()),
        }
        if "timestamp" in edges.columns:
            tmin, tmax = int(edges["timestamp"].min()), int(edges["timestamp"].max())
            info["timestamp_min"] = tmin
            info["timestamp_max"] = tmax
            info["span_days"] = (tmax - tmin) / DAY_SECONDS
            info["unique_days"] = int((edges["timestamp"] // DAY_SECONDS).nunique())
        if "duration" in edges.columns:
            info["duration_positive"] = int((edges["duration"] > 0).sum())
            info["duration_nonpositive"] = int((edges["duration"] <= 0).sum())
        if "rssi" in edges.columns:
            info["rssi_min"] = int(edges["rssi"].min())
            info["rssi_max"] = int(edges["rssi"].max())
        report["layers"][layer] = info

    report["notes"].append(
        "Bluetooth/Facebook are originally undirected; preprocessing writes bidirectional arcs."
    )
    report["notes"].append(
        "Calls/SMS are originally directed (initiator -> recipient)."
    )
    report["notes"].append(
        "Weights for dynamic layers are interaction counts per day (calls: count; missed calls kept with weight 1)."
    )
    return report


def build_network(available_layers: Optional[List[str]] = None) -> dict:
    if available_layers is None:
        available_layers = []
        for name in LAYER_SPECS:
            if (RAW_DIR / name / "edges.csv").is_file() or (RAW_DIR / f"{name}.csv.zip").is_file():
                available_layers.append(name)
    if not available_layers:
        raise RuntimeError(f"No CNS layers found under {RAW_DIR}")

    missing_main = [x for x in ("calls", "sms", "bt") if x not in available_layers]
    id_map, original_ids = build_global_id_map(available_layers)
    n_nodes = len(original_ids)

    # Determine global day axis from dynamic layers that have timestamps
    all_days = set()
    layer_events: Dict[str, pd.DataFrame] = {}
    for layer in available_layers:
        spec = LAYER_SPECS[layer]
        edges, nodes, _ = load_layer_tables(layer)
        ev = edges_to_global(edges, nodes, id_map, directed_raw=spec.directed_raw)
        layer_events[layer] = ev
        if spec.dynamic and "timestamp" in ev.columns:
            all_days.update((ev["timestamp"] // DAY_SECONDS).astype(int).tolist())

    if all_days:
        day_ids = sorted(all_days)
    else:
        day_ids = [0]

    # Build layer structs aligned on the same day axis for dynamic layers
    layer_structs = []
    layer_names = []
    coverage = {}

    for layer in available_layers:
        spec = LAYER_SPECS[layer]
        ev = layer_events[layer]
        undirected = not spec.directed_raw

        if spec.dynamic:
            # reindex onto global day axis (empty snapshots allowed)
            A_cells = []
            W_cells = []
            E_cells = []
            ev = ev.copy()
            ev["day"] = (ev["timestamp"] // DAY_SECONDS).astype(int)
            for d in day_ids:
                sub = ev[ev["day"] == d]
                # reuse aggregator on single-day frame without timestamp split
                tmp = sub.copy()
                if "timestamp" in tmp.columns:
                    # force single bucket
                    tmp["timestamp"] = 0
                A_list, W_list, edge_lists, _ = aggregate_day_weights(
                    tmp, n_nodes, undirected=undirected, weight_mode=spec.weight_mode
                )
                A_cells.append(A_list[0])
                W_cells.append(W_list[0])
                E_cells.append(edge_lists[0])
            num_snap = len(day_ids)
        else:
            # static: replicate same adjacency across all dynamic days for convenience,
            # and also keep a dedicated static copy.
            A_list, W_list, edge_lists, _ = aggregate_day_weights(
                ev.assign(timestamp=0) if "timestamp" not in ev.columns else ev,
                n_nodes,
                undirected=undirected,
                weight_mode=spec.weight_mode,
            )
            A_static, W_static, E_static = A_list[0], W_list[0], edge_lists[0]
            A_cells = [A_static for _ in day_ids]
            W_cells = [W_static for _ in day_ids]
            E_cells = [E_static for _ in day_ids]
            num_snap = len(day_ids)

        active_nodes = set()
        for E in E_cells:
            if E.size:
                active_nodes.update(E[:, 0].astype(int).tolist())
                active_nodes.update(E[:, 1].astype(int).tolist())
        coverage[layer] = {
            "n_active_nodes": len(active_nodes),
            "n_edges_total": int(sum(E.shape[0] for E in E_cells)),
            "n_snapshots_nonempty": int(sum(1 for E in E_cells if E.shape[0] > 0)),
            "directed_raw": spec.directed_raw,
            "dynamic": spec.dynamic,
            "role": spec.role,
            "weight_mode": spec.weight_mode,
        }

        layer_structs.append(
            {
                "name": spec.name,
                "directed_raw": np.bool_(spec.directed_raw),
                "directed_processed": np.bool_(True),
                "dynamic": np.bool_(spec.dynamic),
                "role": spec.role,
                "weight_mode": spec.weight_mode,
                "A": np.array(A_cells, dtype=object),
                "W": np.array(W_cells, dtype=object),
                "edgeList": np.array(E_cells, dtype=object),
                "numSnapshots": np.int32(num_snap),
            }
        )
        layer_names.append(spec.name)

    # Inter-layer coupling: identity-style physical-node sharing proxy (to be confirmed in Stage 2)
    L = len(layer_structs)
    C = np.eye(L, dtype=np.float64)

    network = {
        "V": original_ids.astype(np.float64).reshape(-1, 1),
        "numNodes": np.float64(n_nodes),
        "numLayers": np.float64(L),
        "numSnapshots": np.float64(len(day_ids)),
        "layerNames": np.array(layer_names, dtype=object),
        "layer": np.array(layer_structs, dtype=object),
        "C": C,
        "time": {
            "unit": "day",
            "secondsPerWindow": np.float64(DAY_SECONDS),
            "dayIndex": np.array(day_ids, dtype=np.float64).reshape(-1, 1),
            "t0": np.float64(0),
            "t1": np.float64((max(day_ids) + 1) * DAY_SECONDS if day_ids else DAY_SECONDS),
        },
        "metadata": {
            "dataset": "Copenhagen Networks Study",
            "source_primary": "Netzschleuder copenhagen CSV",
            "source_official": "https://doi.org/10.6084/m9.figshare.7267433",
            "paper": "https://doi.org/10.1038/s41597-019-0325-x",
            "missing_main_layers": missing_main,
            "coverage": coverage,
            "nodeIdConvention": "MATLAB 1-based global ids; V stores original participant ids",
            "facebook_role": "static_ablation",
            "bluetooth_note": "Originally undirected proximity; converted to bidirectional equal-weight arcs",
        },
    }
    return network, available_layers, missing_main, coverage


def save_outputs(network: dict, available: List[str], coverage: dict, report: dict) -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    mat_path = PROC_DIR / "CNS_daily_multilayer.mat"
    # Wrap as Network struct for MATLAB
    savemat(mat_path, {"Network": network}, do_compression=True)

    report_path = PROC_DIR / "inspect_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    coverage_path = PROC_DIR / "layer_coverage.json"
    coverage_path.write_text(
        json.dumps({"available": available, "coverage": coverage}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return mat_path


def write_data_dictionary(report: dict) -> Path:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / "CNS_数据字典.md"
    lines = [
        "# CNS 数据字典（阶段一）",
        "",
        "## 数据来源",
        "",
        "- 官方 Figshare：https://doi.org/10.6084/m9.figshare.7267433",
        "- 论文：https://doi.org/10.1038/s41597-019-0325-x",
        "- 本阶段实际下载：Netzschleuder `copenhagen` CSV zip（Figshare 在当前网络返回 403）",
        "- Netzschleuder 保留时间戳与层语义，节点 `id` 为原始参与者编号",
        "",
        "## 确认决策",
        "",
        "- 主动态层：calls + sms + bluetooth",
        "- facebook：静态 / 消融",
        "- 时间窗：按天（86400 秒）",
        "- 主传播模型（后续阶段）：任务书 IC",
        "",
        "## 各层原始属性 vs 预处理后属性",
        "",
        "| 层 | 原始方向 | 原始权重/强度 | 预处理后方向 | 日快照权重 | 角色 |",
        "|---|---|---|---|---|---|",
        "| calls | 有向（主叫→被叫） | duration（秒；-1/0 为未接通等） | 有向 | 当日通话次数 | 主层 |",
        "| sms | 有向（发送→接收） | 事件本身 | 有向 | 当日短信条数 | 主层 |",
        "| bt | 无向近距离接触 | timestamp + rssi | 双向等权弧 | 当日扫描次数 | 主层 |",
        "| fb_friends | 无向好友 | 无时间戳 | 双向弧 | 静态（复制到各天便于对齐） | 静态/消融 |",
        "",
        "## 字段说明（Netzschleuder CSV）",
        "",
        "### nodes.csv",
        "- `index`：该层局部节点下标（edges 的 source/target）",
        "- `id`：原始参与者 ID（跨层对齐主键）",
        "- `female`：性别标记（1/0/-1）",
        "",
        "### edges.csv",
        "- calls: `source,target,timestamp,duration`",
        "- sms: `source,target,timestamp`",
        "- bt: `source,target,timestamp,rssi`",
        "- fb_friends: `source,target`",
        "",
        "## 检查摘要",
        "",
        "```json",
        json.dumps(report, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 输出",
        "",
        "- `01_Data/processed/CNS_daily_multilayer.mat` → 变量 `Network`",
        "- `Network.layer(l).A{t}` / `W{t}` / `edgeList{t}`",
        "- 全局节点为 MATLAB 1-based；`Network.V` 存原始参与者 ID",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def analyze_and_plot(network: dict, out_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    n = int(network["numNodes"])
    T = int(network["numSnapshots"])
    layers = list(network["layer"])
    summary = {"numNodes": n, "numSnapshots": T, "layers": {}}

    # edges per day per layer
    plt.figure(figsize=(10, 5))
    for layer in layers:
        name = str(layer["name"])
        counts = [int(np.asarray(E).shape[0]) for E in layer["edgeList"]]
        dens = [c / max(n * (n - 1), 1) for c in counts]
        summary["layers"][name] = {
            "edges_per_day": counts,
            "mean_edges": float(np.mean(counts)) if counts else 0.0,
            "max_edges": int(np.max(counts)) if counts else 0,
            "mean_density": float(np.mean(dens)) if dens else 0.0,
            "role": str(layer["role"]),
        }
        if layer["dynamic"]:
            plt.plot(range(1, T + 1), counts, marker="o", label=name)
    plt.xlabel("Day index")
    plt.ylabel("Directed arcs in snapshot")
    plt.title("CNS daily edge counts (dynamic layers)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "cns_daily_edge_counts.png", dpi=150)
    plt.close()

    # weight distribution on a mid dynamic day for available main layers
    mid = T // 2 if T else 0
    for layer in layers:
        name = str(layer["name"])
        W = layer["W"][mid]
        if sparse.issparse(W):
            data = W.data
        else:
            data = np.asarray(W).ravel()
            data = data[data > 0]
        if data.size == 0:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(np.log1p(data), bins=40, color="#3c6e71")
        plt.xlabel("log1p(weight)")
        plt.ylabel("frequency")
        plt.title(f"{name} weight distribution (day {mid + 1})")
        plt.tight_layout()
        plt.savefig(fig_dir / f"cns_weight_hist_{name}_day{mid + 1}.png", dpi=150)
        plt.close()

    summary_path = out_dir / "network_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    print(f"RAW_DIR = {RAW_DIR}")
    available = []
    for name in LAYER_SPECS:
        if (RAW_DIR / name / "edges.csv").is_file() or (RAW_DIR / f"{name}.csv.zip").is_file():
            # validate zip integrity for bt
            z = RAW_DIR / f"{name}.csv.zip"
            if not (RAW_DIR / name / "edges.csv").is_file() and z.is_file():
                try:
                    with zipfile.ZipFile(z, "r") as zf:
                        bad = zf.testzip()
                        if bad is not None:
                            print(f"[warn] corrupt zip {z.name}: {bad}; skip layer")
                            continue
                except zipfile.BadZipFile:
                    print(f"[warn] bad zip {z.name}; skip layer")
                    continue
            available.append(name)
    print("Available layers:", available)

    report = inspect_report(available)
    dict_path = write_data_dictionary(report)
    print("Data dictionary:", dict_path)

    network, available, missing_main, coverage = build_network(available)
    mat_path = save_outputs(network, available, coverage, report)
    print("Saved MAT:", mat_path)
    print("Missing main layers:", missing_main)
    print("Coverage:", json.dumps(coverage, indent=2))

    results_dir = PROJECT_DATA.parent / "06_Results"
    summary = analyze_and_plot(network, results_dir)
    print("Summary:", json.dumps(summary, indent=2)[:1000])


if __name__ == "__main__":
    main()
