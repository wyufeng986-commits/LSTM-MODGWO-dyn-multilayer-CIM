# 异机 Agent 运行流程（可直接执行）

把下面整段交给计算侧 Agent。目标：拉取私有仓 → 跑 Stage-5 正式/半正式 MODGWO → 推回结果。

---

## 0. 环境前提

- Windows + MATLAB（建议 R2021a+，有 Deep Learning Toolbox）
- 已能访问 GitHub（若 `git clone` 出现 `Connection was reset`，先解决代理/网络，再继续）
- GitHub 账号对私有仓有权限：`wyufeng986-commits/LSTM-MODGWO-dyn-multilayer-CIM`

---

## 1. 获取代码

若尚未克隆：

```bat
cd /d E:\多重网络研究
git clone https://github.com/wyufeng986-commits/LSTM-MODGWO-dyn-multilayer-CIM.git
cd LSTM-MODGWO-dyn-multilayer-CIM
```

若已有仓库：

```bat
cd /d E:\多重网络研究\LSTM-MODGWO-dyn-multilayer-CIM
git pull
```

工作目录此后固定为：

```text
...\LSTM-MODGWO-dyn-multilayer-CIM\LSTM_MODGWO_Project
```

---

## 2. 开跑前自检

确认这些文件存在：

- `LSTM_MODGWO_Project\01_Data\processed\CNS_daily_multilayer.mat`
- `LSTM_MODGWO_Project\03_LSTM\models\model_lstm_resid_h4.mat`
- `LSTM_MODGWO_Project\run_formal_modgwo.m`
- `LSTM_MODGWO_Project\05_Experiments\experimentMODGWO_formal.m`

缺任一文件：停止，回报缺失路径，不要盲跑。

---

## 3. 运行策略（必须按顺序）

### Step A：先干跑 `medium`（必做）

1. 打开 `LSTM_MODGWO_Project\run_formal_modgwo.m`
2. 设置：

```matlab
PRESET = 'medium';
```

3. 在 MATLAB 中执行：

```matlab
cd('E:\多重网络研究\LSTM-MODGWO-dyn-multilayer-CIM\LSTM_MODGWO_Project');  % 按实际路径改
run_formal_modgwo
```

或命令行：

```bat
cd /d E:\多重网络研究\LSTM-MODGWO-dyn-multilayer-CIM\LSTM_MODGWO_Project
matlab -batch "run_formal_modgwo"
```

4. 成功判据：
   - 命令正常结束，无报错退出
   - `06_Results\logs\` 出现新的 `stage5_modgwo_medium_*.mat` / `*.diary.txt`
   - diary 末尾有 `Formal run finished` 或 `DONE runId=...`

5. **先提交推送 medium 结果**（即使后面还要跑 formal）：

```bat
cd /d E:\多重网络研究\LSTM-MODGWO-dyn-multilayer-CIM
git add LSTM_MODGWO_Project/06_Results
git commit -m "Add medium MODGWO remote dry-run results"
git push
```

### Step B：再正式 `formal`（A 成功后才做）

1. 改 `run_formal_modgwo.m`：

```matlab
PRESET = 'formal';
```

2. 同样方式运行 `run_formal_modgwo`
3. 跑完后推送：

```bat
cd /d E:\多重网络研究\LSTM-MODGWO-dyn-multilayer-CIM
git add LSTM_MODGWO_Project/06_Results
git commit -m "Add formal MODGWO remote results"
git push
```

---

## 4. 参数约定（不要擅自改）

| 项 | medium | formal |
|----|--------|--------|
| popSize | 20 | 50 |
| maxIter | 20 | 50 |
| archiveSize | 40 | 100 |
| kA, kB | 5 | 5 |
| evalMode | hybrid | hybrid |
| R_final | 20 | 50 |

禁止事项：

- 不要把 `kA/kB` 改成 `0.05N`（当前 LSTM 按 k=5 训练）
- 不要删终局 MC 校正（`finalCorrect` 保持 true）
- 不要在未沟通时改成 `formal_mc`（纯 MC，很慢）
- 本任务**不需要**重新训练 LSTM

---

## 5. 产物清单（推送这些即可）

`LSTM_MODGWO_Project/06_Results/` 下本次新增：

- `logs/stage5_modgwo_<tag>_<timestamp>.mat`
- `logs/stage5_modgwo_<tag>_<timestamp>.diary.txt`
- `logs/stage5_modgwo_<tag>_<timestamp>_config.json`
- `tables/..._archive.csv`
- `tables/..._history.csv`
- `figures/..._pareto.png`
- `figures/..._history.png`

---

## 6. 失败时怎么处理

1. `Connection was reset`（git）：先修网络/代理，再 `git pull/push`，不要改代码硬闯  
2. Preflight 缺 mat/模型：停止并报告缺哪个文件  
3. 缺 Deep Learning Toolbox：停止并报告；不要擅自改成乱参数的 mc 凑合（除非用户明确要求）  
4. MATLAB 报错：把 diary 末尾报错栈全文保留并推送 diary（若可写盘），同时在回复里粘贴关键错误

---

## 7. 完成后向用户汇报的最短模板

```text
状态: medium成功 / formal成功 / 失败
runId: ...
archive数: ...
maxFA: ...
minCost: ...
耗时: ...
已push: 是/否
关键文件: 06_Results/logs/...mat
若失败: <错误摘要>
```

---

## 8. 一句话任务单（可复制）

请在仓库 `LSTM-MODGWO-dyn-multilayer-CIM` 的 `LSTM_MODGWO_Project` 中：先 `git pull`；按 `docs/异机Agent运行流程.md` 先跑 `PRESET=medium`，成功后 commit+push 结果；再跑 `PRESET=formal`，成功后再 commit+push；不要改 k、不要重训 LSTM、不要跳过最终 MC 校正；最后按最短模板汇报。
