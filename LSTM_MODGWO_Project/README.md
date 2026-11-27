# LSTM_MODGWO_Project

MATLAB workspace for **LSTM-assisted Multi-Objective Discrete Grey Wolf Optimizer** on directed, weighted, dynamic multilayer competitive influence maximization.

## Current status

Stages 1–5 smoke complete. Formal Stage-5 MODGWO is prepared for **remote execution** (do not run heavy jobs on the authoring machine by default).

## Sync via GitHub (preferred)

Repo: `https://github.com/wyufeng986-commits/LSTM-MODGWO-dyn-multilayer-CIM` (private)

```bash
git clone https://github.com/wyufeng986-commits/LSTM-MODGWO-dyn-multilayer-CIM.git
cd LSTM-MODGWO-dyn-multilayer-CIM/LSTM_MODGWO_Project
```

## Formal Stage-5 (remote machine)

1. `git pull`
2. Edit `run_formal_modgwo.m`: start with `PRESET = 'medium'`, then `'formal'`.
3. Run `run_formal_modgwo` or `experimentMODGWO_formal('preset','formal')`.
4. Commit & `git push` the new `06_Results/` artifacts, then pull on the writing machine.

Details: `docs/阶段五正式实验_异机运行说明.md`

## Layout

- `01_Data/` network build
- `02_Propagation/` competitive IC + objectives
- `03_LSTM/` residual final-\(F_A\) surrogate
- `04_MODGWO/` discrete multi-objective GWO
- `05_Experiments/` experiment entry points
- `06_Results/` tables / figures / logs

## Related prior code

- Part 2: `多层网络多目标优化`
- Directed-weighted series: `有向加权多层网络影响力最大化`
