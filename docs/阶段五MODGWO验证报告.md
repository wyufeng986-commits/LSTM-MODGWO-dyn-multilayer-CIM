# 阶段五：LSTM-MODGWO 验证报告

**日期：** 2026-11-27  
**状态：** 烟雾测试通过（合法解 + 非支配档案 + MC 终校）

---

## 1. 本阶段完成内容

1. 固定 \(k\) 离散编码与 `repairSeedSet` 约束修复  
2. 潜力 / 随机 / 多样性混合初始化  
3. 三领导狼定向交换 + 多数投票（`discreteUpdate`）  
4. Pareto 档案 + 网格密度截断 / 反密度选狼  
5. 单点交换局部搜索  
6. `mc` / `lstm` / `hybrid` 评价桥接；档案结束强制 MC 校正  
7. 烟雾实验 `experimentMODGWO_smoke`

---

## 2. 烟雾设置

| 项 | 值 |
|----|----|
| 网络 | CNS daily，N=838，L=4，T=28 |
| \(k_A=k_B\) | 5 |
| 模式 | hybrid（残差 LSTM + 周期/终局 MC） |
| 种群 / 迭代 / 档案 | 8 / 4 / 15 |
| \(R_{\mathrm{fast}}, R_{\mathrm{final}}\) | 2 / 5 |

---

## 3. 结果摘要

| 指标 | 值 |
|------|-----|
| 终档规模 | 2（均非支配） |
| 终校后 max \(F_A\) | ≈304.4 |
| 终校后 min Cost | ≈2.234 |
| 评价次数 | ≈69 |
| 运行时间 | ≈4 s |

说明：优化过程中 LSTM 近似 \(F_A\) 曾偏高（≈555），终局 MC 校正后回落到真实量级，说明 **强制校正必要且生效**。

合法性检查：所有档案解 \(|S_A|=k\)、与 \(S_B\) 无交、档案内无被支配解。

产物：

- `06_Results/tables/stage5_modgwo_smoke.csv`
- `06_Results/figures/stage5_modgwo_smoke_pareto.png`
- `06_Results/logs/stage5_modgwo_smoke.mat`

---

## 4. 对后续的含义

- 算法主循环可用，可放大 `pop/iter/R` 做正式对比实验（阶段六/七）。  
- hybrid 适合加速搜索；论文主结果必须以终局 MC 档案为准。  
- 下一步可选：加对比算法入口、消融开关（无 LSTM / 无局部搜索 / 无投票）、HV/IGD 指标。

---

## 5. 复现

```matlab
cd('...\LSTM_MODGWO_Project');
addpath(genpath('.'));
result = experimentMODGWO_smoke();
```
