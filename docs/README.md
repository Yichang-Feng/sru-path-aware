# docs 索引

按用途分四类。**读的顺序通常是 `specs` → `reports` → `evals`**：先看要做什么，再看结果如何、为什么，
最后才看支撑数字。

```
docs/
├── specs/      规格 / 执行手册 / 计划 —— "要做什么"（冻结文档，改动需评审）
├── reports/    分析 / 诊断 / 交接     —— "发生了什么、为什么"
├── evals/      评测与实测输出         —— 支撑数字（.md 摘要 + .json 逐回合数据）
└── papers/     参考文献
```

---

## specs/ —— 要做什么

| 文件 | 内容 | 状态 |
|---|---|---|
| [agent_doing.md](specs/agent_doing.md) | B2W path-aware v2 重设计规格（Stage 0–G）；**附录 A 是 v1.1 当前系统全貌**（训练方式、奖励表、维度、新增机制） | v1.0 已实施；v1.1 见附录 A |
| [agent_do_qwen.md](specs/agent_do_qwen.md) | V1.1 执行手册：三个行为缺陷的定位与修复 | 部分采纳，裁决见 `v1p1_next_plan.md` §0 |
| [v1p1_next_plan.md](specs/v1p1_next_plan.md) | **V1.1 下一步计划**（据 qwen 文档梳理 + P0 实测结果裁决） | 执行中 |
| [navigation_rewards.md](specs/navigation_rewards.md) | 导航奖励项参考表（每一项的语义、量级、陷阱） | 参考 |

## reports/ —— 发生了什么

| 文件 | 内容 |
|---|---|
| [b2w_pathfix_v1_final_report.md](reports/b2w_pathfix_v1_final_report.md) | **v1 4000 iter 全量分析**：目标旅馆、路径跟踪、干净 eval、现场两个失效模式 |
| [v1p1_p0_diagnosis.md](reports/v1p1_p0_diagnosis.md) | **P0 诊断 D1–D8**：低层转身/爬梯能力、冻结/倒走占比、invalid 分桶、free mask 根因 |
| [b2w_pathfix_v1_training_analysis_2026-11-26.md](reports/b2w_pathfix_v1_training_analysis_2026-11-26.md) | 另一份 v1 训练诊断（旅馆机制的独立交叉验证） |
| [phase0_baseline_report.md](reports/phase0_baseline_report.md) | Stage 0：M1–M5 实测、实施取舍、单测与基线重评 |
| [b2w_reward_recalibration_2026-11-24.md](reports/b2w_reward_recalibration_2026-11-24.md) | 奖励重标定 |
| [b2w_timeout_penalty_analysis_2026-11-23.md](reports/b2w_timeout_penalty_analysis_2026-11-23.md) | 超时惩罚分析 |
| [go2w_training_analysis_2026-11-24.md](reports/go2w_training_analysis_2026-11-24.md) | Go2W 训练分析 |
| [b2w_path_aware_handoff_2026-11-25.md](reports/b2w_path_aware_handoff_2026-11-25.md) | 交接文档 |

## evals/ —— 支撑数字

| 文件 | 内容 |
|---|---|
| [phase0_measurements.md](evals/phase0_measurements.md) | M1–M5 的原始测量输出 |
| [phase0_baseline_60s.md](evals/phase0_baseline_60s.md) / [90s](evals/phase0_baseline_90s.md) | 旧策略基线重评（**锁存口径**，已被 `_dwell` 版取代） |
| [phase0_baseline_60s_dwell.md](evals/phase0_baseline_60s_dwell.md) `.json` | 旧策略、驻留口径基线（**验收基线：SR 0.480**） |
| [b2w_pathfix_v1_eval_60s.md](evals/b2w_pathfix_v1_eval_60s.md) / [90s](evals/b2w_pathfix_v1_eval_90s.md) `.json` | v1 策略干净 eval（SR 0.105 / 0.120） |
| [v1p1_p0_eval.md](evals/v1p1_p0_eval.md) `.json` · [summary](evals/v1p1_p0_eval_summary.md) | P0 的 200 局 eval（D3–D8 的原始数据） |
| [v1p1_d3_after_fix.json](evals/v1p1_d3_after_fix.json) | free mask 修复后的 D3 复测（stairs invalid 8.14 → 0.95） |
| [v1p1_p0_probe.json](evals/v1p1_p0_probe.json) | D1/D1b 低层探针原始输出 |

产出这些文件的命令：
`scripts/eval_policy.py`（逐回合 eval）、`scripts/p0_eval_analysis.py`（D3–D8 聚合）、
`scripts/p0_low_level_probe.py`（D1/D1b）、`scripts/p0_stairs_tiles.py`（tile 级掩码检查）、
`scripts/probe_path_geometry.py`（离线路径几何，不启动 Isaac）。

## papers/ —— 参考文献

- Haro et al. 2026 — Path-conditioned RL-based Local Planning for Long-Range Navigation
- Yang et al. 2026 — Spatially-enhanced recurrent memory for long-range mapless navigation
