# docs 索引

按用途分五类。**读的顺序通常是 `specs` → `reports` → `evals`**：先看要做什么，再看结果如何、为什么，
最后才看支撑数字。`archive/` 放已被取代、但仍有史料价值的文档。

```
docs/
├── specs/      规格 / 执行手册 / 计划 —— "要做什么"（冻结文档，改动需评审）
├── reports/    分析 / 诊断             —— "发生了什么、为什么"
├── evals/      评测与实测输出           —— 支撑数字（.md 摘要 + .json 逐回合数据）
├── archive/    已被取代的文档           —— 不要再据此行动
└── papers/     参考文献
```

> 前缀含义：`v1` = 第一次全量（目标旅馆那轮）；`v1.1` = 清零接近奖励 + 朝向势函数；
> `v1.2` = 修势函数采空 + 超时罚 dt 折扣 + 走廊放宽。当前生效参数见 `specs/agent_doing.md` **附录 A.0**。

---

## specs/ —— 要做什么

| 文件 | 内容 | 状态 |
|---|---|---|
| [agent_doing.md](specs/agent_doing.md) | B2W path-aware v2 重设计规格（Stage 0–G） | v1.0 正文冻结；**附录 A 是系统全貌，A.0 是 v1.2 当前口径**（A.1–A.12 写于 v1.1，对应条目已过期） |
| [v1p1_next_plan.md](specs/v1p1_next_plan.md) | V1.1 下一步计划（qwen 文档梳理 + P0 实测裁决） | **已完成**：P0 已跑、P1 已实施；剩余项与风险见 `reports/v1p2_early_analysis_*.md` §5 |
| [agent_do_qwen.md](specs/agent_do_qwen.md) | V1.1 执行手册（另一个模型提出） | **历史提案**：部分采纳，三处被证伪（裁决见 `v1p1_next_plan.md` §0） |
| [navigation_rewards.md](specs/navigation_rewards.md) | 导航奖励项参考表（语义、量级、陷阱） | 参考；§2 表格是 2026-11-24 的**历史**配置，v1.2 当前值见其顶部说明 |

## reports/ —— 发生了什么

| 文件 | 内容 |
|---|---|
| [b2w_pathfix_v1_final_report.md](reports/b2w_pathfix_v1_final_report.md) | **v1 全量分析**：目标旅馆、路径跟踪、干净 eval、现场两个失效模式 |
| [v1p1_p0_diagnosis.md](reports/v1p1_p0_diagnosis.md) | **P0 诊断 D1–D8**：低层转身/爬梯能力、冻结/倒走占比、invalid 分桶、free mask 根因 |
| [v1p1_training_analysis_2026-11-27.md](reports/v1p1_training_analysis_2026-11-27.md) | **v1.1 全量分析**：势函数被采空（15.4 vs π）、超时罚只有 −10，两处修复 |
| [v1p2_early_analysis_2026-11-27.md](reports/v1p2_early_analysis_2026-11-27.md) | **v1.2 早期读数**：修复生效的指纹 + 200 轮拐点是"清零接近奖励"的副作用 + 候选修法 |
| [phase0_baseline_report.md](reports/phase0_baseline_report.md) | Stage 0：M1–M5 实测、实施取舍、单测与基线重评（§3/§5/§6 为 v1.0 口径，见文首提示） |
| [b2w_reward_recalibration_2026-11-24.md](reports/b2w_reward_recalibration_2026-11-24.md) | 奖励重标定（path-aware 之前） |
| [b2w_timeout_penalty_analysis_2026-11-23.md](reports/b2w_timeout_penalty_analysis_2026-11-23.md) | 超时惩罚分析（path-aware 之前；dt 折扣问题的前身） |
| [go2w_training_analysis_2026-11-24.md](reports/go2w_training_analysis_2026-11-24.md) | Go2W 训练分析（path-aware 之前） |

## evals/ —— 支撑数字

| 文件 | 内容 |
|---|---|
| [phase0_measurements.md](evals/phase0_measurements.md) | M1–M5 的原始测量输出 |
| [phase0_baseline_60s_dwell.md](evals/phase0_baseline_60s_dwell.md) `.json` | 旧策略、驻留口径基线（**验收基线：SR 0.480 / SPL 0.339**） |
| [b2w_pathfix_v1_eval_60s.md](evals/b2w_pathfix_v1_eval_60s.md) / [90s](evals/b2w_pathfix_v1_eval_90s.md) `.json` | v1 策略干净 eval（SR 0.105 / 0.120） |
| [v1p1_eval.md](evals/v1p1_eval.md) `.json` | v1.1 策略干净 eval（SR 0.010；到得了目标但停不下来） |
| [v1p1_p0_eval.md](evals/v1p1_p0_eval.md) `.json` · [summary](evals/v1p1_p0_eval_summary.md) | P0 的 200 局 eval（D3–D8 的原始数据） |
| [v1p1_d3_after_fix.json](evals/v1p1_d3_after_fix.json) | free mask 修复后的 D3 复测（stairs invalid 8.14 → 0.95） |
| [v1p1_p0_probe.json](evals/v1p1_p0_probe.json) | D1/D1b 低层探针原始输出 |

> `.json` 是逐回合原始数据，合计约 0.7 MB，用于复核报告里的每个数。若要瘦身，删之前先确认对应报告
> 已经写完并入库；`.md` 摘要都可以单独读。

产出这些文件的命令：
`scripts/eval_policy.py`（逐回合 eval）、`scripts/p0_eval_analysis.py`（D3–D8 聚合）、
`scripts/p0_low_level_probe.py`（D1/D1b）、`scripts/p0_stairs_tiles.py`（tile 级掩码检查）、
`scripts/probe_path_geometry.py`（离线路径几何，不启动 Isaac）。
读训练日志用 `skills/sru-training-analysis/`（按 run 自己的 `params/env.yaml` 换算 + 四条不变量检查）。

## archive/ —— 已被取代，不要再据此行动

| 文件 | 取代者 |
|---|---|
| [b2w_path_aware_handoff_2026-11-25.md](archive/b2w_path_aware_handoff_2026-11-25.md) | `specs/agent_doing.md` 附录 A（系统全貌）；奖励口径早已改写 |
| [b2w_pathfix_v1_training_analysis_2026-11-26.md](archive/b2w_pathfix_v1_training_analysis_2026-11-26.md) | `reports/b2w_pathfix_v1_final_report.md`（结论复核，三处被证伪） |
| [phase0_baseline_60s.md](archive/phase0_baseline_60s.md) / [90s](archive/phase0_baseline_90s.md) | `evals/phase0_baseline_60s_dwell.md`（锁存口径 → 驻留口径） |

## papers/ —— 参考文献

- Haro et al. 2026 — Path-conditioned RL-based Local Planning for Long-Range Navigation
- Yang et al. 2026 — Spatially-enhanced recurrent memory for long-range mapless navigation
