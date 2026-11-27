# Phase 0 基线重评

> **已归档 2026-11-27**。锁存口径（"曾进入 0.5 m"）的基线重评，已被 `docs/evals/phase0_baseline_60s_dwell.md`（驻留口径，SR 0.480）取代。
> 当时 `eval_policy.py` 的"提前结束"判定有 off-by-one，导致 SR 退化回锁存口径，见 `docs/reports/b2w_pathfix_v1_final_report.md` §6。


- checkpoint: `model_6000.pt`
- horizon: 60 s,  episodes: 200,  actor: 1

| bucket | n | SR | SR(old rule) | SPL | median d_path | p90 d_path | median start dist | median planned len | median travelled |
|---|---|---|---|---|---|---|---|---|---|
| all | 200 | 0.580 | 0.580 | 0.449 | 0.91 | 1.99 | 13.9 | 24.6 | 14.4 |
| maze | 57 | 0.737 | 0.737 | 0.577 | 0.76 | 1.95 | 12.2 | 22.4 | 13.9 |
| non_maze | 27 | 0.815 | 0.815 | 0.595 | 0.87 | 2.24 | 12.3 | 24.8 | 14.6 |
| stairs | 52 | 0.346 | 0.346 | 0.263 | 1.03 | 1.48 | 12.8 | 23.8 | 12.7 |
| pits | 64 | 0.531 | 0.531 | 0.425 | 1.13 | 2.41 | 14.8 | 25.3 | 15.2 |
