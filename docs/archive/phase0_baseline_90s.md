# Phase 0 基线重评

> **已归档 2026-11-27**。锁存口径（"曾进入 0.5 m"）的 90 s 基线，已被驻留口径取代；且同样受 `eval_policy.py` 的 off-by-one 影响。
> 见 `docs/reports/b2w_pathfix_v1_final_report.md` §6。


- checkpoint: `model_6000.pt`
- horizon: 90 s,  episodes: 200,  actor: 1

| bucket | n | SR | SR(old rule) | SPL | median d_path | p90 d_path | median start dist | median planned len | median travelled |
|---|---|---|---|---|---|---|---|---|---|
| all | 200 | 0.625 | 0.625 | 0.473 | 0.92 | 2.06 | 15.2 | 23.6 | 16.2 |
| maze | 46 | 0.543 | 0.543 | 0.402 | 0.99 | 1.86 | 15.5 | 28.6 | 16.2 |
| non_maze | 34 | 0.853 | 0.853 | 0.658 | 0.94 | 2.33 | 12.8 | 20.6 | 14.6 |
| stairs | 12 | 0.500 | 0.500 | 0.317 | 0.54 | 1.42 | 10.6 | 16.6 | 13.5 |
| pits | 108 | 0.602 | 0.602 | 0.462 | 1.01 | 2.17 | 16.1 | 24.6 | 16.8 |
