# Phase 0 基线重评

- checkpoint: `model_6000.pt`
- horizon: 60 s,  episodes: 200,  actor: 1

| bucket | n | SR | SR(old rule) | timeout share | SPL | median d_path | p90 d_path | median start dist | median planned len | median travelled | in-goal steps | in-goal speed p50 | longest dwell run p50 / p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 200 | 0.480 | 0.490 | 0.395 | 0.339 | 1.19 | 2.17 | 14.6 | 25.5 | 15.4 | 0 | 0.48 | 0 / 8 |
| maze | 26 | 0.538 | 0.538 | 0.423 | 0.395 | 1.57 | 2.89 | 16.6 | 34.8 | 20.2 | 8 | 0.59 | 8 / 8 |
| stairs | 72 | 0.403 | 0.417 | 0.361 | 0.256 | 1.55 | 2.32 | 14.4 | 27.7 | 15.8 | 0 | 0.51 | 0 / 8 |
| pits | 102 | 0.520 | 0.529 | 0.412 | 0.384 | 0.89 | 1.87 | 14.2 | 22.4 | 14.8 | 7 | 0.41 | 7 / 8 |
