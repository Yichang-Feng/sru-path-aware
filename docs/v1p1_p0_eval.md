# Phase 0 基线重评

- checkpoint: `model_3999.pt`
- horizon: 60 s,  episodes: 200,  actor: 1

| bucket | n | SR | SR(old rule) | timeout share | SPL | median d_path | p90 d_path | median start dist | median planned len | median travelled | in-goal steps | in-goal speed p50 | longest dwell run p50 / p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 200 | 0.110 | 0.495 | 0.675 | 0.048 | 0.34 | 0.65 | 13.9 | 19.5 | 23.8 | 0 | 1.11 | 0 / 8 |
| maze | 19 | 0.211 | 0.684 | 0.684 | 0.108 | 0.34 | 0.68 | 13.3 | 17.7 | 38.1 | 41 | 1.17 | 6 / 8 |
| non_maze | 18 | 0.111 | 0.833 | 0.833 | 0.041 | 0.34 | 0.78 | 13.7 | 17.2 | 52.5 | 107 | 1.09 | 7 / 7 |
| stairs | 69 | 0.058 | 0.232 | 0.681 | 0.024 | 0.34 | 0.47 | 13.4 | 21.6 | 17.2 | 0 | 1.08 | 0 / 7 |
| pits | 94 | 0.128 | 0.585 | 0.638 | 0.053 | 0.35 | 0.69 | 14.1 | 18.9 | 31.6 | 28 | 1.10 | 6 / 8 |
