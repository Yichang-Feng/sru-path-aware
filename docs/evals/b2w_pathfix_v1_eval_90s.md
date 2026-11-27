# Phase 0 基线重评

- checkpoint: `model_3999.pt`
- horizon: 90 s,  episodes: 200,  actor: 1

| bucket | n | SR | SR(old rule) | timeout share | SPL | median d_path | p90 d_path | median start dist | median planned len | median travelled | in-goal steps | in-goal speed p50 | longest dwell run p50 / p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 200 | 0.120 | 0.560 | 0.640 | 0.049 | 0.36 | 0.68 | 16.0 | 21.0 | 32.6 | 10 | 1.08 | 6 / 7 |
| maze | 42 | 0.167 | 0.595 | 0.571 | 0.058 | 0.37 | 0.69 | 16.4 | 21.5 | 41.2 | 38 | 1.07 | 6 / 8 |
| stairs | 60 | 0.083 | 0.417 | 0.650 | 0.040 | 0.35 | 0.58 | 15.6 | 22.7 | 28.8 | 0 | 1.07 | 0 / 7 |
| pits | 98 | 0.122 | 0.633 | 0.663 | 0.050 | 0.36 | 0.71 | 16.1 | 20.6 | 44.7 | 51 | 1.10 | 6 / 7 |
