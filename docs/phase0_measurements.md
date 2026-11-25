# Phase 0 阻塞性实测报告

场景：4×10 tiles, PRM N=300

## M1 B2W 机身尺寸（USD 包围盒，世界系）

| prim | min (x,y,z) | size (x,y,z) |
|---|---|---|
| base_link | [0.8266, -0.1489, 0.645] | [0.8544, 0.2979, 0.3475] |
| robot_total | [0.8092, -0.2603, -0.0626] | [0.8828, 0.5201, 1.0551] |

→ 机身半宽（y）= **0.149 m**，半长（x）= 0.427 m
→ `seg_passable` padding = 半宽 + 0.05 = **0.199 m**

## M2 通道 / 障碍间距分布（clearance = 到最近障碍的半宽，米）

| 统计量 | 全部自由格 | 通道脊线（局部极大，<1.0 m） |
|---|---|---|
| p1 | 0.100 | 0.100 |
| p5 | 0.200 | 0.100 |
| p25 | 0.583 | 0.224 |
| p50 | 1.063 | 0.400 |
| p75 | 1.772 | 0.500 |
| p95 | 3.289 | 0.806 |

→ clearance < 0.45 m 的自由格占比：**19.6%**（窄门/窄通道候选）
→ 通道脊线最小值 0.100 m → 对应最窄净宽 0.200 m

## M3 `free_nodes` 采样是否已含 padding 膨胀

- 每个 tile 上 padding 窗口（`cfg.padding=4` 格）覆盖的自由格比例：均值 26.8%，最大 37.6%
- `free_nodes / nodes` 比例：均值 94.1%
- 结论：`sample_free_point()` 只拒绝 `padding_mask` 中的点，因此 **节点采样已含 padding 膨胀**；但 `free_nodes` 用 `free_mask` 判定（仅地面/平台），二者含义不同。

## M4 `is_traversable` 语义

- 跨墙段 `is_traversable(check_padding=True)` = False
- 跨墙段 `is_traversable(check_padding=False)` = False
- 端点落在障碍内：返回 True（**不抛异常**）
- `max_height_diff` = 50.0 单位 = 0.250 m；楼梯单级 0.2 m = 40 单位 → 可通过

## M5 optimal 路径上 `check_padding=True` 的不可行段比例

- 采样段数：1400（100 条 path 以上）
- `check_padding=True` 判不可行：**6.5%**
- `check_padding=False` 判不可行：2.4%
- 其中段上最小 clearance < 0.45 m（窄段）：108 段，不可行 45.4%
- 开阔段：1292 段，不可行 3.3%

