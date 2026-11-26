# b2w_pathfix_v1 训练诊断报告

> 数据截至 **iter 3230 / 2026-11-26 08:42 CST**，训练仍在进行（4000 iter，约 3 h 后结束）。
> 数据源：`IsaacLab/logs/rsl_rl/b2w_navigation_mdpo/2026-11-25_19-40-08_b2w_pathfix_v1/events.out.tfevents.*`（44 个标量 × 3230 点）。
> 本文所有数字均可由第 9 节命令复现；派生量的公式在第 3 节给出，请勿脱离口径直接引用。

---

## 0. 一句话结论

训练没有崩，**是奖励结构把策略推到了一个"停在目标附近刷分"的局部最优**：每回合总收入的构成从"完成目标为主"翻转为"在目标半径内停留为主"（停留/完成 收入比 0.80 → 6.4），而路径项因为实际横向误差稳定在 ~0.9 m、正好落在 0.8 m 支付门之外而失去牵引力。

---

## 1. 复现信息

| 项 | 值 |
|---|---|
| 任务 | `Isaac-Nav-MDPO-B2W-v0` |
| 启动命令 | `train_b2w_pathaware.sh --task Isaac-Nav-MDPO-B2W-v0 --num_envs 1046 --max_iterations 4000 --seed 60 --run_name b2w_pathfix_v1 --headless --no_debug_vis` |
| 进程 | PID 2637178（`ps` 确认存活，CPU 99%） |
| 启动时间 | 2026-11-25 19:40:08 |
| 代码版本 | 工作区干净；运行时代码为 commit `33ae349`（"Stop path noise from stepping outside the tile"，19:24）。`3c7e555`（19:44）为 wandb 步数修复，启动后才提交，不在本 run 内 |
| 环境步长 | `dt = 0.005 × decimation 40 = 0.2 s`，`episode_length_s = 90` → **450 步/回合上限** |
| 存档 | `model_0/500/1000/1500/2000/2500/3000`，500 iter ≈ 2 h |
| 吞吐 | 1212–1220 fps（采集 9.65 s + 学习 4.08 s/iter），13.7 s/iter |

GPU：RTX 4090 D，17 132 / 24 564 MiB（69.7%），61 ℃，160 W / 425 W，无 ECC / 离线错误。
**显存余量只有 7.4 GB，训练期间不要再起第二个 Isaac 实例**（会把本进程搞死）。

---

## 2. 必须先弄清的四个统计口径

这一节是本次分析里最容易读错的地方，之前我自己也读错过两处。

### 2.1 `Metrics/*` 是**回合结束时刻的快照**，不是全程均值

`Metrics/robot_goal/*` 由 `RobotNavigationGoalCommand._update_metrics()` 在**每个 env reset 的那一步**采样（CommandTerm 的既有行为）。所以：

- `distance_to_path` = **回合结束时**机器人在哪儿离路径多远
- 它是一个**按终止类型分层的条件分布**：成功回合（终点在路径端点上，d≈0）、超时回合、摔倒回合各自贡献样本

**因此 `distance_to_path` 上升 ≠ "机器人平均离路径更远"**，也可能只是"回合的结束构成变了"。两者要分开看。

### 2.2 `Episode_Reward/*` = 每回合该奖励项的**总和 ÷ 90 s**

所以 **每回合总量 = 日志值 × 90**。本报告所有"每回合收入"都按此换算，且已验证：各项之和与 `Train/mean_reward` 在每个窗口都对得上（误差 < 0.3）。

### 2.3 `Episode_Termination/*` 是**比例**，不要再乘 90

各终止项之和略大于 1，因为一个回合可以同时触发多个终止条件（例如 `early_termination` 包含 `time_out`）。

### 2.4 `success_rate` 与 `goal_reached_bonus` 两个口径不一致

- `Metrics/robot_goal/success_rate`：每个 env 最近 **10 个回合**的滚动成功率（`SuccessRateTracker`，`buffer_size=10`），在 reset 步平均 → 最新 **0.291**
- `Episode_Reward/goal_reached_bonus / 25`：实际发放的完成驻留次数 → 最新 **0.212**

差 37%。两者的定义应当一致（`at_goal_navigation` 的成功终止既写 buffer 又打脉冲），因此这是一个**待查的口径问题**，不是分析结论。本报告在需要"成功率"时同时给出两个值。

---

## 3. 派生量的定义

| 派生量 | 公式 | 说明 |
|---|---|---|
| 每回合付费路径米数 | `Episode_Reward/progress_along_path × 90 ÷ 1.2` | 1.2 是 `per_metre` 权重 |
| 每回合在目标半径内步数 | `Episode_Reward/reach_goal × 90 ÷ (2.0 × 0.2)` | `reach_goal_direct` 每步 0.4 |
| 目标形函数收入 | `(reach_goal + reach_goal_xy_soft + reach_goal_xy_tight) × 90` | 三者都只在目标附近/到达后支付 |
| 全称平均 d_path（下界） | `\|path_deviation × 90\| ÷ (ep_len × 0.15 × 0.2) + 0.5` | 由偏差惩罚反推。**是下界**：惩罚在 excess=1.5 m 处封顶，实际偏差更大时会被低估 |

---

## 4. 事实一：全体标量快照

见文末附录 A（44 项，含三个窗口的均值与趋势方向）。要点：

- **零 NaN / Inf**（3230 点全部有限）
- `Loss/explained_variance` **0.041 → 0.914**，critic 拟合良好
- `Loss/kl_divergence` 0.14 → 0.22，无尖峰
- `Loss/learning_rate` 9.94e-4 → 2.20e-4，与 4000 iter 余弦日程一致
- `Policy/log_std_max` 0.100 → **−0.265**，与移动墙日程一致（iter 3230 理论值 ≈ −0.27）
- `Policy/raw_log_std_mean` **从 iter ≈500 起钉死在 0.1035**，长期高于墙面 → clamp 全权负责降压，熵项梯度为零

---

## 5. 事实二：分窗口趋势（每 350 iter）

`d_path_avg*` 为第 3 节定义的反推下界；其余均为窗口均值。

| iter | reward | ep_len | SR | d_path@结束 | d_path_avg* | 付费米数 | 圈内步数 | 目标形函数 | 完成奖励 | 停留/完成 | time_out | base_contact | v@结束 | v→goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–350 | 23.4 | 98 | 0.354 | 0.38 | 0.57 | 11.0 | 9.3 | 7.6 | 9.5 | **0.80** | 0.028 | 0.514 | 0.83 | +0.23 |
| 350–700 | 22.8 | 209 | 0.254 | 0.58 | 0.70 | 8.8 | 23.8 | 19.5 | 4.8 | 4.10 | 0.175 | 0.573 | 0.94 | +0.10 |
| 700–1050 | 20.6 | 218 | 0.213 | 0.57 | 0.71 | 8.8 | 22.2 | 18.2 | 4.7 | 3.84 | 0.187 | 0.569 | 0.93 | +0.10 |
| 1050–1400 | 26.5 | 209 | 0.235 | 0.65 | 0.76 | 9.6 | 27.5 | 22.7 | 5.0 | 4.58 | 0.186 | 0.564 | 0.98 | +0.10 |
| 1400–1750 | 26.1 | 219 | 0.242 | 0.84 | 0.79 | 9.6 | 26.5 | 22.1 | 5.2 | 4.25 | 0.198 | 0.530 | 0.96 | +0.05 |
| 1750–2100 | 33.8 | 214 | 0.280 | 1.24 | 0.84 | 10.7 | 33.0 | 27.6 | 6.0 | 4.61 | 0.188 | 0.503 | 0.98 | −0.01 |
| 2100–2450 | 26.3 | 253 | 0.247 | 1.68 | 0.92 | 9.4 | 32.1 | 26.3 | 4.0 | 6.52 | 0.293 | 0.481 | 0.88 | +0.04 |
| 2450–2800 | 29.0 | 253 | 0.226 | 2.22 | 0.99 | 9.9 | 32.4 | 27.1 | 4.9 | 5.48 | 0.287 | 0.463 | 0.88 | −0.01 |
| 2800–3150 | 38.9 | 259 | 0.291 | 1.57 | 0.91 | 11.3 | 40.1 | 33.8 | 5.6 | 6.04 | 0.282 | 0.433 | 0.90 | −0.04 |
| 3150–3230 | 38.8 | 266 | 0.288 | 1.88 | 0.91 | 11.1 | 41.2 | 34.7 | 4.8 | 7.20 | 0.313 | 0.431 | 0.88 | −0.02 |

**从这里能读出的五件事：**

1. **收入结构在第 350–700 iter 就翻转了**，不是后期崩的：停留/完成比从 0.80 一步跳到 4.10，此后一直在 4–7 之间。
2. **d_path 的时间平均在 iter ≈2100 之后稳定在 0.9 m 左右**（0.57 → 0.70 → 0.71 → 0.76 → 0.79 → 0.84 → 0.92 → 0.99 → 0.91 → 0.91）。它**没有失控发散**，而是停在 0.8 m 支付门的外侧不动了。
3. **`d_path@结束` 仍在涨**（0.38 → 2.22，最新 1.88），说明"死在/耗在远处"的回合越来越多 —— 这与 2.1 节的构成效应一致，不能直接当作"平均跟踪变差"。
4. **付费路径米数基本平**（11.0 → 8.8 → 11.1），而回合长度从 98 步拉到 266 步：单位时间付费进度掉了 ~2.7 倍。
5. **末端速度 `v→goal` 由 +0.23 变成 −0.04**：回合结束那一刻，机器人整体上是在**远离**目标，而且总速率是 0.88 m/s —— 几乎全是切向运动，即在目标周围绕。

---

## 6. 事实三：每回合奖励分解（×90，已与 `mean_reward` 对平）

| 奖励项 | 0–350 | 最新 350 | 变化 |
|---|---:|---:|---|
| `progress_along_path`（跟路径） | +13.23 | +13.44 | 持平 |
| `progress`（直连投影） | +1.18 | +1.19 | 持平 |
| `reach_goal`（半径内每步） | +3.73 | +16.11 | **×4.3** |
| `reach_goal_xy_soft` | +0.46 | +2.02 | ×4.4 |
| `reach_goal_xy_tight` | +3.44 | +15.71 | **×4.6** |
| **目标附近小计** | **+7.63** | **+33.84** | **×4.4** |
| `goal_reached_bonus`（完成驻留） | +9.48 | +5.30 | **−44%** |
| `path_deviation_penalty` | −0.20 | −3.28 | ×16 |
| `action_rate_l1` | −1.34 | −3.40 | ×2.5 |
| `episode_termination` | −5.95 | −4.96 | — |
| `time_out_penalty` | −0.27 | −2.93 | ×11 |
| `joint_acc_l2_joint` | −0.23 | −0.51 | ×2.2 |
| `lateral_movement` | −0.08 | −0.21 | ×2.5 |
| **合计（= `Train/mean_reward`）** | **23.45** | **38.48** | **+64%** |

关键比值：

- 目标附近收入 / 路径收入：**0.58 → 2.52**
- 目标附近收入 / 完成奖励（"停留 vs 完成"）：**0.80 → 6.4**

也就是说，总奖励涨的 15 点里：**+26.2 来自"待在目标附近"，−4.2 来自"真正完成任务"，其余 −7 是惩罚项上升**（偏差 −3.1 / 动作率 −2.1 / 超时 −2.7，终止项回补 +1.0，关节与侧移 −0.4，路径项 +0.2）。

---

## 7. 机制：为什么会被推到这儿

### 7.1 驻留判定与当前速度不兼容（几何约束）

- 成功条件：在 0.5 m 半径内**连续 8 步**（`GOAL_DWELL_TIME_S=1.5` / `step_dt=0.2`，`round(7.5)=8` → 1.6 s），离开即清零（`reset_on_exit=True`）
- 圆直径 1.0 m，直线穿过去的速度上限 = 1.0 / 1.6 = **0.625 m/s**
- 实测回合末端速率 **0.88 m/s**，且 `v→goal ≈ −0.04`

**这是数据里最硬的矛盾**：半径内累计停留 **40.3 步**（占回合 15.3%），但成功率只有 0.21–0.29。它反复进圈，却从未连满 8 步 —— 因为一直在以 0.9 m/s 穿过一个 1 m 的圆。

### 7.2 目标附近的边际收益是沿路径行驶的 3–5 倍

| 位置/行为 | 每步奖励 |
|---|---:|
| 沿路径行驶（1.2/m @ 0.9 m/s） | ≈ **0.22** |
| 目标半径内，err=0.5 m | 0.40（`reach_goal`）+ 0.15（`tight`）≈ **0.55** |
| 目标半径内，err=0（理想） | 0.40 + 0.75 + 0.05 ≈ **1.15** |

而且**完成驻留会终止回合**，把这条逐步收入流水直接切断；实测超时只扣 2.93/回合。局部梯度指向"进圈、别连满 8 步"，而不是"完成"。

### 7.3 路径项的支付门和惩罚走廊错位，导致它在关键区间失效

| 横向误差 d | 能否拿到路径奖励 | 每步偏差惩罚 |
|---|---|---|
| ≤0.5 m | 能 | 0 |
| 0.5–0.8 m（**惩罚/支付错位的 0.3 m 带**） | 能 | 很小 |
| 0.8–2.0 m | **完全拿不到** | 线性，最大 −0.045/步 |
| >2.0 m | 拿不到 | **封顶，不再增加** |

配置：`gate_normal_m = 0.8` / `gate_relaxed_m = 1.2` / `gate_invalid_min_m = 1.6`；`path_deviation_penalty` 的 `corridor_radius = 0.5`、`cap = 1.5`。

实测时间平均 d ≈ **0.91 m**，恰好贴在 0.8 m 门外：按 0.88 m/s 行驶，**约 76% 的位移拿不到任何路径奖励**，而逃逸的代价封顶在 0.045/步。约束形状是"悬崖 + 天花板"，不是梯度。

### 7.4 探索能力在同步衰减

σ：0.85 → 1.10（峰值）→ 0.77；熵系数 → 1.71e-4；LR → 2.20e-4。加上 `raw_log_std_mean` 钉在墙上、熵项梯度为零，**政策没有能力从这个盆地爬出来**，σ 完全由日程表驱动。

---

## 8. 排除项（避免分析跑偏）

| 假设 | 证据 | 结论 |
|---|---|---|
| 训练崩溃 | 3230 点零 NaN/Inf；KL 0.14→0.22 无尖峰；surrogate 平稳 | 否 |
| critic 跟不上 | `explained_variance` 0.914；value loss 25→82（奖励尺度变大所致） | 否 |
| 算力瓶颈 | 1220 fps（开局 932 的 131%）；采集 9.65 s + 学习 4.08 s 稳定 | 否 |
| 路径生成退化 | `invalid_seg_frac` 0.217→0.232、`relax_frac` 0.082→0.089、`pinch_frac` 0.302→0.316、`a_star_failed` 0.032→0.032 全程平 | 否 |
| 地形变难 | 地形课程未接线（`navigation_env_cfg.py:161`） | 否 |
| 过拟合 | 没有验证集；EV 稳定；无 val loss 概念 | 不适用 |
| 机器人平均离路径 2.3 m | 混用了快照与均值；时间平均是 0.91 m（下界） | 否（口径错误） |

---

## 9. 与验收标准的对照

标准来自 `docs/phase0_baseline_report.md` §6 与 `docs/agent_doing.md` §10 G2/G3。

| G2 指标 | 期望 | 实测（最新 350） | 判定 |
|---|---|---|---|
| progress max / p99 | ≤ 1.5 / 无长尾 | **未记录该标量** | 无法判定 |
| `invalid_seg_frac` | 5–10%，稳定 | 0.232，稳定 | ⚠️ 稳定但超出区间 ~2× |
| `relax_frac` | 12–20%，稳定 | 0.089，稳定 | ⚠️ 稳定但低于区间 |
| `blocked_episode` | 0.15 → 0.30 ramp | 0.106 → 0.149 | ⚠️ 低于 ramp |
| `a_star_failed_frac` | 接近 0 | 0.032，稳定 | ⚠️ 约 3% |
| `d_path` 中位数 | 1500 iter 起下降，4000 iter < 0.8 m | 均值 0.91（反推下界）；快照 1.70 | ❌ 未下降且超目标 |
| 路径项 vs 直连项（×90） | 路径 > 直连 | 13.4 vs 1.2 = 11:1 | ✅ |
| `mean_noise_std` 跟墙 | 跟随 | 1.10 → 0.77 ≈ exp(wall) | ✅ |
| `raw_log_std_mean` | 在墙**下方**跟随 | 0.1035 长期**高于**墙 | ❌ |
| `explained_variance` | > 0.5 | 0.914 | ✅ |
| `base_contact` | 持续下降 | 0.514 → 0.430 | ✅ 下降，但仍在 0.4 以上 |
| `time_out` | < 0.15 | **0.296**（最新点 0.315） | ❌ 违反（>0.25 触发"查路径长分布"） |
| `success_rate` | ≥ 基线 + 0.10（0.68） | 0.291 / bonus 口径 0.212 | ❓ 口径不可直接比 |

**G3 成对指标（窄门通过率、挡路绕行成功率、窄门/挡路段 d_path 分位、stairs SR、死胡同计数）训练日志里完全没有记录** —— 即使训练正常，也无法做题面验收。这是验收指标设计上的缺口。

**G4 abort 条件**（1500 iter 时 SR < 基线 − 0.15 = 0.43）在当前口径下无法套用：训练期 `Metrics/*` 带观测噪声与外推，与 `eval_policy.py` 的 Play 协议不可比。若硬套，1500 iter 的 0.24 会直接触发 abort —— 但那是口径造成的，不是事实。

---

## 10. 与上一个 run 的对照（`b2w_sigma_bounded`，2026-11-24，6021 iter）

| 指标 | sigma_bounded 末段 | pathfix_v1 最新 350 |
|---|---:|---:|
| `success_rate` | 0.51 | 0.29 |
| `Train/mean_episode_length` | 149 | 264 |
| `Episode_Termination/base_contact` | 0.345 | 0.430 |
| `Episode_Termination/time_out` | 0.197 | 0.296 |
| `velocity_magnitude`@结束 | 0.618 | 0.882 |
| `Loss/kl_divergence` | 0.20 | 0.22 |

注意：两者任务不同（旧 run 无路径感知层、60 维路径观测、旧类别权重），**只能作为量级参考，不能当作 A/B**。但"末端速率高 43%、base_contact 高 9 个百分点"这一点值得单独注意 —— 新任务的策略整体开得更快、更不稳。

另外，旧 run 的 `Episode_Reward/progress_along_path` 在 0.009 量级（旧口径是百分比），不可比。

---

## 11. 未决问题与其判定方法

### Q1：是"奖励驱动的绕圈"还是"单纯刹不住车"？

两者外在特征相同，现有标量无法区分。判定方法（拿到 model_3000/4000 后）：

1. 在 `eval_policy.py` 里记录**进入 0.5 m 半径后的连续步数分布**与**半径内的速度分布**
   - 若半径内速度集中在 0.8–1.0 m/s 且连续步数多为 2–5 → 控制问题（刹不住）
   - 若半径内速度低而频繁出圈 → 奖励驱动（主动放弃完成）
2. 便宜的反证实验：把 `reach_goal` / `reach_goal_xy_soft` / `reach_goal_xy_tight` 的权重置零（保留 `goal_reached_bonus`），跑 300 iter 看成功率是否立刻回升

### Q2：成功率 0.291 vs 0.212 的差异

`at_goal_navigation` 既写 `goal_reached_buffer`（+1）又打 `goal_dwell_completed` 脉冲，两者应当 1:1。需要确认：

- 同一步里是否有别的终止项也对同一个 env 写了 0（`terminations.py:80/106/134/195/233/272`）
- buffer 是 10 深的滚动窗口，采样时机（reset 步）是否引入偏差

### Q3：`invalid_seg_frac` 0.224 为什么在文档的 5–10% 区间之外

它是稳定的，所以不是退化。最可能的解释是 `infeasible` 类别（类别权重里有 0.20）的路径被设计成"必含一处结构化挡路事件"，会被计入 `~seg_passable`。需要确认该指标的分母是否应该排除 `infeasible` 类别 —— 否则 G2 的这一条判据本身就写错了。

### Q4：`raw_log_std_mean` 为什么钉在墙上

从 iter ~500 起恒为 0.1035（初值 `log(1.0)=0`，墙 `log_std_max_initial=0.1`）。clamp 饱和 → 梯度为 0 → 只能靠墙推动。若这是有意设计（"移动墙面"），需要在文档里写清楚"raw 值不必低于墙"；否则这是一个会让熵项失效的 bug。

---

## 12. 复现命令

提取全部分析数据（不占用 GPU）：

```bash
/home/yichangfeng/miniforge3/envs/sru-pathaware/bin/python - <<'EOF'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p="IsaacLab/logs/rsl_rl/b2w_navigation_mdpo/2026-11-25_19-40-08_b2w_pathfix_v1"
ea=EventAccumulator(p,size_guidance={'scalars':0}); ea.Reload()
for t in sorted(ea.Tags()['scalars']):
    v=[x.value for x in ea.Scalars(t)]
    print(f"{t}\t{v[-1]:.6g}\t{sum(v)/len(v):.6g}")
EOF
```

训练结束后做同口径评测（**必须先确认训练进程已退出**，显存不够并发）：

```bash
cd IsaacLab && PYTHONPATH=/home/yichangfeng/sru-path-aware/sru-navigation-learning \
./isaaclab.sh -p ../sru-navigation-sim/scripts/eval_policy.py \
  --task Isaac-Nav-MDPO-B2W-Play-v0 \
  --checkpoint /home/yichangfeng/sru-path-aware/IsaacLab/logs/rsl_rl/b2w_navigation_mdpo/2026-11-25_19-40-08_b2w_pathfix_v1/model_4000.pt \
  --episodes 200 --episode_length_s 60 --num_envs 20 --headless \
  --out ../docs/b2w_pathfix_v1_eval_60s.md
```

对照基线（60 s）：**SR 0.580 / median d_path 0.91 m / SPL 0.449**。

---

## 13. 建议的判读顺序（给分析用）

1. 先看第 6 节的分解表 —— 确认"停留/完成"比是否继续上升（这是最早、最灵敏的指标）
2. 再看第 5 节的 `d_path_avg*` —— 它是否仍停在 0.9 m（说明跟踪精度已锁定，问题不在控制改进）
3. 再看 `time_out` 与 `base_contact` —— 回合是靠"活着"还是靠"通过"变长
4. 最后用第 12 节的 eval 拿干净数据，才能回答"改奖励还是改控制"

**在拿到干净 eval 之前，不建议改任何奖励权重** —— 现有全部证据都是训练口径，改动目标函数会让"哪一处坏了"永远说不清。

---

## 附录 A：全部 44 个标量

数据截至 iter 3230。方向列：`↑/↓` 表示最新 350 相对首个 350 的变化超过 ±5%，`→` 表示在 ±5% 以内。

| 标量 | 0–350 | 700–1050 | 最新 350 | 最新点 | 方向 |
|---|---:|---:|---:|---:|---|
| `Episode_Reward/action_rate_l1` | -0.01487 | -0.03945 | -0.03774 | -0.03619 | ↓ |
| `Episode_Reward/backward_movement_penalty` | 0 | 0 | 0 | 0 | → |
| `Episode_Reward/episode_termination` | -0.06607 | -0.06988 | -0.05506 | -0.05929 | ↑ |
| `Episode_Reward/goal_reached_bonus` | 0.1053 | 0.05262 | 0.05893 | 0.04977 | ↓ |
| `Episode_Reward/joint_acc_l2_joint` | -0.002598 | -0.005754 | -0.005686 | -0.005864 | ↓ |
| `Episode_Reward/lateral_movement` | -0.0009328 | -0.002214 | -0.002311 | -0.00223 | ↓ |
| `Episode_Reward/path_deviation_penalty` | -0.002197 | -0.01551 | -0.03648 | -0.03519 | ↓ |
| `Episode_Reward/progress` | 0.0131 | 0.01125 | 0.01326 | 0.01106 | → |
| `Episode_Reward/progress_along_path` | 0.147 | 0.1173 | 0.1493 | 0.1506 | → |
| `Episode_Reward/reach_goal` | 0.0414 | 0.09874 | 0.179 | 0.1802 | ↑ |
| `Episode_Reward/reach_goal_xy_soft` | 0.005145 | 0.01243 | 0.02244 | 0.02252 | ↑ |
| `Episode_Reward/reach_goal_xy_tight` | 0.03825 | 0.09081 | 0.1745 | 0.1744 | ↑ |
| `Episode_Reward/rot_movement` | -1.465e-06 | -3.225e-06 | -3.442e-06 | -3.567e-06 | ↓ |
| `Episode_Reward/time_out_penalty` | -0.002965 | -0.02023 | -0.03256 | -0.03191 | ↓ |
| `Episode_Termination/base_contact` | 0.5141 | 0.569 | 0.4302 | 0.4437 | ↓ |
| `Episode_Termination/early_termination` | 0.386 | 0.1958 | 0.215 | 0.1984 | ↓ |
| `Episode_Termination/large_pitch_angle` | 0.08726 | 0.09659 | 0.1063 | 0.09232 | ↑ |
| `Episode_Termination/terrain_fall` | 0 | 0 | 0 | 0 | → |
| `Episode_Termination/time_out` | 0.02768 | 0.187 | 0.296 | 0.3151 | ↑ |
| `Loss/clip_param` | 0.2 | 0.2 | 0.2 | 0.2 | → |
| `Loss/entropy_coef` | 0.001 | 0.0009701 | 0.0001712 | 0.0001153 | ↓ |
| `Loss/explained_variance` | 0.04142 | 0.8648 | 0.9136 | 0.9171 | ↑ |
| `Loss/kl_divergence` | 0.1405 | 0.2044 | 0.2192 | 0.2338 | ↑ |
| `Loss/learning_rate` | 0.0009944 | 0.0008969 | 0.0002198 | 0.0001802 | ↓ |
| `Loss/surrogate` | -0.008121 | -0.01577 | -0.008411 | -0.008713 | ↑ |
| `Loss/value_function` | 25.31 | 64.96 | 81.9 | 82.32 | ↑ |
| `Metrics/robot_goal/a_star_failed_frac` | 0.03182 | 0.0283 | 0.03199 | 0.05208 | → |
| `Metrics/robot_goal/blocked_episode` | 0.1059 | 0.1265 | 0.1486 | 0.2131 | ↑ |
| `Metrics/robot_goal/distance_to_path` | 0.3768 | 0.5663 | 1.696 | 1.807 | ↑ |
| `Metrics/robot_goal/invalid_seg_frac` | 0.2169 | 0.2181 | 0.2324 | 0.2388 | ↑ |
| `Metrics/robot_goal/path_progress` | 0.09582 | 0.06372 | 0.03821 | 0.02648 | ↓ |
| `Metrics/robot_goal/pinch_frac` | 0.3015 | 0.3019 | 0.3158 | 0.3276 | → |
| `Metrics/robot_goal/relax_frac` | 0.08155 | 0.08487 | 0.08854 | 0.1056 | ↑ |
| `Metrics/robot_goal/success_rate` | 0.3543 | 0.213 | 0.2906 | 0.2816 | ↓ |
| `Metrics/robot_goal/velocity_magnitude` | 0.8337 | 0.9302 | 0.8815 | 0.8303 | ↑ |
| `Metrics/robot_goal/velocity_toward_goal` | 0.2323 | 0.1002 | -0.03608 | -0.07883 | ↓ |
| `Perf/collection time` | 13.87 | 10.14 | 9.65 | 9.653 | ↓ |
| `Perf/learning_time` | 4.261 | 4.095 | 4.079 | 4.164 | → |
| `Perf/total_fps` | 932.2 | 1177 | 1220 | 1211 | ↑ |
| `Policy/log_std_max` | 0.1 | 0.09999 | -0.2647 | -0.3123 | ↓ |
| `Policy/mean_noise_std` | 0.8512 | 1.105 | 0.7678 | 0.7317 | ↓ |
| `Policy/raw_log_std_mean` | -0.1743 | 0.1035 | 0.1035 | 0.1035 | ↑ |
| `Train/mean_episode_length` | 97.67 | 217.5 | 264.3 | 258.6 | ↑ |
| `Train/mean_reward` | 23.43 | 20.57 | 38.42 | 35.84 | ↑ |

---

## 附录 B：本文引用的源码位置

| 内容 | 位置 |
|---|---|
| 奖励项与权重 | `sru-navigation-sim/isaaclab_nav_task/navigation/config/b2w/navigation_env_cfg.py:45-110` |
| `path_deviation_penalty` / `progress_along_path` | `.../mdp/rewards.py:145` / `:165` |
| `reach_goal_xyz`（sigmoid 形函数） | `.../mdp/rewards.py:65` |
| `reach_goal_direct` | `.../mdp/rewards.py:285` |
| 三位一体支付门 `/ _v2_path_metrics` | `.../mdp/navigation/path_generator.py:436` |
| 门限常量 `SYNC_RATE/CONT_TOL/PAY_CAP` | `.../mdp/navigation/path_generator.py:40-49` |
| 三档支付门计算 `_compute_gates` | `.../mdp/navigation/path_generator.py:285` |
| 驻留终止 `at_goal_navigation` | `.../mdp/terminations.py:139` |
| `GOAL_DWELL_TIME_S = 1.5` | `.../mdp/navigation/goal_commands.py:50` |
| `SuccessRateTracker`（10 回合滚动） | `.../mdp/navigation/goal_commands.py:309` |
| 指标采样 `_update_metrics`（reset 步） | `.../mdp/navigation/goal_commands.py:747` |
| 地形课程是否接线 | `.../config/b2w/navigation_env_cfg.py:161` |


