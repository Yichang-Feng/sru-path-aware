# Navigation Reward Reference

本文档整理当前工作区中的导航 reward 函数、默认 B2W/Go2W reward 配置，以及容易混淆的 **reference path 路径进度** 与 **single goal 点跟踪** 的区别。

相关源码：

- `sru-navigation-sim/isaaclab_nav_task/navigation/mdp/rewards.py`
- `sru-navigation-sim/isaaclab_nav_task/navigation/navigation_env_cfg.py`
- `sru-navigation-sim/isaaclab_nav_task/navigation/config/b2w/navigation_env_cfg.py`
- `sru-navigation-sim/isaaclab_nav_task/navigation/config/go2w/navigation_env_cfg.py`
- `sru-navigation-sim/isaaclab_nav_task/navigation/mdp/navigation/path_generator.py`
- `sru-navigation-sim/isaaclab_nav_task/navigation/mdp/navigation/goal_commands.py`

---

## 1. Reward 计算约定

IsaacLab 的 `RewardManager.compute()` 会按：

```text
step_reward = raw_term_value * weight * env.step_dt
```

累加到环境中。当前 B2W 的：

```text
sim.dt       = 0.005 s
decimation   = 40
step_dt      = 0.2 s
```

因此一个 `weight=-30` 的 terminal reward，如果只在 timeout 那一步触发，对 episode return 的贡献约为：

```text
-30 * 0.2 = -6.0
```

日志里的 `Episode_Reward/<term>` 通常按每步尺度展示，所以 timeout 项在触发步约为 `-30`。分析训练时必须区分 **配置 weight** 和 **episode return contribution**。

---

## 2. B2W 当前 reward 配置

> ⚠️ **2026-11-25 起 B2W 换了一套以"跟路径"为主目标的奖励**（path-aware v2）。下表是 2026-11-24
> 标定后的**历史**配置，保留用于解释旧 run 的曲线；当前值见
> `docs/phase0_baseline_report.md`，要点：`progress_along_path` 1.2/m（经 `per_metre` 折算成"每米"）、
> `progress` 0.15/m、`reach_goal` 2.0/步、`goal_reached_bonus` +25 一次性、`path_deviation_penalty`
> −0.15/m、`time_out_penalty` −50（horizon 90 s，故 `Episode_Reward/*` 的还原系数是 **×90**）。
> 下面的权重不要直接拿去和新 run 比较。

B2W 使用官方 `RewardsCfg` 作为基底，并在其上做了一次标定（见 §2.3）。2026-11-24 的配置：

| Term | Function | Weight | 含义 |
|---|---|---:|---|
| `joint_acc_l2_joint` | `joint_acc_l2` | `-1e-7` | 腿部与轮毂关节加速度平方和 |
| `lateral_movement` | `lateral_movement` | `-0.02` | 机身横向速度 L1 惩罚，限制横移（原 -0.1） |
| `rot_movement` | `rot_movement` | `-1e-5` | 机身角速度模长 L2 惩罚 |
| `action_rate_l1` | `action_rate_l1` | `-0.02` | 当前 action 与上一 action 的 L1 差（原 -0.1） |
| `episode_termination` | `is_terminated` | `-50.0` | 非 timeout 的真实终止惩罚；不包括 time_out=True 的终止项 |
| `reach_goal_xy_soft` | `reach_goal_xyz` | `0.25` | 宽尺度 goal 奖励；`sigmoid=2.5`, `T_r=1.0`, `p=0.01`, `flat=False` |
| `reach_goal_xy_tight` | `reach_goal_xyz` | `1.5` | 窄尺度 goal 奖励；`sigmoid=0.25`, `T_r=0.1`, `p=0.01`, `flat=True` |
| `progress_along_path` | `progress_along_path` | `0.05` | 沿 reference path 的弧长进度 shortcut reward；`p=0.95`（percent 尺度，见 §4.2） |
| `backward_movement_penalty` | `backward_movement_penalty` | `-0.0` | 后退惩罚，当前关闭 |
| **`progress`** | `progress_to_goal` | **`1.5`** | **新增：朝最终 goal 的密集单点 progress，`(Δd)/step_dt`** |
| **`reach_goal`** | `reach_goal_direct` | **`10.0`** | **新增：进入 0.5 m goal 圈的稀疏到达奖励** |
| `time_out_penalty` | `episode_time_out` | `-100.0` | B2W-only 真实 60 s horizon timeout 惩罚（原 -30） |

### 2.1 `reach_goal_xyz`

该函数是当前 B2W 的 goal reward 主体。

```text
error = ||goal_in_body||              if flat=False
error = ||goal_in_body[:2]||          if flat=True

raw_reward = 1 / (1 + (error / sigmoid)^2) / T_r
```

然后按以下 mask 生效：

- `random < probability`：随机检查；
- 或 `episode_length_buf > T - required_time_at_goal_in_steps`：episode 末尾窗口；
- 或 `time_at_goal > 0`：已经进入过 goal 区域。

此外：

- `flat=False` 会使用 3D 距离；虽然 term 名字叫 `reach_goal_xy_soft`，当前实现并不是只算 xy。
- `flat=True` 才只使用 xy 误差。
- `T_r` 越小，奖励尺度越大；tight 项的 `T_r=0.1` 比 soft 项大 10 倍。
- `ratio=False` 表示不按行走距离比例缩放。
- `required_time_at_goal_in_steps` 当前为 `4.0 / step_dt`，即 4 秒。

### 2.2 `episode_termination` 与 timeout

当前 `is_terminated` 的定义是：

> 只惩罚 `env.termination_manager.terminated`，不包括 `time_out=True` 的 truncation。

B2W 的 termination 配置中：

- `time_out`：60 s horizon，`time_out=True`
- `early_termination`：到达 goal，`time_out=True`
- `terrain_fall`：掉出地形，`time_out=True`
- `base_contact`：真实终止，`time_out=False`
- `large_pitch_angle`：真实终止，`time_out=False`

所以 `episode_termination` 主要会惩罚 base contact 和 large pitch，**不会惩罚普通 timeout**。新增的 `episode_time_out` 只检查：

```text
episode_length_buf >= max_episode_length
```

因此它只惩罚真正耗尽 60 s horizon 的 episode，不惩罚成功到达或 `terrain_fall`。

---

### 2.3 为什么这么标定（B2W 专用）

日志里的 `Episode_Reward/<term>` 已经除以 `max_episode_length_s = 60`，**×60 才是每 episode 的 return 贡献**。按原配置（`action_rate`/`lateral` 各 -0.1、timeout -30）实测换算：

```text
“动起来”要付：action_rate -5.76 + lateral -0.52 + joint_acc -0.28 ≈ -6.6
“站着不动”只付：timeout -30×0.2 = -6.0         ← 惩罚比“动的成本”还小
```

而 `action_rate` / `lateral` / `joint_acc` 在静止时严格为 0，`reach_goal_xyz` 又被门控到
“最后 4 s + 1% 步”，于是 **静止 return（≈ -5.6）高于任何会动的策略（≈ -13）**，策略站住不动是奖励下的最优解。
标定后：每靠近 goal 1 m 得 +1.5，走 15 m ≈ +22，平滑惩罚降到 ≈ -1.3，静止则吃满 -20。

这套配方（密集 progress + 1/5 平滑惩罚 + 到点奖励）已在本机 Go2W 上 A/B 验证：
官方稀疏奖励 SR 0.08，换成该配方后 **iter 25 达 SR 61%、iter 100 达 95%**。
详见 `docs/b2w_timeout_penalty_analysis_2026-11-23.md`。

## 3. Go2W reward helpers

`rewards.py` 中后半部分是为 Go2W / heading 任务追加的 helper。它们与 B2W 的官方 reward 分开，主要目的是让 Go2W 能获得更明确的 goal 方向、航向和制动信号。

### 3.1 `progress_to_goal`

```text
progress_t = previous_goal_distance - current_goal_distance
return progress_t / step_dt
```

奖励机器人**靠近最终 goal**。这是单点 goal progress，不是沿 reference path 的进度。它使用环境的 `_progress_prev_dist` 状态，episode reset 时归零。

### 3.2 `gated_progress_to_goal`

与 `progress_to_goal` 类似，但对正进度加航向门控：

```text
gating = max(cos(heading_error), 0)^power
```

- 只有机头大致朝向 goal 时才给正 progress；
- 远离 goal 的负进度仍然惩罚；
- goal 附近对 gate 做渐隐，避免接近终点时抖动。
- 默认 `power=2.0`, `distance_threshold=0.5`。

它仍然是单点 goal reward。

### 3.3 `heading_alignment`

计算机身朝向与 goal 方向的余弦：

```text
reward = fade(distance) * cos(heading_error)
```

- 正对 goal 为正；
- 背对 goal 为负；
- 在 goal 0.5 m 内渐隐，避免近距离原地旋转。

### 3.4 `heading_progress_to_goal`

对朝向误差做 potential-based progress：

```text
progress = previous_abs_heading_error - current_abs_heading_error
return fade(distance) * active_mask * progress / step_dt
```

奖励“转向目标”而不是持续保持某个绝对朝向。它使用绝对值航向误差，并有 0.2 rad 的激活窗口。

### 3.5 `target_heading_alignment`

用于最终到达点朝向控制：

```text
heading_reward = 0.5 * (1 + cos(yaw_error))
gate = clamp(1 - distance / distance_threshold, 0, 1)
```

`distance_threshold` 内逐渐开启，`is_adjusting_heading` 为真时 gate 锁为 1。该函数用于让机器狗在 goal 附近对准目标航向。

### 3.6 `target_heading_terminal_outcome`

根据 `goal_cmd.heading_terminal_outcome` 给 terminal outcome：

- outcome > 0.5：成功 bonus；
- outcome < -0.5：失败 penalty；
- 其余：0。

默认 `success_bonus=30`, `fail_penalty=15`。函数内部除以 `step_dt`，抵消 RewardManager 的统一 `dt` 缩放。

### 3.7 `target_yaw_progress`

对最终目标 yaw error 做 potential-based progress：

```text
progress = previous_abs_yaw_error - current_abs_yaw_error
return gate(distance) * progress / step_dt
```

奖励朝目标航向转动，接近 goal 时逐渐开启。

### 3.8 `velocity_at_goal_penalty`

在 goal 半径内惩罚速度：

```text
penalty = at_goal * (linear_speed + 0.5 * angular_speed)
```

用于让机器人到点后停稳，但会同时惩罚角速度，可能限制到点后原地调整朝向。

### 3.9 `velocity_at_goal_heading_penalty`

到点后解耦线速度和角速度：

- 始终惩罚线速度；
- 只有在航向已经对齐时才惩罚角速度；
- 允许机器人先原地转向，再制动。

### 3.10 `velocity_at_goal_linear_penalty`

只惩罚 goal 圆内超过 `speed_deadzone` 的线速度，不惩罚角速度，用于需要到点后继续调整朝向的任务。

### 3.11 `reach_goal_direct`

在 goal 半径内直接给 1，否则 0。它是稀疏的 goal-in-radius bonus，不是 progress reward。

### 3.12 `time_step_cost`

每一步返回 1，配合负权重形成恒定 time cost。它是全局时间成本，不是 timeout terminal penalty。

### 3.13 `lateral_movement_l2`

对横向速度超过 deadzone 的部分做平方惩罚：

```text
excess = max(abs(lateral_velocity) - deadzone, 0)
penalty = excess^2
```

用于抑制高速横移/螃蟹步，同时允许小幅自然侧向运动。

### 3.14 `backward_movement_penalty`

只惩罚负的前向速度：

```text
penalty = clamp(-forward_velocity, 0, 1)
```

用于限制倒车。

---

## 4. Reference path 奖励与单点 goal 奖励的区别

这是最容易混淆的部分。

### 4.1 Reference path 是什么

每个 episode 会通过 PRM + A*/GBFS 生成一条参考路径，路径包含多个 waypoint，最终表现为：

```text
target_path: (N, 4) = (direction_xyz, log_distance)
```

路径可能包含：

- optimal：A* 最优路径；
- mildly non-optimal：带轻微绕路；
- highly non-optimal：强绕路；
- infeasible：路径加噪声后可能局部不可行。

当前 path-aware 配置的训练分布是：

```text
optimal:             0%
mildly non-optimal:  0%
highly non-optimal:  50%
infeasible:          50%
```

这是路径条件观测/训练分布，不等于“必须沿路径走”。

### 4.2 `progress_along_path` 是什么

官方 `MultiPath.get_all_path_metrics()` 计算机器人到整条 polyline 的最近投影，然后得到：

```text
progress_along_path = 已走过的路径弧长 / 总路径长度
```

再取相邻 sim step 的正增量：

```text
delta = max(progress_t - progress_{t-1}, 0)
```

本工作区按论文 eq. 12 使用 **percent** 尺度：

```text
progress ∈ [0, 100]      # 已走弧长 / 总弧长 × 100
threshold = 0.15         # = 每步推进 0.15% 路径
```

单步 0.2 s、1.5 m/s 在 10–40 m 的路径上推进 0.75%–3%，远大于 0.15%，因此**会正常触发**。

> 注意：官方 release 的实现除以 `total_len` 后**没有 ×100**（仍是 `[0,1]` fraction），
> 却沿用了 `0.15` 这个为 percent 设计的阈值 → 单步要走完 15% 的路径才会触发，该项实际**永远为 0**。
> 早期日志里该项恒为 `8e-5`，就是这一单位错配的直接结果。这也是"照 release 训"拿不到论文 shortcut 收益的原因。

`progress_along_path` 的特点是：

- 使用**整条 reference path 的弧长坐标**；
- 可以对任意路径形状计算；
- 奖励的是“沿这条参考路径向前推进”；
- 不是 distance-to-path penalty；
- 不是 waypoint-by-waypoint tracking；
- 不是最终 goal 的径向距离；
- 路径绕远时，沿路径推进也可能奖励绕路；
- 路径不可行时，沿不可行路径推进不代表接近最终目标。

它更像论文中的 shortcut reward 候选信号，而不是严格路径跟随控制器。

### 4.3 单点 goal tracking 是什么

单点 goal reward 只依赖：

- 机器人位置到最终 goal 的向量 `p_t`；
- goal 的朝向/yaw；
- 机器人的速度。

典型函数：

- `reach_goal_xyz`
- `reach_goal_direct`
- `progress_to_goal`
- `gated_progress_to_goal`
- `heading_alignment`
- `heading_progress_to_goal`
- `target_heading_alignment`
- `target_yaw_progress`

它们不知道 reference path 的形状，只知道“目标点在哪里”。因此：

- 单点 goal progress 不能表达“走哪条走廊”；
- reference path progress 能表达“沿给定路线走了多少”，但可能奖励坏路径；
- path observation 可以让策略学会利用路径，但没有 path-following reward 时，策略仍然可以忽略路径；
- 当前 B2W 官方 reward 是 goal-driven 的，reference path 主要通过 observation 条件进入策略，而不是通过强路径跟踪奖励进入。

### 4.4 一句话区别

```text
progress_to_goal:
    "我离最终目标更近了吗？"
    单点、无路线形状、目标导向。

progress_along_path:
    "我在给定 polyline 上的弧长坐标前进了吗？"
    路径形状感知、可奖励沿坏路径推进，不等价于靠近最终目标。

distance-to-path:
    "我离参考线有多远？"
    当前没有作为显式 reward 使用。

waypoint following:
    "我是否朝第 i 个 waypoint 靠近/是否到达？"
    当前没有按 waypoint 逐个给 reward。
```

---

## 5. 当前 B2W 与 Go2W 的配置关系

### B2W

- 使用 `B2WRewardsCfg(RewardsCfg)`；
- 增加 `progress`（`progress_to_goal`，weight `1.5`）与 `reach_goal`（`reach_goal_direct`，weight `10`）；
- 平滑惩罚降到官方 Go2W 的量级：`action_rate_l1` / `lateral_movement` 均为 `-0.02`；
- `time_out_penalty=-100`（只罚真正的 60 s horizon timeout）；
- `progress_along_path` 使用论文 eq. 12 的 percent 尺度 + `0.15`（= 0.15%/步）阈值，正常触发；
- `_resample_command` 在采样到本局 spawn 后重新写 root pose，保证机器人起点与参考路径起点一致。

### Go2W

- 使用 `Go2WRewardsCfg(RewardsCfg)`；
- 增加 `progress`（`gated_progress_to_goal`）；
- 增加 `heading_progress`（`heading_progress_to_goal`）；
- 增加 `heading`（`heading_alignment`）；
- 增加 `reach_goal`（`reach_goal_direct`）；
- 增加 `time_cost`（`time_step_cost`）；
- 使用 `lateral_movement_l2` 替代 B2W 的 `lateral_movement`；
- 显式将 `reach_goal_xy_soft` 和 `reach_goal_xy_tight` 设为 `None`；
- 仍从官方 `RewardsCfg` 继承 `progress_along_path`（weight `0.05`），因此该路径弧长 shortcut 项在 Go2W 配置中仍然存在；
- heading variant 还增加 `target_heading` 和到点制动的解耦惩罚。

Go2W 的 `gated_progress_to_goal` 是单点 goal progress 的航向门控版，不是 `progress_along_path`。

---

## 6. 当前改动的结论

1. B2W 的 reward 已按“移动收益 > 移动成本”重新标定，不再逐字沿用 release；
   `progress_along_path` 修回论文的 percent 尺度，root pose 在 spawn 采样后重写。
2. `progress_along_path` 在修正单位后是可用信号，但它是**路径弧长**进度，不等于靠近最终目标；
   真正驱动 B2W 学会行走的是新增的 `progress`（单点 goal progress）。
3. `time_out_penalty` 只惩罚真正 60 s timeout，不惩罚成功终止或 terrain fall；
   其权重必须大于“静止可省下的平滑惩罚”（约 6.0 return），否则罚站仍是最优解。
4. 单点 goal reward 与 reference path progress 是两种不同几何量：
   - 前者决定“是否靠近终点”；
   - 后者决定“沿给参考路径前进了多少”。
5. 训练出现原地不动时，先做 episode-return 分解（日志值 × 60），而不是只看 clock 上的单项曲线。
