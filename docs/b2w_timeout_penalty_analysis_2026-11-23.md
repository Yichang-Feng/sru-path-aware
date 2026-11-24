# B2W 加了 timeout penalty 之后仍然“站着不动”的原因分析

分析对象：`b2w_navigation_mdpo` 的三次 run + 同仓库 Go2W 的历史 A/B。
所有数字来自 `IsaacLab/logs/rsl_rl/**/events.out.tfevents.*`，用
`skills/sru-training-analysis/scripts/analyze_rsl_rl_events.py` 解析。

---

## 0. 结论先行

1. **新增的 `time_out_penalty` 实现是对的，但标定反了。**
   `-30 × step_dt(0.2) = -6.0` 的 episode 代价，恰好**小于“动起来”要付的平滑惩罚**
   （`action_rate_l1` ≈ -5.8 + `lateral_movement` ≈ -0.5 ≈ -6.3）。
   而 `action_rate` / `lateral` / `joint_acc` 在**原地不动时严格为 0**。
   所以“站着不动”的 return 比“动起来”高 6～7 个单位 —— 策略正在**正确地**最大化奖励。
2. **这不是 timeout penalty 的问题。** 同一套官方 reward 在**没有** timeout penalty 的
   B2W run 里（11-21 PPO 13.6k iter、10-19 PPO 70k iter）结局完全一样：站住不动。
   官方 reward 集在 scratch 训练下**根本起不来**，timeout penalty 只是在这个塌缩里加了一个常数。
3. **你的仓库里已经有解药，而且验证过两次。**
   Go2W 用官方稀疏 reward 的 scratch run：SR 8%；
   同一环境换成“密集单点 goal progress + 1/5 的平滑惩罚”：**iter 25 就到 SR 61%，iter 100 到 95%**。
4. 最小改动 = **加一项 `progress_to_goal`（已实现，Go2W 在用）+ 把 `action_rate`/`lateral` 调小 +
   把 timeout 权重提到 -100**。三处配置，不需要改任何函数。

---

## 1. 证据：奖励分解（把日志换算成 episode return）

`RewardManager.reset()` 里 `Episode_Reward/<term> = episode_sum / max_episode_length_s(60)`，
所以**日志值 × 60 = 该项在一个 episode 里的 return 贡献**。这是本次分析最容易被误读的一步
（`time_out_penalty` 日志里是 -0.08，看着“很小”，其实是 -4.8）。

### 1.1 加了 timeout penalty 的 run（`2026-11-23_17-05-55`，窗口 250–300）

| term | weight | 日志值 | ×60 = episode 贡献 | 静止时 |
|---|---:|---:|---:|---:|
| `action_rate_l1` | -0.1 | -0.0960 | **-5.76** | **0** |
| `lateral_movement` | -0.1 | -0.0087 | -0.52 | **0** |
| `joint_acc_l2_joint` | -1e-7 | -0.0046 | -0.28 | **0** |
| `episode_termination` | -50 | -0.0364 | -2.19 | 0（不摔就不触发） |
| `time_out_penalty` | -30 | -0.0765 | **-4.59** | -6.0（100% 超时） |
| `reach_goal_xy_soft` | +0.25 | +0.0010 | +0.06 | ≈ 相同 |
| `reach_goal_xy_tight` | +1.5 | +0.0051 | +0.31 | ≈ 相同 |
| `progress_along_path` | +0.05 | 8e-5 | ≈ 0 | 0 |
| **合计** | | | **-12.97** | **≈ -5.6** |

（合计 -12.97 vs 日志 `Train/mean_reward` = -13.08，闭环。）

对照无 timeout penalty 的 `2026-11-23_14-28-51`（窗口 250–300）：
`-5.92 -0.56 -0.45 -1.51 +0.08 +0.49 = -7.87`（日志 -7.95）✓

### 1.2 三个直接结论

- **静止策略的 return（≈ -5.6）比现行策略（-13）高 7 个单位。**
  加 timeout penalty 之前也一样：静止 ≈ -5.1 vs 实际 -7.9。
- `time_out_penalty` 的日志值从 window(1-50) 的 -0.047 “下降”到 250-300 的 -0.077，
  **不是惩罚变弱，而是超时占比从 40% 涨到 77%**：`Episode_Termination/time_out` 0.41→0.77，
  `base_contact` 0.43→0.20，`mean_episode_length` 174→270。
  策略学到的是“别摔”（-50），换成了“罚站”（-6）。在这套奖励下这是**最优解**。
- `progress_along_path` 恒为 8e-5，等于**这一项不存在**（见 §3.2）。

---

## 2. 为什么官方/论文能跑起来

论文（Haro et al. 2026）的奖励是**纯 goal-driven**，没有任何 timeout penalty：

```
r_t = α1·r_task + α2·r_reg + α3·r_pen + α4·r_shortcut
```

能跑起来靠的是各项之间的**相对标定**，而不是某一项特别大：

1. **`r_task` 的门控设计**（eq. 9）：`1(t > T_max − T_r  ∨  rand < δ_check) / (1 + ‖p_t‖/σ)`。
   每个 episode 只在**最后 4 s + 1%/步的随机抽查**（期望 ~3 次）生效。
   这是刻意的“deadline 语义”：主信号是“60 s 内到达”，而不是全程稠密 shaping。
2. **`r_reg` 用 momentum 滤波后的 action**（eq. 10）：`a_m,t = λa_{t-1}^m + (1-λ)a_t`。
   它的作用是**让高频探索噪声不被全额惩罚**。改成直接 `|a_t − a_{t-1}|` 会把探索一起掐死。
3. **`r_pen` 只罚碰撞和倾角**，不含“动得多就罚”的项。
4. 论文/官方是 **1046 env × 39.5 h / 单张 4090**，并且论文自己写明
   “早期保守起步（先学会别摔）是这套稀疏奖励的固有行为”——
   它的 bootstrap 依赖**足够长的训练 + 足够强的探索**，一旦探索先死（见 §2.2），就再也回不来。

> 注意：**公开 release 不是论文内部代码的逐字复现。** release 的 shortcut reward
> 用 `[0,1]` 的 fraction 去比 `0.15` 阈值（= 单步要走完 15% 路径），实际上**永远不触发**；
> 论文 eq. 12 的 Δp 是 percent，ε=0.15 指 0.15%/步。所以“照 release 训”拿不到论文里的 shortcut 收益。

### 2.1 论文能成而这里不能成的关键：奖励的各项量级

论文侧“朝目标走 1 m 的收益”必须**明显大于**“走 1 m 付掉的平滑惩罚”。
在当前配置里正好相反：走 1 m 的 `reach_goal` 收益 ≈ 0（被门控），
而 `action_rate + lateral` 每 episode 固定收 -6.3，静止即免单。

### 2.2 另一个放大器：熵先死

`2026-11-21_16-24-14` B2W PPO（13.6k iter）的完整轨迹：

| iter | SR | velocity_mag | mean_noise_std |
|---:|---:|---:|---:|
| 50 | 1.3% | 0.60 | 0.73 |
| 119 | **5.2%** | — | — |
| 500 | 2.0% | 0.25 | 0.23 |
| 1500 | 0.7% | 0.14 | 0.086 |
| 5000 | 0.9% | 0.062 | **0.020** |
| 13596 | 0.3% | 0.088 | **0.013** |

先靠探索碰到过 goal（5.2%），随后 σ 塌到 0.01，**再也探索不到**，
剩下唯一能优化的方向就是“消掉所有 penalty”→ 站住。
官方 `ActorCriticSRU` 对 `log_std` **无下界**（你自己的 fork 里在 `update_distribution` 加过
`clamp(log_std, -2.5, 0.2)`，但当前 `rsl_rl` 指向的是官方版本，没有这个 clamp）。

---

## 3. 为什么“加了 timeout penalty”压不住

### 3.1 标定问题（主因）

要让“动”比“不动”划算，需要

```
|timeout_weight| × step_dt  >  action_rate_cost + lateral_cost + margin
           30 × 0.2 = 6.0     vs         5.76 + 0.52     = 6.28
```

**-30 刚好差一点点不够**，而且没有余量留给“学习过程中动作变得更粗糙、惩罚变大”的阶段。
再叠加 `reach_goal` 被门控（不动也照样拿最后 4 s 和 1% 抽查的那点分），
策略没有任何理由动。

### 3.2 `progress_along_path` 实际是 0（第二个“没有前进信号”的原因）

当前 `path_generator.py` 返回的是 **fraction ∈ [0,1]**，而阈值是 `> 0.15`：

```python
progress_along_path = (cum_lens_prev + t_clamped * seg_lens) / total_len   # 0..1
progress = torch.where(progress_diff > 0.15, progress_diff, 0.0)          # 单步要推进 15%
```

B2W 一步 0.2 s、1.5 m/s = 0.3 m；路径 10–40 m → 单步 0.75%–3% ≪ 15%。**永远不触发**
（日志恒 8e-5 就是证据）。所以现在**没有任何一项奖励与“离终点更近”成连续正比**。

> 你在 `309b176` 里改成 `100.0 * ...`（percent，0.15% 阈值）是**对的方向**，
> 但 working tree 把它回退成了 fraction 版本。回退后这一项重新变成死项。

### 3.3 同仓库的 A/B：换密集 progress 就活，不换就死

同一个 `sru-navigation-learning`、同一个 MDPO、同样 1024/1046 env，只有 reward 不同：

| run | 奖励差异 | 最终 SR |
|---|---|---:|
| `go2w…/2026-10-22_15-58-54_mdpo_phase1_scratch` | 官方：soft/tight + ar **-0.1** + lat **-0.1** | **0.080** |
| `go2w…/2026-10-22_10-59-56_phase1_scratch` | `progress_to_goal` 1.5 + `reach_goal_direct` 10 + ar **-0.02** + lat **-0.02** | **0.981** |
| `go2w…/2026-10-22_16-56-11_mdpo_phase1_dense_optb` | 同上 + backward | **0.980** |
| `go2w…/2026-10-26_09-32-33_…gated_head_first` | 门控版 progress | **0.991** |

成功样本的上升速度（`2026-10-22_10-59-56`）：

| iter | SR | mean_reward | time_out |
|---:|---:|---:|---:|
| 10 | 2.8% | -10.5 | 0.59 |
| **25** | **61.2%** | **+52.9** | 0.004 |
| 50 | 86.0% | +61.5 | 0 |
| 100 | 94.6% | +58.8 | 0 |

**25 个 iteration 解决任务**，对比现在的 280 iter 还在 -13。差别只在于
“每走近目标 1 m 给 +1.5” 与 “每 episode 固定付 -6.3 平滑惩罚” 谁大谁小。

---

## 4. 最小化改进

### 4.1 一级（1 行，先止血）

`config/b2w/navigation_env_cfg.py`：

```python
time_out_penalty = RewTerm(func=mdp.episode_time_out, weight=-100.0)   # -30 → -100
```

-30 → -100 意味着超时 = **-20** episode return，静止 ≈ -19.6，动起来 ≈ -7 ~ +10。
差距 12 个单位，足以压过探索噪声。**但这一级只解决“不走会亏”，不提供“往哪走”的密集梯度**，
收敛会比 §4.2 慢很多，且依赖 critic 通过 γ=0.999 把终端惩罚回传 300 步。

### 4.2 二级（推荐，3 处配置，已验证配方）

```python
@configclass
class B2WRewardsCfg(RewardsCfg):
    # 1) 密集单点 goal progress（已在 rewards.py 实现，Go2W 正在用）
    progress = RewTerm(
        func=mdp.progress_to_goal,
        weight=1.5,
        params={"command_name": "robot_goal"},
    )
    # 2) 到点直接奖励（0.5 m 内），给终端一个干净的正信号
    reach_goal = RewTerm(
        func=mdp.reach_goal_direct,
        weight=10.0,
        params={"command_name": "robot_goal", "distance_threshold": 0.5},
    )
    # 3) 把“移动成本”压到奖励之下（沿用官方 Go2W 的 -0.02）
    action_rate_l1 = RewTerm(func=mdp.action_rate_l1, weight=-0.02)
    lateral_movement = RewTerm(func=mdp.lateral_movement, weight=-0.02)
    # 4) 超时惩罚
    time_out_penalty = RewTerm(func=mdp.episode_time_out, weight=-100.0)
```

量级校验：`progress_to_goal` 返回 `Δd / step_dt`，所以 `weight × dt = 1.5`，
即**每靠近目标 1 m 得 +1.5**；走 15 m ≈ +22，扣掉平滑惩罚 ≈ -1.3 → 净 +20。
静止 = -20。**方向明确、量级压倒性**。

可选：`reach_goal_xy_soft` / `reach_goal_xy_tight` 保留或置 `None` 都行；
`progress_along_path`（死项）可顺手修回 percent 版，但**它不是必需的**——
Go2W 的成功样本里它的作用也是 0。

### 4.3 三级（可选，若仍见 σ 塌缩）

`rsl_rl/modules/actor_critic_sru.py::update_distribution` 给 `log_std` 加下界
（你的 fork 里已有 `clamp(-2.5, 0.2)`）：
没有下界时，稀疏奖励场景下 σ 会一路掉到 0.01，探索死掉后再也回不来（§2.2）。

---

## 5. 验证标准（跑起来 1 小时内就能判断）

看这几个量，不要只看 `Train/mean_reward`：

| 指标 | 期望（50 iter 内） | 说明 |
|---|---|---|
| `Metrics/robot_goal/velocity_magnitude` | **> 0.4 且不下降** | 最灵敏的“是不是在走” |
| `Episode_Reward/progress` | 明显为正（> 5） | 密集梯度生效 |
| `Episode_Termination/time_out` | → 0（被 `early_termination` 接管） | 不再罚站 |
| `Metrics/robot_goal/success_rate` | > 0.5 | Go2W 先例是 iter 25 到 61% |
| `Policy/mean_noise_std` | > 0.1 | 低于 0.1 说明熵在死（§4.3） |

**止损线**：到 iter 50 仍是 `SR = 0` 且 `velocity_magnitude` 在下降 → 停，回来查配置是否生效
（`params/env.yaml` 里的 `rewards` 段是最可靠的记录）。

---

## 6. 顺手记下的两个坑

1. **`progress_to_goal` 和 `gated_progress_to_goal` 共用 `env._progress_prev_dist`。**
   两者同时启用会互相污染状态，只能二选一。
2. **`goal_commands.py` 的 root pose 重定位（`309b176` 加的）被回退了。**
   现在 `_reset_idx` 的顺序是：事件先把机器人 teleport 到 `env_origins`（**上一局**的 spawn），
   之后 `command_manager.reset → _resample_command` 才把 `env_origins` 改成**新 spawn** 并重建路径。
   于是本局机器人起点与参考路径起点不一致（同一 terrain tile 内，可能差几十米）。
   这不会直接导致站住不动（goal 观测是独立算的），但它让 path 观测系统性错位，
   对一个 path-conditioned 策略是实打实的输入噪声。**建议一并恢复**。

---

## 7. 已实施（2026-11-23）

按 §4.2 落地，改动三处：

| 文件 | 改动 |
|---|---|
| `config/b2w/navigation_env_cfg.py` | `B2WRewardsCfg` 新增 `progress`(1.5)、`reach_goal`(10)；`action_rate_l1`/`lateral_movement` → -0.02；`time_out_penalty` -30 → -100 |
| `mdp/navigation/goal_commands.py` | 恢复 `309b176` 的 root pose 重定位 + `initial_distance_to_goal` 用 spawn 计算 |
| `mdp/navigation/path_generator.py` | 恢复 percent 尺度（`100.0 * ...`），使 0.15 阈值成为 0.15%/步 |
| `scripts/train.py`（本地 + official 各一份） | 追加 `runner.add_git_repo_to_log(isaaclab_nav_task.__file__)`，让 run 的 `git/` 快照记录**实际 import 的 sim 代码** |

注意：`Go2WNavigationEnvCfg_Paper` 继承 `B2WNavigationEnvCfg`，因此**也会继承这套奖励**；
`Go2WRewardsCfg`（部署用）继承的是基类 `RewardsCfg`，不受影响。
`rsl_rl`（官方 submodule）未改动，§4.3 的 `log_std` 下界仍待定。
