# B2W 奖励重标定：算法改动与训练结果（2026-11-24）

- **改动提交**：`a8e5348`（恢复官方 reward 基线）、`9c5c53b`（B2W 奖励重标定 + 训练记录修正）
- **训练 run**：`IsaacLab/logs/rsl_rl/b2w_navigation_mdpo/2026-11-23_18-45-05_b2w_recalib_1046`
- **配置**：MDPO / 1046 env / seed 60 / `Isaac-Nav-MDPO-B2W-v0`
- **截至本文**：3080 / 15000 iter，14.0 小时（仍在运行）
- **对照 run**：`2026-11-23_17-05-55_official_b2w_mdpo_1046`（官方 reward + `time_out_penalty=-30`）
- **诊断依据**：`docs/reports/b2w_timeout_penalty_analysis_2026-11-23.md`

---

## 1. 摘要

| | 旧 run（17:05，官方 reward + 超时 -30） | 新 run（18:45，重标定） |
|---|---:|---:|
| `Train/mean_reward` | **-13.1** | **+22.3**（窗口峰值 +32.8 @ iter 100–300，单点最高 49.4 @ iter 329） |
| `success_rate` | 0.013 | **0.412** |
| `early_termination`（到点结束） | 0.012 | **0.417** |
| `time_out`（罚站） | 0.770 | **0.033** |
| `velocity_magnitude` | 0.33 m/s | 0.77 m/s |
| `base_contact`（撞/摔） | 0.205 | **0.504** |
| `mean_episode_length` | 264 步 | 97 步 |

**结论：重标定解决了"站着不动"，但没有解决"撞得多"。**

- 失败模式从 **"静止 → 60 s 超时"** 变成了 **"跑起来 → 一半 episode 撞掉"**；
- 成功率从 1.3% 提到 **41%**，且目前仍在缓慢上升（0.36 → 0.41 用了 2000 iter）；
- 但 reward 在 iter 300 之后**从 33 回落到 22**，原因是一个新的失效：**策略噪声 σ 失控（0.71 → 8.5）**，
  它带来的 `action_rate_l1` 代价从 -1.3 涨到 **-12.9 每 episode**，成为现在最大的单项支出。

---

## 2. 算法改动

### 2.1 奖励函数重标定（`config/b2w/navigation_env_cfg.py`，仅 B2W）

动机（定量）：`Episode_Reward/<term>` 被除以 `max_episode_length_s = 60`，所以 **×60 才是每 episode 的 return 贡献**。
按官方 reward 集实测：

```text
"动起来"要付：action_rate_l1 -5.76 + lateral -0.52 + joint_acc -0.28 ≈ -6.6
"站着不动"只付：time_out_penalty (-30 × step_dt 0.2)                = -6.0
```

而 `action_rate` / `lateral` / `joint_acc` 在静止时**严格为 0**，`reach_goal_xyz` 又被门控到
"最后 4 s + 每步 1% 抽查" —— 于是**静止 return（≈ -5.6）高于任何会动的策略（≈ -13）**，
策略站住不动是这套奖励下的最优解。

改动：

| Term | 函数 | 旧 weight | 新 weight | 每 episode 量级 |
|---|---|---:|---:|---|
| `progress`（新增） | `progress_to_goal` | — | **+1.5** | **+1.5 / 前进 1 m** |
| `reach_goal`（新增） | `reach_goal_direct` | — | **+10.0** | +2.0 / 步（0.5 m 圈内，驻留 20 步） |
| `action_rate_l1` | `action_rate_l1` | -0.1 | **-0.02** | ≈ -1.2（σ≈1 时） |
| `lateral_movement` | `lateral_movement` | -0.1 | **-0.02** | ≈ -0.1 |
| `time_out_penalty` | `episode_time_out` | -30 | **-100** | -20（真 60 s 超时） |

`reach_goal_xy_soft/tight`、`episode_termination(-50)`、`joint_acc_l2(-1e-7)`、`progress_along_path(0.05)` 保持官方值。

这套配方**不是新发明的**：它与本机 Go2W 已验证的 scratch 配方一致
（`go2w_navigation_mdpo_phase1/2026-10-22_10-59-56_phase1_scratch`，SR 0.98 —— 而同日使用官方稀疏奖励的
run 只有 SR 0.08）。

### 2.2 路径进度恢复 percent 尺度（`mdp/navigation/path_generator.py`）

```python
progress_along_path = 100.0 * (cum_lens_prev + t_clamped * seg_lens) / total_len
progress = torch.where(progress_diff > 0.15, progress_diff, 0.0)   # 0.15 = 0.15 %/步
```

论文 eq. 12 的 Δp 是 percent；公开 release 除以 `total_len` 后**没有 ×100**，却沿用 `0.15` 阈值
→ 单步要走完 15% 路径，该项**恒为 0**（旧日志恒 `8e-5`）。修正后每步 0.75%–3% ≫ 0.15%，正常触发
（新 run 该项 +0.60/episode）。

### 2.3 root pose 重定位（`mdp/navigation/goal_commands.py`）

`ManagerBasedRLEnv._reset_idx` 的顺序是：`event_manager.apply("reset")` **先**把机器人 teleport 到
`env_origins`，之后 `command_manager.reset → _resample_command` 才采样新 spawn 并重建路径。
也就是说机器人实际站在**上一局**的 spawn，而参考路径从**新** spawn 出发。

修复：采样到本局 spawn 后重新 `write_root_pose_to_sim` / `write_root_velocity_to_sim`，
并用 spawn 计算 `initial_distance_to_goal`。

### 2.4 训练记录修正（`scripts/train.py`）

```python
runner.add_git_repo_to_log(__file__)
runner.add_git_repo_to_log(isaaclab_nav_task.__file__)   # 新增
```

从 official meta-repo 启动时，`__file__` 指向一份**没有被 import** 的 `sru-navigation-sim` 副本，
奖励改动不会进 `git/` 快照，反而会记录一份无关补丁。新 run 的快照已经正确：
`git/sru-path-aware.diff`（27 KB，含全部奖励改动）+ `git/sru-navigation-learning.diff`。

### 2.5 未改动

- `rsl_rl`（官方 submodule）：算法侧一行未动。本次修复不需要它，2026-10-22 的 Go2W 成功 run
  也是纯官方 MDPO。
- PRM / A\* 参数（`k`、padding）与地形采样：保持不动，确保这是单变量实验。

---

## 3. 训练结果

### 3.1 分窗口走势

| window | reward | SR | ep_len | σ | base_contact | early_term | time_out | action_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–50 | 3.8 | 0.063 | 80 | 0.96 | 0.664 | 0.110 | 0.084 | -0.018 |
| 50–100 | 26.2 | 0.211 | 90 | 0.84 | 0.567 | 0.305 | 0.065 | -0.020 |
| **100–300** | **32.8** | 0.329 | 114 | **0.71** | 0.450 | 0.352 | 0.138 | -0.022 |
| 300–600 | 31.8 | 0.366 | 145 | 1.07 | 0.437 | 0.362 | 0.157 | -0.041 |
| 600–1000 | 19.6 | 0.358 | 133 | **3.63** | 0.500 | 0.351 | 0.112 | -0.120 |
| 1000–1500 | 17.2 | 0.360 | 98 | **7.81** | 0.551 | 0.367 | 0.035 | -0.191 |
| 1500–2000 | 22.3 | 0.397 | 94 | 7.82 | 0.520 | 0.404 | 0.031 | -0.189 |
| 2000–2500 | 22.9 | 0.406 | 97 | 7.74 | 0.509 | 0.410 | 0.034 | -0.192 |
| 2500–3076 | 22.3 | **0.412** | 97 | **8.49** | 0.504 | **0.417** | 0.033 | **-0.212** |

三个阶段：

1. **0–50 iter：破冰。** 策略开始移动，`velocity_magnitude` 0.64 → 0.73，`time_out` 从 0.20 掉到 0.08。
2. **50–300 iter：快速上升。** SR 0.06 → 0.33，reward 3.8 → 32.8，`early_termination` 0.11 → 0.35。
3. **300 iter 之后：plateau + σ 失控。** SR 只从 0.37 磨到 0.41，reward 反而回落，σ 从 1.1 一路涨到 8.5。

### 3.2 Episode-return 分解（×60，iter 3080 vs 旧 run）

| term | 旧 run（17:05） | 新 run（18:45 @3080） |
|---|---:|---:|
| `progress` | — | **+13.7** |
| `reach_goal` | — | **+13.8** |
| `reach_goal_xy_tight` | +0.31 | **+11.2** |
| `reach_goal_xy_soft` | +0.06 | +0.43 |
| `progress_along_path` | ≈ 0 | +0.60 |
| `action_rate_l1` | -5.76 | **-12.92** |
| `episode_termination` | -2.19 | -5.62 |
| `joint_acc_l2` | -0.28 | -0.41 |
| `lateral_movement` | -0.52 | -0.11 |
| `time_out_penalty` | -4.59 | -0.58 |
| **合计** | **≈ -13.0** | **≈ +20 ~ +23** |

两个值得注意的点：

- `reach_goal_xy_tight` 从 +0.31 变成 +11.2 —— 说明机器人**真的到得了 0.5 m 圈内**了（以前几乎到不了）。
- `action_rate_l1` 从 -5.76 变成 -12.92 —— **不是**权重调大造成的（权重是从 -0.1 降到 -0.02 的），
  而是 σ 变大后 `|a_t − a_{t−1}|` 本身变大了（σ=8.5 时 raw action 的逐维差分 ≈ 10，3 维合计 ≈ 33/步）。

### 3.3 成功率的分母

`success_rate` 0.412 ≈ `early_termination` 0.417，即**到点判定与成功率一致**。
剩余约 0.50 的 episode 以 `base_contact` 结束（撞障/摔倒），0.073 以 `large_pitch_angle` 结束，
只有 0.033 超时。**当前唯一的大失败模式是碰撞。**

---

## 4. 归因

### 4.1 起作用的

- **密集 `progress` + 到点奖励**：给了"往哪走"的连续梯度，这是从 0 stem 到 0.33 的关键。
  Go2W 的先例（iter 25 到 SR 61%）在这里慢一些（iter 50 才到 0.21），因为 B2W 是双足轮式、
  场景更大、且 σ 后来失控。
- **`time_out_penalty` -100**：超时率从 0.77 直接压到 0.03，罚站解被彻底消除。
  不过按 §3.1 看，真正驱动行为的是 `progress`（机会成本），`-100` 更像保险。
- **`progress_along_path` 修复**：现在有 +0.60/episode 的真实信号（虽然小）。

### 4.2 当前瓶颈：σ 失控（首要）

官方 `ActorCriticSRU.update_distribution` 是

```python
std = self.log_std.exp().expand_as(mean)      # 无任何上下界
```

配合 `entropy_coef = 0.00375`（熵奖励持续把 `log_std` 往上推）和 `schedule="exponential"`（LR 按时间衰减，
不按 KL 自适应），本次 run 里 σ 从 0.71 涨到 8.49，`log_std ≈ 2.14`。

后果：

1. **动作退化成 bang-bang**：`tanh(μ + N(0, 8.5²))` 几乎总是 ±1，靠 `randomize_low_pass_filter_alpha`
   （α ∈ [0.4, 0.9]）平滑成速度指令。这解释了 `base_contact` 长期卡在 0.50 —— 一半的 episode
   是被随机的饱和指令撞掉的。
2. **`action_rate_l1` 线性放大**：-1.3（σ≈0.7）→ -12.9（σ≈8.5）每 episode，成为最大单项支出，
   把 reward 从 33 拉回 22。

对照：旧 run（官方稀疏奖励）σ 会**塌到 0.35 以下**（另一个方向的失控）；Go2W 2026-10-22 那个 98% 的
run 里 σ 也涨到过 20。**这套代码对 σ 没有任何约束，两个方向都见过失控。**

### 4.3 次要瓶颈：critic 跟不上

- `Loss/value_function` 从 10 → **93**，`Loss/kl_divergence` 从 0.065 → **0.34**（是 `desired_kl = 0.01` 的 34 倍）。
- 原因：`value_loss_coef = 0.02` 偏小、`value_clip_param = 0.2` 限制了 critic 单次吸收幅度，
  而奖励量级比之前大 3–5 倍（单步可达 +3），critic 需要重新标定。
- `schedule="exponential"` 不响应 KL（只有 `adaptive` 才会按 KL 调 LR，而官方 MDPO 的 `adaptive`
  等于固定 LR），所以高 KL 没有任何抑制。

### 4.4 与 A\* warning 的关系

新 run 的 console 里 `[GlobalPath] WARNING: A* failed to find a path.` 明显变多，原因是
**reset 次数变多**（episode 从 264 步缩到 ~97 步、`base_contact` 从 0.20 涨到 0.50），
而 `find_path()` 每个 env 每次 reset 跑一次。A\* 的输入（spawn xy / goal xy / tile_id / 高度图）
没有被本次改动触碰，因此**每次尝试的失败率不变**。

影响很小：A\* 失败时兜底成 `start → goal` 直线（15 点），论文训练分布里本来就有 50% 的
`infeasible` 路径，策略被训练成可以忽略坏 hint。代价是启动/采集变慢
（`Perf/total_fps` 1255 → 968），随策略少撞会自行缓解。

---

## 5. 下一步建议（按优先级）

> **更新（2026-11-24）**：#1 与 #2 已实施（commit `304d58b`）。
> `ActorCriticSRU` 的 `log_std` 现在被限制在 `[-2.5, 0.2]`（std ∈ [0.082, 1.22]），
> 由 `RslRlPpoActorCriticCfg.log_std_min/log_std_max` 配置并会写进 `agent.yaml`；
> B2W MDPO 的 `entropy_coef` 从 0.00375 降到 0.001。以下是原始判断，保留作为依据。

| # | 改动 | 位置 | 预期 |
|---|---|---|---|
| 1 | **给 `log_std` 加上界**，如 `torch.clamp(log_std, -2.5, 0.2)`（σ ≤ 1.22） | `rsl_rl/modules/actor_critic_sru.py`（官方 submodule，一行） | σ 收敛，`action_rate` 回到 ≈ -2，`base_contact` 有望从 0.50 降下来 |
| 2 | `entropy_coef` 0.00375 → 0.001（或 0） | `config/b2w/agents/rsl_rl_cfg.py`（纯配置，不动外部 repo） | 减弱把 σ 往上推的力，可与 #1 叠加 |
| 3 | 重新观察 `base_contact` | — | 若 σ 收敛后仍 > 0.4，再考虑碰撞 shaping（例如 `backward_movement_penalty` 或近距离减速项） |
| 4 | 暂不动 PRM / A\* / 地形采样 | — | 保持单变量 |
| 5 | 可选：把 `value_loss_coef` 0.02 → 0.05 | `rsl_rl_cfg.py` | critic 更快跟上更大的奖励量级 |

**关于 #1 的取舍**：它需要改 official `rsl_rl`，会让"官方基线"多一处补丁。但当前证据（σ 8.5、
`action_rate` 成为最大支出、一半 episode 撞掉）表明这不是可选项 —— 除非走 #2 的纯配置路线。
建议先试 #1 + #2 一起，因为 #2 单独大概率不够（熵项只贡献 `-entropy_coef × 1` 的 log_std 梯度）。

**是否重启**：`train.py` 没有 resume 支持，所以只能是 scratch 重跑。当前 run 已经 14 h / 3080 iter，
SR 仍在以 ~0.03 / 1000 iter 的速度上升，reward 已经走平 —— 继续跑 12000 iter 预计到不了 0.6 以上。
建议**尽早重启**，避免再花 3 天验证一个已知受 σ 限制的配置。

---

## 6. 复现信息

```bash
# 本次训练命令
cd /home/yichangfeng/sru-path-aware/IsaacLab
./isaaclab.sh -p /home/yichangfeng/sru-path-aware/sru-navigation-sim/scripts/train.py \
    --task Isaac-Nav-MDPO-B2W-v0 --num_envs 1046 --max_iterations 15000 \
    --seed 60 --run_name b2w_recalib_1046 --headless
```

- 代码版本：`9c5c53b`（`git log -1`），run 的 `git/` 内已存完整 diff
- 配置快照：`run/params/env.yaml`、`run/params/agent.yaml`
- 曲线解析：`python3 skills/sru-training-analysis/scripts/analyze_rsl_rl_events.py <run_dir> --steps ... --windows ...`
- 已知限制：单 seed（60）；Go2W 的可比 run 用的是不同机器人/地形，只作为量级参考。
