# B2W Path-Aware 导航项目交接文档

> 快照时间：2026-11-25 11:00（训练仍在进行，iter ≈ 6020 / 15000）
> 目的：把项目结构、MDP、奖励、路径管线、算法、当前训练结果，以及真机前 play 中观察到的问题，
> 整合成一份可独立阅读的文档，交给另一个 AI 做规划设计 / 归因分析。
> 本文所有路径为绝对路径；代码引用格式 `文件:行号`。

---

## 0. 阅读须知：事实等级

本文用三种标记，请勿混淆：

- **[事实]**：直接读代码 / 配置文件 / 训练日志得到，可复核。
- **[推导]**：由事实计算或推断，计算方法写在旁边。
- **[假设]**：尚无直接证据的解释，附验证方法。

需要特别警惕的度量口径问题（后文多次用到）：

- `Metrics/robot_goal/success_rate`：每个 env 最近 10 个 episode 的成功率滚动平均。
  成功在 episode reset 时判定：`at_goal` 终止 → 1；`base_contact` / `large_pitch` → 0；
  60 s 超时 → 若本 episode 内曾进入 0.5 m 半径（`time_at_goal > 0`）则算 1，否则 0。
- `Metrics/robot_goal/velocity_toward_goal`：**瞬时**世界系速度在"指向目标"方向上的投影，
  按日志窗口求平均。接近目标时该值天然趋 0，不能当作"是否朝目标运动"的直接证据。
- `Episode_Reward/<term>`：RewardManager 记录的是 `sum(term × weight × dt) / 60 s`
  （`IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py:116-122`），
  因此 **该值 × 60 = 每 episode 的该项贡献**。本文第 4.1 节按此还原。
- `Train/mean_reward` ≈ 各 `Episode_Reward` 之和 × 60（已验证：39.5 ≈ 分解总和 39.6）。

---

## 1. 项目总览

### 1.1 仓库与组件 [事实]

工作区：`/home/yichangfeng/sru-path-aware`（单仓库快照，见 `README.md`）

| 目录 | 内容 | 上游 |
|---|---|---|
| `sru-navigation-sim` | IsaacLab 任务：环境、奖励、终止、地形、PRM/A* 路径、`scripts/train.py`、`scripts/play.py` | `leggedrobotics/sru-navigation-sim` `feature/path-aware-rl` @ `da6d5d6` |
| `sru-navigation-learning` | 训练框架 `rsl_rl`：MDPO、ActorCriticSRU、PathEncoder、runner | `leggedrobotics/sru-navigation-learning` `feature/path-aware-rl` @ `e71df2b`（有本地未提交改动，见 1.4） |
| `IsaacLab` | Isaac Sim 4.5.0 / IsaacLab 2.3.2；`IsaacLab/source/isaaclab_nav_task` 是指向 `sru-navigation-sim` 的 symlink | 本地 fork @ `b0542fe2d` |
| `docs/` | 既往分析：`b2w_reward_recalibration_2026-11-24.md`、`b2w_timeout_penalty_analysis_2026-11-23.md`、`go2w_training_analysis_2026-11-24.md`、`navigation_rewards.md`；两篇论文 PDF | — |

环境：conda env `sru-pathaware`（`/home/yichangfeng/miniforge3/envs/sru-pathaware`），Python 3.10、PyTorch 2.7.0+cu126。

### 1.2 任务入口 [事实]

`/home/yichangfeng/sru-path-aware/sru-navigation-sim/isaaclab_nav_task/navigation/config/b2w/__init__.py`：

| Gym ID | env cfg | agent cfg |
|---|---|---|
| `Isaac-Nav-MDPO-B2W-v0` | `B2WNavigationEnvCfg`（训练，1046 envs 由 CLI 指定） | `B2WNavMDPORunnerCfg` |
| `Isaac-Nav-MDPO-B2W-Play-v0` | `B2WNavigationEnvCfg_PLAY`（20 envs，2×2 地形，关掉观测噪声/push） | 同上 |
| `Isaac-Nav-PPO-B2W-v0` / `-Play-v0` | 同上 | `B2WNavPPORunnerCfg`（单策略 PPO，备用） |

启动脚本：`train_b2w_pathaware.sh` / `play_b2w_pathaware.sh`（根目录，封装 `IsaacLab/isaaclab.sh -p .../scripts/train.py|play.py`）。

### 1.3 机器人 [事实]

B2W：双足 + 轮（`sru-navigation-sim/isaaclab_nav_task/navigation/assets/b2w.py`），高层策略输出 SE(2) 速度指令，
由冻结的低层 locomotion 策略执行（`assets/data/Policies/locomotion/b2w/policy_b2w_new_2.pt`，50 Hz）。
用户已确认：**现有 locomotion 足以上下楼梯**，问题不在低层。

### 1.4 相对官方的本地改动 [事实]

| 改动 | 位置 | 说明 |
|---|---|---|
| B2W 奖励重标定（2026-11-24，`9c5c53b`） | `sru-navigation-sim/.../config/b2w/navigation_env_cfg.py:23-65` | 新增 `progress=1.5`、`reach_goal=10.0`、`time_out_penalty=-100`；`lateral/action_rate` 从 -0.1 降到 -0.02。目的：消除"站着不动最优"。见第 4 节 |
| `log_std` 上下界 | `sru-navigation-learning/rsl_rl/modules/actor_critic_sru.py`（当前 run 目录 `git/sru-navigation-learning.diff`） | `clamp(log_std, -2.5, 0.2)`，配置在 `config/b2w/agents/rsl_rl_cfg.py:43-45` |
| `entropy_coef` 0.00375 → 0.001 | 同上 `:53-55` | 减弱熵项对 σ 的推力 |
| `on_policy_runner.py` 对 `get_observations()` 返回 dict 的兼容 | `sru-navigation-learning/rsl_rl/runners/on_policy_runner.py` | 兼容新版 IsaacLab API |
| `train.py` 增加 `--resume` | `sru-navigation-sim/scripts/train.py`（commit `c22d396`） | 可从 checkpoint 续训 |
| play 中绘制 A* 路径 | commit `7654205` | 青色折线 = 策略实际收到的路径 |

⚠️ 训练时 `sru-navigation-learning` 处于 detached HEAD 且有未提交改动；**精确的代码版本以 run 目录下的
`git/sru-navigation-learning.diff` 为准**。

---

## 2. 任务与 MDP

### 2.1 地形 [事实]

`/home/yichangfeng/sru-path-aware/sru-navigation-sim/isaaclab_nav_task/terrains/maze_config.py`：

- 每块 tile 30 m × 30 m，`num_rows=6 × num_cols=30 = 180` 块（与论文一致）。
- 高度场分辨率：水平 0.1 m/格，垂直 0.005 m/单位。
- 子地形比例：`maze` 30%、`non_maze` 20%、`stairs` 30%、`pits` 20%。
- 高度常量（`terrains/terrain_constants.py:51-75`）：平台 +1.0 m，墙 +1.5 m，坑 −1.5 m。
  → **楼梯需要爬升约 1 m**。
- 训练：`difficulty_range=(0.5, 1.0)`，`terrain_generator.curriculum=False`（**没有地形课程**），
  `max_init_terrain_level=10`（初始化时随机分布在所有行）。
- Play：2×2 tile，`max_init_terrain_level=None`。

### 2.2 观测 [事实]

`navigation_env_cfg.py:169-259`，布局 `[proprio, path, image]`（`actor_critic_sru.py:318-330`）：

**Actor（2636 维）**

| 项 | 维度 | 说明 |
|---|---|---|
| `base_lin_vel` | 3 | 带 U(−0.2,0.2) 噪声，带延迟 |
| `base_ang_vel` | 3 | U(−0.1,0.1) |
| `projected_gravity` | 3 | U(−0.1,0.1) |
| `last_action` | 3 | — |
| `target_position` | 4 | **直连目标的单位方向 (3) + log(距离+1) (1)**；10% 概率叠加平移 ≤0.5 m / 旋转 0.1 rad |
| `target_path` | 60 | **15 个路径点 × (单位方向 xyz, 归一化 log 距离)**；10% 概率叠加路径噪声 |
| `depth_image` | 2560 | 64×40 深度 → VAE/CNN 特征 64×5×8；带噪声与延迟 |

**Critic**：额外 `time_normalized`(1) 与 `height_scan`(64×7×7 = 3136)，共 5773 维。

**关键结构事实**：路径在 actor 中先由 `PathEncoder` 压成 **32 维**，
再与 512 维 SRU 隐状态拼接（`actor_critic_sru.py:174`）。
即路径信息带宽远小于本体 + 深度（约 5%），且**不进入 RNN**（论文亦如此，理由是路径在一个 episode 内固定）。

### 2.3 动作与低层控制 [事实]

`sru-navigation-sim/.../mdp/navigation/actions/navigation_se2_actions.py`：

- 策略输出 3 维 `[vx, vy, ω]`（高斯策略，`tanh` 压缩到 [−1,1]）。
- `use_raw_actions=True` → 跳过 affine，仍做 `tanh`；再乘 `policy_scaling=[1.5, 1.0, 1.0]`
  → 速度上限约 **vx 1.5 m/s，vy 1.0 m/s，ω 1.0 rad/s**。
- 低通滤波 `alpha ∈ [0.1, 0.6]`（B2W 配置，每 episode 随机，逐维）。
- 低层策略 50 Hz，高层 5 Hz（`decimation=40`，`sim.dt=0.005` → **step_dt = 0.2 s**）。

### 2.4 Episode、终止与成功判据 [事实]

- 时长 60 s = **300 步**（`episode_length_s=60.0`），`max_episode_length` 不可按 episode 延长。
- 终止项（`navigation_env_cfg.py:381-398`，`mdp/terminations.py`）：

| 终止项 | 条件 | `time_out` 标志 | 是否算成功 |
|---|---|---|---|
| `at_goal` (`early_termination`) | 进入 0.5 m 半径并**连续保持 4 s**（20 步） | 是 | **是** |
| `time_out` | 走满 300 步 | 是 | 若曾进入 0.5 m 则为是 |
| `base_contact` | `base_link / .*hip / .*thigh` 接触力 > 1.0 | 否 | 否 |
| `large_pitch_angle` | pitch > 40° | 否 | 否 |
| `terrain_fall` | 高度 < −2 m | 是 | 否 |

注意：`at_goal` 需要**在目标半径内停留 4 s**（`goal_commands.py:382`、`terminations.py:156-166`），
不是"碰到即成功"。

### 2.5 随机化与延迟 [事实]

`navigation_env_cfg.py:261-344`：摩擦/回弹随机化、初始 yaw 随机（±π）、相机位姿随机（±5°、0.1 m）、
动作尺度随机、`push_robot`（0.2–5 s 间隔，±0.5 m/s）、观测延迟（`max_delay_* = 2` 步）。

### 2.6 策略网络 [事实]

`sru-navigation-learning/rsl_rl/modules/actor_critic_sru.py`：

```
Actor : [proprio(16) + depth(2560)] --cross-attn--> 64 + proprio 16
        --LSTM_SRU(512, 1层)--> h
        concat [h(512), path_embed(32)] --MLP[512,256,128]--> mean(3)
        σ: 全局 log_std（被 clamp 到 [-2.5, 0.2]）
Critic: 同构 + height_scan + time_embed
Path  : PathEncoder(1 层 self-attn + learned query cross-attn, d_model=32, 4 heads, 15 点)
```

`actor_critic_sru.py:425-441`：`log_std` 在构造分布前 clamp；参数本身不被裁剪。

---

## 3. 参考路径管线（问题核心，请重点阅读）

### 3.1 生成流程 [事实]

代码：`sru-navigation-sim/.../mdp/navigation/path_generator.py`、`.../navigation/goal_commands.py`

1. 每块 tile 建一个 PRM（`prm_cfg: N=3000, k=11, max_height_diff=50, padding=4`）。
2. 每个 episode reset 时，从 `spawn_mask` / `valid_mask` 各采一个点作为起点和终点
   （`goal_commands.py:535-611`）。
3. `GlobalPath.find_path()`：
   - 按 `category_weights` 抽路径类别（见 3.2）；
   - A*（`optimal=True`）或**带偏置启发式的 A***（`optimal=False`，等价于 GBFS，
     启发式 `h = 0.1·d_goal + 0.9·d_waypoint`，`path_generator.py:398-410`）；
   - 线视平滑（非最优类别不做 LOS）；
   - 重采样到 **15 个等弧长点**；
   - `infeasible` 类别再对**每一个点**（含起点与终点）叠加半径 ≤ `add_noise` 的均匀 2D 噪声。
4. 路径**在整个 episode 内固定**：只有 reset 时重建（`goal_commands.py:601-611`）；
   每个 step 只做世界系 → 机体系的坐标变换（`path_generator.py:151-182`）。
5. 观测中的路径为 `(15, 4)`：每点 `(单位方向 xyz, log(1+d)/max_i log(1+d_i))`。

### 3.2 训练时的路径类别分布 [事实] —— 最重要的发现之一

`IsaacLab/logs/rsl_rl/b2w_navigation_mdpo/2026-11-24_09-05-54_b2w_sigma_bounded/params/env.yaml`
（对应 `path_generator.py:30-48` 的默认值）：

```yaml
path_categories: [optimal, mildly_non_optimal, highly_non_optimal, infeasible]
category_weights: [0, 0, 0.5, 0.5]
```

即 **训练中从未出现过一条干净的 A* 最优路径**：

- 50%：带偏置启发式的 A*（刻意绕路，不做 LOS 平滑）；
- 50%：同上 + 每点独立 1.0 m 噪声（`add_noise=10 × horizontal_scale 0.1`）。

而论文（`docs/Haro et al. - 2026 - ...pdf`，§IV-B）写的是"**randomly generated either using
standard A\* with the Euclidean heuristic or GBFS with the biased heuristic**"，
即训练分布**同时包含最优路径**，并按概率/β 控制次优程度。本地配置把 optimal 与 mildly 都设成 0，
是对论文训练分布的一次改写，且未见单变量消融记录。

### 3.3 已知的四个路径缺陷 [事实 + 推导]

1. **端点也被加噪**：`add_noise_to_path`（`path_generator.py:563-574`）对 `path[0]` 与 `path[-1]`
   同样加噪。于是路径起点可能离机器人 1 m、路径终点可能离真实目标 1 m。
   训练时 `target_position`（真实目标）与路径尾点系统性不一致 → 策略有充分理由不信任路径。
2. **噪声点之间独立**：先重采样到 15 点再加逐点噪声，形成波长 ≈ 2 m、幅度 ≈ 1 m 的锯齿；
   不是"整条路径整体偏移"。这种形状在楼梯/窄门处会呈现为不可行。
3. **路径静止、不重规划**：机器人偏离后不会重建；路径前段会出现在机体坐标系的身后，
   观测里"身后还有若干点"。若机器人走了另一条路，路径会一直指向原地。
4. **没有局部规划器 / lookahead**：策略拿到整条 15 点路径（含已走过的点），
   没有任何"只喂前方 3–5 m"的裁剪。

### 3.4 路径相关的奖励与度量 [事实]

- `progress_along_path`：机器人到整条 polyline 的最近投影的弧长百分比，
  `progress = 100 × Δ(弧长)/总长`，阈值 0.15%/步（`path_generator.py:281-298`）。
- 机器人到路径的距离 `best_dist` 由 `MultiPath.get_all_path_metrics()` 返回，
  **但当前没有任何日志项记录它**（`MultiPath.update()` 的返回值在 `goal_commands.py:651-653` 被丢弃）。
- `progress_along_path` 奖励权重 0.05，且 95% 概率生效（`reward` 项 `probability: 0.95`）。

---

## 4. 奖励

### 4.1 当前生效的完整奖励表（B2W）[事实]

来源：`sru-navigation-sim/.../config/b2w/navigation_env_cfg.py:23-65`，
实测值来自 run 的 `params/env.yaml` 与 TensorBoard（窗口 5500–5891）。

| term | 函数 | weight | 机制 | `Episode_Reward` | **每 episode 贡献**（×60） |
|---|---|---|---|---|---|
| `progress` | `progress_to_goal` | **+1.5** | 每米缩短到目标的 xy 距离 | 0.262 | **+15.7**（≈10.5 m 净前进） |
| `reach_goal` | `reach_goal_direct` | **+10.0** | 每步在 0.5 m 半径内 | 0.301 | **+18.1** |
| `reach_goal_xy_tight` | `reach_goal_xyz` | +1.5 | σ=0.25 的 sigmoid 塑形（近目标） | 0.249 | +14.9 |
| `reach_goal_xy_soft` | `reach_goal_xyz` | +0.25 | σ=2.5 的 sigmoid（含 z） | 0.009 | +0.64 |
| `progress_along_path` | `progress_along_path` | **+0.05** | 弧长百分比 ×0.95 概率 | 0.0117 | **+0.70（≈14% 路径/局）** |
| `action_rate_l1` | — | −0.02 | 动作变化 L1 | −0.0525 | −3.15 |
| `joint_acc_l2_joint` | — | −1e−7 | 关节加速度 L2 | −0.0059 | −0.35 |
| `lateral_movement` | — | −0.02 | 侧向速度 | −0.0020 | −0.12 |
| `rot_movement` | — | −1e−5 | 角速度范数 | ≈0 | ≈0 |
| `backward_movement_penalty` | — | 0.0 | 禁用（curriculum 500 步后置 0） | 0 | 0 |
| `episode_termination` | `is_terminated` | −50 | 真终止（撞地/倾角） | −0.0495 | −2.97 |
| `time_out_penalty` | `episode_time_out` | **−100** | 走满 60 s | −0.0659 | −3.95 |

合计 ≈ **+39.6 / episode**，与 `Train/mean_reward ≈ 39.5` 一致。

**由该表可读出两个关键比例**：

- 目标类奖励合计 ≈ **+50.0**（含路径项 +0.7），代价类合计 ≈ **−10.5**；
- episode 级："直奔目标"的 `progress` (+15.7) 是"沿路径"的 `progress_along_path` (+0.70) 的 **≈22 倍**。
  换算成"每前进一米"：直连目标 1.5；沿路径 0.05 × (100/L) ≈ 0.125–0.25（L = 20–40 m 路径），
  即 **6–12 倍**。差距在 episode 级更大，恰恰说明机器人**没有**贴着路径走（见 6.4）。

### 4.2 与官方 release 的差异 [事实]

官方（`06e67ad:sru-navigation-sim/.../navigation_env_cfg.py`，即论文发布版）：

| term | 官方 weight | 本地 weight |
|---|---|---|
| `lateral_movement` | −0.1 | −0.02 |
| `action_rate_l1` | −0.1 | −0.02 |
| `progress`（到目标） | **无** | +1.5 |
| `reach_goal`（0.5 m 内每步） | **无** | +10.0 |
| `time_out_penalty` | **无**（`time_out` 终止本身已存在） | −100 |
| `reach_goal_xy_soft/tight` | 0.25 / 1.5 | 同 |
| `progress_along_path` | 0.05 | 同 |
| `episode_termination` | −50 | 同 |

本地改动记录在 `docs/reports/b2w_reward_recalibration_2026-11-24.md`：官方奖励下"站着不动"是最优策略
（站点 return ≈ −5.6，移动 ≈ −13），因此必须给"移动"正奖励。但这次重标定把**直连目标**变成了
主导项，路径项仍停留在 0.05。

### 4.3 与论文 reward 的差异（易被忽略）[事实]

论文 §IV-C 的总奖励是：

```
r_t = α1·r_task + α2·r_reg + α3·r_pen + α4·r_shortcut
```

- `r_task = 1/(1 + p_t²/σ²)`，仅在 `t > T_max − T_r` 或随机 δ 时生效（即"到点"塑形）；
- **`r_shortcut`**：当单步内沿路径的进度 Δp 超过阈值 ε 时给 +Δp，
  显式鼓励"跳过路径里不必要绕路"；
- 论文明确写："any path-following behavior must emerge implicitly from the observation-conditioning
  and training procedure"——**没有路径跟随项**。

本地实现里：

- **没有 `r_shortcut`**（全仓库 grep 无该实现）；
- 反而加了 `progress_along_path` 作为路径跟随项，权重却只有 0.05；
- `progress_to_goal`（1.5）等价于"每一步都奖励靠近目标"，与 `r_shortcut` 的"跳路奖励"语义相反。

### 4.4 冲突分析 [推导]

1. **绕路被直接惩罚**：`progress` 每米 +1.5，绕路时到目标的欧氏距离可能暂时增大 → 负贡献；
   而沿路径只 +0.05/%。绕一条 2 m 冗余的路径，代价约为 −3（直接 progress），
   收益约 +0.1（路径 progress）。**净收益为负是数学上的必然**。
2. **时限压力**：60 s × 实测平均速度（训练期 `velocity_magnitude ≈ 0.59 m/s`）≈ **35 m 可覆盖距离**；
   30×30 m 的格子对角 42 m，任何绕路都容易超时（超时 −20）。
3. **路径可信度**：见 3.2/3.3，训练中 100% 的路径都"不可全信"，且终点与真实目标可能差 1 m。
   策略学到"路径只是弱先验、目标向量才是真值"是**理性**的。
4. **楼梯前置犹豫**：楼梯前需要减速/对准；`progress` 只看到"距离没变"，`base_contact`/`large_pitch`
   终止风险却实实在在，`time_out` 还有 60 s。犹豫是这些力平衡下的合理局部最优 [假设]。
5. **场景 3（走进死胡同）**：若路径绕开障碍，直连目标方向恰好指向死胡同，策略会按 `progress` 走进去；
   由于路径静止且没有局部规划器，进去后路径观测指向身后，策略也没有"回退并跟路径"的训练信号 [假设]。

---

## 5. 算法：MDPO [事实]

`sru-navigation-learning/rsl_rl/algorithms/mdpo.py`

- 双 actor-critic，环境对半分（`num_env_1 = num_envs // 2`），各用一半 rollout；
- 互蒸馏：`distill_coef=0.02` 的对称 KL（`_kl_gaussian`）；
- 其余为 PPO：`clip_param 0.2`、`value_loss_coef 0.02`、`num_learning_epochs 5`、`mini_batches 4`、
  `γ=0.999`、`λ=0.95`、`max_grad_norm 1.0`；
- LR 调度 `schedule="exponential"`（按时间衰减，**不按 KL 自适应**；`desired_kl` 实际不生效）；
- 支持 `min_entropy_coef / entropy_hold_iters / entropy_decay_span / entropy_schedule`，
  但 B2W 配置未启用（`entropy_schedule` 默认 `"fixed"`）；
- **play 只用 `actor_critic_1`**（`on_policy_runner.py:405-411`），且 `act_inference` 输出确定性均值；
  训练日志中的指标是两个策略混合的结果。→ **play 行为与日志 success_rate 不是同一个策略**。

---

## 6. 当前训练 run 结果

### 6.1 Run 信息 [事实]

- 目录：`/home/yichangfeng/sru-path-aware/IsaacLab/logs/rsl_rl/b2w_navigation_mdpo/2026-11-24_09-05-54_b2w_sigma_bounded`
- 命令：`./isaaclab.sh -p .../train.py --task Isaac-Nav-MDPO-B2W-v0 --num_envs 1046 --max_iterations 15000 --seed 60 --run_name b2w_sigma_bounded --headless`
- 进度：6020 / 15000，25.5 h，`Perf/total_fps ≈ 1100`，预计还需 ~34 h；
- LR：1e-3 → 1.5e-4（iter 6000），按当前指数衰减到 15000 时约 **8e-6**；
- **σ 钉死**：iter **1093** 起 `Policy/mean_noise_std` 恒等于 1.2214（= e^0.2），直到 6020 未变；
  checkpoint 里的原始 `log_std = [0.201, 0.205, 0.202]`，说明是优化器把参数顶在 clamp 边界上。

### 6.2 曲线 [事实]

| 窗口 | reward | success | base_contact | time_out | action_rate | σ | 平均 episode 长度 |
|---|---|---|---|---|---|---|---|
| 1–100 | 15.4 | 0.134 | 0.608 | 0.087 | −0.019 | 0.86 | 91 |
| 100–250 | 35.4 | 0.334 | 0.441 | 0.145 | −0.018 | 0.58 | 118 |
| 500–1000 | 33.9 | 0.397 | 0.408 | 0.167 | −0.033 | 0.88 | 151 |
| 1000–1500 | 35.0 | 0.427 | 0.407 | 0.134 | −0.043 | 1.219 | 137 |
| 2000–2500 | 33.0 | 0.450 | 0.360 | 0.170 | −0.048 | 1.221 | 148 |
| 3000–3500 | 36.4 | 0.480 | 0.327 | 0.174 | −0.048 | 1.221 | 144 |
| 4000–4500 | 34.8 | 0.477 | 0.309 | 0.205 | −0.052 | 1.221 | 151 |
| 5000–5500 | 38.4 | 0.496 | 0.275 | 0.220 | −0.053 | 1.221 | 150 |
| 5500–5891 | 39.5 | 0.504 | 0.271 | 0.220 | −0.053 | 1.221 | 151 |
| 5800–6200 | 39.8 | 0.500 | 0.270 | 0.219 | −0.053 | 1.221 | 151 |

终止方式（5500–5891）：`at_goal` 0.464、`base_contact` 0.271、`time_out` 0.220、`large_pitch` 0.069。

### 6.3 与上一 run 的对比 [事实]

基线 `2026-11-23_18-45-05_b2w_recalib_1046`（无 σ 上界，`entropy_coef=0.00375`），同窗口 2500–3000：

| 指标 | recalib | sigma_bounded |
|---|---|---|
| reward | 22.2 | **35.2** |
| success | 0.411 | **0.465** |
| base_contact | 0.504 | **0.337** |
| action_rate | −0.213 | **−0.048** |
| σ | 8.47 | 1.221 |

→ σ 上界止住了崩塌，但只是"把 σ 按在墙上"，不是让 σ 收敛；
成功率增速从 σ 到顶前的约 +0.08/1000 iter（100→1500）降到之后的约 **+0.013/1000 iter**（1500→6200），
慢了约 6 倍。该对比混入了 LR 从 1e-3 衰减到 1.5e-4 的影响，不能全部归因于 σ。

### 6.4 路径跟随的定量证据 [推导]

最近 400 iter：每 episode 净接近目标 **10.5 m**（由 `progress` 还原），
而沿路径的进度只有 **≈14%**。若路径长 ~20 m，14% ≈ 2.8 m。
**机器人实际走的路与参考路径的重合度很低**——这与 play 中"大部分时候不沿路径走"一致。
（该推断依赖路径长度估计，最好用 3.4 提到的 `distance_to_path` 日志直接确认。）

---

## 7. Play 中观察到的问题（用户报告，2026-11-25）

> 以下为原样记录，供后续 AI 作为待解释现象。

1. **楼梯**：路径需要"转身再上下楼梯"时，机器狗只在楼梯前来回挪动，不会转身上楼。
   低层 locomotion 本身足以上下楼。
2. **初始朝向**：机器人一开局就朝向最终目标点，看不出路径点对它的引导作用。
3. **不沿路径**：大部分时候不沿路径走。三个具体情景：
   - 情景 1：路径绕过楼梯通向目标，机器人却选择上楼梯；上到一半犹豫停滞；
   - 情景 2：路径规划了可行的"上楼梯到目标"，机器人却在别处探索，或在楼梯附近前后微动；
   - 情景 3：路径绕开障碍，机器人仍走进死胡同，而非沿路径走。
4. **路径噪声过大且与奖励冲突**：除了"规划路径与直线到目标很接近"的情形，
   其他需要绕障碍的情况机器人都不信任路径，起到反效果；
   即便路径正常（例如"下楼梯后直行到目标"），机器人也会在高台上徘徊，而不是从楼梯下去。

---

## 8. 现象 → 机制假设（按证据强度排序）

| # | 假设 | 证据 | 强度 | 如何验证 |
|---|---|---|---|---|
| H1 | **训练分布里没有一条干净的路径**（optimal 权重 0），策略因此学到"路径不可信" | `env.yaml` 的 `category_weights=[0,0,0.5,0.5]`；论文训练分布含最优路径；`add_noise` 逐点 1 m 且含端点 | 强 | 把权重改成含 optimal（如 `[0.5,0.2,0.2,0.1]`）重训，对比 play 跟随度；或离线统计策略对路径噪声的敏感度 |
| H2 | **奖励结构把"直奔目标"放在约 22 倍权重上**（episode 级），绕路在数学上是负收益；且缺少论文的 `r_shortcut` | 4.1/4.3 的逐项分解；`progress` +15.7 vs `progress_along_path` +0.70 | 强 | 重标定 reward 后重训；或固定策略做 reward 归因（把路径项调大看行为变化） |
| H3 | **60 s 时限与 0.6 m/s 速度不匹配**，长绕路必然超时（−20/次） | 超时率 0.22；平均 episode 151/300 步；论文训练 60 s、评测 120 s | 强 | 延长 horizon 到 120 s 或提高速度上限，观察绕路率 |
| H4 | **路径静止且包含身后点**，偏离后观测误导；没有局部规划 / lookahead | `path_generator.py:151-182`（每步只做坐标变换）；`goal_commands.py:601-611`（仅 reset 重建） | 中 | 记录 `distance_to_path` 曲线；把观测截断为"最近点之后 3–5 m"做 A/B |
| H5 | **σ 恒定在 1.22**（训练噪声不衰减），策略被训练成"在噪声下保守、易震荡" | checkpoint 原始 `log_std≈0.20`；`action_rate` 从 −0.019 涨到 −0.053 | 中 | 加熵退火 / 逐步下调 `log_std_max` 重训；注意 play 是确定性输出，噪声只影响训练期的行为塑形 |
| H6 | **路径噪声形状不合理**（逐点独立 1 m、端点也加噪、先重采样后加噪）→ 楼梯/窄门处路径看起来不可行 | `path_generator.py:369-409, 563-574` | 中 | 只对中间点加噪 + 平滑噪声；统计噪声后路径的可通行率 |
| H7 | **楼梯场景的局部原因**：1 m 爬升、相机下俯 20°、终止对 base/hip/thigh 敏感、无地形课程 | `maze_config.py:46-57`；`terrain_constants.py:51-75`；`navigation_env_cfg.py:109-111` | 弱-中 | play 时记录楼梯前 3 s 的 depth/动作/pitch；单独在 stairs tile 上评估 |
| H8 | **play 与日志不是同一策略**（日志混合两个策略；play 只用 `actor_critic_1`，确定性） | `on_policy_runner.py:405-411` | 事实（非假设） | 分别在两个策略上跑 play，对比 SR |

---

## 9. 建议的下一步（按代价从低到高）

**E0（不改训练的诊断，最优先）**

1. 在 `Metrics` 或观测组里增加 `distance_to_path`（`MultiPath.get_all_path_metrics()[0]`）与
   `progress_along_path` 的日志，用于量化"跟路径程度"。
2. play 时按 episode 记录：初始到目标直线距离、路径总长、最短路径比（spl）、
   是否经过 stairs tile、在楼梯前的停留时长与 pitch 序列。
3. 分别加载 `actor_critic_1` 与 `actor_critic_2` 做 play，确认两者行为是否一致。

**E1（路径分布，单项改动）**
把 `category_weights` 改成包含 optimal，例如 `[0.5, 0.2, 0.2, 0.1]`，
其余不动，重训一个短 run（3000–5000 iter）对比跟随度。这是最可能直接改善"不信任路径"的改动。

**E2（路径噪声形态）**
不对 `path[0]`（起点）与 `path[-1]`（终点）加噪；或改成对整条路径施加平滑的整体偏移 +
少量逐点噪声。同时评估是否把 `add_noise` 从 10 格降到 5 格。

**E3（奖励）**
在 E1 有基线后，做奖励消融：
- 提高 `progress_along_path`（如 0.05 → 0.3），或换成论文的 `r_shortcut`；
- 降低 `progress`（1.5 → 0.5）或把"到目标进度"改成路径势函数；
- 检查 `reach_goal_xy_tight`（+14.9/局）是否鼓励在目标附近徘徊而不进入。

**E4（时限/速度）**
把 `episode_length_s` 提到 120 s（论文评测设置），或提高 `policy_scaling[0]`；
观察超时率与绕路意愿。注意这会改变 MDP，需要与 E3 分开做。

**E5（探索噪声退火）**
B2W 配置启用 `min_entropy_coef=0`、`entropy_hold_iters≈1000`、`entropy_schedule="cosine"`，
或按进度下调 `log_std_max`（0.2 → 0.0 → −0.5）。这是当前 run 已知的遗留问题。

**E6（路径观测）**
把 15 点截断为"最近投影点之后 N 米"，并显式提供 `distance_to_path` 作为观测量，
让策略能判断"我偏离了多少"。属于结构性改动，建议在 E1–E3 明确后再做。

---

## 10. 未解/需要确认的问题

1. `velocity_toward_goal` 从 0.37 掉到 0.03（而速度模 0.59、成功率 0.50），
   该指标的语义是否被"近目标 + 失败 episode 的无效步"污染？需要用 play 的轨迹直接看。
2. 训练日志中 `Episode_Reward` 与 `weight×dt` 的关系已在本文件中对齐（÷60 s），
   但历史文档 `navigation_rewards.md` 的口径可能不同，引用时注意。
3. `reach_goal_xy_tight`（flat, σ=0.25）在 0.5 m 半径内几乎饱和，
   是否与 `reach_goal` 重复计费、导致"停在目标半径里"比"精准到达"更划算？
4. 论文的 `r_shortcut` 与 `β`/类别采样概率在本地缺失，是否有意为之？
5. 楼梯 tile 的梯级尺寸、坡度是否与 B2W 轮腿能力匹配（用户说 locomotion 可以，但仿真里的
   stair 参数来源于 `hf_terrains_maze_cfg.py`，需要核对）。
6. play 的 2×2 地形是否随机落到 stairs tile（影响复现用户描述的场景）。

---

## 11. 关键文件索引（绝对路径）

| 内容 | 文件 |
|---|---|
| B2W 环境/奖励配置 | `/home/yichangfeng/sru-path-aware/sru-navigation-sim/isaaclab_nav_task/navigation/config/b2w/navigation_env_cfg.py` |
| 基础环境（观测/终止/事件） | `.../navigation/navigation_env_cfg.py` |
| 奖励函数实现 | `.../navigation/mdp/rewards.py` |
| 终止函数 | `.../navigation/mdp/terminations.py` |
| 观测函数（路径/目标/深度） | `.../navigation/mdp/observations.py` |
| goal command（采样/路径/度量） | `.../navigation/mdp/navigation/goal_commands.py` |
| PRM / A* / 路径类别 / 噪声 | `.../navigation/mdp/navigation/path_generator.py` |
| 动作/低层策略 | `.../navigation/mdp/navigation/actions/navigation_se2_actions.py` |
| 地形与常量 | `.../terrains/maze_config.py`、`.../terrains/terrain_constants.py` |
| 网络 | `/home/yichangfeng/sru-path-aware/sru-navigation-learning/rsl_rl/modules/actor_critic_sru.py` |
| 路径编码器 | `.../rsl_rl/networks/sru_memory/path_encoder.py` |
| MDPO | `.../rsl_rl/algorithms/mdpo.py` |
| Runner（日志/推理） | `.../rsl_rl/runners/on_policy_runner.py` |
| 当前 run | `.../IsaacLab/logs/rsl_rl/b2w_navigation_mdpo/2026-11-24_09-05-54_b2w_sigma_bounded` |
| 论文 | `.../docs/Haro et al. - 2026 - Path-conditioned Reinforcement Learning-based Local Planning for Long-Range Navigation.pdf` |
| 历史分析 | `.../docs/reports/b2w_reward_recalibration_2026-11-24.md`、`.../docs/specs/navigation_rewards.md` |

---

## 12. 复现命令

训练（当前 run）：

```bash
cd /home/yichangfeng/sru-path-aware
TASK=Isaac-Nav-MDPO-B2W-v0 NUM_ENVS=1046 MAX_ITER=15000 RUN_NAME=b2w_sigma_bounded ./train_b2w_pathaware.sh --headless
```

回放：

```bash
cd /home/yichangfeng/sru-path-aware
./play_b2w_pathaware.sh \
  --checkpoint /home/yichangfeng/sru-path-aware/IsaacLab/logs/rsl_rl/b2w_navigation_mdpo/2026-11-24_09-05-54_b2w_sigma_bounded/model_5500.pt \
  --num_envs 1 --gui
```

解析训练日志（无需 tensorboard）：

```bash
python3 /home/yichangfeng/sru-path-aware/skills/sru-training-analysis/scripts/analyze_rsl_rl_events.py \
  /home/yichangfeng/sru-path-aware/IsaacLab/logs/rsl_rl/b2w_navigation_mdpo/2026-11-24_09-05-54_b2w_sigma_bounded \
  --windows 1-100,500-1000,1000-1500,2000-2500,3000-3500,4000-4500,5000-5500,5500-5891
```
