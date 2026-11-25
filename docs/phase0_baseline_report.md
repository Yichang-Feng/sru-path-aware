# Phase 0 阻塞性实测 + 实施报告（B2W path-aware v2）

对应规格：`docs/agent_doing.md`（Stage 0 → Stage F）。本文件只记录**实测数据、做出的取舍、验证结果**；
规格本身的冻结参数表仍是唯一真值来源。

---

## 1. 五项阻塞性实测（M1–M5）

测量脚本：`sru-navigation-sim/scripts/stage0_measure.py`（4×10 tiles，PRM N=300，仅为测量，不是训练配置）。
原始输出：`docs/phase0_measurements.md`。

| # | 项 | 实测 | 对设计的影响 |
|---|---|---|---|
| **M1** | B2W 机身尺寸 | `base_link` 0.854 × **0.298** × 0.348 m；含轮腿整机 0.883 × **0.520** × 1.055 m | `seg_passable` padding = 0.520/2 + 0.05 = **0.31 m**（不是 PRM 的 0.4 m） |
| **M2** | 通道净宽分布 | 全部自由格 clearance 中位数 1.06 m；**通道脊线**（局部极大）中位数 0.40 m、p95 0.81 m；clearance < 0.45 m 的自由格占 19.6% | 0.8 m 的窄门是**常见**地形特征，不是边角情况 → `pinch_clearance_m = 0.45` 会把这类段全部标为 pinch，严格门控 |
| **M3** | `free_nodes` 是否含 padding | padding 窗口覆盖 26.8% 的自由格；`free_nodes/nodes` = 94.1% | 节点采样**已含** 4 格膨胀；`free_nodes` 用 `free_mask`（地面/平台）判定，与 padding 语义不同 |
| **M4** | `is_traversable` 语义 | 跨墙段 False（`check_padding` 两种取值都是 False）；**端点在障碍内部时返回 True**；`max_height_diff` = 50 单位 = 0.25 m，楼梯单级 0.2 m 可通过 | 端点在障碍内不会报错 → B5 的"注入段显式强制 False"是**必须**的，不能依赖 `is_traversable` |
| **M5** | optimal 路径不可行段比例 | `check_padding=True`：**6.5%** 的段不可行；窄段（clearance<0.45）中 45.4% 不可行，开阔段 3.3% | 6.5% 是"真机可能蹭墙/余量不足"的基线量级；也说明 PRM 的 0.4 m 膨胀会误杀窄门（故改用 0.31 m 度量判据） |

**M1 的取舍说明**：`base_link` 半宽只有 0.149 m，但轮腿展开后整机横向 0.52 m。用 0.31 m 而不是 0.20 m
是保守选择——误判"能过"会把机器人送进实际过不去的缝里，误判"不能过"只会让门控放松一段（代价小得多）。

---

## 2. 附加核查（Stage 0.3）

| 项 | 结论 |
|---|---|
| **G7**：`command_manager.compute()` 与 `reward_manager.compute()` 的先后 | ⚠️ **与规格假设相反**：`ManagerBasedRLEnv.step()` 的顺序是 终止 → 奖励 → reset → **命令** → 观测。因此 §A1 的"在 `_update_command` 产生驻留脉冲"会导致奖励永远晚一步。**采用规格给出的 G7 备选方案**：驻留计数与单步脉冲放在 `at_goal_navigation`（终止阶段，先于奖励），`goal_reached_bonus` 只读脉冲 |
| `DeltaTransformPathNoiseCfg` 是整条刚体变换还是逐点独立 | **整条刚体变换**（每个 env 一组 rotation/translation，10% 概率触发），语义是定位误差 → 保留不动 |
| `CurriculumCfg` 是否接线 `update_terrain_levels` | 没有。只有 `disable_backward_penalty`（500 步后关闭后退惩罚）。地形课程**未接线** |
| stairs 踏步尺寸 | 5 级 × 0.2 m 高，踏面 20/5 = 4 格 = **0.4 m**；0.2 m < `max_height_diff` 0.25 m → PRM 判为可通过（与实测一致） |
| **运行时 `rsl_rl` 到底来自哪个仓库** | ⚠️ **重要**：conda env 里的 editable 安装指向 `~/sru-path-aware-official/sru-navigation-learning`，不是本工作区。也就是说**本工作区 `sru-navigation-learning/` 里的修改在此之前从未进入训练**（两处文件在改动前与 official 完全一致，所以 11-24 那次 run 的分析仍然成立）。已在 `train_b2w_pathaware.sh` / `play_b2w_pathaware.sh` 里把本工作区放到 `PYTHONPATH` 最前面，并验证 `./isaaclab.sh -p -c "import rsl_rl"` 解析到本工作区 |

---

## 3. 实施内容与关键取舍

### 3.1 新增/重写（按 Stage）

| Stage | 文件 | 要点 |
|---|---|---|
| **A** | `mdp/terminations.py`, `mdp/navigation/goal_commands.py`, `mdp/rewards.py` | 连续驻留 1.5 s（8 步，离开即清零）；超时成功改为**当步瞬时**在 0.5 m 内；`goal_reached_bonus`（一次性 +25）；两处 `reach_goal_xyz` 的 `arrive_mask` 改为瞬时 |
| **B** | `mdp/navigation/prm.py`, `path_generator.py` | per-tile clearance map；`segment_passable`（度量判据）；拐角弦 50/50 修复；clearance 自适应噪声（端点锁定、拒绝采样、pinch 段不加噪）；结构化挡路事件；三档门 + 双侧膨胀 |
| **C** | `path_generator.py::MultiPath._v2_path_metrics` | 米数制进度、限速追踪器、棘轮支付（`s_reward = max(...)`，永不欠账） |
| **D** | `config/b2w/navigation_env_cfg.py`, `mdp/rewards.py` | 奖励表按 §2.5；`path_deviation_penalty`；路径类别权重 `[0.45,0.25,0.20,0.10]` |
| **E** | `mdp/observations.py`, 配置, `rl_cfg.py`, `actor_critic_sru` 侧配置 | 局部路径 60 维 + `distance_to_path` 1 维；actor 2636 → **2697**，critic 5773 → **5834**；`path_length` 60 → **120** |
| **F** | `config/b2w/agents/rsl_rl_cfg.py`, `algorithms/mdpo.py`, `runners/on_policy_runner.py` | LR cosine（floor 1e-4）、熵 cosine 退火；`log_std_max` **移动墙面**调度；horizon 90 s；新增 `explained_variance`、`raw_log_std_mean`、`log_std_max` 日志 |

### 3.2 与规格不同的四处（都有原因，评审时可推翻）

1. **驻留计数位置**（见 G7）：在终止函数里而不是命令里。
2. **LOS 平滑用度量判据**：v2 的平滑改用 `segment_passable(0.31)` 而不是 `is_traversable(check_padding=True)`。
   PRM 的 0.4 m 膨胀会放过"擦着 0.8 m 门框侧面切过去"的弦——单元测试里表现为门段的 clearance 只有 0.283 m。
3. **neck 段完全不加噪**：只做拒绝采样不够——它校验的是**稀疏**段，重采样后仍可能切到门框角。因此相邻段任一是 pinch 段的航点直接跳过加噪。
4. **`infeasible` 类别的新语义**：从"整条路径加 1 m 均匀噪声"改为"GBFS 无 LOS + 保证一处结构化挡路事件"。规格 §2.1 给了权重但没重定义该类别；这样既满足原则 2（不制造穿墙航点作为主分布），又保留了"路径本身不可直接走"的样本。

### 3.3 对 Go2W 的影响（2026-11-25 更新）

v2 层抽到了 `navigation/config/path_aware.py::apply_path_aware_v2`，**B2W 与 Go2W 的 path-aware 任务
调用同一个函数**——这是为了避免两个机器人各自漂移（此前就发生过：一边改了、另一边没改，只有等训练
结果不对才发现）。

| 任务 | 状态 |
|---|---|
| `Isaac-Nav-{MDPO,PPO}-B2W-*` | 全部走 v2 |
| `Isaac-Nav-{MDPO,PPO}-Go2W-PathAware-{,-Play,-Dev}-v0` | **重指向** `Go2WNavigationEnvCfg_PathAwareV2`（Phase4 地形 + D435 相机 + v2 层）；player 60 s、训练 90 s |
| `...-Go2W-PathAware-NoCurriculum-*`、Phase1–5、LowObs、Heading、PureMaze | **不变**（仍是 60 s / legacy 管线 / 各自奖励），可复现旧结果 |
| `Go2WPathAwareRewardsCfg` | 直接继承 `B2WRewardsCfg`（奖励项与机器人无关），`Go2WNavigationEnvCfg.__post_init__` 会无条件覆盖 `self.rewards`，所以 v2 子类在其 `super()` 之后重新赋值 |
| 地形覆盖 | Phase4 的 `curriculum=True` + `max_init_terrain_level=0`，而 `CurriculumCfg` 里**没有** `update_terrain_levels`，等于全程只训难度第 0 行。v2 子类关掉课程并把 `max_init_terrain_level` 设成 10（全 6 行），和 B2W 一致 |
| `joint_acc_l2_joint` | 该项来自共享基底 `RewardsCfg`，正则写的是 AoW-D 的 `.*HAA/.*HFE/.*KFE/.*WHEEL`；换成 path-aware 奖励集后必须在 `__post_init__` 里重指到 `.*hip_joint/.*thigh_joint/.*calf_joint/.*foot_joint`，否则 manager 构建时直接 `Not all regular expressions are matched`（Go2W 冒烟抓到） |
| `Go2WNav{MDPO,PPO}RunnerPathAwareV2Cfg` | 直接继承 B2W 的 runner，`seed=42`、`max_iterations=4000`，其余（`path_length=120`、LR/熵退火、`log_std` 移动墙）与 B2W 完全一致 |

Go2W 的观测维度与 B2W 相同（2697 / 5834）：深度项无论 D435 还是 ZED 都是 64×40=2560，height scan 同一网格。

验证：

- `test_b2w_and_go2w_get_identical_v2_layers` / `test_go2w_v2_runners_match_the_b2w_schedule`
  逐字段比对两边的路径配置、终止参数、观测顺序、奖励权重、地形覆盖、关节正则与调度字段（19 项单测全绿）。
- 场景级冒烟（`Isaac-Nav-MDPO-Go2W-PathAware-Dev-v0`，32 envs，3 iter）：**exit 0、无 traceback**，
  policy 2697 / critic 5834、14 项奖励权重与 B2W 一致、`agent.yaml` 里 `path_length: 120` 与
  `log_std_schedule: cosine` 等全部生效。第一次跑时正是它抓到了 `joint_acc_l2_joint` 的关节正则问题。

```bash
./train_go2w_pathaware.sh --task Isaac-Nav-MDPO-Go2W-PathAware-Dev-v0 \
  --num_envs 32 --max_iterations 3 --headless --no_debug_vis
```

---

## 4. 验证

### 4.1 单元测试（`sru-navigation-sim/tests/test_path_pipeline.py`，17 项）

```
cd IsaacLab && ./isaaclab.sh -p -m pytest ../sru-navigation-sim/tests -q
17 passed
```

覆盖：端点锁定、**窄门段必须 passable**（本轮最重要的回归）、注入段必然不可行、注入不改 z、双侧膨胀不 wrap、
三档门、稠密路径等弧长、U 型跳变不付钱、跨无效段限速到 `PAY_CAP`、沿路行走付满、往返不重复付、追踪器单调收敛、
走廊门控、连续驻留/单步脉冲、奖励经济学（站立 −10 < 完成 > 50，路径项 > 5× 直连项）、
**JIT 导出与训练前向数值一致**（30 路点/2697 维，atol 1e-5）。

测试发现并修掉了两个真实 bug：`_compute_relax` 原实现对**批量**张量沿错误维度切片；
门段的 LOS 平滑若沿用 PRM 的 0.4 m 膨胀会放过"擦着门框切角"的弦（clearance 0.283 m）。

### 4.2 冒烟训练

```bash
./train_b2w_pathaware.sh --task Isaac-Nav-MDPO-B2W-v0 --num_envs 32 --max_iterations 10 \
  --headless --no_debug_vis
```

环境构建阶段已确认（`--no_debug_vis`，观测/奖励/终止项按设计生效）：

| 项 | 实测 |
|---|---|
| policy 观测维度 | **2697** = 3+3+3+3+**4**+**1**+**60**+**60**+2560，`distance_to_path` 在 `target_path` 之前、`target_path_local` 紧随其后 ✓ |
| critic 观测维度 | **5834** = 同上 + `time_normalized`(1) + `height_scan_critic`(3136) ✓ |
| 奖励项 | 14 项，含 `progress_along_path` 1.2 / `progress` 0.15 / `reach_goal` 2.0 / `goal_reached_bonus` 25.0 / `reach_goal_xy_tight` 0.75 / `path_deviation_penalty` −0.15 / `time_out_penalty` −50.0 ✓ |
| 终止项 | `early_termination` (time_out=True)、`time_out` ✓ |

（10 iter 的完整结果见文末。）

**冒烟结果**：`Isaac-Nav-MDPO-B2W-v0`、32 envs、10 iterations、**exit 0、无 traceback**，
`model_0.pt` / `model_9.pt` 正常落盘，`wandb` 正常记录。观测维度、奖励表、终止项、四条新标量如上表。

### 4.3 Stage F 的落地验证（F4）

`params/agent.yaml` 已含全部新字段：`path_length: 120`、`log_std_schedule: cosine`、
`log_std_anneal_{start,mid,end}=1000/5000/8000`、`log_std_max_{initial,mid,final}=0.1/-0.6/-1.0`、
`entropy_schedule: cosine`、`entropy_hold_iters: 500`、`entropy_decay_span: 3500`、
`min_learning_rate: 0.0001`、`schedule: cosine`。

TensorBoard 标量（前 4 个 iteration）也已按设计写入：

| 标量 | 实测 | 说明 |
|---|---|---|
| `Policy/raw_log_std_mean` | −0.008 → −0.006 | clamp **前**的 log_std，低于墙面 0.1 ✓ |
| `Policy/log_std_max` | 0.1 | 墙面当前值（iter < 1000）✓ |
| `Policy/mean_noise_std` | 0.99 | clamp 后，与 raw 一致 ✓ |
| `Loss/explained_variance` | ≈0.00 | 未训练 critic，符合预期 ✓ |
| `Loss/learning_rate` | 9.8e-4 → 8.1e-4 | cosine（10 iter 下衰减快属正常）✓ |
| `Loss/entropy_coef` | 0.001 | hold 到 iter 500 ✓ |

> 这两个新标量正是 11-24 那次 run 缺失的关键证据：当时只能从 checkpoint 里反推 `log_std` 钉在墙上。

---

## 5. 基线重评（Stage 0.5）与"不跟路径"的量化

### 5.1 为什么必须重评

旧日志的 `Metrics/robot_goal/success_rate` = 0.504 是**碰边锁存**口径、**两个 actor 混合**、且**带观测噪声与外力推动**，
不能当新管线的对照。重评用 `model_6000`，在**除终止规则外全部回退**的环境里跑（`--legacy_env`：
legacy 管线、`[0,0,0.5,0.5]` 类别、60 维路径观测），只把终止换成连续驻留；Play 协议（无观测噪声、无 push），
每档 200 局、20 envs、地形 4×12。

### 5.2 结果

**60 s（与旧训练同 horizon）**

| bucket | n | SR | SR(旧口径) | SPL | median d_path | p90 d_path | 直线距离 | 计划路径长 | 实际行驶 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | 200 | **0.580** | 0.580 | **0.449** | **0.91 m** | 1.99 m | 13.9 m | 24.6 m | **14.4 m** |
| maze | 57 | 0.737 | 0.737 | 0.577 | 0.76 | 1.95 | 12.2 | 22.4 | 13.9 |
| non_maze | 27 | 0.815 | 0.815 | 0.595 | 0.87 | 2.24 | 12.3 | 24.8 | 14.6 |
| **stairs** | 52 | **0.346** | 0.346 | 0.263 | 1.03 | 1.48 | 12.8 | 23.8 | 12.7 |
| pits | 64 | 0.531 | 0.531 | 0.425 | 1.13 | 2.41 | 14.8 | 25.3 | 15.2 |

**90 s（新 horizon）**

| bucket | n | SR | SR(旧口径) | SPL | median d_path | p90 d_path | 直线距离 | 计划路径长 | 实际行驶 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | 200 | **0.625** | 0.625 | **0.473** | **0.92 m** | 2.06 m | 15.2 m | 23.6 m | **16.2 m** |
| maze | 46 | 0.543 | 0.543 | 0.402 | 0.99 | 1.86 | 15.5 | 28.6 | 16.2 |
| non_maze | 34 | 0.853 | 0.853 | 0.658 | 0.94 | 2.33 | 12.8 | 20.6 | 14.6 |
| stairs | 12 | 0.500 | 0.500 | 0.317 | 0.54 | 1.42 | 10.6 | 16.6 | 13.5 |
| pits | 108 | 0.602 | 0.602 | 0.462 | 1.01 | 2.17 | 16.1 | 24.6 | 16.8 |

### 5.3 怎么读这两张表

1. **"不跟路径"被量化了**：计划路径中位 **24.6 m**，而策略实际只走了 **14.4 m**——几乎就是直线距离
   （13.9 m）。也就是说它是一台"朝目标直线冲 + 局部绕障"的机器，规划路径对它的行为几乎没有影响。
   `median d_path` 0.91 m、`p90` 1.99 m 是同一个结论的另一个视角。
   这正是旧奖励结构下**理性**的结果：路径项每回合只值 0.70，直连项值 15.7。
2. **SPL 0.45 的含义**：成功回合里，它走的距离只有"最短路径"的 1.04 倍——很直，但走的是自己那条直线，
   所以 SPL 高恰恰说明它没在执行规划（规划路径本身平均比直线长 1.8 倍）。
3. **楼梯是最大瓶颈**：60 s 档 stairs SR **0.346**（n=52，样本足够），non_maze 0.815。和 play 里
   "楼梯前徘徊、不转身"的现象一致；90 s 档 stairs 只有 12 局（0.50），**不足以**下结论。
4. **两档不能直接比**：地形是随机分配的，两档的 tile 构成差异很大（stairs 52 vs 12，pits 64 vs 108），
   bucket 级数字的抽样波动 ≈ ±0.1，只有 `all` 行可以跨档比较（0.580 vs 0.625）。
5. **旧口径 SR 与新口径完全相同**：这 400 局里没有"进了 0.5 m 圈又走掉"的回合——策略一到就刹停。
   所以锁存带来的"水分"在这个策略上不是主要问题，**主要问题是它根本没打算走近路以外的路**。
6. 这个 0.58/0.63 比训练日志的 0.504 高，不是矛盾：重评没有观测噪声、没有 push；而日志里是两个 actor 混合的结果。

> **验收基线取 60 s 档 `all`: SR 0.580 / median d_path 0.91 m / SPL 0.449。**
> 新管线要跑赢的是这个，不是 0.504。

### 5.4 复现命令

> 旧 checkpoint 的观测是 60 维路径 / 2636 维，新环境是 120 / 2697 - 直接喂会加载失败。
> `--legacy_env` 把**除终止规则以外**的东西都回退（legacy 管线、`[0,0,0.5,0.5]`、60 维路径），
> 只保留新的连续驻留判定——这正是本次重评要挤掉的水分。

```bash
cd IsaacLab && PYTHONPATH=/home/yichangfeng/sru-path-aware/sru-navigation-learning \
./isaaclab.sh -p ../sru-navigation-sim/scripts/eval_policy.py \
  --task Isaac-Nav-MDPO-B2W-Play-v0 \
  --checkpoint /home/yichangfeng/sru-path-aware/IsaacLab/logs/rsl_rl/b2w_navigation_mdpo/2026-11-24_09-05-54_b2w_sigma_bounded/model_6000.pt \
  --episodes 200 --episode_length_s 60 --num_envs 20 --legacy_env \
  --out ../docs/phase0_baseline_60s.md --headless
```

> `SR(old rule)`（曾进入 0.5 m）与 `SR`（连续驻留）并列给出，两者的差值就是"碰边锁存"灌进 SR 的水分。

### 5.5 当前状态

- 代码：Stage A–F 已实现并提交；17 项单元测试全绿；10 iter 全量冒烟 **exit 0**。
- 基线重评：60 s 与 90 s 两档各 200 局已完成（见 §5.2），交付报告
  `docs/phase0_baseline_60s.md` / `docs/phase0_baseline_90s.md`。
- **待办**：Stage G 的 4000 iter 全量训练（数天），未启动；见 §7、§8。

---

## 6. 全量训练时要盯的面板（规格 §10 G2 的落地版本）

| 指标 | 期望 | 异常含义 |
|---|---|---|
| `Metrics/robot_goal/path_progress` | 训练早期即非零 | 长期为 0 → 门控/门限把支付全挡了 |
| `Episode_Reward/progress_along_path` × 90 vs `Episode_Reward/progress` × 90 | 前者 > 后者 | 反转 → 奖励配平失败 |
| `Metrics/robot_goal/invalid_seg_frac` | 5–10%，稳定 | 上升 → 路径质量退化（膨胀不改变它） |
| `Metrics/robot_goal/relax_frac` | 12–20%，稳定 | 只监控，**不要**和上一条混看 |
| `Metrics/robot_goal/blocked_episode` | ≈0.15→0.30 | 与挡路事件概率的 ramp 对齐 |
| `Metrics/robot_goal/distance_to_path` | 1500 iter 起下降，4000 iter < 0.8 m | 不降 → 查路径项所得占比 |
| `Policy/raw_log_std_mean` vs `Policy/log_std_max` | 跟随墙面分段下降，且**低于**墙面 | 贴着墙面不动 → 调度未生效 |
| `Loss/explained_variance` | > 0.5 | 长期 < 0.2 → critic 跟不上（Phase 2 再处理） |
| `Episode_Termination/base_contact` | 持续下降 | 回升 > 0.4 → 停 |
| `Episode_Termination/time_out` | < 0.15（90 s） | > 0.25 → 查路径长度分布 |
| `Metrics/robot_goal/success_rate` | ≥ 重评基线 + 0.10 | — |

> `Metrics/*` 是**在每个 env reset 的那一步采样**的（CommandTerm 的既有行为），所以它们是"回合结束时刻"的
> 快照，不是全程均值；要看干净的 `d_path` 分位数请用 `scripts/eval_policy.py`。

---

## 7. 复现与回退

- 全量训练（从头，G6）：

```bash
./train_b2w_pathaware.sh --task Isaac-Nav-MDPO-B2W-v0 --num_envs 1046 \
  --max_iterations 4000 --seed 60 --run_name b2w_pathfix_v1 --headless --no_debug_vis
```

> `--no_debug_vis` 关闭命令项的调试图元：每个 env 每段路径一个 cuboid（1046 envs × 14 段 ≈ 1.5 万个实例），
> 而且这些图元的原型 USD 要从 Nucleus 拉取。本机当前无法访问 `omniverse-content-production.s3...`，
> 开着可视化会卡在每个资产 300 s 的超时上。

- 回退：每个 Stage 一个 commit；回退到 `docs/agent_doing.md` 之前的状态即可恢复旧管线
  （`path_pipeline="legacy"` + 旧终止 + 旧奖励表本来就是默认值，所以回退只需回退 B2W 配置里的覆盖）。

---

## 8. 下一步建议

1. **先跑 1500 iter 再决定**。4000 iter 要几天，1500 iter（约 1 天）就足以看出方向：
   `path_progress` 是否非零、`distance_to_path` 是否开始下降、`base_contact` 是否下降。
2. **验收只用同一套评测口径**：`scripts/eval_policy.py`（新环境，不带 `--legacy_env`）× 200 局 × 60 s，
   与本文第 5 节的基线对照。标准是 **SR ≥ 基线 + 0.10**，同时 `median d_path < 0.8 m`。
3. **两组成对指标必须同升**：窄门通过率 ↑ 且 挡路绕行成功率 ↑；只有一组升说明是"过度泛化"而不是学会跟路径。
4. **如果 1500 iter 时 SR < 基线 − 0.15**：停，按规格 §10 G4 回滚到最近一个有问题的 Stage 定位。
   最可能的两个原因：门控把支付全挡了（看 `path_progress` 恒 0）或路径项权重压过了目标项（看 `d_path` 下降但 SR 不升）。
5. Phase 2（规格 §14）等验收通过后再做：挡路率 30 → 35%、噪声上限 0.35 → 0.5 m、TanhNormal + 雅可比修正
   替换移动墙调度。**不要**在 Phase 1 里做地形课程或全局乱噪。
