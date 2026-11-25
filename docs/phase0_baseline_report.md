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

### 3.3 对 Go2W 的影响

- `path_pipeline` 默认 `"legacy"`，`progress_along_path` 的 `per_metre` 默认 False，`at_goal_navigation` 的
  `reset_on_exit` 默认 False，`time_out_navigation` 的 `strict_success` 默认 False → **Go2W 的普通任务行为不变**。
- `Go2WNavigationEnvCfg_Paper` 继承 `B2WNavigationEnvCfg`（本仓库原有设计），因此它**会**继承 v2 管线、
  90 s horizon、新奖励与 2697/5834 维观测；其 runner cfg `Go2WNavMDPORunnerPaperCfg` 继承 B2W 的
  `path_length=120`，两侧一致，不会静默错位。若要保留旧 Paper 对照，应显式回退这几个字段。

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

---

## 5. 当前状态

- 代码：Stage A–F 已实现，单元测试全绿。
- 正在跑：冒烟训练（10 iter）。
- **待办**：Phase 0.5 基线重评（`scripts/eval_policy.py`，model_6000，60 s / 90 s 各 200 局）；
  Stage G 的 4000 iter 全量训练需要数天，未启动。

### 基线重评命令

```bash
cd IsaacLab && ./isaaclab.sh -p ../sru-navigation-sim/scripts/eval_policy.py \
  --task Isaac-Nav-MDPO-B2W-Play-v0 \
  --checkpoint /home/yichangfeng/sru-path-aware/IsaacLab/logs/rsl_rl/b2w_navigation_mdpo/2026-11-24_09-05-54_b2w_sigma_bounded/model_6000.pt \
  --episodes 200 --episode_length_s 60 --num_envs 20 \
  --out ../docs/phase0_baseline_60s.md --headless
```

> ⚠️ 旧日志的 `success_rate` 0.504 是"碰边锁存"口径，**不可**作为新基线对照。

---

## 6. 复现与回退

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
