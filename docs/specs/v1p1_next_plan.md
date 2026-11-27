# B2W Path-Aware V1.1 下一步计划（据 `docs/specs/agent_do_qwen.md` 梳理）

> 输入：`docs/specs/agent_do_qwen.md`（V1.1 执行手册）、`docs/reports/b2w_pathfix_v1_final_report.md`（v1 全量分析）。
> 本文件只做**梳理与裁决**：哪些条目直接执行、哪些被证据推翻、哪些按用户判定改道。
> 两个新前提来自用户：**① 出生朝向对齐路径不可行（不做）；② 楼梯问题必须先查明再改。**

---

## 0. 一页结论

| qwen 文档的条目 | 裁决 |
|---|---|
| P-1 旅馆：三项接近奖励 **weight → 0**（不是速度门） | ✅ **执行**（我用 4000 iter 数据复核，速度门确实只摊薄不封死；`RewardManager` 在 `weight==0` 时 `continue`，已核实 `reward_manager.py:146`） |
| P-2：`heading_progress_to_path` 势函数 + 2 维朝向观测 | ✅ **执行**，但**接口要改写**：`get_local_path_point()` / `get_remaining_path_length()` **不存在**，要用现有的 `get_path_local()`（取末点方向）与目标距离做淡出 |
| P-3 楼梯：`seg_passable` 用端点高差 → 楼梯弦必然 invalid → 改逐点检查（#6） | ❌ **推翻**。`segment_passable` 本来就是**逐格**判定（`prm.py:266-270`），实测 5×0.2 m 楼梯弦 `max_cell_dh = 0.20 m ≤ 0.25`，`pass=True`、invalid 0。**#6 取消** |
| P-3 楼梯是"能力问题、被奖励锁死" | ✅ 接受；并新增 **D1b（低层爬梯能力）** 作为阻塞诊断 |
| #7 拐角保持重采样 | ⚠️ 保留但降级为 P2：拐角确实会把弦推到可通行极限（实测 L 形走廊 1/14 段 clearance 0.30 < 0.31） |
| §4.1 `relax` 方向反了 | ✅ 执行 |
| §4.2 σ 日程与 4000 iter 错配 | ✅ 执行 |
| §4.3 条件完成率 | ✅ 执行（最重要的早期闸门） |
| 出生朝向对齐路径起始切线 | ⛔ **不做**（用户判定不可行）。P-2 只靠奖励 + 观测补梯度 |

---

## 0.5 P0 执行结果（2026-11-26 已完成，详见 `docs/reports/v1p1_p0_diagnosis.md`）

| 诊断 | 结果 | 对计划的影响 |
|---|---|---|
| **D1** | **通过**：低层原地转身 0.28 / 0.55 / 0.98 rad/s（指令 0.3/0.6/1.0），5 s 漂移 ≤0.27 m | **§5 的"弧线转弯"备选取消**；P-2 是奖励/观测问题 |
| **D1b** | 恒定前向指令 8 s 最多爬 +0.18 m（约一级）；0.8 m/s 时走 7 m 却只升 0.13 m（侧滑而非爬升）。但旧策略 stairs SR 0.403 → 系统能爬 | 楼梯是"需学习的动作序列"；P1 的朝向项只补一部分，课程列为后续 |
| **D2** | 楼梯弦本来就 pass（逐格判定） | #6 取消（见上表） |
| **D3** | **已修复（2026-11-26）**：根因是 `compute_free_mask` 只认 ground(0-10)∪platform(195-205)，楼梯踏面(40/80/120/160)整段不在 free mask 里 → clearance=0。改为 `[GROUND_MIN, PLATFORM_MAX]` 后 **invalid 8.14 → 0.95**，platform/tread free 占比 1.00。已合入 P1 | 已合入 P1 commit `b736b4a` |
| **D3b** | `dir_z` 中位 0.37、仅 3% > 0.5 | "楼梯像墙"排除 |
| **D4** | 120–180° 档 SR 0.070 vs 0.119，但 **freeze_frac 0.209 vs 0.000–0.015（15–30 倍）** | P-2 成立，被旅馆掩盖；#5 仍要做 |
| **D5** | freeze 82/200 局 >5%；**倒走 181/200 局 >5%** | 倒走是主要替代策略 |
| **D6** | 冻结时 91% 仍在 0.8 m 支付门内 | **"不可逆悬崖"机制被推翻**，确认是梯度缺失 |
| **D7** | 楼梯 pitch p95 中位 0.1°（max 21.2°） | 不动 40° 阈值 |
| **D8** | 完成回合付满计划路径的 98% | 路径支付有效，问题只在终端行为 |

---

## 1. 楼梯：本轮已完成的离线核查（把"查明"先做掉一半）

工具：新增 `sru-navigation-sim/scripts/probe_path_geometry.py`（**不启动 Isaac**，用真实 `Prm` / `GlobalPath`，
对 stub 掉的 `isaaclab.utils.math.transform_points` 只做恒等）。

```bash
/home/yichangfeng/miniforge3/envs/sru-pathaware/bin/python \
  sru-navigation-sim/scripts/probe_path_geometry.py
```

### 1.1 结果

| 场景 | invalid 段 | pinch 段 | 跨楼梯弦的最大**逐格**高差 | 判定 |
|---|---:|---:|---:|---|
| 平地走廊 | 0 / 14 | 0 / 14 | 0.00 m | pass |
| **5 级 × 0.2 m 楼梯**（0.4 m 踏面，4 m 宽走廊） | **0 / 14** | 0 / 14 | **0.20 m**（阈值 0.25） | **pass** |
| L 形 90° 拐角走廊 | **1 / 14** | 2 / 14 | 0.00 m | 拐角处 clearance 0.30 m < 0.31 → invalid |

### 1.2 三条结论

1. **楼梯弦不会因为高差被判不可通行。** `segment_passable` 沿 Bresenham 逐格比较相邻格高差
   （`prm.py:266-270`），每跨一级只有 0.2 m ≤ `max_height_diff` 0.25 m。
   qwen 文档 §3.3 的"1.5 m 弦跨 0.67 m 高差 → 必然 invalid"是**按端点高差推的**，与实现不符。
2. **`invalid_seg_frac = 0.22` 的来源不是楼梯，而是拐角。** L 形走廊里 LOS 平滑会把弦贴到可通行极限，
   重采样再把某一段压到 0.30 m（< 0.31）就记为 invalid。真实迷宫走廊更窄，这类段只会更多。
   剩余来源仍是 `highly_non_optimal`/`infeasible` 两类非 LOS 路径与注入事件——**需要用 D3 在真实 tile 上分桶确认**。
3. **路径的 z 是折线插值，坡度取决于航点怎么落。** 直走廊楼梯这个场景里，LOS 把整段压成一条弦，
   z 在 18 m 内线性爬升 1.0 m → 每段 `dz 0.07 m / 1.29 m`，`dir_z ≈ 0.05`，看起来像平路。
   **但若楼梯正好落在两个航点之间**（用户描述的"左转 → 上楼梯 → 右转"就是这种），一个 1.4 m 弦要吃
   1.0 m 高差 → `dir_z ≈ 0.6`，观测上就和"前方向上的墙"接近。**这条假说没有被排除**，
   需要在真实 tile 上量（见 D3b），不要先入为主地当成已解决。

### 1.3 对 P-3 的含义

楼梯冻结**不是**路径几何判错导致的（已排除）。剩下两条路，必须先分清：

- **能力**：B2W 到底能不能爬上 0.2 m 踢面、以及在什么进近角/速度下能爬 → **D1b**（阻塞）。
- **梯度**：上楼梯的动作前缀（减速 → 对准 → 爬升 → 顶端转向）在当前奖励里几乎全是零收入，
  只有最慢最危险的中段付费 → 由 **#5 朝向势函数** 部分补上，但仍需评估是否足够。

---

## 2. P0 阻塞诊断（不训练；D1/D1b 决定是否还要动执行器/地形）

| # | 测什么 | 怎么测 | 判据与分支 |
|---|---|---|---|
| **D1** | 低层原地转身 | play 里绕过高层直发 `vx=vy=0, ω∈{0.3,0.6,1.0}` 各 5 s | ω=1.0 时实测 yaw rate **< 0.3 rad/s** 或漂移 **> 0.5 m** → 执行器主因，走 §5 备选 |
| **D1b** | **低层爬楼梯**（新增，直接回答楼梯） | 把机器人放在楼梯正前方 1.5 m，分别以进近角 0°/15°/30°、速度 0.2/0.5/0.8 m/s 给恒定指令 8 s | 正对 0.5 m/s 都上不去 → **能力问题**，先降坡度或做地形课程，不靠奖励硬修；只有斜向失败 → 是 §5(c) 的进近角问题 |
| **D2** | `is_traversable` 对楼梯弦 | **已由 §1 答复**（逐格判定，楼梯 pass） | 不需要再测 |
| **D3** | invalid 段分桶 | 真实 tile 上按「拐角 / 楼梯 / 注入 / 非 LOS 类别」统计，并记录**超出门限多少**（clearance 分布） | 拐角占大头且 clearance 集中在 0.28–0.31 → 是重采样离散化噪声（离线复现到 0.30 vs 门限 0.31），加一个小容差即可，别动管线；真低于 0.2 → 是路径确实穿墙，需要修；楼梯占大头 → **上报**（与 §1.2 矛盾） |
| **D3b** | 楼梯处的**局部路径方向**（新增） | 在含楼梯的 tile 上打印 `get_path_local(6 m)` 末尾几点的 `dir_z`，以及楼梯段前后两个航点的 `dz` | `dir_z > 0.5` → "楼梯像墙"成立，把 `dir_z` 压成坡度角单独一维；`dir_z < 0.2` → 排除，回到能力/梯度问题 |
| **D4** | 按初始 \|Δψ_path\| 四档的 SR | eval 逐回合加字段 | 单调下降、120–180° 档 ≈0 → §2 成立；四档平坦 → **上报，P-2 诊断重审** |
| **D5** | `freeze_frac` / `reverse_frac` | ‖v_xy‖<0.1 且 \|ω\|<0.15 的步数占比；vx<−0.1 占比 | 大角度档显著高 → §2 成立 |
| **D6** | 冻结前 3 s 的 `d_path` 轨迹 | eval 记录 | 冻结前已 >0.8 → 门先关、冻结是后果不是原因 |
| **D7** | 楼梯 tile 的 pitch 分布 vs 40° 阈值 | eval 记录 | p95 > 35° → 上报（先不要动阈值） |
| **D8** | `paid_m` 按结局分层 | 完成 / 旅馆超时 / 冻结超时 / 摔倒 | 量化各问题吃掉的路径米数 |

产出：`docs/reports/v1p1_p0_diagnosis.md`（D1–D8 全部数值 + 分支结论）。

**便宜的同批核查**：注入成功率（`blocked_episode` 0.151 vs 期望 0.37）、`repair_corner_chords` 死代码确认、
实际 seg padding 与 pinch 占比。

---

## 3. P1 补丁（一笔 commit，从头重训）

> **三个问题必须同批**：只加朝向项 → 更快到终点然后掉进旅馆；只修旅馆 → 冻结依旧。

| # | 文件 | 改动 | 针对 |
|---|---|---|---|
| 1 | `config/b2w/navigation_env_cfg.py::B2WRewardsCfg` | `reach_goal.weight = 0.0`；`reach_goal_xy_tight.weight = 0.0`；`reach_goal_xy_soft.weight = 0.0` | P-1 |
| 2 | 同上 | 新增 `velocity_at_goal = RewTerm(func=mdp.velocity_at_goal_penalty, weight=-0.2, params={"command_name":"robot_goal","distance_threshold":0.5})`（函数已存在：`rewards.py:265`） | P-1 |
| 3 | 同上 | 重写 docstring：删掉 "during the dwell ~+3" 的错误预算，写明"目标附近正收入恒 0，唯一正事件是 +25" | P-1 |
| 4 | `mdp/terminations.py::time_out_navigation` | 超时成功**恒记 0**；"超时那一刻在圈内"另存 metric | P-1 |
| 5 | `mdp/rewards.py` + cfg | 新增 `heading_progress_to_path`，**weight = 1.0**、`lookahead = 1.5 m`。**接口改写**：`cmd.get_path_local(1.5)` 返回 `(B,15,4)`，前 15 个点按弧长从"当前位置"排到"前方 1.5 m"（`path_generator.py:491-520`）；取**最后一点**的 `dir_xy` 算 `Δψ = atan2(dir_y, dir_x)`（第 0 点在机器人身上、方向退化，不可用）。淡出用目标距离 `‖cmd._get_unscaled_command()[:, :2]‖`（<1 m 时线性淡出到 0），不需要新接口 | P-2/P-3 |
| 6 | ~~`path_generator.py` 改 `seg_passable` 子采样~~ | **取消**（§1：本来就是逐格判定） | — |
| 7 | `path_generator.py` | 拐角保持重采样（转角 >45° 时把最近的重采样点吸附到原始顶点）→ **P2**，带单测 | P-3(c) |
| 8 | `path_generator.py::_compute_relax` | `relax = inv \| ((left \| right) & ~pinch)` | §4.1 |
| 9 | ~~`[cos Δψ_path, sin Δψ_path]` 观测（+2 维）~~ | **取消（用户 2026-11-26 判定）**：部署时单独给 1.5 m 处的角度既不直观也不一定准；路径点本身已在机体系里，角度可由相对坐标推断。观测维度保持 **2697 / 5834** | — |
| 10 | `path_generator.py` / `goal_commands.py` | 首点方向退化修正：`norm < 0.05` 时用该段切向替代 | P-2 |
| 11 | `config/b2w/agents/rsl_rl_cfg.py` | σ 日程压缩到 4000 iter：`log_std_anneal_start=600, mid=2200, end=3600`；`entropy_hold_iters=400, entropy_decay_span=2600` | §4.2 |
| 12 | ~~观测维度同步~~ | **不需要**：#9 取消后维度仍是 2697 / 5834 | — |
| 13 | `goal_commands.py` / runner / `eval_policy.py` | 新增指标（§4） | 度量 |

**注意**：#5 的势函数必须是**势**（整局总额 = 初始朝向误差 − 末态朝向误差，有界），否则就是一个新的刷分项；
以及**必须用 1.5 m 前视的末点**、不能用第 0 点（第 0 点落在机器人身上，方向向量退化，见 #10）。

> ⚠️ **#8（relax）先别急着上**：D3 显示楼梯 tile 有 58% 的段是"假 invalid"（平台不在 free mask 里导致
> clearance≈0）。`relax = inv | ...` 会把这一大段变成免罚，等于在**最需要严格贴线的地方**取消约束。
> 正确顺序：先查清 free/platform mask（半天），再决定 #8 的写法；或给 relax 加一条"只在路径确实
> 穿过障碍（高度失败）时才放宽"的条件。

---

## 4. 指标（训练期 + eval）

**训练期**（最重要排在前面）
1. `Metrics/robot_goal/conditional_completion_rate` = 完成驻留 / 曾入圈 ← **1000 iter 早期闸门**
2. `Metrics/robot_goal/goal_near_income`（改后应恒 0）
3. `path/invalid_raw_frac` 与 `path/relax_mask_frac` **分列**
4. 单步 `progress` 的 max / p99
5. 窄门通过率、挡路绕行成功率（成对）

**eval**（`eval_policy.py`）
- 按初始 |Δψ_path| 四档分层 SR
- `freeze_frac` / `reverse_frac`
- **楼梯**：进楼梯前 3 s 的停留时长、pitch 序列、是否到达楼梯脚（新增"到楼梯脚率"）
- 楼梯段 / 窄门段的 `d_path` p90

---

## 5. 依赖与分支

| 条件 | 处置 |
|---|---|
| ~~D1 失败（转不动）~~ | **已排除**：D1 通过（0.98 rad/s @ 1.0 指令），不做弧线转弯备选 |
| D1b：开环爬不上，但系统能爬（旧策略 stairs SR 0.403） | P1 的朝向项先补梯度；**若 P1 后 stairs SR 仍 ≈0，再做楼梯课程/降坡度**；⛔ 不加"楼梯专用奖励" |
| D4 四档**平坦** | 未触发：120–180° 档 SR 0.070 vs 0.119、freeze 高 15–30 倍 → P-2 成立，照做 #5 |
| D3 楼梯占 invalid 大头 | **已确认**（8.14/14，clearance≈0）→ 追加一项：查 PRM free/platform mask 为什么排除抬高平台（`scripts/p0_stairs_tiles.py`）；**在它修好之前，楼梯 tile 上的 `relax` 修法会把大段路径变成免罚** |

---

## 6. 训练与验收

1. 冒烟：32 envs × 10 iter，核对 `agent.yaml`（2699 维、σ 日程、14 项奖励权重）。
2. 单测全绿（qwen §8.1 U1–U9；U4 楼梯断言按 §1 结果改为"本来就 pass"，U5 拐角吸附在 P2 补）。
3. 从头 4000 iter（奖励语义 + 观测维度都变，value 必须重学）。⛔ 不做 500 iter resume 闸门——旅馆比改后恒为 0，
   检验它是空转。
4. 1000 iter 早期闸门：条件完成率上升、圈内速度下降、120–180° 档 SR 非零、`freeze_frac` 下降。
5. 最终验收（60 s × 200 局，同口径）：

| 指标 | 门槛 | v1 现值 |
|---|---|---|
| SR（驻留） | ≥ 0.58 | 0.105 |
| 条件完成率 | ≥ 0.80 | 0.19 |
| 圈内速度 p50 | ≤ 0.5 m/s | 1.10 |
| median `d_path` | ≤ 0.5 m | 0.36 ✅ |
| timeout share | ≤ 0.25 | 0.71 |
| 目标附近正收入 | = 0 | 34.2 |
| 120–180° 档 SR | ≥ 0–30° 档的 0.5 倍 | 待测 |
| 楼梯 tile SR | 单列上升 | 0.122 |
