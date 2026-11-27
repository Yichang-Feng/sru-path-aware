# B2W Path-Aware 导航重构执行手册 v1.0（冻结版）

> **文档性质**：本规格经过六轮设计评审后冻结，供执行 Agent 直接实施。
> **Agent 职责边界**：按 Stage 顺序**实现**本规格，遇阻塞项**上报**，⛔ 不得自行重新设计机制、不得"优化"参数、不得跳过单测。
> **若发现规格内部矛盾**：停止该 Stage，输出矛盾点与代码证据，等待裁决。

---

## 0. 背景速览（只读，用于理解决策动机）

### 0.1 现状
- 工作区：`/home/yichangfeng/sru-path-aware`，conda env `sru-pathaware`
- 当前 run：`IsaacLab/logs/rsl_rl/b2w_navigation_mdpo/2026-11-24_09-05-54_b2w_sigma_bounded`，iter≈6020/15000，SR 0.50 停滞
- 现象：忽视路径、直奔目标、死胡同撞墙、楼梯前徘徊、窄门前转身避让

### 0.2 已确诊的根因
| # | 根因 | 证据 |
|---|---|---|
| R1 | 直连目标奖励 1.5/m vs 路径奖励 0.125–0.25/m，路径被压成舍入误差 | 每 episode +15.7 vs +0.70 |
| R2 | `at_goal_navigation` 用 `time_at_goal>0` 锁存，**碰边即成功**，不要求停稳 | `terminations.py` |
| R3 | `log_std` 顶死 clamp 上界（熵项恒正推 + tanh 在环境侧、log_prob 无雅可比修正） | 原始 log_std=0.201–0.205 |
| R4 | 熵退火/LR floor 配置未启用；`schedule="adaptive"` 在 MDPO 中**未实现** | `mdpo.py::_update_learning_rate` |
| R5 | 60s 时限 × 0.59 m/s ≈ 35m，绕路必超时 | 超时率 0.22 |
| R6 | 30–40m 压 15 点 → 间距 2–2.7m，窄门（0.3–0.8m）内无路点 | 等价于失效的 2m carrot |

### 0.3 部署规范（决定所有设计取舍，优先级高于论文）
> **机器狗应大体严格跟随路径。路径几何上总是可行的（含窄门、窄通道），只是可能蹭墙、余量不足。需要的是"遇新障碍局部让行"，不是大幅截弯取直。**

由此推出三条不可动摇的设计原则：
1. **蹭墙 ≠ 不可行**。窄门必须保持**最严格**约束，否则重现"看到门框就转身"。
2. **不制造穿墙航点**作为主分布。真机鲁棒性靠"局部挡路事件"覆盖，不靠全局乱噪。
3. **不奖励大截弯取直**，但**不得饿死正确绕障**。

---

## 1. 全局约束（贯穿所有 Stage）

| # | 约束 | 违反后果 |
|---|---|---|
| **G1** | `MultiPath.get_all_path_metrics()` **非幂等**（消耗内部状态）。全仓库每 sim step **只允许调用一次**，结果缓存共享给奖励/惩罚/日志 | 第二次调用静默返回 0 |
| **G2** | `RewardManager.compute()` 对每项 **×dt(0.2s)**。一次性奖励必须 `/env.step_dt` | +25 变 +5 |
| **G3** | `Episode_Reward` 日志除以 `env.max_episode_length_s`。horizon 改 90s 后还原系数 **×90** | 回报分解全错 |
| **G4** | 新增 cfg 字段必须**同时**声明在 `RslRlPpoAlgorithmCfg`/`RslRlPpoActorCriticCfg` dataclass **和** `MDPO.__init__` 显式签名中 | 只赋值不声明 → `**kwargs` 静默丢弃（Nov24 事故） |
| **G5** | ⛔ `schedule="adaptive"`（MDPO 只有 fixed/linear/cosine/exponential，填 adaptive 会静默冻结 LR） | 训练不动 |
| **G6** | ⛔ 新基线**从头训练**，不从 `model_6000.pt` 续训 | 旧 critic 价值错位 + 旧 actor "忽略路径"先验 |
| **G7** | 实施 Stage A 前必须 grep 确认 `ManagerBasedRLEnv.step()` 中 `command_manager.compute()` 在 `reward_manager.compute()` **之前** | 驻留脉冲时序错误 |
| **G8** | 每 Stage 一个 commit；每 Stage 结束跑冒烟 `--num_envs 32 --max_iterations 10 --headless`，exit 0 且无 traceback 才进下一 Stage | 不可回退 |
| **G9** | ⛔ 修改上游 `path_generator.py` 的**默认值**；所有覆盖只在 B2W 配置中做 | 污染 Go2W |

---

## 2. 冻结参数总表（唯一真值）

### 2.1 路径生成
| 参数 | 值 |
|---|---|
| `category_weights` | `[0.45, 0.25, 0.20, 0.10]` |
| 全局噪声位置 | 重采样**前**的稀疏路点 |
| 全局噪声幅度 | `min(0.35, 0.5 × clearance(p_i))` 米，逐点 |
| 噪声保底 | 拒绝采样（≤5 次）保证点与前后段可行；失败则保留原点 |
| 端点 | **始终锁定**（首=spawn，尾=goal） |
| 拐角弦修复 | 不可行弦 **50% 插回原始拐角点 / 50% 保留并标 invalid**（仅非 pinch 邻域） |
| 挡路事件概率 | `0.15 → 0.30`（iter 0–2000 线性），每局 ≤1 处 |

### 2.2 可行性判据
| 参数 | 值 |
|---|---|
| `seg_passable` padding | **实测机身半宽 + 0.05 m**（⛔ 不是 PRM 的 0.4） |
| `seg_comfortable` padding | 0.4 m（PRM 原值），**仅日志** |
| 计算时机 | 加噪/注入**之后**、`z_offset` **之前**，对最终 15 点的 14 段 |
| `pinch_mask` | 段上 `clearance < 0.45 m` |
| 注入段标记 | **显式强制 False**，不依赖 `is_traversable` 返回 |
| 膨胀 | 前后各 1 段，**切片拼接**（⛔ `torch.roll`），`& ~pinch_mask` |

### 2.3 进度算法（米数制）
| 参数 | 值 |
|---|---|
| `SYNC_RATE` | 1.0 m/步 |
| `CONT_TOL` | 0.5 m |
| `PAY_CAP` | 1.5 m/步 |
| `EPS` | 0.04 m/步 |
| 跨段分类 | 内部**全无效** → 立即同步；**含有效段** → 限速 1 m/步 |
| `s_reward` | 门闭时**也更新**（⛔ 永不记欠账、永不补发） |

### 2.4 三档门 + 走廊
| 段类型 | `seg_gate`（支付门） | 走廊惩罚 |
|---|---|---|
| 正常有效段 | **0.8 m** | 开 |
| 膨胀邻居段（dilated & ~invalid） | **1.2 m** | 关 |
| 真无效段（invalid） | **max(1.6, push + 0.6)** ≈1.6–1.9 | 关 |

走廊惩罚：`weight = −0.15`，`clamp(d_path − 0.5, 0, 1.5) × (relax_mask == 0)`

### 2.5 奖励表
| term | 旧 | **新** |
|---|---|---|
| `progress_along_path` | 0.05（百分比） | **1.2（米数）**，probability **1.0** |
| `progress`（直连） | 1.5 | **0.15** |
| `reach_goal` | 10.0 | **2.0** /步 |
| `goal_reached_bonus` | — | **25.0** 一次性（`/step_dt`） |
| `reach_goal_xy_tight` | 1.5, T_r 0.1 | **0.75, T_r 0.2**，arrive_mask **瞬时** |
| `reach_goal_xy_soft` | 0.25 | 不变 |
| `path_deviation_penalty` | — | **−0.15** |
| `time_out_penalty` | −100 | **−50** |
| `action_rate_l1` / `lateral_movement` | −0.02 | 不变 |
| `joint_acc_l2_joint` / `episode_termination` | −1e−7 / −50 | 不变 |

### 2.6 终止
| 项 | 值 |
|---|---|
| 驻留 | **连续 1.5 s（8 步）**，离开半径**清零**，终止时须在半径内 |
| 超时成功 | 超时**当步瞬时** `d < 0.5`（旧口径另存日志） |

### 2.7 观测（actor 2636 → **2697**）
| 项 | 维度 |
|---|---|
| 全局 15 点（保留原样） | 60 |
| **局部：投影点前方 6 m 重采样 15 点** | **+60** |
| **`distance_to_path` 标量** | **+1** |

### 2.8 训练
| 参数 | 值 |
|---|---|
| `episode_length_s` | **90.0**（`Episode_Reward` 还原 ×90） |
| LR | `cosine`，1e-3 → **floor 1e-4** |
| 熵 | `cosine` 0.001 → **0**，hold 500 / span 3500 |
| `log_std_max` | **调度钩子**：0.1 →(1000–5000 cosine)→ −0.6 →(5000–8000)→ −1.0 |
| `log_std_min` | −2.5（不变） |
| 起点 | **从头训**，seed 60 |
| 目标 iter | 4000（1500 中期检查） |

---

## 3. Stage 0：阻塞性实测与基线重评

> ⚠️ **本 Stage 的 5 项实测是后续所有 Stage 的前置条件。未测出即写代码 = 把猜测硬编码进奖励函数。**

### 0.1 立即动作
```bash
# 停掉当前 run（iter≈6020，边际收益已极低）
# 保留 model_6000.pt 作为评测对照
```

### 0.2 五项阻塞性实测（产出报告，不写训练代码）

| # | 测什么 | 从哪测 | 喂给哪个参数 |
|---|---|---|---|
| M1 | **B2W 实际机身宽度** | `assets/b2w.py` / USD bbox | `seg_passable` 的 padding = 半宽+0.05 |
| M2 | **迷宫通道净宽分布** | `terrains/maze_config.py` + 高度场统计 | 验证走廊 r=0.5 是否安全 |
| M3 | **`prm.free_nodes` 采样是否已含 padding 膨胀** | `mdp/navigation/prm.py` | 判断窄门误判范围 |
| M4 | **`is_traversable` 是否用端点 z 做高差判断**；端点落在障碍内时返回 False 还是抛异常 | `prm.py` | 决定显式标记的必要性 |
| M5 | **optimal 类别下 `check_padding=True` 的 False 段比例**，按窄门/开阔地分桶 | 离线生成 100 条路径统计 | 校准 padding 阈值 + 报警线 |

### 0.3 附加核查
- **G7**：grep `ManagerBasedRLEnv.step()` 确认 command → reward 顺序
- `DeltaTransformPathNoiseCfg(rotation=0.1, translation=0.5, noise_prob=0.1)` 是**整条刚体变换**（≈定位误差，保留）还是**逐点独立**（≈锯齿，需降至 0.2）
- `CurriculumCfg` 中是否存在 `update_terrain_levels` 调用接线（只报告，不开启）
- `hf_terrains_maze_cfg.py` 中 stairs 踏步宽/高

### 0.4 诊断日志接入

**文件**：`mdp/navigation/goal_commands.py`
```python
# _update_command() 末尾（_update_distance_tracking() 之后）：
d, p = self.paths_manager.get_all_path_metrics()   # 全 step 唯一一次调用（G1）
self._cached_path_dist = d
self._cached_path_progress = p

# get_path_metrics() 改为返回缓存；缓存为空时现算一次
def get_path_metrics(self):
    if self._cached_path_dist is None:
        self._cached_path_dist, self._cached_path_progress = \
            self.paths_manager.get_all_path_metrics()
    return self._cached_path_dist, self._cached_path_progress
```
- `_init_metrics()` 注册 `distance_to_path`；`_update_metrics()` 写入
- `MultiPath.update()` 返回值（全局平均偏离）接入 debug 对照

**文件**：`rsl_rl/algorithms/mdpo.py` + `runners/on_policy_runner.py`
- explained variance：`1 − Var(returns−values)/Var(returns+eps)`，两 storage 取均值 → `Loss/explained_variance`
- `Policy/raw_log_std_mean`（clamp **前**）与现有 `mean_noise_std`（clamp 后）并列
- `Loss/entropy_coef`、`Loss/clip_param` 当前值

**文件**：`terminations.py::at_goal_navigation`（临时日志，不改逻辑）
- 终止时 `xy_error >= 0.5` 的比例 → `Metrics/latched_success_frac`

### 0.5 基线重评（⚠️ 决定所有验收对照）
用 `model_6000.pt`，在 **{60s, 90s}** 两种 horizon 下各跑 200 局，按 tile 类型（maze / non_maze / stairs / pits）分桶记录：SR、SPL、初始直线距离、路径长 L、`d_path` 分位数。

> ⚠️ **基线口径警告**：`model_6000` 在"碰边锁存"下训成。换成连续驻留后 SR **必然下降**（可能 0.35–0.45）。**这是挤水分，不是回归。** 所有后续对比一律用重评值，⛔ 不用旧日志 0.504。

### 0.6 Play 工具增强
- `--policy_idx {1,2}`：分别加载双策略（训练日志是两者混合，play 只用 1）
- `--ablate_path`：`target_path` 观测置零，测动作 L2 变化量（路径反事实测试）

**产出**：`docs/reports/phase0_baseline_report.md`

---

## 4. Stage A：终止与终点奖励修复

### A1. 连续驻留重构
**文件**：`goal_commands.py`
```python
# _init_tracking_buffers:
self.required_steps_at_goal = int(round(1.5 / self.env.step_dt))   # 8 步
self.goal_dwell_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

# _update_command() 末尾：
xy_err = torch.norm(self.goal_command_body_unscaled[:, :2], dim=1)
in_radius = xy_err < 0.5
self.steps_at_goal = torch.where(in_radius, self.steps_at_goal + 1,
                                 torch.zeros_like(self.steps_at_goal))   # 离开即清零
self.time_at_goal[in_radius] += self.env.step_dt
self.goal_dwell_completed = (self.steps_at_goal == self.required_steps_at_goal)  # 单步脉冲
```

**文件**：`terminations.py::at_goal_navigation`
```python
termination = goal_cmd.steps_at_goal >= goal_cmd.required_steps_at_goal
# ⛔ 删除 already_at_goal = time_at_goal > 0 的 OR 锁存
# ⛔ 删除函数内的计数累加（已移入 command）
# 保留 success buffer 记 1 的逻辑
```

> **联动说明**：`required_steps_at_goal` 同时被 `reach_goal_xyz` 的 timeup 窗口引用（最后 4s→1.6s），属预期，在 commit message 注明。
> **G7 备选**：若 command 在 reward 之后，脉冲改由 reward 函数自行从 `steps_at_goal` 判断，termination 只读 `>=`。

### A2. 超时口径 + arrive_mask
- `time_out_navigation`：成功判据 `time_at_goal > 0` → **超时当步 `xy_err < 0.5`**；新增 `Metrics/timeout_ever_entered_frac` 保留旧口径
- `rewards.py::reach_goal_xyz`：`arrive_mask = time_at_goal > 0.0` → **瞬时 `xyz_error < 0.5`**

### A3. 一次性到达奖励
```python
def goal_reached_bonus(env, command_name="robot_goal"):
    g = env.command_manager.get_term(command_name)
    return g.goal_dwell_completed.float() / env.step_dt   # G2
```

### A4. 单测（全过才进 Stage B）
脚本驱动一个 env：进圈 → 出圈 → 再进圈
1. 出圈后 `steps_at_goal == 0`
2. 仅**连续 8 步**在圈才触发终止
3. bonus 只在终止步出现一次
4. `latched_success_frac` 恒 0

---

## 5. Stage B：路径管线

### B1. clearance map（构建期，per-tile）
```python
from scipy.ndimage import distance_transform_edt
clearance = distance_transform_edt(~obstacle_mask) * horizontal_scale   # 米
```

### B2. 全局噪声整形
**文件**：`path_generator.py::GlobalPath.find_path`
- 位置：**重采样前**的稀疏路点（平滑后）
- 幅度：`a_i = min(0.35, 0.5 * clearance(p_i))`
- 拒绝采样 ≤5 次：候选点与前后段均 passable 才接受，否则保留原点
- **端点锁定**：`noisy[0] = path[0]; noisy[-1] = path[-1]`

### B3. 拐角弦 50/50
重采样后，对每条弦检查可行性：
- 不可行且**非 pinch 邻域** → **50% 插回原始拐角点修复 / 50% 保留并标 invalid**
- 保留者提供"两端点自由、中间段被挡"的签名（≡ 真机独立箱子挡路）

### B4. 局部挡路事件注入
```python
候选 i ∈ [2, 12]，须同时满足：
  0.30 ≤ clearance(p_i) ≤ 1.00           # 够近，推得进去
  反方向自由宽度 ≥ 1.5 m                   # 有绕行空间
  max|Δz| over p_{i-1..i+1} ≤ 0.15 m      # 排除楼梯/坑边
  与任何 pinch 段弧长距离 ≥ 2 段            # 保护窄门

push = clearance(p_i) + 0.30              # ⛔ 不用固定 0.8–1.2
方向 = 垂直局部路径方向、朝 clearance 更小的一侧
校验 = 推后 clearance(new_xy) == 0（确实进墙），否则放弃本次事件
z    = 保持不变                            # ⛔ 严禁重读 height_map（否则路点飞到墙顶）
```
概率 `p = 0.15 + 0.15 * min(iter/2000, 1)`，每局 ≤1 处。

### B5. 可行性标记（时机关键）
```python
# find_path() 内：noise + 注入之后、z_offset 列表推导之前
pts = path                                # (15,3) heightmap 坐标，未抬升
seg_passable = np.array([
    prm.is_traversable(pts[i], pts[i+1], check_padding=<M1 半宽+0.05>)
    for i in range(len(pts)-1)
], dtype=bool)                            # (14,)

# 注入段显式强制（不依赖 is_traversable 语义，见 M4）
if evt is not None:
    i = evt[0]
    seg_passable[i-1] = False
    seg_passable[i]   = False

seg_comfortable = ... check_padding=4 ...  # 仅日志
pinch_mask = (段上 clearance < 0.45)
```

`MultiPath` 新增张量：`seg_passable (B,14) bool`、`pinch_mask (B,14) bool`、`seg_gate (B,14) float`、`push_per_seg (B,14) float`；`rebuild()` 中逐 env 拷入。

### B6. 双侧膨胀（⛔ 严禁 torch.roll）
```python
inv   = ~self.seg_passable                                  # (B, S=14)
left  = torch.cat([torch.zeros_like(inv[:, :1]), inv[:, :-1]], 1)
right = torch.cat([inv[:, 1:], torch.zeros_like(inv[:, -1:])], 1)
relax = (inv | left | right) & (~self.pinch_mask)
```
> **为何双侧**：深度预览 10 m，机器人在障碍前 1–2 m 就侧移，此刻投影仍在**上游有效段**。只扩下游 → 提前避障被罚 → 训出"贴脸急转"。

### B7. 三档门（构建期算好）
```python
gate = torch.full_like(inv, 0.8, dtype=torch.float32)
gate = torch.where(relax & ~inv, torch.full_like(gate, 1.2), gate)
gate = torch.where(inv, torch.maximum(torch.full_like(gate, 1.6),
                                      self.push_per_seg + 0.6), gate)
self.seg_gate = gate
```

### B8. 稠密路径（供 Stage E）
```python
path_dense_w : (B, M=64, 3)   # 世界系，等弧长重采样，间距 0.5–0.6 m
cum_dense    : (B, M)
```

### B9. 单测
1. 首尾点与 spawn/goal **精确重合**
2. 注入段必 `passable=False`
3. **窄门段必 `passable=True`**（这是本轮最重要的回归测试）
4. 注入后 z 与注入前**完全一致**
5. 末段 invalid 时，第 0 段**不受传染**（验证无 roll）

---

## 6. Stage C：进度算法（米数 + 限速支付）

**文件**：`path_generator.py::MultiPath.get_all_path_metrics` 重写

```python
SYNC_RATE, CONT_TOL, PAY_CAP, EPS = 1.0, 0.5, 1.5, 0.04

# ---- 全局最近投影（保留原计算，供 d_path 与观测）----
d_glob, idx_glob = torch.min(dist, dim=1)
s_glob = (cum_start + t_clamped * seg_lens).gather(1, idx_glob[:, None]).squeeze(1)

# ---- 跳变区间内部是否含有效段（向量化，无 python 循环）----
S = seg_lens.shape[1]                                    # 14
idx_track = (cum_end < self.s_track[:, None]).sum(1).clamp(max=S - 1)
vc = torch.cat([torch.zeros(B, 1, device=dev),
                torch.cumsum(self.seg_passable.float(), 1)], 1)      # (B, 15)
lo = torch.minimum(idx_track, idx_glob)
hi = torch.maximum(idx_track, idx_glob)
n_valid_interior = (vc.gather(1, hi[:, None])
                    - vc.gather(1, (lo + 1).clamp(max=S)[:, None])).squeeze(1).clamp(min=0)
jump_over_invalid = (n_valid_interior <= 0) & ((s_glob - self.s_track).abs() > SYNC_RATE)

# ---- 限速双向同步（不依赖窗口 argmin → 不可能死锁）----
rate = torch.where(jump_over_invalid, 1e9, SYNC_RATE)
s_track_new = self.s_track + torch.clamp(s_glob - self.s_track, -rate, rate)

# ---- 支付门 ----
gate = self.seg_gate.gather(1, idx_glob[:, None]).squeeze(1)
payable = ((s_glob - s_track_new).abs() <= CONT_TOL) & (d_glob <= gate)

# ---- 棘轮支付；超额吞掉，⛔ 永不记欠账 ----
advance  = torch.clamp(s_track_new - self.s_reward, 0.0, PAY_CAP)
progress = torch.where(payable & (advance > EPS), advance, torch.zeros_like(advance))
self.s_reward = torch.maximum(self.s_reward, s_track_new)   # 门闭时也更新 —— 命门
self.s_track  = s_track_new

return d_glob, progress
```

**状态**：`s_track`、`s_reward` 在 `build_all()` / `rebuild()` 中归零（替换原 `last_best_progress`）。

### C1. 五项单测（全过才进 Stage D）
| # | 几何 | 期望累计支付 |
|---|---|---|
| 1 | U 型两臂间距 1 m、臂长 10 m，瞬移到远臂 | **0**（旧实现会跳 ~20 m） |
| 2 | 大绕路 15 m 弦切 | 追赶期 **0**，rejoin 后 ≈(L−15) |
| 3 | 绕障跳 2.6 m（内部全无效） | **1.5**（= PAY_CAP） |
| 4 | 沿路径 100 步 × 0.12 m | ≈**12 m** |
| 5 | 前进 1 m → 后退 1 m → 前进 1 m | 只 **1 m** |

### C2. 附加断言
任意几何下 `|s_track − s_glob|` 单调收敛（无死锁）。

---

## 7. Stage D：奖励与门控

### D1. 新增奖励函数
**文件**：`mdp/rewards.py`
```python
def path_deviation_penalty(env, command_name="robot_goal",
                           corridor_radius=0.5, cap=1.5):
    g = env.command_manager.get_term(command_name)
    d, _ = g.get_path_metrics()                    # 缓存，G1
    relax = g.get_relax_at_nearest()               # (B,) bool，最近段是否在 relax 内
    excess = torch.clamp(d - corridor_radius, 0.0, cap)
    return excess * (~relax).float()
```

### D2. 奖励配置
**文件**：`config/b2w/navigation_env_cfg.py::B2WRewardsCfg`

按 §2.5 表逐项设置。关键项：
```python
progress_along_path = RewTerm(func=mdp.progress_along_path, weight=1.2,
    params={"command_name": "robot_goal", "probability": 1.0})
progress = RewTerm(func=mdp.progress_to_goal, weight=0.15,
    params={"command_name": "robot_goal"})
reach_goal = RewTerm(func=mdp.reach_goal_direct, weight=2.0,
    params={"command_name": "robot_goal", "distance_threshold": 0.5})
goal_reached_bonus = RewTerm(func=mdp.goal_reached_bonus, weight=25.0,
    params={"command_name": "robot_goal"})
reach_goal_xy_tight = RewTerm(func=mdp.reach_goal_xyz, weight=0.75,
    params={"command_name": "robot_goal", "sigmoid": 0.25, "T_r": 0.2,
            "probability": 0.01, "flat": True, "ratio": False})
path_deviation_penalty = RewTerm(func=mdp.path_deviation_penalty, weight=-0.15,
    params={"command_name": "robot_goal", "corridor_radius": 0.5, "cap": 1.5})
time_out_penalty = RewTerm(func=mdp.episode_time_out, weight=-50.0)
```

### D3. 路径分布覆盖（⛔ 不改上游默认，G9）
```python
# B2WNavigationEnvCfg.__post_init__
self.commands.robot_goal.path_cfg.category_weights = [0.45, 0.25, 0.20, 0.10]
```

### D4. 经济学验算（脚本，不训练）
- 站立回报 ≈ **−10**（仅超时罚）
- 完成回报 ≈ **+65**（路径 1.2×25 + bonus 25 + reach ~3 + progress ~2）
- 断言：站立 < 移动

### D5. 更新 docstring
按 **×90** 新口径重写回报分解注释（G3）。

---

## 8. Stage E：观测三项加法

### E1. 局部路径段（向量化）
**文件**：`goal_commands.py` / `path_generator.py`
```python
dense_b = transform_points(self.path_dense_w, inv_pos, inv_rot)      # (B,64,3)
s0 = self.s_track
targets = s0[:, None] + torch.linspace(0, 6.0, 15, device=dev)[None, :]
idx = torch.searchsorted(self.cum_dense, targets).clamp(1, M - 1)
i1, i0 = idx, idx - 1
c0 = self.cum_dense.gather(1, i0)
c1 = self.cum_dense.gather(1, i1)
w  = ((targets - c0) / (c1 - c0 + 1e-9)).clamp(0, 1).unsqueeze(-1)
p0 = dense_b.gather(1, i0.unsqueeze(-1).expand(-1, -1, 3))
p1 = dense_b.gather(1, i1.unsqueeze(-1).expand(-1, -1, 3))
local_b = p0 + w * (p1 - p0)                                          # (B,15,3)
# 复用现有 (单位方向xyz, 归一化log距离) 编码 → (B,15,4) → flatten 60
```

### E2. 观测项注册
- 新增 `target_path_local`（60 维）
- 新增 `distance_to_path`（1 维，读缓存）
- ⛔ **保留全局 15 点**，不得替换（替换会退化成高分辨率 carrot，窄门问题原样带回）

### E3. 网络与导出器同步（易漏，必查）
**文件**：`rsl_rl/modules/actor_critic_sru.py`
- `path_length` 切分逻辑（actor 与 critic 两条路径）
- 两个 PathEncoder 实例的 `masks` / `unpad_trajectories` 处理
- `_ActorCriticSRUExporterSingleCam` / `DualCam` 的 `__constants__`、`path_length`、`num_waypoints`
- `_ActorCriticSRUONNXExporterSingleCam` / `DualCam` 同上

### E4. 单测
导出 JIT/ONNX，与训练前向在同一输入下**数值一致**（atol 1e-5）。

---

## 9. Stage F：训练工程

### F1. 熵退火 + LR（字段声明见 G4）
**文件**：`config/b2w/agents/rsl_rl_cfg.py::B2WNavMDPORunnerCfg.algorithm`
```python
schedule="cosine", learning_rate=1.0e-3, min_learning_rate=1.0e-4,
entropy_coef=0.001, entropy_schedule="cosine",
min_entropy_coef=0.0, entropy_hold_iters=500, entropy_decay_span=3500,
```
（cosine+floor 与熵退火已实现，只需启用。⛔ 不用 `"adaptive"`，G5）

### F2. log_std_max 移动墙调度（新钩子）
**新字段**（`rl_cfg.py` 声明 + `MDPO.__init__` 签名 + `rsl_rl_cfg.py` 赋值）：
```python
log_std_schedule="cosine"
log_std_anneal_start=1000
log_std_anneal_mid=5000
log_std_anneal_end=8000
log_std_max_initial=0.1
log_std_max_mid=-0.6
log_std_max_final=-1.0
```
`MDPO.update()` 在 `_update_clip_param` 后调用 `_update_log_std_bounds(iteration)`：分段 cosine 插值，赋值给 `actor_critic_1/2.log_std_max`。默认 `"fixed"` 不影响任何现有配置。

> ⛔ **不得用"一次性降 clamp"**：`torch.clamp` 在界外**梯度为零**，参数会钉死在新墙上永不回落。必须移动墙面。

顺带给 `B2WNavPPORunnerCfg.policy` 补 `log_std_min=-2.5, log_std_max=0.1`（PPO 消融备用）。

### F3. Horizon
```python
# B2WNavigationEnvCfg.__post_init__
self.episode_length_s = 90.0
```
- 全仓库 grep 硬编码 `60` / `max_episode_length_s`（分析脚本、docs 注释），还原系数改 **×90**
- `B2WNavigationEnvCfg_PLAY` 显式 `episode_length_s = 60.0`（保留论文可比协议）

### F4. 验证
300 iter dev run：
- `agent.yaml` **含全部新字段**
- `Loss/learning_rate` 符合 cosine 且 floor 生效
- `Policy/mean_noise_std` 跟随墙面分段下降
- `raw_log_std_mean` 与 clamp 后值重合或低于墙面

---

## 10. Stage G：训练与验收

### G1. 启动
```bash
# 冒烟
./train_b2w_pathaware.sh --task Isaac-Nav-MDPO-B2W-v0 --num_envs 32 --max_iterations 10 --headless

# 核对 agent.yaml 后全量（⛔ 从头训，G6）
./train_b2w_pathaware.sh --task Isaac-Nav-MDPO-B2W-v0 --num_envs 1046 \
  --max_iterations 4000 --seed 60 --run_name b2w_pathfix_v1 --headless
```

### G2. 监控面板（每 500 iter）
| 指标 | 期望 | 异常处置 |
|---|---|---|
| 单步 progress max / p99 | ≤ **1.5** / 无长尾 | 越界 → Stage C 有 bug，**停** |
| `path/invalid_raw_frac` | 实测基线 ±50%（预期 5–10%） | 越界 → 路径质量退化 |
| `path/relax_mask_frac` | 实测基线 ±50%（预期 12–20%） | 仅监控 |
| `d_path` 中位数 | 1500 iter 起下降，4000 iter **<0.8 m** | 不降 → 查路径项所得占比 |
| 路径项 vs 直连项所得（×90） | 路径项 **>** 直连项 | 反转 → 配平失败 |
| `mean_noise_std` | 跟随 F2 墙面 | 顶墙不动 → 钩子未生效 |
| `explained_variance` | >0.5 | 长期 <0.2 → 记录，Phase 2 处理 |
| `base_contact` 率 | 持续下降 | 回升 >0.4 → **停** |
| `time_out` 率 | <0.15（90s 下） | >0.25 → 查路径长分布 |

> ⚠️ **两个 invalid 指标必须分开**：`invalid_raw_frac` 是**路径质量断言**（膨胀不改变它）；`relax_mask_frac` 是**故意放宽的区域**。混为一谈会让真退化被掩盖。

### G3. 验收（1500 iter 趋势 / 4000 iter 定去留）

**成对指标 —— 必须同升，单升即失败：**
| 指标 | 目标 | 单降的含义 |
|---|---|---|
| **窄门通过率** | ↑ | 降 → 过度泛化，↓block 率 |
| **挡路绕行成功率** | ↑ | 降 → 门控/膨胀失效 |
| 窄门段 `d_path` p90 | **<0.4 m** | 变大 → 门前松绑 |
| 挡路段 `d_path` p90 | **0.6–1.2 m** | 过小→不敢绕；过大→脱缰 |
| 全体 `d_path` 中位数 | <0.8 m | — |
| SR | ≥ **重评基线** +0.10 | — |
| stairs tile SR | 单列上升 | — |
| 死胡同"投影前跳后卡死"计数 | **0** | — |

**定性 play**：开局朝向路径首段而非目标；窄门敢穿；遇挡路能绕并归队；楼梯前徘徊消失。

### G4. Abort 条件
任一 G2 硬异常，或 1500 iter 时 SR < 重评基线 −0.15 → **停**，回滚到最近 Stage 定位。

---

## 11. 禁止事项速查

⛔ `schedule="adaptive"`
⛔ 从旧 checkpoint 续训新基线
⛔ 单独放大 path 权重而不带 Stage C 限速支付
⛔ 一次性降 log_std clamp（界外梯度为零）
⛔ 同 step 二次调用 `get_all_path_metrics`
⛔ `torch.roll` 做掩码膨胀
⛔ 注入路点时重读 height_map 更新 z
⛔ `seg_passable` 用 PRM 的 padding=4（会误判窄门）
⛔ 只膨胀下游一段
⛔ 全局 1.5 m 乱噪 / 制造穿墙航点作为主分布
⛔ 用局部段**替换**全局路径观测
⛔ 用旧日志 SR 0.504 做验收对照
⛔ 新 cfg 字段只赋值不声明
⛔ 改上游 `path_generator.py` 默认权重
⛔ 放物理箱子当障碍（RayCasterCamera + height_scanner + PRM 三重盲，接触传感器却会罚）
⛔ `s_reward` 记欠账 / 补发

---

## 12. 文件索引

| 内容 | 路径（相对 `/home/yichangfeng/sru-path-aware`） |
|---|---|
| B2W 环境/奖励 | `sru-navigation-sim/isaaclab_nav_task/navigation/config/b2w/navigation_env_cfg.py` |
| B2W agent cfg | `sru-navigation-sim/isaaclab_nav_task/navigation/config/b2w/agents/rsl_rl_cfg.py` |
| 基础 rl cfg | `sru-navigation-sim/isaaclab_nav_task/navigation/config/rl_cfg.py` |
| 基础环境 | `sru-navigation-sim/isaaclab_nav_task/navigation/navigation_env_cfg.py` |
| 奖励函数 | `sru-navigation-sim/isaaclab_nav_task/navigation/mdp/rewards.py` |
| 终止函数 | `sru-navigation-sim/isaaclab_nav_task/navigation/mdp/terminations.py` |
| 观测函数 | `sru-navigation-sim/isaaclab_nav_task/navigation/mdp/observations.py` |
| goal command | `sru-navigation-sim/isaaclab_nav_task/navigation/mdp/navigation/goal_commands.py` |
| 路径生成 | `sru-navigation-sim/isaaclab_nav_task/navigation/mdp/navigation/path_generator.py` |
| PRM | `sru-navigation-sim/isaaclab_nav_task/navigation/mdp/navigation/prm.py` |
| 地形 | `sru-navigation-sim/isaaclab_nav_task/terrains/maze_config.py`、`terrain_constants.py` |
| 网络 | `sru-navigation-learning/rsl_rl/modules/actor_critic_sru.py` |
| 路径编码器 | `sru-navigation-learning/rsl_rl/networks/sru_memory/path_encoder.py` |
| MDPO | `sru-navigation-learning/rsl_rl/algorithms/mdpo.py` |
| Runner | `sru-navigation-learning/rsl_rl/runners/on_policy_runner.py` |
| RewardManager | `IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py` |
| 训练/回放脚本 | `sru-navigation-sim/scripts/train.py`、`play.py` |

---

## 13. 执行顺序总结

```
Stage 0（阻塞实测 + 基线重评）
   ↓ 五项实测出结果
Stage A（终止修复）→ 单测 4 项
   ↓
Stage B（路径管线）→ 单测 5 项
   ↓
Stage C（进度算法）→ 单测 5 项 + 无死锁断言
   ↓
Stage D（奖励门控）→ 经济学验算
   ↓
Stage E（观测 + 导出器）→ 数值一致性
   ↓
Stage F（熵/LR/log_std/horizon）→ 300 iter dev 验证
   ↓
Stage G（4000 iter 全量）→ 1500 中期 / 4000 验收
```

**每 Stage 一 commit + 冒烟通过才进下一 Stage。Stage C 与 Stage D 必须同一 commit 或相邻 commit 一起生效。**

---

## 14. Phase 2 预案（验收通过后才执行，本文档不实施）

| 项 | 内容 |
|---|---|
| 挡路率 | 30% → 35% |
| 全局噪声上限 | 0.35 → 0.5 m（**仍 clearance 自适应 + 拒绝采样**） |
| 可选地形升级 | 宽阔平地烘焙孤立立柱进 height field（深度/PRM/接触四者天然一致），再用 B4 推入 |
| 消融臂 A′ | 关闭可行性门控，验证门控是否真的避免"惩罚正确避障" |
| 评测协议 | 2 m 穿墙噪声作 **OOD 压力测试**（记录，不作训练目标）；窄门通过率、门前徘徊时长为主指标 |
| 治本项 | TanhNormal + 雅可比修正，替换移动墙调度 |
| ⛔ 不做 | 全局乱噪、穿墙航点主分布、地形课程（除非 Stage 0 确认接线存在） |
---

# 附录 A：V1.1 当前系统全貌（2026-11-26）

> ## ⚠️ A.0 v1.2 增量（2026-11-27）——本节优先于下面所有小节
>
> 附录 A 正文写的是 **v1.1**（commit `31540fa`）。此后又改了三处，正文对应小节随之过期：
>
> | 项 | A 正文（v1.1） | **当前（v1.2）** | 影响的小节 |
> |---|---|---|---|
> | 观测维度 | 2699 / 5836（+2 维朝向误差） | **2697 / 5834**（+2 已撤销：路径点已含方向信息，显式角度在部署时不可靠） | A.3、A.4、A.12 |
> | `heading_progress_to_path` | `clamp(prev-err)`，实测可达 15.4/回合 | **有界积分器**：`ref = clamp(ref-delta, 0, π)`，整回合 ≤ π | A.5、A.6、A.9 |
> | `time_out_penalty` | 权重 −50（实付 **−10**） | 权重 **−250**（实付 −50）；`episode_terminated` 类一次性项都受 `×dt` 影响 | A.6 |
> | `path_deviation_penalty` | 走廊 0.5 m / cap 1.5 | **走廊 1.5 m / cap 2.0**（0.5 m 不够绕过新出现的障碍） | A.6、A.8 |
> | PRM free mask | `ground ∪ platform` | **`[GROUND_MIN, PLATFORM_MAX]`**（楼梯踏面原本被当障碍） | §2.2、A.8 |
>
> 训练结果：v1.1（`b2w_pathfix_v1p1`）SR 0.10、v1.2（`b2w_pathfix_v1p2`）早期读数见
> `docs/reports/v1p1_training_analysis_2026-11-27.md` 与 `docs/reports/v1p2_early_analysis_2026-11-27.md`。
> 分析工具与判据：`skills/sru-training-analysis/`（会按 run 自己的 `params/env.yaml` 换算，并检查四条不变量）。


> **性质**：本附录是 v1.0 冻结正文的**增量修订**。凡与正文冲突，以本附录为准，具体是：
> §2.5 奖励表、§2.7 观测维度、§2.8 训练配置、以及 §2.2 可行性判据里的 free mask 定义。
> 其余章节（路径管线算法、三档门、进度记账、Stage 划分）保持不变。
>
> 依据：`docs/reports/b2w_pathfix_v1_final_report.md`（v1 全量分析）、
> `docs/reports/v1p1_p0_diagnosis.md`（P0 诊断 D1–D8）、`docs/specs/v1p1_next_plan.md`（裁决）。
> 代码基线：commit `31540fa`（v1.1 P1 全部补丁）。

---

## A.1 训练怎么跑

```bash
# 全量（当前正在跑的那个）
./train_b2w_pathaware.sh --task Isaac-Nav-MDPO-B2W-v0 --num_envs 1046 \
  --max_iterations 4000 --seed 60 --run_name b2w_pathfix_v1p1 --headless --no_debug_vis

# 冒烟（改任何东西之后先跑这个）
./train_b2w_pathaware.sh --task Isaac-Nav-MDPO-B2W-v0 --num_envs 32 --max_iterations 10 \
  --headless --no_debug_vis
```

- **从头训练，禁止 resume**：v1.1 改了奖励语义**和**观测维度（2697→2699），旧 value function 与旧 actor
  维度都对不上，resume 只会把偏见带进来。
- 启动器负责三件事：解析 python 解释器（conda 或 venv 都行）、把本工作区的
  `sru-navigation-learning` / `sru-navigation-sim` **前置到 PYTHONPATH**（否则 `import rsl_rl` 会解析到
  `~/sru-path-aware-official` 那份旧代码）、然后把参数透传给 `isaaclab.sh -p scripts/train.py`。
- 日志：`IsaacLab/logs/rsl_rl/b2w_navigation_mdpo/<时间戳>_<run_name>/`（TensorBoard `events.*` +
  `params/agent.yaml` + `params/env.yaml`），同时写 wandb 项目 `isaaclab_nav_b2w`。
- `--no_debug_vis` 必须开：命令项的调试图元是每个 env 每段路径一个 cuboid（1046×14 ≈ 1.5 万个），
  而且原型 USD 要连 Nucleus，本机连不上会卡在 300 s 超时。

## A.2 时间与规模

| 量 | 值 | 说明 |
|---|---|---|
| 物理步长 | 0.005 s | sim dt |
| 低层 decimation | 4 | 低层 locomotion 50 Hz |
| **策略步长 `dt`** | **0.2 s** | 0.005 × 40（高层 decimation） |
| episode 长度 | **90 s = 450 步** | 训练；Play/评测 60 s（与历史基线可比） |
| 并行 env | **1046** | 单个 RTX 4090，约 13.6 GB 显存；**不能并发第二个 sim** |
| `num_steps_per_env` | 16 | 每 iter 每 env 采 16 步 → **16 736 条/iter** |
| 吞吐 | ≈ 970–1220 fps | 实测 **17.3 s/iter**（v1 是 13.7 s，多了 2 维观测） |
| 4000 iter | ≈ 16–19 h | 1000 iter ≈ 4–5 h |

> **所有 `Episode_Reward/*` 日志 = 每回合该项总和 ÷ 90**（`max_episode_length_s`），所以
> **每回合总量 = 日志值 × 90**；而奖励管理器对每一项都乘 `dt = 0.2`，所以
> **每步实际回报 = weight × 原始值 × 0.2**。这两个换算在全文反复用到。

## A.3 观测与维度（v1.1）

### A.3.1 policy 组 —— **2699 维**

| # | 名称 | 维度 | 说明 |
|---|---|---:|---|
| 0 | `base_lin_vel` | 3 | 机体系线速度 |
| 1 | `base_ang_vel` | 3 | 机体系角速度 |
| 2 | `projected_gravity` | 3 | 姿态 |
| 3 | `last_action` | 3 | 上一步的 `[vx, vy, ω]` 指令 |
| 4 | `target_position` | 4 | 目标在机体系的方向(3) + log 距离(1) |
| 5 | **`path_heading_error`** | **2** | ★ v1.1 新增：`[cos Δψ, sin Δψ]`，到路径 1.5 m 前视点的方位角 |
| 6 | `distance_to_path` | 1 | 到规划路径的横向距离（米） |
| 7 | `target_path` | 60 | 全局路径 15 点 × (dir_xyz, log 距离) |
| 8 | `target_path_local` | 60 | 6 m 前视局部路径 15 点 × 4 |
| 9 | `depth_image` | 2560 | 深度图经预训练 VAE 编码成 64×5×8 的潜特征 |

**proprio 合计 = 3+3+3+3+4+2+1 = 19 维**；path 块 = 120 维；image = 2560 维。

### A.3.2 critic 组 —— **5836 维**

在 policy 之外再加 `time_normalized` (1) 与 `height_scan_critic` (3136 = 64×7×7 的高度图 VAE 潜特征)。

### A.3.3 顺序是硬约束（⚠️ 改观测必须遵守）

网络按**位置**切片，不按名字：

```
actor : [ proprio | path(120) | image(2560) ]
critic: [ proprio | path(120) | time(1) | height(3136) | image(2560) ]
```

所以任何新的 proprioceptive 项必须插在 `target_path` **之前**，任何扩展路径的项必须紧跟 `target_path`。
`config/path_aware.py::insert_obs_terms` 就是干这个的（以 `target_path` 为锚点插入，且幂等）。
`path_heading_error` 就是插在 `distance_to_path` 之前。

> 历史教训：早期版本把新项直接 append，结果落到了 `depth_image` 之后，proprio/path/image 的切片全部错位。

## A.4 网络结构（ActorCriticSRU）

```
actor
  depth(64×5×8) ──┐
                  ├─ self-attn → cross-attn(proprio) → 64 维图像特征
  proprio(19) ────┘
        │
        └→ concat(64 + 19 = 83) → SRU / LSTM 记忆（hidden 512, 1 层）
                                        │
  path(120) → 15×4 → self-attn → cross-attn → path_embed(32)   ← **不过 RNN**
                                        │
                                concat(512 + 32 = 544) → MLP[512,256,128] → 3 维动作 [vx, vy, ω]
```

- 动作维度 3，经动作项 `PerceptiveNavigationSE2Action` 处理：`tanh → ×policy_scaling[1.5,1.0,1.0]
  → 低通滤波（α 随 episode 进度变化）→` 作为速度指令送给低层 locomotion 策略
  （B2W 用 `policy_b2w_new_2.pt`，观测组 `low_level_policy`）。
- **路径不过 RNN，只在 MLP 前拼接**：这意味着路径的时间上下文只能靠观测窗口堆叠，
  这也是为什么 `path_heading_error` 这种"已经算好的误差量"直接喂进去价值很大。
- critic 结构相同，另外多一路 height scan 的 attention 编码与 time embedding。

## A.5 算法与超参（MDPO）

**MDPO = 双 ActorCriticSRU + 互相蒸馏**：两个 actor 各自用自己的 storage 做 PPO，再加一项
`distill_coef = 0.02` 的互相 KL，防止两者过早分叉。评测时只跑 actor 1（`--policy_idx`）。

| 项 | 值 |
|---|---|
| `num_steps_per_env` / `num_learning_epochs` / `num_mini_batches` | 16 / 5 / 4 |
| `gamma` / `lam` | 0.999 / 0.95 |
| `clip_param` / `value_clip_param` / `max_grad_norm` | 0.2 / 0.2 / 1.0 |
| `value_loss_coef` / `distill_coef` | 0.02 / 0.02 |
| `reward_shifting_value` | 0.05（MDPO 训练时奖励整体平移，只为数值稳定，记录时会减回去） |
| 学习率 | 1e-3 → **1e-4 地板**，cosine（⛔ 不能用 `schedule="adaptive"`，MDPO 未实现，会静默冻结 LR） |
| 熵系数 | 1e-3 → 0，cosine，**hold 400 iter，再用 2600 iter 退火** |
| **σ 上墙日程** | `log_std_max`：0.1 → −0.6（600–2200）→ −1.0（2200–3600），cosine 分段 |
| `log_std_min` | −2.5（下界不动） |
| 观测归一化 | 无（`empirical_normalization=False`） |

> **为什么要"移动的墙"**：`log_std` 是被 `clamp` 住的，一旦参数贴在墙上，梯度就是 0，永远不会自己降下来
> （v1 run 里 `raw_log_std_mean` 从 iter 500 起恒等于墙值 0.1035）。只把墙往下挪，才能逼优化器继续压噪声。
> v1.1 把日程压进 4000 iter 内（原来 1000/5000/8000，跑完 4000 轮时 σ 还是 0.61，等于从没进过低噪声区）。

## A.6 奖励表（v1.1，共 16 项）

> 每条都乘 `dt = 0.2`。下表"每步"指**实际到账**的值。

| # | 项 | 权重 | 每步 / 每局 | 语义 |
|---|---|---:|---|---|
| 7 | `progress_along_path` | **1.2** | **1.2 × 新增米数**（≈0.19/步 @0.8 m/s） | **主收入**：沿规划路径的付费进度（米数制、棘轮、三档门） |
| 10 | `progress` | 0.15 | 0.15 × 向目标投影的米数（≈0.024/步） | 直连目标的弱塑形，防"路径断了就完全没方向" |
| 14 | **`heading_progress_to_path`** | **1.0** | 1.0 × 减少的朝向误差(rad)，单步钳 ±0.2 → **≤0.2/步** | ★新增：朝向路径切线的**势函数** |
| 12 | `goal_reached_bonus` | 25.0 | **+25 一次性** | 完成 1.5 s 连续驻留的那一刻 |
| 13 | **`velocity_at_goal`** | **−0.2** | −0.2×(‖v‖+0.5‖ω‖)×0.2 ≈ **−0.04×速度** | ★新增：圈内**只扣不给**，逼它减速 |
| 9 | `path_deviation_penalty` | −0.15 | −0.15 × 超出 0.5 m 走廊的米数（封顶 1.5 m）× 0.2 | 偏离走廊的惩罚；relax 段豁免 |
| 5 | `reach_goal_xy_soft` | **0.0** | 0 | ⛔ v1.1 清零 |
| 6 | `reach_goal_xy_tight` | **0.0** | 0 | ⛔ v1.1 清零 |
| 11 | `reach_goal` | **0.0** | 0 | ⛔ v1.1 清零 |
| 8 | `backward_movement_penalty` | −0.0 | 0 | 保持关闭（启用会封死死胡同脱困与窄处微调） |
| 3 | `action_rate_l1` | −0.02 | −0.02 × mean\|Δa\| × 0.2 | 动作平滑 |
| 1 | `lateral_movement` | −0.02 | −0.02 × \|v_y\| × 0.2 | 抑制横移 |
| 2 | `rot_movement` | −1e-05 | 极小 | 抑制原地打转（数值上几乎可忽略） |
| 0 | `joint_acc_l2_joint` | −1e-07 | 极小 | 关节二阶平滑 |
| 4 | `episode_termination` | −50.0 | **−10/回合** | 任何原因终止都扣 |
| 15 | `time_out_penalty` | −50.0 | **−10/超时回合** | 90 s 跑满 |

**一局的收支（20 m 路径，理想情况）**

| 行为 | 回报 |
|---|---|
| 沿路径走完 + 慢速驻留完成 | +24（路径）+ 25（完成）− 10（终止）≈ **+39** |
| 走完但不完成、绕到超时 | +24 − 10 − 10 ≈ **+4**，且在圈内每分钟都被 `velocity_at_goal` 扣 |
| 全程站在起点 | −10（超时）− 10（终止）≈ **−20** |
| 转身对准路径（180°） | +3.14（朝向势函数全额定金）|

## A.7 终止与成功口径（v1.1）

| 终止项 | 触发条件 | 记成功？ |
|---|---|---|
| `early_termination`（= `at_goal_navigation`） | 在 0.5 m 圈内**连续 8 步**（1.5 s），出圈即清零 | **是**（唯一成功） |
| `time_out` | `episode_length_buf ≥ 450` | **恒否**（v1.1 改） |
| `base_contact` | 机身/髋/大腿接触力 > 1.0 | 否 |
| `large_pitch_angle` | 俯仰 > 40° | 否 |
| `terrain_fall` | 掉到 −2 m 以下 | 否 |

- v1.0 的"超时那一步刚好在圈内就算成功"已删掉：连续驻留规则下，完成一定会先一步终止回合，
  所以**任何超时都是失败**。同时新增指标 `Metrics/robot_goal/in_goal_at_timeout`
  记录"超时那一刻恰好在圈内"的比例——那是旅馆行为的直接指纹（v1 里约 0.3）。
- 成功率的权威口径是 `eval_policy.py`（Play 协议，无观测噪声、无外力）：
  `completed_dwell = 结束早于 horizon 且最长驻留连击 ≥ required-1`（最后一步被 reset 吞掉）。

## A.8 路径管线当前参数

| 参数 | 值 | 位置 |
|---|---|---|
| 采样点数（稀疏/密集） | 15 / 64 | `GlobalPathConfig` |
| 类别权重 | optimal 0.45 / mildly 0.25 / highly 0.20 / infeasible 0.10 | `path_aware.DEFAULT_PATH_CATEGORY_WEIGHTS` |
| 机体半径判据 `seg_padding_m` | **0.31 m** | 实测整机半宽 0.26 + 0.05 |
| pinch 阈值 | 0.45 m | 小于此值视为窄门/窄道 |
| 航点噪声 | ≤ min(0.35 m, 0.5×clearance)，pinch 段不加噪 | |
| 三档支付门 | 正常 0.8 / relax 1.2 / 不可行 max(1.6, push+0.6) | `_compute_gates` |
| 挡路事件概率 | 0.15 → 0.30（2000 iter ramp） | |
| 进度记账 | `SYNC_RATE 1.0 m/步`、`CONT_TOL 0.5`、`PAY_CAP 1.5`、`EPS 0.04` | `path_generator.py` |
| 前视 | 局部路径 6 m；朝向 1.5 m | `LOCAL_PATH_HORIZON_M` / `HEADING_LOOKAHEAD_M` |
| **free mask** | `[GROUND_MIN, PLATFORM_MAX]` = 0–205 单位 | ⚠️ v1.1 修正（见 A.9.5） |

## A.9 新增机制详解（v1.1 相对 v1.0）

### A.9.1 关掉"目标旅馆"：三项接近奖励权重 → 0

**问题**：v1 里 `reach_goal`(2.0) + `tight`(0.75) + `soft`(0.25) 只在 0.5 m 圈内支付，
合计 **0.84/步**；而完成驻留只给 +25 并把回合**终止**。对一条能跑 450 步的回合来说，
"待在圈里刷 0.84/步"永远优于"+25 后结束"——实测每回合圈内收入 34.2，完成奖励只有 5.4，
200 局里有 53 局的最长连击正好停在 **7 步**（差一步，主动不完成）。

**为什么不加速度门**：速度门只是把"绕圈"变成"慢速绕圈"，慢速仍在圈内、仍拿 0.84/步，
要避免完成还得每 7 步出圈一次，摊薄后仍有 ~0.58/步——旅馆容量仍是完成奖励的 8 倍。
**唯一可靠的做法是不给**：`weight=0` 时 `RewardManager` 直接 `continue`，连计算都跳过（零开销）。

**替代的引导**：
- 想"进圈"的塑形由 `progress_along_path` 提供——端点锁定保证路径最后一点就是目标，进度一路付到脚下；
- 想"减速"由 `velocity_at_goal`(−0.2) 提供：圈内**只扣不给**，逼近 0 速度才不挨罚；
- 想"完成"由 +25 提供。

### A.9.2 朝向势函数 `heading_progress_to_path`（新增项，权重 1.0）

```python
p     = get_path_local(1.5)[:, -1, :2]        # 1.5 m 前视点（末点；第 0 点在机器人身上，方向退化）
err   = atan2(p_y, p_x).abs()                  # [0, π]，机体系
delta = clamp(prev_err - err, -0.2, +0.2)      # 单步最多 0.2 rad
r     = fade(dist_to_goal) * delta / dt        # 再 ×dt 后 = fade × delta
```

四条性质（都是刻意的）：

1. **是势函数，不是状态占用**：整局总额 = `err_start − err_end`，有界（≤π）；来回摆头每步 ±0.2 互相抵消，
   净收益 0。这直接对应 v1 的教训——**任何"在某区域每步给钱"的正项都会被刷**。
2. **量级对上**：ω 上限 1.0 rad/s → 每步 0.2 rad → **0.2/步**，与沿路径行驶的 0.19/步同量级。
   180° 转身共带来 **+3.14**，且立刻到账（不是等转完才给）。
3. **单步钳位 = 拐角不尖峰**：90° 急转弯时"胡萝卜"会突跳，不钳位的话势函数会一次性支付整个跳跃；
   钳到 0.2 rad 后，单步上限就是 0.2，与正常转弯无异（单测 U6b 专门断言这一点）。
4. **目标 1 m 内淡出**：终端行为交给驻留判定与 `velocity_at_goal`，朝向项在那里没有意义。

**为什么必须用 1.5 m 前视**：路径首点就在机器人身上，方向向量退化为 0；用瞬时切向又会在噪声里抖。
1.5 m ≈ 0.8 m/s 下 1.8 s 的提前量，既给足拐弯预判，门洞里又仍落在门内。

### A.9.3 朝向误差观测 `path_heading_error`（+2 维）

`[cos Δψ, sin Δψ]`，与 A.9.2 用的是同一个前视点。理由是**路径块不过 RNN**：策略原本只能从
`target_path_local`（15 点 × 4 维）里自己解算方位，现在直接把误差量喂进去（Go2W 的
`target_heading_delayed` 用的是同一套 cos/sin 范式）。

**维度影响**：policy 2697 → **2699**，critic 5834 → **5836**；proprio 17 → 19（actor 与 critic 必须相等，
网络里有断言）。导出器（JIT/ONNX × Single/Dual）的维度全部由 `num_actor_obs` 推导，无需硬编码修改，
但导出后必须跑一遍数值一致性单测。

### A.9.4 超时成功口径归零

v1 的 `strict_success=True` 仍然奖励"超时那一步刚好在 0.5 m 内"——绕圈的机器人约 1/3 的时间在圈里，
于是超时也有 ~1/3 概率白拿成功。v1.1 直接恒记 0，并把这件事降级成一个**指标**
（`in_goal_at_timeout`），因为它本身是有价值的旅馆指纹。

### A.9.5 free mask 修正（楼梯）

**根因**：`Prm.compute_free_mask` 原来只把 `ground(0–10) ∪ platform(195–205)` 当自由格。
楼梯踏面在 **40 / 80 / 120 / 160** 单位，**整段楼梯既不是地面也不是平台**，于是被当成障碍：
clearance 由 free mask 算出来 → 楼梯上 clearance = 0 → 任何跨楼梯的弦都判"不可通行"，
支付门随之放宽到 1.6 m——**恰好在最需要严格贴线的地方取消了约束**。

**修法**：`free = (h ≥ GROUND_MIN) & (h ≤ PLATFORM_MAX)`。墙在 300、坑在 <0，仍然被排除。

**效果**：stairs tile 每局 invalid 段 **8.14 → 0.95**；平台/踏面 free 占比 **1.00**；
invalid 段的最小 clearance 从 0.00 m 回到 0.31 m。

### A.9.6 `relax` 语义修正

```
v1  : relax = (inv | left | right) & ~pinch     # 不可行段若也在 pinch 内 → 不放宽
v1.1: relax = inv | ((left | right) & ~pinch)   # 不可行段本身永远放宽
```

不可行段是**构造上就走不了**（注入的挡路事件、非 LOS 类别、或机体确实过不去的弦），
在那里离开路径是正确行为，不该被走廊惩罚。pinch 只保护"不可行段的邻居"，
避免豁免溢到旁边真正要走严格的门框上。

## A.10 与 v1.0 的差异一览

| 项 | v1.0 | **v1.1** |
|---|---|---|
| `reach_goal` / `tight` / `soft` | 2.0 / 0.75 / 0.25 | **0 / 0 / 0** |
| `velocity_at_goal` | 未注册 | **−0.2** |
| `heading_progress_to_path` | 无 | **1.0（势函数，1.5 m 前视）** |
| policy / critic 维度 | 2697 / 5834 | **2699 / 5836** |
| 超时成功 | 当步在圈内算成功 | **恒 0**，另记 `in_goal_at_timeout` |
| free mask | ground ∪ platform | **`[GROUND_MIN, PLATFORM_MAX]`** |
| `relax` | 不可行段在 pinch 处不免罚 | **不可行段永远免罚** |
| σ 墙日程 | 1000 / 5000 / 8000 | **600 / 2200 / 3600** |
| 熵日程 | hold 500 / span 3500 | **hold 400 / span 2600** |

## A.11 监控面板与闸门

**训练期**（TensorBoard）

| 指标 | 期望 | 异常含义 |
|---|---|---|
| `Episode_Reward/reach_goal*` | **恒 0** | 非 0 → 权重没生效 |
| `Episode_Reward/heading_progress_to_path` | 前 200 轮应为正且逐步收敛到 0 | 长期为负 → 策略在持续背对路径 |
| `Episode_Reward/goal_reached_bonus × 90 / 25` | 上升 | 不升 → 到不了/停不住 |
| `Metrics/robot_goal/in_goal_at_timeout` | 下降 | 不降 → 仍在目标附近空转 |
| `Metrics/robot_goal/success_rate` | 上升 | — |
| `Episode_Termination/time_out` | < 0.25 | > 0.3 → 回合效率差 |
| `Policy/log_std_max` | 600/2200/3600 处为 0.1/−0.6/−1.0 | 不动 → 日程没接上 |
| `Metrics/robot_goal/invalid_seg_frac` | 楼梯 ≈0.07、迷宫/坑洞 ≈0.13 | 楼梯 tile 不再异常偏高（修复前 0.58） |

**1000 轮早期闸门**：条件完成率（完成驻留 / 曾入圈）> 0.4、圈内速度 p50 明显下降、
120–180° 大角度开局 SR 不再趋近 0。

**最终验收**（`eval_policy.py` × 200 局 × 60 s，同口径）：SR ≥ 0.58、条件完成率 ≥ 0.80、
圈内速度 p50 ≤ 0.5 m/s、median `d_path` ≤ 0.5 m、timeout ≤ 0.25、目标附近正收入 = 0。

## A.12 已知限制

1. **楼梯攀爬是学出来的动作序列**：开环恒定前向指令 8 s 最多升 0.18 m（一级），但旧策略 stairs SR 0.403
   说明系统能爬——需要减速/对准/逐级推进，当前靠 `heading_progress_to_path` 只补上"对准"这一段。
   若 P1 之后 stairs SR 仍 ≈0，下一步是楼梯地形课程，⛔ 不是"楼梯专用奖励"。
2. **`backward_movement_penalty` 保持关闭**：v1 里有 90% 的回合出现倒走（`reverse_frac` > 5%），
   但倒走在当前奖励下是合法的替代策略；启用惩罚会顺带封死倒车脱困。
3. **`invalid_seg_frac` 的口径**：它把"注入的挡路事件"和"非 LOS 类别"（权重合计 0.30）都算进去，
   所以稳态就是 0.07–0.13，不能再按 v1.0 写的 5–10% 卡。修复后实测：stairs 0.95/14 ≈ 0.07、
   maze 1.78/14 ≈ 0.13、pits 1.86/14 ≈ 0.13。
4. **地形课程未接线**，`max_init_terrain_level=10`（全部 6 行难度的 tile 一起采样），没有难度递进。
