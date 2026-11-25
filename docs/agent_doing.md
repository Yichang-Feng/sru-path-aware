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

**产出**：`docs/phase0_baseline_report.md`

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