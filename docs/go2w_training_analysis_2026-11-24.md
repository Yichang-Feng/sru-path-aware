# Go2W 训练结果分析（2026-11-24）

- **数据来源**：用户从另一台机器（`CNBESLPD260001`）拷回的 TensorBoard event，
  现位于 `/home/yichangfeng/tb_compare/go2w/`
- **run**：`paper_go2w_mdpo_1046`，实验名 `go2w_navigation_mdpo_phase4`
- **配置**：MDPO / 1046 env / seed 42 / `num_steps_per_env=24` / 24.3 小时 / **4909 / 15000 iter**
- **对照**：B2W 同配方 run
  `b2w_navigation_mdpo/2026-11-23_18-45-05_b2w_recalib_1046`（σ 失控版）
  与 `2026-11-24_09-05-54_b2w_sigma_bounded`（σ 加界版，正在跑）

---

## 1. 摘要

| | 峰值（iter ≈ 300） | 末尾（iter 4000–4909） | 变化 |
|---|---:|---:|---:|
| `Train/mean_reward` | **40.6**（100 轮均值，单点最高 47.1 @ 309） | **16.9** | **−23.7** |
| `success_rate` | **0.596**（100 轮均值，单点最高 0.648 @ 352） | 0.533 | −0.063 |
| `Policy/mean_noise_std` | 1.41 | **11.7**（单点最高 12.98 @ 2383） | **+10.3** |
| `base_contact` | 0.237 | 0.361 | +0.124 |
| `time_out` | 0.173 | 0.068 | −0.105 |
| `velocity_magnitude` | 0.325 | 0.432 | +0.107 |
| `action_rate_l1`（×60） | −2.8 | **−16.6** | **−13.8** |
| `kl_divergence` | 0.183 | 0.403 | +0.22（`desired_kl` 的 40 倍） |
| `mean_episode_length` | 134 | 102 | −32 |

**结论：策略在 iter ≈ 300 学到了，之后 4600 轮一直在退化。** 退化的主因不是任务本身，
而是**探索噪声 σ 从 0.84 一路涨到 12.98**（无上界），把 `tanh` 动作打成 bang-bang，
`action_rate_l1` 从 −2.8 涨到 −16.6 每 episode —— **这一项就占 reward 下降的 60%**。

这与 B2W run（σ 0.71 → 8.49，reward 33 → 22）是**同一个失效模式**，
只是 Go2W 更早、更陡（iter 400 就开始）。

---

## 2. 这次 run 实际跑的是什么代码

这一节决定了"该改哪里"，所以先做取证。三条互相独立的证据指向同一个结论：
**`agent.yaml` 里写的熵 / clip / 学习率退火，在这次 run 里全部没有生效。**

### 2.1 证据一：学习率曲线精确匹配"不退到 min"的官方 cosine

配置里写的是 `schedule="cosine"`、`learning_rate=1e-3`、`min_learning_rate=1e-4`。
实测 `Loss/learning_rate` 在 iter ≈ 4450 处为 **7.973e-4**。

- 官方 `MDPO._update_learning_rate`：`0.5 · lr_max · (1 + cos(π·it/max_iters))`
  → `0.5 · 1e-3 · (1 + cos(π · 4450/15000))` = **7.97e-4** ✅
- 配置期望的版本（退到 `min_learning_rate`）：`1e-4 + 0.9e-3 · 0.797` = 8.17e-4 ❌

也就是说运行时用的是**没有 `min_learning_rate` 语义的官方实现**，
会在训练结束时把 LR 退到 0。

### 2.2 证据二：event 里没有 `Loss/entropy_coef` / `Loss/clip_param`

这两个 tag 在本次 run 的 33 个 tag 里**不存在**。说明记录日志的 runner 没有这两行，
而这两行恰好是"按计划退火"的可见性入口。

### 2.3 证据三：配置文件的字段根本不存在

`installed` 的 `RslRlPpoAlgorithmCfg`（`navigation/config/rl_cfg.py`）在本次改动前只有
14 个字段，**没有** `min_learning_rate` / `min_entropy_coef` / `entropy_hold_iters` /
`entropy_decay_span` / `entropy_schedule` / `min_clip_param` / `clip_schedule`。

`Go2WNavMDPORunnerPhase4Cfg.__post_init__` 里那 7 行 `self.algorithm.xxx = ...` 因此只是给
dataclass 实例**挂了个普通属性**：

1. IsaacLab 的 `class_to_dict` 是按 `__dict__` 序列化的，所以它们**出现在 `agent.yaml` 里**；
2. runner 用 `MDPO(..., **alg_cfg)` 构造算法，而 `MDPO.__init__` 有 `**kwargs`，
   于是这些字段被**静默吞掉**。

**结果：配置读起来是"熵 0.003 → 200 轮后余弦退火到 0.001"，实际是熵恒为 0.003。**

### 2.4 σ > 1.22 直接证明没有上界

`Policy/mean_noise_std` 是 `actor_critic_1.action_std.mean()`，即
`distribution.stddev`。如果 `ActorCriticSRU.update_distribution` 里有
`clamp(log_std, -2.5, 0.2)`，这个值**不可能超过 1.22**。实测 12.98 →
用的 `rsl_rl` 是没有 clamp 的旧版（与官方 submodule `e71df2b` 一致）。

---

## 3. 曲线

### 3.1 分窗口

| window | reward | SR | σ | base_contact | time_out | ep_len | action_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–50 | 16.7 | 0.110 | 0.92 | 0.392 | 0.316 | 144 | −0.031 |
| 50–100 | 30.7 | 0.309 | 0.86 | 0.283 | 0.268 | 126 | −0.027 |
| 100–200 | 34.9 | 0.440 | 0.91 | 0.276 | 0.182 | 118 | −0.027 |
| **200–400** | **39.5** | **0.562** | 1.41 | 0.237 | 0.173 | 134 | −0.047 |
| 400–600 | 36.0 | **0.587** | 2.82 | 0.268 | 0.134 | 129 | −0.093 |
| 600–1000 | 26.3 | 0.567 | **6.60** | 0.319 | 0.087 | 112 | −0.178 |
| 1000–1500 | 20.7 | 0.536 | 9.62 | 0.368 | 0.059 | 98 | −0.228 |
| 1500–2000 | 19.1 | 0.523 | 10.52 | 0.378 | 0.056 | 97 | −0.245 |
| 2000–3000 | 16.1 | 0.526 | **12.05** | 0.376 | 0.053 | 97 | −0.273 |
| 3000–4000 | 17.7 | 0.538 | 11.14 | 0.357 | 0.069 | 103 | −0.274 |
| 4000–4909 | 16.9 | 0.533 | 11.66 | 0.361 | 0.068 | 102 | −0.276 |

三个阶段：

1. **0–200：正常爬升。** SR 0.11 → 0.44，reward 17 → 35，`time_out` 0.32 → 0.18。
2. **200–600：峰值区。** SR 最高 0.587（单点 0.648 @ 352），reward 39.5。此时 σ 已从 0.9 涨到 2.8。
3. **600 之后：σ 接管。** SR 掉回 0.52 并锁死，reward 腰斩，`action_rate_l1` 一路涨到 −0.276。

### 3.2 Episode-return 分解（×60，`episode_length_s = 60`）

| term | 峰值 200–400 | 末尾 4000–4909 | 变化 |
|---|---:|---:|---:|
| `progress`（gated, power 2） | +14.68 | +14.14 | −0.54 |
| `reach_goal`（10, 0.5 m） | +22.61 | +18.71 | −3.90 |
| `heading` | +10.36 | +7.30 | −3.06 |
| `progress_along_path` | +0.73 | +0.70 | −0.03 |
| `heading_progress` | +0.22 | +0.09 | −0.13 |
| **`action_rate_l1`** | **−2.83** | **−16.58** | **−13.75** |
| `episode_termination`（−50） | −2.68 | −3.90 | −1.22 |
| `velocity_at_goal` | −1.16 | −1.12 | +0.04 |
| `backward_movement_penalty` | −1.05 | −1.09 | −0.04 |
| `joint_acc_l2` | −0.87 | −1.09 | −0.22 |
| `lateral_movement` | −0.06 | −0.15 | −0.09 |
| `time_cost` | −0.27 | −0.20 | +0.07 |
| **合计** | **≈ +39.7** | **≈ +16.8** | **−22.9** |

归因（按贡献）：**`action_rate_l1` 60%**、`reach_goal` 17%、`heading` 13%。
`progress` 几乎没变 —— **机器人一直知道该往哪走，是噪声把它晃出了正确动作。**

### 3.3 失败模式

末尾窗口：`early_termination` 0.571（= SR 0.533 量级） / `base_contact` 0.361 /
`time_out` 0.068 / `large_pitch_angle` 0.020。合计 ≈ 1.02。

即：**57% 成功、36% 撞、7% 超时**。"站着不动"的问题在 Go2W 上不存在
（`time_out` 已压到 0.07，`velocity_magnitude` 0.43 m/s），这一点比 B2W 健康。

---

## 4. 与 B2W 的对照

| | B2W 旧 run（18:45，无界 σ） | B2W 新 run（09:05，加界 σ） | Go2W（本次） |
|---|---:|---:|---:|
| 可比 iter | 300–600 | 300–468 | 200–600 |
| `success_rate` | 0.366 | **0.391**（仍上升） | 0.587（峰值后回落） |
| `Train/mean_reward` | 31.8 | **40.1**（仍上升） | 39.5（峰值后回落） |
| `mean_noise_std` 轨迹 | 1.07 → 8.49 | 0.94 → **0.54** | 0.92 → **12.98** |
| `action_rate_l1` 轨迹 | −0.041 → −0.212 | **−0.020 保持** | −0.031 → −0.276 |
| `base_contact` | 0.44 → 0.50 | 0.65 → **0.39** | 0.24 → 0.38 |
| `kl_divergence` | 0.34 | 0.26 | 0.40 |

**B2W 加界后的 run 已经验证了因果链**：σ 不再漂移 → `action_rate_l1` 稳定在 −0.02 →
reward / SR 单调上升到 40.1 / 0.391（旧 run 在同样轮次是 31.8 / 0.366），
`base_contact` 也从 0.65 降到 0.39。

Go2W 的三条曲线形状与 B2W 旧 run 完全同构，只是时间轴压缩了 3–4 倍。
**因此 B2W 的修复可以直接搬到 Go2W，而且更有必要（Go2W 的 σ 涨得更凶）。**

---

## 5. 本次对 Go2W 做的调整

### 5.1 σ 上界（关键）

`Go2WNavBaseRunnerCfg.policy` 现在显式写入 `log_std_min=-2.5, log_std_max=0.2`
（std ∈ [0.082, 1.22]，与 B2W 一致）。实际生效靠的是
`ActorCriticSRU.update_distribution` 里的 clamp（已在前一提交 `72e3818` 加入官方
`rsl_rl`），Go2W 与 B2W 共用同一个类。

### 5.2 让"退火配置"真正生效

`navigation/config/rl_cfg.py` 的 `RslRlPpoAlgorithmCfg` 补齐了 7 个字段：

```text
min_clip_param, clip_schedule            # 0.20 -> 0.15
min_entropy_coef, entropy_hold_iters,
entropy_decay_span, entropy_schedule     # 0.003 -> 0.001
min_learning_rate                        # 1e-3 -> 1e-4
```

默认全部为 `"fixed"` / 0，**不改变任何没有显式要求退火的配置**（B2W、PPO 全部不受影响）。

官方 `rsl_rl/algorithms/mdpo.py` 里实现了 `_update_entropy_coef` / `_update_clip_param`，
并把 cosine 学习率的终点从 0 改成 `min_learning_rate`。
runner 新增 `Loss/entropy_coef`、`Loss/clip_param` 两条曲线，方便下次直接验证。

退火数值（已单测）：

```text
entropy 0.003 -> 0.001   it=0: 0.003000  it=200: 0.003000  it=850: 0.002000  it=1500: 0.001000
clip    0.20  -> 0.15    it=0: 0.2000    it=4500: 0.1897   it=15000: 0.1500
lr      1e-3  -> 1e-4    it=0: 1.0e-3    it=4500: 8.145e-4 it=15000: 1.0e-4
```

### 5.3 未改动

- 奖励集：Go2W 的奖励结构（gated `progress` + `heading` + `reach_goal` + `episode_termination -50`）
  与 B2W 是两套不同配方，Go2W 这套在 300 轮内就打到 SR 0.60，**不需要动**。
- 地形 / PRM / A\* 参数：不动，保持单变量。
- `ppo.py` / `spo.py`：不涉及，未改。

---

## 6. 建议的重跑方式

```bash
cd /home/yichangfeng/sru-path-aware/IsaacLab
./isaaclab.sh -p /home/yichangfeng/sru-path-aware/sru-navigation-sim/scripts/train.py \
    --task Isaac-Nav-MDPO-Go2W-v0 --num_envs 1046 --max_iterations 15000 \
    --seed 42 --run_name go2w_sigma_bounded_1046 --headless
```

> 注意：`train.py` **没有 resume**，`Go2WNavMDPORunnerPhase4Cfg` 是"从 3000 轮接着跑"的
> 课程相位，但单跑时它就是从 0 开始。上面这条命令等价于本次 run 的复现，只多了 σ 上界与退火。

### 起跑后 50–100 iter 看这几个量

| 指标 | 期望 | 不对时怎么办 |
|---|---|---|
| `Policy/mean_noise_std` | ≤ 1.22，且**跟随熵系数下行** | 仍 > 1.22 → clamp 没生效，查 `agent.yaml` 的 `log_std_max` |
| `Loss/entropy_coef` | 200 iter 后开始下降，1500 iter 到 0.001 | 恒 0.003 → 退火没接上 |
| `Loss/learning_rate` | 收尾停在 1e-4 而不是 0 | — |
| `Episode_Reward/action_rate_l1` | 稳定在 −0.03 ~ −0.08 | 涨到 −0.2 以上 → σ 仍失控 |
| `Episode_Termination/base_contact` | 从 0.39 往下 | 不降 → 撞障另有原因，考虑加权 `episode_termination` |

### 判读口径

不要只看 `Train/mean_reward`。本次 run 的教训是：**reward 高的那一版（iter 300）
和末尾那一版（iter 4900）在 SR 上只差 0.06，但 reward 差了 23.7，全是噪声惩罚。**
应以 `success_rate` + `base_contact` + `action_rate_l1` 三者一起看。

---

## 7. 尚未解决的问题

1. **`base_contact` 0.36 的天花板。** σ 收敛后预计会降（B2W 已从 0.65 → 0.39），
   但如果仍在 0.3 以上，需要单独的碰撞 shaping；当前 `episode_termination=-50` 折算到每
   episode 只有 −3.9，而成功一次给 +18.7，**策略撞一次的边际成本偏低**。
2. **critic 跟不上。** `Loss/value_function` 从 5.6 涨到 28.8，B2W 新 run 更是到 72。
   `value_loss_coef=0.02` 偏小，后续可试 0.05。
3. **单 seed。** 消融要再补 seed 43/44 才能排除随机性。

---

## 8. 复现信息

```bash
# 曲线解析
python3 skills/sru-training-analysis/scripts/analyze_rsl_rl_events.py \
    /home/yichangfeng/tb_compare/go2w \
    --tags "Train/mean_reward,Metrics/robot_goal/success_rate,Policy/mean_noise_std" \
    --windows 0-200,200-400,400-1000,1000-2000,2000-3000,3000-4000,4000-4909
```

- 原始 event：`/home/yichangfeng/tb_compare/go2w/events.out.tfevents.1795401492.CNBESLPD260001.1735784.0`
- 配置快照：同目录 `params/env.yaml`、`params/agent.yaml`
- 该 run 的 `git/sru-path-aware.diff` **为空**（只有未跟踪的 `env_sru_path/`），
  因此本次改动的代码版本无法从 run 目录反查，只能靠 §2 的三条曲线证据推断。
