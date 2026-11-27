---
name: sru-training-analysis
description: 分析 SRU/B2W/Go2W 的 IsaacLab/RSL-RL 训练日志，用当前 path-aware 奖励表的算术把 TensorBoard 标量换算成"每回合收入/每次超时实付/势函数上界"等可判读的量，并跑一组针对已知失效模式（目标旅馆、势函数被采空、超时罚被 dt 打折、终端动作丢失）的检查。当用户要求分析训练曲线、对比两次 run、排查 reward 崩塌/成功率下滑、判断本地改动是否生效时使用；不用于机采数据或真机部署日志。
---

# SRU 训练日志分析

## 目标

把"训练行不行"判断成可复核的证据链：**同一轮次比较 → 收入分解（换算到物理单位）→ 不变量检查 → 行为指标 → 代码/配置对照 → 原因排序**。

这个仓库的训练诊断有一个特殊之处：**奖励表里的每一项都被 `RewardManager` 乘过 `dt`，`Episode_Reward/*` 又被除以 `max_episode_length_s`**。所以"weight 写了多少"和"实际付了多少"经常差 5 倍以上，直接看 `Episode_Reward` 的原始数字必然误判。`scripts/analyze_rsl_rl_events.py` 会读 run 自己的 `params/env.yaml` 把这一切换算回来。

---

## 核心工作流

### 1. 定位 run

- 路径：`IsaacLab/logs/rsl_rl/<experiment>/<timestamp>_<run_name>/`
- 需要的文件：`events.out.tfevents.*`、`params/env.yaml`（奖励权重、horizon、sim.dt、decimation）、`params/agent.yaml`（max_iterations）、`git/*.diff`
- 对齐原则：**只比同一轮次**。不同 run 的轮次分布不同时，先按同样的窗口切。

### 2. 一条命令出表 + 检查

```bash
python3 /home/yichangfeng/sru-path-aware/skills/sru-training-analysis/scripts/analyze_rsl_rl_events.py \
  <runA> <runB> --windows 0-250,250-500,500-1000
```

- 不传 `--windows` 时自动按 250 轮分桶；多个 run 会在末尾打一张 cross-run summary（每个 run 取最后一个有效桶 + FAIL/WARN 计数）。
- `--tags-only` 保留旧的原始 tag 输出；`--steps 100,500 --tags A,B` 查指定轮次的原始值；`--list-tags` 列出全部标量；`--json out.json` 导出。
- 只读：脚本不启动仿真、不改任何东西。

### 3. 读表（每列都是"每回合"或"每次事件"）

| 列 | 含义 | 健康值 |
|---|---|---|
| `SR` | `Metrics/robot_goal/success_rate`（10 回合滚动，reset 时刻快照） | 与 `comp` 同量级；差得远说明口径不一致 |
| `comp` | 完成/局 = `goal_reached_bonus × T / weight` | 越高越好；**明显低于历史最好桶**就是终端动作在退化 |
| `ep_len` | 回合步数 | 稳定；翻倍 = 有东西在拖时间 |
| `timeout` | 跑满 horizon 的比例（不含 `early_termination`） | < 0.25 |
| `fall` | `base_contact` | < 0.45 |
| `devX` | **超出偏差走廊**的横向偏移（由惩罚反推，下界） | 走廊 1.5 m 时 < 0.3 m |
| `paid_m` | 每回合被付费的路径米数 = `progress_along_path × T / weight` | 接近计划路径长度 |
| `head` | 朝向势函数每回合实付 | **必须 ≤ π (3.14)** |
| `near` | 目标半径内的正收入 | **必须 = 0** |
| `to$` | **一次 horizon 超时实际付了多少** = `time_out_penalty × T / 超时率` | ≈ −50（不是 weight 的字面值） |
| `EV` | explained variance | > 0.5 |

### 4. 四条不变量（脚本会逐条 PASS/WARN/FAIL）

1. **目标旅馆**：`near` 必须恒 0。任何"待在目标半径内就每步付钱"的项，配上"完成即终止"，都会让"绕圈不完成"成为最优。
2. **势函数上界**：`head`（以及任何 potential 型奖励）每回合不得超过它的定义域上界（朝向 = π）。超了就是钳位/参照量写错，能量被采空。
3. **dt 折扣**：一次性惩罚（超时、终止）的实付 = `weight × dt`。`to$` 明显大于 −35 就是这个问题，不是"惩罚不够狠"。
4. **终端动作**：`comp` 相对历史最好桶掉一半以上 = 完成动作丢失。先看是不是噪声墙（`Policy/log_std_max` 何时开始下移）还没走到，再看奖励里有没有稠密终端信号。

### 5. 与代码/配置对照

- 先确认 `params/env.yaml` 里实际生效的权重（脚本已读，`config:` 那行会打印路径）。若显示 `defaults`，说明该 run 没有 `params/`，`comp/paid_m/devX/to$` 四列会是 `-`，不要当 0 读。
- 代码侧重点看：`config/b2w/navigation_env_cfg.py`（奖励表）、`config/path_aware.py`（观测插入、horizon）、`mdp/navigation/goal_commands.py`（驻留、指标定义）。
- `Metrics/*` 是 **reset 时刻的快照**，不是全程均值：`distance_to_path` 高不代表平均跟踪差，要看按终止类型分层。
- MDPO 的 `Loss/kl_divergence` 是双 actor 蒸馏 KL，不是 PPO adaptive KL。

### 6. 输出结论

先给同轮次对照表（附 `comp/head/near/to$` 四列），再给机制，最后给原因排序。要明确区分"这条是数据直接支撑的"与"这条是推断、需要什么额外证据"。

---

## 本项目已经踩过的坑（对照用）

| 指纹 | 判据 | 根因 |
|---|---|---|
| `near`≈+34/局、SR 0.28 | 目标半径内每步付 0.84 | 旅馆（v1） |
| `head`≈15/局（>π）、SR 0.62→0.10 | 势函数钳位 + `prev := err` | 势被采空（v1.1） |
| `to$`≈−10 | `weight −50 × dt 0.2` | dt 折扣（v1、v1.1） |
| `comp` 从 0.6 掉到 0.28、`ep_len` 88→205 | 接近奖励清零后终端无稠密信号 + σ≈1.0 | 终端动作丢失（v1p2） |
| `invalid`≈0.22 且只在 stairs tile | 楼梯踏面 40–160 不在 free mask | PRM free mask（已修，修后 0.09） |
| `devX` 大但 `paid_m` 正常 | 走廊太窄（0.5 m），不是跟踪问题 | 已放宽到 1.5 m |

---

## 操作边界

- 默认只读：不改训练代码、配置或正在运行的训练任务。
- 若用户要求修改，先说明 why / how / risk & rollback，并跑单测。
- 不把仿真训练日志分析扩展成真机部署任务。
- 输出中引用文件时用绝对路径。
