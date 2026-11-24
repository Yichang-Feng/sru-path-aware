---
name: sru-training-analysis
description: 分析 SRU/B2W/Go2W 的 IsaacLab/RSL-RL 训练日志，解析 TensorBoard event、按轮次/窗口对比 reward 与稳定性指标，并结合官方代码与本地 diff 判断训练失败或偏离的原因。当用户要求分析训练曲线、对比两次训练 run、排查 reward 崩塌/不收敛、判断本地修改是否偏离官方实现时使用；不用于机采数据或真机部署日志。
---

# SRU 训练日志分析

## 目标

把“训练行不行”判断成可复核的证据链：**同一轮次比较 → reward 分解 → 稳定性/行为指标 → 配置与代码 diff → 原因排序**。不要只用最终 reward 或轮次少来解释失败。

## 核心工作流

1. **定位 run**
   - 常见路径：`IsaacLab/logs/rsl_rl/<experiment>/<timestamp>_<run_name>/`
   - 每个 run 读取：`events.out.tfevents.*`、`params/agent.yaml`、`params/env.yaml`、`git/*.diff`。
   - 若同时有官方 run 和本地修改 run，先确认两者 `agent.yaml` 是否一致，再用 `git/*.diff` 确认 sim/learning 侧改动。

2. **解析 event**
   - 使用 `scripts/analyze_rsl_rl_events.py`，无需 tensorboard 依赖。
   - 至少报告：
     - `Train/mean_reward`
     - `Metrics/robot_goal/success_rate`
     - `Metrics/robot_goal/velocity_toward_goal`
     - `Episode_Reward/*`（尤其 `action_rate_l1`、`joint_acc*`、`progress_along_path`、`reach_goal_xy_*`）
     - `Episode_Termination/base_contact`、`time_out`
     - `Policy/mean_noise_std`
     - `Loss/learning_rate`、`Loss/kl_divergence`

   示例：

   ```bash
   python3 /home/yichangfeng/sru-path-aware/skills/sru-training-analysis/scripts/analyze_rsl_rl_events.py \
     /abs/path/to/official_run \
     /abs/path/to/local_run \
     --steps 25,50,100,200,300,500,700 \
     --windows 1-50,100-200,250-350,400-500,600-700
   ```

3. **判断是否失败**
   - “轮次少”只能解释最终性能不足，不能解释 reward 先升后降。
   - 若同时出现 `noise_std↑`、`KL↑`、`action penalty↑`、`base_contact↑`、`velocity_toward_goal↓`，判为优化失稳/early collapse。
   - 若 reward 变高但 `success_rate`、`base_contact`、`velocity_toward_goal` 基本不变，说明提升主要来自 reward shaping，不是行为改善。

4. **对照代码与论文**
   - 官方 release 是复现论文结果的 ground truth；论文文字是高层意图，常省略 `α/β/ε/λ`、reward scale、采样比例等细节。
   - 本地“按论文文字修正”若改变 reward 触发频率或惩罚函数，应视为第三个实现，而不是官方复现。
   - MDPO 中 `Loss/kl_divergence` 是双 actor 的 distillation KL；`desired_kl` 在 `schedule=exponential` 下不调 LR。不要把它解释成 PPO adaptive 的 policy KL。

5. **输出结论**
   - 先给“同轮次差异表”，再给机制解释，最后给原因排序。
   - 明确区分：
     - 复现论文结果：优先信官方代码；
     - 解释论文意图：论文更可信但不完整；
     - 本地修改版：只有在单变量消融后才能判断是否更正确。

## 操作边界

- 默认只读分析；不要修改训练代码、配置或正在运行的训练任务。
- 若用户要求修改，先按仓库 AGENTS.md 说明 why/how/risk & rollback。
- 不把仿真训练日志分析扩展成真机部署任务；真机相关改动的同步/编译/验证另走 `sru-deploy` skill。
- 输出中引用文件时用绝对路径。
