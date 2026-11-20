# sru-path-aware

可部署的 path-aware RL 训练工作区快照。仓库：`git@github.com:Yichang-Feng/sru-path-aware.git`。

在其他电脑快速部署（conda / Miniforge）：

```bash
git clone git@github.com:Yichang-Feng/sru-path-aware.git
cd sru-path-aware
bash setup.sh
```

前置条件：Miniforge/conda、已创建 `sru-pathaware` 环境，且其中装有 Isaac Sim 4.5.0、IsaacLab 2.3.2、PyTorch 2.7.0+cu126、Python 3.10。`setup.sh` 只做 symlink 校验和 editable 安装，不负责安装 Isaac Sim/PyTorch。训练需至少 32 个环境并保证 GPU 有足够空闲显存。

本仓库是单仓库快照：三个上游子项目的 `.git` 已剥离，所有本地兼容性补丁已直接包含在源码中；不依赖原 `sru-learning` 工作区的代码或数据。

## 目录来源

| 目录 | 来源 | Commit |
|---|---|---|
| `sru-navigation-learning` | `leggedrobotics/sru-navigation-learning` 分支 `feature/path-aware-rl` | `e71df2b` |
| `sru-navigation-sim` | `leggedrobotics/sru-navigation-sim` 分支 `feature/path-aware-rl` | `da6d5d6` |
| `IsaacLab` | 本地 IsaacLab 仓库（isaac-sim/IsaacLab） | `b0542fe2d` |

`IsaacLab/source/isaaclab_nav_task` 是指向 `sru-navigation-sim` 的 symlink。

上游 `sru-navigation-learning` 按 README 只包含 `rsl_rl` 训练框架；IsaacLab 任务与 `scripts/train.py`、`scripts/play.py` 在 `sru-navigation-sim`。因此这里的目录结构与上游 README 是一致的，不再混入原 sru-learning 的日志、模型和脚本。

## Conda 环境与一键部署

使用 `sru-pathaware` 环境；本仓库的 `setup.sh` 会把 `rsl_rl`、所有 `isaaclab*` 包和 `isaaclab_nav_task` 的 editable 安装指向当前克隆目录。原 `sru-learning` 环境不需要改动。

若 `setup.sh` 未执行或路径发生变化，可手动重绑：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sru-pathaware
P="$(pwd)"
python -m pip install --no-deps -e "$P/sru-navigation-learning"
python -m pip install --no-deps -e "$P/sru-navigation-sim"
for m in isaaclab isaaclab_assets isaaclab_contrib isaaclab_mimic isaaclab_tasks isaaclab_rl; do
  python -m pip install --no-deps -e "$P/IsaacLab/source/$m"
done
```

## 训练与回放

B2W：

```bash
./train_b2w_pathaware.sh --num_envs 32 --max_iterations 10 --headless
./play_b2w_pathaware.sh --task Isaac-Nav-PPO-B2W-Play-v0 --checkpoint <model.pt> --num_envs 1 --headless
```

Go2W（本工作区新增的 path-aware 任务）：

```bash
./train_go2w_pathaware.sh --num_envs 32 --max_iterations 10 --headless
./play_go2w_pathaware.sh --checkpoint <model.pt> --num_envs 1 --headless
```

可用任务 ID：

- `Isaac-Nav-PPO-B2W-v0`、`Isaac-Nav-MDPO-B2W-v0` 及对应 `*-Play-v0` / `*-Dev-v0`
- `Isaac-Nav-PPO-Go2W-PathAware-v0`、`Isaac-Nav-MDPO-Go2W-PathAware-v0` 及对应 `*-Play-v0` / `*-Dev-v0`

## Go2W 说明

path-aware sim 分支没有 Go2W 任务；本工作区从原 sru-learning sim 分支 `sru-navigation-sim`（`e6c75f2`）以**加法方式**移植了：

- Go2W 机器人资产、低层运动策略、地形配置
- Go2W 任务/agent 配置，并注册 `*-PathAware-*` 任务 ID
- 原分支独有的 heading/goal 辅助函数（追加，不覆盖 path-aware 代码）
- Go2W 仿真 raycast camera 改用 RealSense D435 内参（87°×58° @ 640×480 直接 resize 到 64×40）；安装位姿设为 `(0.285, 0.0, 0.09)`、20° 下俯（相对自带相机 +1cm 前向、+4cm 上方、横向居中）。噪声模型和相机域随机化参数保持不变。

Go2W 的 Phase1-5、LowObs、Heading 等原任务 ID 也一并在本工作区注册；但只有 `*-PathAware-*`  aliases 明确保证继承 `target_path` 观测和 `progress_along_path` 奖励。所有移植改动都在 `sru-navigation-sim` 中，尚未提交。

## 本工作区的兼容性补丁

这些补丁只作用于本工作区，尚未提交到各仓库：

- `IsaacLab/source/isaaclab/isaaclab/utils/io/pickle.py` 与 `__init__.py`：补回训练脚本需要的 `dump_pickle` / `load_pickle`
- `IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter.py`：对旧 Isaac Sim 缺少的 `set_merge_fixed_ignore_inertia` 做存在性检查
- `sru-navigation-sim/.../navigation/mdp/navigation/goal_commands.py`：允许首次 reset 前读取路径观测时惰性初始化或返回零路径
- `sru-navigation-learning/rsl_rl/runners/on_policy_runner.py`：兼容 IsaacLab/Gymnasium 字典式观测
- `sru-navigation-sim/scripts/play.py`：新增 `--max_steps`，并兼容字典式观测
- `sru-navigation-sim/.../terrains/__init__.py`：低障碍地形改为懒加载，避免阻塞主任务导入

## 冒烟测试注意事项

- 训练至少使用 32 个环境；`--num_envs 1` 会让 PPO 小批次为空并触发 `batch_size=0` 的 reshape 错误。
- 首次环境构建会为每个地形构建 PRM（默认每地形 3000 个采样点，共约 180 个地形），通常需要约 10-12 分钟；训练脚本超时需相应放宽。

## 验证结果（2026-11-20）

释放显存后，四项无头冒烟测试全部通过（exit 0，无 traceback）：

| 测试 | 任务 | 结果 |
|---|---|---|
| B2W 训练 | `Isaac-Nav-PPO-B2W-v0`，32 envs / 1 iter | 完成迭代，保存 `IsaacLab/logs/rsl_rl/b2w_navigation_ppo/2026-11-20_14-11-06_smoke_b2w_v7/model_0.pt` |
| B2W 回放 | `Isaac-Nav-PPO-B2W-Play-v0`，`--max_steps 2` | 成功加载 checkpoint 并完成 2 步推理 |
| Go2W 训练 | `Isaac-Nav-PPO-Go2W-PathAware-v0`，32 envs / 1 iter | 完成迭代，`progress_along_path` 奖励参与计算，保存 `IsaacLab/logs/rsl_rl/go2w_navigation_ppo_phase4/2026-11-20_14-24-15_smoke_go2w_v1/model_0.pt` |
| Go2W 回放 | `Isaac-Nav-PPO-Go2W-PathAware-Play-v0`，`--max_steps 2` | 成功加载 checkpoint 并完成 2 步推理 |

## 仿真前置条件

Isaac Sim 需要足够的空闲显存。测试前 `llama.cpp/llama-server` 占用约 23GB，导致无头仿真出现 Vulkan `ERROR_OUT_OF_DEVICE_MEMORY`；用户释放后（GPU 空闲约 23.7GB）上述测试全部通过。若后续显存再次被占满，需先释放或换用其他 GPU。
