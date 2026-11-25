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

先激活环境（或显式指定）：四个 launcher 按 **`$CONDA_ENV_DIR` > 当前激活的 conda env (`$CONDA_PREFIX`) >
作者机器上的默认路径** 解析 Python，所以换机器/换用户名只需要 `conda activate <env>`，不用改脚本。

```bash
conda activate <你的环境名>          # 例如 env_sru_path
# 或：CONDA_ENV_DIR=/path/to/env ./train_go2w_pathaware.sh ...
```

B2W：

```bash
# 冒烟
./train_b2w_pathaware.sh --num_envs 32 --max_iterations 10 --headless
./play_b2w_pathaware.sh --task Isaac-Nav-PPO-B2W-Play-v0 --checkpoint <model.pt> --num_envs 1 --headless

# 论文对齐：180 地形 / 1046 envs / 10 m 深度相机
./train_b2w_pathaware.sh --task Isaac-Nav-PPO-B2W-v0 --num_envs 1046 --max_iterations 15000 --headless
```

Go2W（本工作区新增的 path-aware 任务）：

```bash
# 冒烟（Phase4 地形 + D435 相机 + path-aware v2 层）
./train_go2w_pathaware.sh --num_envs 32 --max_iterations 10 --headless --no_debug_vis
./play_go2w_pathaware.sh --checkpoint <model.pt> --num_envs 1 --headless

# 全量：与 B2W 完全相同的 MDPO 超参 / 20 cm 台阶地形 / 90 s horizon / 4000 iter
./train_go2w_pathaware.sh --task Isaac-Nav-MDPO-Go2W-PathAware-v0 --num_envs 1046 \
  --max_iterations 4000 --seed 42 --run_name go2w_pathfix_v1 --headless --no_debug_vis

# 论文对齐：无课程 + ZED-X 相机 + 与 B2W 相同的 PPO 超参（该 alias 未走 v2 层）
./train_go2w_pathaware.sh --task Isaac-Nav-PPO-Go2W-PathAware-NoCurriculum-v0 --num_envs 1046 --max_iterations 15000 --headless
./play_go2w_pathaware.sh --task Isaac-Nav-PPO-Go2W-PathAware-NoCurriculum-Play-v0 --checkpoint <model.pt> --num_envs 1 --headless
```

可用任务 ID：

- `Isaac-Nav-PPO-B2W-v0`、`Isaac-Nav-MDPO-B2W-v0` 及对应 `*-Play-v0` / `*-Dev-v0`
- `Isaac-Nav-PPO-Go2W-PathAware-v0`、`Isaac-Nav-MDPO-Go2W-PathAware-v0` 及对应 `*-Play-v0` / `*-Dev-v0`
  —— **2026-11-25 起这两个 ID 指向 path-aware v2**（与 B2W 同一层：v2 路径管线、米数进度、1.5 s 驻留、
  90 s horizon、2697/5834 维观测）。旧的 60 维任务仍可通过 Phase/NoCurriculum ID 访问，行为不变
- `Isaac-Nav-PPO-Go2W-PathAware-NoCurriculum-v0` 与 `Isaac-Nav-PPO-Go2W-PathAware-NoCurriculum-Play-v0`（论文对齐）

GUI 回放时的坐标系（`robot_goal` 的 debug 可视化，`*-Play-*` 任务默认开启）：

- 青色折线：每个 env 的规划路径，即策略观测里实际使用的 `MultiPath.paths_w`（PRM 上 A*、视线平滑、重采样后的点）
- 蓝色方块：目标点；橙色方块：出生点
- 绿色箭头：期望速度方向；红色箭头：当前速度方向

注意路径类别：走 v2 层的任务（B2W 全部、Go2W `*-PathAware-*`）是
`optimal 45% / mildly 25% / highly 20% / infeasible 10%`，噪声 ≤0.35 m 且端点锁定，画出来应该基本贴合可走走廊；
只有旧任务（Go2W Phase/LowObs/NoCurriculum）仍是 `highly_non_optimal` / `infeasible` 各 50% 加 1 m 噪声，
看起来明显绕路是**旧设定的有意行为**，不是可视化错误。`--headless` 下标记照常更新但不会显示，
训练时建议加 `--no_debug_vis`（1046 envs 下路径标记是 ~1.5 万个 cuboid 实例，且原型 USD 要从 Nucleus 拉）。

## Go2W 说明

path-aware sim 分支没有 Go2W 任务；本工作区从原 sru-learning sim 分支 `sru-navigation-sim`（`e6c75f2`）以**加法方式**移植了：

- Go2W 机器人资产、低层运动策略、地形配置
- Go2W 任务/agent 配置，并注册 `*-PathAware-*` 任务 ID
- 原分支独有的 heading/goal 辅助函数（追加，不覆盖 path-aware 代码）
- Go2W 仿真 raycast camera 改用 RealSense D435 内参（87°×58° @ 640×480 直接 resize 到 64×40）；安装位姿设为 `(0.285, 0.0, 0.09)`、20° 下俯（相对自带相机 +1cm 前向、+4cm 上方、横向居中）。噪声模型和相机域随机化参数保持不变。
- 论文对齐任务 `*-PathAware-NoCurriculum-*` 在此基础上改用 ZED-X 内参（105°×78° @ 64×40）、10 m 测距并关闭地形课程；原 D435 / Phase4 任务保持不变。

Go2W 的 Phase1-5、LowObs、Heading 等原任务 ID 也一并在本工作区注册，行为保持移植时的状态；只有
`*-PathAware-*` aliases 走 path-aware v2（与 B2W 同一层，见上一节）。所有移植改动都在 `sru-navigation-sim` 中，已提交。

## Path-aware v2（2026-11-25）

B2W 全部任务与 Go2W 的 `*-PathAware-*` 任务现在共用同一层实现
（`sru-navigation-sim/isaaclab_nav_task/navigation/config/path_aware.py::apply_path_aware_v2`），
所以两个机器人不会再各自漂移：

| 项 | v2 |
|---|---|
| 路径管线 | clearance 自适应噪声（≤0.35 m，端点锁定、窄门段不加）、拐角修复、结构化挡路事件 |
| 路径类别 | `[0.45, 0.25, 0.20, 0.10]` |
| 可行性判据 | 度量 0.31 m（实测机身半宽 + 0.05），不是 PRM 的 0.4 m 膨胀 |
| 进度奖励 | 米数制 + 限速追踪 + 三档支付门（1.2/m），直连进度 0.15/m |
| 到达判定 | 连续驻留 1.5 s（离开清零），一次性 +25；超时按当步位置判成功 |
| horizon | 90 s（`Episode_Reward/*` 的还原系数变成 ×90） |
| 观测 | 2697 / 5834（全局路径 60 + 局部 6 m 路径 60 + `distance_to_path` 1） |
| 训练 | MDPO：LR cosine→1e-4、熵退火到 0、`log_std_max` 0.1→−0.6→−1.0 移动墙 |

完整规格见 `docs/agent_doing.md`；M1–M5 实测、实施取舍、17+2 项单测与基线重评见 `docs/phase0_baseline_report.md`。

两个运行前提（都在这台机器上踩过）：

1. conda env 里的 `rsl_rl` 是 editable 安装，指向 `~/sru-path-aware-official`。两个 launcher 已把本工作区的
   `sru-navigation-learning` 放到 `PYTHONPATH` 最前面；手工起训练时也要加，否则本仓库对 MDPO 的改动**不生效**
   （`**kwargs` 会静默吞掉新参数）。
2. 本机目前访问不到 `omniverse-content-production.s3...`，训练/评测请加 `--no_debug_vis`，否则会卡在
   每项 Nucleus 资产 300 s 的超时上。

## 本工作区的兼容性补丁

这些补丁只作用于本工作区，已随本仓库的历史提交：

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
