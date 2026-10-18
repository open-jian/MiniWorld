<div align="center">

# MiniWorld

### Democratizing the Training of Video World Models from Scratch

<a href="https://zhao-yian.github.io/MiniWorld/"><img src="https://img.shields.io/badge/Project-Page-1f6feb?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2608.01127"><img src="https://img.shields.io/badge/arXiv-2608.01127-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="https://huggingface.co/zhaoyian01/MiniWorld"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-ffcc4d?style=for-the-badge" alt="Hugging Face"></a>

[安装](#安装) · [推理](#流式推理) · [训练](#训练) · [引用](#引用) · [English](README.md)

</div>

## 简介

MiniWorld 是一个用于**从零训练流式视频世界模型**的紧凑框架。它不依赖对预训练双向视频生成器的改造，而是用块因果（block-causal）Video Diffusion Transformer 配合 Rectified Flow，直接学习因果的下一状态预测。

同一套架构支持两种控制模态：

- **DROID：** 用底层机器人动作做具身世界建模。
- **RealEstate10K：** 用相机位姿做可控场景预测。

MiniWorld 融合了 chunk 级 Diffusion Forcing、非递减噪声调度、长上下文续训以及滚动 KV 流式推理。完整模型可以在单台 8 卡服务器上用几天时间训完，为研究长时程生成、时序记忆以及训推一致性提供一个透明的基线。

## 效果展示

每一格都是 `1B` 模型的 253 帧流式 rollout，只给定一帧观测和对应的控制信号。

<p align="center">
  <a href="https://zhao-yian.github.io/MiniWorld/assets/demo_droid.mp4"><img src="assets/demo_droid.webp" width="49%" alt="DROID 动作条件 rollout"></a>
  <a href="https://zhao-yian.github.io/MiniWorld/assets/demo_re10k.mp4"><img src="assets/demo_re10k.webp" width="49%" alt="RealEstate10K 相机条件 rollout"></a>
</p>
<p align="center">
  <sub>左：<b>DROID</b> 动作条件 rollout；右：<b>RealEstate10K</b> 相机条件 rollout，均为 2 倍速播放。点击任一宫格可查看原分辨率视频，全部 100 段 rollout 见<a href="https://zhao-yian.github.io/MiniWorld/">项目主页</a>。</sub>
</p>

## 方法概览

<p align="center">
  <img src="assets/miniworld_overview.png" width="100%" alt="MiniWorld 方法概览">
</p>

训练时，预训练的 Wan2.2 VAE 把视频映射为紧凑的 latent 序列。MiniWorld 将这些序列切分成时间维度上的 chunk，用 chunk 级非递减噪声调度训练一个块因果 Video DiT。推理时，已完成的 chunk 被提交进结构化的滚动 KV 缓存，同时一个有界的 in-flight 窗口在异步去噪。

核心组件：

- **块因果 Video DiT**：chunk 内部双向注意力，chunk 之间因果注意力。
- **统一条件注入**：机器人动作与相机位姿通过 AdaLN-LoRA 调制走同一条路径。
- **面向 chunk 的概率传播（CoPP）**：保证非递减扩散调度的稳定性。
- **长上下文续训**：从短片段逐步扩展到 253 帧序列。
- **结构化滚动 KV 缓存**：常驻 sink 帧 + FIFO 历史。
- **流水线式异步去噪**：在推理时权衡质量与吞吐。

## 更新日志

- **2026-07：** 发布 DROID 与 RealEstate10K 的训练、流式推理与吞吐基准测试代码。

## 安装

### 环境要求

- Linux，且具备 NVIDIA CUDA GPU
- Python 3.11
- 与 CUDA 匹配的 PyTorch 2.x
- FlashAttention
- 复现默认训练配方建议 8 卡；推理单卡即可

MiniWorld 依赖 CUDA、NCCL、bf16 和 FlashAttention，目前不支持纯 CPU 与 macOS 运行。

### 创建环境

以下命令均在仓库根目录执行：

```bash
conda create -n miniworld python=3.11 -y
conda activate miniworld

# 安装与你的 CUDA 版本匹配的 PyTorch，这里以 CUDA 12.4 为例。
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# FlashAttention 编译时需要读取已安装的 PyTorch，因此要加 --no-build-isolation。
pip install -r requirements.txt --no-build-isolation
```

验证安装：

```bash
python -c "import torch, flash_attn; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)"
```

> [!IMPORTANT]
> 本仓库刻意保持轻量，目前不提供 `setup.py` 或 `pyproject.toml`。请在仓库根目录执行命令；若要在其他位置调用这些 Python 模块，请在命令前加上 `PYTHONPATH=.`。

## 数据与权重

### MiniWorld 权重

DROID 与 RealEstate10K 的预训练权重托管在 [zhaoyian01/MiniWorld](https://huggingface.co/zhaoyian01/MiniWorld)：

```bash
hf download zhaoyian01/MiniWorld \
  --include "MiniWorld_1b_droid.pt" \
  --local-dir checkpoints/miniworld
```

完整列表见[模型配置](#模型配置)。

### Wan2.2 VAE

训练和推理都需要官方的高压缩率 `Wan2.2_VAE.pth`，来自 [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/main/Wan2.2_VAE.pth)：

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
hf download Wan-AI/Wan2.2-TI2V-5B \
  --include "Wan2.2_VAE.pth" \
  --local-dir checkpoints/wan2.2
```

下载后的 VAE 路径为：

```text
checkpoints/wan2.2/Wan2.2_VAE.pth
```

### DROID

DROID 加载器要求 [DreamZero-DROID-Data](https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data) 使用的 LeRobot-v2 目录结构：

```text
/path/to/droid_lerobot/
├── meta/
├── data/
└── videos/
```

可以这样下载数据集：

```bash
hf download GEAR-Dreams/DreamZero-DROID-Data \
  --type dataset \
  --local-dir /path/to/droid_lerobot
```

### RealEstate10K

RealEstate10K [官方页面](https://google.github.io/realestate10k/download.html)只提供相机轨迹文本，视频需要自行从 YouTube 下载并切分。推荐直接使用 [DFoT](https://huggingface.co/kiwhansong/DFoT/tree/main/datasets) 发布的预处理版本，视频和位姿标注都已经准备好：

```bash
hf download kiwhansong/DFoT \
  --include "datasets/RealEstate10K_Full.tar.gz.part-*" \
  --local-dir /path/to/re10k_download

cat /path/to/re10k_download/datasets/RealEstate10K_Full.tar.gz.part-* \
  | tar -xzf - -C /path/to/re10k
```

解压后包含训练集和测试集两套目录，视频与位姿按文件名一一对应：

```text
/path/to/re10k/
├── training_256/
│   └── {clip_id}.mp4
├── training_poses/
│   └── {clip_id}.pt
├── test_256/
└── test_poses/
```

把 `DATA_ROOT` 指向 `training_256/`、`POSE_DIR` 指向 `training_poses/` 即可，无需做任何格式转换：位姿是逐帧 18 维张量，正是 `miniworld/data/re10k.py` 期望的布局。视频为 256×256、10 fps，加载时会被缩放到 240×320。


## 模型配置

| 模型 | 层数 | 宽度 | 头数 | 参数量 | 权重 |
| --- | ---: | ---: | ---: | ---: | --- |
| `B` | 12 | 768 | 12 | 0.12B | -- |
| `L` | 24 | 1024 | 16 | 0.39B | -- |
| `0.5B` | 28 | 1152 | 16 | 0.55B | [DROID](https://huggingface.co/zhaoyian01/MiniWorld/blob/main/MiniWorld_0_5b_droid.pt) · [RealEstate10K](https://huggingface.co/zhaoyian01/MiniWorld/blob/main/MiniWorld_0_5b_re10k.pt) |
| `1B` | 28 | 1536 | 12 | 1B | [DROID](https://huggingface.co/zhaoyian01/MiniWorld/blob/main/MiniWorld_1b_droid.pt) · [RealEstate10K](https://huggingface.co/zhaoyian01/MiniWorld/blob/main/MiniWorld_1b_re10k.pt) |
| `3B` | 32 | 2560 | 20 | 3B | -- |

所有启动脚本默认使用 `MODEL=1B`。已发布的权重都放在 Hugging Face 仓库
[zhaoyian01/MiniWorld](https://huggingface.co/zhaoyian01/MiniWorld)。

## 流式推理

默认采样器的配置为：

- 以 1 帧观测作为初始上下文；
- 8 个 in-flight chunk + 24 个 chunk 的滚动 KV 缓存，即 64 帧的有效注意力窗口；
- 1 帧常驻 sink；
- 100 步去噪，CFG scale 为 2.0；
- rollout 长度 64 个 latent 帧，对应 253 帧 RGB。

生成的视频保存在 `${SAMPLE_DIR}/pred/`。

推理优化统一由仓库根目录的 [`optimization_config.toml`](optimization_config.toml)
控制。当前默认配置：

```toml
schema_version = 1
chunkwise_flash_attention = true
framewise_conditioning = true
fused_norm_modulation = true
reuse_rope_frequencies = true
fused_residual_gate_add = true
```

将 `chunkwise_flash_attention` 改为 `false` 后，流式 `forward_with_cache` 使用原来的带 mask SDPA。
`true` 表示请求开启，CPU、FP32 或不支持的硬件/head_dim 仍按原逻辑回退。
配置不包含 sink、窗口、采样步数或精度；后续优化实现时再加入对应开关。

`framewise_conditioning = true` 将共享调制、各层 AdaLN/LoRA 和输出层条件投影
改为每个 latent 帧计算一次，再向空间 token 广播。15×20 个 token 的情况下，
这些投影处理的行数减少 300 倍，不代表整体推理加速 300 倍。
调制张量保留大小为 1 的空间维度，不再物化重复的逐 token 调制张量。
仅作用于 eval 模式下、使用全局或逐帧 timestep 的 `forward_with_cache`；
逐 token timestep 和普通/训练 `forward` 保留原布局，空间变化的 pose 调制不折叠。
每次前向（包括各去噪步和 CFG 分支）重新计算，checkpoint 字段不变。
较小的 GEMM 可能改变 BF16 舍入结果，需对实际任务验证画质。
仓库配置、Python 配置以及省略该字段时均默认开启；显式设为 `false` 可使用原来的逐 token 投影路径。

`fused_norm_modulation = true` 将各 block 和输出层的 hidden-state RMSNorm 与
AdaLN 合并为两个 Triton kernel，直接读取原有 channel-major 特征排列，不复制整个
张量来调整布局。保留 FP32 统计与权重、原 epsilon，以及归一化结果和 `1 + scale`
的 BF16 舍入；归约顺序仍可能造成少量数值差异。仅用于关闭梯度的 cached eval、
CUDA BF16 特征、FP32 norm 权重及支持的逐 token／逐帧调制布局。
不支持的输入或未安装 Triton 时保留原计算路径，运行错误不会被静默忽略。
Q/K norm、条件编码 norm 和残差相加保持独立。显式设为 `false` 可以在保留其他
优化的同时单独比较融合效果。

`reuse_rope_frequencies = true` 在关闭梯度的 cached eval 中，将原来的 RoPE
正余弦系数供同一次前向的 Q/K 和各层共用。原来每层分别为 Q 和 K 获取系数，
28 层共 56 次；开启后同一次前向只获取一次。保留原构表设备、dtype 和使用 checkpoint
已有表的特殊路径；系数只存在于当前调用，不跨去噪步缓存，不改变历史淘汰与位置重编号。
普通前向、训练、自定义 RoPE 实现或 forward hook 保留原路径；设为 `false` 可单独对照。

`fused_residual_gate_add = true` 将每个 block 两处的 `gate * update` 和残差相加
合并到一个 Triton kernel。保留乘法结果先舍入 BF16、再相加的顺序，不改成只舍入
一次的融合乘加。直接按实际 stride 读取并保留残差布局，省去 gated-update 中间张量，
不增加长期缓存。仅作用于关闭梯度、受支持的 CUDA BF16 cached eval；普通前向、训练、
不支持的输入或未安装 Triton 时保留原操作。设为 `false` 只关闭这一项，其他开关独立，
checkpoint 字段、sink、历史窗口和采样设置不变。

直接运行采样入口时可加 `--optimization_config /path/to/experiment.toml`；
上述 DROID、RE10K 采样脚本和吞吐脚本支持 `OPTIMIZATION_CONFIG=/path/to/experiment.toml`。
Python 调用可以向 `DenoiserConfig` 或 `MiniWorldModel` 传入
`optimization_config="/path/to/experiment.toml"`。不指定时使用当前代码仓库内的默认文件，
不依赖启动目录。每个模型只在创建时读取一次；修改文件后需重新创建模型。

优先级为：`forward_with_cache(use_chunk_flash_attn=True/False)` 的显式单次覆盖
> 模型已加载的配置。省略该参数（或传 `None`）则继承配置；已有 A/B benchmark 的
显式覆盖仍有效。未知字段、非布尔开关、错误版本或不存在的文件会报错。
采样启动日志及 `${SAMPLE_DIR}/optimization_config.json` 保存本次实际加载的配置快照，
Python 流式推理的 `last_eval_meta["optimization_config"]` 也记录相同信息。

### DROID 动作条件生成

```bash
DATA_ROOT=/path/to/droid_lerobot \
CKPT=/path/to/droid_last.pt \
VAE_CKPT=checkpoints/wan2.2/Wan2.2_VAE.pth \
bash scripts/sample_droid.sh
```

### RealEstate10K 相机条件生成

```bash
DATA_ROOT=/path/to/re10k/videos \
POSE_DIR=/path/to/re10k/poses \
CKPT=/path/to/re10k_last.pt \
VAE_CKPT=checkpoints/wan2.2/Wan2.2_VAE.pth \
bash scripts/sample_re10k.sh
```

### 常用参数覆盖

shell 脚本把主要的推理开关都暴露成了环境变量：

```bash
GPU=0 \
TOTAL_LEN=64 \
CFG_SCALE=2.0 \
SAMPLE_NUM_VIDEOS=10 \
STREAM_INFLIGHT_CHUNKS=8 \
STREAM_MAX_CACHE_CHUNKS=24 \
STREAM_SINK_SIZE=1 \
bash scripts/sample_droid.sh
```

`TOTAL_LEN` 设定 rollout 的 latent 帧数，可以超过训练窗口，因为流式推理会把注意力跨度限制在有界范围内：用 64 帧的 checkpoint 设 `TOTAL_LEN=96` 能得到 381 帧 RGB。两个 stream chunk 数量则是在质量、显存和延迟之间做权衡。

### 自定义相机轨迹

用 RealEstate10K 训练的 checkpoint 可以按程序化生成的相机轨迹让一张静态图动起来：

```bash
PYTHONPATH=. python -m miniworld.sample \
  --dataset re10k \
  --init_image /path/to/first_frame.png \
  --custom_camera_trajectory orbit_right \
  --checkpoint /path/to/re10k_last.pt \
  --vae_checkpoint checkpoints/wan2.2/Wan2.2_VAE.pth \
  --sample_dir samples/re10k_orbit_right \
  --wm_model 1B \
  --total_len 64 \
  --sample_num_videos 1 \
  --trajectory_magnitude 3.0
```

可选的轨迹类型：

```text
static, forward, backward, pan_left, pan_right, tilt_up, tilt_down,
orbit_right, orbit_left, spiral, zoom_in, zoom_out
```

这里不需要 `--data_root` 和 `--pose_dir`，因为位姿是程序化生成的。`--sample_num_videos` 指定的每个样本都会重新采噪声，所以一条命令可以产出同一条轨迹的多个变体。

**如何选择运动幅度。** `--trajectory_magnitude` 是唯一真正值得调的参数。已发布的 RealEstate10K checkpoint 是在原始（未归一化）平移量上训练的，默认值 `1.0` 只会让相机移动约 0.5 个单位，画面看起来几乎是静止的。另外轨迹会均匀铺满 `--total_len`，所以同样的幅度在更短的 rollout 里意味着更快的逐帧运动。在 `--total_len 64` 下的大致可用区间是：

| `--trajectory_magnitude` | 效果 |
| --- | --- |
| 1.0 | 几乎静止 |
| 2.0 | 运动轻微，全程稳定 |
| 3.0 | 运动明显且稳定 —— 推荐默认值 |
| 5.0 | 运动较快，末尾有轻微拖影 |
| 8.0 | 后半段结构崩坏 |

想保持相同的视觉速度，就让幅度随 rollout 长度一起缩放：`--total_len 96` 时约为 `4.5`。如果 rollout 后段画面变差，优先把幅度调小。

`--init_image` 是必填的。`--init_pose` 是可选的，作用仅仅是用某个真实 RealEstate10K 片段的内参覆盖默认内参；不传的话内参来自 `--trajectory_focal_norm`（默认 `0.5`，与 RealEstate10K 的典型取值一致）。无论是否传入，运动本身始终遵循所选的轨迹类型。

## 训练

公开的启动脚本把论文里的短时程 + 长时程续训配方落成了四个具体的课程阶段：

| 阶段 | latent 帧数 | 默认 batch/GPU | 时长 | 学习率 | 初始化 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 6 | 8 | 100 epoch | `1e-4` | 从零开始 |
| 2 | 16 | 4 | 50 epoch | `2e-5` | 阶段 1 |
| 3 | 32 | 1 | 30k step | `2e-5` | 阶段 2 |
| 4 | 64 | 1 | 30k step | `2e-5` | 阶段 3 |

阶段 1、2 按固定的 epoch 数训练（`STAGE1_EPOCHS`、`STAGE2_EPOCHS`）；长上下文阶段则按固定的优化器步数训练（`STAGE3_MAX_TRAIN_STEPS`、`STAGE4_MAX_TRAIN_STEPS`），这样它们的开销不会随数据集规模变化。

所有阶段都使用 240×320 的视频、latent chunk size 为 2、bf16 混合精度以及 Muon 优化器。

### 在 DROID 上训练

```bash
DATA_ROOT=/path/to/droid_lerobot \
VAE_CKPT=checkpoints/wan2.2/Wan2.2_VAE.pth \
MODEL=1B \
bash scripts/train_droid.sh
```

最终 checkpoint：

```text
outputs/droid_1B/stage4_lf64/last.pt
```

### 在 RealEstate10K 上训练

```bash
DATA_ROOT=/path/to/re10k/videos \
POSE_DIR=/path/to/re10k/poses \
VAE_CKPT=checkpoints/wan2.2/Wan2.2_VAE.pth \
MODEL=1B \
bash scripts/train_re10k.sh
```

最终 checkpoint：

```text
outputs/re10k_1B/stage4_lf64/last.pt
```

### 分布式与资源配置覆盖

两个脚本都通过 `torchrun` 启动，默认起 8 个本地进程。多机配置无需修改脚本，直接用环境变量传入：

```bash
NNODES=2 \
NODE_RANK=0 \
NPROC_PER_NODE=8 \
MASTER_ADDR=10.0.0.1 \
MASTER_PORT=12471 \
bash scripts/train_droid.sh
```

换模型规模或输出目录：

```bash
MODEL=3B OUTPUT_DIR=outputs/droid_3B bash scripts/train_droid.sh
MODEL=0.5B OUTPUT_DIR=outputs/re10k_0.5B bash scripts/train_re10k.sh
```

单卡实验时，设 `NPROC_PER_NODE=1` 并按需调小各阶段的 batch size：

```bash
NPROC_PER_NODE=1 \
STAGE1_BATCH_SIZE=1 \
STAGE2_BATCH_SIZE=1 \
STAGE3_BATCH_SIZE=1 \
STAGE4_BATCH_SIZE=1 \
bash scripts/train_droid.sh
```

### 视频长度缓存（仅 RealEstate10K）

一个片段需要 `4 * (latent_frames - 1) + 1` 帧原始视频，因此 RealEstate10K 会在训练开始前先过滤掉长度不够的视频。帧数没有任何地方记录，只能把每个视频打开读一遍，而且每个 rank 每次启动都要重复扫描一次。

`FILTER_CACHE_DIR` 是可选项，它会把扫描结果以 JSON 持久化，键是「文件列表 + 所需帧数」的哈希：

```bash
DATA_ROOT=/path/to/re10k/videos \
POSE_DIR=/path/to/re10k/poses \
VAE_CKPT=/path/to/Wan2.2_VAE.pth \
FILTER_CACHE_DIR=/path/to/re10k/filter_cache \
bash scripts/train_re10k.sh
```

每个课程阶段需要的帧数不同，因而各自对应一条缓存记录，但重跑和续训都能直接复用。不设 `FILTER_CACHE_DIR` 也能正常训练，只是每次都要重新扫一遍。

`scripts/train_droid.sh` 没有这个选项，因为 DROID 根本不需要扫描：它是 LeRobot 格式的数据集，episode 长度直接从 `meta/episodes.jsonl` 读取，过滤过程完全不碰视频文件。

### 日志

训练过程中，rank 0 每 `--log_every` 步会把 loss、学习率和吞吐记录到 [Weights & Biases](https://wandb.ai)。先执行一次 `wandb login`，然后用 `WANDB_PROJECT` 修改项目名（默认 `miniworld`）；每个课程阶段是一个独立的 run，以其输出目录命名。

每 `--image_log_every` 步（默认 1000），rank 0 还会额外记录两段视频，都是真值与模型输出左右并排：

| 面板 | 内容 | 怎么看 |
| --- | --- | --- |
| `train/recon_video` | 对刚训练过的那个 batch 做单步去噪、预测出的 clean latent，caption 标注了当步的噪声水平 `t` | 去噪目标的健全性检查。它很早就会看起来不错并一直不错，因为从较低的 `t` 走一步本来就简单。 |
| `train/gen_video` | 用 EMA 权重、从一个固定片段完整 rollout 出 `--latent_frames` 帧，caption 标注步数 | 真正反映采样质量随训练变化的指标。 |


## 吞吐基准测试

DROID 的基准测试会关闭 CFG、先做预热、跳过视频写盘，并记录 chunk 级的耗时：

```bash
DATA_ROOT=/path/to/droid_lerobot \
CKPT=/path/to/droid_last.pt \
VAE_CKPT=checkpoints/wan2.2/Wan2.2_VAE.pth \
bash scripts/benchmark_droid_throughput.sh
```

结果写入：

```text
${SAMPLE_DIR}/throughput_timing.jsonl
```

每一行会报告首个 chunk 的延迟、稳态端到端吞吐、DiT 吞吐以及 VAE 吞吐。

## 仓库结构

```text
MiniWorld/
├── miniworld/
│   ├── miniworld.py          # 块因果 Video DiT
│   ├── denoiser.py           # AR 扩散与滚动 KV 生成
│   ├── train.py              # 训练入口
│   ├── sample.py             # 流式推理入口
│   ├── conditioning/         # 动作、位姿与轨迹条件
│   ├── data/                 # DROID 与 RealEstate10K 数据加载
│   └── vae/                  # Wan2.2 VAE 编解码
├── scripts/                  # 训练、采样与基准测试启动脚本
├── assets/                   # README 配图与效果示例
├── requirements.txt
└── README.md
```

## 引用

论文已发布在 arXiv：[arXiv:2608.01127](https://arxiv.org/abs/2608.01127)。

如果 MiniWorld 对你的研究有帮助，欢迎引用：

```bibtex
@article{zhao2026miniworld,
  title   = {MiniWorld: Democratizing the Training of Video World Models from Scratch},
  author  = {Zhao, Yian and Zheng, Ruochong and Guo, Hongcan and Yan, Yu and Zhang, Jian and Chen, Jie},
  journal = {arXiv preprint arXiv:2608.01127},
  year    = {2026}
}
```
