# AIDE — AI 生成图像三分支检测器

**论文：** [A Sanity Check for AI-generated Image Detection](https://arxiv.org/abs/2406.19435)
**会议：** ICLR 2025
**作者：** Shilin Yan, Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu, Weidi Xie
**单位：** 小红书 · 中国科学技术大学 · 上海交通大学


## 项目简介

AIDE（**A**I-generated **I**mage **DE**tector）是一个用于检测 AI 生成图像的多分支检测框架。
本仓库在原始双分支版本基础上，新增了 **NPR（自然模式残差）分支**，构成三分支架构，并提供完整的训练、评测和对比流程。

核心贡献：
- **Chameleon 数据集**：收录真实场景中经过精心调整的 AI 生成图像，显著难于现有基准
- **AIDE 三分支模型**：同时利用视觉伪影、语义特征和噪声模式三类信号进行检测
- **公平对比工具**：支持三分支与双分支在同一 AIGCBenchmark 上的等协议评测


## 方法

### 三分支架构

```
输入图像 (batch, 5, 3, 256, 256)
     │
     ├─── 分支一：HPF + 双 ResNet-50
     │         DCT 频域分解 → 高通滤波(30核 SRM) → ResNet-50 × 2
     │         输出：2048 维特征
     │
     ├─── 分支二：冻结 ConvNeXt-XXL（OpenCLIP）
     │         原始 RGB 图像 → ConvNeXt-XXL trunk（冻结）
     │         3072 → 256 线性投影
     │         输出：256 维特征
     │
     └─── 分支三：NPR 残差分支（新增）
               x_npr = x - upsample(downsample(x, 0.5))
               → NPR-ResNet-50 → 512 → 128 线性投影
               输出：128 维特征（训练时 dropout=0.3）

融合：
  [2048 + 256] → Linear(1024) + GELU      # 保留双分支融合表示
  [1024 + 128] → Linear(512) → Linear(2)  # 最终分类
```

双分支（旧版 `AIDE_2BRANCH`）不含 NPR 分支，直接 `[2048 + 256] → Linear(1024) → Linear(2)`。


## 环境安装

```bash
# 1. 克隆仓库
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. 安装 PyTorch（需要 CUDA 11.8）
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch

# 3. 安装其余依赖
pip install -r requirements.txt
```

**测试环境：**

| 依赖 | 版本 |
|------|------|
| CUDA | 11.8 |
| Python | 3.10 |
| PyTorch | 2.0.1 |
| open-clip-torch | 2.24.0 |
| timm | 0.9.6 |


## 预训练权重

所有权重统一放置于 `pretrained_ckpts/` 目录：

```
pretrained_ckpts/
├── resnet50.pth                  # ImageNet 预训练 ResNet-50
├── open_clip_pytorch_model.bin   # OpenCLIP ConvNeXt-XXL 权重
└── NPR.pth                       # NPR 特征提取器预训练权重
```

| 权重文件 | 来源 |
|----------|------|
| `resnet50.pth` | [torchvision 官方](https://download.pytorch.org/models/resnet50-0676ba61.pth) |
| `open_clip_pytorch_model.bin` | [OpenCLIP convnext_xxlarge](https://huggingface.co/laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup) |
| `NPR.pth` | [原始 NPR 论文仓库](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) |

训练好的三分支 checkpoint 可从 [Google Drive](https://drive.google.com/drive/folders/1qx76UFvDpgCxaPLBCmsA2WY-SSzeJrd4?usp=sharing) 下载。


## 数据集准备

### 训练集

采用 [CNNSpot](https://github.com/peterwang512/CNNDetection) 和 [GenImage](https://github.com/Andrew-Zhu/GenImage) 的训练划分。

目录结构（`0_real` 和 `1_fake` 固定命名）：

```
train_data/
├── 0_real/
│   ├── img_001.jpg
│   └── ...
└── 1_fake/
    ├── img_001.jpg
    └── ...
```

### 测试集：AIGCBenchmark

从 [AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark) 下载，包含 16 个生成器子目录：

```
AIGCBenchmark/
├── progan/            # 每个子目录内含 0_real/ 和 1_fake/
├── stylegan/
├── biggan/
├── cyclegan/
├── stargan/
├── gaugan/
├── stylegan2/
├── whichfaceisreal/
├── ADM/
├── Glide/
├── Midjourney/
├── stable_diffusion_v_1_4/
├── stable_diffusion_v_1_5/
├── VQDM/
├── wukong/
└── DALLE2/
```

> 评测脚本通过 `len(subdirs) == 16` 自动识别 AIGCBenchmark 协议（见 `main_finetune.py:347`）。


## 训练三分支模型

修改 `scripts/train_3branch.sh` 中的路径变量：

```bash
TRAIN_DATA="/path/to/train_data"
EVAL_DATA="/path/to/val_data"
RESNET_PATH="pretrained_ckpts/resnet50.pth"
CONVNEXT_PATH="pretrained_ckpts/open_clip_pytorch_model.bin"
NPR_PATH="pretrained_ckpts/NPR.pth"
OUTPUT_DIR="results/3branch_train"
```

然后运行：

```bash
bash scripts/train_3branch.sh
```

**关键超参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 4 | 单卡 batch size |
| `--update_freq` | 4 | 梯度累积步数（等效 batch 16） |
| `--blr` | 1e-4 | 基础学习率 |
| `--epochs` | 20 | 训练轮数 |
| `--freeze_npr` | True | 冻结 NPR 主干，只训练融合层和分类器 |
| `--npr_branch_dropout` | 0.3 | 训练时 NPR 分支 dropout |
| `--use_amp` | True | 混合精度，节省显存 |

多卡训练在命令前加 `torchrun --nproc_per_node=<N>` 即可。

训练完成后，最佳 checkpoint 保存于 `results/3branch_train/checkpoint-best.pth`。


## AIGCBenchmark 公平评测

使用同一脚本、同一协议分别评测三分支和双分支，确保结果可直接对比。

### 评测三分支（你训练的新模型）

```bash
bash scripts/eval_aigcbenchmark.sh 3branch \
    results/3branch_train/checkpoint-best.pth \
    /path/to/train_data \
    /path/to/AIGCBenchmark
```

输出：`results/eval_3branch/checkpoint-best.pth_AIGCBenchmark.csv`

### 评测双分支（官方预训练 checkpoint）

```bash
bash scripts/eval_aigcbenchmark.sh 2branch \
    pretrained_ckpts/official_2branch.pth \
    /path/to/train_data \
    /path/to/AIGCBenchmark
```

输出：`results/eval_2branch/official_2branch.pth_AIGCBenchmark.csv`

**公平性保证：** 两次评测共用同一 `TestDataset`（无增强）、同一 `evaluate()` 函数和同一 sklearn 指标计算，结果直接可比。


## 对比结果汇总

```bash
python tools/compare_benchmark.py \
    --csv3 results/eval_3branch/checkpoint-best.pth_AIGCBenchmark.csv \
    --csv2 results/eval_2branch/official_2branch.pth_AIGCBenchmark.csv \
    --save results/comparison.csv
```

输出示例：

```
----------------------------------------------------------
Generator                         ACC-3b   ACC-2b     ΔACC     AP-3b    AP-2b      ΔAP
----------------------------------------------------------
progan                            0.9812   0.9754   +0.0058   0.9901   0.9867   +0.0034
stylegan                          0.9345   0.9201   +0.0144   0.9512   0.9388   +0.0124
...
----------------------------------------------------------
MEAN                                                +0.0081                     +0.0062
```


## 引用

如果本项目对你的研究有帮助，请引用原始论文：

```bibtex
@article{yan2024sanity,
  title={A Sanity Check for AI-generated Image Detection},
  author={Yan, Shilin and Li, Ouxiang and Cai, Jiayin and Hao, Yanbin and
          Jiang, Xiaolong and Hu, Yao and Xie, Weidi},
  journal={arXiv preprint arXiv:2406.19435},
  year={2024}
}
```


## 致谢

本项目基于 [ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)，并参考了以下工作：
[CNNSpot](https://github.com/peterwang512/CNNDetection) ·
[AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark) ·
[GenImage](https://github.com/Andrew-Zhu/GenImage) ·
[DNF](https://github.com/YichiCS/DNF) ·
[NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
