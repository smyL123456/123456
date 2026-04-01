# AIDE — AI 生成图像检测项目

## 项目简介

本项目是一个面向 AI 生成图像检测的多分支检测框架，提供三分支模型的训练、评测与结果对比流程，适用于以统一协议完成模型训练和 AIGCBenchmark 测试。

## 模型结构

### 三分支架构

```text
输入图像
  ├── 分支一：HPF + ResNet-50
  │   DCT 频域分解 → 高通滤波 → 特征提取
  ├── 分支二：ConvNeXt-XXL（OpenCLIP）
  │   原始 RGB 图像 → 语义特征提取 → 线性投影
  └── 分支三：NPR 残差分支
      自然模式残差 → NPR-ResNet-50 → 线性投影

融合：
  [分支一 + 分支二] → 融合层
  [融合结果 + 分支三] → 分类层
```

仓库中同时保留双分支与三分支实现，便于在相同评测协议下进行比较。

## 环境安装

```bash
# 1. 克隆仓库
https://github.com/smyL123456/123456
cd 123456

# 2. 安装 PyTorch（CUDA 11.8）
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch

# 3. 安装其余依赖
pip install -r requirements.txt
```

### 推荐环境

| 依赖 | 版本 |
|------|------|
| CUDA | 11.8 |
| Python | 3.10 |
| PyTorch | 2.0.1 |
| open-clip-torch | 2.24.0 |
| timm | 0.9.6 |

## 预训练权重

请将所需权重统一放置在 `pretrained_ckpts/` 目录：

```text
pretrained_ckpts/
├── resnet50.pth
├── open_clip_pytorch_model.bin
└── NPR.pth
```

训练输出默认保存在 `results/` 目录下。

## 数据集准备

### 训练集

训练集仅采用 [CNNSpot](https://github.com/peterwang512/CNNDetection) 的训练划分。

目录结构要求如下（`0_real` 和 `1_fake` 为固定命名）：

```text
train_data/
├── 0_real/
│   ├── img_001.jpg
│   └── ...
└── 1_fake/
    ├── img_001.jpg
    └── ...
```

### 测试集

测试集采用 [AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark)。评测时需要保证测试目录下包含各生成器子目录，并维持数据集原始结构。

示例：

```text
AIGCBenchmark/
├── progan/
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

## 训练三分支模型

先根据本地环境修改 `scripts/train_3branch.sh` 中的路径：

```bash
TRAIN_DATA="/path/to/train_data"
EVAL_DATA="/path/to/val_data"
RESNET_PATH="pretrained_ckpts/resnet50.pth"
CONVNEXT_PATH="pretrained_ckpts/open_clip_pytorch_model.bin"
NPR_PATH="pretrained_ckpts/NPR.pth"
OUTPUT_DIR="results/3branch_train"
```

然后执行：

```bash
bash scripts/train_3branch.sh
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 4 | 单卡 batch size |
| `--update_freq` | 4 | 梯度累积步数 |
| `--blr` | 1e-4 | 基础学习率 |
| `--epochs` | 20 | 训练轮数 |
| `--freeze_npr` | True | 冻结 NPR 主干 |
| `--npr_branch_dropout` | 0.3 | NPR 分支 dropout |
| `--use_amp` | True | 混合精度训练 |

训练完成后，最佳模型默认保存为 `results/3branch_train/checkpoint-best.pth`。

## 模型评测

### AIGCBenchmark 评测三分支模型

```bash
bash scripts/eval_aigcbenchmark.sh 3branch \
    results/3branch_train/checkpoint-best.pth \
    /path/to/train_data \
    /path/to/AIGCBenchmark
```

输出文件默认位于：

```text
results/eval_3branch/checkpoint-best.pth_AIGCBenchmark.csv
```

### 对比双分支结果

```bash
bash scripts/eval_aigcbenchmark.sh 2branch \
    pretrained_ckpts/official_2branch.pth \
    /path/to/train_data \
    /path/to/AIGCBenchmark
```

## 结果汇总

可使用以下脚本汇总双分支与三分支在 AIGCBenchmark 上的结果：

```bash
python tools/compare_benchmark.py \
    --csv3 results/eval_3branch/checkpoint-best.pth_AIGCBenchmark.csv \
    --csv2 results/eval_2branch/official_2branch.pth_AIGCBenchmark.csv \
    --save results/comparison.csv
```
