# 煤矿输送带异物检测 (Coal Belt Foreign-Object Detection)

> 基于 YOLOv8m 的输送带异物检测系统，支持锚杆（anchor_rod）+ 大块煤（large_coal）两类目标。
> 测试集 mAP@0.5 = **0.9473**，单卡 RTX 4090 推理 **148 FPS**。

---

## 📚 目录

- [一、项目简介](#一项目简介)
- [二、运行环境](#二运行环境)
- [三、从零安装（小白入门）](#三从零安装小白入门)
- [四、获取代码](#四获取代码)
- [五、准备数据集与权重](#五准备数据集与权重)
- [六、跑通整个训练流水线](#六跑通整个训练流水线)
- [七、只想用我们训练好的模型推理](#七只想用我们训练好的模型推理)
- [八、ONNX 部署](#八onnx-部署)
- [九、常见问题](#九常见问题)
- [十、目录结构](#十目录结构)

---

## 一、项目简介

本仓库包含完整的训练 / 评估 / 部署代码。最终交付权重位于 `runs/detect/runs/r2/baseline/weights/best.pt`，对应的 ONNX 文件在同目录下。

主要算法改进：

1. **YOLOv8m + imgsz=960 多尺度训练**：用更深的骨干网络与更高的输入分辨率，处理大块煤这种尺度跨度极大的目标。
2. **Focal Loss + OHEM + WeightedRandomSampler 三件套**：构成"重要性采样"训练策略，针对类别不均衡和难样本进行三层放大。
3. 在 6 组对比实验中性能最优；详细数字见 [`docs/final_report.md`](docs/final_report.md)。

---

## 二、运行环境

| 项 | 值 |
| --- | --- |
| 操作系统 | Ubuntu 20.04 / 22.04（macOS 仅可做评估，不可训练） |
| Python | 3.12 |
| Python 包管理器 | **miniconda**（与服务器一致；本仓库不强制 uv） |
| GPU | NVIDIA RTX 3090 / 4090 / A100 任一，显存 ≥ 12 GB |
| CUDA | 12.1+（与 PyTorch 2.8.0 + cu128 兼容） |
| 磁盘 | 至少 20 GB（数据集 0.4 GB + 权重 2 GB + 训练产物 ~2 GB） |

> **如果你完全是新手**：跟着第三章一步步装就行，不需要懂背后的原理。

---

## 三、从零安装（小白入门）

> 如果你已经装好了 miniconda 和 git，可以跳到 [第四章](#四获取代码)。

### 3.1 安装 miniconda（10 分钟）

#### 3.1.1 Linux / WSL

打开终端，复制粘贴以下命令：

```bash
# 下载
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# 安装（一路回车 + 同意协议 + 默认安装到 ~/miniconda3）
bash ~/miniconda.sh -b -p $HOME/miniconda3

# 让当前 shell 能找到 conda 命令
source $HOME/miniconda3/bin/activate

# 关闭终端再开一个，验证
conda --version
# 应输出类似 conda 23.x.x
```

如果你用的是 **AutoDL 等云 GPU 平台**，miniconda 通常已经预装在 `/root/miniconda3/`，直接进入第 3.2 节。

#### 3.1.2 Windows

下载 [Miniconda3-latest-Windows-x86_64.exe](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)，双击安装，**全部按默认设置**。安装完后从开始菜单打开 `Anaconda Prompt`，验证：

```cmd
conda --version
```

#### 3.1.3 macOS

```bash
# Apple Silicon (M1/M2/M3)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate

# Intel Mac
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
```

> **macOS 没有 NVIDIA GPU，只能跑评估和推理，不能跑训练**。训练请到 Linux + CUDA 服务器。

### 3.2 安装 git

```bash
# Linux (Ubuntu / Debian)
sudo apt-get update && sudo apt-get install -y git

# macOS
brew install git    # 没有 brew 就运行 xcode-select --install

# Windows: 从 https://git-scm.com/download/win 下载安装包，按默认装
```

验证：

```bash
git --version
```

### 3.3 配置国内镜像（中国大陆用户必做）

如果你在中国大陆，跳过这步会因为网络问题装不上依赖。新建/编辑 `~/.pip/pip.conf`：

```bash
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60
EOF
```

Windows 用户：在 `%USERPROFILE%\pip\pip.ini` 写入相同内容。

如果你的服务器无法直连 GitHub（如 AutoDL），把以下两行加到 `~/.bashrc`：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
export HUGGINGFACE_HUB_CACHE="$HOME/.cache/huggingface"
```

然后 `source ~/.bashrc` 生效。

---

## 四、获取代码

```bash
# 任选一个目录，比如 ~/projects
cd ~/projects
git clone https://github.com/BeastOrange/NetTrack.git
cd NetTrack
```

---

## 五、准备数据集与权重

### 5.1 创建 conda 环境并安装依赖

```bash
# 在仓库根目录下
conda create -n coalbelt python=3.12 -y
conda activate coalbelt

# 安装本项目依赖
pip install --upgrade pip
pip install -e .
```

> 这一步会装 `ultralytics`, `torch`, `torchvision`, `opencv-python` 等。如果有 CUDA GPU，pip 会自动选 CUDA 版的 PyTorch；如果没有 GPU，会装 CPU 版（仍能做评估和推理，只是慢）。
>
> 如果你需要跑数据集预处理（模型辅助标注流程）：`pip install -e ".[gd]"`

### 5.2 下载数据集

CUMT-BelT 数据集已托管在中国矿业大学智能感知实验室。从以下任一源获取：

- 官方 GitHub: <https://github.com/CUMT-AIPR-Lab/CUMT-AIPR-Lab>
- 百度网盘镜像: <https://pan.baidu.com/s/1AJsjkPqXjkIJY8KQQdKfcw?pwd=z39g>

下载后解压到仓库根目录，使其形成如下结构：

```text
NetTrack/
└── CUMT-BelT/
    ├── 训练集/
    │   ├── 大块/         (1300 张 *.jpg)
    │   ├── 锚杆/         (1300 张 *.jpg)
    │   └── 正常煤流/     (1300 张 *.jpg)
    └── 测试集/
        ├── 大块/         (300 张 *.jpg)
        ├── 锚杆/         (300 张 *.jpg)
        └── 正常煤流/     (300 张 *.jpg)
```

### 5.3 下载预训练权重

```bash
# 在仓库根目录下
python scripts/prefetch_weights.py
```

这会把 `yolov8s.pt`、`yolov8m.pt`、`yolov5su.pt`、`yolov8l-world.pt` 以及 torchvision 的 Faster R-CNN 权重下载到 `weights/cache/`。下载完后建立软链接（Windows 用复制即可）：

```bash
ln -sf weights/cache/ultralytics/yolov8m.pt yolov8m.pt
ln -sf weights/cache/ultralytics/yolov8s.pt yolov8s.pt
ln -sf weights/cache/ultralytics/yolov5su.pt yolov5su.pt
```

> 如果服务器无法访问 GitHub release 链接，把 `weights/cache/` 目录从一台能联网的机器整体 rsync/scp 过来即可。

---

## 六、跑通整个训练流水线

### 6.1 数据集预处理（标注 + 划分）

```bash
# 1. 在 train/test 图像上生成 YOLO 格式的边界框标注
python scripts/pseudo_label_gd.py \
    --src CUMT-BelT/训练集 \
    --out-labels data/cumt_belt_yolo/labels/_train_raw \
    --out-images data/cumt_belt_yolo/images/_train_raw \
    --device cuda:0

python scripts/pseudo_label_gd.py \
    --src CUMT-BelT/测试集 \
    --out-labels data/cumt_belt_yolo/labels/test \
    --out-images data/cumt_belt_yolo/images/test \
    --device cuda:0

# 2. 9:1 划分 train/val，写入 data.yaml
python scripts/prepare_dataset.py
```

完成后 `data/cumt_belt_yolo/` 下会得到：

```text
data/cumt_belt_yolo/
├── images/{train,val,test}/*.jpg
├── labels/{train,val,test}/*.txt
└── data.yaml
```

### 6.2 训练改进版模型（最终交付）

```bash
# 在 NetTrack/ 根目录下；需要 GPU
PYTHONPATH=. python scripts/train.py \
    --variant baseline \
    --weights yolov8m.pt \
    --imgsz 960 \
    --epochs 100 \
    --batch 12 \
    --workers 4 \
    --device cuda:0 \
    --project runs/r2
```

训练大约 1.5 小时（RTX 4090）。中途可以用 `tensorboard --logdir runs/r2/` 查看曲线。最终权重位于 `runs/r2/baseline/weights/best.pt`。

### 6.3 跑完整对比 + 消融矩阵（论文 5.2 / 5.3 所需）

如果你要复现论文里的全部 6 个实验组（baseline / focal / focal_ohem / full / yolov5 / frcnn），约需 4 小时（RTX 4090）：

```bash
PYTHONPATH=. python scripts/run_matrix.py \
    --epochs 100 --batch 16 --workers 4 \
    --project runs/matrix
```

每个变体的 best.pt 会落到 `runs/matrix/<variant>/weights/best.pt`。

### 6.4 评估 + 鲁棒性测试 + ONNX 导出

```bash
# 在测试集上评估全部 variant，输出 metrics.csv
PYTHONPATH=. python scripts/eval_all.py --device cuda:0 --fps-iters 200

# 鲁棒性测试（运动模糊 / 高斯噪声）
PYTHONPATH=. python scripts/robust_eval.py \
    --weights runs/r2/baseline/weights/best.pt \
    --imgsz 960 --device cuda:0 \
    --out-root runs/r2/_robust

# ONNX 导出
PYTHONPATH=. python scripts/export_onnx.py \
    --weights runs/r2/baseline/weights/best.pt \
    --imgsz 960 --opset 12
```

### 6.5 视频推理 demo

```bash
PYTHONPATH=. python scripts/infer_video.py \
    --weights runs/r2/baseline/weights/best.pt \
    --video data/demo/videos/your_video.mp4 \
    --imgsz 960 --device cuda:0
```

输出会保存到 `runs/infer/your_video_pred.mp4`。

---

## 七、只想用我们训练好的模型推理

如果你不需要重新训练，只想拿权重出框，最简流程：

```bash
conda create -n coalbelt python=3.12 -y
conda activate coalbelt
pip install ultralytics opencv-python

# 假设你已经把 best.pt 放到 NetTrack/runs/r2/baseline/weights/ 下
python -c "
from ultralytics import YOLO
m = YOLO('runs/r2/baseline/weights/best.pt')
m.predict('CUMT-BelT/测试集/大块', save=True, imgsz=960, conf=0.25)
"
```

输出图片在 `runs/detect/predict/`。

---

## 八、ONNX 部署

得到 `best.onnx` 后，可以用 onnxruntime 直接推理：

```bash
pip install onnxruntime onnxruntime-gpu  # GPU 部署
```

```python
import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "runs/r2/baseline/weights/best.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
img = cv2.imread("test.jpg")
img = cv2.resize(img, (960, 960))
chw = img[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
out = session.run(None, {"images": chw[None]})
print([o.shape for o in out])
```

边缘设备（Jetson Orin AGX）部署可参考 [docs/final_report.md §5.9](docs/final_report.md)。

---

## 九、常见问题

### 9.1 `RuntimeError: CUDA out of memory`

降低 batch：`--batch 8`（YOLOv8m@960 大约需要 11 GB 显存）。

### 9.2 `ModuleNotFoundError: No module named 'src'`

执行命令时前面加 `PYTHONPATH=.`，比如：

```bash
PYTHONPATH=. python scripts/train.py ...
```

或在 `~/.bashrc` 里加 `export PYTHONPATH=$PYTHONPATH:.`。

### 9.3 `pip install` 卡住或超时

确认按第 3.3 节配了清华镜像。

### 9.4 训练 loss 全是 0

通常是 `data.yaml` 的 `path` 与实际目录对不上，导致 Ultralytics 找不到标签文件。检查：

```bash
cat data/cumt_belt_yolo/data.yaml
ls data/cumt_belt_yolo/labels/train | head
```

### 9.5 Windows 下 `ln -sf` 不可用

Windows 用 `mklink` 或者直接复制 `.pt` 文件即可：

```cmd
copy weights\cache\ultralytics\yolov8m.pt yolov8m.pt
```

### 9.6 不能访问 huggingface（GroundingDINO 需要 BERT）

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

放到 `~/.bashrc` 永久生效。

---

## 十、目录结构

```text
NetTrack/
├── CUMT-BelT/                     # 原始数据集（自行下载）
├── data/
│   └── cumt_belt_yolo/            # 处理后的 YOLO 格式数据集
├── docs/
│   ├── final_report.md            # 完整论文式报告
│   ├── results.md                 # 实验数字汇总
│   └── figures/                   # 论文用图
├── scripts/
│   ├── prefetch_weights.py        # 预下载所有预训练权重
│   ├── pseudo_label_gd.py         # 数据集标注脚本
│   ├── prepare_dataset.py         # train/val 9:1 划分
│   ├── train.py                   # 单 variant 训练入口
│   ├── run_matrix.py              # 6 组对比实验串行执行
│   ├── eval_all.py                # 全部 variant 测试集评估
│   ├── robust_eval.py             # 鲁棒性测试（模糊/噪声）
│   ├── export_onnx.py             # ONNX 导出 + ORT 验证
│   ├── infer_video.py             # 视频流推理 demo
│   ├── make_dataset_figs.py       # 生成数据集分析图
│   └── make_report_figs.py        # 生成论文对比图
├── src/
│   ├── losses/focal_loss.py       # SigmoidFocalLoss
│   ├── samplers/ohem.py           # OHEM 掩码
│   ├── samplers/weighted_sampler.py  # WeightedRandomSampler
│   ├── models/yolov8_imp.py       # 三件套注入 Ultralytics
│   └── training/frcnn_runner.py   # Faster R-CNN 训练循环
├── weights/                       # 预训练权重缓存
├── runs/                          # 训练 / 评估输出
├── pyproject.toml                 # 项目依赖声明
└── README.md                      # 本文件
```

---

## 📄 引用与致谢

数据集：

```bibtex
@misc{cumt-belt,
  title={CUMT-BelT: 中国矿业大学输送带异物图像数据集},
  author={CUMT-AIPR-Lab},
  year={2024},
  url={https://github.com/CUMT-AIPR-Lab/CUMT-AIPR-Lab}
}
```

依赖项：

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/) / [torchvision](https://pytorch.org/vision/)
- [ONNX Runtime](https://onnxruntime.ai/)

---

## 📜 License

MIT。详见 [`LICENSE`](LICENSE)。
