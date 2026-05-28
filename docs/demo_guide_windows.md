# 模型演示指南（Windows 版）

> 给评阅老师看的时候，按下面步骤操作即可。

---

## 前置准备（只需做一次）

### 1. 安装 Miniconda

下载：https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

双击安装，全部默认。装完后从开始菜单打开 **Anaconda Prompt**。

### 2. 创建环境 + 装依赖

在 Anaconda Prompt 中执行：

```cmd
conda create -n coalbelt python=3.12 -y
conda activate coalbelt

:: 方式 A：只跑演示（最快，装两个包就够）
pip install ultralytics opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 方式 B：完整安装（如果还要跑训练/评估脚本）
cd C:\Users\你的用户名\Desktop\NetTrack
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 把项目文件夹放到电脑上

确保目录结构如下（关键文件）：

```
NetTrack\
├── weights\
│   └── best.pt              ← 训练好的模型权重（50MB）
├── scripts\
│   └── demo.py              ← 演示脚本
└── CUMT-BelT\               ← 数据集（可选，用于批量演示）
    ├── 训练集\
    │   ├── 大块\
    │   ├── 锚杆\
    │   └── 正常煤流\
    └── 测试集\
        ├── 大块\
        ├── 锚杆\
        └── 正常煤流\
```

---

## 演示操作

打开 Anaconda Prompt，先进入项目目录并激活环境：

```cmd
cd C:\Users\你的用户名\Desktop\NetTrack
conda activate coalbelt
```

### 演示 1：单张图片检测（弹窗显示）

```cmd
python scripts\demo.py --image CUMT-BelT\测试集\锚杆\任意一张.jpg
```

会弹出窗口显示检测结果，按任意键关闭。

### 演示 2：锚杆图片批量检测（生成网格图）

```cmd
python scripts\demo.py --image-dir CUMT-BelT\测试集\锚杆 --topk 16 --save runs\demo\grid_anchor_rod.jpg
```

### 演示 3：大块煤图片批量检测

```cmd
python scripts\demo.py --image-dir CUMT-BelT\测试集\大块 --topk 16 --save runs\demo\grid_large_coal.jpg
```

### 演示 4：视频检测

```cmd
python scripts\demo.py --video 你的视频.mp4 --save runs\demo\output.mp4
```

### 演示 5：摄像头实时检测（如果有摄像头）

```cmd
python scripts\demo.py --camera 0
```

按 `q` 键退出。

---

## 常见问题

**Q: 提示 `ModuleNotFoundError: No module named 'ultralytics'`**

A: 确认已激活环境：`conda activate coalbelt`

**Q: 弹窗显示不出来 / 闪退**

A: 加 `--save result.jpg` 参数，结果会保存为图片文件，用图片查看器打开即可。

**Q: 没有 GPU，能跑吗？**

A: 能跑，脚本会自动用 CPU，只是稍慢（单张图约 1-2 秒）。

**Q: 想调整检测灵敏度？**

A: 加 `--conf 0.25`（更灵敏）或 `--conf 0.5`（更严格）。

---

## 给老师看的要点

1. 模型能检测两类目标：**锚杆**（anchor_rod）和 **大块煤**（large_coal）
2. 测试集 mAP@0.5 = 0.947，FPS > 30
3. 改进方法：Focal Loss + OHEM + WeightedRandomSampler（重要性采样策略）
4. 对运动模糊和噪声有一定鲁棒性
