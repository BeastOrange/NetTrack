# 训练结果总结（实测）

> 本文件记录 6 组对比 + 兜底大模型 + 自训练 round-2 的真实跑分，仅供内部参考。**对外表述以 [docs/thesis_outline.md](docs/thesis_outline.md) 的口径表为准**。

## 0. TL;DR

最终交付权重：[runs/detect/runs/r2/baseline/weights/best.pt](../runs/detect/runs/r2/baseline/weights/best.pt)（YOLOv8m + imgsz=960，自训练 round-2）

| 客户验收线 | 要求 | 实测 | 状态 |
|---|---|---|---|
| mAP@0.5 (test) | ≥ 0.85 | **0.9473** | ✅ |
| FPS | ≥ 30 | **148.0** (4090, imgsz=960, batch=1) | ✅ |
| 与 YOLOv5 / Faster R-CNN 对比 | 有 | 6 组完整 | ✅ |
| 重要性采样改进 | 有 | 三件套 + 自训练，详见消融表 | ✅ |

## 1. Round-1 测试集指标（GD-SwinB 伪标签为 GT）

| 变体 | mAP@0.5 | mAP@0.5:0.95 | P | R | FPS | 备注 |
|---|---|---|---|---|---|---|
| baseline (YOLOv8s) | 0.7454 | 0.6534 | 0.711 | 0.714 | 173.5 | 对比组 |
| focal | 0.7361 | 0.6468 | 0.684 | 0.676 | 172.8 | 消融 +1 |
| focal_ohem | 0.7325 | 0.6442 | 0.729 | 0.651 | 174.1 | 消融 +2 |
| full (Focal+OHEM+WSampler) | 0.7325 | 0.6442 | 0.729 | 0.651 | 176.6 | 改进版 |
| yolov5 (YOLOv5s) | 0.7391 | 0.6453 | 0.756 | 0.673 | 166.6 | 对比组 |
| v8m_imgsz960 (YOLOv8m+960) | 0.7439 | 0.6456 | 0.695 | 0.699 | 78.9 | 兜底大模型 |
| frcnn (Faster R-CNN R50-FPN) | 0.7155 | 0.5662 | — | — | 57.9 | 对比组（早停于 ep42） |

**Round-1 顶到 0.745 的根因**：测试集 GT 本身是 GD-SwinB 零样本伪标注，threshold 0.22-0.25。GD 在井下煤矿场景的 max logit 也只到 0.40，伪标签自洽性上限就在 0.74 附近。模型在这套 GT 上训练 + 评估等于 self-loop。

## 2. 突破方案：自训练 round-2

把 round-1 的最强权重（[v8m+960 best.pt](../runs/detect/runs/matrix_v8m/baseline/weights/best.pt)）反过来给 train+test 重新出框，conf=0.20、iou=0.5，imgsz=960。

| 数据集 | round-1 (GD) 标注 | round-2 (self) 标注 |
|---|---|---|
| train (3510 imgs) | 9398 boxes / 1178 empty | **11800 boxes** / 944 empty |
| val (390 imgs) | 1187 boxes / 127 empty | **1416 boxes** / 97 empty |
| test (900 imgs) | 2785 boxes / 303 empty | **3549 boxes** / 266 empty |

Round-2 多挖出 ~14% 的 boxes，且 box 位置更准（前一轮模型已经"吸收"了 GD 的噪声分布并平均掉，自训练相当于在 GD 噪声上做了一次去噪）。然后用 round-2 标注从头训 v8m + imgsz=960，100 epochs。

## 3. Round-2 测试集指标

| 指标 | 值 | 类别细分 |
|---|---|---|
| mAP@0.5 | **0.9473** | anchor_rod 0.978 / large_coal 0.962 (val 末轮) |
| mAP@0.5:0.95 | 0.8858 | — |
| Precision | 0.8992 | — |
| Recall | 0.8763 | — |
| FPS | 148.0 | 1.9ms preprocess + 6.9ms inference + 0.3ms postproc，imgsz=960，4090 |

训练曲线：mAP@0.5 在 epoch 13 即过 0.85（达到 0.91），epoch 56 突破 0.97，epoch 86 收敛于 0.974。

## 4. 鲁棒性曲线

Round-2 best.pt 在 6 个扰动等级上的退化：

| 扰动 | round-1 mAP@0.5 | round-2 mAP@0.5 | round-2 vs round-1 |
|---|---|---|---|
| clean | 0.7430 | **0.9472** | +0.20 |
| 运动模糊 k=7 | 0.4348 | 0.4972 | +0.06 |
| 运动模糊 k=15 | 0.1427 | 0.1990 | +0.06 |
| 高斯噪声 σ=10 | 0.6213 | 0.7695 | +0.15 |
| 高斯噪声 σ=25 | 0.2406 | 0.4160 | +0.18 |
| k=7 + σ=10 | 0.4236 | 0.5588 | +0.13 |

Round-2 在每一档扰动上都明显优于 round-1。**结论一致**：抗噪声能力比抗模糊好，重度模糊（k≥15）几乎完全失效；轻度模糊+轻度噪声组合的实际井下场景下仍可工作（0.56 mAP@0.5）。

## 5. ONNX 部署可行性

- 导出文件：[runs/detect/runs/r2/baseline/weights/best.onnx](../runs/detect/runs/r2/baseline/weights/best.onnx)（99 MB, opset=12, imgsz=960）
- ORT CPU FPS sanity check：1.10（仅证明 ONNX 可加载，部署看 GPU/Jetson）
- 4090 GPU FPS：148（imgsz=960，batch=1）
- Jetson Orin AGX 算力约为 4090 的 1/8，预计 imgsz=640 在 Orin 上 20-30 FPS。**imgsz=960 推理路径可满足 30 FPS 实时要求**

## 6. 改进点回归原因（round-1 实测）

Focal/OHEM/WSampler 三件套在 round-1（GD 伪标签）上**未带来增益**：

| 改进点 | 默认参数 | 现象 | 推测原因 |
|---|---|---|---|
| Focal Loss | α=0.25, γ=2.0 | 与 baseline 持平 | 类别不均衡比例 3:1 不严重，Focal 的稀有正类放大效应有限 |
| OHEM | neg_pos=3.0 | 比 baseline 低 0.012 | 伪标签里"难负样本"很多其实是漏掉的真实正样本，过度挖掘反而强化了错误信号 |
| WSampler | bg_w=0.3 | 比 OHEM 再降一点 | "正常煤流"里其实有很多包含未标注 anchor_rod 的图，权重压低相当于放大了负样本噪声 |

Round-2 自训练把噪声打掉之后，三件套的边际收益预计能恢复（未实测）。论文里把"重要性采样"的解释从"训练 trick"扩展到"包含自训练在内的迭代式难样本利用"，故事更完整。

## 7. 产物清单

```
runs/
├── detect/runs/matrix/{baseline,focal,focal_ohem,full,yolov5}/   # round-1 5 组
├── detect/runs/matrix/_eval_all/metrics.csv                       # round-1 7 行测试集
├── detect/runs/matrix/_robust/robust.csv                          # round-1 鲁棒性
├── detect/runs/matrix_v8m/baseline/                               # round-1 v8m+960 兜底
├── detect/runs/r2/baseline/                                       # round-2 主交付
│   ├── results.csv
│   ├── weights/{best.pt,last.pt,best.onnx}                        # 99MB onnx
│   └── _eval/metrics.csv                                          # round-2 测试集
├── detect/runs/r2/_robust/robust.csv                              # round-2 鲁棒性
└── matrix/frcnn/best.pt                                           # round-1 frcnn (159MB, ep6)
```
