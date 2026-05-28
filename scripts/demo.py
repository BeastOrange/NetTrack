"""
一键演示脚本：煤矿传送带异物检测模型

功能：
  1. 对单张图片进行检测并弹窗显示结果
  2. 对文件夹中的图片批量检测，生成网格图
  3. 对视频进行实时检测

使用方法：
  # 单张图片检测（弹窗显示）
  python scripts/demo.py --image path/to/image.jpg

  # 文件夹批量检测（生成网格图）
  python scripts/demo.py --image-dir path/to/images/ --topk 16

  # 视频检测
  python scripts/demo.py --video path/to/video.mp4

  # 摄像头实时检测
  python scripts/demo.py --camera 0

可选参数：
  --weights   模型权重路径（默认 weights/best.pt）
  --imgsz     推理分辨率（默认 960）
  --conf      置信度阈值（默认 0.3）
  --device    推理设备（默认自动选择 cuda/cpu）
  --save      保存结果到指定路径
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ["anchor_rod", "large_coal"]
CLASS_NAMES_CN = ["锚杆", "大块煤"]
COLORS = [(0, 200, 255), (255, 180, 0)]  # BGR


def draw_detections(image: np.ndarray, results) -> np.ndarray:
    out = image.copy()
    if results.boxes is None or len(results.boxes) == 0:
        return out
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES_CN[cls_id]}({CLASS_NAMES[cls_id]}) {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def demo_single_image(model, img_path: Path, args):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] 无法读取图片: {img_path}")
        return
    results = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
    annotated = draw_detections(img, results)

    n_boxes = len(results.boxes) if results.boxes is not None else 0
    print(f"[INFO] 检测到 {n_boxes} 个目标")
    for box in (results.boxes or []):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"       - {CLASS_NAMES_CN[cls_id]} (conf={conf:.3f})")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotated)
        print(f"[INFO] 结果已保存: {save_path}")
    else:
        cv2.imshow("Detection Result", annotated)
        print("[INFO] 按任意键关闭窗口")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def demo_image_dir(model, img_dir: Path, args):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in extensions)
    if not images:
        print(f"[ERROR] 文件夹中没有图片: {img_dir}")
        return

    topk = min(args.topk, len(images))
    import random
    random.seed(42)
    selected = random.sample(images, topk)

    cols = 4
    rows = (topk + cols - 1) // cols
    cell_size = 416
    canvas = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

    for idx, img_path in enumerate(selected):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        oh, ow = img.shape[:2]
        results = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        img_resized = cv2.resize(img, (cell_size, cell_size))

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 = int(x1 / ow * cell_size)
                y1 = int(y1 / oh * cell_size)
                x2 = int(x2 / ow * cell_size)
                y2 = int(y2 / oh * cell_size)
                color = COLORS[cls_id % len(COLORS)]
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 2)
                label = f"{CLASS_NAMES[cls_id]} {conf:.1f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(img_resized, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img_resized, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        r_idx = idx // cols
        c_idx = idx % cols
        canvas[r_idx * cell_size:(r_idx + 1) * cell_size,
               c_idx * cell_size:(c_idx + 1) * cell_size] = img_resized

    save_path = Path(args.save) if args.save else Path("runs/demo/grid_result.jpg")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[INFO] 网格图已保存: {save_path} ({rows}x{cols}, {topk} 张图)")


def demo_video(model, video_source, args):
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source)
        print(f"[INFO] 打开摄像头 {video_source}")
    else:
        cap = cv2.VideoCapture(str(video_source))
        print(f"[INFO] 打开视频: {video_source}")

    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频源")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
        print(f"[INFO] 输出视频: {save_path}")

    print("[INFO] 按 'q' 退出")
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf,
                                verbose=False)[0]
        annotated = draw_detections(frame, results)
        frame_count += 1

        if writer:
            writer.write(annotated)
        else:
            cv2.imshow("Coal Belt Detection (press q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
        print(f"[INFO] 视频保存完成，共 {frame_count} 帧")
    else:
        cv2.destroyAllWindows()
    print(f"[INFO] 处理完成，共 {frame_count} 帧")


def parse_args():
    p = argparse.ArgumentParser(
        description="煤矿传送带异物检测 - 演示脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--weights", type=str, default="weights/best.pt",
                   help="模型权重路径 (default: weights/best.pt)")
    p.add_argument("--imgsz", type=int, default=960,
                   help="推理分辨率 (default: 960)")
    p.add_argument("--conf", type=float, default=0.3,
                   help="置信度阈值 (default: 0.3)")
    p.add_argument("--device", type=str, default=None,
                   help="推理设备, e.g. cuda:0 / cpu (default: auto)")
    p.add_argument("--save", type=str, default=None,
                   help="保存结果路径 (不指定则弹窗显示)")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="单张图片路径")
    group.add_argument("--image-dir", type=str, help="图片文件夹路径")
    group.add_argument("--video", type=str, help="视频文件路径")
    group.add_argument("--camera", type=int, nargs="?", const=0,
                       help="摄像头编号 (default: 0)")

    p.add_argument("--topk", type=int, default=16,
                   help="网格图最多显示几张 (default: 16)")
    return p.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] 权重文件不存在: {weights_path}")
        print("        请确保 weights/best.pt 在项目根目录下")
        sys.exit(1)

    device = args.device
    if device is None:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 加载模型: {weights_path} (device={device})")
    model = YOLO(str(weights_path))

    if args.image:
        demo_single_image(model, Path(args.image), args)
    elif args.image_dir:
        demo_image_dir(model, Path(args.image_dir), args)
    elif args.video:
        demo_video(model, Path(args.video), args)
    elif args.camera is not None:
        demo_video(model, args.camera, args)


if __name__ == "__main__":
    main()
