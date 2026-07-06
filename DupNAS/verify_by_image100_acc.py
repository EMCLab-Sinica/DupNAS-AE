# python3.9 verify_by_image100_acc.py --limit 200 --batch-size 32 --provider cpu
# python3.9 verify_by_image100_acc.py --batch-size 32 --provider cpu
# python3.9 verify_by_image100_acc.py --batch-size 64 --provider cuda

#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "onnxruntime is required. Install by: pip install onnxruntime or onnxruntime-gpu"
    ) from e


#DEFAULT_ROOT = Path("/4TB/aeuser/DupNAS-AE/DupNAS")
DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_VAL_DIR = DEFAULT_ROOT / "NASBase/dataset/IMAGE100/val"

MODEL_DIRS = [
    DEFAULT_ROOT / "genonnx/shuffle",
    DEFAULT_ROOT / "genonnx/mbv2",
    DEFAULT_ROOT / "genonnx/incept",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_ir_from_name(path: Path):
    """
    Example:
      shuffle-im100-vm96-dupnas_w0.5_ir128.onnx -> 128
    """
    m = re.search(r"_ir(\d+)", path.stem)
    return int(m.group(1)) if m else None


def get_input_info(session, onnx_path: Path):
    inp = session.get_inputs()[0]
    name = inp.name
    shape = inp.shape

    # Try to infer NCHW / NHWC and input resolution.
    # Common: [1, 3, 128, 128]
    # Dynamic can be ['batch', 3, 'height', 'width']
    layout = "NCHW"
    input_size = None

    if len(shape) == 4:
        s = shape

        # NCHW
        if s[1] == 3:
            layout = "NCHW"
            if isinstance(s[2], int):
                input_size = s[2]

        # NHWC
        elif s[3] == 3:
            layout = "NHWC"
            if isinstance(s[1], int):
                input_size = s[1]

    if input_size is None:
        input_size = parse_ir_from_name(onnx_path)

    if input_size is None:
        raise ValueError(
            f"Cannot infer input resolution for {onnx_path.name}. "
            f"ONNX input shape is {shape}. Please pass --input-size."
        )

    return name, shape, layout, input_size


def build_image_list(val_dir: Path):
    """
    Expects ImageFolder-style validation directory:

      val/
        class_0/
          xxx.JPEG
        class_1/
          yyy.JPEG

    Label index follows sorted class folder order.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    class_dirs = sorted([p for p in val_dir.iterdir() if p.is_dir()])

    if not class_dirs:
        raise RuntimeError(f"No class folders found under {val_dir}")

    samples = []
    class_to_idx = {}

    for idx, class_dir in enumerate(class_dirs):
        class_to_idx[class_dir.name] = idx
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.suffix.lower() in exts:
                samples.append((img_path, idx))

    if not samples:
        raise RuntimeError(f"No validation images found under {val_dir}")

    return samples, class_to_idx


def resize_shorter_side(img: Image.Image, size: int):
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(round(h * size / w))
    else:
        new_h = size
        new_w = int(round(w * size / h))
    return img.resize((new_w, new_h), Image.BILINEAR)


def center_crop(img: Image.Image, crop_size: int):
    w, h = img.size
    left = int(round((w - crop_size) / 2.0))
    top = int(round((h - crop_size) / 2.0))
    return img.crop((left, top, left + crop_size, top + crop_size))


def preprocess_image(img_path: Path, input_size: int, layout: str):
    img = Image.open(img_path).convert("RGB")

    # Standard ImageNet validation preprocessing:
    # resize shorter side to input_size / 0.875, then center crop.
    resize_size = int(round(input_size / 0.875))
    img = resize_shorter_side(img, resize_size)
    img = center_crop(img, input_size)

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    if layout == "NCHW":
        arr = np.transpose(arr, (2, 0, 1))

    return arr.astype(np.float32)


def softmax_or_logits_topk(output, topk=5):
    """
    ONNX output is usually logits. Top-k is the same before/after softmax.
    """
    output = np.asarray(output)

    if output.ndim > 2:
        output = output.reshape(output.shape[0], -1)

    k = min(topk, output.shape[1])
    topk_idx = np.argpartition(-output, kth=k - 1, axis=1)[:, :k]

    # Sort the selected top-k indices by score.
    row_indices = np.arange(output.shape[0])[:, None]
    topk_scores = output[row_indices, topk_idx]
    order = np.argsort(-topk_scores, axis=1)
    topk_idx = topk_idx[row_indices, order]

    return topk_idx


def evaluate_onnx(
    onnx_path: Path,
    samples,
    batch_size: int,
    provider: str,
    forced_input_size=None,
):
    providers = []

    available = ort.get_available_providers()
    if provider == "cuda" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(onnx_path), providers=providers)

    input_name, input_shape, layout, inferred_size = get_input_info(session, onnx_path)
    input_size = forced_input_size if forced_input_size is not None else inferred_size

    output_names = [o.name for o in session.get_outputs()]

    total = 0
    top1_correct = 0
    top5_correct = 0

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]

        images = [
            preprocess_image(img_path, input_size=input_size, layout=layout)
            for img_path, _ in batch
        ]
        labels = np.array([label for _, label in batch], dtype=np.int64)

        x = np.stack(images, axis=0).astype(np.float32)

        outputs = session.run(output_names, {input_name: x})
        logits = outputs[0]

        topk_idx = softmax_or_logits_topk(logits, topk=5)

        pred1 = topk_idx[:, 0]
        top1_correct += int(np.sum(pred1 == labels))

        for i, label in enumerate(labels):
            if label in topk_idx[i]:
                top5_correct += 1

        total += len(batch)

    top1 = top1_correct / total * 100.0
    top5 = top5_correct / total * 100.0

    return {
        "onnx": str(onnx_path),
        "model_name": onnx_path.name,
        "input_shape": str(input_shape),
        "layout": layout,
        "input_size": input_size,
        "num_images": total,
        "top1_acc": top1,
        "top5_acc": top5,
        "provider": providers[0],
    }


def collect_onnx_files(model_dirs):
    files = []
    for d in model_dirs:
        if not d.exists():
            print(f"[WARN] Missing model dir: {d}")
            continue
        files.extend(sorted(d.glob("*.onnx")))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=DEFAULT_VAL_DIR,
        help="Image100 validation folder.",
    )
    parser.add_argument(
        "--onnx-dirs",
        type=Path,
        nargs="*",
        default=MODEL_DIRS,
        help="Folders containing ONNX models.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "onnx_image100_accuracy.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--provider",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Use cuda if onnxruntime-gpu is installed.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Force input image size. If not set, infer from ONNX shape or filename _irXX.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick test, e.g., --limit 200.",
    )
    args = parser.parse_args()

    print(f"[INFO] Validation dir: {args.val_dir}")
    samples, class_to_idx = build_image_list(args.val_dir)

    if args.limit is not None:
        samples = samples[: args.limit]
        print(f"[INFO] Limit enabled: using first {len(samples)} images")

    print(f"[INFO] Number of classes: {len(class_to_idx)}")
    print(f"[INFO] Number of validation images: {len(samples)}")

    onnx_files = collect_onnx_files(args.onnx_dirs)
    if not onnx_files:
        raise RuntimeError("No ONNX files found.")

    print(f"[INFO] Number of ONNX models: {len(onnx_files)}")
    print(f"[INFO] Output CSV: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_name",
        "onnx",
        "input_shape",
        "layout",
        "input_size",
        "num_images",
        "top1_acc",
        "top5_acc",
        "provider",
    ]

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, onnx_path in enumerate(onnx_files, start=1):
            print("=" * 80)
            print(f"[{i}/{len(onnx_files)}] Evaluating: {onnx_path}")

            try:
                result = evaluate_onnx(
                    onnx_path=onnx_path,
                    samples=samples,
                    batch_size=args.batch_size,
                    provider=args.provider,
                    forced_input_size=args.input_size,
                )

                writer.writerow(result)
                f.flush()

                print(
                    f"[RESULT] {onnx_path.name}: "
                    f"Top-1={result['top1_acc']:.2f}%, "
                    f"Top-5={result['top5_acc']:.2f}%, "
                    f"input={result['input_size']}, "
                    f"layout={result['layout']}"
                )

            except Exception as e:
                print(f"[ERROR] Failed on {onnx_path.name}: {e}")
                writer.writerow(
                    {
                        "model_name": onnx_path.name,
                        "onnx": str(onnx_path),
                        "input_shape": "ERROR",
                        "layout": "ERROR",
                        "input_size": args.input_size if args.input_size else "ERROR",
                        "num_images": len(samples),
                        "top1_acc": "ERROR",
                        "top5_acc": "ERROR",
                        "provider": args.provider,
                    }
                )
                f.flush()

    print("=" * 80)
    print(f"[DONE] Saved accuracy results to: {args.output}")


if __name__ == "__main__":
    main()