# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train yolo26n with or without the YOLOv5-style objectness branch on an Ultralytics Benchmark dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from objexp import ObjectnessTrainer

DATASETS = {
    "coco8": "coco8.yaml",
    "deeppcb": "ul://ultralytics/datasets/deeppcb",
    "dspcbsd": "ul://ultralytics/datasets/dspcbsd-pcb-surface-defect",
    "tn5000": "ul://ultralytics/datasets/tn5000-thyroid-us",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        (argparse.Namespace): Parsed arguments with arm, data, model and training overrides.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "objectness"), required=True, help="experiment arm")
    parser.add_argument("--data", required=True, help="dataset key from DATASETS or a data config/URI")
    parser.add_argument("--model", default="yolo26n.pt", help="pretrained weights or model YAML")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs (benchmark protocol: 100)")
    parser.add_argument("--batch", type=int, default=16, help="batch size (benchmark protocol: 16)")
    parser.add_argument("--project", default="objectness", help="save directory parent under runs/detect")
    return parser.parse_args()


def main() -> None:
    """Launch one training run of the selected arm with the benchmark protocol."""
    args = parse_args()
    data = DATASETS.get(args.data, args.data)
    data_key = (Path(data).parent.name or Path(data).stem) if data.endswith((".yaml", ".yml")) else args.data
    name = f"{data_key}-{args.arm}"
    overrides = dict(
        data=data,
        epochs=args.epochs,
        patience=100,
        imgsz=640,
        batch=args.batch,
        seed=0,
        workers=4,
        cls_remap=True,
        channels_last=True,
        save_json=True,
        plots=True,
        fraction=[1000, 1.0, 0.0],  # benchmark protocol: train on up to 1000 images, full val, no test
        project=args.project,
        name=name,
    )
    model = YOLO(args.model)
    if args.arm == "objectness":
        model.train(trainer=ObjectnessTrainer, **overrides)
    else:
        model.train(**overrides)


if __name__ == "__main__":
    main()
