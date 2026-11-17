# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Compare objectness vs baseline runs: accuracy, per-class AP, TP confidence distributions and confusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import yaml_load


class StatsValidator(DetectionValidator):
    """Detection validator that keeps the raw per-prediction match statistics for offline analysis."""

    def get_stats(self) -> dict:
        """Snapshot concatenated raw stats before the metrics pipeline clears them.

        Returns:
            (dict): Metrics results dict from the standard pipeline.
        """
        self.raw_stats = {k: np.concatenate(v, 0) for k, v in self.metrics.stats.items() if v}
        return super().get_stats()


def collect(run_dir: Path, out_dir: Path) -> dict:
    """Re-run validation for one training run and collect accuracy, score and confusion statistics.

    Args:
        run_dir (Path): Training run directory containing args.yaml and weights/best.pt.
        out_dir (Path): Directory for validation artifacts.

    Returns:
        (dict): Collected statistics including per-class AP, TP confidences and the confusion matrix.
    """
    model = YOLO(run_dir / "weights" / "best.pt")
    data = yaml_load(run_dir / "args.yaml")["data"]
    model.val(data=data, imgsz=640, batch=16, plots=True, save_json=True, verbose=True, project=out_dir, name="val")
    v, m = model.validator, model.metrics
    raw = v.raw_stats
    tp50 = raw["tp"][:, 0]  # IoU 0.5 correct-class matches, one entry per prediction
    names = [v.names[i] for i in m.ap_class_index]
    return {
        "names": names,
        "map50_95": m.mean_results()[3],
        "map50": m.mean_results()[2],
        "precision": m.mean_results()[0],
        "recall": m.mean_results()[1],
        "per_class": [m.class_result(i) for i in range(len(names))],  # (p, r, ap50, ap) per class
        "conf": raw["conf"],
        "tp50": tp50,
        "gt_count": len(raw["target_cls"]),
        "recall_at": {t: float((tp50 & (raw["conf"] >= t)).sum() / max(len(raw["target_cls"]), 1)) for t in (0.25, 0.5)},
        "confusion": m.confusion_matrix.matrix.copy(),
    }


def write_report(dataset: str, results: dict[str, dict], out_dir: Path) -> None:
    """Write the markdown comparison report and TP-confidence histogram.

    Args:
        dataset (str): Dataset name for the report title and histogram file.
        results (dict[str, dict]): Collected statistics keyed by arm name.
        out_dir (Path): Output directory for report.md and tp_conf_hist.png.
    """
    lines = [f"# Objectness A/B — {dataset}", "", "## Summary", ""]
    lines.append("| arm | mAP50-95 | mAP50 | precision | recall | mean TP conf | median TP conf | recall@0.25 | recall@0.5 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for arm, r in results.items():
        tp_conf = r["conf"][r["tp50"]]
        lines.append(
            f"| {arm} | {r['map50_95']:.4f} | {r['map50']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{tp_conf.mean():.4f} | {np.median(tp_conf):.4f} | {r['recall_at'][0.25]:.4f} | {r['recall_at'][0.5]:.4f} |"
        )

    names = next(iter(results.values()))["names"]
    lines += ["", "## Per-class AP50-95", "", "| class | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    for i, name in enumerate(names):
        lines.append(f"| {name} | " + " | ".join(f"{r['per_class'][i][3]:.4f}" for r in results.values()) + " |")

    for arm, r in results.items():
        cm = r["confusion"][:-1, :-1]  # drop background row/col
        pairs = [(cm[i, j], names[i], names[j]) for i in range(len(names)) for j in range(len(names)) if i != j]
        pairs.sort(reverse=True)
        lines += ["", f"## Inter-class confusion — {arm} (top pairs, gt → pred)", ""]
        lines += [f"- {a} → {b}: {int(n)}" for n, a, b in pairs[:5] if n > 0]

    lines += ["", "## Files", "", "- `tp_conf_hist.png` — TP confidence histograms at IoU 0.5"]
    lines += [f"- `{arm}-val/` — validation artifacts (confusion matrices, predictions.json)" for arm in results]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 41)
    for arm, r in results.items():
        tp_conf = r["conf"][r["tp50"]]
        ax.hist(tp_conf, bins=bins, alpha=0.5, label=f"{arm} (mean {tp_conf.mean():.3f})", density=True)
    ax.set(xlabel="TP confidence (IoU 0.5)", ylabel="density", title=f"True-positive confidence — {dataset}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tp_conf_hist.png", dpi=150)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        (argparse.Namespace): Parsed arguments with run pairs and output directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, metavar="ARM=DIR", help="arm name and training run dir")
    parser.add_argument("--out", default="analysis", help="output directory")
    return parser.parse_args()


def main() -> None:
    """Collect per-run statistics and write the comparison report."""
    args = parse_args()
    results = {}
    run_name = ""
    for spec in args.runs:
        arm, _, path = spec.partition("=")
        out_dir = Path(args.out) / f"{arm}-val"
        out_dir.mkdir(parents=True, exist_ok=True)
        results[arm] = collect(Path(path), out_dir)
        run_name = run_name or yaml_load(Path(path) / "args.yaml")["name"]
    dataset = run_name.rsplit("-", 1)[0]  # run names follow "<dataset>-<arm>"
    out_dir = Path(args.out)
    write_report(dataset, results, out_dir)
    print(f"report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
