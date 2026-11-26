"""TensorBoard logging wrapper around `torch.utils.tensorboard.SummaryWriter`."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter

from src.validate_labels import BBox
from src.viz import CLASS_COLORS, ascii_safe


def render_gt_vs_pred(
    image_path: Path,
    gt_boxes: list[BBox],
    pred_boxes: list[tuple[int, float, float, float, float, float]],
    class_names: list[str],
    max_width: int = 1280,
) -> np.ndarray:
    """Render side-by-side Ground Truth (left) vs Predictions (right) RGB array."""
    with Image.open(image_path) as im:
        base = im.convert("RGB")

    w, h = base.size
    line_w = max(2, min(4, w // 350))

    def _draw_panel(boxes_with_conf: list[tuple[int, float, float, float, float, float | None]], header: str) -> Image.Image:
        canvas = base.copy()
        draw = ImageDraw.Draw(canvas)
        for cid, xc, yc, bw, bh, conf in boxes_with_conf:
            color = CLASS_COLORS[cid % len(CLASS_COLORS)]
            x1 = max(0.0, (xc - bw / 2.0) * w)
            y1 = max(0.0, (yc - bh / 2.0) * h)
            x2 = min(float(w - 1), (xc + bw / 2.0) * w)
            y2 = min(float(h - 1), (yc + bh / 2.0) * h)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
            cname = class_names[cid] if 0 <= cid < len(class_names) else f"cls_{cid}"
            txt = f"{cname} {conf:.2f}" if conf is not None else cname
            ty = max(22.0, y1 - 13.0)
            draw.rectangle([x1, ty, min(float(w), x1 + len(txt) * 6 + 6), ty + 13], fill=color)
            draw.text((x1 + 3, ty + 1), txt, fill=(0, 0, 0))
        draw.rectangle([0, 0, w, 22], fill=(18, 24, 38))
        draw.text((8, 5), ascii_safe(header), fill=(255, 255, 255))
        return canvas

    gt_tuples = [(b.class_id, b.x_center, b.y_center, b.width, b.height, None) for b in gt_boxes]
    pred_tuples = [(cid, xc, yc, bw, bh, conf) for cid, xc, yc, bw, bh, conf in pred_boxes]

    left = _draw_panel(gt_tuples, f"GT ({len(gt_tuples)}) | {image_path.name}")
    right = _draw_panel(pred_tuples, f"PRED ({len(pred_tuples)}) | {image_path.name}")

    half_w = max(160, max_width // 2)
    if left.width > half_w:
        new_h = max(1, int(round(left.height * (half_w / left.width))))
        left = left.resize((half_w, new_h), Image.Resampling.LANCZOS)
        right = right.resize((half_w, new_h), Image.Resampling.LANCZOS)

    panel = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
    panel.paste(left, (0, 0))
    panel.paste(right, (left.width, 0))
    return np.asarray(panel, dtype=np.uint8)


class TensorBoardLogger:
    """Local TensorBoard logger wrapping `SummaryWriter`."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def __enter__(self) -> TensorBoardLogger:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def log_run_metadata(
        self,
        config: dict[str, Any],
        git_info: dict[str, Any],
        dataset_info: dict[str, Any],
        env_info: dict[str, Any],
        mode: str,
    ) -> None:
        """Log run config, git version, dataset metadata, and environment as Markdown text."""
        cfg_md = "```json\n" + json.dumps({"mode": mode, **config}, indent=2, ensure_ascii=False) + "\n```"
        env_md = "```json\n" + json.dumps({"git": git_info, "env": env_info}, indent=2, ensure_ascii=False) + "\n```"
        ds_md = "```json\n" + json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n```"
        self.writer.add_text("run/config", cfg_md, global_step=0)
        self.writer.add_text("run/environment", env_md, global_step=0)
        self.writer.add_text("run/dataset", ds_md, global_step=0)

    def log_epoch_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log finite scalar metrics for a single epoch."""
        for tag, value in sorted(metrics.items()):
            if value is None:
                continue
            fval = float(value)
            if math.isfinite(fval):
                self.writer.add_scalar(tag, fval, global_step=epoch)

    def log_confusion_matrix(
        self,
        matrix: np.ndarray,
        class_names: list[str],
        step: int,
    ) -> None:
        """Render and log the detection confusion matrix as a TensorBoard image."""
        arr = np.asarray(matrix, dtype=np.float64)
        n_rows, n_cols = arr.shape
        if n_rows == len(class_names) + 1:
            labels = [*class_names, "background"]
        elif n_rows == len(class_names):
            labels = list(class_names)
        else:
            labels = [f"cls_{i}" for i in range(n_rows)]

        col_labels = labels[:n_cols]
        fig_size = max(7.0, 0.65 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85), dpi=130)
        im = ax.imshow(arr, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(n_cols),
            yticks=np.arange(n_rows),
            xticklabels=col_labels,
            yticklabels=labels[:n_rows],
            ylabel="Predicted",
            xlabel="True",
            title=f"Confusion Matrix (step {step})",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = arr.max() / 2.0 if arr.size and arr.max() > 0 else 0.5
        for i in range(n_rows):
            for j in range(n_cols):
                val = arr[i, j]
                if val > 0:
                    ax.text(
                        j,
                        i,
                        f"{int(round(val))}" if abs(val - round(val)) < 1e-3 else f"{val:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if val > thresh else "black",
                    )

        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[..., :3].copy()
        plt.close(fig)
        self.writer.add_image("eval/confusion_matrix", rgb, global_step=step, dataformats="HWC")

    def log_prediction_samples(
        self,
        samples: list[tuple[str, np.ndarray]],
        step: int,
    ) -> None:
        """Log GT-vs-Pred RGB sample panels under `predictions/`."""
        for tag, rgb_img in samples:
            self.writer.add_image(f"predictions/{tag}", rgb_img, global_step=step, dataformats="HWC")

    def log_hparams(
        self,
        hparams: dict[str, Any],
        metrics: dict[str, float],
    ) -> None:
        """Log primitive hyperparameters and summary metrics to the HPARAMS tab."""
        clean_hp: dict[str, int | float | str | bool] = {}
        for k, v in hparams.items():
            if isinstance(v, (bool, int, float, str)):
                clean_hp[k] = v
            elif v is not None:
                clean_hp[k] = str(v)

        clean_metrics: dict[str, float] = {}
        for k, v in metrics.items():
            if v is not None and math.isfinite(float(v)):
                clean_metrics[k] = float(v)

        if clean_hp and clean_metrics:
            self.writer.add_hparams(clean_hp, clean_metrics, run_name=".")

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
