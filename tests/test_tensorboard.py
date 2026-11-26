"""Unit tests for TensorBoard logger, visualization, and trainer callbacks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.run_experiment import _extract_epoch_scalars, _make_fit_epoch_callback
from src.tensorboard_logger import TensorBoardLogger, render_gt_vs_pred
from src.validate_labels import BBox


def test_tensorboard_logger_writes_all_artifact_types(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (320, 240), color=(40, 80, 120)).save(img_path)

    gt = [BBox(class_id=1, x_center=0.5, y_center=0.5, width=0.2, height=0.2)]
    pred = [(1, 0.52, 0.48, 0.22, 0.19, 0.87)]
    class_names = ["bicycle", "car", "motorcycle"]

    panel = render_gt_vs_pred(img_path, gt, pred, class_names, max_width=640)
    assert panel.ndim == 3
    assert panel.shape[2] == 3
    assert panel.dtype == np.uint8

    tb_dir = tmp_path / "tb"
    with TensorBoardLogger(tb_dir) as logger:
        logger.log_run_metadata(
            config={"epochs": 1, "imgsz": 320},
            git_info={"commit_short": "abc1234", "branch": "main", "is_dirty": False},
            dataset_info={"nc": 3, "manifest_sha256": "deadbeef", "manifest_verified": True},
            env_info={"python": "3.12", "torch": "2.15.0", "device": "cpu"},
            mode="dry_run",
        )
        logger.log_epoch_metrics(
            epoch=1,
            metrics={
                "loss/train_box": 1.25,
                "loss/val_box": 1.10,
                "metrics/precision": 0.5,
                "metrics/recall": 0.4,
                "metrics/mAP50": 0.45,
                "metrics/mAP50_95": 0.25,
                "lr/pg0": 0.01,
            },
        )
        cm = np.array([[2, 0, 0, 1], [0, 5, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0]], dtype=np.float32)
        logger.log_confusion_matrix(cm, class_names, step=1)
        logger.log_prediction_samples([("val_sample_0", panel)], step=1)
        logger.log_hparams({"epochs": 1, "imgsz": 320, "mode": "dry_run"}, {"hparam/mAP50": 0.45})

    acc = EventAccumulator(str(tb_dir))
    acc.Reload()
    tags = acc.Tags()

    assert "loss/train_box" in tags["scalars"]
    assert "metrics/mAP50" in tags["scalars"]
    assert "metrics/mAP50_95" in tags["scalars"]
    assert "eval/confusion_matrix" in tags["images"]
    assert "predictions/val_sample_0" in tags["images"]
    assert len(tags["tensors"]) >= 1


def test_fit_epoch_callback_filters_final_eval_and_duplicates(tmp_path: Path) -> None:
    tb_dir = tmp_path / "tb_cb"
    history: list[dict[str, float]] = []
    logged: set[int] = set()

    def _fake_trainer(epoch_idx: int, map50: float, box_loss: float) -> SimpleNamespace:
        return SimpleNamespace(
            epoch=epoch_idx,
            tloss=(box_loss, 1.0, 0.5),
            label_loss_items=lambda tloss, prefix="train": {
                f"{prefix}/box_loss": float(tloss[0]),
                f"{prefix}/cls_loss": float(tloss[1]),
                f"{prefix}/dfl_loss": float(tloss[2]),
            },
            metrics={
                "val/box_loss": box_loss + 0.1,
                "metrics/precision(B)": 0.7,
                "metrics/recall(B)": 0.3,
                "metrics/mAP50(B)": map50,
                "metrics/mAP50-95(B)": map50 * 0.6,
            },
            lr={"lr/pg0": 0.001},
        )

    with TensorBoardLogger(tb_dir) as logger:
        cb = _make_fit_epoch_callback(logger, epochs=5, epoch_history=history, logged_epochs=logged)
        for ep in range(5):
            cb(_fake_trainer(ep, map50=0.1 + 0.05 * ep, box_loss=1.5 - 0.1 * ep))
        # Duplicate epoch 5 call should be ignored
        cb(_fake_trainer(4, map50=0.99, box_loss=0.1))
        # Ultralytics final_eval() increments trainer.epoch to 5 (step 6 > epochs=5)
        cb(_fake_trainer(5, map50=0.42, box_loss=1.1))

    assert len(history) == 5
    acc = EventAccumulator(str(tb_dir))
    acc.Reload()
    train_steps = [e.step for e in acc.Scalars("loss/train_box")]
    map_steps = [e.step for e in acc.Scalars("metrics/mAP50")]
    assert train_steps == [1, 2, 3, 4, 5]
    assert map_steps == [1, 2, 3, 4, 5]
    assert "final/mAP50" in acc.Tags()["scalars"]
    assert acc.Scalars("final/mAP50")[0].value == pytest.approx(0.42)

    extracted = _extract_epoch_scalars(_fake_trainer(0, map50=0.25, box_loss=1.2))
    assert extracted["loss/train_box"] == pytest.approx(1.2)
    assert extracted["metrics/mAP50"] == pytest.approx(0.25)
