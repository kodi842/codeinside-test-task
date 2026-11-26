"""CLI entrypoint for Ultralytics YOLO experiments with TensorBoard logging."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from src.prepare_dataset import verify_manifest
from src.tensorboard_logger import TensorBoardLogger, render_gt_vs_pred
from src.utils import (
    IMAGE_EXTENSIONS,
    find_repo_root,
    get_git_info,
    load_yaml,
    resolve_device,
    set_seed,
    sha256_file,
)
from src.validate_labels import BBox, validate_label_file


def _rel_to_repo(path: Path, repo_root: Path) -> str:
    pr = path.resolve()
    return pr.relative_to(repo_root).as_posix() if pr.is_relative_to(repo_root) else path.name


def _resolve_model_spec(model_name: str, repo_root: Path) -> str:
    """Cache bare `.pt` weights inside `<repo_root>/weights/` instead of repo root."""
    p = Path(model_name)
    if p.parent == Path(".") and model_name.endswith(".pt"):
        weights_dir = repo_root / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        return str((weights_dir / model_name).resolve())
    return model_name


def _resolve_relative_data_root(data_yaml_path: Path, data_cfg: dict[str, Any]) -> Path:
    base = data_yaml_path.parent.resolve()
    raw_path = data_cfg.get("path")
    if raw_path is None or str(raw_path).strip() in ("", "."):
        return base
    p = Path(str(raw_path))
    return p if p.is_absolute() else (base / p).resolve()


def _extract_class_names(data_cfg: dict[str, Any]) -> list[str]:
    names = data_cfg.get("names", {})
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names.keys(), key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(n) for n in names]
    raise ValueError(f"Unsupported 'names' format in dataset.yaml: {type(names).__name__}")


def _extract_epoch_scalars(trainer: Any) -> dict[str, float]:
    """Extract standardized scalar metrics from an Ultralytics `BaseTrainer` instance."""
    scalars: dict[str, float] = {}

    if getattr(trainer, "tloss", None) is not None and hasattr(trainer, "label_loss_items"):
        train_losses = trainer.label_loss_items(trainer.tloss, prefix="train")
        mapping = {
            "train/box_loss": "loss/train_box",
            "train/cls_loss": "loss/train_cls",
            "train/dfl_loss": "loss/train_dfl",
        }
        for src_k, dst_k in mapping.items():
            if src_k in train_losses:
                scalars[dst_k] = float(train_losses[src_k])

    trainer_metrics = getattr(trainer, "metrics", None) or {}
    metric_map = {
        "val/box_loss": "loss/val_box",
        "val/cls_loss": "loss/val_cls",
        "val/dfl_loss": "loss/val_dfl",
        "metrics/precision(B)": "metrics/precision",
        "metrics/recall(B)": "metrics/recall",
        "metrics/mAP50(B)": "metrics/mAP50",
        "metrics/mAP50-95(B)": "metrics/mAP50_95",
    }
    for src_k, dst_k in metric_map.items():
        if src_k in trainer_metrics:
            scalars[dst_k] = float(trainer_metrics[src_k])

    lr_dict = getattr(trainer, "lr", None) or {}
    if "lr/pg0" in lr_dict:
        scalars["lr/pg0"] = float(lr_dict["lr/pg0"])

    return scalars


def _make_fit_epoch_callback(
    tb_logger: TensorBoardLogger,
    epochs: int,
    epoch_history: list[dict[str, float]],
    logged_epochs: set[int],
) -> Callable[[Any], None]:
    """Build `on_fit_epoch_end` callback separating training epochs from `final_eval()`."""

    def _on_fit_epoch_end(trainer: Any) -> None:
        epoch_num = int(getattr(trainer, "epoch", 0)) + 1
        scalars = _extract_epoch_scalars(trainer)

        # Ultralytics `final_eval()` increments trainer.epoch to `epochs` (so epoch_num = epochs + 1).
        # Route that post-training `best.pt` evaluation to `final/*` instead of creating a bogus step.
        if epoch_num > epochs:
            final_only = {
                f"final/{k.split('/', 1)[1]}": v
                for k, v in scalars.items()
                if k.startswith("metrics/")
            }
            if final_only:
                tb_logger.log_epoch_metrics(epochs, final_only)
            return

        if epoch_num in logged_epochs:
            return
        logged_epochs.add(epoch_num)
        tb_logger.log_epoch_metrics(epoch_num, scalars)
        epoch_history.append({"epoch": float(epoch_num), **scalars})

    return _on_fit_epoch_end


def _collect_val_predictions(
    model: YOLO,
    val_images_dir: Path,
    val_labels_dir: Path,
    class_names: list[str],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    max_samples: int,
) -> list[tuple[str, np.ndarray]]:
    """Run inference on up to `max_samples` validation images and render GT-vs-Pred panels."""
    val_images = sorted(
        (p for p in val_images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name,
    )[:max_samples]

    panels: list[tuple[str, np.ndarray]] = []
    for idx, img_path in enumerate(val_images):
        lbl_path = val_labels_dir / f"{img_path.stem}.txt"
        gt_boxes: list[BBox] = []
        if lbl_path.exists():
            gt_boxes, _ = validate_label_file(lbl_path, num_classes=len(class_names))

        res = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            save=False,
            verbose=False,
        )[0]

        preds: list[tuple[int, float, float, float, float, float]] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xywhn = res.boxes.xywhn.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for (xc, yc, bw, bh), cid, cval in zip(xywhn, clss, confs):
                preds.append((int(cid), float(xc), float(yc), float(bw), float(bh), float(cval)))

        panel = render_gt_vs_pred(img_path, gt_boxes, preds, class_names)
        panels.append((f"val_sample_{idx}", panel))
    return panels


def _safe_metric(val: Any) -> float:
    fval = float(val)
    return fval if math.isfinite(fval) else 0.0


def _run_yolo_session(
    model_spec: str,
    data_yaml: Path,
    yolo_project_dir: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    nbs: int,
    device: str,
    seed: int,
    workers: int,
    lr0: float,
    optimizer: str,
    patience: int,
    conf_threshold: float,
    iou_threshold: float,
    num_vis_samples: int,
    class_names: list[str],
    tb_logger: TensorBoardLogger,
) -> dict[str, Any]:
    """Execute Ultralytics YOLO training, checkpoint check, and val + test evaluation."""
    model = YOLO(model_spec)
    epoch_history: list[dict[str, float]] = []
    logged_epochs: set[int] = set()
    resolved_opt_info: dict[str, Any] = {}

    def _on_train_start(trainer: Any) -> None:
        opt = getattr(trainer, "optimizer", None)
        if opt is not None:
            resolved_opt_info["optimizer_class"] = type(opt).__name__
            if getattr(opt, "param_groups", None):
                resolved_opt_info["initial_lr"] = float(opt.param_groups[0].get("lr", lr0))

    model.add_callback("on_train_start", _on_train_start)
    model.add_callback(
        "on_fit_epoch_end",
        _make_fit_epoch_callback(tb_logger, epochs, epoch_history, logged_epochs),
    )

    model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        nbs=nbs,
        device=device,
        seed=seed,
        workers=workers,
        lr0=lr0,
        optimizer=optimizer,
        patience=patience,
        project=str(yolo_project_dir.resolve()),
        name="train",
        exist_ok=True,
        plots=False,
        verbose=False,
    )

    weights_dir = yolo_project_dir / "train" / "weights"
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"
    eval_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
    if not eval_ckpt.exists():
        raise FileNotFoundError(f"No checkpoint saved under {weights_dir}")

    eval_model = YOLO(str(eval_ckpt))
    val_results = eval_model.val(
        data=str(data_yaml.resolve()),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(yolo_project_dir.resolve()),
        name="val",
        exist_ok=True,
        plots=False,
        verbose=False,
    )

    final_metrics = {
        "metrics/precision": _safe_metric(val_results.box.mp),
        "metrics/recall": _safe_metric(val_results.box.mr),
        "metrics/mAP50": _safe_metric(val_results.box.map50),
        "metrics/mAP50_95": _safe_metric(val_results.box.map),
    }

    if hasattr(val_results, "confusion_matrix") and val_results.confusion_matrix is not None:
        cm_matrix = np.asarray(val_results.confusion_matrix.matrix, dtype=np.float64)
        tb_logger.log_confusion_matrix(cm_matrix, class_names=class_names, step=epochs)

    # Evaluate best checkpoint on held-out test split if configured
    data_cfg = load_yaml(data_yaml)
    data_root = _resolve_relative_data_root(data_yaml, data_cfg)
    test_metrics: dict[str, float] = {}
    if data_cfg.get("test") and (data_root / str(data_cfg["test"])).exists():
        test_results = eval_model.val(
            data=str(data_yaml.resolve()),
            split="test",
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(yolo_project_dir.resolve()),
            name="test",
            exist_ok=True,
            plots=False,
            verbose=False,
        )
        test_metrics = {
            "test/precision": _safe_metric(test_results.box.mp),
            "test/recall": _safe_metric(test_results.box.mr),
            "test/mAP50": _safe_metric(test_results.box.map50),
            "test/mAP50_95": _safe_metric(test_results.box.map),
        }
        tb_logger.log_epoch_metrics(epochs, test_metrics)

    val_images_dir = data_root / str(data_cfg["val"])
    val_labels_dir = data_root / "labels" / Path(str(data_cfg["val"])).name
    panels = _collect_val_predictions(
        model=eval_model,
        val_images_dir=val_images_dir,
        val_labels_dir=val_labels_dir,
        class_names=class_names,
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        max_samples=num_vis_samples,
    )
    tb_logger.log_prediction_samples(panels, step=epochs)

    return {
        "best_checkpoint": best_ckpt if best_ckpt.exists() else None,
        "last_checkpoint": last_ckpt if last_ckpt.exists() else None,
        "final_metrics": final_metrics,
        "test_metrics": test_metrics,
        "resolved_optimizer": resolved_opt_info,
        "epoch_history": epoch_history,
    }


def run_experiment(
    config_path: Path,
    data_yaml: Path,
    dry_run: bool = False,
    overrides: dict[str, Any] | None = None,
    strict_manifest: bool = False,
    require_clean: bool = False,
) -> Path:
    """Run YOLO training or dry-run and write logs to TensorBoard."""
    repo_root = find_repo_root(config_path)
    git_info = get_git_info(repo_root)
    if git_info.get("is_dirty"):
        if require_clean:
            raise RuntimeError("Working tree has uncommitted changes (--require-clean is set).")
        print("[WARN] Git working tree is dirty; recorded run code version may not be reproducible.")

    cfg = load_yaml(config_path)
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v

    mode = "dry_run" if dry_run else "train"
    epochs = 1 if dry_run else int(cfg.get("epochs", 5))
    imgsz = 320 if dry_run else int(cfg.get("imgsz", 640))
    batch = min(int(cfg.get("batch", 8)), 4) if dry_run else int(cfg.get("batch", 8))
    nbs = batch if dry_run else int(cfg.get("nbs", batch))
    seed = int(cfg.get("seed", 42))
    workers = int(cfg.get("workers", 0))
    lr0 = float(cfg.get("lr0", 0.001))
    optimizer = str(cfg.get("optimizer", "AdamW"))
    patience = int(cfg.get("patience", 10))
    conf_threshold = float(cfg.get("conf_threshold", 0.25))
    iou_threshold = float(cfg.get("iou_threshold", 0.5))
    num_vis_samples = int(cfg.get("num_vis_samples", 6))
    model_name = str(cfg.get("model", "yolo11n.pt"))
    model_spec = _resolve_model_spec(model_name, repo_root)
    exp_name = str(cfg.get("experiment_name", "yolo_experiment"))
    runs_dir = Path(str(cfg.get("runs_dir", "runs")))
    manifest_rel = str(cfg.get("manifest_path", "reports/manifest.sha256"))
    manifest_path = (repo_root / manifest_rel) if not Path(manifest_rel).is_absolute() else Path(manifest_rel)

    set_seed(seed)
    device = resolve_device(str(cfg.get("device", "auto")))

    data_cfg = load_yaml(data_yaml)
    class_names = _extract_class_names(data_cfg)
    data_root = _resolve_relative_data_root(data_yaml, data_cfg)

    manifest_verified = False
    manifest_sha256 = None
    manifest_errors: list[str] = []
    if manifest_path.exists():
        manifest_sha256 = sha256_file(manifest_path)
        manifest_verified, manifest_errors = verify_manifest(manifest_path, data_root, root_dir=repo_root)
        if not manifest_verified:
            msg = f"Manifest check warning ({manifest_path}): {manifest_errors[:3]}"
            if strict_manifest:
                raise RuntimeError(msg)
            print(f"[WARN] {msg}")
    elif strict_manifest:
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_dryrun" if dry_run else ""
    run_id = f"{timestamp}_{exp_name}{suffix}"
    run_dir = runs_dir / run_id
    tb_dir = run_dir / "tb"
    yolo_dir = run_dir / "yolo"
    run_dir.mkdir(parents=True, exist_ok=True)

    env_info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "requested_device": str(cfg.get("device", "auto")),
        "resolved_device": device,
    }
    dataset_info = {
        "data_yaml": _rel_to_repo(data_yaml, repo_root),
        "nc": len(class_names),
        "names": class_names,
        "manifest_sha256": manifest_sha256,
        "manifest_verified": manifest_verified,
    }
    effective_config = {
        "experiment_name": exp_name,
        "model": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "nbs": nbs,
        "device": device,
        "seed": seed,
        "workers": workers,
        "lr0": lr0,
        "optimizer": optimizer,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "num_vis_samples": num_vis_samples,
    }

    def _run_with_logger(target_device: str, current_env: dict[str, Any], current_cfg: dict[str, Any]) -> dict[str, Any]:
        with TensorBoardLogger(tb_dir) as tb_logger:
            tb_logger.log_run_metadata(
                config=current_cfg,
                git_info=git_info,
                dataset_info=dataset_info,
                env_info=current_env,
                mode=mode,
            )
            out = _run_yolo_session(
                model_spec=model_spec,
                data_yaml=data_yaml,
                yolo_project_dir=yolo_dir,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                nbs=nbs,
                device=target_device,
                seed=seed,
                workers=workers,
                lr0=lr0,
                optimizer=optimizer,
                patience=patience,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                num_vis_samples=num_vis_samples,
                class_names=class_names,
                tb_logger=tb_logger,
            )
            has_nan = any(
                not math.isfinite(v)
                for ep in out["epoch_history"]
                for v in ep.values()
            )
            if target_device == "mps" and has_nan:
                raise RuntimeError("NaN detected on MPS backend; falling back to CPU.")

            hparam_metrics = {
                "hparam/precision": out["final_metrics"]["metrics/precision"],
                "hparam/recall": out["final_metrics"]["metrics/recall"],
                "hparam/mAP50": out["final_metrics"]["metrics/mAP50"],
                "hparam/mAP50_95": out["final_metrics"]["metrics/mAP50_95"],
            }
            if "test/mAP50" in out["test_metrics"]:
                hparam_metrics["hparam/test_mAP50"] = out["test_metrics"]["test/mAP50"]
            tb_logger.log_hparams({"mode": mode, **current_cfg}, hparam_metrics)
            return out

    try:
        session_out = _run_with_logger(device, env_info, effective_config)
    except RuntimeError as exc:
        if device != "mps" or isinstance(exc.__cause__, FileNotFoundError):
            raise
        print(f"[WARN] MPS execution failed ({exc}); clearing tb/ and retrying cleanly on CPU.")
        if tb_dir.exists():
            shutil.rmtree(tb_dir)
        device = "cpu"
        effective_config["device"] = "cpu"
        env_info["resolved_device"] = "cpu"
        env_info["mps_fallback"] = str(exc)
        session_out = _run_with_logger("cpu", env_info, effective_config)

    run_meta = {
        "run_id": run_id,
        "mode": mode,
        "timestamp": timestamp,
        "config": effective_config,
        "resolved_optimizer": session_out["resolved_optimizer"],
        "git": git_info,
        "environment": env_info,
        "dataset": dataset_info,
        "final_metrics": session_out["final_metrics"],
        "test_metrics": session_out["test_metrics"],
        "checkpoints": {
            "best": _rel_to_repo(session_out["best_checkpoint"], repo_root) if session_out["best_checkpoint"] else None,
            "last": _rel_to_repo(session_out["last_checkpoint"], repo_root) if session_out["last_checkpoint"] else None,
        },
        "tensorboard_dir": _rel_to_repo(tb_dir, repo_root),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Ultralytics YOLO experiment with TensorBoard logging.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML experiment config.")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset.yaml.")
    parser.add_argument("--dry-run", action="store_true", help="Run fast 1-epoch sanity pipeline verification.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size.")
    parser.add_argument("--device", type=str, default=None, help="Override device (auto/cpu/mps/cuda).")
    parser.add_argument("--model", type=str, default=None, help="Override YOLO model weights/config.")
    parser.add_argument("--name", type=str, default=None, help="Override experiment_name.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Override runs directory.")
    parser.add_argument("--manifest", type=Path, default=None, help="Override manifest.sha256 path.")
    parser.add_argument("--strict-manifest", action="store_true", help="Fail if manifest verification fails.")
    parser.add_argument("--require-clean", action="store_true", help="Fail if git working tree has uncommitted edits.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    overrides = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "model": args.model,
        "experiment_name": args.name,
        "seed": args.seed,
        "runs_dir": str(args.runs_dir) if args.runs_dir is not None else None,
        "manifest_path": str(args.manifest) if args.manifest is not None else None,
    }
    run_dir = run_experiment(
        config_path=args.config,
        data_yaml=args.data,
        dry_run=args.dry_run,
        overrides=overrides,
        strict_manifest=args.strict_manifest,
        require_clean=args.require_clean,
    )
    print(f"Experiment completed: {run_dir}")
    print(f"View logs with: tensorboard --logdir {run_dir.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
