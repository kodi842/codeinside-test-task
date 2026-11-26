"""Unit tests for leakage-free dataset preparation, RGB export, and manifest verification."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from src.prepare_dataset import (
    _clean_output_dir,
    build_leakage_groups,
    compute_dhash,
    export_image_rgb,
    extract_session_key,
    prepare_dataset,
    split_groups_stratified,
    verify_manifest,
    write_manifest,
)


def _save_pattern_image(path: Path, seed: int, mode: str = "RGB", fmt: str | None = None) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    arr[:, :32, :] = (arr[:, :32, :] // 2) + (seed * 37 % 120)
    if mode == "RGBA":
        alpha = np.full((64, 64, 1), 200, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=2)
    img = Image.fromarray(arr, mode=mode)
    img.save(path, format=fmt)


def test_leakage_groups_dhash_session_and_manual(tmp_path: Path) -> None:
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    for x in range(64):
        base[:, x, :] = (x * 4) % 256
    p1 = tmp_path / "camA_01.png"
    p2 = tmp_path / "camA_02.png"
    Image.fromarray(base).save(p1)
    tweaked = base.copy()
    tweaked[10:15, 10:15] = 255
    Image.fromarray(tweaked).save(p2)

    s1 = tmp_path / "1_170709082023.png"
    s2 = tmp_path / "11_170709082023.png"
    _save_pattern_image(s1, seed=101)
    _save_pattern_image(s2, seed=202)

    # Unrelated images linked manually (for example day and night frames from one camera)
    day = tmp_path / "cam_day.png"
    night = tmp_path / "cam_night.png"
    _save_pattern_image(day, seed=401)
    _save_pattern_image(night, seed=777)

    uniq = tmp_path / "unrelated.jpg"
    _save_pattern_image(uniq, seed=303)

    assert extract_session_key("1_170709082023.png") is not None
    assert extract_session_key("1_170709082023.png") == extract_session_key("11_170709082023.png")
    assert extract_session_key("0 - 2025-04-16 16-59-56-449_00009570.jpg") == extract_session_key(
        "0 - 2025-04-16 16-59-56-449_00026790.jpg"
    )

    with Image.open(p1) as im1, Image.open(p2) as im2:
        dist = int(np.sum(compute_dhash(im1) != compute_dhash(im2)))
        assert dist <= 12

    groups = build_leakage_groups(
        [p1, p2, s1, s2, day, night, uniq],
        dhash_threshold=12,
        manual_groups=[["cam_day.png", "cam_night.png"]],
    )
    group_sets = [set(p.name for p in g) for g in groups]
    assert {"camA_01.png", "camA_02.png"} in group_sets
    assert {"1_170709082023.png", "11_170709082023.png"} in group_sets
    assert {"cam_day.png", "cam_night.png"} in group_sets
    assert {"unrelated.jpg"} in group_sets


def test_split_determinism_zero_leakage_rare_and_cluster_cap(tmp_path: Path) -> None:
    groups = []
    stem_to_classes: dict[str, Counter[int]] = {}
    # Group 0 has 5 frames (>50% of val/test target ~3), so it stays in train
    big_grp = []
    for f_idx in range(5):
        p = tmp_path / f"bigcam_f{f_idx}.jpg"
        big_grp.append(p)
        stem_to_classes[p.stem] = Counter({1: 2, 3: 1})
    groups.append(big_grp)

    for g_idx in range(1, 14):
        p = tmp_path / f"grp{g_idx}_f0.jpg"
        cls_counter: Counter[int] = Counter({1: 2})
        if g_idx == 13:
            cls_counter[5] = 1  # singleton rare class
        elif g_idx in (2, 4, 6):
            cls_counter[3] = 1
        stem_to_classes[p.stem] = cls_counter
        groups.append([p])

    split_a = split_groups_stratified(groups, stem_to_classes, ratios=(0.7, 0.15, 0.15), seed=42)
    split_b = split_groups_stratified(groups, stem_to_classes, ratios=(0.7, 0.15, 0.15), seed=42)

    assert {k: [p.name for p in v] for k, v in split_a.items()} == {
        k: [p.name for p in v] for k, v in split_b.items()
    }

    train_names = {p.name for p in split_a["train"]}
    assert "grp13_f0.jpg" in train_names
    assert all(f"bigcam_f{i}.jpg" in train_names for i in range(5))


def test_prepare_dataset_raw_only_rgba_cyrillic_dynamic_report_and_manifest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "mini"
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (raw_dir / "classes.txt").write_text("bicycle\ncar\nbus\n", encoding="utf-8")

    rgba_name = "Снимок экрана 1.png"
    _save_pattern_image(images_dir / rgba_name, seed=1, mode="RGBA", fmt="PNG")
    (labels_dir / "Снимок экрана 1.txt").write_text("1 0.5 0.95 0.2 0.12\n99 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    jpg_name = "rgb_frame.jpg"
    _save_pattern_image(images_dir / jpg_name, seed=2, mode="RGB", fmt="JPEG")
    orig_jpg_bytes = (images_dir / jpg_name).read_bytes()
    (labels_dir / "rgb_frame.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    _save_pattern_image(images_dir / "neg_empty.png", seed=3, mode="RGB", fmt="PNG")
    (labels_dir / "neg_empty.txt").write_text("", encoding="utf-8")

    for i in range(4, 11):
        _save_pattern_image(images_dir / f"extra_{i}.jpg", seed=i * 10, mode="RGB", fmt="JPEG")
        (labels_dir / f"extra_{i}.txt").write_text("1 0.4 0.4 0.2 0.2\n2 0.7 0.7 0.1 0.1\n", encoding="utf-8")

    _save_pattern_image(images_dir / "unlabeled_drop.jpg", seed=999, mode="RGB", fmt="JPEG")

    out_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"

    summary = prepare_dataset(
        input_dir=raw_dir,
        output_dir=out_dir,
        reports_dir=reports_dir,
        seed=42,
        raw_only=True,
        root_dir=tmp_path,
    )
    assert summary["used_curated_labels"] is False

    exported_imgs = {p.name for p in out_dir.glob("images/**/*") if p.is_file()}
    assert "unlabeled_drop.jpg" not in exported_imgs
    assert "neg_empty.png" in exported_imgs
    assert rgba_name in exported_imgs
    assert jpg_name in exported_imgs

    exported_rgba = next(out_dir.glob(f"images/**/{rgba_name}"))
    with Image.open(exported_rgba) as im:
        assert im.mode == "RGB"
    exported_jpg = next(out_dir.glob(f"images/**/{jpg_name}"))
    assert exported_jpg.read_bytes() == orig_jpg_bytes

    rgba_lbl = next(out_dir.glob("labels/**/Снимок экрана 1.txt"))
    rgba_lines = [l for l in rgba_lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rgba_lines) == 1
    assert rgba_lines[0].startswith("1 ")
    jpg_lbl = next(out_dir.glob("labels/**/rgb_frame.txt"))
    jpg_lines = [l for l in jpg_lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(jpg_lines) == 1

    yaml_text = (out_dir / "dataset.yaml").read_text(encoding="utf-8")
    assert str(tmp_path) not in yaml_text
    ds_cfg = yaml.safe_load(yaml_text)
    assert "path" not in ds_cfg
    assert ds_cfg["train"] == "images/train"
    assert ds_cfg["nc"] == 3

    # Report should reflect the input dataset only
    report_text = (reports_dir / "dataset_report.md").read_text(encoding="utf-8")
    assert "COCO" not in report_text
    assert "class_id=11" not in report_text
    assert "unlabeled_drop.jpg" in report_text

    manifest_path = reports_dir / "manifest.sha256"
    ok, errs = verify_manifest(manifest_path, out_dir, root_dir=tmp_path)
    assert ok, errs

    exported_jpg.write_bytes(b"corrupted")
    ok_bad, errs_bad = verify_manifest(manifest_path, out_dir, root_dir=tmp_path)
    assert not ok_bad
    assert any("rgb_frame.jpg" in e for e in errs_bad)


def test_clean_output_dir_refuses_non_dataset_dir(tmp_path: Path) -> None:
    unsafe_dir = tmp_path / "unsafe"
    unsafe_dir.mkdir()
    (unsafe_dir / "README.md").write_text("do not delete", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to wipe"):
        _clean_output_dir(unsafe_dir)
    assert (unsafe_dir / "README.md").exists()


def test_manifest_outside_repo_root_fallback(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    external_out = tmp_path / "external_processed"
    (external_out / "images" / "train").mkdir(parents=True)
    (external_out / "images" / "train" / "a.txt").write_text("ok", encoding="utf-8")
    manifest = repo_root / "manifest.sha256"

    entries = write_manifest(external_out, manifest, root_dir=repo_root)
    assert "images/train/a.txt" in entries
    ok, errs = verify_manifest(manifest, external_out, root_dir=repo_root)
    assert ok, errs


def test_export_image_rgb_determinism(tmp_path: Path) -> None:
    src = tmp_path / "alpha.png"
    _save_pattern_image(src, seed=77, mode="RGBA", fmt="PNG")
    dst1 = tmp_path / "out1.png"
    dst2 = tmp_path / "out2.png"
    assert export_image_rgb(src, dst1) is True
    assert export_image_rgb(src, dst2) is True
    assert dst1.read_bytes() == dst2.read_bytes()
