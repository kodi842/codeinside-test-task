"""Unit tests for dataset image and YOLO label validation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from src.validate_labels import (
    BBox,
    audit_dataset,
    clamp_yolo_box,
    compute_iou_xywh,
    validate_label_file,
)


def _create_image(path: Path, size: tuple[int, int] = (64, 64), mode: str = "RGB") -> None:
    color = (120, 140, 160, 255) if mode == "RGBA" else (120, 140, 160)
    img = Image.new(mode, size, color=color)
    img.save(path)


def _write_classes(path: Path, n: int = 11) -> Path:
    names = [f"cls_{i}" for i in range(n)]
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return path


def test_valid_and_empty_negative_labels(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    classes_path = _write_classes(tmp_path / "classes.txt", 11)

    _create_image(images_dir / "valid.jpg")
    (labels_dir / "valid.txt").write_text("1 0.5 0.5 0.2 0.3\n10 0.2 0.2 0.1 0.1\n", encoding="utf-8")

    _create_image(images_dir / "negative.png")
    (labels_dir / "negative.txt").write_text("", encoding="utf-8")

    result = audit_dataset(images_dir, labels_dir, classes_path)
    assert result.is_valid
    assert len(result.errors) == 0
    assert len(result.warnings) == 0
    assert len(result.images) == 2
    by_name = {item.image_path.name: item for item in result.images}
    assert len(by_name["valid.jpg"].boxes) == 2
    assert len(by_name["negative.png"].boxes) == 0
    assert by_name["negative.png"].has_label_file is True


def test_missing_label_and_orphan_label(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    classes_path = _write_classes(tmp_path / "classes.txt", 11)

    _create_image(images_dir / "no_lbl.jpg")
    (labels_dir / "ghost.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    result = audit_dataset(images_dir, labels_dir, classes_path)
    assert not result.is_valid
    codes = {issue.code for issue in result.issues}
    assert "missing_label" in codes
    assert "orphan_label" in codes
    assert any(p.name == "ghost.txt" for p in result.orphan_labels)


def test_unknown_class_and_malformed_lines(tmp_path: Path) -> None:
    lbl = tmp_path / "bad.txt"
    lbl.write_text(
        "\n".join(
            [
                "11 0.5 0.5 0.2 0.2",
                "-1 0.5 0.5 0.2 0.2",
                "1 0.5 0.5 0.2",
                "1 abc 0.5 0.2 0.2",
                "1 nan 0.5 0.2 0.2",
                "1 inf 0.5 0.2 0.2",
                "2 0.5 0.5 0.2 0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    boxes, issues = validate_label_file(lbl, num_classes=11)
    assert len(boxes) == 1
    assert boxes[0].class_id == 2
    unknown_issues = [i for i in issues if i.code == "unknown_class"]
    malformed_issues = [i for i in issues if i.code == "malformed_line"]
    assert len(unknown_issues) == 2
    assert len(malformed_issues) == 4
    assert all(i.severity == "error" for i in unknown_issues + malformed_issues)


def test_invalid_coords_and_corner_oob(tmp_path: Path) -> None:
    lbl = tmp_path / "coords.txt"
    lbl.write_text(
        "\n".join(
            [
                "1 1.2 0.5 0.2 0.2",    # center out of [0, 1]
                "1 0.5 0.5 0.0 0.2",    # zero width
                "1 0.5 0.5 -0.1 0.2",   # negative width
                "1 0.5 0.95 0.2 0.12",  # y2 = 1.01 > 1 (small 1% corner OOB)
                "1 0.02 0.5 0.06 0.2",  # x1 = -0.01 < 0 (small 1% corner OOB)
                "1 0.5 0.99 0.2 0.2",   # y2 = 1.09 > 1 (large 9% overshoot triggers bbox_suspicious)
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    boxes, issues = validate_label_file(lbl, num_classes=11)
    assert len(boxes) == 0
    invalid_coords = [i for i in issues if i.code == "invalid_coords"]
    corner_oob = [i for i in issues if i.code == "bbox_out_of_bounds"]
    suspicious = [i for i in issues if i.code == "bbox_suspicious"]
    assert len(invalid_coords) == 3
    assert len(corner_oob) == 2
    assert len(suspicious) == 1


def test_clamp_yolo_box() -> None:
    clamped = clamp_yolo_box(0.5, 0.95, 0.2, 0.2)
    assert clamped is not None
    xc, yc, w, h = clamped
    assert xc == pytest.approx(0.5)
    assert w == pytest.approx(0.2)
    assert yc - h / 2 >= 0.0
    assert yc + h / 2 == pytest.approx(1.0)
    assert h == pytest.approx(0.15)

    # Completely outside [0, 1], so clamping returns None
    assert clamp_yolo_box(1.5, 0.5, 0.2, 0.2) is None
    assert clamp_yolo_box(0.5, 0.5, 0.0, 0.2) is None


def test_duplicate_boxes_and_corrupt_image(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    classes_path = _write_classes(tmp_path / "classes.txt", 11)

    (images_dir / "corrupt.jpg").write_bytes(b"not-a-valid-jpeg")
    (labels_dir / "corrupt.txt").write_text(
        "1 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    result = audit_dataset(images_dir, labels_dir, classes_path)
    assert not result.is_valid
    assert any(i.code == "corrupt_image" and i.severity == "error" for i in result.issues)
    assert any(i.code == "duplicate_box" and i.severity == "warning" for i in result.issues)
    assert compute_iou_xywh(BBox(1, 0.5, 0.5, 0.2, 0.2), BBox(1, 0.5, 0.5, 0.2, 0.2)) == pytest.approx(1.0)
