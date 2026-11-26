"""Image integrity and YOLO annotation validator."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, UnidentifiedImageError

from src.utils import IMAGE_EXTENSIONS, load_classes


@dataclass(frozen=True)
class BBox:
    """Normalized YOLO bounding box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    line_number: int = 0


@dataclass(frozen=True)
class ValidationIssue:
    """Single validation error or warning."""

    severity: Literal["error", "warning"]
    code: str
    file: str
    line: int | None
    message: str


@dataclass
class ImageAudit:
    """Audit result for a single image and its paired label file."""

    image_path: Path
    label_path: Path | None
    width: int
    height: int
    mode: str
    fmt: str
    boxes: list[BBox]
    is_corrupt: bool
    has_label_file: bool


@dataclass
class DatasetAuditResult:
    """Aggregated audit report for a YOLO image/label directory pair."""

    classes: list[str]
    images: list[ImageAudit]
    orphan_labels: list[Path]
    issues: list[ValidationIssue]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_classes": len(self.classes),
            "num_images": len(self.images),
            "num_boxes": sum(len(img.boxes) for img in self.images),
            "num_orphan_labels": len(self.orphan_labels),
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "is_valid": self.is_valid,
            "issues": [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "file": i.file,
                    "line": i.line,
                    "message": i.message,
                }
                for i in self.issues
            ],
        }


def clamp_yolo_box(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    min_size: float = 1e-6,
) -> tuple[float, float, float, float] | None:
    """Clamp YOLO box corners to `[0, 1]`; return `None` if degenerate."""
    values = (x_center, y_center, width, height)
    if any(not math.isfinite(v) for v in values) or width <= 0.0 or height <= 0.0:
        return None

    x1 = max(0.0, min(1.0, x_center - width / 2.0))
    y1 = max(0.0, min(1.0, y_center - height / 2.0))
    x2 = max(0.0, min(1.0, x_center + width / 2.0))
    y2 = max(0.0, min(1.0, y_center + height / 2.0))

    new_w = x2 - x1
    new_h = y2 - y1
    if new_w < min_size or new_h < min_size:
        return None

    new_xc = (x1 + x2) / 2.0
    new_yc = (y1 + y2) / 2.0
    return (new_xc, new_yc, new_w, new_h)


def compute_iou_xywh(a: BBox, b: BBox) -> float:
    """Compute Intersection over Union between two normalized `BBox` instances."""
    ax1, ay1 = a.x_center - a.width / 2.0, a.y_center - a.height / 2.0
    ax2, ay2 = a.x_center + a.width / 2.0, a.y_center + a.height / 2.0
    bx1, by1 = b.x_center - b.width / 2.0, b.y_center - b.height / 2.0
    bx2, by2 = b.x_center + b.width / 2.0, b.y_center + b.height / 2.0

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0.0 else 0.0


def validate_label_file(
    label_path: Path,
    num_classes: int,
    dup_iou_threshold: float = 0.99,
    corner_tol: float = 1e-4,
    suspicious_tol: float = 0.02,
) -> tuple[list[BBox], list[ValidationIssue]]:
    """Parse and validate a single YOLO `.txt` annotation file."""
    boxes: list[BBox] = []
    issues: list[ValidationIssue] = []
    filename = label_path.name

    raw_text = label_path.read_text(encoding="utf-8")
    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 5:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="malformed_line",
                    file=filename,
                    line=line_no,
                    message=f"Expected 5 fields, got {len(parts)}: {stripped!r}",
                )
            )
            continue

        try:
            class_id = int(parts[0])
            xc, yc, w, h = (float(token) for token in parts[1:])
        except ValueError:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="malformed_line",
                    file=filename,
                    line=line_no,
                    message=f"Non-numeric field in line: {stripped!r}",
                )
            )
            continue

        if any(not math.isfinite(v) for v in (xc, yc, w, h)):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="malformed_line",
                    file=filename,
                    line=line_no,
                    message=f"Non-finite coordinate in line: {stripped!r}",
                )
            )
            continue

        if class_id < 0 or class_id >= num_classes:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unknown_class",
                    file=filename,
                    line=line_no,
                    message=f"class_id={class_id} outside valid range [0, {num_classes - 1}]",
                )
            )
            continue

        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_coords",
                    file=filename,
                    line=line_no,
                    message=f"Center/size out of bounds: xc={xc}, yc={yc}, w={w}, h={h}",
                )
            )
            continue

        x1, y1 = xc - w / 2.0, yc - h / 2.0
        x2, y2 = xc + w / 2.0, yc + h / 2.0
        if x1 < -corner_tol or y1 < -corner_tol or x2 > 1.0 + corner_tol or y2 > 1.0 + corner_tol:
            max_overshoot = max(-x1, -y1, x2 - 1.0, y2 - 1.0)
            if max_overshoot > suspicious_tol:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="bbox_suspicious",
                        file=filename,
                        line=line_no,
                        message=(
                            f"Large corner overshoot ({max_overshoot:.4f} > {suspicious_tol}): "
                            f"likely shifted/corrupted box requiring manual inspection "
                            f"(x1={x1:.4f}, y1={y1:.4f}, x2={x2:.4f}, y2={y2:.4f})"
                        ),
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="bbox_out_of_bounds",
                        file=filename,
                        line=line_no,
                        message=(
                            f"Box corners exceed [0, 1]: x1={x1:.4f}, y1={y1:.4f}, "
                            f"x2={x2:.4f}, y2={y2:.4f}"
                        ),
                    )
                )
            continue

        candidate = BBox(
            class_id=class_id,
            x_center=xc,
            y_center=yc,
            width=w,
            height=h,
            line_number=line_no,
        )
        for existing in boxes:
            if existing.class_id == candidate.class_id and (
                compute_iou_xywh(existing, candidate) >= dup_iou_threshold
            ):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="duplicate_box",
                        file=filename,
                        line=line_no,
                        message=(
                            f"Near-duplicate box of class {class_id} "
                            f"(matches line {existing.line_number})"
                        ),
                    )
                )
                break

        boxes.append(candidate)

    return boxes, issues


def _inspect_image(image_path: Path) -> tuple[int, int, str, str, bool, str | None]:
    """Verify image readability and return `(width, height, mode, fmt, is_corrupt, error)`."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
            return img.width, img.height, img.mode, img.format or "", False, None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return 0, 0, "", "", True, str(exc)


def audit_dataset(
    images_dir: Path,
    labels_dir: Path,
    classes_path: Path,
) -> DatasetAuditResult:
    """Audit all images and paired YOLO `.txt` labels in a dataset directory."""
    classes = load_classes(classes_path)
    num_classes = len(classes)

    image_paths = sorted(
        (
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda p: p.name,
    )
    label_paths = {
        p.stem: p
        for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    }

    issues: list[ValidationIssue] = []
    image_audits: list[ImageAudit] = []
    image_stems = {p.stem for p in image_paths}

    for img_path in image_paths:
        width, height, mode, fmt, is_corrupt, err_msg = _inspect_image(img_path)
        if is_corrupt:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="corrupt_image",
                    file=img_path.name,
                    line=None,
                    message=f"Failed to decode image: {err_msg}",
                )
            )

        lbl_path = label_paths.get(img_path.stem)
        boxes: list[BBox] = []
        has_label_file = lbl_path is not None
        if lbl_path is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_label",
                    file=img_path.name,
                    line=None,
                    message=f"Missing label file for image {img_path.name}",
                )
            )
        else:
            boxes, file_issues = validate_label_file(lbl_path, num_classes=num_classes)
            issues.extend(file_issues)

        image_audits.append(
            ImageAudit(
                image_path=img_path,
                label_path=lbl_path,
                width=width,
                height=height,
                mode=mode,
                fmt=fmt,
                boxes=boxes,
                is_corrupt=is_corrupt,
                has_label_file=has_label_file,
            )
        )

    orphan_labels = sorted(
        (p for stem, p in label_paths.items() if stem not in image_stems),
        key=lambda p: p.name,
    )
    for orphan in orphan_labels:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="orphan_label",
                file=orphan.name,
                line=None,
                message=f"Label file {orphan.name} has no matching image",
            )
        )

    return DatasetAuditResult(
        classes=classes,
        images=image_audits,
        orphan_labels=orphan_labels,
        issues=issues,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate YOLO images and labels.")
    parser.add_argument("--images", type=Path, required=True, help="Directory with images.")
    parser.add_argument("--labels", type=Path, required=True, help="Directory with YOLO .txt labels.")
    parser.add_argument("--classes", type=Path, required=True, help="Path to classes.txt.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = audit_dataset(args.images, args.labels, args.classes)
    total_boxes = sum(len(img.boxes) for img in result.images)
    print(
        f"Audited {len(result.images)} images, {total_boxes} valid boxes, "
        f"{len(result.classes)} classes | "
        f"{len(result.errors)} errors, {len(result.warnings)} warnings"
    )
    for issue in result.issues:
        loc = f":{issue.line}" if issue.line is not None else ""
        print(f"  [{issue.severity.upper()}] {issue.code} {issue.file}{loc} - {issue.message}")
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
