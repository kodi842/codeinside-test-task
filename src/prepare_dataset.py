"""Leakage-free dataset preparation, RGB export, reporting, and SHA-256 manifest."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.utils import find_repo_root, load_yaml, save_yaml, sha256_file
from src.validate_labels import (
    BBox,
    DatasetAuditResult,
    audit_dataset,
    clamp_yolo_box,
    compute_iou_xywh,
)
from src.viz import draw_yolo_boxes, resize_max_width


def compute_dhash(image: Image.Image, hash_size: int = 8) -> np.ndarray:
    """Compute a 64-bit difference hash (dHash) boolean array for an image."""
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size),
        Image.Resampling.LANCZOS,
    )
    arr = np.asarray(gray, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).flatten()


def extract_session_key(filename: str) -> str | None:
    """Extract camera-session grouping key from known timestamped filename patterns."""
    stem = Path(filename).stem

    # e.g. "0 - 2025-04-16 16-59-56-449_00009570" -> "0 - 2025-04-16 16-59-56-449"
    m_dashcam = re.match(r"^(0\s*-\s*\d{4}-\d{2}-\d{2}\s+\d{2}-\d{2}-\d{2}-\d+)_\d+$", stem)
    if m_dashcam:
        return f"dashcam:{m_dashcam.group(1)}"

    # e.g. "1_170709082023", "11_170709082023" -> "suffix:170709082023"
    m_suffix = re.match(r"^\d+_(\d{10,})$", stem)
    if m_suffix:
        return f"suffix:{m_suffix.group(1)}"

    # e.g. "10_9aug202317009", "102_9aug202317009" -> "camdate:9aug202317009"
    m_cam_date = re.match(r"^\d+_(\d+[a-z]{3}\d{4,})$", stem, flags=re.IGNORECASE)
    if m_cam_date:
        return f"camdate:{m_cam_date.group(1).lower()}"

    return None


def load_scene_groups(path: Path) -> list[list[str]]:
    """Load manual scene/camera group overrides from a YAML file."""
    data = load_yaml(path)
    raw_groups = data.get("groups", {})
    if not isinstance(raw_groups, dict):
        raise ValueError(f"Expected 'groups' mapping in {path}")
    groups: list[list[str]] = []
    for members in raw_groups.values():
        if isinstance(members, list) and len(members) >= 2:
            groups.append([str(m) for m in members])
    return groups


def build_leakage_groups(
    image_paths: list[Path],
    dhash_threshold: int = 12,
    manual_groups: list[list[str]] | None = None,
) -> list[list[Path]]:
    """Group visually near-duplicate, same-session, or manually linked frames via Union-Find."""
    ordered = sorted(image_paths, key=lambda p: p.name)
    n = len(ordered)
    if n == 0:
        return []

    parent = list(range(n))
    name_to_idx = {p.name: idx for idx, p in enumerate(ordered)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for grp in manual_groups or []:
        present = [name_to_idx[name] for name in grp if name in name_to_idx]
        for idx in present[1:]:
            union(present[0], idx)

    dhashes: list[np.ndarray | None] = []
    session_map: dict[str, list[int]] = defaultdict(list)

    for idx, path in enumerate(ordered):
        skey = extract_session_key(path.name)
        if skey is not None:
            session_map[skey].append(idx)
        try:
            with Image.open(path) as im:
                dhashes.append(compute_dhash(im))
        except Exception:
            dhashes.append(None)

    for indices in session_map.values():
        for idx in indices[1:]:
            union(indices[0], idx)

    for i in range(n):
        hi = dhashes[i]
        if hi is None:
            continue
        for j in range(i + 1, n):
            hj = dhashes[j]
            if hj is None:
                continue
            dist = int(np.sum(hi != hj))
            if dist <= dhash_threshold:
                union(i, j)

    buckets: dict[int, list[Path]] = defaultdict(list)
    for idx, path in enumerate(ordered):
        buckets[find(idx)].append(path)

    groups = [sorted(paths, key=lambda p: p.name) for paths in buckets.values()]
    groups.sort(key=lambda grp: grp[0].name)
    return groups


def split_groups_stratified(
    groups: list[list[Path]],
    stem_to_classes: dict[str, Counter[int]],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Assign whole image groups to train/val/test without leakage and balancing classes."""
    if len(ratios) != 3 or any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError(f"Invalid split ratios: {ratios}")

    norm_sum = sum(ratios)
    norm_ratios = tuple(r / norm_sum for r in ratios)
    splits: tuple[str, ...] = ("train", "val", "test")
    total_images = sum(len(g) for g in groups)
    target_counts = {sp: total_images * r for sp, r in zip(splits, norm_ratios)}

    group_classes: list[Counter[int]] = []
    class_group_freq: Counter[int] = Counter()
    for grp in groups:
        gc: Counter[int] = Counter()
        for path in grp:
            gc.update(stem_to_classes.get(path.stem, Counter()))
        group_classes.append(gc)
        for cid in gc:
            class_group_freq[cid] += 1

    rng = random.Random(seed)
    indices = list(range(len(groups)))
    rng.shuffle(indices)

    def _rarity_key(idx: int) -> tuple[int, int, int, str]:
        cids = list(group_classes[idx].keys())
        min_freq = min((class_group_freq[c] for c in cids), default=9999)
        is_singleton_cls = 0 if min_freq == 1 else 1
        return (is_singleton_cls, min_freq, -len(groups[idx]), groups[idx][0].name)

    indices.sort(key=_rarity_key)

    assigned: dict[str, list[Path]] = {sp: [] for sp in splits}
    split_img_counts: dict[str, int] = {sp: 0 for sp in splits}
    split_cls_counts: dict[str, Counter[int]] = {sp: Counter() for sp in splits}

    for g_idx in indices:
        grp = groups[g_idx]
        gc = group_classes[g_idx]
        has_unique_class = any(class_group_freq[cid] == 1 for cid in gc)

        best_split = "train"
        best_score = -1e18

        for sp_idx, sp in enumerate(splits):
            if norm_ratios[sp_idx] == 0:
                continue

            score = 0.0
            if has_unique_class:
                score += 500.0 if sp == "train" else -500.0

            # Prevent a single multi-frame cluster from dominating val or test
            if sp in ("val", "test"):
                max_cluster_share = max(2, int(round(target_counts[sp] * 0.5)))
                if len(grp) > max_cluster_share:
                    score -= 300.0

            # Reward covering classes not yet present in this split
            for cid in gc:
                if split_cls_counts[sp][cid] == 0:
                    weight = 40.0 if sp == "train" else 22.0
                    if sp != "train" and split_cls_counts["train"][cid] > 0:
                        weight = 35.0
                    score += weight / max(1, class_group_freq[cid] ** 0.5)

            # Ensure val and test receive at least one group when 3 or more groups exist
            if len(groups) >= 3 and len(assigned[sp]) == 0 and sp != "train" and not has_unique_class:
                score += 45.0

            # Prefer splits still below their target image count
            deficit = target_counts[sp] - split_img_counts[sp]
            score += deficit * 4.0
            if split_img_counts[sp] + len(grp) > target_counts[sp] + 1.0:
                overflow = (split_img_counts[sp] + len(grp)) - target_counts[sp]
                penalty_scale = 8.0 if sp == "train" else 16.0
                score -= overflow * penalty_scale

            score += rng.random() * 1e-4

            if score > best_score:
                best_score = score
                best_split = sp

        assigned[best_split].extend(grp)
        split_img_counts[best_split] += len(grp)
        split_cls_counts[best_split].update(gc)

    for sp in splits:
        assigned[sp].sort(key=lambda p: p.name)
    return assigned


def export_image_rgb(src_path: Path, dst_path: Path) -> bool:
    """Export image as RGB: byte-copy RGB files, deterministically convert RGBA/other modes."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as im:
        mode = im.mode
        fmt = im.format

    if mode == "RGB":
        shutil.copyfile(src_path, dst_path)
        return False

    with Image.open(src_path) as im:
        rgb = im.convert("RGB")
        if dst_path.suffix.lower() == ".png" or fmt == "PNG":
            rgb.save(dst_path, format="PNG", compress_level=6, optimize=False)
        else:
            rgb.save(dst_path, format="JPEG", quality=95)
    return True


def _resolve_manifest_base(output_dir: Path, root_dir: Path | None) -> Path:
    """Return `root_dir` if `output_dir` resides under it, otherwise `output_dir` itself."""
    out_resolved = output_dir.resolve()
    base = (root_dir or find_repo_root(output_dir)).resolve()
    return base if out_resolved.is_relative_to(base) else out_resolved


def write_manifest(
    output_dir: Path,
    manifest_path: Path,
    root_dir: Path | None = None,
) -> dict[str, str]:
    """Write sorted SHA-256 manifest with paths relative to `root_dir` (or `output_dir`)."""
    out_resolved = output_dir.resolve()
    base = _resolve_manifest_base(out_resolved, root_dir)
    entries: dict[str, str] = {}

    files = sorted(
        (
            p
            for p in out_resolved.rglob("*")
            if p.is_file() and p.name != ".gitkeep" and p.suffix != ".cache"
        ),
        key=lambda p: p.relative_to(base).as_posix(),
    )
    lines: list[str] = []
    for file_path in files:
        rel_posix = file_path.relative_to(base).as_posix()
        digest = sha256_file(file_path)
        entries[rel_posix] = digest
        lines.append(f"{digest}  {rel_posix}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return entries


def verify_manifest(
    manifest_path: Path,
    output_dir: Path,
    root_dir: Path | None = None,
) -> tuple[bool, list[str]]:
    """Read-only verification of `output_dir` against `manifest_path`."""
    if not manifest_path.exists():
        return False, [f"Manifest file not found: {manifest_path}"]

    out_resolved = output_dir.resolve()
    base = _resolve_manifest_base(out_resolved, root_dir)
    errors: list[str] = []
    expected: dict[str, str] = {}

    for lineno, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"Malformed manifest line {lineno}: {stripped!r}")
            continue
        digest, rel_path = parts
        expected[rel_path] = digest

    for rel_path, expected_digest in sorted(expected.items()):
        target = base / rel_path
        if not target.exists():
            errors.append(f"Missing file from manifest: {rel_path}")
            continue
        actual_digest = sha256_file(target)
        if actual_digest != expected_digest:
            errors.append(
                f"Checksum mismatch for {rel_path}: expected {expected_digest[:12]}..., got {actual_digest[:12]}..."
            )

    if out_resolved.exists():
        actual_files = {
            p.relative_to(base).as_posix()
            for p in out_resolved.rglob("*")
            if p.is_file() and p.name != ".gitkeep" and p.suffix != ".cache"
        }
        for extra in sorted(actual_files - set(expected.keys())):
            errors.append(f"Unexpected file not in manifest: {extra}")

    return len(errors) == 0, errors


def _clean_output_dir(output_dir: Path) -> None:
    """Safely clear a processed dataset directory while refusing to wipe unrelated directories."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    children = [c for c in output_dir.iterdir() if c.name != ".gitkeep"]
    if children:
        allowed_names = {"images", "labels", "dataset.yaml"}
        has_marker = (output_dir / "dataset.yaml").is_file()
        all_known = all(c.name in allowed_names for c in children)
        if not (has_marker or all_known):
            raise ValueError(
                f"Refusing to wipe non-dataset directory {output_dir}: "
                f"unexpected entries {[c.name for c in children if c.name not in allowed_names]}"
            )

    for child in children:
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _sanitize_raw_boxes(raw_label_path: Path | None, num_classes: int) -> list[BBox]:
    """Fallback sanitization in `--raw-only` mode: parse, clamp corners, deduplicate."""
    if raw_label_path is None or not raw_label_path.exists():
        return []
    cleaned: list[BBox] = []
    for lno, line in enumerate(raw_label_path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            continue
        try:
            cid = int(parts[0])
            xc, yc, w, h = (float(x) for x in parts[1:])
        except ValueError:
            continue
        if cid < 0 or cid >= num_classes:
            continue
        clamped = clamp_yolo_box(xc, yc, w, h)
        if clamped is None:
            continue
        cand = BBox(cid, *clamped, line_number=lno)
        if any(e.class_id == cand.class_id and compute_iou_xywh(e, cand) >= 0.99 for e in cleaned):
            continue
        cleaned.append(cand)
    return cleaned


def _render_samples(
    samples_dir: Path,
    split_map: dict[str, list[Path]],
    stem_to_boxes: dict[str, list[BBox]],
    raw_labels_dir: Path,
    classes: list[str],
    changelog: dict[str, Any] | None,
) -> list[str]:
    """Save lightweight JPEG bbox visualizations to `reports/samples/`."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    for old in samples_dir.glob("*.jpg"):
        old.unlink()

    saved: list[str] = []
    all_images_by_name = {p.name: p for paths in split_map.values() for p in paths}

    # 1. Select comparison panels dynamically from changelog (fixed_errors first, then supplemented/newly_labeled)
    if changelog:
        status_priority = {"fixed_errors": 0, "supplemented": 1, "newly_labeled": 2}
        candidates = [
            (status_priority[entry["status"]], name)
            for name, entry in changelog.items()
            if entry.get("status") in status_priority and name in all_images_by_name
        ]
        candidates.sort()
        # Keep all fixed_errors + up to 5 total comparison panels
        selected = candidates[:5]
        for idx, (_, img_name) in enumerate(selected, start=1):
            img_path = all_images_by_name[img_name]
            stem = img_path.stem
            with Image.open(img_path) as im:
                raw_path = raw_labels_dir / f"{stem}.txt"
                raw_boxes: list[BBox] = []
                if raw_path.exists():
                    for lno, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), 1):
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                raw_boxes.append(
                                    BBox(
                                        int(parts[0]),
                                        float(parts[1]),
                                        float(parts[2]),
                                        float(parts[3]),
                                        float(parts[4]),
                                        lno,
                                    )
                                )
                            except ValueError:
                                continue
                left = draw_yolo_boxes(
                    im,
                    raw_boxes,
                    classes,
                    title=f"RAW ({len(raw_boxes)} boxes)" if raw_path.exists() else "RAW (NO .txt LABEL)",
                )
                cur_boxes = stem_to_boxes.get(stem, [])
                right = draw_yolo_boxes(
                    im,
                    cur_boxes,
                    classes,
                    title=f"CURATED ({len(cur_boxes)} boxes) | {img_name}",
                )

            left_r = resize_max_width(left, 640)
            right_r = resize_max_width(right, 640)
            panel = Image.new("RGB", (left_r.width + right_r.width, max(left_r.height, right_r.height)))
            panel.paste(left_r, (0, 0))
            panel.paste(right_r, (left_r.width, 0))
            out_name = f"compare_{idx}_{changelog[img_name]['status']}.jpg"
            panel.save(samples_dir / out_name, format="JPEG", quality=88)
            saved.append(out_name)

    # 2. Representative samples from each split
    for sp in ("train", "val", "test"):
        split_paths = split_map.get(sp, [])
        ranked = sorted(split_paths, key=lambda p: len(stem_to_boxes.get(p.stem, [])), reverse=True)
        for idx, img_path in enumerate(ranked[:2], start=1):
            with Image.open(img_path) as im:
                boxes = stem_to_boxes.get(img_path.stem, [])
                vis = draw_yolo_boxes(
                    im,
                    boxes,
                    classes,
                    title=f"{sp.upper()} | {img_path.name} ({len(boxes)} boxes)",
                )
                vis = resize_max_width(vis, 1280)
                out_name = f"{sp}_sample_{idx}.jpg"
                vis.save(samples_dir / out_name, format="JPEG", quality=88)
                saved.append(out_name)

    return saved


def _summarize_issues_by_code(raw_audit: DatasetAuditResult) -> dict[str, list[str]]:
    """Group human-readable `file[:line]` locations by issue code from `raw_audit`."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for issue in raw_audit.issues:
        loc = f"`{issue.file}:{issue.line}`" if issue.line is not None else f"`{issue.file}`"
        grouped[issue.code].append(loc)
    return grouped


def _format_issue_comment(locations: list[str], empty_msg: str, max_show: int = 4) -> str:
    if not locations:
        return empty_msg
    head = ", ".join(locations[:max_show])
    extra = f" и ещё {len(locations) - max_show}" if len(locations) > max_show else ""
    return f"Файлы: {head}{extra}"


def _write_dataset_report(
    report_path: Path,
    raw_audit: DatasetAuditResult,
    active_audit: DatasetAuditResult | None,
    used_curated: bool,
    used_scene_groups: bool,
    excluded_missing: list[Path],
    groups: list[list[Path]],
    split_map: dict[str, list[Path]],
    stem_to_boxes: dict[str, list[BBox]],
    converted_rgba: list[str],
    classes: list[str],
    seed: int,
    ratios: tuple[float, float, float],
    dhash_threshold: int,
    changelog: dict[str, Any] | None,
    sample_files: list[str],
) -> None:
    """Write Markdown audit and split report from the collected dataset statistics."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    raw_issues_by_code = Counter(i.code for i in raw_audit.issues)
    issue_locs = _summarize_issues_by_code(raw_audit)
    raw_cls_counts = Counter(b.class_id for img in raw_audit.images for b in img.boxes)

    cur_cls_counts: dict[str, Counter[int]] = {sp: Counter() for sp in ("train", "val", "test")}
    cur_img_cls_counts: dict[str, Counter[int]] = {sp: Counter() for sp in ("train", "val", "test")}
    all_areas: list[float] = []
    for sp, paths in split_map.items():
        for p in paths:
            boxes = stem_to_boxes.get(p.stem, [])
            seen_cids = set()
            for b in boxes:
                cur_cls_counts[sp][b.class_id] += 1
                seen_cids.add(b.class_id)
                all_areas.append(b.width * b.height)
            for cid in seen_cids:
                cur_img_cls_counts[sp][cid] += 1

    total_boxes_curated = sum(sum(c.values()) for c in cur_cls_counts.values())
    total_imgs_exported = sum(len(v) for v in split_map.values())
    multi_groups = [g for g in groups if len(g) > 1]
    stem_to_split = {p.stem: sp for sp, paths in split_map.items() for p in paths}
    corrupt_count = sum(1 for i in raw_audit.images if i.is_corrupt)

    lines: list[str] = [
        "# Отчёт по подготовке и аудиту датасета",
        "",
        "## 1. Общая сводка",
        "",
        f"- Входные изображения оставлены без изменений; активная разметка: "
        f"{'`data/curated_labels/`' if used_curated else 'сырые аннотации (режим санитизации)'}.",
        f"- Группировка сцен: `dHash <= {dhash_threshold}` и шаблоны сессий камер"
        + (", дополненные `data/scene_groups.yaml`." if used_scene_groups else "."),
        f"- Разбиение: `seed={seed}`, целевые доли `train/val/test = {ratios[0]:.2f}/{ratios[1]:.2f}/{ratios[2]:.2f}`.",
        f"- Входных изображений: {len(raw_audit.images)} (битых: {corrupt_count}); экспортировано: {total_imgs_exported}"
        + (f" (исключено без `.txt`: {len(excluded_missing)})" if excluded_missing else "") + ".",
        f"- К `RGB` приведено не-RGB файлов: {len(converted_rgba)}.",
        f"- Валидных боксов: {sum(len(i.boxes) for i in raw_audit.images)} в сырых данных и {total_boxes_curated} в итоговой выборке.",
        "",
        "## 2. Аудит исходной разметки",
        "",
        "| Тип проблемы | Количество | Файлы |",
        "|---|---:|---|",
        f"| Отсутствует `.txt` файл (`missing_label`) | {raw_issues_by_code.get('missing_label', 0)} | {_format_issue_comment(issue_locs['missing_label'], 'Отсутствующих файлов нет')} |",
        f"| Неизвестный класс (`unknown_class`) | {raw_issues_by_code.get('unknown_class', 0)} | {_format_issue_comment(issue_locs['unknown_class'], 'Неизвестных классов нет')} |",
        f"| Выход за `[0, 1]` больше 2% (`bbox_suspicious`) | {raw_issues_by_code.get('bbox_suspicious', 0)} | {_format_issue_comment(issue_locs['bbox_suspicious'], 'Подозрительных смещений нет')} |",
        f"| Небольшой выход за `[0, 1]` (`bbox_out_of_bounds`) | {raw_issues_by_code.get('bbox_out_of_bounds', 0)} | {_format_issue_comment(issue_locs['bbox_out_of_bounds'], 'Мелких выходов за край нет')} |",
        f"| Некорректные координаты или формат (`invalid_coords`/`malformed_line`) | {raw_issues_by_code.get('invalid_coords', 0) + raw_issues_by_code.get('malformed_line', 0)} | {_format_issue_comment(issue_locs['invalid_coords'] + issue_locs['malformed_line'], 'Ошибок формата и размера нет')} |",
        f"| Битые изображения (`corrupt_image`) | {raw_issues_by_code.get('corrupt_image', 0)} | {_format_issue_comment(issue_locs['corrupt_image'], f'Все {len(raw_audit.images)} изображений декодируются без ошибок')} |",
        f"| Дубликаты боксов (`duplicate_box`) | {raw_issues_by_code.get('duplicate_box', 0)} | {_format_issue_comment(issue_locs['duplicate_box'], 'Повторов с IoU >= 0.99 нет')} |",
        "",
    ]

    if active_audit is not None:
        lines.extend([
            f"Повторная проверка активной разметки: `{len(active_audit.errors)} errors, {len(active_audit.warnings)} warnings`.",
            "",
        ])

    if changelog:
        status_counts = Counter(entry["status"] for entry in changelog.values())
        lines.extend([
            "## 3. Сводка журнала правок (`data/annotation_changelog.json`)",
            "",
        ])
        for st_name, cnt in sorted(status_counts.items()):
            lines.append(f"- `{st_name}`: {cnt} кадров")
        lines.append("")

    lines.extend([
        "## 4. Защита от утечки данных (кластеры сцен)",
        "",
        f"Найдено {len(multi_groups)} многокадровых кластеров ({sum(len(g) for g in multi_groups)} из {total_imgs_exported} экспортированных изображений). "
        "Каждый кластер целиком помещён в один сплит:",
        "",
        "| # | Размер | Сплит | Файлы кластера |",
        "|---:|---:|---|---|",
    ])
    for idx, grp in enumerate(multi_groups, start=1):
        sp = stem_to_split.get(grp[0].stem, "-")
        names_str = ", ".join(f"`{p.name}`" for p in grp)
        lines.append(f"| {idx} | {len(grp)} | `{sp}` | {names_str} |")

    lines.extend([
        "",
        "## 5. Итоговый сплит и распределение классов",
        "",
        "| Сплит | Изображений | Доля | Боксов |",
        "|---|---:|---:|---:|",
    ])
    for sp in ("train", "val", "test"):
        n_im = len(split_map[sp])
        frac = (n_im / total_imgs_exported * 100.0) if total_imgs_exported else 0.0
        n_bx = sum(cur_cls_counts[sp].values())
        lines.append(f"| `{sp}` | {n_im} | {frac:.1f}% | {n_bx} |")

    lines.extend([
        "",
        "| ID | Класс | Сырые валидные | Итого | Train (боксы/кадры) | Val (боксы/кадры) | Test (боксы/кадры) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for cid, cname in enumerate(classes):
        tr_b, tr_i = cur_cls_counts["train"][cid], cur_img_cls_counts["train"][cid]
        va_b, va_i = cur_cls_counts["val"][cid], cur_img_cls_counts["val"][cid]
        te_b, te_i = cur_cls_counts["test"][cid], cur_img_cls_counts["test"][cid]
        tot_b = tr_b + va_b + te_b
        lines.append(
            f"| {cid} | `{cname}` | {raw_cls_counts[cid]} | {tot_b} | "
            f"{tr_b} / {tr_i} | {va_b} / {va_i} | {te_b} / {te_i} |"
        )

    if all_areas:
        arr_a = np.asarray(all_areas, dtype=np.float64)
        zero_classes = [cname for cid, cname in enumerate(classes) if (cur_cls_counts["train"][cid] + cur_cls_counts["val"][cid] + cur_cls_counts["test"][cid]) == 0]
        rare_classes = [cname for cid, cname in enumerate(classes) if 0 < (cur_cls_counts["train"][cid] + cur_cls_counts["val"][cid] + cur_cls_counts["test"][cid]) <= 10]
        lines.extend([
            "",
            "## 6. Размеры объектов и технические детали",
            "",
            f"- Нормализованная площадь бокса (`w * h`): min `{arr_a.min():.6f}`, median `{np.median(arr_a):.5f}`, mean `{arr_a.mean():.5f}`, max `{arr_a.max():.4f}`.",
            f"- Мелкие объекты (площадь меньше 0.002): {int(np.sum(arr_a < 0.002))} из {len(all_areas)} ({np.mean(arr_a < 0.002) * 100:.1f}%).",
            f"- Редкие классы (не более 10 боксов): {', '.join(f'`{c}`' for c in rare_classes) if rare_classes else 'нет'}"
            + (f"; отсутствуют: {', '.join(f'`{c}`' for c in zero_classes)}" if zero_classes else "") + ".",
            f"- Цветовые каналы и пути: {len(converted_rgba)} файлов приведены к `RGB`, остальные `RGB`-файлы скопированы побайтово с исходными именами в UTF-8.",
            "",
        ])

    if sample_files:
        lines.extend([
            "## 7. Примеры визуализаций (`reports/samples/`)",
            "",
        ])
        for sf in sample_files:
            lines.append(f"- [{sf}](samples/{sf})")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    reports_dir: Path = Path("reports"),
    labels_dir: Path | None = None,
    scene_groups_path: Path | None = None,
    raw_only: bool = False,
    seed: int = 42,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    dhash_threshold: int = 12,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit, split without leakage, export RGB data + dataset.yaml, and write report/manifest."""
    repo_root = (root_dir or find_repo_root(input_dir)).resolve()
    raw_images_dir = input_dir / "images"
    raw_labels_dir = input_dir / "labels"
    classes_path = input_dir / "classes.txt"

    raw_audit = audit_dataset(raw_images_dir, raw_labels_dir, classes_path)
    classes = raw_audit.classes
    raw_image_paths = [img.image_path for img in raw_audit.images if not img.is_corrupt]
    raw_stems = {p.stem for p in raw_image_paths}

    used_curated = False
    active_labels_dir = labels_dir
    if active_labels_dir is None and not raw_only:
        candidate_curated = repo_root / "data" / "curated_labels"
        if candidate_curated.is_dir():
            curated_stems = {
                p.stem
                for p in candidate_curated.iterdir()
                if p.is_file() and p.suffix.lower() == ".txt"
            }
            if curated_stems == raw_stems:
                active_labels_dir = candidate_curated
                used_curated = True

    active_audit: DatasetAuditResult | None = None
    stem_to_boxes: dict[str, list[BBox]] = {}
    excluded_missing: list[Path] = []
    exportable_images: list[Path] = []

    if active_labels_dir is not None:
        active_audit = audit_dataset(raw_images_dir, active_labels_dir, classes_path)
        if not active_audit.is_valid:
            raise ValueError(
                f"Active labels in {active_labels_dir} failed validation with {len(active_audit.errors)} errors."
            )
        for img_audit in active_audit.images:
            if img_audit.is_corrupt:
                continue
            exportable_images.append(img_audit.image_path)
            stem_to_boxes[img_audit.image_path.stem] = img_audit.boxes
    else:
        for img_audit in raw_audit.images:
            if img_audit.is_corrupt:
                continue
            if not img_audit.has_label_file:
                excluded_missing.append(img_audit.image_path)
                continue
            exportable_images.append(img_audit.image_path)
            stem_to_boxes[img_audit.image_path.stem] = _sanitize_raw_boxes(
                img_audit.label_path,
                num_classes=len(classes),
            )

    manual_groups: list[list[str]] | None = None
    used_scene_groups = False
    resolved_groups_path = scene_groups_path
    if resolved_groups_path is None and not raw_only:
        candidate_sg = repo_root / "data" / "scene_groups.yaml"
        if candidate_sg.is_file():
            resolved_groups_path = candidate_sg
    if resolved_groups_path is not None:
        loaded_groups = load_scene_groups(resolved_groups_path)
        export_names = {p.name for p in exportable_images}
        matching_groups = [g for g in loaded_groups if sum(1 for n in g if n in export_names) >= 2]
        if matching_groups:
            manual_groups = matching_groups
            used_scene_groups = True

    stem_to_classes = {
        stem: Counter(b.class_id for b in boxes)
        for stem, boxes in stem_to_boxes.items()
    }
    groups = build_leakage_groups(
        exportable_images,
        dhash_threshold=dhash_threshold,
        manual_groups=manual_groups,
    )
    split_map = split_groups_stratified(
        groups,
        stem_to_classes=stem_to_classes,
        ratios=split_ratios,
        seed=seed,
    )

    _clean_output_dir(output_dir)
    converted_rgba: list[str] = []

    for sp, paths in split_map.items():
        img_out_dir = output_dir / "images" / sp
        lbl_out_dir = output_dir / "labels" / sp
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for src_img in paths:
            dst_img = img_out_dir / src_img.name
            if export_image_rgb(src_img, dst_img):
                converted_rgba.append(src_img.name)

            boxes = stem_to_boxes.get(src_img.stem, [])
            lbl_lines = [
                f"{b.class_id} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}"
                for b in boxes
            ]
            (lbl_out_dir / f"{src_img.stem}.txt").write_text(
                "\n".join(lbl_lines) + ("\n" if lbl_lines else ""),
                encoding="utf-8",
            )

    dataset_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": {idx: name for idx, name in enumerate(classes)},
    }
    save_yaml(dataset_yaml, output_dir / "dataset.yaml")

    changelog: dict[str, Any] | None = None
    changelog_path = repo_root / "data" / "annotation_changelog.json"
    if used_curated and changelog_path.exists():
        changelog = json.loads(changelog_path.read_text(encoding="utf-8"))

    manifest_path = reports_dir / "manifest.sha256"
    manifest_entries = write_manifest(output_dir, manifest_path, root_dir=repo_root)
    sample_files = _render_samples(
        samples_dir=reports_dir / "samples",
        split_map=split_map,
        stem_to_boxes=stem_to_boxes,
        raw_labels_dir=raw_labels_dir,
        classes=classes,
        changelog=changelog,
    )
    _write_dataset_report(
        report_path=reports_dir / "dataset_report.md",
        raw_audit=raw_audit,
        active_audit=active_audit,
        used_curated=used_curated,
        used_scene_groups=used_scene_groups,
        excluded_missing=excluded_missing,
        groups=groups,
        split_map=split_map,
        stem_to_boxes=stem_to_boxes,
        converted_rgba=converted_rgba,
        classes=classes,
        seed=seed,
        ratios=split_ratios,
        dhash_threshold=dhash_threshold,
        changelog=changelog,
        sample_files=sample_files,
    )

    return {
        "used_curated_labels": used_curated,
        "used_scene_groups": used_scene_groups,
        "exported_images": len(exportable_images),
        "excluded_missing": len(excluded_missing),
        "num_groups": len(groups),
        "split_counts": {sp: len(paths) for sp, paths in split_map.items()},
        "manifest_files": len(manifest_entries),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset, split, report, and SHA-256 manifest.")
    parser.add_argument("--input", type=Path, required=True, help="Input raw dataset directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output processed dataset directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Optional override labels directory.")
    parser.add_argument("--scene-groups", type=Path, default=None, help="Optional scene_groups.yaml override.")
    parser.add_argument("--raw-only", action="store_true", help="Do not auto-apply data/curated_labels/.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Reports output directory.")
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test ratios.",
    )
    parser.add_argument("--dhash-threshold", type=int, default=12, help="Max dHash Hamming distance to group.")
    parser.add_argument(
        "--verify-manifest",
        action="store_true",
        help="Read-only check of existing --output against reports/manifest.sha256.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.verify_manifest:
        manifest_path = args.reports_dir / "manifest.sha256"
        ok, errs = verify_manifest(manifest_path, args.output)
        if ok:
            print(f"Manifest verification PASSED ({manifest_path}).")
            return 0
        print(f"Manifest verification FAILED ({len(errs)} issues):")
        for err in errs:
            print(f"  - {err}")
        return 1

    summary = prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        reports_dir=args.reports_dir,
        labels_dir=args.labels_dir,
        scene_groups_path=args.scene_groups,
        raw_only=args.raw_only,
        seed=args.seed,
        split_ratios=tuple(args.split_ratios),
        dhash_threshold=args.dhash_threshold,
    )
    print(
        f"Prepared dataset: exported={summary['exported_images']}, "
        f"curated={summary['used_curated_labels']}, scene_groups={summary['used_scene_groups']}, "
        f"groups={summary['num_groups']}, splits={summary['split_counts']}, "
        f"manifest_files={summary['manifest_files']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
