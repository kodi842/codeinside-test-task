"""Shared visualization helpers for dataset reports and TensorBoard samples."""

from __future__ import annotations

from PIL import Image, ImageDraw

from src.validate_labels import BBox

# Distinct high-contrast RGB tuples for the 11 traffic classes
CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 230, 118),    # 0: bicycle
    (56, 189, 248),   # 1: car
    (255, 193, 7),    # 2: motorcycle
    (168, 85, 247),   # 3: bus
    (249, 115, 22),   # 4: truck
    (234, 88, 12),    # 5: lorry
    (20, 184, 166),   # 6: trailer
    (59, 130, 246),   # 7: police
    (236, 72, 153),   # 8: ambulance
    (239, 68, 68),    # 9: fire_truck
    (163, 230, 53),   # 10: pedestrian
)

_CYR_TO_LAT = str.maketrans({
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "E", "Ж": "Zh",
    "З": "Z", "И": "I", "Й": "Y", "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O",
    "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "Kh", "Ц": "Ts",
    "Ч": "Ch", "Ш": "Sh", "Щ": "Shch", "Ъ": "", "Ы": "Y", "Ь": "", "Э": "E", "Ю": "Yu", "Я": "Ya",
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
})


def ascii_safe(text: str) -> str:
    """Transliterate Cyrillic to ASCII for Pillow's default bitmap font."""
    return text.translate(_CYR_TO_LAT).encode("ascii", errors="replace").decode("ascii")


def resize_max_width(img: Image.Image, max_width: int = 1280) -> Image.Image:
    """Downscale image preserving aspect ratio when wider than `max_width`."""
    if img.width <= max_width:
        return img
    new_h = max(1, int(round(img.height * (max_width / img.width))))
    return img.resize((max_width, new_h), Image.Resampling.LANCZOS)


def draw_yolo_boxes(
    image: Image.Image,
    boxes: list[BBox],
    classes: list[str],
    title: str | None = None,
) -> Image.Image:
    """Draw colored YOLO bounding boxes and an optional ASCII-safe header onto an RGB copy."""
    canvas = image.convert("RGB")
    w, h = canvas.size
    draw = ImageDraw.Draw(canvas)
    line_w = max(2, min(4, w // 400))

    for b in boxes:
        color = CLASS_COLORS[b.class_id % len(CLASS_COLORS)]
        x1 = max(0.0, (b.x_center - b.width / 2.0) * w)
        y1 = max(0.0, (b.y_center - b.height / 2.0) * h)
        x2 = min(float(w - 1), (b.x_center + b.width / 2.0) * w)
        y2 = min(float(h - 1), (b.y_center + b.height / 2.0) * h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        label = classes[b.class_id] if 0 <= b.class_id < len(classes) else f"cls_{b.class_id}"
        ty = max(2.0, y1 - 13)
        draw.rectangle([x1, ty, min(float(w), x1 + len(label) * 7 + 6), ty + 13], fill=color)
        draw.text((x1 + 3, ty + 1), label, fill=(0, 0, 0))

    if title:
        safe_title = ascii_safe(title)
        draw.rectangle([0, 0, w, 22], fill=(18, 24, 38))
        draw.text((8, 5), safe_title, fill=(255, 255, 255))
    return canvas
