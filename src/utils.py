"""Shared utilities for dataset validation, preparation, and experiment runs."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_repo_root(start: Path | None = None) -> Path:
    """Locate repository root by walking upward for `.git` or `configs/`."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (
            (candidate / "src").is_dir() and (candidate / "test_data").is_dir()
        ):
            return candidate
    return current


def load_classes(classes_path: Path) -> list[str]:
    """Load non-empty class names from a YOLO `classes.txt` file."""
    lines = classes_path.read_text(encoding="utf-8").splitlines()
    classes = [line.strip() for line in lines if line.strip()]
    if not classes:
        raise ValueError(f"No classes found in {classes_path}")
    return classes


def set_seed(seed: int) -> None:
    """Set deterministic RNG seeds for Python, NumPy, and PyTorch (if installed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(requested: str = "auto") -> str:
    """Resolve training or inference device (`auto` selects `cuda`, then `mps`, then `cpu`)."""
    req = str(requested).strip().lower()
    if req and req != "auto":
        return req
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def get_git_info(repo_root: Path | None = None) -> dict[str, Any]:
    """Return git commit, branch, and dirty flag without failing outside git."""
    cwd = str((repo_root or find_repo_root()).resolve())

    def _run_git(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return None

    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    return {
        "commit": commit or "unknown",
        "commit_short": commit[:8] if commit else "unknown",
        "branch": branch or "unknown",
        "is_dirty": bool(status) if status is not None else None,
    }


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def save_yaml(data: dict[str, Any], path: Path) -> None:
    """Write a YAML mapping to disk with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
