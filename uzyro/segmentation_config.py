from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class SegmentationConfig:
    model_dir: Path
    subject_model: Path
    sky_model: Path
    input_size: int = 320
    prefer_gpu: bool = True

    @classmethod
    def from_environment(cls) -> "SegmentationConfig":
        default_root = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "UZYRO" / "models" / "segmentation"
        root = Path(os.environ.get("UZYRO_SEGMENTATION_DIR", default_root))
        subject = Path(os.environ.get("UZYRO_SUBJECT_MODEL", root / "subject.onnx"))
        if not subject.is_file() and (root / "foreground.onnx").is_file():
            subject = root / "foreground.onnx"
        sky = Path(os.environ.get("UZYRO_SKY_MODEL", root / "sky.onnx"))
        try:
            size = max(128, min(1024, int(os.environ.get("UZYRO_SEGMENTATION_SIZE", "320"))))
        except ValueError:
            size = 320
        prefer_gpu = os.environ.get("UZYRO_SEGMENTATION_GPU", "1").strip().lower() not in {"0", "false", "off"}
        return cls(root, subject, sky, size, prefer_gpu)


__all__ = ["SegmentationConfig"]
