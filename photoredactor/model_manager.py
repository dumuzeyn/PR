from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
from typing import Callable
from urllib.request import Request, urlopen
import zipfile


ENGINE_RELEASE_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
ProgressCallback = Callable[[str, int, int], None]


class ModelManagerError(RuntimeError):
    pass


class DownloadCancelled(ModelManagerError):
    pass


@dataclass(frozen=True)
class LocalModelSpec:
    model_id: str
    name: str
    description: str
    filename: str
    url: str
    source_url: str
    license_name: str
    license_url: str
    size: int
    sha256: str
    recommended_steps: int = 24
    recommended_cfg: float = 7.0

    @property
    def size_gb(self) -> float:
        return self.size / (1024 ** 3)


MODEL_CATALOG = (
    LocalModelSpec(
        model_id="realistic-vision-51-inpaint",
        name="Realistic Vision 5.1 Inpainting",
        description="Быстрая фотомодель для предметов, портретов и естественного заполнения выделенной области.",
        filename="Realistic_Vision_V5.1_fp16-no-ema-inpainting.safetensors",
        url=(
            "https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/"
            "Realistic_Vision_V5.1_fp16-no-ema-inpainting.safetensors"
        ),
        source_url="https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE",
        license_name="CreativeML Open RAIL-M",
        license_url="https://huggingface.co/spaces/CompVis/stable-diffusion-license",
        size=2_132_679_782,
        sha256="dbcf026dd6498122e239b06fa83a2cecca9e3ce2a39f50bef000594bd2b28ad2",
        recommended_steps=22,
        recommended_cfg=7.0,
    ),
    LocalModelSpec(
        model_id="sd15-inpaint",
        name="Stable Diffusion 1.5 Inpainting",
        description="Официальная универсальная модель для точного заполнения по маске и расширения изображения.",
        filename="sd-v1-5-inpainting.ckpt",
        url=(
            "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/resolve/main/"
            "sd-v1-5-inpainting.ckpt"
        ),
        source_url="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting",
        license_name="CreativeML Open RAIL-M",
        license_url="https://huggingface.co/spaces/CompVis/stable-diffusion-license",
        size=4_265_437_280,
        sha256="c6bbc15e3224e6973459ba78de4998b80b50112b0ae5b5c67113d56b4e366b19",
        recommended_steps=26,
        recommended_cfg=7.5,
    ),
)
MODEL_BY_ID = {item.model_id: item for item in MODEL_CATALOG}
LCM_ACCELERATOR = LocalModelSpec(
    model_id="lcm-lora-sdv1-5",
    name="LCM-LoRA SD 1.5",
    description="Открытый ускоритель генерации Stable Diffusion 1.5 до 2–8 шагов.",
    filename="pytorch_lora_weights.safetensors",
    url=(
        "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/"
        "cf2fced511dbe7e26c8d1d397e728fbab875db4b/pytorch_lora_weights.safetensors"
    ),
    source_url="https://huggingface.co/latent-consistency/lcm-lora-sdv1-5",
    license_name="OpenRAIL++",
    license_url="https://huggingface.co/latent-consistency/lcm-lora-sdv1-5",
    size=134_621_556,
    sha256="8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4",
    recommended_steps=6,
    recommended_cfg=1.5,
)


@dataclass(frozen=True)
class HardwareProfile:
    gpu_names: tuple[str, ...]
    nvidia_vram_mb: int
    ram_mb: int
    recommended_backend: str

    @property
    def summary(self) -> str:
        gpu = ", ".join(self.gpu_names) or "только CPU"
        vram = f", {self.nvidia_vram_mb // 1024} ГБ VRAM" if self.nvidia_vram_mb else ""
        return f"{gpu}{vram}; RAM {max(1, round(self.ram_mb / 1024))} ГБ"


def _windows_gpu_names() -> tuple[str, ...]:
    if os.name != "nt":
        return ()

    class DisplayDevice(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32), ("DeviceName", ctypes.c_wchar * 32),
            ("DeviceString", ctypes.c_wchar * 128), ("StateFlags", ctypes.c_uint32),
            ("DeviceID", ctypes.c_wchar * 128), ("DeviceKey", ctypes.c_wchar * 128),
        ]

    names: list[str] = []
    index = 0
    while True:
        device = DisplayDevice()
        device.cb = ctypes.sizeof(device)
        if not ctypes.windll.user32.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
            break
        name = device.DeviceString.strip()
        if name and name not in names and not name.lower().startswith("microsoft basic"):
            names.append(name)
        index += 1
    return tuple(names)


def _nvidia_vram_mb() -> int:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return 0
    try:
        result = subprocess.run(
            [executable, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return max(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _ram_mb() -> int:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys // (1024 * 1024))
    return 0


def detect_hardware() -> HardwareProfile:
    names = _windows_gpu_names()
    lowered = " ".join(names).lower()
    backend = "cuda" if "nvidia" in lowered else "vulkan" if names else "cpu"
    return HardwareProfile(names, _nvidia_vram_mb(), _ram_mb(), backend)


def _sha256(path: Path, progress: ProgressCallback | None = None, cancel: threading.Event | None = None) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            if cancel is not None and cancel.is_set():
                raise DownloadCancelled("Загрузка отменена")
            digest.update(chunk)
            done += len(chunk)
            if progress:
                progress("Проверка SHA-256", done, total)
    return digest.hexdigest()


def _download(
    url: str, target: Path, expected_size: int, expected_sha256: str,
    progress: ProgressCallback | None, cancel: threading.Event | None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset == expected_size:
        if _sha256(partial, progress, cancel).lower() != expected_sha256.lower():
            partial.unlink(missing_ok=True)
            raise ModelManagerError("SHA-256 загруженного файла не совпадает с опубликованным значением")
        os.replace(partial, target)
        return
    headers = {"User-Agent": "PhotoRedactor/1.0"}
    if 0 < offset < expected_size:
        headers["Range"] = f"bytes={offset}-"
    elif offset > expected_size:
        offset = 0
        partial.unlink(missing_ok=True)
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        append = offset > 0 and getattr(response, "status", 200) == 206
        if not append:
            offset = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as output:
            done = offset
            while chunk := response.read(1024 * 1024):
                if cancel is not None and cancel.is_set():
                    raise DownloadCancelled("Загрузка отменена")
                output.write(chunk)
                done += len(chunk)
                if progress:
                    progress("Загрузка", done, expected_size)
    if partial.stat().st_size != expected_size:
        raise ModelManagerError("Размер загруженного файла не совпадает с опубликованным размером")
    if _sha256(partial, progress, cancel).lower() != expected_sha256.lower():
        partial.unlink(missing_ok=True)
        raise ModelManagerError("SHA-256 загруженного файла не совпадает с опубликованным значением")
    os.replace(partial, target)


class ModelStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.engines_dir = self.root / "engines"

    def model_path(self, model: LocalModelSpec | str) -> Path:
        spec = MODEL_BY_ID[model] if isinstance(model, str) else model
        return self.models_dir / spec.model_id / spec.filename

    def model_installed(self, model: LocalModelSpec | str) -> bool:
        spec = MODEL_BY_ID[model] if isinstance(model, str) else model
        path = self.model_path(spec)
        manifest = path.parent / "manifest.json"
        if not path.is_file() or path.stat().st_size != spec.size or not manifest.is_file():
            return False
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            return data.get("sha256") == spec.sha256 and data.get("size") == spec.size
        except (OSError, ValueError):
            return False

    def accelerator_path(self) -> Path:
        return self.model_path(LCM_ACCELERATOR)

    def accelerator_installed(self) -> bool:
        return self.model_installed(LCM_ACCELERATOR)

    def install_model(
        self, model: LocalModelSpec | str, progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
    ) -> Path:
        spec = MODEL_BY_ID[model] if isinstance(model, str) else model
        path = self.model_path(spec)
        if self.model_installed(spec):
            return path
        _download(spec.url, path, spec.size, spec.sha256, progress, cancel)
        (path.parent / "manifest.json").write_text(
            json.dumps(asdict(spec), ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return path

    def remove_model(self, model: LocalModelSpec | str) -> None:
        path = self.model_path(model).parent.resolve()
        root = self.models_dir.resolve()
        if root not in path.parents:
            raise ModelManagerError("Недопустимый путь модели")
        shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _engine_asset_suffix(backend: str) -> str:
        values = {
            "cuda": "bin-win-cuda12-x64.zip",
            "vulkan": "bin-win-vulkan-x64.zip",
            "cpu": "bin-win-cpu-x64.zip",
        }
        if backend not in values:
            raise ModelManagerError(f"Неизвестный движок: {backend}")
        return values[backend]

    def _latest_engine_assets(self, backend: str) -> list[dict[str, object]]:
        request = Request(ENGINE_RELEASE_API, headers={"Accept": "application/vnd.github+json", "User-Agent": "PhotoRedactor/1.0"})
        with urlopen(request, timeout=30) as response:
            release = json.loads(response.read().decode("utf-8"))
        suffixes = [self._engine_asset_suffix(backend)]
        if backend == "cuda":
            suffixes.append("cudart-sd-bin-win-cu12-x64.zip")
        result: list[dict[str, object]] = []
        for suffix in suffixes:
            asset = next((item for item in release.get("assets", []) if str(item.get("name", "")).endswith(suffix)), None)
            if not asset:
                raise ModelManagerError(f"В актуальном выпуске stable-diffusion.cpp нет сборки {suffix}")
            digest = str(asset.get("digest", ""))
            if not digest.startswith("sha256:"):
                raise ModelManagerError("Выпуск движка не содержит проверяемую SHA-256 подпись")
            result.append({
                "tag": str(release["tag_name"]), "name": str(asset["name"]),
                "url": str(asset["browser_download_url"]), "size": int(asset["size"]),
                "sha256": digest.split(":", 1)[1], "backend": backend,
            })
        return result

    def engine_executable(self, backend: str) -> Path | None:
        state_path = self.engines_dir / f"{backend}.json"
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            path = (self.engines_dir / str(data["relative_executable"])).resolve()
            if self.engines_dir.resolve() in path.parents and path.is_file():
                return path
        except (OSError, KeyError, ValueError):
            pass
        return None

    def install_engine(
        self, backend: str, progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
    ) -> Path:
        existing = self.engine_executable(backend)
        if existing:
            return existing
        assets = self._latest_engine_assets(backend)
        release_tag = str(assets[0]["tag"])
        downloads = self.engines_dir / "downloads"
        archives: list[Path] = []
        for asset in assets:
            archive = downloads / str(asset["name"])
            _download(str(asset["url"]), archive, int(asset["size"]), str(asset["sha256"]), progress, cancel)
            archives.append(archive)
        destination = self.engines_dir / release_tag / backend
        staging = Path(tempfile.mkdtemp(prefix="extract-", dir=self.engines_dir))
        try:
            for archive in archives:
                with zipfile.ZipFile(archive) as bundle:
                    root = staging.resolve()
                    for member in bundle.infolist():
                        target = (staging / member.filename).resolve()
                        if root != target and root not in target.parents:
                            raise ModelManagerError("Архив движка содержит небезопасный путь")
                    bundle.extractall(staging)
            executable = next(staging.rglob("sd-cli.exe"), None)
            if executable is None:
                raise ModelManagerError("В архиве движка отсутствует sd-cli.exe")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(staging), str(destination))
            executable = destination / executable.relative_to(staging)
            relative = executable.resolve().relative_to(self.engines_dir.resolve())
            self.engines_dir.mkdir(parents=True, exist_ok=True)
            (self.engines_dir / f"{backend}.json").write_text(
                json.dumps({"tag": release_tag, "backend": backend, "assets": assets, "relative_executable": str(relative)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            for archive in archives:
                archive.unlink(missing_ok=True)
            return executable.resolve()
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def ensure_ready(
        self, model_id: str, backend: str, progress: ProgressCallback | None = None,
        cancel: threading.Event | None = None,
    ) -> tuple[Path, Path]:
        model = MODEL_BY_ID.get(model_id)
        if model is None:
            raise ModelManagerError("Выбранная локальная модель отсутствует в каталоге")
        engine = self.install_engine(backend, progress, cancel)
        weights = self.install_model(model, progress, cancel)
        self.install_model(LCM_ACCELERATOR, progress, cancel)
        return engine, weights


__all__ = [
    "DownloadCancelled", "HardwareProfile", "LCM_ACCELERATOR", "LocalModelSpec", "MODEL_BY_ID", "MODEL_CATALOG",
    "ModelManagerError", "ModelStore", "detect_hardware",
]
