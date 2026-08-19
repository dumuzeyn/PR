from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class SecurityLimits:
    max_image_file_bytes: int = 8 * 1024**3
    max_project_file_bytes: int = 32 * 1024**3
    max_image_pixels: int = 268_435_456
    max_dimension: int = 100_000
    max_layers: int = 4_096
    max_archive_entries: int = 250_000
    max_archive_member_bytes: int = 512 * 1024**2
    max_archive_total_bytes: int = 32 * 1024**3
    max_archive_ratio: int = 1_000
    max_tile_bytes: int = 128 * 1024**2
    max_tiles_per_array: int = 262_144
    max_json_bytes: int = 64 * 1024**2
    max_action_bytes: int = 256 * 1024**2
    max_action_steps: int = 10_000
    max_batch_jobs: int = 2_000
    max_batch_items: int = 100_000
    max_metadata_bytes: int = 8 * 1024**2
    max_icc_bytes: int = 16 * 1024**2
    max_string_chars: int = 65_536
    max_name_chars: int = 512
    max_json_depth: int = 48
    max_json_nodes: int = 1_000_000
    max_plugin_message_bytes: int = 8 * 1024**2
    max_process_output_bytes: int = 2 * 1024**2

    def memory_budget_bytes(self) -> int:
        fallback = 4 * 1024**3
        if os.name != "nt":
            return fallback
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
                    ("total", ctypes.c_uint64), ("available", ctypes.c_uint64),
                    ("total_page", ctypes.c_uint64), ("available_page", ctypes.c_uint64),
                    ("total_virtual", ctypes.c_uint64), ("available_virtual", ctypes.c_uint64),
                    ("available_extended", ctypes.c_uint64),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return max(1024**3, min(8 * 1024**3, int(status.available * 0.45)))
        except (AttributeError, OSError):
            pass
        return fallback


LIMITS = SecurityLimits()

__all__ = ["LIMITS", "SecurityLimits"]
