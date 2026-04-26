# Pre-flight and runtime disk-space checks for long-running collectors.
import shutil
from pathlib import Path


def free_bytes(path):
    return shutil.disk_usage(path).free


def precheck(target_path, min_free_gb):
    free = free_bytes(target_path)
    if free < min_free_gb * 1024**3:
        raise RuntimeError(
            f"Disk too small at {target_path}: need {min_free_gb}GB, have {free/1024**3:.2f}GB"
        )
    return free


class DiskFuse:
    def __init__(self, target_path, floor_gb):
        self.target_path = Path(target_path)
        self.floor = floor_gb * 1024**3

    def check(self):
        f = free_bytes(self.target_path)
        if f < self.floor:
            raise RuntimeError(
                f"Disk fuse triggered at {self.target_path}: "
                f"{f/1024**3:.2f}GB free < {self.floor/1024**3:.2f}GB floor"
            )
        return f
