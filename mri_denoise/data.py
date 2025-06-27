import os
import subprocess


def download_ds005239(destination: str = "data") -> None:
    """Download OpenNeuro dataset ds005239 version 1.0.1."""
    os.makedirs(destination, exist_ok=True)
    cmd = [
        "openneuro",
        "download",
        "--dataset",
        "ds005239",
        "--version",
        "1.0.1",
        "--destination",
        destination,
    ]
    subprocess.check_call(cmd)
