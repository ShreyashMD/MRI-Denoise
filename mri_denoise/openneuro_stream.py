import os
import tempfile
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import nibabel as nib
import requests


class OpenNeuroStreamer:
    """Stream NIfTI files directly from the OpenNeuro S3 bucket."""

    def __init__(self, dataset_id: str = "ds002306") -> None:
        self.dataset_id = dataset_id
        self.base_url = f"https://s3.amazonaws.com/openneuro.org/{dataset_id}"

    def check_url_exists(self, url: str) -> bool:
        """Return True if the given URL exists."""
        try:
            response = requests.head(url, timeout=15)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def stream_nii_file(self, url: str) -> Optional[np.ndarray]:
        """Download a NIfTI file to a temporary path and return its data."""
        print(f"Streaming: {os.path.basename(url)}")
        temp_path: Optional[str] = None
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                temp_path = tmp.name
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
            img = nib.load(temp_path)
            return img.get_fdata(dtype=np.float32)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error streaming file: {exc}")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def load_files_for_chunk(
        self,
        subjects: Iterable[str],
        num_files: int,
        seen_urls: Optional[Set[str]] = None,
    ) -> Tuple[List[np.ndarray], Set[str]]:
        """Stream multiple files for a chunk."""
        data: List[np.ndarray] = []
        urls = seen_urls or set()

        for subj in subjects:
            if len(data) >= num_files:
                break
            for run in range(1, 6):
                if len(data) >= num_files:
                    break
                url = (
                    f"{self.base_url}/{subj}/func/"
                    f"{subj}_task-training_run-{run:02d}_bold.nii.gz"
                )
                if url in urls:
                    continue
                if self.check_url_exists(url):
                    array = self.stream_nii_file(url)
                    if array is not None:
                        data.append(array)
                        urls.add(url)
        return data, urls


def main() -> None:
    """CLI entry point for streaming a small dataset chunk."""
    import argparse

    parser = argparse.ArgumentParser(description="Stream data from OpenNeuro")
    parser.add_argument(
        "--dataset-id",
        default="ds002306",
        help="OpenNeuro dataset identifier",
    )
    parser.add_argument(
        "--subjects", type=int, default=10, help="Number of subjects to scan"
    )
    parser.add_argument(
        "--files", type=int, default=20, help="Number of files to stream"
    )
    parser.add_argument("--output", required=True, help="Path to save the output .npy")
    args = parser.parse_args()

    pool = [f"sub-{i:02d}" for i in range(1, args.subjects + 1)]
    streamer = OpenNeuroStreamer(args.dataset_id)
    chunk_data, _ = streamer.load_files_for_chunk(pool, args.files, set())
    if chunk_data:
        np.save(args.output, np.array(chunk_data, dtype=np.float32))
        print(f"Saved {len(chunk_data)} files to {args.output}")
    else:
        print("No data was streamed.")


if __name__ == "__main__":
    main()
