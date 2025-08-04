import os, time, requests, tempfile, numpy as np, nibabel as nib
from typing import List, Optional

class OpenNeuroStreamer:
    def __init__(self, dataset_id: str = "ds002306"):
        self.dataset_id = dataset_id
        self.base_url = f"https://s3.amazonaws.com/openneuro.org/{dataset_id}"

    def check_url_exists(self, url: str) -> bool:
        try:
            response = requests.head(url, timeout=15)
            return response.status_code == 200
        except:
            return False

    def stream_nii_file(self, url: str) -> Optional[np.ndarray]:
        print(f"Streaming: {os.path.basename(url)}")
        temp_path = None
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
                temp_path = temp_file.name
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
            img = nib.load(temp_path)
            return img.get_fdata(dtype=np.float32)
        except Exception as e:
            print(f"Error streaming file: {e}")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def load_selected_files(self):
        selected_files = [
            ("sub-01", range(1, 6)), 
            ("sub-02", range(1, 6)),  
            ("sub-03", range(1, 3))   
        ]

        data = []
        for subject, runs in selected_files:
            for run in runs:
                url = f"{self.base_url}/{subject}/func/{subject}_task-training_run-{run:02d}_bold.nii.gz"
                if self.check_url_exists(url):
                    d = self.stream_nii_file(url)
                    if d is not None:
                        data.append(d)
        return data


# === RUN THE DOWNLOADER ===
print("NOTEBOOK 1A: Downloading Specific Files (sub-01 to sub-03, run-02)")
streamer = OpenNeuroStreamer()
chunk_data = streamer.load_selected_files()
if chunk_data:
    np.save("/kaggle/working/fmri_dataset_chunk_1.npy", np.array(chunk_data, dtype=np.float32))
    print("\nChunk saved successfully.")
