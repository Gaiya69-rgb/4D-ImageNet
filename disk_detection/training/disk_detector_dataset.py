import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class BraggDataset(Dataset):
    def __init__(self, config, data_path_list):
        self.config = config
        self.data_files = []

        # 1. Collect all .pkl files
        for path in data_path_list:
            if not os.path.isdir(path):
                print(f"Warning: Dataset path does not exist: {path}")
                continue
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".pkl"):
                        self.data_files.append(os.path.join(root, file))

        if not self.data_files:
            raise FileNotFoundError(f"No .pkl files found in the specified paths: {data_path_list}")

        print(f"Found {len(self.data_files)} total .pkl files.")

        # 2. Load all data into memory
        self.all_items = []
        print("Loading data into memory... this may take a moment.")
        for file_path in tqdm(self.data_files, desc="Loading .pkl files"):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.all_items.extend(data["item_list"])

        print(f"Total number of simulation samples loaded: {len(self.all_items)}")

    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, idx):
        item = self.all_items[idx]
        image = item["dp"].astype(np.float16)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)  # Add channel dimension: 1, H, W
        image = torch.from_numpy(image).half()
        max_disks = self.config.get("disk_count", 64)
        qi_raw = item["qi_list"]
        qj_raw = item["qj_list"]
        valid_mask = qi_raw != -1
        qi_valid = qi_raw[valid_mask]
        qj_valid = qj_raw[valid_mask]
        disk_array = np.full((max_disks, 2), -1.0, dtype=np.float32)
        num_points = min(len(qi_valid), max_disks)
        if num_points > 0:
            disk_array[:num_points, 0] = qj_valid[:num_points]
            disk_array[:num_points, 1] = qi_valid[:num_points]
        disk = torch.from_numpy(disk_array)
        return image.half(), disk