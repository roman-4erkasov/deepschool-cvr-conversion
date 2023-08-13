import os
import typing as tp


import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils import preprocess_image, MAX_UINT8

class PlanetDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_folder: str,
        target_image_size=(MAX_UINT8, MAX_UINT8)
    ):
        self.df = df
        self.image_folder = image_folder

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_folder, f'{row.image_name}.jpg')
        labels = np.array(row.drop(["image_name"]), dtype="float32")
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        data = {'image': image, 'labels': labels}
        return data['image'], data['labels']

    def __len__(self) -> int:
        return len(self.df)
