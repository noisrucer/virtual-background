import torch
import cv2
import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, img_dir, matting_dir, csv_df, mode='train', transform=None):
        self.img_dir = img_dir
        self.matting_dir = matting_dir
        self.transform = transform

        self.df = csv_df


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = row['img_path']
        matting_path = row['matting_path']

        # Load image
        img_bgr = cv2.imread(img_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load matting
        matting = cv2.imread(matting_path, cv2.IMREAD_UNCHANGED)
        matting_alpha = matting[:, :, 3]

        # Mask
        mask = (matting_alpha != 0) * 1
        mask = mask.astype(np.uint8)

        # Data Augmentation
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img, mask
