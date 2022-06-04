import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

from .dataset import CustomDataset

def get_transform(mode='train'):
    if mode == 'train':
        transform = A.Compose([
        A.HorizontalFlip(0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
        ])

    elif mode == 'val':
        transform = A.Compose([
            A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

    return transform


def get_dataloaders(img_dir, matting_dir, csv_path, batch_size=16, shuffle=True, num_workers=2, train_split=0.8):
    print(">>> Constructing DataLoaders...")
    # Transform
    train_transform = get_transform('train')
    val_transform = get_transform('val')

    # Split train/val
    df = pd.read_csv(csv_path)
    num_data = len(df)
    num_train = int(num_data * train_split)

    train_df = df.iloc[0: num_train]
    val_df = df.iloc[num_train:]
    val_df.reset_index(inplace=True)
    val_df = val_df.drop(columns=['index'])

    # Construct datasets
    train_dataset = CustomDataset(img_dir, matting_dir, train_df, mode='train', transform=train_transform)
    val_dataset = CustomDataset(img_dir, matting_dir, val_df, mode='val', transform=val_transform)

    # Construct dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers
    )

    return train_loader, val_loader

