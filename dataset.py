from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning import LightningDataModule
import pandas as pd
import numpy as np
from img_preprocess import get_train_transforms, get_test_transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import torch


class ItemsDataset(Dataset):
    def __init__(
        self, parquet_file_path, images_path, img_transforms, is_labeled
    ) -> None:
        super().__init__()

        self.df = pd.read_parquet(parquet_file_path)
        self.images_path = images_path
        self.img_transforms = img_transforms
        self.is_labeled = is_labeled
        self.tags_columns = self.df.columns[
            self.df.columns.str.startswith("title_has_tag_")
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = np.asarray(Image.open(self.images_path / f'{row["product_id"]}.jpg'))
        image = self.img_transforms(image=image)["image"]

        result = {
            # "image": image,
            "title": row["title"],
            "description": row["description"],
            "tags_features": torch.tensor(
                row[self.tags_columns].to_numpy().astype(np.float32)
            ),
        }

        label = row["target"] if self.is_labeled else -1
        return result, label, index


class ItemsDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 16,
        k_fold_idx=0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_fold_idx = k_fold_idx

    def setup(self, stage=None):
        train_path = Path("images/train")
        test_path = Path("images/test")

        self.train_dataset = ItemsDataset(
            "preprocessed_train.parquet",
            images_path=train_path,
            img_transforms=get_train_transforms(),
            is_labeled=True,
        )
        self.val_dataset = ItemsDataset(
            "preprocessed_train.parquet",
            images_path=train_path,
            img_transforms=get_test_transforms(),
            is_labeled=True,
        )

        train_df = pd.read_parquet("preprocessed_train.parquet")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        k_folds = list(skf.split(train_df, train_df["target"]))

        self.train_indices, self.val_indices = (
            k_folds[self.k_fold_idx][0],
            k_folds[self.k_fold_idx][1],
        )

        self.train_dataset = Subset(self.train_dataset, self.train_indices)
        self.val_dataset = Subset(self.val_dataset, self.val_indices)

        train_df = train_df.iloc[self.train_indices]
        self.train_dataset_balance = torch.tensor(
            train_df["target"].value_counts().sort_index().to_numpy()
        )

        self.test_dataset = ItemsDataset(
            "preprocessed_test.parquet",
            images_path=test_path,
            img_transforms=get_test_transforms(),
            is_labeled=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
