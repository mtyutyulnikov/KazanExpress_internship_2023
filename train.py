from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import random
import numpy as np
from model import ClassifierNet
import torch
from dataset import ItemsDataModule
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    dm = ItemsDataModule(batch_size=3, num_workers=20, k_fold_idx=0)
    dm.setup()

    # Попытка исправить баланс классов за счет весов в лоссе. Стало только хуже
    # train_classes_count = dm.train_dataset_balance
    # train_weights = 1/train_classes_count
    # train_weights /= train_weights.sum()

    default_net = ClassifierNet(
        {"num_classes": 730},
        ce_weights=None,
    )  # 730 классов после очистки датасета от классов с менее чем 10 семплами

    # WandB url : wandb.ai/mtyutyulnikov/KazanExpress
    wandb_logger = WandbLogger(log_model=True, project="KazanExpress")

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=30,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                monitor="val_f1",
                mode="max",
            ),
            EarlyStopping(monitor="val_f1", mode="max", patience=5),
        ],
        default_root_dir="models",
        logger=wandb_logger,
    )

    trainer.fit(
        default_net,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )
