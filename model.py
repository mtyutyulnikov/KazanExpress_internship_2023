from pytorch_lightning import LightningModule
from torch import nn
from ast import Dict
import torch.optim as optim
import torch
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
)
from transformers import AutoTokenizer, AutoModel
from torchmetrics.classification import F1Score


class ClassifierNet(LightningModule):
    def __init__(
        self,
        model_hparams: Dict,
        ce_weights=None,
    ):
        """Initialize a classification model
        Args:
            model_hparams (Dict): keys:  num_classes: int
            optimizer_name (str): Adam or SGD
            optimizer_hparams (Dict): lr: float, weight_decay: float
        """
        super().__init__()
        self.save_hyperparameters()

        num_classes = model_hparams["num_classes"]

        if ce_weights is not None:
            self.loss_module = nn.CrossEntropyLoss(weight=ce_weights)
        else:
            self.loss_module = nn.CrossEntropyLoss()

        self.softmax = nn.Softmax(dim=-1)

        # Not used in final version
        # base_img_model = resnet18(ResNet18_Weights.DEFAULT) # img_emb = 1280
        # base_img_model = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT)  # img_emb = 512
        # self.image_model = nn.Sequential(*list(base_img_model.children())[:-1])  # output batch_size*img_emb*1*1

        self.text_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")

        self.title_model = AutoModel.from_pretrained(
            "cointegrated/rubert-tiny2"
        )  # output batch_size*312

        self.classifier = nn.Sequential(
            nn.Linear(312 + 739, 512),  # 739 tags features
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        self.f1_metric = F1Score(
            "multiclass", num_classes=num_classes, average="weighted"
        )

    def forward(self, title, description, tags_features):
        text = [f"{t}. {d}" for t, d in zip(title, description)]

        text_tokenized = self.text_tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )

        text_model_output = self.title_model(
            **{k: v.to(self.title_model.device) for k, v in text_tokenized.items()}
        )

        text_embed = text_model_output.last_hidden_state[:, 0, :]
        text_embed = torch.nn.functional.normalize(text_embed)
        features = torch.cat((text_embed, tags_features), dim=1)

        logits = self.classifier(features)
        return logits

    def predict_step(self, batch, batch_idx):
        x, labels, idx = batch

        logits = self.forward(x["title"], x["description"], x["tags_features"])

        preds = logits.argmax(dim=-1)
        probs = self.softmax(logits)
        return preds, labels, logits, idx, probs

    def training_step(self, batch, batch_idx):
        x, labels, idx = batch

        logits = self.forward(x["title"], x["description"], x["tags_features"])
        loss = self.loss_module(logits, labels)

        acc = (logits.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1_metric(logits.argmax(dim=-1), labels)

        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss)
        self.log("train_f1", f1)

        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, labels, idx = batch

        logits = self.forward(x["title"], x["description"], x["tags_features"])

        acc = (logits.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1_metric(logits.argmax(dim=-1), labels)

        self.log("val_acc", acc)
        self.log("val_f1", f1)

        loss = self.loss_module(logits, labels)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            [
                {"params": self.classifier.parameters(), "lr": 1e-3},
                # {"params": self.image_model.parameters(), "lr": 1e-4},
                {"params": self.text_model.parameters(), "lr": 1e-4},
            ],
        )

        lmbda = lambda epoch: 0.8**epoch
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        return [optimizer], [scheduler]
