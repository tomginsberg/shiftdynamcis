import warnings
from glob import glob
from os.path import join
from typing import Optional, Callable

import attr
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from nn.losses.entropy import HighEntropyLoss
import pandas as pd


@attr.s(auto_attribs=True)
class PQStruct:
    accuracy: float = None
    p_logits: torch.Tensor = None
    q_logits: torch.Tensor = None
    p_accuracy: float = None
    q_accuracy: float = None
    p_labels: torch.Tensor = None
    q_labels: torch.Tensor = None

    def to_dict(self, minimal: bool = False):
        if minimal:
            return {
                'accuracy': self.accuracy,
                'p_accuracy': self.p_accuracy,
                'q_accuracy': self.q_accuracy,
            }
        return attr.asdict(self)

    def dataframe(self, minimal: bool = False):
        return pd.DataFrame([self.to_dict(minimal=minimal)])


class PQModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, alpha=None, optim_func: Callable = None, num_classes=10):
        super().__init__()
        self.model = model
        self.loss = HighEntropyLoss(alpha=alpha)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)

        self.test_struct = PQStruct()
        self.optim_func = optim_func

    def set_alpha(self, alpha):
        self.loss.alpha = alpha

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        logits = self.model(x)
        loss = self.loss(logits=logits, labels=y, mask=mask)
        return loss

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_id):
        x, y, mask = batch
        logits = self.model(x)

        p_mask = mask == 1
        q_mask = mask == 2

        return dict(p_logits=logits[p_mask],
                    q_logits=logits[q_mask],
                    p_labels=y[p_mask],
                    q_labels=y[q_mask])

    def test_epoch_end(self, outputs):
        p_logits = torch.cat([x['p_logits'] for x in outputs], dim=0)
        q_logits = torch.cat([x['q_logits'] for x in outputs], dim=0)

        p_labels = torch.cat([x['p_labels'] for x in outputs], dim=0)
        q_labels = torch.cat([x['q_labels'] for x in outputs], dim=0)

        p_acc = (p_logits.argmax(dim=1) == p_labels).float().mean()
        q_acc = (q_logits.argmax(dim=1) == q_labels).float().mean()
        acc = (torch.cat([p_logits, q_logits], dim=0).argmax(dim=1) == torch.cat([p_labels, q_labels],
                                                                                 dim=0)).float().mean()

        self.test_struct = PQStruct(
            p_logits=p_logits.cpu(),
            q_logits=q_logits.cpu(),
            accuracy=acc.cpu().item(),
            p_accuracy=p_acc.cpu().item(),
            q_accuracy=q_acc.cpu().item(),
            p_labels=p_labels.cpu(),
            q_labels=q_labels.cpu(),
        )

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        logits = self.model(x)
        self.val_acc(logits.argmax(dim=1), y)

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute())
        self.val_acc.reset()

    def configure_optimizers(self):
        if self.optim_func is None:
            return torch.optim.Adam(self.model.parameters())
        return self.optim_func(self.model.parameters())


class PQEnsemble(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, detectors: Optional[torch.nn.ModuleList] = None):
        super().__init__()
        if detectors is None:
            self.ensemble = torch.nn.ModuleList([base_model])
        else:
            self.ensemble = torch.nn.ModuleList([base_model, *detectors])

    def add_detector(self, detector: torch.nn.Module):
        self.ensemble.append(detector)

    @staticmethod
    def load_from_checkpoint(base_model: torch.nn.Module,
                             checkpoint_directory: str,
                             model_cls: pl.LightningModule,
                             sorting_func: Optional[Callable[[str], int]] = None) -> 'PQEnsemble':
        checkpoints = glob(join(checkpoint_directory, '*.ckpt'))
        if len(checkpoints) == 0:
            warnings.warn(f'No checkpoints found in {checkpoint_directory}, using base model only')
            return PQEnsemble(base_model, None)

        if sorting_func is not None:
            checkpoints.sort(key=sorting_func)

        return PQEnsemble(
            base_model,
            torch.nn.ModuleList(
                [model_cls.load_from_checkpoint(c) for c in checkpoints]
            )
        )

    def forward(self, x):
        return torch.stack([model(x) for model in self.ensemble], dim=1)


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.wait = 0
        assert mode in ['min', 'max']
        self.mode = mode

    def update(self, metric):
        if self.best is None:
            self.best = metric
            return False
        if self.mode == 'min':
            if metric < self.best - self.min_delta:
                self.best = metric
                self.wait = 0
                return False
            else:
                self.wait += 1
                return self.wait >= self.patience
        elif self.mode == 'max':
            if metric > self.best + self.min_delta:
                self.best = metric
                self.wait = 0
                return False
            else:
                self.wait += 1
                return self.wait >= self.patience
