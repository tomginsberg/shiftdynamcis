import pytorch_lightning as pl
import torch

from typing import Callable, Optional, Sequence

from torch.utils.data import Dataset, DataLoader


class LabelCollector(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.predicted_labels = []
        self.true_labels = []

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat: torch.Tensor = self.model(x).argmax(dim=1)
        self.predicted_labels.extend(y_hat.cpu().tolist())
        self.true_labels.extend(y.cpu().tolist())

    def compute_accuracy(self):
        return torch.eq(torch.tensor(self.predicted_labels), torch.tensor(self.true_labels)).float().mean().item()

    def get_labels(self, mode='predicted'):
        if mode == 'predicted':
            return self.predicted_labels
        elif mode == 'true':
            return self.true_labels
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def reset(self):
        self.predicted_labels = []
        self.true_labels = []


def infer_labels(model: torch.nn.Module, dataset: Dataset | Sequence[Dataset], batch_size: Optional[int] = None,
                 num_workers=64, gpus=[0],
                 verbose=True, return_accuracy=False):
    tr = pl.Trainer(gpus=gpus, max_epochs=1, enable_model_summary=False, logger=False)
    if isinstance(dataset, Dataset):
        dataset = [dataset]

    results = []
    for d in dataset:
        dl = DataLoader(d, batch_size=batch_size if batch_size else len(dataset),
                        num_workers=num_workers,
                        drop_last=False)

        lc = LabelCollector(model=model)
        tr.validate(lc, dl, verbose=False)
        if verbose:
            print(f'Inferred labels for {len(d)} samples. Accuracy: {lc.compute_accuracy():.3f}')
        results.append(lc.get_labels(mode='predicted'))
        if return_accuracy:
            results[-1] = [results[-1], (lc.compute_accuracy())]

    if len(dataset) == 1:
        if return_accuracy:
            return results[0][0], results[0][1]
        return results[0]
    return results
