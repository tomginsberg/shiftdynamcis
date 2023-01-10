import os

import torch

from utils.config import Config


def resnet18_trained_on_cifar10(path=None, domain_classifier=False):
    if path is None:
        cfg = Config()
        path = os.path.join(cfg.checkpoint_path(), 'cifar', 'base_model.pt')

    model = torch.load(path)

    if domain_classifier:
        model = _to_binary_output(model)
    return model


def _to_binary_output(model):
    assert hasattr(model.model, 'fc'), 'Model must have a fully connected layer named fc'
    model.model.fc = torch.nn.Linear(model.model.fc.in_features, 2)
    return model
