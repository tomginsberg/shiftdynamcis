from typing import Optional

import torch
import torch.nn.functional as F


class HighEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None, use_random_vectors=False, alpha=None):
        super(HighEntropyLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha

    def forward(self, logits, labels, mask):
        return high_entropy_loss(logits, labels,
                                 weight=self.weight,
                                 mask=mask,
                                 alpha=self.alpha)


def high_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, alpha: Optional[float] = None,
                      weight=None) -> torch.Tensor:
    """

    :param logits: (batch_size, num_classes) tensor of logits
    :param labels: (batch_size,) tensor of labels
    :param mask: (batch_size,) mask
    :param alpha: (float) weight of q samples
    :param weight:  (torch.Tensor) weight for each sample_data, default=None do not apply weighting
    :return: (tensor, float) the disagreement cross entropy loss
    """
    mask: torch.Tensor = (mask == 0)
    if mask.all():
        # if all labels are positive, then use the standard cross entropy loss
        return F.cross_entropy(logits, labels)

    if alpha is None:
        alpha = 1 / (1 + (~mask).float().sum())

    num_classes = logits.shape[1]

    q_logits, q_labels = logits[~mask], labels[~mask]
    ce_n = -q_logits.sum(dim=1) / num_classes + torch.logsumexp(q_logits, dim=1)

    if torch.isinf(ce_n).any() or torch.isnan(ce_n).any():
        raise RuntimeError('NaN or Infinite loss encountered for ce-q')

    if (~mask).all():
        return (ce_n * alpha).mean()

    p_logits, p_labels = logits[mask], labels[mask]
    ce_p = F.cross_entropy(p_logits, p_labels, reduction='none', weight=weight)
    return torch.cat([ce_n * alpha, ce_p]).mean()


if __name__ == '__main__':
    import unittest


    class TestHighEntropyLoss(unittest.TestCase):
        logits = torch.tensor([[1., 2., 3.], [4., 5., 6.], [1., 1.2, .98], [-3., 4., 2.]])
        labels = torch.tensor([2, 0, 1, 1])

        def __init__(self, *args, **kwargs):
            super(TestHighEntropyLoss, self).__init__(*args, **kwargs)

        def test_high_entropy_loss(self):
            mask = torch.tensor([True, True, True, True])
            self.assertTrue(torch.allclose(high_entropy_loss(self.logits, self.labels, mask),
                                           F.cross_entropy(self.logits, self.labels)))

        def test_high_entropy_loss_mask(self):
            mask = torch.tensor([True, True, False, True])
            self.assertTrue(
                torch.allclose(high_entropy_loss(self.logits, self.labels, mask, alpha=1), torch.tensor(1.7266)))
