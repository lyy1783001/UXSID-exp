from torch import nn
from fuxictr.pytorch.torch_utils import get_loss

__all__ = [
    "DinLoss",
]


class DinLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss_fn = get_loss(loss)

    def forward(self, y_pred, y_true, reduction="mean"):
        return self.loss_fn(y_pred, y_true, reduction=reduction)
