import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    2D-CNN on EEG as a 3-scale image (3, 512, 512).

    Input: (B, 3, 512, 512) — 3 temporal crops (10s / 25s / 50s) as RGB channels,
    built by _signals_to_image in dataset.py.
    """

    def __init__(
        self,
        backbone: str = 'efficientnet_b5',
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        super().__init__()
        self.net = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            global_pool='',
        )
        n_feat = self.net.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_feat, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.drop(x)
        return F.log_softmax(self.fc(x), dim=1)
