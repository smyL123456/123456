import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.AIDE import AIDE_2BRANCH


class DummyTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Module()
        self.head.global_pool = nn.Identity()
        self.head.flatten = nn.Identity()

    def forward(self, x):
        b = x.shape[0]
        return torch.randn(b, 3072, 8, 8, device=x.device)


class DummyClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.trunk = DummyTrunk()


def main():
    model = AIDE_2BRANCH(
        resnet_path=None,
        convnext_path=None,
        openclip_model=DummyClipModel(),
    )
    x = torch.randn(2, 5, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 2), y.shape
    print("ok")


if __name__ == "__main__":
    main()
