import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.npr_feature import build_npr_feature_extractor


def main():
    model = build_npr_feature_extractor(freeze=True)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        feat = model(x)
    assert feat.shape == (2, 512), feat.shape
    assert all(not p.requires_grad for p in model.parameters())
    print("ok")


if __name__ == "__main__":
    main()
