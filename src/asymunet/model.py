"""A small U-Net: 3 input maps (depth, ICP asymmetry, validity) -> 1 logit map."""

from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 16, depth: int = 3) -> None:
        super().__init__()
        widths = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        c = in_channels
        for w in widths[:-1]:
            self.down.append(_block(c, w))
            c = w
        self.bottom = _block(c, widths[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.up.append(nn.ConvTranspose2d(w * 2, w, 2, stride=2))
            self.dec.append(_block(w * 2, w))
        self.head = nn.Conv2d(widths[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)
