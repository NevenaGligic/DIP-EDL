"""
PyTorch port of CNN_Model_1D from:
  https://github.com/superdreamliner/LAMOST-Spectra-Classifier

Architecture
------------
- n_branches parallel Conv1d branches with different kernel sizes
- Each branch: num_blocks × (Conv1d → BatchNorm1d → ReLU → MaxPool1d(3, stride=3))
- All branch outputs concatenated and flattened  ← .feature is stored here
- Dense head: [128, 64, 32] with BatchNorm, optional Dropout, then output layer

Default parameters reproduce the published architecture:
  5 branches, kernel sizes [3,5,7,9,11], 3 conv blocks per branch, 32 filters,
  dense layers [128, 64, 32].
"""

import torch
import torch.nn as nn


class Conv1dMultiBranchNet(nn.Module):

    def __init__(
        self,
        input_shape: int,
        num_classes: int = 2,
        kernel_sizes: tuple = (3, 5, 7, 9, 11),
        num_blocks: int = 3,
        num_filters: int = 32,
        dense_units: tuple = (128, 64, 32),
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── parallel conv branches ────────────────────────────────────────────
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            layers = []
            in_ch = 1
            for _ in range(num_blocks):
                layers += [
                    nn.Conv1d(in_ch, num_filters, kernel_size=ks,
                              padding=ks // 2, bias=False),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=3, stride=3),
                ]
                in_ch = num_filters
            self.branches.append(nn.Sequential(*layers))

        # ── feature dimension after conv (needed by GDA) ─────────────────────
        conv_out_len = input_shape
        for _ in range(num_blocks):
            conv_out_len = conv_out_len // 3          # each MaxPool1d(3, stride=3)
        self.flat_dim = len(kernel_sizes) * num_filters * conv_out_len

        # ── dense head ────────────────────────────────────────────────────────
        dense_layers = []
        in_dim = self.flat_dim
        for units in dense_units:
            dense_layers += [
                nn.Linear(in_dim, units),
                nn.BatchNorm1d(units),
                nn.ReLU(inplace=True),
            ]
            if dropout_rate > 0.0:
                dense_layers.append(nn.Dropout(p=dropout_rate))
            in_dim = units
        dense_layers.append(nn.Linear(in_dim, num_classes))
        self.dense = nn.Sequential(*dense_layers)

        # Populated on every forward pass so DAEDL's fit_gda can read it.
        self.feature = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        branch_outs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outs, dim=1)   # (B, n_branches * num_filters, L')
        x = x.view(x.size(0), -1)           # (B, flat_dim)
        self.feature = x.clone().detach()   # flat conv output (kept for reference)
        penultimate = self.dense[:-1](x)    # (B, dense_units[-1]) — last hidden layer
        self.penultimate_feature = penultimate.clone().detach()
        return self.dense[-1](penultimate)
