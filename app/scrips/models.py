"""
Module: models.py

Description:
Contains a collection of PyTorch model architectures used for regression and classification tasks
in the MLB pitcher prediction pipeline.

Models included:
- DeepModel_res: Deep neural network for regression.
- ResNetModel: Residual neural network with skip connections for regression.
- ShallowModel: Lightweight model for fast predictions.
- DeepModel_class: Deep model for both classification and regression.
- TransformerModel: Transformer-based architecture for tabular data.

Author: Lance Santerre
"""
import torch
import torch.nn as nn


class DeepModel_class(nn.Module):
    def __init__(self, n_cont, output_type="classification"):
        super().__init__()
        self.output_type = output_type

        self.model = nn.Sequential(
            nn.Linear(n_cont, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() if output_type == "classification" else nn.Identity(),
        )

    def forward(self, x_cont):
        return self.model(x_cont)


class DeepModel_res(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


class ResNetModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2)
        )

        # Residual Block 1 (256 → 256)
        self.block1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2))

        # Residual Block 2 (256 → 128) with projection for residual
        self.block2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2))
        self.residual_proj2 = nn.Linear(256, 128)

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_layer(x)
        residual1 = x

        x = self.block1(x)
        x = x + residual1  # Residual connection 1

        residual2 = x
        x = self.block2(x)
        x = x + self.residual_proj2(residual2)  # Projected residual connection 2

        return self.output_layer(x)


class ShallowModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


class DeepModel_class(nn.Module):
    def __init__(self, n_cont, output_type="regression"):
        super().__init__()
        layers = []
        sizes = [n_cont, 512, 256, 128, 64, 32]
        for i in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[i], sizes[i + 1]),
                nn.BatchNorm1d(sizes[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.4),
            ]
        layers += [nn.Linear(sizes[-1], 1)]
        if output_type == "classification":
            layers += [nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, cont_x):
        return self.net(cont_x)


class TransformerModel(nn.Module):
    def __init__(self, n_cont, output_type="regression"):
        super().__init__()
        self.linear_proj = nn.Linear(n_cont, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out = nn.Linear(128, 1)
        self.act = nn.Sigmoid() if output_type == "classification" else nn.Identity()

    def forward(self, cont_x):
        # cont_x shape: [B, n_cont]
        x = self.linear_proj(cont_x).unsqueeze(1)  # → [B, 1, 128]
        x = self.transformer(x).squeeze(1)  # → [B, 128]
        return self.act(self.out(x))  # → [B, 1]
