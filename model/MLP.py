import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim, dims=[256, 128, 64], dropout=0):
        super().__init__()
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(nn.PReLU(num_parameters=1, init=0.25))
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)