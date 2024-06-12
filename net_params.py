import torch
from torch import nn


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    
    def forward(self, X):    
        linear = torch.matmul(X, torch.sigmoid(self.weight.data))
        return linear


class AutoTomo(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(AutoTomo, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            MyLinear(output_size, input_size),
        )

    def forward(self, x):
        h = self.encoder(x)
        x = self.decoder(h)
        return h, x
