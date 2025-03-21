import torch
import torch.nn as nn

import torch
import torch.nn as nn

class RegressionNN(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(RegressionNN, self).__init__()
        layer_sizes = [input_size] + hidden_sizes + [1]
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        return self.layers[-1](x)
