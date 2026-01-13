import torch
import torch.nn as nn

class CCRNet(nn.Module):
    """
    Small MLP for tabular binary classification
    Input: (batch, 11)
    Output (batch, 1) logits (raw scores)
    """

    def __init__(self, input_dim: int = 11, hidden_dim: int = 16):
        super().__init__()

        # Layers and activations
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # construct the forward pass through layers
        return self.net(x)