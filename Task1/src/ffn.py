import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)
