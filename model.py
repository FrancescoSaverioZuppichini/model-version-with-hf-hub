from torch import nn


class BoringModel(nn.Sequential):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.in_dense = nn.Linear(2, hidden_size)
        self.in_act = nn.ReLU()
        self.out_dense = nn.Linear(hidden_size, 4)
