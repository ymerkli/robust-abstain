import torch.nn as nn


class MiniNet(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_classes: int,
        ):
        super(MiniNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.out = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.out(x)
        return x


def get_mininet(**kwargs):
    return MiniNet(**kwargs)

mininet = get_mininet