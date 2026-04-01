import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeNet-5 with configurable Batch Normalization (conv layers) and Dropout (FC layers).

    Configurations tested:
        1. use_dropout=True,  use_bn=True   -> FC-Dropout + CONV-BN
        2. use_dropout=True,  use_bn=False  -> FC-Dropout + CONV-noBN
        3. use_dropout=False, use_bn=True   -> FC-noDropout + CONV-BN
        4. use_dropout=False, use_bn=False  -> FC-noDropout + CONV-noBN

    Architecture (MNIST 28x28 input):
        CONV1: 1->6,  5x5  -> [BN] -> ReLU -> MaxPool2d(2)  => 6x12x12
        CONV2: 6->16, 5x5  -> [BN] -> ReLU -> MaxPool2d(2)  => 16x4x4
        Flatten: 256
        FC1: 256->120 -> ReLU -> [Dropout]
        FC2: 120->84  -> ReLU -> [Dropout]
        FC3: 84->10
    """

    def __init__(self, use_dropout: bool = True, use_bn: bool = True, dropout_rate: float = 0.5):
        super().__init__()
        self.use_dropout = use_dropout
        self.use_bn = use_bn

        def conv_block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=5)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.ReLU(), nn.MaxPool2d(kernel_size=2)]
            return layers

        self.conv = nn.Sequential(
            *conv_block(1, 6),
            *conv_block(6, 16),
        )

        fc_layers = [
            nn.Linear(256, 120), nn.ReLU(),
        ]
        if use_dropout:
            fc_layers.append(nn.Dropout(dropout_rate))
        fc_layers += [nn.Linear(120, 84), nn.ReLU()]
        if use_dropout:
            fc_layers.append(nn.Dropout(dropout_rate))
        fc_layers.append(nn.Linear(84, 10))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
