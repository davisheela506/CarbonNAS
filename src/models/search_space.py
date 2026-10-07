import torch.nn as nn


class SearchSpaceCNN(nn.Module):
    def __init__(
        self,
        channels=(32, 64),
        kernel_size=3,
        use_pooling=True,
        num_classes=10,
    ):
        super().__init__()

        layers = []
        in_channels = 3

        for out_channels in channels:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())

            if use_pooling:
                layers.append(nn.MaxPool2d(2))

            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        spatial_size = 32

        if use_pooling:
            spatial_size //= 2 ** len(channels)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_channels * spatial_size * spatial_size,
                128,
            ),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)