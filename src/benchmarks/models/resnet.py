import torch.nn as nn

from torch import Tensor
from typing import Type

class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv3d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels*self.expansion,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels*self.expansion)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out


class ResNet(nn.Module):
    def __init__(
            self,
            mode: str,
            input_channels: int,
            latent_dim: int,
            dropout: float,
            device: str,
            num_layers: int = 18,
            output_dim: int = 100,
            block: Type[BasicBlock] = BasicBlock
    ) -> None:
        super(ResNet, self).__init__()

        self.mode = mode
        self.latent_dim = latent_dim

        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock`
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 16
        # All ResNets (18 to 152) contain a Conv3d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv3d(
            in_channels=input_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        in_channels_per_layer = [self.in_channels, 32, 64, 128]
        self.layer1 = self._make_layer(block, in_channels_per_layer[0], layers[0])
        self.layer2 = self._make_layer(block, in_channels_per_layer[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, in_channels_per_layer[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, in_channels_per_layer[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.dropout = nn.Dropout(p=dropout)
        self.latent_head = nn.Linear(128, latent_dim)

        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(latent_dim, output_dim)
        )

    def _make_layer(
            self,
            block: Type[BasicBlock],
            out_channels: int,
            blocks: int,
            stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm3d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        features = self.avgpool(x).squeeze()

        features = self.dropout(features)
        features = self.latent_head(features)

        smoothed_features = features

        if self.mode == "encoder":
            return smoothed_features, features

        elif self.mode == "regressor":
            x = self.regressor(smoothed_features)
            return x, features

        elif self.mode == "classifier":

            x = self.classifier(smoothed_features)
            x = nn.functional.log_softmax(x, dim=-1)
            return x, features


class ResNet_64ch(nn.Module):
    def __init__(
            self,
            mode: str,
            input_channels: int,
            latent_dim: int,
            dropout: float,
            device: str,
            num_layers: int = 18,
            output_dim: int = 100,
            block: Type[BasicBlock] = BasicBlock
    ) -> None:
        super(ResNet_64ch, self).__init__()

        self.mode = mode
        self.latent_dim = latent_dim

        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock`
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv3d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv3d(
            in_channels=input_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        in_channels_per_layer = [self.in_channels, 128, 256, 512]
        self.layer1 = self._make_layer(block, in_channels_per_layer[0], layers[0])
        self.layer2 = self._make_layer(block, in_channels_per_layer[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, in_channels_per_layer[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, in_channels_per_layer[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.dropout = nn.Dropout(p=dropout)
        self.latent_head = nn.Linear(512, latent_dim)

        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(latent_dim, output_dim)
        )

    def _make_layer(
            self,
            block: Type[BasicBlock],
            out_channels: int,
            blocks: int,
            stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm3d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        features = self.avgpool(x).squeeze()

        features = self.dropout(features)
        features = self.latent_head(features)

        smoothed_features = features

        if self.mode == "encoder":
            return smoothed_features, features

        elif self.mode == "regressor":
            x = self.regressor(smoothed_features)
            return x, features

        elif self.mode == "classifier":

            x = self.classifier(smoothed_features)
            x = nn.functional.log_softmax(x, dim=-1)
            return x, features