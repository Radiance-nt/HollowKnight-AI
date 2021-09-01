import torch
import torch.nn as nn
import torch.nn.functional as F
from Tool.FrameGetter import FrameGetter


def normalization_layer(num_channels: int, norm_type: str, groups: int, is_2d: bool = True):
    if norm_type == "batch_norm":
        return nn.BatchNorm2d(num_channels) if is_2d else nn.BatchNorm1d(num_channels)
    elif norm_type == "group_norm":
        return nn.GroupNorm(groups, num_channels)
    elif norm_type == "layer_norm":
        assert is_2d is False, "LayerNorm can only be used on MLP layers."
        return nn.LayerNorm(num_channels)
    elif norm_type == "none":
        return nn.Identity()
    else:
        assert False, "Unknown normalization type: {}".format(norm_type)


class PreactResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 norm_type: str,
                 groups: int):
        super(PreactResBlock, self).__init__()

        use_bias = norm_type == "none"
        self.bn1 = normalization_layer(in_channels, norm_type, groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=use_bias)
        self.bn2 = normalization_layer(out_channels, norm_type, groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        shortcut = self.downsample(x) if hasattr(self, "downsample") else x

        y = x
        y = self.conv1(F.relu(self.bn1(y)))
        y = self.conv2(F.relu(self.bn2(y)))
        return y + shortcut


class ResEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_dims: int,
                 num_layers: int = 3,
                 start_channels: int = 16,
                 norm_type: str = "group_norm",
                 groups: int = 8):
        super().__init__()
        # network architecture
        # initial conv
        use_bias = norm_type == "none"
        layers = [
            nn.Conv2d(in_channels, start_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            normalization_layer(start_channels, norm_type, groups)
        ]
        # res blocks
        last_channels = num_channels = start_channels
        for idx in range(num_layers):
            layers.append(PreactResBlock(last_channels, num_channels, 2, norm_type, groups))
            layers.append(PreactResBlock(num_channels, num_channels, 1, norm_type, groups))
            last_channels = num_channels
            num_channels *= 2

        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_dims)

    def forward(self, x):
        # reshape N FS C H W --> N C*FS H W
        if len(x.shape) == 5:
            x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        # uint8 --> float
        if x.dtype is torch.uint8:
            x = x.to(torch.float) / 255

        x = self.layers(x)
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        if x.shape[0] == 1:
            x = self.fc[6](x)
        else:
            x = self.fc(x)
        return x


if __name__ == "__main__":
    framebuffer = FrameGetter()
    obs = framebuffer.get_frame()
    print(obs.shape)
    if not isinstance(obs, torch.Tensor):
        obs = torch.Tensor(obs).unsqueeze(0).unsqueeze(0)
    encoder = ResEncoder(in_channels=1, out_dims=256)
    encoder(obs)
