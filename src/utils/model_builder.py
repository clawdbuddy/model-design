import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_depthwise=False):
        super().__init__()
        if use_depthwise:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, bias=False)
            )
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def build_model(spec: dict) -> nn.Module:
    layers = []
    in_channels = spec.get("input_channels", 3)

    for layer in spec["layers"]:
        layer_type = layer["type"]

        if layer_type == "Conv2d":
            out_ch = layer["out_channels"]
            layers.append(ConvBlock(in_channels, out_ch, kernel_size=layer.get("kernel_size", 3), stride=layer.get("stride", 1), padding=layer.get("padding", 1)))
            in_channels = out_ch

        elif layer_type == "DepthwiseConv2d":
            out_ch = layer["out_channels"]
            layers.append(ConvBlock(in_channels, out_ch, kernel_size=layer.get("kernel_size", 3), padding=layer.get("padding", 1), use_depthwise=True))
            in_channels = out_ch

        elif layer_type == "MaxPool2d":
            layers.append(nn.MaxPool2d(kernel_size=layer.get("kernel_size", 2), stride=layer.get("stride", 2)))

        elif layer_type == "AvgPool2d":
            layers.append(nn.AdaptiveAvgPool2d(1))

        elif layer_type == "Linear":
            layers.append(nn.Flatten())
            layers.append(nn.Linear(in_channels, layer["out_features"]))
            in_channels = layer["out_features"]

    return nn.Sequential(*layers)
