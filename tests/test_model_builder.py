import pytest
import torch

from src.utils.model_builder import build_model


def test_build_and_forward_shape(base_spec):
    model = build_model(base_spec)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_param_count_positive(base_spec):
    model = build_model(base_spec)
    params = sum(p.numel() for p in model.parameters())
    assert params > 0


def test_depthwise_block_forward():
    spec = {
        "input_channels": 3,
        "layers": [
            {"type": "DepthwiseConv2d", "out_channels": 16, "kernel_size": 3, "padding": 1},
            {"type": "AvgPool2d"},
            {"type": "Linear", "out_features": 5},
        ],
    }
    model = build_model(spec)
    with torch.no_grad():
        out = model(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 5)


def test_channels_follow_layer_specs(base_spec):
    model = build_model(base_spec)
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert convs[0].in_channels == 3
    assert convs[0].out_channels == 32
    assert convs[1].in_channels == 32
    assert convs[1].out_channels == 64


def test_missing_required_field_raises(base_spec):
    del base_spec["layers"][0]["out_channels"]
    with pytest.raises(KeyError):
        build_model(base_spec)
