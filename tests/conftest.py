import sys
from pathlib import Path

import pytest

# 保证以仓库根目录为包根（import src.*）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def base_spec():
    """与 SearchAgent 初始架构一致的基准规格"""
    return {
        "input_channels": 3,
        "layers": [
            {"type": "Conv2d", "out_channels": 32, "kernel_size": 3, "padding": 1},
            {"type": "Conv2d", "out_channels": 64, "kernel_size": 3, "padding": 1},
            {"type": "MaxPool2d", "kernel_size": 2},
            {"type": "AvgPool2d"},
            {"type": "Linear", "out_features": 10},
        ],
    }


@pytest.fixture
def tiny_config():
    return {
        "constraints": {
            "max_params": 500000,
            "input_shape": [3, 32, 32],
            "input_channels": 3,
            "num_classes": 10,
            "device": "cpu",
        },
        "scoring": {"method": "naswot", "batch_size": 8},
        "proposal": {"style": "nngpt", "temperature": 0.7},
        "search": {"max_rounds": 2, "candidates_per_round": 2, "early_stop_patience": 3},
    }
