import math

import torch.nn as nn

from src.scoring.naswot import NASWOTScorer


def test_score_is_finite_float(base_spec):
    from src.utils.model_builder import build_model

    scorer = NASWOTScorer()
    model = build_model(base_spec)
    score = scorer(model, input_shape=(3, 32, 32), batch_size=8)
    assert isinstance(score, float)
    assert math.isfinite(score)


def test_model_without_relu_scores_zero():
    scorer = NASWOTScorer()
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    assert scorer(model, input_shape=(2, 2), batch_size=4) == 0.0
