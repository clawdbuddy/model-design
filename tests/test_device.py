import torch

from src.scoring.naswot import NASWOTScorer
from src.search.agent import SearchAgent
from src.utils.device import resolve_device
from src.utils.model_builder import build_model


def test_cpu_passthrough():
    assert resolve_device("cpu") == torch.device("cpu")


def test_cuda_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda") == torch.device("cpu")


def test_cuda_used_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == torch.device("cuda")


def test_scorer_runs_on_requested_device(base_spec):
    scorer = NASWOTScorer()
    model = build_model(base_spec)
    score = scorer(model, input_shape=(3, 32, 32), batch_size=8, device=torch.device("cpu"))
    assert isinstance(score, float)


def test_agent_resolves_device_from_config(tiny_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tiny_config["constraints"]["device"] = "cuda"
    agent = SearchAgent(config=tiny_config, mode="mock")
    assert agent.device == torch.device("cpu")
