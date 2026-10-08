from src.search.agent import SearchAgent


def test_search_loop_mock_mode(tiny_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # SearchAgent 把产物写到相对路径 outputs/
    agent = SearchAgent(config=tiny_config, mode="mock")
    best = agent.search(max_rounds=2, candidates_per_round=2)

    assert best["score"] > -float("inf")
    assert best["id"].startswith("r")
    assert best["params"] > 0
    assert best["valid"] is True

    assert (tmp_path / "outputs" / "checkpoint.json").exists()
    assert (tmp_path / "outputs" / "reports" / "final_report.json").exists()
    assert len(agent.history) == 4  # 2 轮 × 每轮 2 个候选


def test_oversized_candidates_marked_invalid(tiny_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tiny_config["constraints"]["max_params"] = 10  # 远小于任何候选
    agent = SearchAgent(config=tiny_config, mode="mock")
    best = agent.search(max_rounds=1, candidates_per_round=2)

    assert all(r["valid"] is False for r in agent.history)
    # 无有效候选时保持初始空状态
    assert best["spec"] is None


def test_evaluate_candidate_reports_build_error(tiny_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = SearchAgent(config=tiny_config, mode="mock")
    broken = {"architecture_spec": {"input_channels": 3, "layers": [{"type": "Linear"}]}}
    result = agent._evaluate_candidate(broken, "r1_cX")

    assert result["valid"] is False
    assert result["score"] == 0.0
    assert "error" in result
