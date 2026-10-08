from src.proposal.mock import MockProposer

VALID_TYPES = {"Conv2d", "DepthwiseConv2d", "MaxPool2d", "AvgPool2d", "Linear"}


def _assert_valid_candidate(candidate):
    assert "architecture_spec" in candidate
    assert "strategy" in candidate
    spec = candidate["architecture_spec"]
    assert spec["layers"]
    for layer in spec["layers"]:
        assert layer["type"] in VALID_TYPES


def test_generate_returns_requested_count(base_spec):
    proposer = MockProposer()
    candidates = proposer.generate(base_spec, num_candidates=3, feedback={})
    assert len(candidates) == 3
    for c in candidates:
        _assert_valid_candidate(c)


def test_strategy_labels_by_slot(base_spec):
    """策略标签按候选槽位打：第 0 个恒为 exploit，其余为 explore
    （即使 feedback 无 best_spec、第 0 个实际走的是随机变异）"""
    proposer = MockProposer()
    candidates = proposer.generate(base_spec, num_candidates=2, feedback={})
    assert candidates[0]["strategy"] == "exploit"
    assert candidates[1]["strategy"] == "explore"


def test_first_candidate_exploits_best_spec(base_spec):
    proposer = MockProposer()
    feedback = {"best_spec": base_spec}
    candidates = proposer.generate(base_spec, num_candidates=3, feedback=feedback)
    assert candidates[0]["strategy"] == "exploit"
    assert [c["strategy"] for c in candidates[1:]] == ["explore", "explore"]


def test_generate_does_not_mutate_input(base_spec):
    proposer = MockProposer()
    before = str(base_spec)
    proposer.generate(base_spec, num_candidates=3, feedback={"best_spec": base_spec})
    assert str(base_spec) == before
