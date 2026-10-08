import pytest

from src.proposal.factory import create_proposer
from src.proposal.mock import MockProposer
from src.scoring.factory import create_scorer
from src.scoring.naswot import NASWOTScorer


def test_create_scorer_naswot():
    scorer = create_scorer({"method": "naswot"})
    assert isinstance(scorer, NASWOTScorer)


def test_create_scorer_default_is_naswot():
    assert isinstance(create_scorer({}), NASWOTScorer)


def test_create_scorer_unknown_raises():
    with pytest.raises(ValueError):
        create_scorer({"method": "nope"})


def test_create_proposer_mock_mode():
    assert isinstance(create_proposer(mode="mock"), MockProposer)


def test_api_mode_without_key_falls_back_to_mock():
    assert isinstance(create_proposer(mode="api", api_key=None), MockProposer)
