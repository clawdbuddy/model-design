from src.proposal.mock import MockProposer
from src.proposal.openai_adapter import OpenAIProposer, HAS_OPENAI


def create_proposer(mode: str, api_key: str = None, model: str = "gpt-4", config: dict = None):
    if mode == "api" and api_key and HAS_OPENAI:
        return OpenAIProposer(api_key=api_key, model=model, config=config)
    else:
        return MockProposer(config=config)
