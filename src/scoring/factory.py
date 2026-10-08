from src.scoring.naswot import NASWOTScorer


def create_scorer(config: dict):
    method = config.get("method", "naswot")
    if method == "naswot":
        return NASWOTScorer()
    else:
        raise ValueError(f"Unknown scoring method: {method}")
