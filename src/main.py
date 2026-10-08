#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search.agent import SearchAgent
from src.utils.io import load_config
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "api"], default="mock")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="gpt-4")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--candidates-per-round", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("model_design")
    logger.info(f"Starting in {args.mode} mode")

    agent = SearchAgent(config=config, mode=args.mode, api_key=args.api_key, model=args.model)
    best = agent.search(max_rounds=args.max_rounds, candidates_per_round=args.candidates_per_round)

    logger.info(f"Best: {best['id']} | Score: {best['score']:.2e} | Params: {best['params']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
