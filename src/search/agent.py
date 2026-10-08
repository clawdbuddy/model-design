import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.scoring.factory import create_scorer
from src.proposal.factory import create_proposer
from src.utils.device import resolve_device
from src.utils.model_builder import build_model
from src.utils.io import save_json


class SearchAgent:
    def __init__(self, config: Dict, mode: str = "mock", api_key: Optional[str] = None, model: str = "gpt-4"):
        self.config = config
        self.mode = mode
        self.constraints = config.get("constraints", {})
        self.device = resolve_device(self.constraints.get("device", "cpu"))
        self.scorer = create_scorer(config.get("scoring", {}))
        self.proposer = create_proposer(mode=mode, api_key=api_key, model=model, config=config.get("proposal", {}))
        self.history = []
        self.best = {"score": -float('inf'), "spec": None, "id": None}
        self.round = 0
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

    def search(self, max_rounds: int = 5, candidates_per_round: int = 3) -> Dict:
        print("=" * 70)
        print("🔬 Model Architecture Search")
        print(f"   Scorer: {self.scorer.name}")
        print(f"   Proposer: {self.proposer.name}")
        print(f"   Mode: {self.mode}")
        print(f"   Device: {self.device}")
        print("=" * 70)

        current_spec = self._get_initial_spec()

        for round_num in range(1, max_rounds + 1):
            self.round = round_num
            print(f"\n{'='*70}")
            print(f"🔄 Round {round_num}/{max_rounds}")
            print(f"{'='*70}")

            print(f"\n🤖 Generating {candidates_per_round} candidates...")
            candidates = self.proposer.generate(current_spec=current_spec, num_candidates=candidates_per_round, feedback=self._get_feedback())

            print(f"\n⚡ Scoring with {self.scorer.name}...")
            round_results = []

            for i, candidate in enumerate(candidates):
                result = self._evaluate_candidate(candidate, f"r{round_num}_c{i}")
                round_results.append(result)
                status = "✅" if result["valid"] else "❌"
                print(f"  {status} {result['id']}: score={result['score']:.2e}, params={result['params']:,}")

            valid_results = [r for r in round_results if r["valid"]]
            if valid_results:
                round_best = max(valid_results, key=lambda x: x["score"])
                if round_best["score"] > self.best["score"]:
                    improvement = (round_best["score"] - self.best["score"]) / max(abs(self.best["score"]), 1e-10)
                    print(f"\n🎯 New best: {round_best['id']} (+{improvement:.1%})")
                    self.best = round_best
                    current_spec = round_best["spec"]

            self.history.extend(round_results)
            self._save_checkpoint()

        self._save_final_report()
        return self.best

    def _evaluate_candidate(self, candidate: Dict, candidate_id: str) -> Dict:
        spec = candidate["architecture_spec"]
        try:
            model = build_model(spec)
            score = self.scorer(model, self.constraints.get("input_shape", (3, 32, 32)), device=self.device)
            params = sum(p.numel() for p in model.parameters())
            result = {"id": candidate_id, "spec": spec, "score": score, "params": params, "valid": params <= self.constraints.get("max_params", float('inf')), "strategy": candidate.get("strategy", "unknown"), "timestamp": time.time()}
        except Exception as e:
            result = {"id": candidate_id, "spec": spec, "score": 0.0, "params": 0, "valid": False, "error": str(e), "timestamp": time.time()}
        save_json(self.output_dir / "candidates" / f"{candidate_id}.json", result)
        return result

    def _get_initial_spec(self) -> Dict:
        return {"input_channels": self.constraints.get("input_channels", 3), "layers": [{"type": "Conv2d", "out_channels": 32, "kernel_size": 3, "padding": 1}, {"type": "Conv2d", "out_channels": 64, "kernel_size": 3, "padding": 1}, {"type": "MaxPool2d", "kernel_size": 2}, {"type": "AvgPool2d"}, {"type": "Linear", "out_features": self.constraints.get("num_classes", 10)}]}

    def _get_feedback(self) -> Dict:
        return {"best_score": self.best["score"] if self.best["score"] > -float('inf') else None, "best_spec": self.best["spec"], "history": self.history[-5:] if self.history else []}

    def _save_checkpoint(self):
        save_json(self.output_dir / "checkpoint.json", {"round": self.round, "best": self.best, "history_length": len(self.history)})

    def _save_final_report(self):
        report = {"best": self.best, "config": self.config, "total_candidates": len(self.history), "rounds": self.round}
        save_json(self.output_dir / "reports" / "final_report.json", report)
        print(f"\n💾 Report saved to {self.output_dir}/reports/final_report.json")
