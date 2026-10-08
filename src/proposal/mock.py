import copy
import random
from typing import Dict, List


class MockProposer:
    name = "MockProposer"

    def __init__(self, config: Dict = None):
        self.config = config or {}

    def generate(self, current_spec: Dict, num_candidates: int, feedback: Dict) -> List[Dict]:
        candidates = []
        for i in range(num_candidates):
            if i == 0 and feedback.get("best_spec"):
                candidate = self._mutate_from_best(feedback["best_spec"])
            else:
                candidate = self._random_spec(current_spec)
            candidate["strategy"] = "exploit" if i == 0 else "explore"
            candidates.append(candidate)
        return candidates

    def _random_spec(self, base_spec: Dict) -> Dict:
        spec = copy.deepcopy(base_spec)
        for layer in spec["layers"]:
            if layer["type"] in ["Conv2d", "DepthwiseConv2d"]:
                if random.random() < 0.5:
                    layer["out_channels"] = random.choice([16, 32, 64, 128, 256])
        return {"architecture_spec": spec}

    def _mutate_from_best(self, best_spec: Dict) -> Dict:
        spec = copy.deepcopy(best_spec)
        mutation_type = random.choice(["widen", "deepen", "shrink"])
        if mutation_type == "widen":
            for layer in spec["layers"]:
                if layer["type"] in ["Conv2d", "DepthwiseConv2d"]:
                    layer["out_channels"] = min(layer["out_channels"] * 2, 512)
                    break
        elif mutation_type == "deepen":
            insert_pos = random.randint(0, len(spec["layers"]) - 1)
            spec["layers"].insert(insert_pos, {"type": "DepthwiseConv2d", "out_channels": 64, "kernel_size": 3, "padding": 1})
        else:
            for layer in spec["layers"]:
                if layer["type"] in ["Conv2d", "DepthwiseConv2d"]:
                    layer["out_channels"] = max(layer["out_channels"] // 2, 8)
        return {"architecture_spec": spec}
