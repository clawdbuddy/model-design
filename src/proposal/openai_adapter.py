import json
from typing import Dict, List

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class OpenAIProposer:
    name = "OpenAIProposer"

    def __init__(self, api_key: str, model: str = "gpt-4", config: Dict = None):
        if not HAS_OPENAI:
            raise ImportError("Please install openai: pip install openai")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.config = config or {}

    def generate(self, current_spec: Dict, num_candidates: int, feedback: Dict) -> List[Dict]:
        prompt = f"""You are a neural architecture search expert.

Current architecture: {json.dumps(current_spec)}
Best score so far: {feedback.get('best_score', 'N/A')}

Generate {num_candidates} improved architecture variants as JSON array.
Each item should have: architecture_spec (with layers), rationale, strategy.
Max params: {self.config.get('max_params', 500000)}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        if isinstance(data, dict):
            data = [data]
        return data[:num_candidates]
