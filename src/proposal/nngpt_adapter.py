"""
基于 ABrain-One/NN-GPT 的提示词策略
参考其闭环发现流程：AST变异 → LLM生成 → 性能反馈
"""
import json
from pathlib import Path

class NNGPTPromptAdapter:
    """NN-GPT 风格的提示词生成器"""
    
    def __init__(self):
        # 加载 NN-GPT 的提示词模板（如果公开）
        self.template = self._load_template()
    
    def _load_template(self):
        """NN-GPT 的核心提示词结构"""
        # 基于论文和代码仓库的提示词设计
        return """
You are an expert in neural architecture search, specializing in channel configuration optimization.

Current Architecture Code:
{architecture_code}

Performance Feedback:
- NASWOT Score: {score}
- Previous Best: {best_score}
- Failure Modes: {failures}

Task: Modify the channel widths to improve performance while maintaining tensor shape consistency.

Constraints:
- Parameter budget: {param_limit}
- When changing a layer's channels, you MUST adjust all coupled components (residual connections, concat layers, etc.)

Output Format:
Return a JSON object with:
1. "channel_changes": list of {layer_idx, old_channels, new_channels}
2. "coupled_adjustments": list of descriptions for synchronized changes
3. "rationale": explanation for this modification
"""
    
    def create_prompt(self, current_spec, feedback, constraints):
        """生成 NN-GPT 风格的提示词"""
        
        # 将架构规格转为类代码表示（NN-GPT 使用代码作为输入）
        arch_code = self._spec_to_pseudo_code(current_spec)
        
        prompt = self.template.format(
            architecture_code=arch_code,
            score=feedback.get("current_score", "N/A"),
            best_score=feedback.get("best_score", "N/A"),
            failures=feedback.get("failure_modes", []),
            param_limit=constraints.get("max_params", "unknown")
        )
        
        return prompt
    
    def _spec_to_pseudo_code(self, spec):
        """将 JSON 规格转为伪代码（NN-GPT 使用代码表示）"""
        lines = []
        for i, layer in enumerate(spec.get("layers", [])):
            if layer["type"] == "Conv2d":
                lines.append(f"layer_{i} = Conv2d({layer['out_channels']}, k={layer.get('kernel_size', 3)})")
            elif layer["type"] == "DepthwiseConv2d":
                lines.append(f"layer_{i} = DepthwiseConv2d({layer['out_channels']})")
        return "\n".join(lines)
    
    def parse_response(self, response_text):
        """解析 LLM 返回的通道修改"""
        try:
            # 提取 JSON 部分
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
            
            result = json.loads(json_str)
            
            # 转换为架构规格修改
            modifications = {
                "channel_changes": result.get("channel_changes", []),
                "coupled_adjustments": result.get("coupled_adjustments", []),
                "rationale": result.get("rationale", "")
            }
            
            return modifications
            
        except Exception as e:
            print(f"解析 NN-GPT 响应失败: {e}")
            return None


# 使用示例
def create_adapter():
    return NNGPTPromptAdapter()
