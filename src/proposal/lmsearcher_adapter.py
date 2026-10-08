"""
基于 Ashone3/LM-Searcher 的 NCode 编码
将架构转为统一数值字符串，便于 LLM 理解
"""
import json

class NCodeEncoder:
    """LM-Searcher 的 NCode 编码器"""
    
    # 层类型映射（根据 LM-Searcher 论文）
    TYPE_MAPPING = {
        "Conv2d": "C",
        "DepthwiseConv2d": "D",
        "MaxPool2d": "P",
        "AvgPool2d": "A",
        "Linear": "L",
        "BatchNorm2d": "B",
        "ReLU": "R"
    }
    
    def encode(self, spec):
        """
        将架构规格编码为 NCode 字符串
        示例: "C32_3-B-R-C64_3-B-R-P2-D128_3-B-R-A-L10"
        """
        codes = []
        
        for layer in spec.get("layers", []):
            layer_type = layer["type"]
            code = self.TYPE_MAPPING.get(layer_type, "X")
            
            # 添加参数信息
            if layer_type in ["Conv2d", "DepthwiseConv2d"]:
                code += f"{layer['out_channels']}_{layer.get('kernel_size', 3)}"
            elif layer_type == "Linear":
                code += f"{layer['out_features']}"
            elif layer_type in ["MaxPool2d", "AvgPool2d"]:
                code += f"{layer.get('kernel_size', 2)}"
            
            codes.append(code)
        
        return "-".join(codes)
    
    def decode(self, ncode_str):
        """将 NCode 解码回架构规格（简化版）"""
        layers = []
        
        for part in ncode_str.split("-"):
            layer_type_code = part[0]
            params = part[1:] if len(part) > 1 else ""
            
            # 反向映射
            type_reverse = {v: k for k, v in self.TYPE_MAPPING.items()}
            layer_type = type_reverse.get(layer_type_code, "Unknown")
            
            layer = {"type": layer_type}
            
            if layer_type in ["Conv2d", "DepthwiseConv2d"] and params:
                ch, k = params.split("_")
                layer["out_channels"] = int(ch)
                layer["kernel_size"] = int(k)
                layer["padding"] = int(k) // 2
            elif layer_type == "Linear" and params:
                layer["out_features"] = int(params)
            elif layer_type in ["MaxPool2d", "AvgPool2d"] and params:
                layer["kernel_size"] = int(params)
                layer["stride"] = int(params)
            
            layers.append(layer)
        
        return {"layers": layers}


class LMSearcherAdapter:
    """LM-Searcher 风格的架构生成器"""
    
    def __init__(self):
        self.encoder = NCodeEncoder()
    
    def create_prompt(self, current_ncode, performance_feedback, constraints):
        """创建 LM-Searcher 风格的排序任务提示词"""
        
        prompt = f"""You are a neural architecture search expert using NCode representation.

Current Architecture (NCode):
{current_ncode}

Performance:
- Current NASWOT: {performance_feedback.get('score', 'N/A')}
- Best so far: {performance_feedback.get('best_score', 'N/A')}

Task: Generate {constraints.get('num_candidates', 3)} improved architecture variants.

NCode Format: TypeParams-TypeParams-...
- C: Conv2d (e.g., C32_3 = 32 channels, 3x3 kernel)
- D: DepthwiseConv2d
- P: MaxPool2d
- A: AvgPool2d
- L: Linear (e.g., L10 = 10 classes)

Constraints:
- Max parameters: {constraints.get('max_params', 'unknown')}
- Input: {constraints.get('input_shape', (3, 32, 32))}

Output: JSON array of NCode strings, e.g., ["C32_3-B-R-C64_3-B-R-P2-L10", ...]
"""
        return prompt
    
    def parse_variants(self, response_text):
        """解析 LLM 返回的 NCode 变体"""
        try:
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            ncode_list = json.loads(response_text[json_start:json_end])
            
            # 解码为架构规格
            variants = []
            for ncode in ncode_list:
                spec = self.encoder.decode(ncode)
                variants.append({
                    "architecture_spec": spec,
                    "ncode": ncode
                })
            
            return variants
            
        except Exception as e:
            print(f"解析 NCode 失败: {e}")
            return []


def create_adapter():
    return LMSearcherAdapter()
