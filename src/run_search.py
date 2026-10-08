import json
import os
from datetime import datetime

# 导入开源适配器
from scoring.naswot_adapter import create_scorer
from proposal.nngpt_adapter import create_adapter as create_nngpt
from proposal.lmsearcher_adapter import create_adapter as create_lmsearcher

# 可选：真实 API
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class ModelDesignWithLibs:
    """使用开源框架包的模型设计 Agent"""
    
    def __init__(self, constraints, api_key=None, use_nngpt=True):
        self.constraints = constraints
        self.use_nngpt = use_nngpt  # True: NN-GPT风格, False: LM-Searcher风格
        
        # 初始化组件
        self.scorer = create_scorer()
        self.proposer = create_nngpt() if use_nngpt else create_lmsearcher()
        
        # API 客户端
        self.client = None
        if api_key and HAS_OPENAI:
            self.client = OpenAI(api_key=api_key)
        
        # 状态
        self.history = []
        self.best = {"score": -1, "spec": None}
        
        os.makedirs("results", exist_ok=True)
    
    def search(self, max_rounds=5, candidates_per_round=3):
        """执行搜索"""
        
        print("=" * 70)
        print("🔬 模型架构搜索（基于开源框架包）")
        print(f"评分器: {self.scorer.name}")
        print(f"提案器: {'NN-GPT' if self.use_nngpt else 'LM-Searcher'}")
        print(f"API: {'OpenAI' if self.client else 'Mock'}")
        print("=" * 70)
        
        # 初始架构
        current_spec = self._get_initial_spec()
        
        for round_num in range(1, max_rounds + 1):
            print(f"\n{'='*70}")
            print(f"🔄 Round {round_num}/{max_rounds}")
            print(f"{'='*70}")
            
            # Step 1: 生成候选
            candidates = self._generate_candidates(
                current_spec, 
                candidates_per_round
            )
            
            # Step 2: 评分
            print(f"\n⚡ Scoring {len(candidates)} candidates...")
            round_results = []
            
            for i, candidate in enumerate(candidates):
                spec = candidate["architecture_spec"]
                
                # 构建模型（简化：假设有 build_model 函数）
                model = self._build_model(spec)
                
                # 使用官方 NASWOT 评分
                score = self.scorer(model, self.constraints["input_shape"])
                params = sum(p.numel() for p in model.parameters())
                
                result = {
                    "id": f"r{round_num}_c{i}",
                    "spec": spec,
                    "score": score,
                    "params": params,
                    "valid": params <= self.constraints["max_params"]
                }
                
                round_results.append(result)
                
                status = "✅" if result["valid"] else "❌"
                print(f"  {status} {result['id']}: score={score:.2e}, params={params:,}")
            
            # Step 3: 更新最优
            valid_results = [r for r in round_results if r["valid"]]
            if valid_results:
                round_best = max(valid_results, key=lambda x: x["score"])
                
                if round_best["score"] > self.best["score"]:
                    print(f"\n🎯 New best: {round_best['id']} "
                          f"({self.best['score']:.2e} → {round_best['score']:.2e})")
                    self.best = round_best
                    current_spec = round_best["spec"]
            
            self.history.extend(round_results)
        
        self._save_results()
        return self.best
    
    def _get_initial_spec(self):
        """初始架构"""
        return {
            "input_channels": 3,
            "layers": [
                {"type": "Conv2d", "out_channels": 32, "kernel_size": 3, "padding": 1},
                {"type": "Conv2d", "out_channels": 64, "kernel_size": 3, "padding": 1},
                {"type": "MaxPool2d", "kernel_size": 2},
                {"type": "AvgPool2d"},
                {"type": "Linear", "out_features": self.constraints.get("num_classes", 10)}
            ]
        }
    
    def _generate_candidates(self, current_spec, num):
        """生成候选"""
        
        if not self.client:
            # Mock 模式：随机变异
            return self._mock_generate(current_spec, num)
        
        # 准备反馈
        feedback = {
            "score": "N/A",
            "best_score": f"{self.best['score']:.2e}" if self.best['score'] > 0 else "N/A",
            "failure_modes": []
        }
        
        # 生成提示词
        if self.use_nngpt:
            prompt = self.proposer.create_prompt(current_spec, feedback, self.constraints)
        else:
            ncode = self.proposer.encoder.encode(current_spec)
            prompt = self.proposer.create_prompt(ncode, feedback, self.constraints)
        
        # 调用 API
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # 解析
        if self.use_nngpt:
            modifications = self.proposer.parse_response(response.choices[0].message.content)
            # 应用修改到当前架构...
            return self._apply_modifications(current_spec, modifications, num)
        else:
            variants = self.proposer.parse_variants(response.choices[0].message.content)
            return variants[:num]
    
    def _mock_generate(self, current_spec, num):
        """模拟生成（无 API 时使用）"""
        import random
        candidates = []
        
        for i in range(num):
            # 简单变异：改变通道数
            new_spec = json.loads(json.dumps(current_spec))  # 深拷贝
            
            for layer in new_spec["layers"]:
                if layer["type"] in ["Conv2d", "DepthwiseConv2d"]:
                    if random.random() < 0.5:
                        layer["out_channels"] = random.choice([16, 32, 64, 128])
            
            candidates.append({
                "architecture_spec": new_spec,
                "strategy": "mock_mutation"
            })
        
        return candidates
    
    def _build_model(self, spec):
        """构建 PyTorch 模型（简化版）"""
        import torch.nn as nn
        
        layers = []
        in_ch = spec.get("input_channels", 3)
        
        for layer in spec["layers"]:
            if layer["type"] == "Conv2d":
                out_ch = layer["out_channels"]
                layers.extend([
                    nn.Conv2d(in_ch, out_ch, layer.get("kernel_size", 3), 
                             padding=layer.get("padding", 1), bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ])
                in_ch = out_ch
            elif layer["type"] == "DepthwiseConv2d":
                out_ch = layer["out_channels"]
                layers.extend([
                    nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ])
                in_ch = out_ch
            elif layer["type"] == "MaxPool2d":
                layers.append(nn.MaxPool2d(2))
            elif layer["type"] == "AvgPool2d":
                layers.append(nn.AdaptiveAvgPool2d(1))
            elif layer["type"] == "Linear":
                layers.extend([
                    nn.Flatten(),
                    nn.Linear(in_ch, layer["out_features"])
                ])
        
        return nn.Sequential(*layers)
    
    def _save_results(self):
        """保存结果"""
        report = {
            "best": self.best,
            "history": self.history[-10:],  # 最近10条
            "config": {
                "scorer": self.scorer.name,
                "proposer": "NN-GPT" if self.use_nngpt else "LM-Searcher"
            }
        }
        
        with open(f"results/search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Results saved to results/")


# ==================== 运行 ====================

if __name__ == "__main__":
    constraints = {
        "max_params": 500000,
        "input_shape": (3, 32, 32),
        "num_classes": 10,
        "num_candidates": 3
    }
    
    # 使用 NN-GPT 风格（需要 API key 才能真实运行）
    # agent = ModelDesignWithLibs(constraints, api_key="sk-...", use_nngpt=True)
    
    # 使用 LM-Searcher 风格
    # agent = ModelDesignWithLibs(constraints, api_key="sk-...", use_nngpt=False)
    
    # Mock 模式（无需 API）
    agent = ModelDesignWithLibs(constraints, api_key=None, use_nngpt=True)
    best = agent.search(max_rounds=3, candidates_per_round=3)
    
    print(f"\n🏆 Best architecture: {best['id']}")
    print(f"   Score: {best['score']:.2e}")
    print(f"   Params: {best['params']:,}")
