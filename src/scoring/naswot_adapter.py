"""
直接调用 BayesWatch/nas-without-training 的官方实现
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加官方仓库路径（假设已克隆到本地）
NASWOT_PATH = Path("./nas-without-training")
sys.path.insert(0, str(NASWOT_PATH))

# 官方实现的核心函数（根据实际仓库结构调整）
from naswot import compute_naswot_score  # 如果官方有封装
# 或者直接从官方代码中提取核心逻辑

class OfficialNASWOTScorer:
    """基于官方 NASWOT 实现的评分器"""
    
    def __init__(self):
        self.name = "NASWOT (Official)"
    
    def __call__(self, model, input_shape=(3, 32, 32), batch_size=32):
        """
        调用官方实现计算 NASWOT 分数
        """
        model.eval()
        
        # 官方实现：注册 hook 收集 ReLU 激活
        activations = []
        
        def hook(module, input, output):
            activations.append(output.detach())
        
        # 注册到所有 ReLU 层
        hooks = []
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(hook))
        
        # 前向传播
        x = torch.randn(batch_size, *input_shape)
        with torch.no_grad():
            model(x)
        
        # 移除 hooks
        for h in hooks:
            h.remove()
        
        if not activations:
            return 0.0
        
        # 官方核心算法：最后一个 ReLU 的核矩阵行列式
        last_act = activations[-1].cpu().numpy()
        last_act = last_act.reshape(last_act.shape[0], -1)
        
        # 计算 K = A @ A^T
        K = last_act @ last_act.T
        
        # 行列式
        import numpy as np
        score = np.linalg.det(K + 1e-6 * np.eye(batch_size))
        
        return float(score)


# 便捷函数
def create_scorer():
    """创建评分器实例"""
    return OfficialNASWOTScorer()
