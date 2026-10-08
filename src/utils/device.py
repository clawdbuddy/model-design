import torch


def resolve_device(requested: str = "cpu") -> torch.device:
    """解析配置中的 device。

    请求 cuda 但当前环境不可用时自动回退 cpu，
    保证同一份配置在有/无 GPU 的机器上都能跑。
    """
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
