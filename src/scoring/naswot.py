import numpy as np
import torch
import torch.nn as nn


class NASWOTScorer:
    name = "NASWOT"

    def __call__(self, model: nn.Module, input_shape=(3, 32, 32), batch_size=32, device=None) -> float:
        device = torch.device(device) if device is not None else torch.device("cpu")
        model = model.to(device)
        model.eval()
        activations = []

        def hook(module, input, output):
            activations.append(output.detach().cpu().numpy())

        hooks = []
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(hook))

        x = torch.randn(batch_size, *input_shape, device=device)
        with torch.no_grad():
            model(x)

        for h in hooks:
            h.remove()

        if not activations:
            return 0.0

        last_act = activations[-1].reshape(activations[-1].shape[0], -1)
        K = last_act @ last_act.T
        score = np.linalg.det(K + 1e-6 * np.eye(batch_size))
        return float(score)
