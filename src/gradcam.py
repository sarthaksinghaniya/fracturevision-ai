import numpy as np
import torch
import torch.nn.functional as F


def get_efficientnet_target_layer(model):
    """Return EfficientNet last conv block, typically model.model.features[-1]."""
    if hasattr(model, "model") and hasattr(model.model, "features"):
        return model.model.features[-1]
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return model.backbone.blocks[-1]
    raise AttributeError("Could not locate EfficientNet target layer.")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.feature_maps = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __del__(self):
        self.remove_hooks()

    def _extract_logits(self, model_output):
        if torch.is_tensor(model_output):
            return model_output
        if isinstance(model_output, dict):
            if "multi_logits" in model_output:
                return model_output["multi_logits"]
            if "logits" in model_output:
                return model_output["logits"]
        raise TypeError("Unsupported model output for Grad-CAM.")

    def generate(self, input_tensor, class_idx):
        if input_tensor.ndim != 4:
            raise ValueError("input_tensor must be 4D: [batch, channels, height, width].")
        if input_tensor.size(0) != 1:
            raise ValueError("Grad-CAM currently expects batch size 1.")

        device = next(self.model.parameters()).device
        x = input_tensor.to(device)
        self.feature_maps = None
        self.gradients = None

        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        logits = self._extract_logits(output)
        score = logits[:, int(class_idx)].sum()
        score.backward()

        if self.feature_maps is None or self.gradients is None:
            raise RuntimeError("Failed to capture features/gradients. Check target_layer selection.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)

        cam = cam[0, 0]
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        heatmap = cam.detach().cpu().numpy().astype(np.float32)
        return heatmap
