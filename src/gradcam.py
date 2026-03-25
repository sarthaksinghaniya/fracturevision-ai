import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_efficientnet_target_layer(model):
    """Return EfficientNet last conv block, typically model.model.features[-1]."""
    if hasattr(model, "model") and hasattr(model.model, "features"):
        return model.model.features[-1]
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return model.backbone.blocks[-1]
    raise AttributeError("Could not locate EfficientNet target layer.")


def resolve_target_layer(model):
    """Resolve a robust Grad-CAM target layer across wrapped model architectures."""
    print(model)
    if hasattr(model, "model") and hasattr(model.model, "features"):
        return model.model.features[-1]

    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is not None:
        return last_conv

    return get_efficientnet_target_layer(model)


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


def overlay_heatmap(original_image, heatmap, alpha=0.4):
    """Overlay a normalized heatmap (0..1) on an RGB image."""
    if hasattr(original_image, "convert"):
        original = np.array(original_image.convert("RGB"))
    else:
        original = np.asarray(original_image)
        if original.ndim == 2:
            original = np.stack([original, original, original], axis=-1)
        if original.shape[-1] == 4:
            original = original[..., :3]

    original = original.astype(np.uint8)
    h, w = original.shape[:2]

    heatmap_uint8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(original, 1.0 - alpha, heatmap_color, alpha, 0.0)
    return blended
