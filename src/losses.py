import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        if target.ndim == 1:
            target_dist = F.one_hot(target, num_classes=num_classes).float()
            if self.label_smoothing > 0 and num_classes > 1:
                smooth = self.label_smoothing / (num_classes - 1)
                target_dist = target_dist * (1.0 - self.label_smoothing) + (1.0 - target_dist) * smooth
        else:
            target_dist = target.float()

        ce_loss = -(target_dist * log_probs)
        if self.weight is not None:
            ce_loss = ce_loss * self.weight.unsqueeze(0)
        ce_loss = ce_loss.sum(dim=1)

        pt = (probs * target_dist).sum(dim=1).clamp(min=1e-6, max=1.0)
        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
