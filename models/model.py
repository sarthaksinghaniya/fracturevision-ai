import torch
import torch.nn as nn
import timm


def combine_head_probabilities(multi_logits, binary_logits, non_fracture_index):
    multi_probs = torch.softmax(multi_logits, dim=1)
    binary_fracture_probs = torch.sigmoid(binary_logits)
    adjusted_probs = multi_probs.clone()
    adjusted_probs[:, non_fracture_index] = adjusted_probs[:, non_fracture_index] * (1.0 - binary_fracture_probs)
    fracture_mask = torch.ones(multi_logits.size(1), dtype=torch.bool, device=multi_logits.device)
    fracture_mask[non_fracture_index] = False
    adjusted_probs[:, fracture_mask] = adjusted_probs[:, fracture_mask] * binary_fracture_probs.unsqueeze(1)
    adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return adjusted_probs


class FractureClassifier(nn.Module):
    def __init__(self, model_name="efficientnet_b3", pretrained=True, num_classes=2, dropout=0.3):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = self.backbone.num_features

        self.shared_head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.BatchNorm1d(in_features // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        hidden_features = in_features // 2
        self.multi_classifier = nn.Linear(hidden_features, num_classes)
        self.binary_classifier = nn.Linear(hidden_features, 1)

    def forward_features(self, x):
        features = self.backbone.forward_features(x)
        pooled = self.pool(features)
        embeddings = self.shared_head(pooled)
        return features, embeddings

    def forward_multitask(self, x):
        _, embeddings = self.forward_features(x)
        multi_logits = self.multi_classifier(embeddings)
        binary_logits = self.binary_classifier(embeddings).squeeze(1)
        return {
            "multi_logits": multi_logits,
            "binary_logits": binary_logits,
        }

    def forward(self, x):
        return self.forward_multitask(x)["multi_logits"]
