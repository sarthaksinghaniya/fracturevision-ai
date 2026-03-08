import torch
import torch.nn as nn
import timm

class FractureClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True, num_classes=1):
        super(FractureClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # Get the number of features from the backbone
        if hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes)
            )
        elif hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes)
            )
        else:
            raise ValueError("Model does not have a classifier or fc layer")

    def forward(self, x):
        return self.model(x)
