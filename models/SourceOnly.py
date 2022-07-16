import torch.nn as nn
from torchvision import models
import torch


class SourceOnlyModel(nn.Module):
    def __init__(self, num_classes, option='resnet18', use_bottleneck=True, bottleneck_width=256, width=1024):
        super(SourceOnlyModel, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=False)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=True)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=True)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=True)
        mod = list(model_ft.children())
        mod.pop()
        self.base_network = nn.Sequential(*mod)
        self.use_bottleneck = use_bottleneck
        self.num_classes = num_classes

        self.bottleneck_layer = nn.Sequential(
            nn.Linear(self.dim, bottleneck_width),
            nn.BatchNorm1d(bottleneck_width),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        hidden_dim = bottleneck_width if self.use_bottleneck else self.dim
        self.classifier_layer = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, s_inputs, s_outputs, t_inputs=None, alpha=1.0):
        s_inputs = torch.cat(s_inputs, dim=0)
        s_outputs = torch.cat(s_outputs, dim=0)
        s_feats = self.base_network(s_inputs).view(s_inputs.shape[0], -1)
        if self.use_bottleneck:
            s_feats = self.bottleneck_layer(s_feats)
        s_preds = self.classifier_layer(s_feats)
        loss = self.criterion(s_preds, s_outputs)

        return loss

    def inference(self, x):
        x = self.base_network(x).view(x.shape[0], -1)
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)
        return self.classifier_layer(x)

    def get_features(self, x):
        x = self.base_network(x).view(x.shape[0], -1)
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)
        return x
