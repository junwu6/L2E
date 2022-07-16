import torch
import torch.nn as nn
from torchvision import models
from models.helper import *


class M3SDAModel(nn.Module):
    def __init__(self, num_classes, option='resnet18', use_bottleneck=True, bottleneck_width=256, width=1024):
        super(M3SDAModel, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
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

    def forward(self, s_inputs, s_outputs, t_inputs, alpha):
        all_s_inputs = torch.cat(s_inputs, dim=0)
        all_s_feats = self.base_network(all_s_inputs).view(all_s_inputs.shape[0], -1)
        t_feats = self.base_network(t_inputs).view(t_inputs.shape[0], -1)
        if self.use_bottleneck:
            all_s_feats = self.bottleneck_layer(all_s_feats)
            t_feats = self.bottleneck_layer(t_feats)

        s_features, class_losses = [], []
        for i in range(len(s_inputs)):
            s_feats = all_s_feats[s_inputs[i].shape[0]*i:s_inputs[i].shape[0]*(i+1)]
            s_preds = self.classifier_layer(s_feats)
            class_loss = self.criterion(s_preds, s_outputs[i])
            class_losses.append(class_loss)
            s_features.append(s_feats)

        loss_msda = msda_regulizer(s_features, t_feats)
        loss = torch.mean(torch.stack(class_losses)) + loss_msda * alpha * 0.01
        return loss

    def inference(self, x):
        x = self.base_network(x).view(x.shape[0], -1)
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)
        return self.classifier_layer(x)
