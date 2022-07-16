import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from torchvision import models


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANNModel(nn.Module):
    def __init__(self, num_classes, option='resnet18', use_bottleneck=True, bottleneck_width=256, width=1024):
        super(DANNModel, self).__init__()
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

        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, 2),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, s_inputs, s_outputs, t_inputs, alpha):
        all_s_inputs = torch.cat(s_inputs, dim=0)
        all_s_outputs = torch.cat(s_outputs, dim=0)
        all_s_feats = self.base_network(all_s_inputs).view(all_s_inputs.shape[0], -1)
        t_feats = self.base_network(t_inputs).view(t_inputs.shape[0], -1)
        if self.use_bottleneck:
            all_s_feats = self.bottleneck_layer(all_s_feats)
            t_feats = self.bottleneck_layer(t_feats)
        all_s_preds = self.classifier_layer(all_s_feats)
        loss = self.criterion(all_s_preds, all_s_outputs)

        all_t_feats = torch.cat([t_feats]*len(s_inputs), dim=0)
        domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([all_s_feats, all_t_feats], dim=0), alpha))
        domain_labels = np.array([0] * all_s_feats.shape[0] + [1] * all_t_feats.shape[0])
        domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=t_inputs.device)
        loss += self.criterion(domain_preds, domain_labels) * 0.1
        return loss

    def inference(self, x):
        x = self.base_network(x).view(x.shape[0], -1)
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)
        return self.classifier_layer(x)

