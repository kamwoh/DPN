import torch
import torch.nn as nn
from torchvision.models import alexnet

from models import register_network


@register_network('alexnet')
class AlexNet(nn.Module):
    def __init__(self, nbit, nclass, pretrained=False, freeze_weight=False, **kwargs):
        super(AlexNet, self).__init__()

        model = alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool = model.avgpool
        fc = []
        for i in range(6):
            fc.append(model.classifier[i])
        self.fc = nn.Sequential(*fc)

        in_features = model.classifier[6].in_features
        self.ce_fc = nn.Linear(in_features, nclass)
        self.hash_fc = nn.Linear(in_features, nbit, bias=False)

        nn.init.normal_(self.hash_fc.weight, std=0.01)
        # nn.init.zeros_(self.hash_fc.bias)

        self.extrabit = 0

        if freeze_weight:
            for param in self.features.parameters():
                param.requires_grad_(False)
            for param in self.fc.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        u = self.ce_fc(x)
        v = self.hash_fc(x)
        return u, v

