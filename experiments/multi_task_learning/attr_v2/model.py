from reidlib.models.resnet import ResNet50
import torch
from torch import nn
import math
from torchvision import models
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from reidlib.models.weight_init import weights_init_kaiming, weights_init_classifier


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride=1, pretrained=True, nr_type=10, nr_color=11):
        super(Baseline, self).__init__()
        self.pretrained = pretrained
        self.backbone = ResNet50(last_stride)
        if self.pretrained:
            self.backbone._ImageNet_pretrained()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.bn = nn.BatchNorm1d(self.in_planes)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)

        self.type_fc = nn.Linear(self.in_planes, nr_type, bias=True)
        self.type_fc.apply(weights_init_classifier)

        self.color_fc = nn.Linear(self.in_planes, nr_color, bias=True)
        self.color_fc.apply(weights_init_classifier)

    def forward(self, x):
        x = self.backbone(x)
        global_feats = self.gap(x)  # (b, 2048, 1, 1)
        global_feats = global_feats.view(
            global_feats.shape[0], -1)  # flatten to (bs, 2048)
        feats = self.bn(global_feats)  # normalize for angular softmax
        logits_type = self.type_fc(feats)
        logits_color = self.color_fc(feats)
        if self.training:
            cls_score = self.fc(feats)
            return feats, cls_score, logits_type, logits_color
        else:
            return feats, logits_type, logits_color
