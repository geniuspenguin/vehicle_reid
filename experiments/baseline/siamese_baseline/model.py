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

    def __init__(self, num_classes, last_stride=1, pretrained=True):
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

        self.fc_attn = nn.Linear(self.in_planes, self.in_planes)
        self.bn_attn = nn.BatchNorm1d(self.in_planes)
        self.bn_attn.bias.requires_grad_(False)
        self.sigmoid = nn.Sigmoid()

        self.fc_attn.apply(weights_init_kaiming)
        self.bn_attn.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.backbone(x)
        global_feats = self.gap(x)  # (b, 2048, 1, 1)
        global_feats = global_feats.view(
            global_feats.shape[0], -1)  # flatten to (bs, 2048)
        feats = self.bn(global_feats)  # normalize for angular softmax

        mask = self.fc_attn(feats)
        mask = self.bn_attn(mask)
        mask = self.sigmoid(mask)

        if self.training:
            cls_score = self.fc(feats)
            return feats, cls_score, mask
        else:
            feats = nn.functional.normalize(feats, dim=1, p=2)
            return feats, mask

class BinaryClassifier(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.in_planes = in_planes
        self.bce = nn.Linear(self.in_planes, 2, bias=False)
        self.bce.apply(weights_init_classifier)

    def forward(self, a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(0)
        subtract = torch.abs(a - b)
        subtract = subtract.reshape(-1, self.in_planes)
        logit = self.bce(subtract)
        return logit

def idlabel2pairlabel(labels1, labels2):
    labels1 = labels1.unsqueeze(1)
    labels2 = labels2.unsqueeze(0)
    same_mat = torch.eq(labels1, labels2).long()
    return same_mat.reshape(-1)
