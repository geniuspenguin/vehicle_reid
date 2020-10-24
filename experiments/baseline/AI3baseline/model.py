from reidlib.models.resnet_ibn_a import resnet50_ibn_a
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
        self.backbone = resnet50_ibn_a(last_stride, pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.bn = nn.BatchNorm1d(self.in_planes)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)

    def forward(self, x):
        x = self.backbone(x)
        global_feats = self.gap(x)  # (b, 2048, 1, 1)
        global_feats = global_feats.view(
            global_feats.shape[0], -1)  # flatten to (bs, 2048)
        feats = self.bn(global_feats)  # normalize for angular softmax
        if self.training:
            cls_score = self.fc(feats)
            return global_feats, cls_score
        else:
            return feats

if __name__ == '__main__':
    model = Baseline(num_classes=576)
    from torchsummary import summary
    summary(model, (3, 224, 224))
