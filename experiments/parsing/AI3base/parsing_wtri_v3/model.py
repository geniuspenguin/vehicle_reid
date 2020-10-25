from reidlib.models.resnet_ibn_a import resnet50_ibn_a, r50ibna_path
import torch
from torch import nn
import math
from torchvision import models
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from reidlib.models.weight_init import weights_init_kaiming, weights_init_classifier
from reidlib.utils.parsing import get_reshape_mask, MaskAveragePooling


class Backbone(nn.Module):
    def __init__(self, last_stride=1, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.backbone = resnet50_ibn_a(last_stride)
        if pretrained:
            self.backbone.load_param(r50ibna_path)

    def forward(self, x):
        x = self.backbone(x)
        return x

class main_branch(nn.Module):
    def __init__(self, nr_class, in_planes=2048):
        super().__init__()
        self.in_planes = in_planes
        self.nr_class = nr_class

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(self.in_planes)
        self.bn.bias.requires_grad_(False)

        # self.fc_1024 = nn.Linear(self.in_planes, 1024)
        # self.bn_1024 = nn.BatchNorm1d(1024)

        self.fc_cls = nn.Linear(self.in_planes, nr_class, bias=False)

        self.bn.apply(weights_init_kaiming)
        # self.fc_1024.apply(weights_init_kaiming)
        # self.bn_1024.apply(weights_init_kaiming)
        self.fc_cls.apply(weights_init_classifier)

    def forward(self, x):
        gapx = self.gap(x)
        gapx = gapx.view(gapx.shape[0], -1)
        feat = self.bn(gapx)

        # feats = self.fc_1024(gapx)
        # feats = self.bn_1024(feats)

        if self.training:
            logits = self.fc_cls(feat)
            return gapx, logits
        else:
            feat = nn.functional.normalize(feat, dim=1, p=2)
            return feat

class parsing_branch(nn.Module):
    def __init__(self, nr_class, in_planes, midnum):
        super().__init__()
        self.nr_class = nr_class
        self.in_planes = in_planes
        self.maskap = MaskAveragePooling()
        self.bn = nn.BatchNorm1d(self.in_planes)
        self.fc_mid = nn.Linear(self.in_planes, midnum)
        self.bn_mid = nn.BatchNorm1d(midnum)
        self.fc_cls = nn.Linear(midnum, self.nr_class, bias=False)

        self.fc_mid.apply(weights_init_kaiming)
        self.bn_mid.apply(weights_init_kaiming)
        self.fc_cls.apply(weights_init_classifier)
        self.bn.apply(weights_init_kaiming)
        
    def forward(self, x, mask):
        mask = get_reshape_mask(mask, x.shape[-2:])
        mapx = self.maskap(x, mask)
        mapx = self.bn(mapx)

        feats = self.fc_mid(mapx)
        feats = self.bn_mid(feats)

        if self.training:
            logits = self.fc_cls(feats)
            return feats, logits
        else:
            return feats

if __name__ == '__main__':
    a = torch.randn((1, 3, 224, 224))
    model = Backbone().cpu()
    model.eval()
    b = model(a)
    print(b.shape)