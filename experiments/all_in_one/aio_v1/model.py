from reidlib.models.resnet import ResNet50
import torch
from torch import nn
import math
from torchvision import models
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from reidlib.models.weight_init import weights_init_kaiming, weights_init_classifier
from reidlib.utils.parsing import get_reshape_mask, MaskAveragePooling

class Backbone(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride=1, pretrained=True, nr_type=10, nr_color=11):
        super(Backbone, self).__init__()
        self.pretrained = pretrained
        self.backbone = ResNet50(last_stride)
        if self.pretrained:
            self.backbone._ImageNet_pretrained()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.bn = nn.BatchNorm1d(self.in_planes)
        self.bn.bias.requires_grad_(False)  # no shift
        # self.fc = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bn.apply(weights_init_kaiming)
        # self.fc.apply(weights_init_classifier)

        self.type_fc = nn.Linear(self.in_planes, nr_type)
        self.type_fc.apply(weights_init_classifier)

        self.color_fc = nn.Linear(self.in_planes, nr_color)
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
            return x, logits_type, logits_color
        else:
            return x, logits_type, logits_color

class main_branch(nn.Module):
    def __init__(self, nr_class, in_planes=2048):
        super().__init__()
        self.in_planes = in_planes
        self.nr_class = nr_class

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(self.in_planes)

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
        gapx = self.bn(gapx)

        # feats = self.fc_1024(gapx)
        # feats = self.bn_1024(feats)

        if self.training:
            logits = self.fc_cls(gapx)
            return gapx, logits
        else:
            return gapx

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
    a = torch.arange(16).reshape(1, 1, 4, 4)
    b = torch.randn((1, 1, 4, 4))
    x = torch.cat([a, b], axis=1)
    x = torch.cat([x, x], axis=0)
    mask = torch.Tensor([
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 0, 0]
    ]).reshape(1, 1, 4, 4)
    print(a)
    mask = torch.cat([mask, mask], axis=0)
    model = parsing_branch(2, 2)
    model.eval()
    model(x, mask)