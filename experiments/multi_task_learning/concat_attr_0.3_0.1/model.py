from reidlib.models.resnet import ResNet50
import torch
from torch import nn
import math
from torchvision import models
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from reidlib.models.weight_init import weights_init_kaiming, weights_init_classifier
from torchsummary import summary


class Baseline(nn.Module):
    in_planes = 1024

    def __init__(self, num_classes, last_stride=1, pretrained=True, nr_type=10, nr_color=11):
        super(Baseline, self).__init__()
        self.pretrained = pretrained
        self.backbone = ResNet50(last_stride)
        if self.pretrained:
            self.backbone._ImageNet_pretrained()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.feat_fc = nn.Linear(2048, self.in_planes)
        self.feat_fc.apply(weights_init_kaiming)
        self.bn = nn.BatchNorm1d(self.in_planes)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc = nn.Linear(self.in_planes * 2, self.num_classes, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)

        self.type_fc = nn.Linear(self.in_planes, self.in_planes // 2)
        self.type_bn = nn.BatchNorm1d(self.in_planes // 2)
        self.type_cls_fc = nn.Linear(self.in_planes // 2, nr_type, bias=False)
        self.type_fc.apply(weights_init_kaiming)
        self.type_bn.apply(weights_init_kaiming)
        self.type_cls_fc.apply(weights_init_classifier)

        self.color_fc = nn.Linear(self.in_planes, self.in_planes // 2)
        self.color_bn = nn.BatchNorm1d(self.in_planes //2)
        self.color_cls_fc = nn.Linear(self.in_planes // 2, nr_color, bias=False)
        self.color_fc.apply(weights_init_kaiming)
        self.color_bn.apply(weights_init_kaiming)
        self.color_cls_fc.apply(weights_init_classifier)

    def forward(self, x):
        x = self.backbone(x)
        global_feats = self.gap(x)  # (b, 2048, 1, 1)
        global_feats = global_feats.view(
            global_feats.shape[0], -1)  # flatten to (bs, 2048)
        feats = self.feat_fc(global_feats)
        feats = self.bn(feats)  # normalize for angular softmax

        f_type = self.type_fc(feats)
        f_type = self.type_bn(f_type)

        f_color = self.color_fc(feats)
        f_color = self.color_bn(f_color)

        logits_type = self.type_cls_fc(f_type)
        logits_color = self.color_cls_fc(f_color)

        feats = torch.cat([feats, f_type, f_color], axis=1)

        if self.training:
            cls_score = self.fc(feats)
            return feats, cls_score, logits_type, logits_color
        else:
            feats = nn.functional.normalize(feats, dim=1, p=2)
            return feats, logits_type, logits_color


if __name__ == '__main__':
    summary(Baseline(10), (3, 224, 224))