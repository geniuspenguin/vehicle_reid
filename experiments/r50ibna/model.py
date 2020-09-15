from reidlib.models.resnet_ibn import resnet50_ibn_a
import torch.nn as nn
import torch


class r50ibn_reid_cls(nn.Module):
    def __init__(self, nr_class, nr_feature=2048):
        super().__init__()
        self.backbone = resnet50_ibn_a(pretrained=True)
        self.fc1 = nn.Linear(2048, nr_feature)
        self.bn1 = nn.BatchNorm1d(2048)
        self.gap = nn.AdaptiveAvgPool2d(1)
        nn.init.zeros_(self.bn1.bias)
        self.bn1.bias.requires_grad = False
        self.fc_cls = nn.Linear(nr_feature, nr_class)

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)  # (b, 2048, 1, 1)
        x = x.view(x.shape[0], -1)  # flatten to (bs, 2048)
        f = self.fc1(x)
        f_bn = self.bn1(f)
        p = self.fc_cls(f_bn)
        return f, f_bn, p
