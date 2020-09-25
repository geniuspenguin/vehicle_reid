import torch
from torch.nn.functional import interpolate
import torch.nn as nn

'''
0: 背景
1: 正面
2: 背面
3: 顶面
4: 侧面
'''

def get_weight(masks):
    if masks.dtype != torch.float:
        masks = masks.float()
    total_mask = torch.sum(masks, (1, 2, 3)).reshape(masks.shape[0], 1)
    part_mask = torch.sum(masks, (2, 3))
    weights = part_mask / (total_mask+1)
    return weights

def get_reshape_mask(mask: torch.Tensor, target_shape):
    if mask.dtype != torch.float:
        mask = mask.float()
    return interpolate(mask, target_shape, mode='bilinear', align_corners=False)

class MaskAveragePooling(nn.Module):
    def __init__(self, eps=1):
        super().__init__()
        self.eps = eps

    def forward(self, feat, mask):
        if mask.dtype != torch.float:
            mask = mask.float()
        assert mask.shape[1] == 1
        feat = feat * mask # B C H W
        total_mask = torch.sum(mask, (2, 3)) + self.eps # B 1
        feat = torch.sum(feat, (2, 3)) # B C
        return feat / total_mask


if __name__ == '__main__':
    masks = torch.randint(0, 2, (10, 4, 16, 16))
    img = torch.randn((10, 64, 16, 16)) * 100
    mmap = MaskAveragePooling()
    img = mmap(img, masks[:, 1:2, ...])
    print(img.shape)
    print(get_weight(masks), get_weight(masks).shape, get_weight(masks).sum(axis=1))




