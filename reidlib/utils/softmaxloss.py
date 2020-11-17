import torch
from reidlib.models.weight_init import weights_init_kaiming, weights_init_classifier
import torch.nn as nn
import torch.nn.functional as F
import math

class AMSoftmax(nn.Module):
    def __init__(self, mid_num, nr_class, scale, margin, eps=1e-8):
        super().__init__()
        self.nr_cls = nr_class
        self.scale = scale
        self.margin = margin
        self.eps = eps

        self.w = nn.Parameter(torch.randn((mid_num, nr_class)), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.w, gain=1)
        # self.apply(weights_init_kaiming)
        

    def forward(self, x, y):
        y_onehot = nn.functional.one_hot(y, self.nr_cls)
        # norm_w = torch.sqrt(torch.sum(self.w ** 2, axis=0)).reshape(1, -1)   # (1, C)
        # norm_x = torch.sqrt(torch.sum(x ** 2, axis=1)).reshape(-1, 1)   # (B, 1)
        norm_w = torch.norm(self.w, p=2, dim=0).clamp(min=self.eps).reshape(1, -1)
        norm_x = torch.norm(x, p=2, dim=1).clamp(min=self.eps).reshape(-1, 1)
        score = torch.matmul(x, self.w) #  (B, C)
        score = score / norm_w
        score = score / norm_x
        margin = self.margin * y_onehot
        m_score = self.scale * (score - margin)
        # score = torch.exp(score)
        # p = score / torch.sum(score, axis=1, keepdim=True)
        # p = p * y_onehot
        # p = torch.sum(p, axis=1)
        loss = self.ce(m_score, y)
        return loss, score

class ArcfaceLoss(nn.Module):
    def __init__(self, mid_num, nr_class, scale, margin, eps=1e-8):
        super().__init__()
        self.nr_cls = nr_class
        self.scale = scale
        self.margin = margin
        self.eps = eps

        self.w = nn.Parameter(torch.randn((mid_num, nr_class)), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.w, gain=1)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # self.apply(weights_init_kaiming)
        

    def forward(self, x, y):
        y_onehot = nn.functional.one_hot(y, self.nr_cls)
        # norm_w = torch.sqrt(torch.sum(self.w ** 2, axis=0)).reshape(1, -1)   # (1, C)
        # norm_x = torch.sqrt(torch.sum(x ** 2, axis=1)).reshape(-1, 1)   # (B, 1)
        norm_w = torch.norm(self.w, p=2, dim=0).clamp(min=self.eps).reshape(1, -1)
        norm_x = torch.norm(x, p=2, dim=1).clamp(min=self.eps).reshape(-1, 1)
        cosine = torch.matmul(x, self.w) #  (B, C)
        cosine = cosine / norm_w
        cosine = cosine / norm_x

        sine = torch.sqrt(1 - (cosine ** 2).clamp(0, 1))
        margin = cosine * self.cos_m - sine * self.sin_m

        m_score = cosine * (1 - y_onehot) + margin * y_onehot
        m_score = self.scale * (m_score - margin)
        # score = torch.exp(score)
        # p = score / torch.sum(score, axis=1, keepdim=True)
        # p = p * y_onehot
        # p = torch.sum(p, axis=1)
        loss = self.ce(m_score, y)
        return loss, cosine
        
if __name__ == '__main__':
    loss = AMSoftmax(mid_num=50, nr_class=10, scale=30, margin=0.25)

    for name, param in loss.named_parameters():
        print('name:', name, '\n', 'param shape:', param.shape)
    y = torch.arange(0, 5)
    x = torch.randn((5, 50))
    l = loss(x, y)
    print(l)