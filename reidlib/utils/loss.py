import torch
import torch.nn as nn
import torch.nn.functional as F
from reidlib.utils.metrics import get_L2distance_matrix


def _triplet_hard_loss(qry, gry, q_ids, g_ids, margin, sqrt=True, valid_ap=None):
    # test_q = qry.unsqueeze(0).expand(qry.shape[0], gry.shape[0], qry.shape[1]).reshape(-1, qry.shape[1])
    # test_g = gry.unsqueeze(1).expand(gry.shape[0], qry.shape[0],qry.shape[1]).reshape(-1, qry.shape[1])
    # distance = F.pairwise_distance(test_q, test_g).reshape(qry.shape[0], gry.shape[0])
    distance = get_L2distance_matrix(qry, gry, sqrt)
    positive_mask = (q_ids.reshape(-1, 1) == g_ids.reshape(1, -1)).float()
    negative_mask = 1 - positive_mask
    if not valid_ap and qry.shape[0] == gry.shape[0]:
        valid_ap = 1 - torch.diag(torch.ones(qry.shape[0]))
        if positive_mask.is_cuda:
            valid_ap = valid_ap.cuda()
        positive_mask = valid_ap * positive_mask
    dist_ap = distance * positive_mask
    dist_ap_hard = dist_ap.max(axis=1, keepdims=False)[0]

    dist_an = distance * negative_mask
    max_dist_an = dist_an.max()
    dist_an = distance * negative_mask + max_dist_an * (1 - negative_mask)
    dist_an_hard = dist_an.min(axis=1, keepdims=False)[0]

    tri = (dist_ap_hard - dist_an_hard + margin).clamp(min=0)
    tri_loss = tri.mean()
    return tri_loss, dist_ap_hard.mean(), dist_an_hard.mean()


# def triplet_hard_loss(feat, labels, margin, sqrt=True):
#     return _triplet_hard_loss(feat, feat, labels, labels, margin=margin, sqrt=sqrt)

class triplet_hard_loss(nn.Module):
    def __init__(self, margin=0.25, sqrt=True):
        super().__init__()
        self.margin = margin
        self.sqrt = sqrt
        self.dist_ap_hard = None
        self.dist_an_hard = None

    def forward(self, x, labels):
        loss, self.dist_ap_hard, self.dist_an_hard = _triplet_hard_loss(
            x, x, labels, labels, margin=self.margin, sqrt=self.sqrt)
        return loss
    
    def get_mean_hard_dist(self):
        return self.dist_ap_hard, self.dist_an_hard


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feat_dim=2048):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(
            torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(
            0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(
            batch_size, self.num_classes) + torch.pow(self.centers, 2).sum(
                dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes, device=x.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12,
                                max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss

if __name__ == '__main__':
    la = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    a = torch.tensor([
        [1, 0, 0],
        [0, 3, 0],
        [0, 0, 0],
        [1, 1, 0],
        [1, 10,0],
        [0, 0, 0]
    ], dtype=torch.float32)
    la = torch.tensor([1,1,1,2,2,2])
    lossfunc_my = triplet_hard_loss()
    loss = lossfunc_my(a, la)
