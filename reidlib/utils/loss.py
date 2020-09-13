import torch
import torch.nn as nn
import torch.nn.functional as F


def get_L2distance_matrix(a: torch.tensor, b: torch.tensor, sqrt=True, epsilon=1):
    # a(A, X) b(B, X)
    ab = torch.matmul(a, b.t())
    a_square = (a ** 2).sum(axis=1, keepdims=True)
    b_square = (b ** 2).sum(axis=1, keepdims=True).permute((1, 0))
    square_l2 = a_square + b_square - 2*ab
    square_l2 = square_l2.clamp(min=0)
    if sqrt:
        l2 = torch.sqrt(square_l2)
        return l2
    return square_l2


def _triplet_hard_loss(qry, gry, q_ids, g_ids, margin, sqrt=True, valid_ap=None):
    distance = get_L2distance_matrix(qry, gry, sqrt)
    print(distance, distance.dtype)
    positive_mask = (q_ids.reshape(-1, 1) == g_ids.reshape(1, -1)).float()
    negative_mask = 1 - positive_mask
    if not valid_ap and qry.shape[0] == gry.shape[0]:
        valid_ap = 1 - torch.diag(torch.ones(qry.shape[0]))
        print(type(positive_mask))
        if positive_mask.is_cuda:
            valid_ap = valid_ap.cuda()
        positive_mask = valid_ap * positive_mask
    dist_ap = distance * positive_mask
    dist_ap_hard = dist_ap.max(axis=1, keepdims=False)[0]

    dist_an = distance * negative_mask
    max_dist_an = dist_an.max()
    dist_an = distance * negative_mask + max_dist_an * (1 - negative_mask)
    dist_an_hard = dist_an.min(axis=1, keepdims=False)[1]

    tri = (dist_an_hard - dist_ap_hard + margin).clamp(min=0)
    tri_loss = tri.mean()
    return tri_loss


# def triplet_hard_loss(feat, labels, margin, sqrt=True):
#     return _triplet_hard_loss(feat, feat, labels, labels, margin=margin, sqrt=sqrt)

class triplet_hard_loss(nn.Module):
    def __init__(self, margin=0.25, sqrt=True):
        self.margin = margin
        self.sqrt = sqrt

    def foward(self, x, labels):
        loss = _triplet_hard_loss(
            x, x, labels, labels, margin=self.margin, sqrt=self.sqrt)
        return loss


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
        distmat.addmm_(1, -2, x, self.centers.t())

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
