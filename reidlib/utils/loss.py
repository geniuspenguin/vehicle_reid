import torch
import torch.nn as nn
import torch.nn.functional as F
from reidlib.utils.metrics import get_L2distance_matrix, get_L2distance_matrix_attn


def _triplet_hard_loss(qry, gry, q_ids, g_ids, margin, sqrt=True):
    # test_q = qry.unsqueeze(0).expand(qry.shape[0], gry.shape[0], qry.shape[1]).reshape(-1, qry.shape[1])
    # test_g = gry.unsqueeze(1).expand(gry.shape[0], qry.shape[0],qry.shape[1]).reshape(-1, qry.shape[1])
    # distance = F.pairwise_distance(test_q, test_g).reshape(qry.shape[0], gry.shape[0])
    distance = get_L2distance_matrix(qry, gry, sqrt)
    positive_mask = (q_ids.reshape(-1, 1) == g_ids.reshape(1, -1)).float()
    negative_mask = 1 - positive_mask
    dist_ap = distance * positive_mask
    dist_ap_hard = dist_ap.max(axis=1, keepdims=False)[0]

    dist_an = distance * negative_mask
    max_dist_an = dist_an.max()
    dist_an = distance * negative_mask + max_dist_an * (1 - negative_mask)
    dist_an_hard = dist_an.min(axis=1, keepdims=False)[0]

    tri = (dist_ap_hard - dist_an_hard + margin).clamp(min=0)
    tri_loss = tri.mean()
    return tri_loss, dist_ap_hard.mean(), dist_an_hard.mean()

def _weighted_triplet_hard_loss(distance, q_ids, g_ids, weights, margin, soft_margin=True, sqrt=True, mask=None):
    positive_mask = (q_ids.reshape(-1, 1) == g_ids.reshape(1, -1)).float()
    negative_mask = 1 - positive_mask
    dist_ap = distance * positive_mask
    dist_ap_hard, idx_ap_hard = dist_ap.max(axis=1)
    weight_ap_hard = weights[torch.arange(q_ids.shape[0]), idx_ap_hard]
    wdist_ap_hard = weight_ap_hard * dist_ap_hard

    dist_an = distance * negative_mask
    max_dist_an = dist_an.max()
    dist_an = distance * negative_mask + max_dist_an * (1 - negative_mask)
    dist_an_hard, idx_an_hard = dist_an.min(axis=1)
    weight_an_hard = weights[torch.arange(q_ids.shape[0]), idx_an_hard]
    wdist_an_hard = weight_an_hard * dist_an_hard

    if soft_margin:
        margin = weight_an_hard * weight_ap_hard * margin

    tri = (wdist_ap_hard - wdist_an_hard + margin).clamp(min=0)
    tri_loss = tri.mean()
    return tri_loss

def _weighted_triplet_loss(distance, q_ids, g_ids, weights, margin, soft_margin=True, sqrt=True, mask=None, relu_on_wtri=False):
    '''
        weighted triplet loss with batch-all sampler
    '''
    pos_mask = torch.eq(q_ids.unsqueeze(1), g_ids.unsqueeze(0))
    neg_mask = ~pos_mask

    pos_dis = distance[pos_mask]
    neg_dis = distance[neg_mask]

    pos_w = weights[pos_mask]
    neg_w = weights[neg_mask]

    if soft_margin:
        w_margin = pos_w.unsqueeze(1) * neg_w.unsqueeze(0)
        margin = margin * w_margin
    
    triplet = pos_dis.unsqueeze(1) - neg_dis.unsqueeze(0) + margin

    w_pos_dis = pos_w * pos_dis
    w_neg_dis = neg_w * neg_dis
    w_triplet = w_pos_dis.unsqueeze(1) - w_neg_dis.unsqueeze(0) + margin

    if relu_on_wtri:
        mask = (w_triplet > 0).float()
    else:
        mask = (triplet > 0).float()
    valid_triplet = w_triplet * mask
    loss = valid_triplet.mean()
    return loss

def _weighted_triplet_loss_v4(distance, q_ids, g_ids, weights, margin):
    '''
        weighted triplet loss with batch-all sampler
    '''
    pos_mask = torch.eq(q_ids.unsqueeze(1), g_ids.unsqueeze(0))
    neg_mask = ~pos_mask

    pos_dis = distance[pos_mask]
    neg_dis = distance[neg_mask]

    pos_w = weights[pos_mask]
    neg_w = weights[neg_mask]

    w_tri = pos_w.unsqueeze(1) * neg_w.unsqueeze(0)
    triplet = pos_dis.unsqueeze(1) - neg_dis.unsqueeze(0) + margin
    mask = (triplet > 0).float()
    valid_triplet = triplet * mask * w_tri
    loss = valid_triplet.mean()
    return loss

def _triplet_hard_loss_new(distance, q_ids, g_ids, margin):
    positive_mask = (q_ids.reshape(-1, 1) == g_ids.reshape(1, -1)).float()
    negative_mask = 1 - positive_mask
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


class triplet_hard_loss_base(nn.Module):
    def __init__(self, margin=1.2, sqrt=True):
        super().__init__()
        self.margin = margin
        self.sqrt = sqrt
        self.dist_ap_hard = None
        self.dist_an_hard = None

    def get_dist(self, x):
        return get_L2distance_matrix(x, x, sqrt=self.sqrt)

    def forward(self, x, labels):
        dist = self.get_dist(x)
        loss, self.dist_ap_hard, self.dist_an_hard = _triplet_hard_loss_new(
            dist, labels, labels, margin=self.margin)
        return loss

    def get_mean_hard_dist(self):
        return self.dist_ap_hard, self.dist_an_hard


class weighted_triplet_hard_loss(triplet_hard_loss_base):
    def __init__(self, margin, sqrt=True, soft_margin=True):
        super().__init__(margin=margin, sqrt=sqrt)
        self.soft_margin = soft_margin

    def forward(self, x, labels, w):
        assert w.ndim == 1
        weights = w.unsqueeze(1) * w.unsqueeze(0)
        distance = self.get_dist(x)
        loss = _weighted_triplet_hard_loss(distance, labels, labels, weights,
                                           margin=self.margin, soft_margin=self.soft_margin)
        return loss
        
    def get_mean_hard_dist(self):
        raise NotImplementedError

class weighted_triplet_batch_all_loss(triplet_hard_loss_base):
    def __init__(self, margin, sqrt=True, soft_margin=True, relu_on_wtri=False):
        super().__init__(margin=margin, sqrt=sqrt)
        self.soft_margin = soft_margin
        self.relu_on_wtri = relu_on_wtri

    def forward(self, x, labels, w):
        assert w.ndim == 1
        weights = w.unsqueeze(1) * w.unsqueeze(0)
        distance = self.get_dist(x)
        loss = _weighted_triplet_loss(distance, labels, labels, weights, relu_on_wtri=self.relu_on_wtri,
                                           margin=self.margin, soft_margin=self.soft_margin)
        return loss
        
    def get_mean_hard_dist(self):
        raise NotImplementedError

class weighted_triplet_batch_all_loss_v4(triplet_hard_loss_base):
    def __init__(self, margin, sqrt=True, soft_margin=True, relu_on_wtri=False):
        super().__init__(margin=margin, sqrt=sqrt)
        self.soft_margin = soft_margin
        self.relu_on_wtri = relu_on_wtri

    def forward(self, x, labels, w):
        assert w.ndim == 1
        weights = w.unsqueeze(1) * w.unsqueeze(0)
        distance = self.get_dist(x)
        loss = _weighted_triplet_loss_v4(distance, labels, labels, weights, margin=self.margin)
        return loss
        
    def get_mean_hard_dist(self):
        raise NotImplementedError

class triplet_batch_all_loss(triplet_hard_loss_base):
    def __init__(self, margin, sqrt=True, soft_margin=True, relu_on_wtri=False):
        super().__init__(margin=margin, sqrt=sqrt)
        self.soft_margin = soft_margin
        self.relu_on_wtri = relu_on_wtri

    def forward(self, x, labels):
        weights = torch.ones((x.shape[0], x.shape[0])).cuda()
        distance = self.get_dist(x)
        loss = _weighted_triplet_loss(distance, labels, labels, weights, relu_on_wtri=self.relu_on_wtri,
                                           margin=self.margin, soft_margin=self.soft_margin)
        return loss
        
    def get_mean_hard_dist(self):
        raise NotImplementedError


class tri_hard_attn(triplet_hard_loss_base):
    def __init__(self, temp=0.1, margin=1.2, sqrt=True):
        super().__init__(margin, sqrt)
        self.temp = temp

    def get_dist(self, x, x_mask):
        return get_L2distance_matrix_attn(x, x, x_mask, x_mask, temp=self.temp, sqrt=self.sqrt)

    def forward(self, x, x_mask, labels):
        dist = self.get_dist(x, x_mask)
        loss, self.dist_ap_hard, self.dist_an_hard = _triplet_hard_loss_new(
            dist, labels, labels, margin=self.margin)
        return loss, dist


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

class weight_cross_entropy(nn.Module):
    def __init__(self, mask_thres, eps=1):
        super().__init__()
        self.mask_thres = mask_thres
        self.eps = eps

    def forward(self, logits, labels, weight):
        ''' logits(B, C), weight(B)
        '''
        assert labels.dtype == torch.long
        mask = weight >= self.mask_thres
        # print(mask.sum())
        logp = -torch.log_softmax(logits, dim=1)
        onehot = F.one_hot(labels, num_classes=logp.shape[1])
        logp += self.eps
        logpy = (logp * onehot).sum(axis=1)
        logpy = logpy.reshape(-1)
        loss = logpy * mask
        loss = loss.sum() / (mask.sum() + self.eps)
        return loss

# multi-name class, no time to fix
thres_cross_entropy = weight_cross_entropy

class thres_weight_cross_entropy(nn.Module):
    def __init__(self, mask_thres, eps=1):
        super().__init__()
        self.mask_thres = mask_thres
        self.eps = eps

    def forward(self, logits, labels, weight):
        ''' logits(B, C), weight(B)
        '''
        assert labels.dtype == torch.long
        mask = weight >= self.mask_thres
        # print(mask.sum())
        logp = -torch.log_softmax(logits, dim=1)
        onehot = F.one_hot(labels, num_classes=logp.shape[1])
        logp += self.eps
        logpy = (logp * onehot).sum(axis=1)
        logpy = logpy.reshape(-1)
        loss = logpy * mask * weight
        loss = loss.sum() / (mask.sum() + self.eps)
        return loss
        

if __name__ == '__main__':
    # logit = torch.randn(4, 5)
    # label = torch.randint(0, 5, (4,)).long()
    # weight = torch.randn(4)
    # wce = weight_cross_entropy(0.3)
    # ce = nn.CrossEntropyLoss()
    # loss1 = wce(logit, label, weight)
    # loss2 = ce(logit, label)
    # print(loss1, loss2)
    from reidlib.utils.parsing import get_weight
    a = torch.tensor([
        [1, 0, 0],
        [0, 3, 0],
        [0, 0, 0],
        [1, 1, 0],
        [1, 10, 0],
        [0, 0, 0]
    ], dtype=torch.float32)
    mask = torch.randint(0, 2, (6, 4, 16, 16))
    w = get_weight(mask)
    print(w.shape, w[:, 1:2].shape, w[:, 1].shape)
    la = torch.tensor([1, 1, 1, 2, 2, 2])
    lossfunc_my = weighted_triplet_hard_loss(margin=1.2)
    loss = lossfunc_my(a, w[:, 1], la)

