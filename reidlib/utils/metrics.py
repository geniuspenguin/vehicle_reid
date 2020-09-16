import torch
import numpy as np


def get_L2distance_matrix(a: torch.tensor, b: torch.tensor, sqrt=True, epsilon=1e-20):
    # a(A, X) b(B, X)
    ab = torch.matmul(a, b.t())
    a_square = (a ** 2).sum(axis=1, keepdims=True)
    b_square = (b ** 2).sum(axis=1, keepdims=True).permute((1, 0))
    square_l2 = a_square + b_square - 2*ab
    square_l2 = square_l2.clamp(min=0)
    if sqrt:
        zero_mask = torch.eq(square_l2, 0).float()
        square_l2 = square_l2 + zero_mask * epsilon
        l2 = torch.sqrt(square_l2)
        l2 = l2 * (1 - zero_mask)
        return l2
    return square_l2

# def get_L2distance_matrix_attn(a: torch.tensor, b: torch.tensor, a_mask: torch.tensor, b_mask: torch.tensor)


def get_L2distance_matrix_numpy(a: np.ndarray, b: np.ndarray, sqrt=True):
    ab = np.matmul(a, b.T)
    a_square = (a ** 2).sum(axis=1, keepdims=True)
    b_square = (b ** 2).sum(axis=1, keepdims=True)
    b_square = np.transpose(b_square, (1, 0))
    square_l2 = a_square + b_square - 2*ab
    square_l2 = np.clip(square_l2, 0, None)
    if sqrt:
        l2 = np.sqrt(square_l2)
        return l2
    return square_l2


def get_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """
        offical implement by Liu,
        Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    def percent(x): return x/100
    return list(map(percent, res))
