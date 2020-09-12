import torch
def get_L2distance_matrix(a: torch.tensor, b: torch.tensor, sqrt=True, epsilon=1):
    # a(A, X) b(B, X)
    ab = torch.matmul(a, b.t())
    a_square = (a ** 2).sum(axis=1, keepdims=True)
    b_square = (b ** 2).sum(axis=1, keepdims=True).permute((1, 0))
    square_l2 = a_square + b_square - 2*ab
    square_l2 = square_l2.clamp(min=0)
    if sqrt:
        # zero_mask = torch.eq(square_l2, 0).float()
        # square_l2 += zero_mask * epsilon
        l2 = torch.sqrt(square_l2)
        # l2 *= (1 - zero_mask)
        return l2
    return square_l2

def triplet_hard_loss(qry, gry, q_ids, g_ids, margin, sqrt=True, valid_ap=None):
    distance = get_L2distance_matrix(qry, gry, sqrt)
    print(distance, distance.dtype)
    positive_mask = (q_ids.reshape(-1, 1) == g_ids.reshape(1, -1)).float()
    negative_mask = 1- positive_mask
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
    
if __name__ == '__main__':
    a = torch.randn((8, 256), dtype=torch.float32).cuda()
    b = torch.randn((8, 256), dtype=torch.float32).cuda()
    # a = torch.ones((4, 256), dtype=torch.float32).cuda()
    # b = torch.cat([torch.zeros(4,256), torch.full((4,256),3, dtype=torch.float32)], axis=0).cuda()
    a_id = torch.randint(0, 12, (8, 1), dtype=torch.long).cuda()
    b_id = torch.randint(0, 12, (8, 1), dtype=torch.long).cuda()
    loss = triplet_hard_loss(a, b, a_id, b_id, 0.25)
    print(loss)


