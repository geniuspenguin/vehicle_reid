from reidlib.dataset.dataset import Veri776
from reidlib.dataset.sampler import PKSampler_coverage
from reidlib.utils.logger import Logger
from config import Config
from reidlib.models.resnet_ibn import resnet50_ibn_a
from reidlib.utils.loss import triplet_hard_loss
import torch

def cross_entropy_and_triple_thard_loss(features, pros, labels, margin=0.25, weight_tri=1, weight_ce=1):
    tri_loss = triplet_hard_loss(features, features, labels, labels, margin)
    ce_func = torch.nn.CrossEntropyLoss()
    if features.is_cuda:
        ce_func = ce_func.cuda()
    ce_loss = ce_func(pros, labels)
    print(ce_loss, tri_loss)
    return weight_tri * tri_loss + weight_ce * ce_loss

# f = torch.rand((4, 256)).cuda()
# pros = torch.randn((4, 10)).cuda()
# labels = target = torch.empty(4, dtype=torch.long).random_(10).cuda()
# print(cross_entropy_and_triple_thard_loss(f, pros, labels))

def train(model, train_loader, val_loader, lossfunc, config, logger: Logger):
    log = logger.log_and_print
    log_add_scalar = logger.add_scalar
    train_start_time = time.time()