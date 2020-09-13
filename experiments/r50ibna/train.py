from reidlib.dataset.dataset import Veri776
from reidlib.dataset.sampler import PKSampler_coverage
from reidlib.utils.logger import Logger
from config import Config
from reidlib.models.resnet_ibn import resnet50_ibn_a
from reidlib.utils.loss import triplet_hard_loss, CenterLoss
import torch
from model import r50ibn_reid_cls
import time


logger = Logger(log_dir=Config.log_dir)


# def cross_entropy_and_triple_thard_loss(features, pros, labels, margin=0.25, weight_tri=1, weight_ce=1, weight_center=0.0005):
#     tri_loss = triplet_hard_loss(features, labels, margin)
#     ce_func = torch.nn.CrossEntropyLoss()
#     ce_loss = ce_func(pros, labels)
#     cl_func = CenterLoss()
#     if features.is_cuda:
#         ce_func = ce_func.cuda()
#         cl_func = cl_func.cuda()
#     ce_loss, cl_loss = ce_func(features, labels), cl_func(features, labels)
#     # print(ce_loss, tri_loss)
#     return weight_tri * tri_loss + weight_ce * ce_loss + weight_center * cl_loss

# f = torch.rand((4, 256)).cuda()
# pros = torch.randn((4, 10)).cuda()
# labels = target = torch.empty(4, dtype=torch.long).random_(10).cuda()
# print(cross_entropy_and_triple_thard_loss(f, pros, labels))


def get_weight_decay_param(model):
    conv_and_fc_param_list = []
    conv_and_fc_param_name_list = []
    for name, param in model.named_parameters():
        if 'bn' not in name:
            conv_and_fc_param_list.append(param)
            conv_and_fc_param_name_list.append(name)
    logger.info(tag='setting', 'weight decay {} for '.format(Config.weight_decay) + f"{conv_and_fc_param_name_list}")
    weight_decay_setting = [
        {
            'params': conv_and_fc_param_list,
            'weight_decay': Config.weight_decay,
        }
    ]
    return weight_decay_setting


def lr_multi_func(epoch):
    if epoch <= 10:
        return epoch / 10
    if epoch <= 40:
        return 1
    if epoch <= 70:
        return 0.1
    else:
        return 0.01


def train(model, train_loader, val_loader):
    log = logger.info
    log_add_scalar = logger.add_scalar
    train_start_time = time.time()
    optimizer = torch.optim.Adam(weight_decay_setting_list, lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multi_func)

    best_mAP = 0.0
    best_mAP_epoch = None

    logger.info('trainning', 'start trainning')
    for epoch in range(1, Config.epoch + 1):
        epoch_start_time = time.time()
        logger.info('trainning', 'start epoch {:>4}!'.format(epoch))
        for i, (imgs, labels, cids) in enumerate(train_loader):
            batch_start_time = time.time()
            f, f_bn, p = 


