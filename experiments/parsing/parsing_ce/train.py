from reidlib.dataset.dataset import Veri776_train, Veri776_test
from reidlib.dataset.sampler import PKSampler
from reidlib.utils.logger import Logger, sec2min_sec, model_summary
from config import Config, config_info
from reidlib.models.resnet_ibn import resnet50_ibn_a
from reidlib.utils.loss import triplet_hard_loss, weighted_triplet_hard_loss, weight_cross_entropy
from reidlib.utils.metrics import get_cmc_map, get_L2distance_matrix_numpy, accuracy
import torch
from model import Backbone, main_branch, parsing_branch
import time
import numpy as np
import torchvision.transforms as transforms
import collections
import os
from tqdm import tqdm
from reidlib.utils.utils import no_grad_func
import argparse
import torch.cuda.amp as amp
from reidlib.utils.parsing import get_weight
from reidlib.utils.timer import wait

batch_step = 1
logger = Logger(log_dir=Config.log_dir)


def check_config_dir():
    for dir in [Config.experiment_dir, Config.data_dir, Config.model_dir, Config.log_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)


def mean(vlist: list):
    return sum(vlist)/len(vlist)


def parm_list_with_Wdecay(model):
    conv_and_fc_param_list, bn_param_list = [], []
    for name, param in model.named_parameters():
        if 'bn' in name:
            bn_param_list.append(param)
        else:
            conv_and_fc_param_list.append(param)

    param_list = [
        {
            'params': conv_and_fc_param_list,
            'weight_decay': Config.weight_decay,
        },
        {
            'params': bn_param_list,
            'weight_decay': 0.0,
        },
    ]
    return param_list


def parm_list_with_Wdecay_multi(models):
    conv_and_fc_param_list, bn_param_list = [], []
    for model in models:
        for name, param in model.named_parameters():
            if 'bn' in name:
                bn_param_list.append(param)
            else:
                conv_and_fc_param_list.append(param)

    param_list = [
        {
            'params': conv_and_fc_param_list,
            'weight_decay': Config.weight_decay,
        },
        {
            'params': bn_param_list,
            'weight_decay': 0.0,
        },
    ]
    return param_list


def lr_multi_func(epoch):
    epoch += 1
    if epoch <= 10:
        return epoch / 10
    if epoch <= 40:
        return 1
    if epoch <= 70:
        return 0.1
    else:
        return 0.01


@no_grad_func
def test(model, branches, test_loader, losses, epoch, nr_query=Config.nr_query):
    '''
    return: cmc1, mAP
    test model on testset and save result to log.
    '''
    val_start_time = time.time()
    model.eval()
    all_features, all_labels, all_cids = [], [], []
    history = collections.defaultdict(list)
    for branch in branches:
        branch.eval()
    for i, (imgs, labels, cids) in tqdm(enumerate(test_loader), desc='testing on epoch-{}'.format(epoch), total=len(test_loader)):
        imgs, labels, cids = imgs.cuda(), labels.cuda(), cids.cuda()
        x = model(imgs)
        f_main = branches[0](x)
        triplet_hard_loss = losses['triplet_hard_loss'][0](f_main, labels)
        history['triplet_hard_loss'].append(float(triplet_hard_loss))
        all_features.append(f_main.cpu().detach().numpy())
        all_labels.append(labels.cpu().detach().numpy())
        all_cids.append(cids.cpu().detach().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_cids = np.concatenate(all_cids, axis=0)
    q_f, g_f = all_features[:nr_query], all_features[nr_query:]
    q_ids, g_ids = all_labels[:nr_query], all_labels[nr_query:]
    q_cids, g_cids = all_cids[:nr_query], all_cids[nr_query:]

    print('Computing CMC and mAP')
    distance_matrix = get_L2distance_matrix_numpy(q_f, g_f)
    cmc, mAP = get_cmc_map(distance_matrix, q_ids, g_ids, q_cids, g_cids)
    val_end_time = time.time()
    time_spent = sec2min_sec(val_start_time, val_end_time)

    text = 'testing epoch {:>3}, time spent: [{:>3}mins{:>3}s]:##'.format(
        epoch, time_spent[0], time_spent[1])
    text += '|CMC1:{:>5.4f} |mAP:{:>5.4f}'.format(cmc[0], mAP)
    for k, vlist in history.items():
        v = float(mean(vlist))
        text += '|{}:{:>5.4f} '.format(k, v)
        logger.add_scalar('TEST/'+k, v, epoch)
    logger.info('testing', text)

    logger.add_scalar('TEST/cmc1', cmc[0], epoch)
    logger.add_scalar('TEST/cmc5', cmc[4], epoch)
    logger.add_scalar('TEST/cmc10', cmc[9], epoch)
    logger.add_scalar('TEST/mAP', mAP, epoch)
    return cmc, mAP


def get_lr_from_optim(optim):
    for param_group in optim.param_groups:
        return param_group['lr']


def train_one_epoch(model, branches, nr_mask, train_loader, losses, optimizer, scheduler, epoch):
    global batch_step

    epoch_start_time = time.time()
    logger.info('training', 'Start training epoch-{}, lr={:.6}'.format(epoch,
                                                                       get_lr_from_optim(optimizer)))

    scaler = amp.GradScaler()
    model.train()
    for branch in branches:
        branch.train()
    history = collections.defaultdict(list)
    for i, (imgs, labels, masks) in enumerate(train_loader):

        batch = i + 1
        batch_start_time = time.time()

        imgs, labels, masks = imgs.cuda(), labels.cuda(), masks.float().cuda()

        with amp.autocast():
            loss = 0
            parsing_celoss = [0] * nr_mask
            parsing_triloss = [0] * nr_mask
            x = model(imgs)
            weights = get_weight(masks)
            f_main, p = branches[0](x)
            ce_loss = losses['cross_entropy_loss'][0](p, labels)
            triplet_hard_loss = losses['triplet_hard_loss'][0](f_main, labels)
            loss += Config.weight_ce[0] * ce_loss
            loss += Config.weight_tri[0] * triplet_hard_loss
            for b, branch in enumerate(branches):
                if b == 0:  # main branch
                    continue
                mask = masks[:, b-1: b, ...]
                f, logit = branch(x, mask)
                w = weights[:, b-1]
                pce_loss = losses['cross_entropy_loss'][b](logit, labels, w)
                # ptriplet_hard_loss = losses['triplet_hard_loss'][b](
                #     f, w, labels)
                # ptriplet_hard_loss = losses['triplet_hard_loss'][0](
                #     f, labels)
                parsing_celoss[b-1] = Config.weight_ce[b] * pce_loss
                # parsing_triloss += Config.weight_tri[b] * ptriplet_hard_loss
            loss = loss + sum(parsing_celoss) + sum(parsing_triloss)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        acc = accuracy(p, labels)[0]
        batch_end_time = time.time()
        time_spent = batch_end_time - batch_start_time
        dist_ap, dist_an = losses['triplet_hard_loss'][0].get_mean_hard_dist()
        perform = {}
        for i in range(nr_mask):
            perform.update({'p%d_ce' % (i+1): float(parsing_celoss[i])})
        # for i in range(nr_mask):
        #     perform.update({'p%d_tri' % (i+1): float(parsing_triloss[i])})
        perform.update({'ce': float(Config.weight_ce[0] * ce_loss),
                        'tri': float(Config.weight_tri[0] * triplet_hard_loss),
                        'dap': float(dist_ap),
                        'dan': float(dist_an),
                        'acc': float(acc),
                        't': float(time_spent)})

        if i % Config.batch_per_log == 0:
            stage = (epoch, batch)
            text = ''
            for k, v in perform.items():
                text += '|{}:{:<6.4f} '.format(k, float(v))
            logger.info('training', text, stage=stage)

        for k, v in perform.items():
            history[k].append(float(v))
            if k != 'time(s)':
                logger.add_scalar('TRAIN_b/'+k, v, batch_step)
        batch_step += 1

    scheduler.step()

    epoch_end_time = time.time()
    time_spent = sec2min_sec(epoch_start_time, epoch_end_time)

    text = 'Finish training epoch {}, time spent: {}mins {}secs, performance:\n##'.format(
        epoch, time_spent[0], time_spent[1])
    for k, vlist in history.items():
        v = mean(vlist)
        text += '|{}:{:>5.4f} '.format(k, v)
        if k != 'time(s)':
            logger.add_scalar('TRAIN_e/'+k, v, epoch)
    logger.info('training', text)


def save_checkpoint(save_dict):
    torch.save(save_dict, Config.checkpoint_path)


def load_checkpoint():
    ret = torch.load(Config.checkpoint_path)
    logger.info('global', 'loading checkpoint[epoch-{}] from {}'.format(
        ret['epoch'], Config.checkpoint_path), time_report=False)
    return ret


def prepare(args):
    resume_from_checkpoint = args.resume_from_checkpoint

    prepare_start_time = time.time()
    logger.info('global', 'Start preparing.')
    check_config_dir()
    logger.info('setting', config_info(), time_report=False)

    model = Backbone()
    model = model.cuda()
    logger.info('setting', model_summary(model), time_report=False)
    logger.info('setting', str(model), time_report=False)

    branches = [main_branch(Config.nr_class, Config.in_planes),
                parsing_branch(Config.nr_class, Config.in_planes),
                parsing_branch(Config.nr_class, Config.in_planes),
                parsing_branch(Config.nr_class, Config.in_planes),
                parsing_branch(Config.nr_class, Config.in_planes)]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(Config.input_shape),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(10),
        transforms.RandomCrop(Config.input_shape),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(Config.input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    trainset = Veri776_train(transforms=train_transforms,
                             need_mask=True, bg_switch=Config.p_bgswitch)
    testset = Veri776_test(transforms=test_transforms)

    pksampler = PKSampler(trainset, p=Config.P, k=Config.K)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=Config.batch_size, sampler=pksampler, num_workers=Config.nr_worker, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=Config.batch_size, sampler=torch.utils.data.SequentialSampler(testset), num_workers=Config.nr_worker, pin_memory=True)

    weight_decay_setting = parm_list_with_Wdecay_multi([model] + branches)
    optimizer = torch.optim.Adam(weight_decay_setting, lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_multi_func)

    losses = {}
    losses['cross_entropy_loss'] = [torch.nn.CrossEntropyLoss(),
                                    weight_cross_entropy(Config.ce_thres[0]), weight_cross_entropy(
                                        Config.ce_thres[1]),
                                    weight_cross_entropy(Config.ce_thres[2]), weight_cross_entropy(Config.ce_thres[3])]
    losses['triplet_hard_loss'] = [triplet_hard_loss(margin=Config.triplet_margin),
                                   weighted_triplet_hard_loss(
                                       margin=Config.branch_margin, soft_margin=Config.soft_marigin),
                                   weighted_triplet_hard_loss(
                                       margin=Config.branch_margin, soft_margin=Config.soft_marigin),
                                   weighted_triplet_hard_loss(
                                       margin=Config.branch_margin, soft_margin=Config.soft_marigin),
                                   weighted_triplet_hard_loss(
                                       margin=Config.branch_margin, soft_margin=Config.soft_marigin)]

    for k in losses.keys():
        if isinstance(losses[k], list):
            for i in range(len(losses[k])):
                losses[k][i] = losses[k][i].cuda()
        else:
            losses[k] = losses[k].cuda()

    for i in range(len(branches)):
        branches[i] = branches[i].cuda()

    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(Config.checkpoint_path):
        checkpoint = load_checkpoint()
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    # continue training for next the epoch of the checkpoint, or simply start from 1
    start_epoch += 1

    ret = {
        'start_epoch': start_epoch,
        'model': model,
        'branches': branches,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'losses': losses}

    prepare_end_time = time.time()
    time_spent = sec2min_sec(prepare_start_time, prepare_end_time)
    logger.info('global', 'Finish preparing, time spend: {}mins {}s.'.format(
        time_spent[0], time_spent[1]))

    return ret


def start(model, branches, train_loader, test_loader, optimizer, scheduler, losses, start_epoch,):
    train_start_time = time.time()

    best_mAP = 0.0
    best_mAP_epoch = 0
    best_top1 = 0.0
    best_top1_epoch = 0

    logger.info('global', 'Start training.')
    for epoch in range(start_epoch, Config.epoch + 1):
        train_one_epoch(model, branches, Config.nr_mask, train_loader, losses,
                        optimizer, scheduler, epoch)

        if epoch % Config.epoch_per_test == 0:
            cmc, mAP = test(model, branches, test_loader, losses, epoch)
            top1 = cmc[0]
            if top1 > best_top1:
                best_top1 = top1
                best_top1_epoch = epoch
            if mAP > best_mAP:
                best_mAP = mAP
                best_mAP_epoch = epoch

        if epoch % Config.epoch_per_save == 0:
            if Config.epoch_per_test % Config.epoch_per_save != 0:
                cmc, mAP = test(model, branches, test_loader, losses, epoch)
            file_name = 'epoch-{:0>3}'.format(epoch) + '.pth'
            save_dict = {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'top1': cmc[0],
                         'mAP': mAP}
            path = os.path.join(Config.model_dir, file_name)
            logger.info('global', 'Save model to {}'.format(path))
            torch.save(save_dict, path)

        save_dict = {'epoch': epoch,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict()}
        save_checkpoint(save_dict)
    train_end_time = time.time()
    time_spent = sec2min_sec(train_start_time, train_end_time)

    text = 'Finish training, time spent: {:>3}mins {:>3}s'.format(
        time_spent[0], time_spent[1])
    logger.info('global', text)
    text = '##FINISH## best mAP:{:>5.4f} in epoch {:>3}; best top1:{:>5.4f} in epoch{:>3}'.format(
        best_mAP, best_mAP_epoch, best_top1, best_top1_epoch)
    logger.info('global', text)


def main(args):
    kargs = prepare(args)
    start(**kargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--resume_from_checkpoint',
                        action='store_true', default=False)
    parser.add_argument('-w', '--wait_m', type=int, default=0)
    args = parser.parse_args()
    if args.wait_m:
        wait(m=args.wait_m)
    main(args)