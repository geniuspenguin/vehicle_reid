from reidlib.dataset.dataset import Veri776_train, Veri776_test
from reidlib.dataset.sampler import PKSampler
from reidlib.utils.logger import Logger, sec2min_sec, model_summary
from config import Config, config_info
from reidlib.models.resnet_ibn import resnet50_ibn_a
from reidlib.utils.loss import triplet_hard_loss, tri_hard_attn
from reidlib.utils.metrics import get_cmc_map, get_L2distance_matrix_numpy, accuracy, get_L2distance_matrix_attn_batch
import torch
import time
import numpy as np
import torchvision.transforms as transforms
import collections
import os
from tqdm import tqdm
from reidlib.utils.utils import no_grad_func
import argparse
import torch.cuda.amp as amp
from model import Baseline, BinaryClassifier, idlabel2pairlabel

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
def test(model, bclassifier, test_loader, losses, epoch, nr_query=Config.nr_query):
    '''
    return: cmc1, mAP
    test model on testset and save result to log.
    '''
    val_start_time = time.time()
    model.eval()
    bclassifier.eval()
    logger.info('testing', 'Start testing')
    all_features_gpu, all_labels, all_cids, all_mask = [], [], [], []

    for i, (imgs, labels, cids) in tqdm(enumerate(test_loader), desc='extracting features', total=len(test_loader)):
        imgs, labels, cids = imgs.cuda(), labels.cuda(), cids.cuda()
        f_norm, f_mask = model(imgs)
        all_features_gpu.append(f_norm)
        all_labels.append(labels)
        all_mask.append(f_mask)
        all_cids.append(cids)

    features = torch.cat(all_features_gpu, axis=0)
    alllabels = torch.cat(all_labels, axis=0)
    allmask = torch.cat(all_mask, axis=0)
    allcids = torch.cat(all_cids, axis=0)

    q_f_gpu, g_f_gpu = features[:nr_query, ...], features[nr_query:, ...]
    q_ids, g_ids = alllabels[:nr_query, ...], alllabels[nr_query:, ...]
    q_cids, g_cids = allcids[:nr_query], allcids[nr_query:]
    q_mask, g_mask = allmask[:nr_query], allmask[nr_query:]

    pk_pros = []
    for start in tqdm(range(0, q_f_gpu.shape[0], Config.eval_P), desc='computing similarity'):
        end = min(start + Config.eval_P, q_f_gpu.shape[0])
        p_pros = []
        for kstart in range(0, g_f_gpu.shape[0], Config.eval_K):
            kend = min(kstart + Config.eval_K, g_f_gpu.shape[0])
            pros = bclassifier(
                q_f_gpu[start: end, ...], g_f_gpu[kstart: kend, ...])
            pros = pros.reshape(end - start, kend - kstart, 2)
            p_pros.append(pros)
        p_pros = torch.cat(p_pros, axis=1)
        pk_pros.append(p_pros)
    pros = torch.cat(pk_pros, axis=0)
    pairlabels = idlabel2pairlabel(q_ids, g_ids)
    acc = accuracy(pros.reshape(-1, 2), pairlabels)[0]
    distance_matrix = pros[:, :, 0].cpu().detach().numpy()
    q_ids = q_ids.cpu().detach().numpy()
    g_ids = g_ids.cpu().detach().numpy()
    q_cids = q_cids.cpu().detach().numpy()
    g_cids = g_cids.cpu().detach().numpy()

    print('Compute CMC and mAP')
    cmc, mAP = get_cmc_map(distance_matrix, q_ids, g_ids, q_cids, g_cids)
    val_end_time = time.time()
    time_spent = sec2min_sec(val_start_time, val_end_time)

    text = 'Finish testing epoch {:>3}, time spent: [{:>3}mins{:>3}s], performance:\n##'.format(
        epoch, time_spent[0], time_spent[1])
    text += 'W/O attention> |CMC1:{:>5.4f} |mAP:{:>5.4f} |ACC:{:>5.4f} '.format(
        cmc[0], mAP, acc)
    logger.info('testing', text)

    logger.add_scalar('TEST/cmc1', cmc[0], epoch)
    logger.add_scalar('TEST/cmc5', cmc[4], epoch)
    logger.add_scalar('TEST/mAP', mAP, epoch)
    logger.add_scalar('TEST/acc', acc, epoch)
    return cmc, mAP, acc


def get_lr_from_optim(optim):
    for param_group in optim.param_groups:
        return param_group['lr']


def train_one_epoch(model, bclassifier, train_loader, losses, optimizer, scheduler, epoch):
    global batch_step

    epoch_start_time = time.time()
    logger.info('training', 'Start training epoch-{}, lr={:.6}'.format(epoch,
                                                                       get_lr_from_optim(optimizer)))

    scaler = amp.GradScaler()
    model.train()
    bclassifier.train()
    history = collections.defaultdict(list)
    for i, (imgs, labels) in enumerate(train_loader):
        batch = i + 1
        batch_start_time = time.time()

        imgs, labels = imgs.cuda(), labels.cuda()
        pair_labels = idlabel2pairlabel(labels, labels)
        with amp.autocast():
            f_bn, p, f_mask = model(imgs)
            p_same = bclassifier(f_bn, f_bn)
            id_loss = losses['cross_entropy_loss'](p, labels)
            pair_loss = losses['pair_loss'](p_same, pair_labels)
            loss = Config.weight_ce * id_loss
            loss += Config.weight_pairloss * pair_loss

    
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        sortmask = torch.sort(f_mask, dim=1)[0]
        bottom20mean = sortmask[:, :20].mean()
        top20mean = sortmask[:, -20:].mean()

        acc = accuracy(p, labels)[0]
        acc_pair = accuracy(p_same, pair_labels)[0]
        batch_end_time = time.time()
        time_spent = batch_end_time - batch_start_time

        perform = {'id_loss': float(Config.weight_ce * id_loss),
                   'pair_loss': float(Config.weight_pairloss * pair_loss),
                   'acc_id': float(acc),
                   'acc_pair': float(acc_pair),
                #    'mtop20': float(top20mean),
                #    'mbot20': float(bottom20mean),
                   'time(s)': float(time_spent)}

        if i % Config.batch_per_log == 0:
            stage = (epoch, batch)
            text = ''
            for k, v in perform.items():
                text += '|{}:{:<8.4f} '.format(k, float(v))
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

    model = Baseline(num_classes=Config.nr_class).cuda()
    logger.info('setting', model_summary(model), time_report=False)
    logger.info('setting', str(model), time_report=False)

    bclassifier = BinaryClassifier(Config.in_planes).cuda()

    train_transforms = transforms.Compose([
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

    trainset = Veri776_train(transforms=train_transforms)
    testset = Veri776_test(transforms=test_transforms)

    pksampler = PKSampler(trainset, p=Config.P, k=Config.K)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=Config.batch_size, sampler=pksampler, num_workers=Config.nr_worker, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=Config.batch_size, sampler=torch.utils.data.SequentialSampler(testset), num_workers=Config.nr_worker, pin_memory=True)

    weight_decay_setting = parm_list_with_Wdecay(model)
    weight_decay_setting_bc = parm_list_with_Wdecay(bclassifier)
    weight_decay_setting += weight_decay_setting_bc
    optimizer = torch.optim.Adam(weight_decay_setting, lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_multi_func)

    losses = {}
    losses['cross_entropy_loss'] = torch.nn.CrossEntropyLoss()
    losses['pair_loss'] = torch.nn.CrossEntropyLoss()
    for k in losses.keys():
        losses[k] = losses[k].cuda()

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
        'bclassifier': bclassifier,
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


def start(model, bclassifier, train_loader, test_loader, optimizer, scheduler, losses, start_epoch,):
    train_start_time = time.time()

    # best_mAP_a = 0.0
    # best_mAP_a_epoch = 0
    # best_top1_a = 0.0
    # best_top1_a_epoch = 0

    best_mAP = 0.0
    best_mAP_epoch = 0
    best_top1 = 0.0
    best_top1_epoch = 0

    logger.info('global', 'Start training.')
    for epoch in range(start_epoch, Config.epoch + 1):
        train_one_epoch(model, bclassifier, train_loader, losses,
                        optimizer, scheduler, epoch)

        if epoch % Config.epoch_per_test == 0:
            cmc, mAP, _ = test(model, bclassifier, test_loader, losses, epoch)
            top1 = cmc[0]
            if top1 > best_top1:
                best_top1 = top1
                best_top1_epoch = epoch
            if mAP > best_mAP:
                best_mAP = mAP
                best_mAP_epoch = epoch

        if epoch % Config.epoch_per_save == 0:
            if Config.epoch_per_test % Config.epoch_per_save != 0:
                cmc, mAP, _ = test(model, bclassifier,
                                   test_loader, losses, epoch)
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
    # text = '##FINISH## best mAP_a:{:>5.4f} in epoch {:>3}; best top1_a:{:>5.4f} in epoch{:>3}'.format(
    #     best_mAP_a, best_mAP_a_epoch, best_top1_a, best_top1_a_epoch)
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
    args = parser.parse_args()
    main(args)
