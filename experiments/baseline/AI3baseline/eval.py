import argparse
import torch
import numpy as np
import glob
import os.path as osp
from config import Config
from model import Baseline
from reidlib.utils.utils import no_grad_func
import torchvision.transforms as transforms
from reidlib.dataset.dataset import Veri776_train, Veri776_test
import time
from reidlib.utils.metrics import get_cmc_map, get_L2distance_matrix_numpy, accuracy
from reidlib.utils.logger import Logger, sec2min_sec, model_summary
import collections
from tqdm import tqdm

text_line = []
text_log_path = osp.join(Config.data_dir, 'eval_log.txt')
eval_log_path = osp.join(Config.data_dir, 'eval_results.pth')
device = 'cuda'

@no_grad_func
def test(model, name, test_loader, nr_query=Config.nr_query):
    '''
    return: cmc1, mAP
    test model on testset and save result to log.
    '''
    val_start_time = time.time()
    model.eval()
    all_features, all_labels, all_cids = [], [], []
    all_gapx = []
    history = collections.defaultdict(list)
    for i, (imgs, labels, cids) in tqdm(enumerate(test_loader), desc='testing on epoch-{}'.format(name), total=len(test_loader)):
        imgs, labels, cids = imgs.cuda(), labels.cuda(), cids.cuda()
        x_gap, x = model(imgs)
        all_gapx.append(x_gap.cpu().detach().numpy())
        all_features.append(x.cpu().detach().numpy())
        all_labels.append(labels.cpu().detach().numpy())
        all_cids.append(cids.cpu().detach().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_cids = np.concatenate(all_cids, axis=0)
    all_gapx = np.concatenate(all_gapx, axis=0)
    q_f, g_f = all_features[:nr_query], all_features[nr_query:]
    q_ids, g_ids = all_labels[:nr_query], all_labels[nr_query:]
    q_cids, g_cids = all_cids[:nr_query], all_cids[nr_query:]
    q_gapx, g_gapx = all_gapx[:nr_query], all_gapx[nr_query:]

    print('Computing CMC and mAP')
    distance_matrix = get_L2distance_matrix_numpy(q_f, g_f)
    cmc, mAP = get_cmc_map(distance_matrix, q_ids, g_ids, q_cids, g_cids)

    distance_matrix_gapx = get_L2distance_matrix_numpy(q_gapx, g_gapx)
    cmc_gapx, mAP_gapx = get_cmc_map(
        distance_matrix_gapx, q_ids, g_ids, q_cids, g_cids)

    val_end_time = time.time()
    time_spent = sec2min_sec(val_start_time, val_end_time)

    print('testing epoch {:>3}, time spent: [{:>3}mins{:>3}s]:##'.format(
        name, time_spent[0], time_spent[1]))
    text = '[feat_after_bn] mAP:{:>5.4f} |CMC1:{:>5.4f} |CMC5:{:>5.4f} |CMC10:{:>5.4f}'.format(
        mAP, cmc[0], cmc[4], cmc[9])
    text += '\t [feat_after_gap] mAP:{:>5.4f} |CMC1:{:>5.4f} |CMC5:{:>5.4f} |CMC10:{:>5.4f}'.format(
        mAP_gapx, cmc_gapx[0], cmc_gapx[4], cmc_gapx[9])
    for k, vlist in history.items():
        v = float(sum(vlist)/len(vlist))
        text += '|{}:{:>5.4f} '.format(k, v)

    print(text)
    text_line.append(text)
    ret_dict = {'cmc': cmc,
                'cmc_gapx': cmc_gapx,
                'mAP': mAP,
                'mAP_gapx': mAP_gapx}
    return ret_dict

if __name__ == '__main__':
    model = Baseline(num_classes=Config.nr_class)
    model = model.to(device)
    test_transforms = transforms.Compose([
        transforms.Resize(Config.input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    testset = Veri776_test(transforms=test_transforms)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=Config.batch_size,
        sampler=torch.utils.data.SequentialSampler(testset),
        num_workers=Config.nr_worker, pin_memory=True)

    eval_results = {}

    model_paths = glob.glob(osp.join(Config.model_dir, '*.pth'))
    model_paths = sorted(model_paths)
    history = collections.defaultdict(list)
    for model_path in model_paths:
        states = torch.load(model_path)
        model.load_state_dict(states['model'])
        model_name = model_path.split('/')[-1]
        eval_result = test(model, model_name, test_loader)
        eval_results[model_name] = eval_result

        history['mAP'].append(eval_result['mAP'])
        history['CMC1'].append(eval_result['cmc'][0])
        history['CMC5'].append(eval_result['cmc'][4])
        history['CMC10'].append(eval_result['cmc'][9])

        history['mAP_gapx'].append(eval_result['mAP_gapx'])
        history['CMC1_gapx'].append(eval_result['cmc_gapx'][0])
        history['CMC5_gapx'].append(eval_result['cmc_gapx'][4])
        history['CMC10_gapx'].append(eval_result['cmc_gapx'][9])
    
    with open(text_log_path, 'w') as eval_log_file:
        eval_log_file.write('\n'.join(text_line))

    for metric, values in history.items():
        print('[{:>5}] max:{:5.4f} min:{:5.4f}'.format(metric, max(values), min(values)))
    torch.save(eval_results, eval_log_path)

