import os.path as osp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from reidlib.dataset.dataset import Veri776_train, Veri776_test, Sub_benchmark
from config import Config
import argparse
import collections
from reidlib.utils.metrics import get_cmc_map, get_L2distance_matrix_numpy, accuracy
import numpy as np
from tqdm import tqdm
import glob
from model import Backbone, main_branch
import pandas as pd
from reidlib.utils.utils import no_grad_func
import time


def post_dists(data, a_type, a_color):
    dists = data['dist']
    qt, gt, qc, gc = data['query_type'], data['gallery_type'], \
        data['query_color'], data['gallery_color']

    t_mask = (qt.reshape(-1, 1) != gt.reshape(1, -1))
    c_mask = (qc.reshape(-1, 1) != gc.reshape(1, -1))

    return dists + a_type * t_mask + a_color * c_mask
if __name__ == '__main__':
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('-p', '--path')
    # args = argparser.parse_args()

    # model_dir = args.path

    start_time = time.time()
    data_path = osp.join(Config.data_dir, 'dists.pth')
    data = torch.load(data_path)
    q_ids, g_ids, q_cids, g_cids = data['ids_cids']

    dists = post_dists(data, a_type=3, a_color=3)

    cmc, mAP = get_cmc_map(dists, q_ids, g_ids, q_cids, g_cids)
    perform_name = ['mAP', 'cmc1', 'cmc5', 'cmc10']

    old_perf = [(name, value) for name, value in data['performance'][:4]]
    new_perf = [(name, value) for name, value in zip(perform_name, [mAP, cmc[0], cmc[4], cmc[9]])]
    print('before:{}'.format(old_perf))
    print('after :{}'.format(new_perf))
    



