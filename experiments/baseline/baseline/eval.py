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
from model import Baseline
import pandas as pd
from reidlib.utils.utils import no_grad_func
import time

@no_grad_func
def test(model, test_loader, nr_query=Config.nr_query):
    '''
    return: cmc1, mAP
    test model on testset and save result to log.
    '''
    model.eval()
    all_features, all_labels, all_cids = [], [], []
    history = collections.defaultdict(list)

    for i, (imgs, labels, cids) in tqdm(enumerate(test_loader), total=len(test_loader)):
        imgs, labels, cids = imgs.cuda(), labels.cuda(), cids.cuda()
        f_norm = model(imgs)
        all_features.append(f_norm.cpu().detach().numpy())
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
    return cmc, mAP, distance_matrix

if __name__ == '__main__':
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('-p', '--path')
    # args = argparser.parse_args()

    # model_dir = args.path

    start_time = time.time()
    model_dir = Config.model_dir

    test_transforms = transforms.Compose([
        transforms.Resize(Config.input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    testset = Sub_benchmark(transforms=test_transforms)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=Config.batch_size, sampler=torch.utils.data.SequentialSampler(testset), num_workers=Config.nr_worker, pin_memory=True)

    nr_query = testset.nr_query

    perform = []    # 0:mAP, 1:cmc1, 2:cmc5, 3:cmc10
    index = []
    columns = ['mAP', 'cmc1', 'cmc5', 'cmc10']
    model = Baseline(Config.nr_class).cuda()

    best_dist = {}
    best_mAP = 0

    model_paths = glob.glob(osp.join(model_dir, '*.pth'))
    model_paths = sorted(model_paths)

    for path in model_paths:
        model_name = path.split('/')[-1].replace('.pth', '')
        print('testing ', model_name)
        index.append(model_name)

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])

        cmc, mAP, dis_matrix = test(model, test_loader, nr_query=nr_query)
        one_perform = [mAP, cmc[0], cmc[4], cmc[9]]
        perform.append(one_perform)

        name_perform = list(zip(columns, one_perform))

        if mAP > best_mAP:
            best_mAP = mAP
            best_dist['dist'] = dis_matrix
            best_dist['model'] = model_name
            best_dist['performance'] = name_perform

        print(name_perform)
    
    dataframe = pd.DataFrame(np.array(perform), index=index, columns=columns)
    print(dataframe)
    print(dataframe.describe().T['max'])
    df_save_path = osp.join(Config.data_dir, 'eval_result.csv')
    dists_save_path = osp.join(Config.data_dir, 'dists.pth')

    dataframe.to_csv(df_save_path)
    torch.save(best_dist, dists_save_path)

    end_time = time.time()
    time_spent = end_time - start_time
    print('time spent: {} mins {} secs'.format(time_spent // 60, time_spent % 60))