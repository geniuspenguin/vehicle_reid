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

def getpr_from_confusion(m):
    assert m.shape[0] == m.shape[1]
    tp = np.diag(m)
    tp_fn = np.sum(m, axis=1)
    tp_fn = np.clip(tp_fn, 1, None)
    tp_fp = np.sum(m, axis=0)
    tp_fp = np.clip(tp_fp, 1, None)
    pre_cls = tp / tp_fp
    rec_cls = tp / tp_fn
    pre_total = tp.sum() / tp_fp.sum()
    rec_total = tp.sum() / tp_fn.sum()
    return pre_cls, rec_cls, pre_total, rec_total

@no_grad_func
def test(model, main_branch, test_loader, nr_query=Config.nr_query):
    '''
    return: cmc1, mAP
    test model on testset and save result to log.
    '''
    model.eval()
    main_branch.eval()
    all_features, all_labels, all_cids = [], [], []
    all_pred_t, all_pred_c = [], []
    type_confuse = np.zeros((10, 10))
    color_confuse = np.zeros((11, 11))

    for i, (imgs, labels, cids, types, colors) in tqdm(enumerate(test_loader), total=len(test_loader)):
        imgs, labels, cids = imgs.cuda(), labels.cuda(), cids.cuda()
        x, p_type, p_color = model(imgs)
        f = main_branch(x)

        pred_t = torch.argmax(p_type, dim=1)
        pred_c = torch.argmax(p_color, dim=1)
        for k in range(types.shape[0]):
            pred_tid = pred_t[k]
            label_tid = types[k]
            type_confuse[label_tid, pred_tid] += 1

            pred_cid = pred_c[k]
            label_cid = colors[k]
            color_confuse[label_cid, pred_cid] += 1
        all_features.append(f.cpu().detach().numpy())
        all_labels.append(labels.cpu().detach().numpy())
        all_cids.append(cids.cpu().detach().numpy())
        all_pred_t.append(pred_t.cpu().detach().numpy())
        all_pred_c.append(pred_c.cpu().detach().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_cids = np.concatenate(all_cids, axis=0)
    all_pred_t = np.concatenate(all_pred_t, axis=0)
    all_pred_c = np.concatenate(all_pred_c, axis=0)

    q_f, g_f = all_features[:nr_query], all_features[nr_query:]
    q_ids, g_ids = all_labels[:nr_query], all_labels[nr_query:]
    q_cids, g_cids = all_cids[:nr_query], all_cids[nr_query:]
    q_predt, g_predt = all_pred_t[:nr_query], all_pred_t[nr_query:]
    q_predc, g_predc = all_pred_c[:nr_query], all_pred_c[nr_query:]

    print('Computing CMC and mAP')
    distance_matrix = get_L2distance_matrix_numpy(q_f, g_f)
    cmc, mAP = get_cmc_map(distance_matrix, q_ids, g_ids, q_cids, g_cids)
    ids_cids = (q_ids, g_ids, q_cids, g_cids)
    return cmc, mAP, distance_matrix, q_predt, g_predt, q_predc, g_predc, \
        type_confuse, color_confuse, ids_cids

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
    testset = Sub_benchmark(transforms=test_transforms, need_attr=True)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=Config.batch_size, sampler=torch.utils.data.SequentialSampler(testset), num_workers=Config.nr_worker, pin_memory=True)

    nr_query = testset.nr_query

    perform = []    # 0:mAP, 1:cmc1, 2:cmc5, 3:cmc10
    index = []
    columns = ['mAP', 'cmc1', 'cmc5', 'cmc10']
    colors = ['yellow', 'orange', 'green', 'gray', 'red', \
        'blue', 'white', 'gold', 'brown', 'black']
    types = ['sedan', 'suv', 'van', 'hatchback', \
        'mpv', 'pickup', 'bus', 'truck', 'estate']

    columns = columns + ['pre_' + x for x in types] + ['pre_type'] + ['rec_' + x for x in types] + ['rec_type']
    columns = columns + ['pre_' + x for x in colors] + ['pre_color'] + ['rec_' + x for x in colors] + ['rec_color']


    model = Backbone(num_classes=Config.nr_class).cuda()
    mainbranch = main_branch(Config.nr_class, Config.in_planes).cuda()

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
        mainbranch.load_state_dict(checkpoint['branch0'])

        cmc, mAP, dis_matrix, qt, gt, qc, gc, type_confuse, color_confuse, ids_cids \
             = test(model, mainbranch, test_loader, nr_query=nr_query)
        one_perform = [mAP, cmc[0], cmc[4], cmc[9]]
        pre_type, rec_type, pre_t_total, rec_t_total = getpr_from_confusion(type_confuse)
        pre_type, rec_type = pre_type[1:], rec_type[1:]
        pre_color, rec_color, pre_c_total, rec_c_total = getpr_from_confusion(color_confuse)
        pre_color, rec_color = pre_color[1:], rec_color[1:]
        one_perform.extend(list(pre_type) + [pre_t_total] + list(rec_type) + [rec_t_total])
        one_perform.extend(list(pre_color) + [pre_c_total] + list(rec_color) + [rec_c_total])

        perform.append(one_perform)

        name_perform = list(zip(columns, one_perform))

        if mAP > best_mAP:
            best_mAP = mAP
            best_dist['dist'] = dis_matrix
            best_dist['model'] = model_name
            best_dist['performance'] = name_perform
            best_dist['query_type'] = qt
            best_dist['gallery_type'] = gt
            best_dist['query_color'] = qc
            best_dist['gallery_color'] = gc
            best_dist['ids_cids'] = ids_cids

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