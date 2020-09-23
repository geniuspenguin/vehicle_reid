import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import pandas as pd

def mask_weight_summary(dic, sample = -1, prefix='data'):
    history = [[] for i in range(4)]
    if sample > 0:
        rng = np.random.choice(list(dic.values()), sample)
    else:
        rng = dic.values()
    for v in rng:
        mask = cv2.imread(v['mask_path'], cv2.IMREAD_UNCHANGED)
        mask = np.array(mask)
        masks =mask_2_layer(mask)
        weight = get_weight(masks)
        for i in range(len(weight)):
            history[i].append(float(weight[i]))
    for i, line in enumerate(history):
        line = np.array(line)
        df_describe = pd.DataFrame(line)
        desc = df_describe.describe().round(3)
        print(desc)
        desc = desc.T[['mean', 'std', '50%', '75%']]
        plt.hist(line, bins='auto')
        plt.title('distribution of weight #{}\n{}'.format(i, str(desc).strip()))
        plt.savefig('./output/{}-weight{}.jpg'.format(prefix, i))
        plt.clf()
    # for i, line in enumerate(history):
    #     print('##{}##: '.format(i), [round(n, 3) for n in line])

def mask_2_layer(mask, map_values=[1, 2, 3, 4]):
    mask_map = []
    for v in map_values:
        vmap = (mask == v)[None, ...]
        mask_map.append(vmap)
    layer = np.concatenate(mask_map, axis=0)
    return layer

def get_weight(masks):
    total_mask = np.sum(masks)
    mask = np.sum(masks, (1, 2)).reshape(masks.shape[0], -1)
    return mask / (total_mask + 1)

def update_weight_info(dic):
    for v in dic.values():
        mask = cv2.imread(v['mask_path'], cv2.IMREAD_UNCHANGED)
        mask = np.array(mask)
        masks =mask_2_layer(mask)
        weights = get_weight(masks)
        v['mask_weights'] = list(weights.reshape(-1,))

def summary_weights(dic):
    history = [[] for i in range(4)]
    for item in dic.values():
        for i in range(4):
            history[i].append(item['mask_weights'][i])
    return history      

if __name__ == '__main__':
    pickle_path = '/home/peng/Documents/data/VeRi/data_info.pkl'
    dic = pickle.load(open(pickle_path, 'rb'))
    # train = dic['train']
    # mask_weight_summary(train, sample=-1, prefix='train')
    for k in dic.keys():
        for m, n in dic[k].items():
            for c in n['mask_weights']:
                if not isinstance(c, float):
                    print(m, n)
