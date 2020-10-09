import torch
import argparse
from glob import glob
import os
from collections import defaultdict
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('-p', '--path', default='./')
    args = argp.parse_args()
    history = defaultdict(list)
    paths = glob(os.path.join(args.path, '*.pth'))
    paths = sorted(paths)
    for f in paths:
        print('loading ', f, end=' ')
        ret = torch.load(f, map_location='cpu')
        delete_keys = []
        for k, v in ret.items():
            if isinstance(v, int) or isinstance(v, float) or k == 'mAP' or k == 'top1':
                history[k].append(v)
                print('{}: {:.5f}'.format(k, v), end=' ')
            else:
                delete_keys.append(k)
        for k in delete_keys:
            ret.pop(k)
        torch.save(ret, f)
        print('')

    for k, v in history.items():
        print("[{}]max:{:.5f}  min:{:.5f}".format(k, max(v), min(v)))
