import torch.utils.data as data
from glob import glob
from collections import defaultdict
import os
from PIL import Image
import pickle
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
from reidlib.dataset.transform import background_switch

rootdir = '/home/peng/Documents/data/VeRi'
train_path = '/home/peng/Documents/data/VeRi/image_train'
query_path = '/home/peng/Documents/data/VeRi/image_query'
gallery_path = '/home/peng/Documents/data/VeRi/image_test'
pickle_path = '/home/peng/Documents/data/VeRi/data_info.pkl'
subbm_path = '/home/peng/Documents/data/VeRi/sub_benchmark.pkl'
type_lines = open(os.path.join(rootdir, 'list_type.txt'), 'r').readlines()
color_lines = open(os.path.join(rootdir, 'list_color.txt'), 'r').readlines()



def generate_attr_map(lines):
    name2id = {}
    for line in lines:
        args = line.split()
        idx = int(args[0])
        name = args[1]
        name2id[name] = idx
    return name2id


type_map = generate_attr_map(type_lines)
color_map = generate_attr_map(color_lines)


def generate_id_map(data_info):
    vid2ivid = {}
    sample2sid = {}
    cid2icid = {}
    infos = []
    ivid2sids = defaultdict(list)
    for name, info in data_info.items():
        if name not in sample2sid:
            sample2sid[name] = len(sample2sid)
        sid = sample2sid[name]
        vid = info['vehicleID']
        if vid not in vid2ivid:
            vid2ivid[vid] = len(vid2ivid)
        ivid = vid2ivid[vid]
        info.update({'sid': sid, 'ivid': ivid})
        cid = info['cameraID']
        if cid not in cid2icid:
            cid2icid[cid] = len(cid2icid)
        icid = cid2icid[cid]
        info.update({'icid': icid})
        infos.append(info)
        ivid2sids[ivid].append(sid)
    return infos, ivid2sids


def mask_2_layer(mask, map_values=[1, 2, 3, 4]):
    mask_map = []
    for v in map_values:
        vmap = (mask == v)[None, ...]
        mask_map.append(vmap)
    layer = np.concatenate(mask_map, axis=0)
    return layer


class Veri776_train(data.Dataset):
    def __init__(self, pickle_path=pickle_path, img_shape=(224, 224), transforms=None,
                 need_attr=False, need_mask=False, bg_switch=0):
        super().__init__()
        self.pickle_path = pickle_path
        self.sample_info = pickle.load(open(pickle_path, 'rb'))['train']
        self.metas, self.label_to_samples = generate_id_map(self.sample_info)
        self.need_attr = need_attr
        self.need_mask = need_mask
        self.transforms = transforms
        self.nr_id = len(self.label_to_samples)
        self.nr_sample = len(self.metas)
        self.img_shape = img_shape
        self.bg_switch = bg_switch  # probs to switch background using segmentation

        self.type2itid = None
        self.color2iclid = None
        # self.imgs_ram = self._load_imgs_from_metas()
        print('veri776: {} imgs with {} ids'.format(self.nr_sample, self.nr_id))

    def _load_imgs_from_metas(self):
        imgs = []
        for meta in tqdm(self.metas, desc='loading imgs into RAM'):
            img = Image.open(meta['path']).convert('RGB')
            imgs.append(img)
        return imgs

    def __getitem__(self, idx):
        path = self.metas[idx]['path']
        label = self.metas[idx]['ivid']
        cid = self.metas[idx]['icid']
        ret = {}

        # img = self.imgs_ram[idx]
        img = Image.open(path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)
        ret = [img, label]

        if self.need_attr:
            typeid = int(self.metas[idx]['typeID'])
            colorid = int(self.metas[idx]['colorID'])
            ret.extend([cid, typeid, colorid])

        if self.need_mask:
            mask = Image.open(self.metas[idx]['mask_path'])
            resize = transforms.Resize(self.img_shape)
            mask = resize(mask)
            mask = np.array(mask)
            layer = mask_2_layer(mask)
            ret.append(layer)
        return ret

    def __len__(self):
        return self.nr_sample


class Veri776_test(data.Dataset):
    def __init__(self, pickle_path=pickle_path, img_shape=(224, 224),
                 transforms=None, need_attr=False, need_mask=False):
        self.pickle_path = pickle_path
        infos = pickle.load(open(pickle_path, 'rb'))
        self.need_attr = need_attr
        self.q_info = infos['query']
        self.g_info = infos['gallery']
        self.metas = self.relabel(self.q_info, self.g_info)
        self.transforms = transforms
        self.need_mask = need_mask
        self.img_shape = img_shape
        self.nr_query = len(self.q_info)
        self.nr_gallery = len(self.g_info)
        # self.imgs_ram = self._load_imgs_from_metas()

    def _load_imgs_from_metas(self):
        imgs = []
        for meta in tqdm(self.metas, desc='loading imgs into RAM'):
            img = Image.open(meta['path']).convert('RGB')
            imgs.append(img)
        return imgs

    def relabel(self, q_infos, g_infos):
        vid2ivid = {}
        cid2icid = {}
        infos = []
        for _, info in q_infos.items():
            vid = info['vehicleID']
            cid = info['cameraID']
            if vid not in vid2ivid:
                vid2ivid[vid] = len(vid2ivid)
            if cid not in cid2icid:
                cid2icid[cid] = len(cid2icid)
            icid = cid2icid[cid]
            ivid = vid2ivid[vid]
            sid = len(infos)
            info['ivid'] = ivid
            info['icid'] = icid
            info['sid'] = sid
            infos.append(info)

        for _, info in g_infos.items():
            vid = info['vehicleID']
            cid = info['cameraID']
            if vid not in vid2ivid:
                vid2ivid[vid] = len(vid2ivid)
            if cid not in cid2icid:
                cid2icid[cid] = len(cid2icid)
            icid = cid2icid[cid]
            ivid = vid2ivid[vid]
            sid = len(infos)
            info['ivid'] = ivid
            info['icid'] = icid
            info['sid'] = sid
            infos.append(info)
        return infos

    def __getitem__(self, idx):
        path = self.metas[idx]['path']
        label = self.metas[idx]['ivid']
        cid = self.metas[idx]['icid']

        # img = self.imgs_ram[idx]
        img = Image.open(path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        ret = [img, label, cid]
        if self.need_attr:
            typeid = int(self.metas[idx]['typeID'])
            colorid = int(self.metas[idx]['colorID'])
            ret.extend([typeid, colorid])
        if self.need_mask:
            resize = transforms.Resize((self.img_shape))
            mask = Image.open(self.metas[idx]['mask_path'])
            mask = resize(mask)
            mask = np.array(mask)
            layer = mask_2_layer(mask)
            ret.append(layer)
        return ret

    def __len__(self):
        return len(self.metas)

    def get_num_query(self):
        return len(self.q_info)

    def get_num_gallery(self):
        return (self.g_info)

class Sub_benchmark(Veri776_test):
    def __init__(self, pickle_path=pickle_path, img_shape=(224, 224),
                 transforms=None, need_attr=False, need_mask=False):
        self.pickle_path = pickle_path
        infos = pickle.load(open(subbm_path, 'rb'))
        self.q_info = infos['query']
        self.g_info = infos['gallery']
        self.metas = self.relabel(self.q_info, self.g_info)
        self.transforms = transforms
        self.need_mask = need_mask
        self.img_shape = img_shape
        self.nr_query = len(self.q_info)
        self.nr_gallery = len(self.g_info)
        self.need_attr = need_attr

# def collate_func(batch):
#     inps, targets, cids = [], [], []
#     make_tensor = lambda x: torch.Tensor(x)
#     for item in batch:
#         img, target, cid = map(make_tensor, item)
#         inps.append(img)
#         targets.append(target)
#         cids.append(cid)
#     return torch.cat(inps, axis = 0), torch.cat(targets, axis=0), torch.cat(cids, axis=0)


if __name__ == '__main__':
    import torch
    import torchvision.transforms as transforms
    from tqdm import tqdm
    from torchvision.utils import make_grid, save_image
    resize_target = (300, 300)
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_target),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(10),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    trainset = Veri776_train(transforms=train_transforms,
                             need_attr=True, need_mask=True, bg_switch=1)
    testset = Veri776_test(transforms=test_transforms,
                           need_attr=True, need_mask=True)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=256, sampler=torch.utils.data.SequentialSampler(trainset), pin_memory=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=40, sampler=torch.utils.data.SequentialSampler(testset), pin_memory=True)
    for i, (img, labels, cid, ts, cs, mask) in tqdm(enumerate(train_loader), desc='###', total=len(train_loader)):
        # img = make_grid(img, padding=2)
        # save_image(img, open('./img.jpg', 'w'))
        print(mask.shape)
        break
        # continue