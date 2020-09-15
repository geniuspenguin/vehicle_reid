import torch.utils.data as data
from glob import glob
from collections import defaultdict
import os
import cv2
from PIL import Image
import pickle

train_path = '/home/peng/Documents/data/VeRi/image_train'
query_path = '/home/peng/Documents/data/VeRi/image_query'
gallery_path = '/home/peng/Documents/data/VeRi/image_test'
pickle_path = '/home/peng/Documents/data/VeRi/data_info.pkl'


def generate_id_map(data_info):
    vid2ivid = {}
    sample2sid = {}
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
        infos.append(info)
        ivid2sids[ivid].append(sid)
    return infos, ivid2sids


class Veri776_train(data.Dataset):
    def __init__(self, pickle_path=pickle_path, transforms=None):
        super().__init__()
        self.pickle_path = pickle_path
        self.sample_info = pickle.load(open(pickle_path, 'rb'))['train']
        self.metas, self.label_to_samples = generate_id_map(self.sample_info)

        self.transforms = transforms
        self.nr_id = len(self.label_to_samples)
        self.nr_sample = len(self.metas)
        print('veri776: {} imgs with {} ids'.format(self.nr_sample, self.nr_id))

    def __getitem__(self, idx):
        path = self.metas[idx]['path']
        label = self.metas[idx]['ivid']
        # img = cv2.imread(path)
        img = Image.open(path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return self.nr_sample


class Veri776_test(data.Dataset):
    def __init__(self, pickle_path=pickle_path, transforms=None):
        self.pickle_path = pickle_path
        infos = pickle.load(open(pickle_path, 'rb'))
        self.q_info = infos['query']
        self.g_info = infos['gallery']
        self.test = self.relabel(self.q_info, self.g_info)
        self.transforms = transforms

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
        path = self.test[idx]['path']
        label = self.test[idx]['ivid']
        cid = self.test[idx]['icid']

        img = Image.open(path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img, label, cid

    def __len__(self):
        return len(self.test)

    def get_num_query(self):
        return len(self.q_info)

    def get_num_gallery(self):
        return (self.g_info)


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
    resize_target = (300, 300)
    train_transforms = transforms.Compose([
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

    trainset = Veri776_train(transforms=train_transforms)
    testset = Veri776_test(transforms=test_transforms)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=40, sampler=torch.utils.data.SequentialSampler(trainset), pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=40, sampler=torch.utils.data.SequentialSampler(testset), pin_memory=True)
    for i, (img, labels, cid) in tqdm(enumerate(test_loader), desc='###', total=len(test_loader)):
        if i > 10:
            break
        print(img.shape, labels.shape, cid.shape)
