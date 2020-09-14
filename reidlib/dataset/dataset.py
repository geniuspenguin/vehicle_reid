import torch.utils.data as data
from glob import glob
import collections
import os
import cv2
from PIL import Image

train_path = '/home/peng/Documents/data/VeRi/image_train'
query_path = '/home/peng/Documents/data/VeRi/image_query'
gallery_path = '/home/peng/Documents/data/VeRi/image_test'


def generate_idmap(folder):
    names = glob(os.path.join(folder, '*.jpg'))
    idx_to_path, labelid_to_idxs = {}, collections.defaultdict(list)
    label_to_labelid = {}
    idx_to_labelid = {}
    idx_to_cid = {}
    camera_to_cid = {}
    for i, path in enumerate(names):
        idx_to_path[i] = path
        parts = path.split('/')[-1].split('_')
        label = parts[0]
        camera = parts[1]
        if label not in label_to_labelid:
            label_to_labelid[label] = len(label_to_labelid)
        if camera not in camera_to_cid:
            camera_to_cid[camera] = len(camera_to_cid)
        labelid = label_to_labelid[label]
        labelid_to_idxs[labelid].append(i)
        idx_to_labelid[i] = labelid
        idx_to_cid[i] = camera_to_cid[camera]
    return idx_to_path, labelid_to_idxs, idx_to_labelid, idx_to_cid, camera_to_cid


class Veri776_train(data.Dataset):
    def __init__(self, path=train_path, transforms=None):
        super().__init__()
        self.path = path
        self.sample_to_path, self.label_to_samples, self.sample_to_label, self.sample_to_cid, self.camera_to_cid \
            = generate_idmap(
                path)
        self.transforms = transforms
        self.nr_id = len(self.label_to_samples)
        self.nr_sample = len(self.sample_to_label)
        print('veri776: {} imgs with {} ids'.format(self.nr_sample, self.nr_id))

    def __getitem__(self, idx):
        path = self.sample_to_path[idx]
        label = self.sample_to_label[idx]
        cid = self.sample_to_cid[idx]
        # img = cv2.imread(path)
        img = Image.open(path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img, label, cid

    def __len__(self):
        return self.nr_sample


class Veri776_test(data.Dataset):
    def __init__(self, query_path=query_path,
                 gallery_path=gallery_path, transforms=None):
        self.q_path = query_path
        self.g_path = gallery_path
        self.idx_to_path, _, self.idx_to_label, self.idx_to_cid, self.camera_to_cid = generate_idmap(
            self.q_path)
        self.nr_query = len(self.idx_to_path)
        g_idx_to_path, _, g_idx_to_label, g_idx_to_cid, g_camera_to_cid = generate_idmap(
            self.g_path)
        self.nr_gallery = len(g_idx_to_path)
        self.idx_to_path.update(g_idx_to_path)
        self.idx_to_label.update(g_idx_to_label)
        self.idx_to_cid.update(g_idx_to_cid)
        self.camera_to_cid.update(g_camera_to_cid)
        self.transforms = transforms

    def __getitem__(self, idx):
        path = self.idx_to_path[idx]
        label = self.idx_to_label[idx]
        cid = self.idx_to_cid[idx]

        img = Image.open(path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img, label, cid

    def __len__(self):
        return len(self.idx_to_path)

    def get_num_query(self):
        return self.nr_query

    def get_num_gallery(self):
        return self.nr_gallery

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
    resize_target = (300,300)
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