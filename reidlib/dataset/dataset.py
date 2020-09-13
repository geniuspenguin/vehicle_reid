import torch.utils.data as data
from glob import glob
import collections
import os
import cv2
from PIL import Image


def generate_idmap(folder):
    names = glob(os.path.join(folder, '*.jpg'))
    idx_to_path, labelid_to_idxs = {}, collections.defaultdict(list)
    label_to_labelid = {}
    idx_to_labelid = {}
    idx_to_cameraid = {}
    for i, path in enumerate(names):
        idx_to_path[i] = path
        parts = path.split('/')[-1].split('_')
        label = parts[0]
        camera_id = parts[1]
        if label not in label_to_labelid:
            label_to_labelid[label] = len(label_to_labelid)
        labelid = label_to_labelid[label]
        labelid_to_idxs[labelid].append(i)
        idx_to_labelid[i] = labelid
        idx_to_cameraid[i] = camera_id
    return idx_to_path, labelid_to_idxs, idx_to_labelid, idx_to_cameraid


class Veri776(data.Dataset):
    def __init__(self, path, transforms=None):
        super().__init__()
        self.path = path
        self.sample_to_path, self.label_to_samples, self.sample_to_label, self.sample_to_cid \
            = generate_idmap(
                path)
        self.transforms = transforms

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
        return len(self.sample_to_path)
