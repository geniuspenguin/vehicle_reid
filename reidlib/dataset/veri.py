import os
import glob
from collections import defaultdict
import xml.etree.ElementTree as ET
import pickle

rootdir = '/home/peng/Documents/data/VeRi'
train_dir = os.path.join(rootdir, 'image_train')
query_dir = os.path.join(rootdir, 'image_query')
test_dir = os.path.join(rootdir, 'image_test')
train_xml_path = os.path.join(rootdir, 'train_label.xml')
test_xml_path = os.path.join(rootdir, 'test_label.xml')
save_path = os.path.join(rootdir, 'data_info.pkl')

query_name = open(os.path.join(rootdir, 'name_query.txt'), 'r').readlines()
query_set = [item.strip() for item in query_name]

train_name = open(os.path.join(rootdir, 'name_train.txt'), 'r').readlines()
train_set = [item.strip() for item in train_name]


mask_path = '/home/peng/Documents/data/VeRi/masks'
train_mask_path = os.path.join(mask_path, 'train/')
query_mask_path = os.path.join(mask_path, 'query/')
gallery_mask_path = os.path.join(mask_path, 'gallery/')


def get_dict(path):
    xmlp = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(path, parser=xmlp)
    root = tree.getroot()
    name2attr = {}
    vid2id = {}
    vid2attr = {}
    for child in root:
        for sub in child:
            attr = sub.attrib
            vid = attr['vehicleID']
            file_name = attr['imageName']
            if vid not in vid2id:
                vid2id[vid] = len(vid2id)
            intid = vid2id[vid]
            attr['intVID'] = intid
            vid2attr[vid] = attr
            name2attr[file_name] = attr
    return name2attr, vid2attr


def update_data_info(dirname, name2attr, vid2attr):
    miss = 0
    for fpath in glob.glob(os.path.join(dirname, '*.jpg')):
        fname = fpath.split('/')[-1]
        # print(fpath, fname, fpath.split('/')[-1].split('_')[0])
        if fname not in name2attr:
            parts = fpath.split('/')[-1].split('_')
            vid = parts[0]
            if vid not in vid2attr:
                miss += 1
                continue
            name2attr[fname] = {}
            name2attr[fname].update(vid2attr[vid])
        name2attr[fname]['path'] = fpath
    print(dirname, 'miss %d' % miss)


def generate_data_info():

    tname2attr_tmp, tvid2attr = get_dict(train_xml_path)
    vname2attr, vvid2attr = get_dict(test_xml_path)
    update_data_info(train_dir, tname2attr_tmp, tvid2attr)
    update_data_info(query_dir, vname2attr, vvid2attr)
    update_data_info(test_dir, vname2attr, vvid2attr)

    tname2attr = {}
    for name, attr in tname2attr_tmp.items():
        if name != attr['path'].split('/')[-1]:
            print('##', name, attr)
        if name in train_set:
            tname2attr[name] = attr

    qname2attr, gname2attr = {}, {}
    for name, attr in vname2attr.items():
        if name in query_set:
            qname2attr[name] = attr
        else:
            gname2attr[name] = attr
    update_mask_info(tname2attr, train_mask_path)
    update_mask_info(qname2attr, query_mask_path)
    update_mask_info(gname2attr, gallery_mask_path)
    save_pkl = {'train': tname2attr,
                'query': qname2attr, 'gallery': gname2attr}
    for k, x in save_pkl.items():
        print(k, len(x))
    with open(os.path.join(rootdir, 'data_info.pkl'), 'wb') as f:
        pickle.dump(save_pkl, f)

def update_mask_info(metas, path):
    name2path = {}
    for f in glob.glob(os.path.join(path, '*.png')):
        name = f.split('/')[-1]
        name = name.replace('.png', '.jpg')
        name2path[name] = f
    for k, meta in metas.items():
        imgname = meta['imageName']
        if imgname in name2path:
            meta.update({'mask_path': name2path[imgname]})

if __name__ == '__main__':
    # generate_data_info()

    with open(os.path.join(rootdir, 'data_info.pkl'), 'rb') as f:
        info = pickle.load(f)

    for k, dic in info.items():
        print(k)
        for i, (k, v) in enumerate(dic.items()):
            if i < 3:
                print(k, v)
            if 'mask_path' not in v:
                print('###', k, v)