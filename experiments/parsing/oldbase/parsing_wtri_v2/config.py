import os


class Config:
    experiment_name = 'res50ibna'

    experiment_dir = os.path.dirname(__file__)
    data_dir = os.path.join(experiment_dir, 'expr_logs')
    model_dir = os.path.join(data_dir, 'models')
    log_dir = os.path.join(data_dir, 'log')

    checkpoint_path = os.path.join(log_dir, 'latest.pth')

    input_shape = (224, 224)

    nr_class = 576
    in_planes = 2048

    epoch = 120
    P = 32
    K = 4
    batch_size = int(P*K)

    weight_decay = 5e-4
    lr = 3.5e-4
    triplet_margin = 1.2
    # 0: main brach loss 1: front branch loss 2: backward branch loss
    # 3: top branch loss 4: side branch loss
    weight_ce = [1, 0.1, 0.1, 0.1, 0.1]
    weight_tri = [1, 1, 1, 1, 1]

    batch_per_log = 1
    epoch_per_test = 5
    epoch_per_save = 5

    nr_query = 1678
    nr_test = 11579

    w_type = 1
    w_color = 1

    nr_worker = 6

    p_bgswitch = 0
    nr_mask = 4

    branch_margin = 1.2
    soft_marigin = True
    ce_thres = [0.6, 0.6, 1, 0.4]
    # 0: front branch loss 1: backward branch loss
    # 2: top branch loss 3: side branch loss


def config_info():
    attrs = ['%s:%s' % (k, v)
             for k, v in Config.__dict__.items() if '__' not in k]
    return '\n'.join(attrs)
