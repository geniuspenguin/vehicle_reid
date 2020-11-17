import os


class Config:
    experiment_name = 'res50ibna'

    experiment_dir = os.path.dirname(__file__)
    data_dir = os.path.join(experiment_dir, 'expr_logs')
    model_dir = os.path.join(data_dir, 'models')
    log_dir = os.path.join(data_dir, 'log')

    checkpoint_path = os.path.join(log_dir, 'latest.pth')

    input_shape = (320, 320)

    nr_class = 576
    in_planes = 2048

    epoch = 90
    P = 6
    K = 8
    batch_size = int(P*K)

    weight_decay = 5e-4
    triplet_margin = 0
    weight_ce = 1
    weight_tri = 1

    batch_per_log = 1
    epoch_per_test = 3
    epoch_per_save = 3

    nr_query = 1678
    nr_test = 11579

    w_type = 1
    w_color = 1

    nr_worker = 4

    # warmup lr
    momentum = 0.9
    base_lr = 0.01
    milestones = [40, 70]
    gamma = 0.1
    warmup_factor = 0.01
    warmup_epoch = 10


def config_info():
    attrs = ['%s:%s' % (k, v)
             for k, v in Config.__dict__.items() if '__' not in k]
    return '\n'.join(attrs)