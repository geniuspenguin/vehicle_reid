import os

class Config:
    experiment_dir = os.path.dirname(__file__)
    data_dir = os.path.join(experiment_dir, 'expr_data')
    model_dir = os.path.join(data_dir, 'models')
    log_dir = os.path.join(data_dir, 'log')

    epoch = 120
    P = 64
    K = 4
    batch_size = int(P*K)

    weight_decay = 5e-4

    batch_per_log = 1
    epoch_per_test = 10
    epoch_per_save = 10


