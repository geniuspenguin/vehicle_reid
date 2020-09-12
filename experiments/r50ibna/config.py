class Config:
    epoch = 120
    P = 64
    K = 4
    batch_size = int(P*K)

    lr = lambda x: 3.5e-4
    warm_up_epochs = 10
    weight_decay = 5e-4

    batch_per_log = 1
    epoch_per_test = 10
    epoch_per_save = 10
