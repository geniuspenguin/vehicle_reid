from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, path='./runs'):
        self.writer = SummaryWriter(log_dir=path)
        self.add_text = self.writer.add_text
        self.add_scalar = self.writer.add_scalar

    def log_and_print(self, tag, text):
        self.add_text(tag, text)
        print('{:>8}:{}'.format(tag, text))

# if __name__ == '__main__':
#     import numpy as np
#     logger = Logger()
#     for n_iter in range(100):
#         logger.add_scalar('Loss/train', np.random.random(), n_iter)
#         logger.add_scalar('Loss/test', np.random.random(), n_iter)
#         logger.add_scalar('Accuracy/train', np.random.random(), n_iter)
#         logger.add_scalar('Accuracy/test', np.random.random(), n_iter)
#         logger.add_text(str(n_iter), str(n_iter) + 'step')
#         logger.log_and_print('test', str(n_iter)+'th')