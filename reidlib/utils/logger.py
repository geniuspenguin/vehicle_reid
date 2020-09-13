from torch.utils.tensorboard import SummaryWriter
import time


def time_passed(start, end, rnd=1):
    return round(end - start, rnd)


class Logger(SummaryWriter):
    def info(self, tag, text, time_report=True):
        if time_report:
            text = time.strftime("%H:%M:%S {}: ".format(
                tag), time.localtime()) + text
        super().add_text(tag, text)
        print('{:>8}:{}'.format(tag, text))

    def add_text(self, tag, text, time_report=True):
        if time_report:
            text = time.strftime("%H:%M:%S {}: ".format(
                tag), time.localtime()) + text
        super().add_text(tag, text)

# if __name__ == '__main__':
#     import numpy as np
#     logger = Logger()
#     for n_iter in range(100):
#         logger.add_scalar('Loss/train', np.random.random(), n_iter)
#         logger.add_scalar('Loss/test', np.random.random(), n_iter)
#         logger.add_scalar('Accuracy/train', np.random.random(), n_iter)
#         logger.add_scalar('Accuracy/test', np.random.random(), n_iter)
#         logger.add_text('train', str(n_iter) + 'step, loss')
#         logger.info('test', str(n_iter)+'acc')
