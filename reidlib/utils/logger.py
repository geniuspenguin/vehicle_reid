
import time

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from termcolor import colored

def sec2min_sec(start, end):
    secs = end - start
    secs = int(secs)
    mins = secs // 60
    secs = secs % 60
    return mins, secs


class Logger(SummaryWriter):
    def __init__(self,time_color='green', tag_color='magenta', stage_color='blue', **kargs):
        super().__init__(**kargs)
        self.time_color = time_color
        self.tag_color = tag_color
        self.stage_color = stage_color

    def info(self, tag, text, stage=None, time_report=True):
        clock = ''
        if time_report:
            clock = time.strftime("%H:%M:%S", time.localtime())
        clock_print = '[{}]'.format(clock) if clock else ''
        clock_print = colored(clock_print, self.time_color)

        tag_print = '[{}]'.format(tag)
        tag_print = colored(tag_print, self.tag_color)

        stage_str = ''
        if stage:
            e, b = stage
            stage_str = 'e:{:0>3} b:{:0>4}'.format(e, b)
        stage_print = '[{}]'.format(stage_str) if stage_str else ''
        stage_print = colored(stage_print, self.stage_color)

        text = stage_str + text

        print('{}{}{}{}'.format(clock_print, tag_print, stage_print, text))
        super().add_text(tag, '{}{}'.format(clock, text))

    def add_text(self, tag, text, time_report=True):
        clock = ''
        if time_report:
            clock = time.strftime("[%H:%M:%S] ", time.localtime())
        super().add_text(tag, '{}{}'.format(clock, text))

def model_summary(model, input_shape=(3, 224, 224)):
    if not next(model.parameters()).is_cuda:
        model = model.cuda()
    summary_state = summary(model, input_shape, verbose=0)
    return str(summary_state)

# if __name__ == '__main__':
#     from torchvision import models
#     vgg = models.vgg16()
#     s = model_summary(vgg)
#     print('####\n', s)
#     import numpy as np
#     logger = Logger()
#     for n_iter in range(100):
#         logger.add_scalar('Loss/train', np.random.random(), n_iter)
#         logger.add_scalar('Loss/test', np.random.random(), n_iter)
#         logger.add_scalar('Accuracy/train', np.random.random(), n_iter)
#         logger.add_scalar('Accuracy/test', np.random.random(), n_iter)
#         logger.add_text('train', str(n_iter) + 'step, loss')
#         logger.info('test', str(n_iter)+'acc')
