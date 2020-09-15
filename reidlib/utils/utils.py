import torch


def no_grad_func(func):
    def func_nograd(*args, **kargs):
        with torch.no_grad():
            ret = func(*args, **kargs)
        return ret
    return func_nograd
