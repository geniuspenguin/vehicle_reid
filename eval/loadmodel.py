import torch
import torch.nn as nn

def import_parsing_model(path, backbone, branch):
    checkpoint = torch.load(path)
    backbone.load_state_dict(checkpoint['model'])
    branch.load_state_dict(checkpoint['branch0'])
    return nn.Sequential(backbone, branch)

def import_aio_model(path, backbone, branch):
    checkpoint = torch.load(path)
    backbone.load_state_dict(checkpoint['model'])
    branch.load_state_dict(checkpoint['branch0'])

def import_baseline(path, backbone):
    checkpoint = torch.load(path)
    backbone.load_state_dict(checkpoint['model'])
    