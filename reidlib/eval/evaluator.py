from torch.utils.data.dataloader import DataLoader, SequentialSampler

class Evaluator:

    def __init__(self, feature_extractor, testset, dim_out=0):
        self.extractor = feature_extractor
        self.testset = testset