import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import itertools
import copy

# Both samplers are passed a data_source (likely your dataset) that has following members:
# * label_to_samples - mapping of label ids (zero based integer) to samples for that label


class PKSampler(Sampler):

    def __init__(self, data_source, p=64, k=16):
        super().__init__(data_source)
        self.p = p
        self.k = k
        self.data_source = data_source

    def __iter__(self):
        pk_count = len(self) // (self.p * self.k)
        for _ in range(pk_count):
            labels = np.random.choice(np.arange(
                len(self.data_source.label_to_samples.keys())), self.p, replace=False)
            for l in labels:
                indices = self.data_source.label_to_samples[l]
                replace = True if len(indices) < self.k else False
                for i in np.random.choice(indices, self.k, replace=replace):
                    yield i

    def __len__(self):
        pk = self.p * self.k
        samples = ((len(self.data_source) - 1) // pk + 1) * pk
        return samples


def grouper(iterable, n):
    it = itertools.cycle(iter(iterable))
    for _ in range((len(iterable) - 1) // n + 1):
        yield list(itertools.islice(it, n))

# full label coverage per 'epoch'


class PKSampler_coverage(Sampler):

    def __init__(self, data_source, p=64, k=16):
        super().__init__(data_source)
        self.p = p
        self.k = k
        self.data_source = data_source

    def __iter__(self):
        rand_labels = np.random.permutation(
            np.arange(len(self.data_source.label_to_samples.keys())))
        for labels in grouper(rand_labels, self.p):
            for l in labels:
                indices = self.data_source.label_to_samples[l]
                replace = True if len(indices) < self.k else False
                for j in np.random.choice(indices, self.k, replace=replace):
                    yield j

    def __len__(self):
        num_labels = self.data_source.nr_id
        samples = ((num_labels - 1) // self.p + 1) * self.p * self.k
        return samples

