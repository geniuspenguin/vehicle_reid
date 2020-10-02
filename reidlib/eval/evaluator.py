# from torch.utils.data.dataloader import DataLoader, SequentialSampler

# class Evaluator:

#     def __init__(self, feature_extractor, testset, dim_out=0, batch_size):
#         self.extractor = feature_extractor
#         self.testset = testset
#         seq_sampler = SequentialSampler(testset)
#         self.test_loader = DataLoader(testset, batch_size=batch_size, sampler=seq_sampler, pin_memory=True)
#         self.nr_query = testset.get_num_query()