import random
import numpy as np
import torch
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence


class CustomBatchSampler(Sampler):

    def __init__(self, dataset, batch_size, shuffle=True, bucket_size=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if not bucket_size:
            self.bucket_size = 50 * batch_size
        else:
            self.bucket_size = bucket_size

    def __iter__(self,):
        index_array = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(index_array)

        bucket_num = int(np.ceil(len(self.dataset) / self.bucket_size))
        for ib in range(bucket_num):
            bucket_idx = index_array[ib * self.bucket_size: (ib + 1) * self.bucket_size]
            bucket = [idx for idx in bucket_idx if idx < len(self.dataset)]
            bucket = sorted(bucket, key=lambda idx: self.dataset[idx][1].sum().item(), reverse=True)

            batch_num = int(np.ceil(len(bucket) / self.batch_size))
            batches = []
            for i in range(batch_num):
                batch_idx = range(i * self.batch_size, (i + 1) * self.batch_size)
                batch = [bucket[idx] for idx in batch_idx if idx < len(bucket)]
                batches.append(batch)
            np.random.shuffle(batches)

            for batch in batches:
                yield batch

    def __len__(self,):
        return int(np.ceil(len(self.dataset)//self.batch_size))


def dynamic_padding_collate_fn(batch_list):
    batch_uncollated = [[] for i in range(4)]

    for features in batch_list:
        length = features[1].sum().item()
        for i, feature in enumerate(features):
            if i < 2:
                batch_uncollated[i].append(feature[:length])
            else:
                batch_uncollated[i].append(torch.tensor([feature.item()]))

    batch_collated = []
    for batch in batch_uncollated:
        batch_collated.append(pad_sequence(batch, batch_first=True))

    return batch_collated


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()