import random
import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.data import (
    Sampler,
    Dataset
)


class WikiDataset(Dataset):
    """
    """
    def __init__(self, wiki_dataset, cachedir='.'):
        self.ds = wiki_dataset
        cache = os.path.join(cachedir, 'cache_wiki')
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        if not os.path.exists(cache):
            self.indices = []
            for i, article in enumerate(tqdm(wiki_dataset, desc='Preparing dataset')):
                article = article['text']
                paragraphs = article.split('\n_START_PARAGRAPH_\n')[1:]
                self.indices.extend([(i, j) for j in range(len(paragraphs))])
            torch.save(self.indices, cache)
        else:
            self.indices = torch.load(cache)

    def __getitem__(self, index):
        i, j = self.indices[index]
        article = self.ds[i]['text']
        text = article.split('\n_START_PARAGRAPH_\n')[1:][j]
        text = text.split('\n_START_SECTION_\n')[0]
        text = text.replace('_NEWLINE_', '\n')

        return text

    def __len__(self):
        return len(self.indices)


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
        return int(np.ceil(len(self.dataset) // self.batch_size))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()
