import random
import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.data import (
    Sampler,
    Dataset
)


def find_subsequences(text, seq):
    """
    Looks for specified subsequences within the given tensor.

    Args:
        * text - 1D tensor,
        * seq - 1D tensor,

    Returns:
        * a list of tuples, where each tuple contains a start and end indices
        of matching subsequences.
    """
    start_ids = (text == seq[0]).nonzero()
    seq_len = len(seq)
    is_subsequence = torch.ones(len(start_ids), dtype=bool)

    for i, start in enumerate(start_ids):
        if (start + len(seq) - 1) >= len(text):
            is_subsequence[i] = False
            continue
        for j, value in enumerate(seq[1:], 1):
            if text[start + j] != value:
                is_subsequence[i] = False
                break

    return [(s, s + seq_len) for i, s in enumerate(start_ids) if is_subsequence[i]]


class WikiDataset(Dataset):
    """
    A wrapper for 'wiki40b' dataset. Upon indexing, returns a section
    text without any titles and with '_NEWLINE_' substituted for '\n'.
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
    """
    Custom batch sampler to be used as an argument to pytorch DataLoader. Returns
    batches with examples of similar length in order to minimize padding.
    
    It devides the shuffled dataset into buckets of size 50 * batch_size,
    sorts each bucket according to the length of its sequences, and shuffles
    the buckets.
    """
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
