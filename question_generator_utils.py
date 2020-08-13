import os
import json
import torch
import logging
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, IterableDataset
from tqdm import tqdm, trange
from utils import find_subsequences


def dynamic_padding_collate_fn(batch_list):
    batch_uncollated = [[] for i in range(3)]

    for features in batch_list:
        length = features[1].sum().item()
        for i, feature in enumerate(features):
            batch_uncollated[i].append(feature[:length])

    batch_collated = []
    for batch in batch_uncollated:
        batch_collated.append(pad_sequence(batch, batch_first=True))

    return batch_collated


def preprocess_dataset(dataset, tokenizer):
    eos = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)
    q_start = torch.tensor(tokenizer.encode('question:'), dtype=torch.long)
    q_end = torch.tensor(tokenizer.encode(':question'), dtype=torch.long)

    tensors = [[] for i in range(3)]
    for i in trange(len(dataset)):
        example = dataset[i]

        context_start_idx = (example[2] == 1).nonzero()[0].item()
        try:
            context_end_idx = (example[1] == 0).nonzero()[0].item()
        except:
            context_end_idx = len(example[0]) - 1
        ans_start = example[3] - context_start_idx
        ans_end = example[4] - context_start_idx

        context = example[0][context_start_idx: context_end_idx]
        question = example[0][: context_start_idx]
        answer = example[0][example[3]: example[4] + 1]

        input_ids = torch.cat([
            context,
            eos,
            answer,
            eos,
            q_start,
            question,
            q_end,
            eos
        ])

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        token_type_ids = torch.cat([
            torch.zeros(len(context) + 1, dtype=torch.long),
            torch.ones(len(answer) + 1, dtype=torch.long),
            2 * torch.ones(len(question) + 3, dtype=torch.long)
        ])
        token_type_ids[ans_start: ans_end + 1] = 1

        tensors[0].append(input_ids)
        tensors[1].append(attention_mask)
        tensors[2].append(token_type_ids)

    tensors_padded = []
    for i, sequences in enumerate(tqdm(tensors)):
        padded = pad_sequence(sequences, batch_first=True)
        tensors_padded.append(padded)

    new_dataset = TensorDataset(*tensors_padded)
    return new_dataset


class SyntheticAnswersDataset(IterableDataset):

    def __init__(self, path, tokenizer):
        self.path_dir = path
        self.file_list = os.listdir(self.path_dir)
        self.file_list = [file for file in self.file_list if 'answers' in file]
        self.tokenizer = tokenizer
        self.eos_token = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        self.q_start_token = torch.tensor(self.tokenizer.encode('question:'), dtype=torch.long)

    def __iter__(self):
        for file in self.file_list:
            with open(os.path.join(self.path_dir, file), 'r') as f:
                examples = json.load(f)

            for example in examples:
                context = example['context']
                context_encoded = torch.tensor(self.tokenizer.encode(context), dtype=torch.long)
                generated_answers = example['generated_answers']

                for answer in generated_answers:
                    answer_text = ' ' + answer['answer']
                    answer_text_encoded = torch.tensor(self.tokenizer.encode(answer_text), dtype=torch.long)
                    subsequences = find_subsequences(context_encoded, answer_text_encoded)
                    if len(subsequences) == 0:
                        answer_text = answer['answer']
                        answer_text_encoded = torch.tensor(self.tokenizer.encode(answer_text), dtype=torch.long)
                        subsequences = find_subsequences(context_encoded, answer_text_encoded)
                        if len(subsequences) == 0:
                            continue
                    seq_num = answer['seq_num']
                    try:
                        ans_start, ans_end = subsequences[seq_num - 1]
                    except IndexError:
                        continue

                    input_ids = torch.cat([
                        context_encoded,
                        self.eos_token,
                        answer_text_encoded,
                        self.eos_token,
                        self.q_start_token
                    ])

                    token_type_ids = torch.cat([
                        torch.zeros(len(context_encoded) + 1, dtype=torch.long),
                        torch.ones(len(answer_text_encoded) + 1, dtype=torch.long),
                        2 * torch.ones(1, dtype=torch.long)
                    ])
                    token_type_ids[ans_start: ans_end + 1] = 1

                    yield (context, input_ids, token_type_ids)
