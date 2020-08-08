import torch
import torch.nn as nn
import numpy as np
import logging
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    TensorDataset,
    Sampler
)
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_utils import (
    PoolerStartLogits,
    PoolerEndLogits
)
from tqdm import tqdm, trange


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


def preprocess_dataset(dataset):
    tensors = [[] for i in range(4)]
    for i in trange(len(dataset)):
        example = dataset[i]
        context_start_idx = (example[2] == 1).nonzero()[0].item()

        input_ids = example[0][context_start_idx - 1:]
        input_ids[0] = 101  # change to [CLS] token
        tensors[0].append(input_ids)

        attention_mask = example[1][context_start_idx - 1:]
        tensors[1].append(attention_mask)

        start_position = example[3] - context_start_idx + 1
        tensors[2].append(start_position.unsqueeze(-1))

        end_position = example[4] - context_start_idx + 1
        tensors[3].append(end_position.unsqueeze(-1))

    tensors_padded = []
    for i, sequences in enumerate(tqdm(tensors)):
        padded = pad_sequence(sequences, batch_first=True)
        if i > 1:
            tensors_padded.append(padded.squeeze(-1))
        else:
            tensors_padded.append(padded)

    new_dataset = TensorDataset(*tensors_padded)
    return new_dataset


class BertModelWithXLNetHead(BertPreTrainedModel):
    """
    BERT-type model of choice with a question answering head inspired by XLNet,
    i.e. performing a join probability prediction over spans rather than independently
    over start and end logits

    (code adapted from huggingface)
    """
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.start_logits_pooler = PoolerStartLogits(config)
        self.end_logits_pooler = PoolerEndLogits(config)
        self.loss_fn = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        start_positions,
        end_positions,
        max_ans_length=30,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]

        p_mask_start = 1 - attention_mask.clone()
        p_mask_start[:, 0] = 1

        start_logits = self.start_logits_pooler(hidden_states, p_mask=p_mask_start)

        p_mask_end = p_mask_start.clone()
        for i, s in enumerate(start_positions):
            p_mask_end[i, :s] = 1
            p_mask_end[i, s + max_ans_length:] = 1

        end_logits = self.end_logits_pooler(hidden_states, start_positions=start_positions, p_mask=p_mask_end)

        start_loss = self.loss_fn(start_logits, start_positions)
        end_loss = self.loss_fn(end_logits, end_positions)
        loss = start_loss + end_loss

        return (loss, )

    def generate(
        self,
        input_ids,
        attention_mask,
        num_answers=1,
        max_ans_length=30,
        top_p_value=0.9,
        top_k_value=5,
        temperature=1.0,

    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]  # B X L X H

        # Start positions
        p_mask_start = 1 - attention_mask.clone()  # B x L
        p_mask_start[:, 0] = 1

        start_logits = self.start_logits_pooler(hidden_states, p_mask=p_mask_start)
        start_top_logits = self.top_k_top_p_filtering(
            start_logits,
            top_k=top_k_value,
            top_p=top_p_value,
            min_tokens_to_keep=num_answers
        )
        start_top_probs = F.softmax(start_top_logits / temperature, dim=-1)  # shape B x L
        start_sampled_ids = torch.multinomial(start_top_probs, num_samples=num_answers)  # shape B x N

        # End positions
        bs, slen, hsz = hidden_states.size()
        hidden_states_expanded = hidden_states.unsqueeze(1).expand(bs, num_answers, slen, hsz)  # B x N x L x H

        start_positions = start_sampled_ids[..., None, None].expand(bs, num_answers, slen, hsz)  # B x N x 1 x H
        start_states = hidden_states_expanded.gather(-2, start_positions)  # B x N x 1 x H
        start_states = start_states.expand(bs, num_answers, slen, hsz)

        p_mask_end = p_mask_start.clone().unsqueeze(1).repeat(1, num_answers, 1)  # B x N x L
        for bi in range(len(p_mask_end)):
            for pi, idx in enumerate(start_sampled_ids[bi]):
                p_mask_end[bi, pi, : idx] = 1
                p_mask_end[bi, pi, idx + max_ans_length:] = 1

        end_logits = self.end_logits_pooler(
            hidden_states_expanded,
            start_states=start_states,
            p_mask=p_mask_end
        )  # B x N x L

        end_top_logits = self.top_k_top_p_filtering(
            end_logits,
            top_k=top_k_value,
            top_p=top_p_value,
            min_tokens_to_keep=1
        )  # B x N x L
        end_top_probs = F.softmax(end_top_logits / temperature, dim=-1)
        end_top_probs_view = end_top_probs.view(-1, slen)  # B*N x L
        end_sampled_ids = torch.multinomial(end_top_probs_view, num_samples=1)  # shape B*N x 1
        end_sampled_ids = end_sampled_ids.unsqueeze(-1).view(bs, -1).contiguous()  # B x N

        return start_sampled_ids, end_sampled_ids

    @staticmethod
    def top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (batch size, vocabulary size)
                if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Make sure we keep at least min_tokens_to_keep per batch example in the output
            Originally from: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
            Adapted by: https://github.com/huggingface/transformers/blob/d5bc32ce92ace9aaec7752e0b89d51ba18903a1b/src/transformers/generation_utils.py
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value

        return logits


# class BertModelWithXLNetHeadForONNX(BertPreTrainedModel):
#     """
#     BERT-type model of choice with a question answering head inspired by XLNet,
#     i.e. performing a join probability prediction over spans rather than independently
#     over start and end logits

#     (code adapted from huggingface)
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         self.start_logits_pooler = PoolerStartLogits(config)
#         self.end_logits_pooler = PoolerEndLogits(config)
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.bert_onnx = None

#         self.init_weights()
    
#     def forward(self, *args, **kwargs):
#         pass
    
#     def generate(
#         self,
#         input_ids,
#         attention_mask,
#         device=torch.device('cpu'),
#         num_answers=1,
#         max_ans_length=30,
#         top_p_value=0.9,
#         top_k_value=5,
#         temperature=1.0,

#     ):
#         inputs_onnx = {
#             'input_ids': input_ids.numpy(),
#             'attention_mask': attention_mask.numpy(),
#             'token_type_ids': torch.zeros_like(attention_mask, dtype=torch.long).numpy()
#         }
#         outputs = self.bert_onnx.run(None, inputs_onnx)
#         hidden_states = torch.tensor(outputs[0], device=device)  # B x L x H

#         # Start positions
#         p_mask_start = 1 - attention_mask.to(device)  # B x L
#         p_mask_start[:, 0] = 1

#         start_logits = self.start_logits_pooler(hidden_states, p_mask=p_mask_start)
#         start_top_logits = self.top_k_top_p_filtering(
#             start_logits,
#             top_k=top_k_value,
#             top_p=top_p_value,
#             min_tokens_to_keep=num_answers
#         )
#         start_top_probs = F.softmax(start_top_logits / temperature, dim=-1)  # shape B x L
#         start_sampled_ids = torch.multinomial(start_top_probs, num_samples=num_answers)  # shape B x N

#         # End positions
#         bs, slen, hsz = hidden_states.size()
#         hidden_states_expanded = hidden_states.unsqueeze(1).expand(bs, num_answers, slen, hsz)  # B x N x L x H

#         start_positions = start_sampled_ids[..., None, None].expand(bs, num_answers, slen, hsz)  # B x N x 1 x H
#         start_states = hidden_states_expanded.gather(-2, start_positions)  # B x N x 1 x H
#         start_states = start_states.expand(bs, num_answers, slen, hsz)

#         p_mask_end = p_mask_start.clone().unsqueeze(1).repeat(1, num_answers, 1)  # B x N x L
#         for bi in range(len(p_mask_end)):
#             for pi, idx in enumerate(start_sampled_ids[bi]):
#                 p_mask_end[bi, pi, : idx] = 1
#                 p_mask_end[bi, pi, idx + max_ans_length:] = 1

#         end_logits = self.end_logits_pooler(
#             hidden_states_expanded,
#             start_states=start_states,
#             p_mask=p_mask_end
#         )  # B x N x L

#         end_top_logits = self.top_k_top_p_filtering(
#             end_logits,
#             top_k=top_k_value,
#             top_p=top_p_value,
#             min_tokens_to_keep=1
#         )  # B x N x L
#         end_top_probs = F.softmax(end_top_logits / temperature, dim=-1)
#         end_top_probs_view = end_top_probs.view(-1, slen)  # B*N x L
#         end_sampled_ids = torch.multinomial(end_top_probs_view, num_samples=1)  # shape B*N x 1
#         end_sampled_ids = end_sampled_ids.unsqueeze(-1).view(bs, -1).contiguous()  # B x N

#         return start_sampled_ids, end_sampled_ids

#     @staticmethod
#     def top_k_top_p_filtering(
#         logits: torch.Tensor,
#         top_k: int = 0,
#         top_p: float = 1.0,
#         filter_value: float = -float("Inf"),
#         min_tokens_to_keep: int = 1,
#     ) -> torch.Tensor:
#         """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
#             Args:
#                 logits: logits distribution shape (batch size, vocabulary size)
#                 if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
#                 if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
#                 Make sure we keep at least min_tokens_to_keep per batch example in the output
#             Originally from: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
#             Adapted by: https://github.com/huggingface/transformers/blob/d5bc32ce92ace9aaec7752e0b89d51ba18903a1b/src/transformers/generation_utils.py
#         """
#         if top_k > 0:
#             top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
#             # Remove all tokens with a probability less than the last token of the top-k
#             indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#             logits[indices_to_remove] = filter_value

#         if top_p < 1.0:
#             sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#             cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

#             # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
#             sorted_indices_to_remove = cumulative_probs > top_p
#             if min_tokens_to_keep > 1:
#                 # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
#                 sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
#             # Shift the indices to the right to keep also the first token above the threshold
#             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#             sorted_indices_to_remove[..., 0] = 0

#             # scatter sorted tensors to original indexing
#             indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
#             logits[indices_to_remove] = filter_value

#         return logits


def generator_collate_fn(batch, tokenizer, max_context_length):
    lengths = [len(b) for b in batch]
    batch = [tokenizer.encode(b, add_special_tokens=False) for b in batch]
    batch = [b[:max_context_length - 2] for b in batch]

    batch_extra = [[tokenizer.cls_token_id] + b + [tokenizer.sep_token_id] for b in batch]
    batch_tensors = [torch.tensor(b, dtype=torch.long) for b in batch_extra]
    batch_collated = pad_sequence(batch_tensors, batch_first=True)

    # add attention_mask
    attention_mask = torch.ones_like(batch_collated)
    for i in range(len(attention_mask)):
        attention_mask[i][lengths[i]:] = 0

    return (batch_collated, attention_mask)


class GeneratorBatchSampler(Sampler):
    """
    """
    def __init__(self, dataset, tokenizer, batch_size, shuffle, min_context_length):
        self.ds = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_context_length = min_context_length
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    def __iter__(self):
        index_array = list(range(len(self.ds)))
        # print(len(index_array))
        # print(type(index_array))
        if self.shuffle:
            np.random.shuffle(index_array)

        batch_ids = []
        for i in index_array:
            text = self.ds[i]
            tokenized = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokenized) >= self.min_context_length:
                batch_ids.append(i)
            if len(batch_ids) == self.batch_size:
                yield batch_ids
                batch_ids = []

    def __len__(self):
        return int(np.ceil(len(self.ds) / self.batch_size))
