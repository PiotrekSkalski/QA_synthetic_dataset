import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
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

        p_mask_start = 1 - attention_mask.clone()
        p_mask_start[:, 0] = 1  # B x L

        start_logits = self.start_logits_pooler(hidden_states, p_mask=p_mask_start)
        start_probs = F.softmax(start_logits / temperature, dim=-1)  # B X L

        start_top_probs, start_top_idxs = torch.topk(
            start_probs, top_k_value, dim=-1
        )  # B x top-k
        start_top_probs, start_top_idx, = self.top_p(
            start_top_probs, start_top_idxs,
            p=top_p_value, dim=-1
        )  # list, shape B, of tensors, shape top-p

        # sample start positions
        start_idxs = []
        for i in range(len(start_top_probs)):
            with_replacement = False if len(start_top_probs[i]) > num_answers else True
            indices = torch.multinomial(start_top_probs[i],
                                        num_samples=num_answers,
                                        replacement=with_replacement
                                        )
            start_idxs.append(start_top_idx[i][indices].unsqueeze(0))
        start_idxs = torch.cat(start_idxs, dim=0)  # shape B x N

        bs, slen, hsz = hidden_states.size()
        hidden_states_expanded = hidden_states.unsqueeze(1).expand(bs, num_answers, slen, hsz)  # B x N x L x H

        start_positions = start_idxs[:, :, None, None].expand(bs, num_answers, slen, hsz)  # B x N x 1 x H
        start_states = hidden_states_expanded.gather(-2, start_positions)  # B x N x 1 x H
        start_states = start_states.expand(bs, num_answers, slen, hsz)

        p_mask_end = p_mask_start.clone().unsqueeze(1).repeat(1, num_answers, 1)  # B x N x L
        for bi in range(len(p_mask_end)):
            for pi, idx in enumerate(start_idxs[bi]):
                p_mask_end[bi, pi, : idx] = 1
                p_mask_end[bi, pi, idx + max_ans_length:] = 1

        end_logits = self.end_logits_pooler(hidden_states_expanded,
                                            start_states=start_states,
                                            p_mask=p_mask_end
                                            )  # B x N x L
        end_probs = F.softmax(end_logits / temperature, dim=-1)

        end_top_probs, end_top_idxs = torch.topk(
            end_probs, top_k_value, dim=-1
        )  # B x N x top-k
        end_idxs = []
        for end_top_probs_n, end_top_idxs_n in zip(
            end_top_probs.split(1, dim=1,),
            end_top_idxs.split(1, dim=1)
        ):
            end_top_probs_n, end_top_idx_n, = self.top_p(
                end_top_probs_n.squeeze(1), end_top_idxs_n.squeeze(1),
                p=top_p_value, dim=-1
            )  # list, shape B, of tensors, shape top-p

            # sample start positions
            end_idxs_n = []
            for i in range(len(end_top_probs_n)):
                indices = torch.multinomial(end_top_probs_n[i], 1)
                end_idxs_n.append(end_top_idx_n[i][indices].unsqueeze(0))
            end_idxs_n = torch.cat(end_idxs_n, dim=0)  # shape B
            end_idxs.append(end_idxs_n.unsqueeze(1))
        end_idxs = torch.cat(end_idxs, dim=1)

        return start_idxs, end_idxs

    @staticmethod
    def top_p(tensor, indices, p, dim):
        cumsum = tensor.cumsum(dim=dim)
        top_p_idxs = [(t > p).nonzero()[0] if (t > p).any() else torch.tensor(len(t)) for t in cumsum]

        top_p_tensors = [t[: idx + 1] for t, idx in zip(tensor, top_p_idxs)]
        top_p_indices = [i[: idx + 1] for i, idx in zip(indices, top_p_idxs)]

        return top_p_tensors, top_p_indices
