# coding=utf-8
#
#
# Modified by Piotr Skalski from https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_squad.py
#
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import logging
import os
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset
)

from data_processor import load_and_cache_examples
from utils import (
    CustomBatchSampler,
    dynamic_padding_collate_fn,
    set_seed
)

from transformers import (
    AdamW,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_utils import (
    PoolerStartLogits,
    PoolerEndLogits
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


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
        input_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
        max_ans_length=30,
        top_p_value=0.9,
        top_k_value=5,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]
        # logger.debug('\nhidden_states : {}'.format(hidden_states.shape))
        # logger.debug('hidden_states : {}'.format(hidden_states))
        # logger.debug('attention_mask : {}'.format(attention_mask.shape))
        # logger.debug('attention_mask : {}'.format(attention_mask))

        p_mask_start = 1 - attention_mask.clone()
        p_mask_start[:, 0] = 1
        # logger.debug('p_mask_start : {}'.format(p_mask_start.shape))
        # logger.debug('p_mask_start : {}'.format(p_mask_start))

        if start_positions is not None and end_positions is not None:
            start_logits = self.start_logits_pooler(hidden_states, p_mask=p_mask_start)

            # during training, compute the end logits based on the ground truth of the start position
            p_mask_end = p_mask_start.clone()
            for i, s in enumerate(start_positions):
                p_mask_end[i, :s] = 1
                p_mask_end[i, s + max_ans_length:] = 1
            # logger.debug('p_mask_end : {}'.format(p_mask_end.shape))
            # logger.debug('p_mask_end : {}'.format(p_mask_end))
            end_logits = self.end_logits_pooler(hidden_states, start_positions=start_positions, p_mask=p_mask_end)

            # logger.debug('start_logits : {}'.format(start_logits.shape))
            # logger.debug('start_logits : {}'.format(start_logits))
            # logger.debug('end_logits : {}'.format(end_logits.shape))
            # logger.debug('end_logits : {}'.format(end_logits))

            start_loss = self.loss_fn(start_logits, start_positions)
            end_loss = self.loss_fn(end_logits, end_positions)
            loss = start_loss + end_loss
            # logger.debug('loss : {}'.format(loss.item()))

            return (loss, )

        else:
            start_logits = self.start_logits_pooler(hidden_states, p_mask=p_mask_start)
            start_probs = F.softmax(start_logits, dim=-1)  # B X L

            start_top_probs, start_top_idxs = torch.topk(
                start_probs, top_k_value, dim=-1
            )  # B x top-k
            start_top_probs, start_top_idxs = top_p(
                start_top_probs, start_top_idxs,
                p=top_p_value, dim=-1
            )  # list, shape B, of tensors, shape top-p

            # iterate over batches
            max_start, max_end, max_probs = [], [], []
            for i, start_top_idx in enumerate(start_top_idxs):  # shape top-p
                hidden_states_expanded = hidden_states[i].unsqueeze(0).expand(len(start_top_idx), -1)  # top-p x L x H
                p_mask_end = p_mask_start[i].clone().unsqueeze(0).expand(len(start_top_idx), -1)  # top-p x L
                for pi, idx in enumerate(start_top_idx):
                    p_mask_end[pi, : idx] = 1
                    p_mask_end[pi, idx + max_ans_length:] = 1

                end_logits = self.end_logits_pooler(hidden_states_expanded,
                                                    start_positions=start_top_idx,
                                                    p_mask=p_mask_end
                                                    )
                end_probs = F.softmax(end_logits, dim=-1)
                end_max_probs, end_max_idxs = torch.max(end_probs, dim=-1)

                cum_probs = start_top_probs[i] * end_max_probs
                max_id = torch.argmax(cum_probs)

                max_probs.append(cum_probs[max_id])
                max_start.append(start_top_idx[max_id])
                max_end.append(end_max_idxs[max_id])

            return (torch.tensor(max_start), torch.tensor(max_end), torch.tensor(max_probs))


def top_p(tensor, indices, p, dim):
    cumsum = tensor.cumsum(dim=dim)
    top_p_idxs = [(t > p).nonzero()[0] for t in cumsum]

    top_p_tensors = [t[: idx + 1] for t, idx in zip(tensor, top_p_idxs)]
    top_p_indices = [i[: idx + 1] for i, idx in zip(indices, top_p_idxs)]

    return top_p_tensors, top_p_indices


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


def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'TB_writer'))

    if args.dynamic_batching:
        train_sampler = CustomBatchSampler(train_dataset, args.train_batch_size)
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=dynamic_padding_collate_fn
        )
    else:
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=args.train_batch_size, num_workers=1)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")

    # Added here for reproductibility
    set_seed(args)

    loss_cum = None
    # torch.autograd.set_detect_anomaly(True)
    for _ in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", smoothing=0.05)
        for step, batch_cpu in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            batch = tuple(t.to(args.device) for t in batch_cpu)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "start_positions": batch[2].squeeze(-1),
                "end_positions": batch[3].squeeze(-1),
                "max_ans_length": args.max_ans_length,
            }

            outputs = model(**inputs)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    if loss_cum is None:
                        loss_cum = loss.detach()
                    else:
                        loss_cum += loss.detach()

            else:
                loss.backward()
                if loss_cum is None:
                    loss_cum = loss.detach()
                else:
                    loss_cum += loss.detach()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log train metrics
                if (not global_step % args.train_logging_steps) and args.train_logging_steps > 0:
                    tb_writer.add_scalar('train_loss', loss_cum.item() / args.train_logging_steps, global_step)

                    loss_cum = None
                # Log dev metrics
                if args.dev_logging_steps > 0 and global_step % args.dev_logging_steps == 0 and args.evaluate_during_training:
                    dev_loss = evaluate(args, dev_dataset, model)
                    tb_writer.add_scalar("dev_loss", dev_loss, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)

                # Save model checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        tb_writer.close()


def evaluate(args, dev_dataset, model):
    if args.dynamic_batching:
        dev_sampler = CustomBatchSampler(dev_dataset, args.dev_batch_size)
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_sampler=dev_sampler,
            num_workers=1,
            collate_fn=dynamic_padding_collate_fn
        )
    else:
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler,
                                    batch_size=args.dev_batch_size, num_workers=1)

    model.eval()
    iterator = tqdm(dev_dataloader, desc="Evaluation", smoothing=0.05)
    loss_cum = None
    num_batch = 0
    for step, batch_cpu in enumerate(iterator):
        num_batch += 1

        batch = tuple(t.to(args.device) for t in batch_cpu)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "start_positions": batch[2].squeeze(-1),
            "end_positions": batch[3].squeeze(-1),
            "max_ans_length": args.max_ans_length,
        }

        with torch.no_grad():
            outputs = model(**inputs)
            if loss_cum is None:
                loss_cum = outputs[0]
            else:
                loss_cum += outputs[0]

    model.train()

    return loss_cum.item() / num_batch


def main():
    parser = get_parser()
    args = parser.parse_args()

    if not args.model_name:
        args.model_name = args.model_path

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    config = BertConfig.from_pretrained(
        args.config_name if args.config_name else args.model_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = BertModelWithXLNetHead.from_pretrained(
        args.model_path,
        from_tf=bool(".ckpt" in args.model_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    train_dataset = preprocess_dataset(train_dataset)

    dev_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    dev_dataset = preprocess_dataset(dev_dataset)

    train(args, train_dataset, dev_dataset, model, tokenizer)
    logging.info('Finished training !')

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Good practice: save your training arguments together with the trained model
    logger.info("Saving final model checkpoint to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Set DEBUG logger level",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Model name used in file names",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument(
        "--max_ans_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated.",
    )

    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training.")
    parser.add_argument("--dev_batch_size", default=8, type=int, help="Batch size for evaluation.")
    parser.add_argument("--dynamic_batching", action='store_true', help="Do dynamic padding and batching")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )

    parser.add_argument("--train_logging_steps", type=int, default=500, help="Log training loss every X updates steps.")
    parser.add_argument("--dev_logging_steps", type=int, default=500, help="Log dev loss every X updates steps.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="Whether to run evaluations on dev set during training.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    return parser


if __name__ == "__main__":
    main()