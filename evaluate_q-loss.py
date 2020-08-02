import logging
from types import SimpleNamespace
import os
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
)

from data_processor import load_and_cache_examples
from question_generator_utils import (
    dynamic_padding_collate_fn,
    preprocess_dataset
)
from utils import (
    CustomBatchSampler,
    set_seed
)

from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


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
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    iterator = tqdm(dev_dataloader, desc="Evaluation", smoothing=0.05)
    loss_cum = None
    num_batch = 0
    for step, batch_cpu in enumerate(iterator):
        num_batch += 1

        batch = tuple(t.to(args.device) for t in batch_cpu)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }

        with torch.no_grad():
            outputs = model(**inputs)

            # Calculate loss of just the question part
            q_mask = (inputs['token_type_ids'] == 2)
            masked_labels = inputs['input_ids'].masked_fill(~q_mask, 0)
            shift_labels = masked_labels[..., 1:].contiguous()

            lm_logits = outputs[0]
            shift_logits = lm_logits[..., : -1, :].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))

            if loss_cum is None:
                loss_cum = loss
            else:
                loss_cum += loss

    return loss_cum.item() / num_batch


def main():
    args = SimpleNamespace(
        debug=False,
        data_dir='data',
        train_file=None,
        directory='question_generator_gpt2-medium',
        model_name='gpt2-medium',
        do_lower_case=True,
        max_seq_length=448,
        overwrite_cache=False,
        cache_dir=None,
        dynamic_batching=True,
        dev_batch_size=4,
        fp16=True,
        fp16_opt_level='O2',
        no_cuda=False,
        seed=42,
    )

    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO
    )
    tb_writer = SummaryWriter(os.path.join(args.directory, 'TB_writer_q-loss_evaluation'))

    # Set seed
    set_seed(args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            from apex import amp
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Evaluate checkpoints
    def sort_fn(x):
        return int(x.split('-')[-1])
    checkpoints = [x for x in os.listdir(args.directory) if 'checkpoint' in x]
    checkpoints = sorted(checkpoints, key=sort_fn)
    for checkpoint in checkpoints:
        logging.info(f'Evaluating checkpint : {checkpoint}')
        global_step = checkpoint.split('-')[-1]

        # Load pretrained model and tokenizer
        config = GPT2Config.from_pretrained(
            os.path.join(args.directory, checkpoint),
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        tokenizer = GPT2Tokenizer.from_pretrained(
            os.path.join(args.directory, checkpoint),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer.encode = partial(tokenizer.encode, is_pretokenized=True, truncation=True)
        tokenizer.encode_plus = partial(tokenizer.encode_plus, is_pretokenized=True, truncation=True)

        model = GPT2LMHeadModel.from_pretrained(
            os.path.join(args.directory, checkpoint),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
        if args.fp16:
            model = amp.initialize(model, opt_level=args.fp16_opt_level)

        # Load data
        dev_dataset = load_and_cache_examples(args, tokenizer, 'quest_gen', evaluate=True, gpt=True)
        dev_dataset = preprocess_dataset(dev_dataset, tokenizer)

        # Evaluation
        loss = evaluate(args, dev_dataset, model)
        tb_writer.add_scalar('dev_loss', loss, global_step)


if __name__ == '__main__':
    main()
