import argparse
import logging
import os
import json
from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertConfig
)
from nlp import load_dataset
# from tensorboardX import SummaryWriter

from answer_generator_utils import (
    BertModelWithXLNetHead,
    generator_collate_fn,
    GeneratorBatchSampler
)
from utils import (
    set_seed,
    WikiDataset,
    find_subsequences
)


logger = logging.getLogger(__name__)
# `os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def generate_answers(args, start_idxs, end_idxs, input_ids):
    """
    Args:
        args - parser from argparse;
        start_idxs - tensor; shape: batch x num_gen_ans
        end_idxs - tensor; shape: batch x num_gen_answer
        input_ids - tensor; shape: batch x max_seq_length
    Returns:
        answers - list of lists; shape: batch x num_gen_ans;
                  the inner list consists of tuples: (answer_tokens, int)
    """
    answers = []
    for i, (start_idx, end_idx) in enumerate(zip(start_idxs, end_idxs)):
        answer = []
        context = input_ids[i]
        for start, end in zip(start_idx, end_idx):
            ans = context[start: end + 1]
            subsequences = find_subsequences(context, ans)
            for i, (s, e) in enumerate(subsequences, 1):
                if s == start:
                    answer.append((ans, i))
                    break
        answers.append(answer)

    return answers


def generate(args, tokenizer, model):
    # tb_writer = SummaryWriter(os.path.join(args.output_dir, 'TB_writer'))

    dataset = WikiDataset(load_dataset('wiki40b', 'en')['train'])
    loader = DataLoader(
        dataset,
        batch_sampler=GeneratorBatchSampler(dataset,
                                            tokenizer,
                                            args.batch_size,
                                            True,
                                            args.max_context_length // 2),
        num_workers=args.threads,
        collate_fn=partial(generator_collate_fn,
                           tokenizer=tokenizer,
                           max_context_length=args.max_context_length)
    )

    num_iterations = 0
    num_generated_examples = 0
    file_num = 0

    examples = []
    while num_generated_examples < args.num_examples:
        for batch in tqdm(loader, desc='Generating...', smoothing=0.95):
            num_iterations += 1

            inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'max_ans_length': args.max_ans_length,
                'num_answers': args.ans_per_context,
                'top_k_value': args.ans_per_context + 2,
                'top_p_value': 0.9
            }

            with torch.no_grad():
                start_idxs, end_idxs = model.generate(**inputs)

            answers = generate_answers(args, start_idxs.detach().cpu(),
                                       end_idxs.detach().cpu(),
                                       batch[0])
            for i, answer_set in enumerate(answers):
                context_length = batch[1][i].sum() - 2
                context = tokenizer.decode(batch[0][i][1: context_length - 1])
                generated_answers = []

                for answer_tokens, seq_num in answer_set:
                    # check for duplicates
                    answer_text = tokenizer.decode(answer_tokens)
                    duplicates = [(answer_text == gen['answer'] and seq_num == gen['seq_num'])
                                  for gen in generated_answers]
                    if any(duplicates):
                        continue

                    generated_answers.append(
                        {
                            'answer': answer_text,
                            'seq_num': seq_num
                        }
                    )
                    num_generated_examples += 1

                examples.append(
                    {
                        'context': context,
                        'generated_answers': generated_answers
                    }
                )

            # Log numbers
            if not num_iterations % args.log_steps:
                pass
                # tb_writer.add_scalar('num_generated_examples', num_generated_examples, num_iterations)

            # Save examples
            if (not num_iterations % args.save_steps) or (num_generated_examples >= args.num_examples):
                filename = os.path.join(args.output_dir, f'answers_{file_num}.json')
                with open(filename, 'w') as outfile:
                    json.dump(examples, outfile)
                    examples = []
                    file_num += 1

                if num_generated_examples >= args.num_examples:
                    break
    # tb_writer.close()


def main():
    parser = get_parser()
    args = parser.parse_args()

    if not args.model_name:
        args.model_name = args.model_path

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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.eval()
    model.to(args.device)

    if args.fp16:
        try:
            import apex
            from apex import amp
            apex.amp.register_half_function(torch, "einsum")
            model_bert = amp.initialize(model.bert, opt_level=args.fp16_opt_level)
            model.bert = model_bert
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    generate(args, tokenizer, model)


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
        "--max_context_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--max_ans_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated.",
    )
    parser.add_argument(
        "--num_examples",
        default=3000000,
        type=int,
        help="Number of context-amswers pairs to generate."
    )
    parser.add_argument(
        "--ans_per_context",
        default=3,
        type=int,
        help="Number of answers generated for each context paragraph."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size for generation.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save generated context-answer pairs every 'save_steps' generated pair.")
    parser.add_argument("--log_steps", type=int, default=100, help="Log numbers every 'log_steps' batch steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
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
    parser.add_argument("--threads", type=int, default=1, help="Multiple threads for data loading")

    return parser


if __name__ == '__main__':
    main()
