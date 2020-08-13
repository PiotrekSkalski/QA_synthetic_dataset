import os
import json
import logging
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    BertForQuestionAnswering,
    BertTokenizer
)
from question_generator_utils import SyntheticAnswersDataset
from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def generate(args, GPT2_tokenizer, GPT2_model, BERT_tokenizer, BERT_model):
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'TB_writer'))

    dataset = SyntheticAnswersDataset(args.generated_answers_path, GPT2_tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.threads
    )

    num_iterations = 1
    num_generated_examples = 0
    num_generated_answerable_questions = 0
    num_generated_unanswerable_questions = 0
    file_num = 0

    q_start_token = torch.tensor(GPT2_tokenizer.encode('question:'), dtype=torch.long, device=args.device)
    q_end_token = torch.tensor(GPT2_tokenizer.encode(':question'), dtype=torch.long, device=args.device)

    examples = []
    for batch in tqdm(loader, desc='Generating...', smoothing=0.95):
        context = batch[0][0]
        output_sequences = GPT2_model.generate(
            input_ids=batch[1].to(args.device),
            token_type_ids=batch[2].to(args.device),
            max_length=args.max_q_length + batch[1].shape[-1],
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=True,
            num_return_sequences=args.num_pregenerated_questions
        )

        # Decode questions
        questions = []
        for output in output_sequences:
            q_start_idx = (output.squeeze() == q_start_token).nonzero()[0]
            q_end_indices = (output.squeeze() == q_end_token).nonzero()
            if len(q_end_indices):
                q_end_idx = q_end_indices[0]
            else:
                continue
            question = GPT2_tokenizer.decode(output[q_start_idx + 1: q_end_idx],
                                             clean_up_tokenization_spaces=True)
            questions.append(question.strip())
        # tb_writer.add_scalar('num_pregenerated_questions', len(questions), num_iterations)
        if questions == []:
            continue

        # Rating as the probability of the [CLS] token, i.e. unanswerable question,
        # from the pretrained BERT model
        ratings, is_answerable, logits = rate_questions(args, BERT_tokenizer, BERT_model, context, questions)
        # tb_writer.add_scalar('num_answerable_questions', is_answerable.sum().item(), num_iterations)
        if not torch.any(is_answerable):
            continue

        # add lowest rated answerable questions
        answerable_questions = [questions[i] for i in range(len(questions)) if is_answerable[i]]
        answerable_q_ratings = ratings[is_answerable]
        answerable_start_logits = logits[0][is_answerable]
        answerable_end_logits = logits[1][is_answerable]
        num_answerable = len(answerable_questions)
        answerable_q_sorted, ids = torch.sort(answerable_q_ratings)
        for i in range(min(num_answerable, args.num_answerable_questions)):
            examples.append({
                'context': context,
                'question': answerable_questions[ids[i].item()],
                'teacher_start_logits': answerable_start_logits[ids[i]].tolist(),
                'teacher_end_logits': answerable_end_logits[ids[i]].tolist()
            })
            num_generated_answerable_questions += 1
            num_generated_examples += 1

        # add highest rated unanswerable questions
        unanswerable_questions = [questions[i] for i in range(len(questions)) if not is_answerable[i]]
        unanswerable_q_ratings = ratings[~is_answerable]
        unanswerable_start_logits = logits[0][~is_answerable]
        unanswerable_end_logits = logits[1][~is_answerable]
        num_unanswerable = len(unanswerable_questions)
        unanswerable_q_sorted, ids = torch.sort(unanswerable_q_ratings)
        for i in range(min(num_unanswerable, args.num_unanswerable_questions)):
            examples.append({
                'context': context,
                'question': unanswerable_questions[ids[i].item()],
                'teacher_start_logits': unanswerable_start_logits[ids[i]].tolist(),
                'teacher_end_logits': unanswerable_end_logits[ids[i]].tolist()
            })
            num_generated_unanswerable_questions += 1
            num_generated_examples += 1

        num_iterations += 1
        # Log numbers
        if not num_iterations % args.log_steps:
            tb_writer.add_scalar('num_generated_answerable_questions',
                                 num_generated_answerable_questions, num_iterations)
            tb_writer.add_scalar('num_generated_unanswerable_questions',
                                 num_generated_unanswerable_questions, num_iterations)

        # Save examples
        if (not num_iterations % args.save_steps) or (num_generated_examples >= args.num_examples):
            filename = os.path.join(args.output_dir, f'questions_{file_num}.json')
            with open(filename, 'w') as outfile:
                json.dump(examples, outfile)
            examples = []
            file_num += 1

            if num_generated_examples >= args.num_examples:
                break
    tb_writer.close()


def main():
    parser = get_parser()
    args = parser.parse_args()

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
        level=logging.INFO
    )
    logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)

    # Load pretrained question generation model and tokenizer
    GPT2_tokenizer = GPT2Tokenizer.from_pretrained(
        args.question_generation_model,
        do_lower_case=args.do_lower_case
    )
    GPT2_model = GPT2LMHeadModel.from_pretrained(
        args.question_generation_model
    )
    GPT2_model.prepare_inputs_for_generation = prepare_inputs_for_generation
    GPT2_model.eval()
    GPT2_model.to(args.device)

    BERT_tokenizer = BertTokenizer.from_pretrained(
        args.answering_model,
        do_lower_case=args.do_lower_case
    )
    BERT_model = BertForQuestionAnswering.from_pretrained(
        args.answering_model
    )
    BERT_model.eval()
    BERT_model.to(args.device)

    logging.info("Parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            from apex import amp
            amp.register_half_function(torch, "einsum")
            GPT2_model = amp.initialize(GPT2_model, opt_level=args.fp16_opt_level)
            BERT_model = amp.initialize(BERT_model, opt_level=args.fp16_opt_level)
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    generate(args, GPT2_tokenizer, GPT2_model, BERT_tokenizer, BERT_model)


def rate_questions(args, BERT_tokenizer, BERT_model, context, questions):
    batch_contexts = [context] * len(questions)
    inputs = BERT_tokenizer(batch_contexts, questions, padding=True, return_tensors='pt')
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    with torch.no_grad():
        start_logits, end_logits = BERT_model(**inputs)[:2]

    token_type_ids = inputs['token_type_ids'].bool()
    attention_mask = inputs['attention_mask'].bool()
    start_logits.masked_fill_(torch.logical_or(token_type_ids, ~attention_mask), -float('inf'))
    end_logits.masked_fill_(torch.logical_or(token_type_ids, ~attention_mask), -float('inf'))
    
    is_answerable = ~ torch.logical_and(
        (start_logits[:, 0].unsqueeze(-1) >= start_logits).all(dim=-1),
        (end_logits[:, 0].unsqueeze(-1) >= end_logits).all(dim=-1)
    )
    
    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)

    ratings = start_probs[:, 0] * end_probs[:, 0]

    return ratings, is_answerable, (start_logits, end_logits)


def prepare_inputs_for_generation(input_ids, past, **kwargs):
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        token_type_ids = kwargs['token_type_ids'][:, -1].unsqueeze(-1)
    else:
        token_type_ids = kwargs['token_type_ids']
    return {'input_ids': input_ids, 'past': past, 'use_cache': kwargs['use_cache'],
            'token_type_ids': token_type_ids}


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question_generation_model",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained question generation GPT2 model",
    )
    parser.add_argument(
        "--answering_model",
        default=None,
        type=str,
        required=True,
        help="Pretrained question answering BERT model",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory for generated question-context pairs.",
    )

    parser.add_argument(
        "--generated_answers_path",
        default=None,
        type=str,
        help="Path to directory with generated answer-context pairs."
    )
    parser.add_argument(
        "--max_q_length",
        default=64,
        type=int,
        help="The maximum length of generated questions."
    )
    parser.add_argument(
        "--num_pregenerated_questions",
        default=16,
        type=int,
        help="Number of questions to pregenerate for each context-answer pair"
    )
    parser.add_argument(
        "--num_answerable_questions",
        default=4,
        type=int,
        help="Maximum number of answerable question to keep."
    )
    parser.add_argument(
        "--num_unanswerable_questions",
        default=1,
        type=int,
        help="Maximum number of unanswerable question to keep."
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature used for generation.")
    parser.add_argument("--top_k", type=int, default=30, help="The k value for top-k generation.")
    parser.add_argument("--top_p", type=float, default=0.9, help="The p value for top-p (nucleus) generation.")
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--threads", default=0, type=int, help="Number of workers for the dataloader.")

    parser.add_argument("--log_steps", type=int, default=500, help="Log numbers every X update steps.")
    parser.add_argument("--num_examples", type=int, default=3000000, help="Total number of question-context pairs to generate.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save examples every updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
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

    return parser


if __name__ == '__main__':
    main()
