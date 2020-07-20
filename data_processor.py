
import os
import wget
import logging
import torch
from torch.utils.data import TensorDataset
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV2Processor


logger = logging.getLogger(__name__)


def download_squad():
    if not os.path.exists('data'):
        os.makedirs('data')
        wget.download('https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
                      out='data/train-v2.0.json')
        wget.download('https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',
                      out='data/dev-v2.0.json')


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file_train = os.path.join(
        input_dir,
        "cached_ans_gen_{}_{}_train".format(
            list(filter(None, args.model_name.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )
    cached_features_file_dev = os.path.join(
        input_dir,
        "cached_ans_gen_{}_{}_dev".format(
            list(filter(None, args.model_name.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )
    # Init features and dataset from cache if it exists
    if (os.path.exists(cached_features_file_train)
        and not args.overwrite_cache
        and not evaluate
    ):
        logger.info("Loading features from cached file: {}".format(cached_features_file_train))
        features_and_dataset = torch.load(cached_features_file_train)
        train_features, train_ds, train_examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    elif (os.path.exists(cached_features_file_dev)
          and not args.overwrite_cache
          and evaluate
    ):
        logger.info("Loading features from cached file: {}".format(cached_features_file_dev))
        features_and_dataset = torch.load(cached_features_file_dev)
        dev_features, dev_ds, dev_examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )

    # Preprocess examples into features if not already in cache
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        if not args.data_dir and not args.train_file:
            raise ImportError("Please specify --data_dir or {}".format('--train_file'))
        else:
            processor = SquadV2Processor()
            examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        train_examples = examples[: len(examples) * 9 // 10]
        dev_examples = examples[len(examples) * 9 // 10:]

        train_features = squad_convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True,
            threads=args.threads,
        )
        dev_features = squad_convert_examples_to_features(
            examples=dev_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True,
            threads=args.threads,
        )

        train_ds = feats_to_ds(train_features)
        dev_ds = feats_to_ds(dev_features)

        logger.info("Saving train features into cached file %s", cached_features_file_train)
        torch.save({"features": train_features, "dataset": train_ds, "examples": train_examples},
                   cached_features_file_train)
        logger.info("Saving dev features into cached file %s", cached_features_file_dev)
        torch.save({"features": dev_features, "dataset": dev_ds, "examples": dev_examples},
                   cached_features_file_dev)
    if evaluate:
        if output_examples:
            return dev_ds, dev_examples, dev_features
        else:
            return dev_ds
    else:
        if output_examples:
            return train_ds, train_examples, train_features
        else:
            return train_ds


def feats_to_ds(features):
    # Select only features with answers
    features = [feature for feature in features if not feature.is_impossible]

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_masks,
        all_token_type_ids,
        all_start_positions,
        all_end_positions,
        all_cls_index,
        all_p_mask,
        all_feature_index
    )

    return dataset


if __name__ == '__main__':
    download_squad()
