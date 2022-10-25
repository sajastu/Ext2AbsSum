#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import json
import logging

import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import wandb

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock

from models.GSum_trainer import TGSumTrainer
# from models.modeling_TGSum import TGSumForConditionalGeneration
from models.sequential_TGSum.modeling_TGSum import TGSumForConditionalGeneration
from models.tokenization_TGSum import TGSumTokenizer
from models.tokenization_TGSum_extractor import TGSumTokenizerExt
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed, LEDForConditionalGeneration, LEDConfig, BartConfig, LEDTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
from utility import _combine_model_inputs

check_min_version("4.20.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_info()


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        print(name)
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

set_global_logging_level(logging.ERROR, ["transformers.models.led.modeling_led", "models.tokenization_TGSum", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])

try:
    os.makedirs('descriptions/')
except:
    pass

with open('description.txt') as fR:
    desc_text = ''
    for l in fR:
          desc_text += l


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    labeling: str = field(
        default='all',
        metadata={
            "help": "Which labeling we want to do?",
            'choices': ["dynamic", "pre-selected"]
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    ext_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    mode: str = field(
        default='train',
        metadata={"help": "Mode: between 'train' and 'test'"},
    )

    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: str = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    topic_file_path: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    # 7168
    # 6144
    max_source_length: Optional[int] = field(
        default=6144,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=150,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=3,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "parquet"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "parquet"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    global LAST_ID_FILENAME
    LAST_ID_FILENAME = f'last_id_{training_args.run_name}.txt'
    if os.path.exists(f'last_id_{training_args.run_name}.txt'):
        os.remove(f'last_id_{training_args.run_name}.txt')

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:

        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]

        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None)


    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # AutoConig
    config = LEDConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # config.gradient_checkpointing = True
    #TGSumTokenizer

    # abs_tokenizer = TGSumTokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     use_fast=model_args.use_fast_tokenizer,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    abs_tokenizer = LEDTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    #TGSumForConditionalGeneration
    # model, loading_info = TGSumForConditionalGeneration.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     use_topic=True,
    #     is_test=bool( model_args.mode == "test" ),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    model = LEDForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.config = config

    # add <sect> labels...
    # special_tokens_dict = {'additional_special_tokens': ['<sect>', '</sect>']}
    # tokenizer.add_special_tokens(special_tokens_dict)
    # intialize word embeddings...

    if model_args.mode != 'test':
        model.resize_token_embeddings(len(abs_tokenizer))
        model.led.shared.weight.requires_grad = False
        model.led.shared.weight[-1, :] = model.led.shared.weight[2, :]
        model.led.shared.weight[-2, :] = model.led.shared.weight[0, :]
        model.led.shared.weight.requires_grad = True


    model.config.num_beams = 4
    model.config.max_length = 256
    model.config.min_length = 100
    model.config.length_penalty = 1.0
    model.config.no_repeat_ngram_size = 3

    if model.config.decoder_start_token_id is None and isinstance(abs_tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(abs_tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = abs_tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = abs_tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # import pdb;pdb.set_trace()
    if (
        hasattr(model.config, "max_encoder_position_embeddings")
        and model.config.max_encoder_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if isinstance(abs_tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{abs_tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        abs_tokenizer.src_lang = data_args.lang
        abs_tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            abs_tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    # def preprocess_function(examples):
    #     # remove pairs where at least one record is None
    #     inputs, targets, src_ids, tgt_ids, topic_info_global, topic_info_section, ext_labels, section_scores, \
    #     headings, inputs_tokenized,target_tokenized =[], [], [], [], [], [], [], [], [], [], []
    #     for i in range(len(examples[text_column])):
    #         if examples[text_column][i] is not None and examples[summary_column][i] is not None:
    #             input_sents = []
    #             input_sents_tokenized = []
    #             section_heading = []
    #             for sect in examples[text_column][i]:
    #                 sent_sects = []
    #                 for sent in sect:
    #                     sent_sects.append(sent.replace(' </s>', '').replace(' <s>', '').replace(' <mask>', '') \
    #                           .replace('<s>', '').replace('</s>', '').replace('<mask>', '') \
    #                           .replace('\n', ' ').strip().lower())
    #                 input_sents.append(' <SENTTT> '.join(sent_sects.copy()))
    #
    #             for sect in examples[text_column + '_tokenized'][i]:
    #                 sent_sects = []
    #                 for sent in sect:
    #                     sent_sects.append(sent)
    #
    #                 input_sents_tokenized.append(sent_sects.copy())
    #
    #             section_heading.append([e.split(' <COMBINED> ') for e in examples['section_headings'][i]])
    #             headings.append(section_heading)
    #             inputs.append(input_sents)
    #             inputs_tokenized.append(input_sents_tokenized)
    #             src_ids.append(examples["paper_id"][i])
    #             # if examples["paper_id"][i] == "SP:bd9472600b9e7e4b407b0b2572179bc8cab7f272":
    #             #     import pdb;pdb.set_trace()
    #             topic_info_global.append(json.loads(examples["topic_info_global"][i]))
    #             # topic_info_section.append(json.loads(examples["topic_info_section"][i]))
    #             ext_labels.append(examples["ext_labels"][i])
    #             section_scores.append(examples["section_scores"][i])
    #
    #             valid_targets = [j for j, e in enumerate(examples[summary_column][i]) if len(e.strip())>0]
    #
    #             targets.extend([e.strip().lower() for j, e in enumerate(examples[summary_column][i]) if j in valid_targets])
    #             target_tokenized.append([e.strip().lower() for j, e in enumerate(examples[summary_column][i]) if j in valid_targets])
    #             tgt_ids.extend(len(valid_targets) * [examples["paper_id"][i]])
    #
    #     topic_info_tuple = {"topic_info_global": topic_info_global}
    #
    #     # ext_model_inputs = ext_tokenizer(
    #     #     inputs,
    #     #     # inputs_tokenized=inputs_tokenized,
    #     #     # target_tokenized=target_tokenized,
    #     #     max_length=data_args.max_source_length,
    #     #     padding=padding,
    #     #     truncation=True,
    #     #     doc_ids=src_ids,
    #     #     section_headings=headings,
    #     #     topic_info_tuple=topic_info_tuple,
    #     #     ext_labels=ext_labels,
    #     #     section_scores=section_scores,
    #     #     labeling=model_args.labeling,
    #     #  )
    #
    #     model_inputs = abs_tokenizer(
    #         inputs,
    #         # inputs_tokenized=inputs_tokenized,
    #         # target_tokenized=target_tokenized,
    #         topic_file_path=data_args.topic_file_path,
    #         max_length=data_args.max_source_length,
    #         padding=padding,
    #         truncation=True,
    #         doc_ids=src_ids,
    #         section_headings=headings,
    #         topic_info_tuple=topic_info_tuple,
    #         ext_labels=ext_labels,
    #         section_scores=section_scores,
    #         labeling=model_args.labeling,
    #      )
    #
    #     # Setup the tokenizer for targets
    #     with abs_tokenizer.as_target_tokenizer():
    #         labels = abs_tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True, doc_ids=['tgt-'+t for t in tgt_ids],is_target=True)
    #
    #
    #     # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    #     # padding in the loss.
    #     if padding == "max_length" and data_args.ignore_pad_token_for_loss:
    #         labels["input_ids"] = [
    #             [(l if l != abs_tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #         ]
    #
    #     model_inputs["attention_mask"] = model_inputs.attention_mask
    #
    #     # create 0 global_attention_mask lists
    #     model_inputs["global_attention_mask"] = len(model_inputs["input_ids"]) * [
    #         [0 for _ in range(len(model_inputs["attention_mask"][0]))]
    #     ]
    #
    #     # since above lists are references, the following line changes the 0 index for all samples
    #     model_inputs["global_attention_mask"][0][0] = 1
    #
    #     model_inputs = _combine_model_inputs(model_inputs, labels, tgt_ids, targets)
    #
    #     # model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(' '.join(sum(examples[text_column][i], [])))
                targets.append(examples[summary_column][i][0])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = abs_tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with abs_tokenizer.as_target_tokenizer():
            labels = abs_tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != abs_tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]


        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        global eval_dataset_raw
        eval_dataset_raw = eval_dataset

        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else abs_tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        abs_tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric_loader = load_metric("rouge")
    from rouge_score import rouge_scorer
    metrics = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]

        return preds

    def combinte_predictions_labels(preds, summaries):
        preds_all, decoded_all, paper_ids_all = [], [], []
        for p_id, summaries in summaries.items():

            for summary in summaries:
                decoded_all.append(summary)
                preds_all.append(preds[p_id])
                paper_ids_all.append(p_id)

        return paper_ids_all, preds_all, decoded_all

    def compute_metrics(eval_preds):

        class bcolors:
            HEADER = '\033[95m'
            OKBLUE = '\033[94m'
            OKCYAN = '\033[96m'
            OKGREEN = '\033[92m'
            WARNING = '\033[93m'
            FAIL = '\033[91m'
            ENDC = '\033[0m'
            BOLD = '\033[1m'
            UNDERLINE = '\033[4m'

        preds, labels, doc_ids = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = abs_tokenizer.batch_decode(preds, skip_special_tokens=True)
        # if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)


        # Some simple post-processing
        decoded_preds = postprocess_text(decoded_preds)
        # read summaries from file...
        val_dataset = eval_dataset_raw
        val_paper_ids = val_dataset['paper_id']

        val_paper_summaries = []

        for summs in val_dataset['summary']:
            val_paper_summaries.append(postprocess_text(summs))

        p_ids, decoded_preds_all, decoded_labels_all = combinte_predictions_labels(
                                                        {k: v for k, v in zip(doc_ids, decoded_preds)},
                                                        {k: v for k, v in zip(val_paper_ids, val_paper_summaries)}
        )
        before_aggregated = {}

        randomlist = [0, 2, 5, 6, 9]

        logger.info(f" Calculating element-wise predictions: {len(p_ids)}")
        print()
        counter = 0
        for p_id, pred, summ in zip(p_ids, decoded_preds_all, decoded_labels_all):

            # if counter in randomlist:



            results_f = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
            results_r = {"rouge1": 0, "rouge2":0, "rougeL": 0}

            for rouge_metric in metrics:
                scores = scorer.score(summ.strip(), pred.strip())
                try:
                    results_f[rouge_metric] = scores[rouge_metric].fmeasure
                    results_r[rouge_metric] = scores[rouge_metric].recall
                except:
                    pass


            if p_id not in before_aggregated.keys():
                before_aggregated[p_id] = {
                    "scores": [{'recall': results_r, 'fmeasure': results_f}],
                    "pred": pred,
                    "gold_summs": [summ]
                }
            else:
                # before_aggregated[p_id].append({'recall': results_r, 'fmeasure': results_f})
                before_aggregated[p_id]["scores"].append({'recall': results_r, 'fmeasure': results_f})
                before_aggregated[p_id]["gold_summs"].append(summ)

            counter += 1

        # results = metric_loader.compute(predictions=decoded_preds_all, references=decoded_labels_all, use_stemmer=True, use_aggregator=False)


        # pair results with paper_ids
        all_results = {'rouge1_f':[], 'rouge2_f': [], 'rougeL_f':[], 'rouge1_r': [], 'rouge2_r': [], 'rougeL_r': []}
        logger.info(f" Averaging against multiple summaries: {len(before_aggregated)}")
        print()
        for p_id, paper_info in before_aggregated.items():
            # num_summaries = len(paper_info["gold_summs"])
            for j, metric in enumerate(['rouge1_f', 'rouge2_f', 'rougeL_f', 'rouge1_r', 'rouge2_r', 'rougeL_r']):
                metric_type = "recall" if '_r' in metric else "fmeasure"
                real_metric = metric.replace('_f', '').replace('_r', '')
                avg_score_metric_wise = np.average([s[metric_type][real_metric] for s in paper_info["scores"]])
                all_results[metric].append(
                    avg_score_metric_wise
                )
                # add avg_score to the dict list.
                if metric_type == 'fmeasure':
                    before_aggregated[p_id][f'{real_metric}'] = avg_score_metric_wise

        # store before_aggregated to wandb
        # styling first
        entities_to_be_saved_final = {
            'paper_id': [],
            'pred': [],
            'gold_summs': [],
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
        }

        counter = -1
        for p_id, paper_info in before_aggregated.items():
            counter += 1

            entities_to_be_saved_final['paper_id'].append(p_id)

            new_summs_ent = []
            for k, v in paper_info.items():
                if k=='gold_summs':
                    for sum_idx, sum in enumerate(v):
                        new_summs_ent.append(
                            {
                                f'summary_{sum_idx}': sum,
                                'scores': [paper_info['scores'][sum_idx]['fmeasure']['rouge1'], paper_info['scores'][sum_idx]['fmeasure']['rouge2'], paper_info['scores'][sum_idx]['fmeasure']['rougeL']]
                            }
                        )

                    entities_to_be_saved_final[k].append(new_summs_ent)
                    continue
                if k=="scores":
                    continue
                else:
                    entities_to_be_saved_final[k].append(v)

            if counter in randomlist:
                print(f'{bcolors.WARNING}ID:{bcolors.ENDC}' + p_id)
                print(f'{bcolors.OKBLUE}Pred:{bcolors.ENDC}' + entities_to_be_saved_final['pred'][-1].replace("\n", " "))
                for idx, sum in enumerate(entities_to_be_saved_final['gold_summs'][-1]):
                    print(f'{bcolors.OKGREEN}Summary_{idx}:{bcolors.ENDC}' + sum[f'summary_{idx}'].replace("\n", " ") + f'{bcolors.BOLD}{sum["scores"]}{bcolors.ENDC}')
                print()
                print('-'*70)
                print()



        # if wb_logger is not None:

            ## format multiple summaries...
        new_lst_gold_sums = []
        for j, sums_info_lst in enumerate(entities_to_be_saved_final['gold_summs']):
            new_lst_gold_sum = json.dumps(sums_info_lst, indent=2)
            new_lst_gold_sums.append(new_lst_gold_sum)
        entities_to_be_saved_final['gold_summs'] = new_lst_gold_sums


        df = pd.DataFrame(entities_to_be_saved_final)



        df.to_csv(f'val_results_100_150_src6144_b5_n3.csv', index=False, encoding='utf-8-sig')

            # wb_logger.save(
            #     f'results_{RUN_NAME}_{last_id+0.5}.csv'
            # )


        # Extract a few results from ROUGE
        # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = all_results
        # mean over all results...
        result = {k: np.mean(v) * 100 for k, v in result.items()}

        prediction_lens = [np.count_nonzero(pred != abs_tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    # Initialize wandb to get full control access for logging
    global RUN_NAME
    RUN_NAME = None
    global wb_logger
    wb_logger = None
    # if training_args.report_to != "none":

        # wb_logger = wandb.init(
        #     project="huggingface",
        #     name=training_args.run_name,
        #     tags=["TGSum"],
        # )
        # RUN_NAME = training_args.run_name
    # else:
    #     wb_logger=None

    # Initialize our Trainer
    # trainer = TGSumTrainer(
    #     model=model,
    #     args=training_args,
    #     loading_info=loading_info,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=eval_dataset if training_args.do_eval else None,
    #     tokenizer=abs_tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    # )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=abs_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = abs_tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()