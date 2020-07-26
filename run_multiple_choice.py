# coding=utf-8
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
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils_multiple_choice import MultipleChoiceDataset, Split, processors

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    name_gender_or_race: str = field(
        default=None, metadata={"help": "choices=['male', 'female', 'african', 'chinese']"}
    )
    augment: bool = field(
        default=False, metadata={"help": "Perform data augmentation on the training set"}
    )
    perturbation_num: int = field(
        default=0, metadata={"help": "How many perturbation to perform per example"}
    )
    compute_features: bool = field(
        default=False, metadata={"help": "Compute and return training features"}
    )



@dataclass
class TrainingArguments(TrainingArguments):
    n_runs: int = field(
        default=1, metadata={"help": "Number of runs"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = MultipleChoiceDataset(
                        data_dir=data_args.data_dir,
                        tokenizer=tokenizer,
                        task=data_args.task_name,
                        max_seq_length=data_args.max_seq_length,
                        overwrite_cache=data_args.overwrite_cache,
                        mode=Split.train,
                        name_gender_or_race=data_args.name_gender_or_race,
                        augment=data_args.augment,
                        perturbation_num=data_args.perturbation_num,
                    )

    if data_args.compute_features:
        features_file = os.path.join(training_args.output_dir, "features")
        torch.save(train_dataset.features, features_file)
        return train_dataset.features

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()

        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    runs = []

    for i in range(training_args.n_runs):
        set_seed(i)

        logger.info("Starting run number " + str(i))

        test_dataset = MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
            name_gender_or_race=data_args.name_gender_or_race,
        )

        predictions, label_ids, metrics = trainer.predict(test_dataset)
        predictions_file = os.path.join(training_args.output_dir, "test_predictions" + '-' + str(i))
        labels_ids_file = os.path.join(training_args.output_dir, "test_labels_id" + '-' + str(i))
        torch.save(predictions, predictions_file)
        torch.save(label_ids, labels_ids_file)

        examples_ids = []
        for input_feature in test_dataset.features:
            examples_ids.append(input_feature.example_id)

        examples_ids_file = os.path.join(training_args.output_dir, "examples_ids" + '-' + str(i))
        torch.save(examples_ids, examples_ids_file)

        if trainer.is_world_master():
            logger.info("***** Test results *****")
            for key, value in metrics.items():
                logger.info("  %s = %s", key, value)

        runs.append(metrics['eval_acc'])

    runs_file = os.path.join(training_args.output_dir, 'runs')
    torch.save(torch.tensor(runs), runs_file)

    return runs


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
