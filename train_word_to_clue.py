from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, Features, Sequence, Value

import pandas as pd

from torch.utils.data import DataLoader
import torch

from torch.utils.data import IterableDataset
from transformers import BertTokenizerFast
import random
import string

import os

DATASET_PATH = 'data/'
PREFIX = ""
EPOCHS = 5

INCLUDED_DAYS = [0, 1, 2, 3, 4, 5, 6, 7]

MAX_INPUT_LENGTH = 16
MAX_TARGET_LENGTH = 128

BATCH_SIZE = 32

LOAD_FROM = "t5-base"

RUN_NAME = "t5-base-attpt-2"

tokenizer = T5Tokenizer.from_pretrained(LOAD_FROM)
model = T5ForConditionalGeneration.from_pretrained(LOAD_FROM)

def get_dataset():
    train_dataset = pd.read_csv(os.path.join(DATASET_PATH, 'train/entries.csv'))
    test_dataset = pd.read_csv(os.path.join(DATASET_PATH, 'test/entries.csv'))
    val_dataset = pd.read_csv(os.path.join(DATASET_PATH, 'val/entries.csv'))

    train_dataset.dropna(inplace=True)
    test_dataset.dropna(inplace=True)
    val_dataset.dropna(inplace=True)

    train_dataset.drop(columns=['Unnamed: 0'], inplace=True)
    test_dataset.drop(columns=['Unnamed: 0'], inplace=True)
    val_dataset.drop(columns=['Unnamed: 0'], inplace=True)

    train_dataset = train_dataset[train_dataset['weekday'].isin(INCLUDED_DAYS)]
    test_dataset = test_dataset[test_dataset['weekday'].isin(INCLUDED_DAYS)]
    val_dataset = val_dataset[val_dataset['weekday'].isin(INCLUDED_DAYS)]

    train_dataset = Dataset.from_pandas(train_dataset)
    test_dataset = Dataset.from_pandas(test_dataset)
    val_dataset = Dataset.from_pandas(val_dataset)
    
    return train_dataset, test_dataset, val_dataset

class WordToClueDataset(IterableDataset):
    def __init__(self, entry_path : str, number_repetitions : int = 1):
        super().__init__()

        word_list = pd.read_csv(entry_path)
        word_list.dropna(inplace=True)

        self.word_list = word_list['word'].tolist()
        self.puzzle_list = word_list['clue'].tolist()

        self.num_examples = len(self.word_list)

        self.input_cache = {}
        self.target_cache = {}

        self.stopping_condition = self.num_examples * number_repetitions

    def __iter__(self):
        i = 0
        while(1):
            word = self.word_list[i % self.num_examples]
            puzzle = self.puzzle_list[i % self.num_examples]

            if word in self.input_cache:
                tokenized_input = self.input_cache[word]
            else:
                tokenized_input = tokenizer(word, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_INPUT_LENGTH)
                self.input_cache[word] = tokenized_input

            if puzzle in self.target_cache:
                tokenized_target = self.target_cache[puzzle]
            else:
                tokenized_target = tokenizer(puzzle, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_TARGET_LENGTH).input_ids
                self.target_cache[puzzle] = tokenized_target

            labels_example = torch.where(tokenized_target == 0, torch.tensor(-100), tokenized_target)
            result = {"input_ids": tokenized_input["input_ids"].squeeze(), "attention_mask": tokenized_input["attention_mask"].squeeze(), "labels": labels_example.squeeze()}

            yield result

            i += 1

            if i >= self.stopping_condition:
                break

    def __len__(self):
        return self.stopping_condition


def preprocess_examples(examples):
  # encode the code-docstring pairs
  words = examples['word']
  clues = examples['clue']
  
  inputs = [PREFIX + code for code in words]
  model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True)

  # encode the summaries
  labels = tokenizer(clues, max_length=MAX_TARGET_LENGTH, padding="max_length", truncation=True).input_ids

  # important: we need to replace the index of the padding tokens by -100
  # such that they are not taken into account by the CrossEntropyLoss
  labels_with_ignore_index = []
  for labels_example in labels:
    labels_example = [label if label != 0 else -100 for label in labels_example]
    labels_with_ignore_index.append(labels_example)
  
  model_inputs["labels"] = labels_with_ignore_index

  return model_inputs


def main():
    train_ds = WordToClueDataset(entry_path = os.path.join(DATASET_PATH, 'train/entries.csv'), number_repetitions=1)
    test_ds = WordToClueDataset(entry_path = os.path.join(DATASET_PATH, 'test/entries.csv'), number_repetitions=1)
    val_ds = WordToClueDataset(entry_path = os.path.join(DATASET_PATH, 'val/entries.csv'), number_repetitions=1)

    features = Features({
        "input_ids": Sequence(Value('int64')),
        "attention_mask": Sequence(Value('int64')),
        "labels": Sequence(Value('int64')),
    })

    train_ds = Dataset.from_generator(lambda: iter(train_ds), features)
    test_ds = Dataset.from_generator(lambda: iter(test_ds), features)
    val_ds = Dataset.from_generator(lambda: iter(val_ds), features)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"results/word_to_clue/{RUN_NAME}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        logging_steps=len(train_ds) // (BATCH_SIZE * 40),
        save_steps=len(train_ds) // (BATCH_SIZE * 40),
        evaluation_strategy="steps",
        run_name=RUN_NAME,
        num_train_epochs=EPOCHS,
        report_to="wandb",
        #fp16=True,
    )

    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,  # your training dataset
        eval_dataset=val_ds,  # your validation dataset
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the test dataset
    test_results = trainer.evaluate(test_ds)
    print(test_results)

if __name__ == '__main__':
    main()