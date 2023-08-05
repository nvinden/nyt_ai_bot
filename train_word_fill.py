from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, Features, Sequence, Value


import pandas as pd
import torch

from torch.utils.data import DataLoader, IterableDataset
from transformers import BertTokenizerFast
import random
import string
from collections import defaultdict
import json

import os

random.seed(42)

N_REPETITIONS = 5

DATASET_PATH = 'data/'
PREFIX = ""

INCLUDED_DAYS = [0, 1, 2, 3, 4, 5, 6, 7]

MAX_INPUT_LENGTH = 30
MAX_TARGET_LENGTH = 30
BATCH_SIZE = 32
LOAD_FROM = "t5-base"
RUN_NAME = "t5-base-better-ds"

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

class WordMaskingDataset(IterableDataset):
    def __init__(self, entry_path : str, number_masks : int = 1):
        super().__init__()

        word_list = pd.read_csv(entry_path)
        word_list.dropna(inplace=True)

        word_list = word_list['word'].tolist()

        self.input_cache = {}
        self.target_cache = {}

        #Load word list from data/word_dictionary.json
        with open('data/words_dictionary.json') as f:
            word_dictionary = json.load(f)

        word_list.extend(word_dictionary)
        word_list = list(set(word_list))
        self.word_list = word_list
        self.word_list = [word for word in self.word_list if len(word) >= 3]
        random.shuffle(self.word_list)

        self.num_examples = len(self.word_list)
        self.stopping_condition = self.num_examples
        self.n_masks = number_masks

    def mask_word(self, word):
        og_word = word
        word_list = []
        while len(word_list) < self.n_masks:
            n_to_randomize = random.randint(1, len(word))
            indices_to_randomize = set(random.sample(range(len(word)), n_to_randomize))
            word = ''.join([letter if i not in indices_to_randomize else '*' for i, letter in enumerate(og_word)])
            word_list.append(word)
            word_list = list(set(word_list))
        return word_list

    def __iter__(self):
        i = 0
        while(1):
            word = self.word_list[i % self.num_examples]
            masked_words = self.mask_word(word)

            if word in self.input_cache:
                tokenized_input = self.input_cache[word]
            else:
                tokenized_input = tokenizer(word, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_INPUT_LENGTH)
                self.input_cache[word] = tokenized_input

            for masked_word in masked_words:
                if masked_word in self.target_cache:
                    tokenized_target = self.target_cache[masked_word]
                else:
                    tokenized_target = tokenizer(word, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_TARGET_LENGTH).input_ids
                    self.target_cache[masked_word] = tokenized_target

                labels_example = torch.where(tokenized_target == 0, torch.tensor(-100), tokenized_target)
                result = {"input_ids": tokenized_input["input_ids"].squeeze(), "attention_mask": tokenized_input["attention_mask"].squeeze(), "labels": labels_example.squeeze()}

                yield result

            i += 1

            if i >= self.stopping_condition:
                break

    def __len__(self):
        return self.stopping_condition

def main():
    word_masking_dataset = WordMaskingDataset(entry_path = os.path.join(DATASET_PATH, 'train/entries.csv'), number_masks=N_REPETITIONS)

    features = Features({
        "input_ids": Sequence(Value('int64')),
        "attention_mask": Sequence(Value('int64')),
        "labels": Sequence(Value('int64')),
    })

    dataset = Dataset.from_generator(lambda: iter(word_masking_dataset), features)

    # Split the dataset into 90% training+validation and 10% testing
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    # Split the training set further into 80% training and 20% validation
    train_val_split = dataset["train"].train_test_split(test_size=0.05 + 0.05 * (0.05), seed=42)
    train_ds = train_val_split["train"]
    val_ds = train_val_split["test"]
    test_ds = dataset["test"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./results/word_fill/{RUN_NAME}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        logging_steps=len(train_ds) // (BATCH_SIZE * 10 * N_REPETITIONS),
        save_steps=len(train_ds) // (BATCH_SIZE * 5 * N_REPETITIONS),
        evaluation_strategy="steps",
        run_name='Train Word Fill',
        num_train_epochs=3,
        learning_rate=2.5e-6, 
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