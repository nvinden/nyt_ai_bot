from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, Features, Sequence, Value
from itertools import chain


import pandas as pd

from torch.utils.data import DataLoader
import torch

from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import random
import string
from collections import defaultdict
import json

import os

DATASET_PATH = 'data/'
PREFIX = ""
EPOCHS = 1

INCLUDED_DAYS = [0, 1, 2, 3, 4, 5, 6, 7]

MAX_INPUT_LENGTH = 16
MAX_TARGET_LENGTH = 128

BATCH_SIZE = 64

LOAD_FROM = "bert-base-uncased"

RUN_NAME = "bert-base-attpt-3-new-ds"

tokenizer = AutoTokenizer.from_pretrained(LOAD_FROM)
model = AutoModelForSequenceClassification.from_pretrained(LOAD_FROM, num_labels=2)  # Binary classification (legit vs. random)

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

class WordLegitDataset(IterableDataset):
    def __init__(self, entry_path : str, rand_ratio : float):
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
        self.word_list = [word.upper() for word in self.word_list if len(word) >= 3]
        random.shuffle(self.word_list)

        self.num_examples = len(self.word_list)
        self.stopping_condition = self.num_examples
        self.n_masks = rand_ratio

        self.num_examples = len(self.word_list)

        self.input_cache = {}
        self.target_cache = {}

        self.stopping_condition = self.num_examples * rand_ratio

        rand_list = list(range(self.num_examples + int(rand_ratio * self.num_examples)))
        random.shuffle(rand_list)

        word_strat_ratios = {
            0: 1, # Use real words
            1: 0.5, # Stitch words
            2: 3, # Generate completely fake
            3: 1, # modify word
            4: 0.5, # modify word extend
        }

        real_word_indexes = [[k] * int(v * self.num_examples) for k, v in word_strat_ratios.items()]
        unpacked_list = list(chain(*real_word_indexes))
        random.shuffle(unpacked_list)
        self.real_word_indexes = unpacked_list

        word_length_list = defaultdict(int)
        for word in self.word_list:
            word_length_list[len(word)] += 1
        word_length_list_sum = sum(word_length_list.values())
        word_length_list = {k: v / word_length_list_sum for k, v in word_length_list.items()}
        self.word_length_list = word_length_list

    def mask_word(self, word):
        n_to_randomize = random.randint(0, len(word))
        indices_to_randomize = set(random.sample(range(len(word)), n_to_randomize))
        word = ''.join([letter if i not in indices_to_randomize else '*' for i, letter in enumerate(word)])
        return word

    def randomly_generate_word(self):
        word_length = random.choices(list(self.word_length_list.keys()), k=1, weights=list(self.word_length_list.values()))[0]
        word = ''.join(random.choice(string.ascii_uppercase) for _ in range(word_length))
        return word
    
    def modify_word(self, word):
        n_to_randomize = random.randint(0, len(word) // 2)
        indices_to_randomize = set(random.sample(range(len(word)), n_to_randomize))
        word = ''.join([letter if i not in indices_to_randomize else random.choice(string.ascii_uppercase) for i, letter in enumerate(word)])
        return word
    
    def modify_word_extend(self, word):
        random_snippet = random.randint(1, 3)
        random_index = random.randint(0, len(word) - random_snippet)
        random_num_repetitions = random.randint(1, 12 // random_snippet)
        insert_word = word[random_index:random_index + random_snippet] * (random_num_repetitions + 1)
        word = word[:random_index] + insert_word + word[random_index + random_snippet:]
        return word
    
    def modify_word_stitch(self, words):
        return ''.join(words)

    def __iter__(self):
        i = 0

        while(1):
            try:
                for act_num in range(7):
                    if act_num == 0: # Use real words
                        word = self.word_list[i]
                        label = torch.tensor(1, dtype = torch.long)
                    elif act_num == 1: # Use real masked word
                        word = self.word_list[i]
                        word = self.mask_word(word)
                        label = torch.tensor(1, dtype = torch.long)
                    elif act_num == 2: # Generate completely fake w/o mask
                        word = self.randomly_generate_word()
                        label = torch.tensor(0, dtype = torch.long)
                    elif act_num == 3: # Generate completely fake w mask
                        word = self.randomly_generate_word()
                        word = self.mask_word(word)
                        label = torch.tensor(0, dtype = torch.long)
                    elif act_num == 4: # modify word
                        word = self.word_list[i % self.num_examples]
                        word = self.modify_word(word)
                        label = torch.tensor(0, dtype = torch.long)
                    elif act_num == 5: # modify word extend
                        word = self.word_list[i % self.num_examples]
                        word = self.modify_word_extend(word)
                        label = torch.tensor(0, dtype = torch.long)
                    elif act_num == 6: # Stitch words
                        word_list = []
                        while sum([len(word) for word in word_list]) < 8:
                            word_list.append(self.word_list[(i + len(word_list)) % len(self.word_list)])
                        word = self.modify_word_stitch(word_list)
                        label = torch.tensor(0, dtype = torch.long)

                    word = word.upper()

                    if word in self.input_cache:
                        tokenized_input = self.input_cache[word]
                    else:
                        tokenized_input = tokenizer(word, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_INPUT_LENGTH)
                        self.input_cache[word] = tokenized_input

                    result = {"input_ids": tokenized_input["input_ids"].squeeze(), "attention_mask": tokenized_input["attention_mask"].squeeze(), "labels": label}

                    yield result

            except Exception:
                #print("Exception with word: ", self.word_list[i % self.num_examples], flush = True)
                #print("Action number: ", self.real_word_indexes[i], flush = True)
                continue

            i += 1

            if i >= len(self.word_list):
            #if i >= 1000:
                break

    def __len__(self):
        return len(self.word_list)

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
    train_ds = WordLegitDataset(entry_path = os.path.join(DATASET_PATH, 'entries.csv'), rand_ratio=5.0)

    features = Features({
        "input_ids": Sequence(Value('int64')),
        "attention_mask": Sequence(Value('int64')),
        "labels": Value('int64'),
    })

    dataset = Dataset.from_generator(lambda: iter(train_ds), features)

    # Split the dataset into 90% training+validation and 10% testing
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    # Split the training set further into 80% training and 20% validation
    train_val_split = dataset["train"].train_test_split(test_size=0.05 + 0.05 * (0.05), seed=42)
    train_ds = train_val_split["train"]
    val_ds = train_val_split["test"]
    test_ds = dataset["test"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"results/word_to_clue/{RUN_NAME}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        logging_steps=len(train_ds) // (BATCH_SIZE * 20 * EPOCHS),
        save_steps=len(train_ds) // (BATCH_SIZE * 20 * EPOCHS),
        evaluation_strategy="steps",
        run_name='run_name',
        num_train_epochs=EPOCHS,
        #fp16=True,
        learning_rate=2.5e-5,
        remove_unused_columns=False,  # Important to avoid unnecessary warnings with Wandb
        report_to="wandb",  # Report metrics to Wandb
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