import os
import json
import random
import itertools
import math

from datetime import datetime

import torch
import pandas as pd
import numpy as np

NYT_DIRECTORY = "puzzles"

weekday_to_int = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
    "Other": 7
}

MAX_NUMBER_COMBINATIONS = 10

def get_all_json():
    file_names = []

    # walk through all subdirectories
    for root, dirs, files in os.walk(NYT_DIRECTORY):
        for file in files:
            # check if the file is a .json file
            if file.endswith('.json'):
                file_names.append(os.path.join(root, file))

    return file_names

def puzzle_to_pytorch(puzz_string, meta_data):
    board = [[]]
    puzzle = [[]]
    puzzle_letters = [[]]

    for letter in puzz_string:
        if letter == " ":
            board[-1].append(0)
            puzzle[-1].append(0)
            puzzle_letters[-1].append(' ')
        elif letter == "\n":
            board.append([])
            puzzle.append([])
            puzzle_letters.append([])
            
        else:
            board[-1].append(1)

            # Assign letter A to 1, B to 2, etc.
            puzzle[-1].append(ord(letter) - 64)
            puzzle_letters[-1].append(letter)

    puzzle = puzzle[:-1]
    board = board[:-1]
    puzzle_letters = puzzle_letters[:-1]

    # Converting these to tensors
    board = torch.tensor(board)
    puzzle = torch.tensor(puzzle)

    return board, puzzle, puzzle_letters

def puzzle_legal(data):
    board = [[]]
    puzzle = [[]]
    puzzle_letters = [[]]

    puzz_string = data["puzzle"]

    for letter in puzz_string:
        if letter == " ":
            board[-1].append(0)
            puzzle[-1].append(0)
            puzzle_letters[-1].append(' ')
        elif letter == "\n":
            board.append([])
            puzzle.append([])
            
        else:
            board[-1].append(1)

            # Assign letter A to 1, B to 2, etc.
            puzzle[-1].append(ord(letter) - 64)

    puzzle = puzzle[:-1]
    board = board[:-1]

    if len(board) != data["metadata"]["rows"]:
        return False
    
    for i in range(len(board)):
        if len(board[i]) != data["metadata"]["columns"]:
            return False
        
    return True

def generate_incomplete_words(data_entries):
    incomplete_words = []
    for entry in data_entries:
        word = entry["word"]

        combinations = random_combinations(list(range(len(word))), MAX_NUMBER_COMBINATIONS, len(word))

        combinations = combinations[:min(MAX_NUMBER_COMBINATIONS, len(combinations))]
        for combination in combinations:
            blanked_words = word
            
            for letter in combination:
                blanked_words = blanked_words[:letter] + "*" + blanked_words[letter + 1:]

            incomplete_words.append({"word": word, "blanked_word": blanked_words})

    return incomplete_words


def random_combinations(iterable, l, word_length):

    if 2**word_length > l:
        combinations = ()
        word_lengths = [math.comb(word_length, n) for n in range(word_length + 1)]
        while len(combinations) < l and sum(word_lengths) > 0:
            num_to_sample = random.choices(list(range(word_length + 1)), word_lengths, k=1)[0]

            combination = random.sample(iterable, num_to_sample)
            combination.sort()

            if combination not in combinations:
                combinations += (combination,)
                word_lengths[num_to_sample] -= 1

        return combinations
    else:
        combinations = []
        for r in range(word_length):
            combinations.extend(list(itertools.combinations(iterable, r)))
        return list(itertools.combinations(iterable, l))


def main():
    # Goes through every .json file under folder "puzzles" and opens them

    # get all files in the directory
    files = get_all_json()

    data_entries = []
    metadata_entries = []
    all_data = []
    pt_puzzles = []

    id_number = 0
    for json_files in files:
        # Open json as a dictionary
        with open(json_files, 'r') as f:
            data = json.load(f)

            if not puzzle_legal(data):
                print(f"{str(id_number).zfill(5)}) illegal")
                continue
            else:
                print(f"{str(id_number).zfill(5)}) legal")

            id = str(data['metadata']['date']['month']) + "-" + str(data['metadata']['date']['day']) + "-" + str(data['metadata']['date']['year'])

            # Takes day, month, and year and finds if it is a Monday, Tuesday, etc.
            # create a datetime object
            try:
                date = datetime(data['metadata']['date']['year'], data['metadata']['date']['month'], data['metadata']['date']['day'])
                weekday = date.strftime("%A")
                weekday = weekday_to_int[weekday]
            except:
                #print(data['metadata']['date']['year'], data['metadata']['date']['month'], data['metadata']['date']['day'])
                weekday = 7

            #print(data['metadata']['date']['month'], data['metadata']['date']['day'], data['metadata']['date']['year'], weekday)

            for value in data['key']['across'].values():
                value['id'] = id
                value['id_number'] = id_number
                value['weekday'] = weekday

                data_entries.append(value)

            for value in data['key']['down'].values():
                value['id'] = id
                value['id_number'] = id_number
                value['weekday'] = weekday

                data_entries.append(value)

            curr_metadata = data['metadata']
            curr_metadata['id'] = id
            curr_metadata['id_number'] = id_number
            curr_metadata['weekday'] = weekday

            curr_metadata["month"] = data['metadata']['date']['month']
            curr_metadata["day"] = data['metadata']['date']['day']
            curr_metadata["year"] = data['metadata']['date']['year']

            curr_metadata["puzzle"] = data['puzzle']

            curr_metadata.pop('date', None)

            metadata_entries.append(curr_metadata)

            all_data.append(data)

            # Adding section for the boards
            puzzle, board, puzzle_letters = puzzle_to_pytorch(data["puzzle"], data['metadata'])

            pt_puzzles.append({"id": id, "id_number": id_number, "puzzle": puzzle, "board": board, "puzzle_letters": puzzle_letters})

            # Incomplete words dataset
            #incomplete_word_batch = generate_incomplete_words(data_entries)
            #incomplete_word_batch = [{"id": id, "id_number": id_number, "word": entry["word"], "blanked_word": entry["blanked_word"]} for entry in incomplete_word_batch]
            #incomplete_words.extend(incomplete_word_batch)

            id_number += 1

    if os.path.isdir("data") == False:
        os.mkdir("data")

    if os.path.isdir("data/train") == False:
        os.mkdir("data/train")
    
    if os.path.isdir("data/test") == False:
        os.mkdir("data/test")

    if os.path.isdir("data/val") == False:
        os.mkdir("data/val")

    ttv_ids = list(range(0, len(all_data)))

    # shuffle the list
    random.shuffle(ttv_ids)

    # calculate the indices for splitting
    index1 = int(len(ttv_ids) * 0.8)
    index2 = int(len(ttv_ids) * 0.9)

    # split the list
    train_ids = ttv_ids[:index1]
    test_ids = ttv_ids[index1:index2]
    val_ids = ttv_ids[index2:]

    # Entries
    train_entries = []
    test_entries = []
    val_entries = []

    for entry in data_entries:
        if entry["id_number"] in train_ids:
            train_entries.append(entry)
        elif entry["id_number"] in test_ids:
            test_entries.append(entry)
        elif entry["id_number"] in val_ids:
            val_entries.append(entry)

    

    # Metadata
    train_metadata = []
    test_metadata = []
    val_metadata = []

    for id in train_ids:
        train_metadata.append(metadata_entries[id])

    for id in test_ids:
        test_metadata.append(metadata_entries[id])

    for id in val_ids:
        val_metadata.append(metadata_entries[id])

    # All data
    train_all_data = []
    test_all_data = []
    val_all_data = []

    for id in train_ids:
        train_all_data.append(all_data[id])

    for id in test_ids:
        test_all_data.append(all_data[id])

    for id in val_ids:
        val_all_data.append(all_data[id])

    # Puzzles
    train_puzzles = []
    test_puzzles = []
    val_puzzles = []

    for id in train_ids:
        train_puzzles.append(pt_puzzles[id])

    for id in test_ids:
        test_puzzles.append(pt_puzzles[id])

    for id in val_ids:
        val_puzzles.append(pt_puzzles[id])

    torch.save(pt_puzzles, "data/puzzles.pt")
    torch.save(train_puzzles, "data/train/puzzles.pt")
    torch.save(test_puzzles, "data/test/puzzles.pt")
    torch.save(val_puzzles, "data/val/puzzles.pt")

    # Incomplete words
    '''
    train_incomplete_words = []
    test_incomplete_words = []
    val_incomplete_words = []

    for entry in incomplete_words:
        if entry["id_number"] in train_ids:
            train_incomplete_words.append(entry)
        elif entry["id_number"] in test_ids:
            test_incomplete_words.append(entry)
        elif entry["id_number"] in val_ids:
            val_incomplete_words.append(entry)
    

    train_incomplete_words_df = pd.DataFrame(train_incomplete_words)
    test_incomplete_words_df = pd.DataFrame(test_incomplete_words)
    val_incomplete_words_df = pd.DataFrame(val_incomplete_words)

    train_incomplete_words_df.to_csv("data/train/incomplete_words.csv")
    test_incomplete_words_df.to_csv("data/test/incomplete_words.csv")
    val_incomplete_words_df.to_csv("data/val/incomplete_words.csv")
    '''

    # Save data_entroes and metadata_entries as a csv
    entries_df = pd.DataFrame(data_entries)
    train_entries_df = pd.DataFrame(train_entries)
    test_entries_df = pd.DataFrame(test_entries)
    val_entries_df = pd.DataFrame(val_entries)

    entries_df.to_csv("data/entries.csv")
    train_entries_df.to_csv("data/train/entries.csv")
    test_entries_df.to_csv("data/test/entries.csv")
    val_entries_df.to_csv("data/val/entries.csv")

    train_metadata_df = pd.DataFrame(train_metadata)
    test_metadata_df = pd.DataFrame(test_metadata)
    val_metadata_df = pd.DataFrame(val_metadata)

    train_metadata_df.to_csv("data/train/metadata.csv")
    test_metadata_df.to_csv("data/test/metadata.csv")
    val_metadata_df.to_csv("data/val/metadata.csv")

    # Save all_data as a json
    with open("data/train/all_data.json", 'w') as f:
        json.dump(train_all_data, f, indent=4)

    with open("data/test/all_data.json", 'w') as f:
        json.dump(test_all_data, f, indent=4)

    with open("data/val/all_data.json", 'w') as f:
        json.dump(val_all_data, f, indent=4)

if __name__ == '__main__':
    main()