import os
import random
from copy import deepcopy
from collections import defaultdict
import Levenshtein
from itertools import combinations
import string


import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

import dask

from datetime import datetime

import argparse
from pathlib import Path

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification

WORD_FILL_MODEL_PATH = "results/word_fill/t5-base-better-ds/checkpoint-229500"
WORD_TO_CLUE_MODEL_PATH = "results/word_to_clue/results_mtw/checkpoint-24000"
WORD_LEGIT_MODEL_PATH = "results/word_to_clue/bert-base-attpt-3-new-ds/checkpoint-55520"

NUMBER_OF_THREADS = 4

MAX_INPUT_LENGTH = 30
MAX_TARGET_LENGTH = 30

NUM_POTENTIAL_WORDS_PER_ROUND = 100
BEAM_WIDTH = 32
BEAM_MULTIPLIER = 1.4
PUZZLE_BEAM_WIDTH = 24
MAX_NUMBER_PUZZLES_ON_DECK = 100
COMPLETE_BOARD_LIST = 5
MAX_TOP_K_VALS = 500

JW_SIMILARITY_THRESHOLD = 0.70

PREFIX = ""

USE_CHATGPT = False

# Styles:
# 1. random: Completely random squares in places
# 2. past: Gets a random past board

# Fill strategies:
# 1. longest: Fill in the longest words first
# 2. remaining_letters: Fill in the sols with the most remaining letters first
# 3. impacted_letters: Fill in the sols with the most impacted letters first
BOARD_CONFIG = {
    "style": "past",
    "fill_strat": "least_likely_word",

    "verbose": False
}

class Board():
    def __init__(self, past_puzzles = None, board_string = None):
        assert BOARD_CONFIG["style"] in ["random", "past"]

        self.style = BOARD_CONFIG["style"]
        self.fill_strat = BOARD_CONFIG["fill_strat"]

        while(1):
            if board_string is None:
                if self.style == "random":
                    self.board = self.generate_random_board()
                elif self.style == "past":
                    self.board = self.generate_past_board(past_puzzles = past_puzzles)
            else:
                self.board = self.generate_board_from_text(board_string)

            self.sols = self.update_sols_from_board(self.board)

            longer = False
            for sol in self.sols.values():
                if sol["total_sol_length"] >= 10:
                    longer = True
                    break
            
            if not longer:
                break

        self.space_effects_sol = self.generate_space_effects_sol_dict()

    def generate_board_from_text(self, board_text):
        board_text = board_text.split("\n")
        board_text = [line.strip() for line in board_text]
        board_text = [[char for char in my_string if not char.isspace()] for my_string in board_text]

        return board_text

    def generate_random_board(self):
        pass

    def generate_past_board(self, past_puzzles = None):
        if past_puzzles == None:
            past_puzzles = torch.load("data/puzzles.pt")

        board = []

        rand_puzzle = random.choice(past_puzzles)

        for row in rand_puzzle['puzzle_letters']:
            curr_row = []
            for col in row:
                if col == " ":
                    curr_row.append("*")
                else:
                    curr_row.append("_")
            board.append(curr_row)

        return board
    
    def generate_space_effects_sol_dict(self):
        ses = defaultdict(list)

        for sol in self.sols.values():
            coor = [sol["row"], sol["col"]]

            for _ in range(sol["total_sol_length"]):
                coor_key = str(coor[0]) + "_" + str(coor[1])
                ses[coor_key].append(sol["sol_number"])
                if sol["direction"] == "across":
                    coor[1] += 1
                else:
                    coor[0] += 1

        return ses
                
    def update_sols_from_board(self, board):
        sol_number = 1
        sol_id = 1

        sols = {}

        if BOARD_CONFIG['verbose']: self._print_sol_no_board(board, print_with_directions=True)

        for row_num in range(len(board)):
            for col_num in range(len(board[0])):
                across_legal, down_legal = self._is_legal_sol_spot(board, row_num, col_num, return_direction=True)
                for legality, direction in zip([across_legal, down_legal], ["across", "down"]):
                    if not legality:
                        continue

                    sol_length = self._get_length_of_sol(board, row_num, col_num, direction)

                    coor = [row_num, col_num]
                    sol = ""
                    for _ in range(sol_length):
                        sol += board[coor[0]][coor[1]]
                        if direction == "across":
                            coor[1] += 1
                        else:
                            coor[0] += 1

                    sol_entry = {
                        "sol_number": sol_number,
                        "sol_id": sol_id,
                        "row": row_num,
                        "col": col_num,
                        "direction": direction,
                        "total_sol_length": sol_length,
                        "total_unfilled_letters": sol.count("_"),
                        "sol": sol
                    }

                    sols[sol_number] = sol_entry

                    sol_number += 1

                if across_legal and down_legal:
                    sol_id += 1

        return sols

    def _get_length_of_sol(self, board, row_num, col_num, direction):
        assert direction in ["across", "down"]

        direction_legality = self._is_legal_sol_spot(board, row_num, col_num, return_direction=True)

        assert direction == "across" and direction_legality[0] == True or direction == "down" and direction_legality[1] == True

        if direction == "across":
            additive_dir = [0, 1]
        else:
            additive_dir = [1, 0]

        count = 1
        coor = [row_num + additive_dir[0], col_num + additive_dir[1]]
        while coor[0] < len(board) and coor[1] < len(board[0]) and board[coor[0]][coor[1]] != "*":
            coor[0] += additive_dir[0]
            coor[1] += additive_dir[1]
            count += 1

        return count
    
    def _print_sol_no_board(self, board, print_with_directions = False):
        sol_number = 1

        for row_num in range(len(board)):
            for col_num in range(len(board[0])):
                curr_char = board[row_num][col_num]
                across_legal, down_legal = self._is_legal_sol_spot(board, row_num, col_num, return_direction=True)
                across_letter = "a" if across_legal else " "
                down_letter = "d" if down_legal else " "
                if self._is_legal_sol_spot(board, row_num, col_num):
                    
                    if print_with_directions:
                        print(str(sol_number).zfill(2) + across_letter + down_letter, end=" ")
                    else:
                        print(str(sol_number).zfill(2), end=" ")
                    sol_number += 1
                elif curr_char == "*":
                    if print_with_directions:
                        print("**  ", end=" ")
                    else:
                        print("**", end=" ")
                else:
                    if print_with_directions:
                        print("__  ", end=" ")
                    else:
                        print("__", end=" ")
            print()

        print()

    def get_next_move(self, model = None, tokenizer = None, word_cache = None):
        sorted_sols = deepcopy(list(self.sols.values()))

        if self.fill_strat == "remaining_letters":
            sorted_sols = sorted(sorted_sols, key=lambda x: x["total_unfilled_letters"], reverse=True)
        elif self.fill_strat == "sol_length":
            sorted_sols = sorted(sorted_sols, key=lambda x: x["total_sol_length"], reverse=True)
        elif self.fill_strat == "impacted_letters":
            sorted_sols = sorted(sorted_sols, key=lambda x: x["total_sol_length"] - x["total_unfilled_letters"], reverse=True)
        elif self.fill_strat == "random":
            random.shuffle(sorted_sols)
        elif self.fill_strat == "random_by_remaining_letter":
            entry_with_remain_letters = [entry["total_unfilled_letters"] for entry in sorted_sols]
            sorted_sols = random.choices(sorted_sols, weights=entry_with_remain_letters, k=len(sorted_sols))
        elif self.fill_strat == "least_likely_word":
            assert model is not None, "Must provide model to use least_likely_word fill_strat"
            assert word_cache is not None, "Must provide word_cache to use least_likely_word fill_strat"
            board_scores, word_cache = get_board_score(self, model, tokenizer, word_cache, return_all_word_scores=True)
            data_with_board = list(zip(sorted_sols, board_scores))
            sorted_sols = sorted(data_with_board, key=lambda x: x[1], reverse=False)
            sorted_sols = [entry[0] for entry in sorted_sols]
        elif self.fill_strat == "hybrid_longest_letters_remaining":
            number_of_complete_words = sum([1 for entry in sorted_sols if entry["total_unfilled_letters"] == 0])
            if number_of_complete_words <= 6:
                sorted_sols = sorted(sorted_sols, key=lambda x: x["total_sol_length"], reverse=True)
            else:
                sorted_sols = sorted(sorted_sols, key=lambda x: x["total_sol_length"] - x["total_unfilled_letters"], reverse=False)

        for sol in sorted_sols:
            if sol["total_unfilled_letters"] == 0:
                continue

            return sol
        
        return None

    def _is_legal_sol_spot(self, board, row_num, col_num, return_direction = False):
        a_legal = False
        d_legal = False

        if col_num < 0 or col_num >= len(board[0]) or board[row_num][col_num] == "*":
            a_legal = False
        elif col_num == 0 or board[row_num][col_num - 1] == "*":
            a_legal = True

        if row_num < 0 or row_num >= len(board) or board[row_num][col_num] == "*":
            d_legal = False
        elif row_num == 0 or board[row_num - 1][col_num] == "*":
            d_legal = True
        
        if return_direction:
            return [a_legal, d_legal]
        else: return a_legal or d_legal

    def make_move(self, sol_word, next_move):
        # Changing the board
        loc = [next_move["row"], next_move["col"]]
        additive_dir = [0, 1] if next_move["direction"] == "across" else [1, 0]

        for letter in sol_word:
            curr_char = self.board[loc[0]][loc[1]]
            assert curr_char == "_" or curr_char == letter

            self.board[loc[0]][loc[1]] = letter
            loc[0] += additive_dir[0]
            loc[1] += additive_dir[1]

        id_no = next_move["sol_number"]
        self.sols = self.update_sols(id_no, sol_word)

        return deepcopy(self)
    
    def update_all_sols(self):
        for sol_id in self.sols.keys():
            self.sols = self.update_sols(sol_id, self.sols[sol_id]["sol"])

        return self.sols

    def update_sols(self, id_no, sol_word):
        # Changin the solution entry for the new filled in move
        sols = deepcopy(self.sols)

        sols[id_no]["total_unfilled_letters"] = 0
        sols[id_no]["sol"] = list(sol_word)
        
        start_coor = [sols[id_no]["row"], sols[id_no]["col"]]
        for i in range(len(sol_word)):
            curr_row, curr_col = [start_coor[0], start_coor[1] + i] if sols[id_no]["direction"] == "across" else [start_coor[0] + i, start_coor[1]]
            curr_key = str(curr_row) + "_" + str(curr_col)

            effected_sols = self.space_effects_sol[curr_key]

            for sol_id in effected_sols:
                if sol_id == id_no:
                    continue

                sols[sol_id] = self.update_one_sol(sols, sol_id)

        return deepcopy(sols)

    def update_one_sol(self, sols, sol_id):
        sols = deepcopy(sols)

        sol = sols[sol_id]

        row_num = sol["row"]
        col_num = sol["col"]

        if sol["direction"] == "across":
            additive_dir = [0, 1]
        else:
            additive_dir = [1, 0]

        coor = [row_num, col_num]
        new_word = ""
        while coor[0] < len(self.board) and coor[1] < len(self.board[0]) and self.board[coor[0]][coor[1]] != "*":
            new_word = new_word + self.board[coor[0]][coor[1]]
            coor[0] += additive_dir[0]
            coor[1] += additive_dir[1]

        sol["total_unfilled_letters"] = new_word.count("_")
        sol["sol"] = list(new_word)

        return sol

    def __str__(self) -> str:
        out = ""
        for board_row in self.board:
            out = out + " ".join(board_row) + "\n"

        return out

def get_next_best_moves(board, next_move, fill_model, fill_tokenizer, legal_tokens, legal_tokenizer, legal_model):
    if next_move is None:
        return []

    input_word = next_move["sol"]
    input_word = "".join(input_word).replace("_", "*")

    line = PREFIX + input_word
    inputs = fill_tokenizer(line, return_tensors="pt")

    complete_words = []
    already_used_words = set([''.join(sol["sol"]) for sol in board.sols.values() if sol["total_unfilled_letters"] == 0])

    # Generate initial output
    output = torch.tensor([[0, 3]], dtype=torch.long)
    curr_input = inputs['input_ids']

    # Illegal actions mask
    illegal_actions_mask = torch.ones((32128), dtype=torch.bool)
    illegal_actions_mask[legal_tokens] = False

    fringe = [{"path": deepcopy(output.squeeze()), "path_list": deepcopy(output.squeeze().tolist()), "prob": 0, "past_word": "123456789"} for _ in range(BEAM_WIDTH)]
    round = 1
    while len(fringe) > 0:
        #print("Round: ", round)
        if round > 1:
            output = torch.tensor([fringe[i]["path_list"] for i in range(len(fringe))], dtype=torch.long)

        if len(output) == 0:
            break

        next_token_logits = fill_model(curr_input.repeat(output.shape[0], 1), decoder_input_ids=output).logits
        next_token_logits = next_token_logits[:, -1, :]

        next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
        if round > 1: # Ignores the first round
            to_add = torch.tensor([fringe[i]["prob"] for i in range(next_token_log_probs.size(0))])
            next_token_log_probs = next_token_log_probs + to_add.unsqueeze(-1)
            del to_add

        # Masking out illegal tokens
        next_token_log_probs[:, illegal_actions_mask] = -float("inf")

        topk_values, topk_indices_flat = torch.topk(next_token_log_probs.reshape(-1), min(next_token_log_probs.shape[0] * next_token_log_probs.shape[1], MAX_TOP_K_VALS), dim=-1)
        rows, cols = topk_indices_flat // next_token_log_probs.size(1), topk_indices_flat % next_token_log_probs.size(1)

        del next_token_logits
        del topk_indices_flat

        row = rows[:len(legal_tokens) * next_token_log_probs.shape[0]]
        col = cols[:len(legal_tokens) * next_token_log_probs.shape[0]]
        topk_values = topk_values[:len(legal_tokens) * next_token_log_probs.shape[0]]

        new_fringe = []
        for row, col, log_prob in zip(rows, cols, topk_values):
            past_fringe = fringe[row] if round > 1 else fringe[0]

            token = torch.tensor([col])
            temp_output = torch.cat([past_fringe["path"], token], dim=-1)

            new_decoded_output = fill_tokenizer.decode(temp_output, skip_special_tokens=True)
            #print(new_decoded_output)

            if new_decoded_output == past_fringe["past_word"] or len(new_decoded_output) == 0:
                continue

            if check_part_word_legality(input_word, new_decoded_output, already_used_words):
                if check_word_completed(new_decoded_output, len(input_word), complete_words):
                    complete_words.append((new_decoded_output, log_prob.item()))
                    continue

                if not check_word_can_fringe(new_decoded_output, new_fringe):
                    continue

                new_fringe_entry = {"path": temp_output, "path_list": temp_output.squeeze().tolist(), "prob": log_prob.item(), "past_word": new_decoded_output}
                new_fringe.append(new_fringe_entry)

            if len(new_fringe) == int(BEAM_WIDTH * BEAM_MULTIPLIER):
                break

        fringe = new_fringe
        round += 1

        #fringe = sort_fringe(fringe, legal_tokenizer, legal_model, len(input_word))
        fringe = random.choices(fringe, k=min(BEAM_WIDTH, len(fringe)))

        #print("Round: ", round - 1, flush = True)
        #for ent in fringe:
        #    print(ent["past_word"], ent["prob"], flush = True)
        #print("", flush = True)

        #print()

        #if len(complete_words) >= NUM_POTENTIAL_WORDS_PER_ROUND:
        #    break

    decoded_outputs = sorted(complete_words, key=lambda x: x[1], reverse=True)[:NUM_POTENTIAL_WORDS_PER_ROUND]

    # No solutions found: Generate random word solutions
    if len(decoded_outputs) == 0:
        new_word = ''.join([random.choice(string.ascii_uppercase) if letter == "*" else letter for letter in input_word])
        return [(new_word, -100.0), ]

    return decoded_outputs

def sort_fringe(fringe, legal_tokenizer, legal_model, word_length):
    if len(fringe) == 0:
        return fringe
    
    new_fringe = sorted(fringe, key=lambda x: generate_legit_predictions(legal_model, legal_tokenizer, word_list = x["past_word"] + "*" * (word_length - len(x["past_word"])), log_probabilties=False), reverse=True)

    return new_fringe[:BEAM_WIDTH]

def check_part_word_legality(input_word, decoded_output, already_used_words):
    if len(decoded_output) > len(input_word):
        return False
    
    for letter in decoded_output:
        # Check if letter is capitalized
        if not letter.isupper():
            return False
        
        if letter == "*":
            return False
    
    for i in range(len(decoded_output)):
        if decoded_output[i] != input_word[i] and input_word[i] != "*":
            return False
        if not (decoded_output[i].isupper() or decoded_output[i] == "*"):
            return False
        
    for au_word in already_used_words:
        if decoded_output in au_word:
            return False
            
    return True

def check_word_can_fringe(decoded_output, fringe): 
    return True
        
def check_word_completed(decoded_output, input_word_len, completed_words):
    if len(decoded_output) != input_word_len:
        return False
    
    # Check to see if JW is too close for each completed word
    for completed_word in completed_words:
        jw_score = Levenshtein.jaro_winkler(decoded_output, completed_word[0])
        if jw_score > 0.85:
            return False
    
    return True

def generate_legit_predictions(model, tokenizer, word_list, log_probabilties=True):
    inputs = tokenizer(word_list, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits

    if log_probabilties == True:
        probabilities = torch.log_softmax(logits, dim=-1)
    else:
        probabilities = torch.softmax(logits, dim=-1)

    return probabilities[:, 1]

def get_optimal_next_boards(word_legit_cache, boards_with_percentages, next_moves, sol_list, legit_model, legit_tokenizer):
    board_score_list = []

    move_list = []
    # Flatten the board list from 2d to 1d
    for i in range(len(sol_list)):
        for j in range(len(sol_list[i])):
            if len(boards_with_percentages[i]) == 0:
                continue

            #print(f"i: {i}, j: {j}, len(boards_with_percentages): {len(boards_with_percentages)}, len(boards_with_percentages[i]): {len(boards_with_percentages[i])}")
            move_list.append((deepcopy(boards_with_percentages[i][0]), next_moves[i], sol_list[i][j], i, boards_with_percentages[i][1]))

    for board, next_move, sol, board_idx, past_log_pred in move_list:
        next_board = board.make_move(sol[0], next_move)
        
        board_score, word_legit_cache = get_board_score(next_board, legit_model, legit_tokenizer, word_legit_cache)

        board_score_list.append((next_board, board_score))

    board_score_list = sorted(board_score_list, key=lambda x: x[1], reverse=True)

    #board_simularity_scores = []

    #for i, (board, _) in enumerate(board_score_list):
    #    curr_board_simularity_scores = []
    #    for j, (board2, _) in enumerate(board_score_list):
    #        if i == j:
    #            continue

    #        jw_similarity = calculate_board_jw_sim(board, board2)
    #        curr_board_simularity_scores.append(jw_similarity)
    #    board_simularity_scores.append(sum(curr_board_simularity_scores) / len(curr_board_simularity_scores))

    #out_board_list = [(board_score_list[i][0], board_score_list[i][1] + board_simularity_scores[i]) for i in range(len(board_score_list))]
    out_board_list = [(board_score_list[i][0], board_score_list[i][1]) for i in range(len(board_score_list))]
    out_board_list = sorted(out_board_list, key=lambda x: x[1], reverse=True)

    return out_board_list, word_legit_cache

def get_board_score(board, legit_model, legit_tokenizer, word_legit_cache, add_min_score = False, return_all_word_scores = False):
    sol_vals = [''.join(sol_val['sol']).replace("_", "*") for sol_val in board.sols.values()]
    sol_vals_uncached = [sol_val for sol_val in sol_vals if sol_val not in word_legit_cache]

    word_scores = []

    # Calculating percentages of unknown words and adding them to the cache
    if len(sol_vals_uncached) > 0:
        unique_sol_val_list = list(sol_vals_uncached)
        legit_predictions = generate_legit_predictions(legit_model, legit_tokenizer, unique_sol_val_list, log_probabilties=False)
        for sol_val_curr, legit_prediction in zip(unique_sol_val_list, legit_predictions):
            word_legit_cache[sol_val_curr] = legit_prediction.item()

    for sol_val in sol_vals:
        word_scores.append(np.log(max(word_legit_cache[sol_val], 0.00000001)))

    master_percent = sum(word_scores) / len(board.sols)

    if add_min_score == True:
        complete_words_indexes = [i for i, sol_val in enumerate(board.sols.values()) if sol_val['sol'].count("_") == 0]
        if len(complete_words_indexes) > 0:
            complete_word_scores = [word_scores[i] for i in complete_words_indexes]
            master_percent += min(complete_word_scores)
        else:
            master_percent += 0.1

    if return_all_word_scores == True:
        return word_scores, word_legit_cache

    return master_percent, word_legit_cache

def calculate_board_jw_sim(board1, board2):
    completed_sols_1 = [''.join(sol_val['sol']) for sol_val in board1.sols.values() if sol_val['sol'].count("_") == 0 and len(sol_val['sol']) > 5]
    completed_sols_2 = [''.join(sol_val['sol']) for sol_val in board2.sols.values() if sol_val['sol'].count("_") == 0 and len(sol_val['sol']) > 5]

    if len(completed_sols_1) == 0 or len(completed_sols_2) == 0:
        return 0.0
    
    jw_score_total = list()
    count = 0
    
    for word1 in completed_sols_1:
        for word2 in completed_sols_2:
            jw_score = Levenshtein.jaro_winkler(word1, word2)
            jw_score_total.append(jw_score)
            count += 1

    return sum(jw_score_total) / count

def get_all_combinations(input_list):
    all_combinations = []
    for r in range(1, len(input_list) + 1):
        all_combinations.extend(list(combinations(input_list, r)))
    return all_combinations

def generate_legal_vocab(fill_tokenizer, use_saved=False):
    save_location = "data/word_legit_cache.pt"

    if use_saved == True and os.path.exists(save_location):
        word_legit_cache = torch.load(save_location)
        return word_legit_cache
    
    vocab = fill_tokenizer.get_vocab()
    legal_vocab = defaultdict(list)

    legal_tokens = []
    
    for word, token_no in vocab.items():
        og_word = word

        if len(word) == 0:
            continue

        if word[0] == 0:
            is_cont = True
        else:
            is_cont = False
            word = word[1:]

        if word.isupper() == False:
            continue

        if word.isalpha() == False:
            continue

        legal_tokens.append(token_no)

        #all_star_combinations = get_all_combinations(range(len(word)))

        #print(og_word)

        #for combination in all_star_combinations:
        #    new_word = ''.join([letter if i not in combination else "*" for i, letter in enumerate(word)])
        #    legal_vocab[new_word].append(token_no)

        #legal_vocab[word].append(token_no)


    return np.array(legal_tokens)

def parallel_get_next_best_moves(boards, next_moves, fill_model, fill_tokenizer, legal_tokens, legit_tokenizer, legit_model):
    num_threads_per_chunk = NUMBER_OF_THREADS
    num_chunks = math.ceil(len(boards) / num_threads_per_chunk)

    results = []

    for chunk in range(num_chunks):
        start_idx = chunk * num_threads_per_chunk
        end_idx = min((chunk + 1) * num_threads_per_chunk, len(boards))

        lazy_results = [
            dask.delayed(get_next_best_moves)(boards[i], next_moves[i], fill_model, fill_tokenizer, legal_tokens, legit_tokenizer, legit_model)
            for i in range(start_idx, end_idx)
        ]

        dask_out = dask.compute(*lazy_results, scheduler='processes')
        results.extend(dask_out)

    return results

def save_board(board, output_path):

    if os.path.exists(output_path) == False:
        os.mkdir(output_path)

    board_text = str(board)
    board_key = board_text.split("\n")[0].replace(" ", "")
    random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    with open(os.path.join(output_path, f"puzzle_{board_key}_{random_string}.txt"), "w") as f:
        f.write(board_text)

def get_unique_boards(board_list):
    seen = set()
    no_repeats = []
    for board in board_list:            
        key = str(board[0])
        if key not in seen:
            no_repeats.append(board)
            seen.add(key)

    return no_repeats

def main(num_puzzles, output_path):
    past_puzzles = torch.load("data/puzzles.pt")

    fill_tokenizer = T5Tokenizer.from_pretrained(WORD_FILL_MODEL_PATH)
    fill_model = T5ForConditionalGeneration.from_pretrained(WORD_FILL_MODEL_PATH)

    legit_tokenizer = AutoTokenizer.from_pretrained(WORD_LEGIT_MODEL_PATH)
    legit_model = AutoModelForSequenceClassification.from_pretrained(WORD_LEGIT_MODEL_PATH, num_labels=2)  # Binary classification (legit vs. random)

    legal_tokens = generate_legal_vocab(fill_tokenizer, use_saved=True)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_path, time_str)

    word_legit_cache = {}

    for batch_no in range(num_puzzles):
        completed_boards = 0

        boards_with_score = [[Board(past_puzzles = past_puzzles), _] for _ in range(PUZZLE_BEAM_WIDTH)]
        boards_with_score = [[board[0], get_board_score(board[0], legit_model, legit_tokenizer, word_legit_cache)[0]] for board in boards_with_score]

        round = 1
        while len(boards_with_score) > 0:
        #try:
            curr_board_batch = boards_with_score[:PUZZLE_BEAM_WIDTH]
            boards_with_score = boards_with_score[PUZZLE_BEAM_WIDTH:]
            boards = [board for board, _ in curr_board_batch]

            next_moves = [board.get_next_move(model = legit_model, tokenizer = legit_tokenizer, word_cache = word_legit_cache) for board in boards]

            sol_list = parallel_get_next_best_moves(boards, next_moves, fill_model, fill_tokenizer, legal_tokens, legit_tokenizer, legit_model)

            best_boards, word_legit_cache = get_optimal_next_boards(word_legit_cache, curr_board_batch, next_moves, sol_list, legit_model, legit_tokenizer)

            # Saving models if they are complete
            best_boards_temp = []
            for board in best_boards:
                for sol in board[0].sols.values():
                    if "_" in sol["sol"]:
                        best_boards_temp.append(board)
                        break
                else:
                    save_board(board[0], output_path)
                    completed_boards += 1
            best_boards = best_boards_temp
                
            if completed_boards >= COMPLETE_BOARD_LIST:
                boards_with_score = []
                break


            boards_with_score = best_boards

            boards_with_score = get_unique_boards(boards_with_score)

            boards_with_score = sorted(boards_with_score, key = lambda x: x[1], reverse=True)
            boards_with_score = boards_with_score[:MAX_NUMBER_PUZZLES_ON_DECK]

            # Clear the print screen
            print("\033c", end="")

            print("Round: ", round)
            for i, (board, score) in enumerate(boards_with_score):
                if i < 3:
                    print("Board: ", i + 1)
                    print(board)
                    n_complete_solutions = len([1 for sol in board.sols.values() if "_" not in sol["sol"]])
                    print("Score: ", score)
                    print("Number of complete solutions: ", n_complete_solutions)
                    print()
                else:
                    break

            for row in range(15):
                boards_in_row = [i for i in range(len(boards_with_score)) if i % 15 == row]
                print("|", end="")
                for board_num in boards_in_row:
                    board = boards_with_score[board_num][0]
                    score = boards_with_score[board_num][1]
                    n_complete_solutions = len([1 for sol in board.sols.values() if "_" not in sol["sol"]])

                    print(f"{str(board_num + 1).zfill(2)} {score:.2f} {str(n_complete_solutions).zfill(3)} |", end="")

                print()

                if len(boards_with_score) <= row + 1:
                    break


            round += 1

            if round > 100:
                break
                #except Exception:
                #    for board in boards_with_score:
                #        save_board(board[0], output_path)

if __name__ == "__main__":
    # Command line parameters:
    # 1. Number of puzzles to generate
    # 2. Output path

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-puzzles", help="The number of puzzles to generate", type=int, default=500)
    parser.add_argument("--output-path", help="The path where the puzzles will be saved", type=str, default="./ai_puzzles")
    args = parser.parse_args()

    num_puzzles = args.num_puzzles
    output_path = args.output_path

    Path(output_path).mkdir(parents=True, exist_ok=True)

    main(**vars(args))
