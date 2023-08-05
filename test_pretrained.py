

def train():
    pass

if __name__ == '__main__':
    train()

# Messing around with the model

# FILL IN THE WORD
'''
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

input_txt = "E. A. [MASK]. I. N. G."
input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids']
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

outputs = model(input_ids)
predictions = outputs.logits

predicted_token_ids = torch.argmax(predictions, dim=2)

for mask_index in mask_token_index:
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_ids[0, mask_index].item())
    print("For mask at index {}: predicted token is {}".format(mask_index, predicted_token))

# Convert the ids to tokens
output_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids[0].tolist())

# Join the tokens to get the output sentence
output_sentence = "".join(output_tokens)
output_sentence = output_sentence.replace("[CLS]","").replace("[SEP]","")
print("Output sentence:", output_sentence)
'''

# MAKE A NYT CLUE TO THIS WORD

# pip install accelerate
'''
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large")

# Encode input text
input_ids = tokenizer.encode("What is a good response to a funny meme sent by your dead grandmother", return_tensors='pt')

# Generate text
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, do_sample=True, temperature=0.7)

# Decode the output
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
'''

# pip install accelerate
'''

from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

PREFIX = "Make New York Times Crossword Clue for: "

# specify the path to the saved weights
model_path = "results_mtw/checkpoint-24000"

# load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

word_list = ["ZOO", "SQUISHMELLOW", "SUNNY", "ATHOME", "LEBRONJAMES"]
#word_list = ["EDEN", "MATH", "SCIENCE", "RULE", "LEADER", "CUBE", "CAKE", "BUM", "INABOWL", "EXTREMELY", "FART", "PRETENSE", "POST", "MICHAELJFOX"]
# word_list = ["RAT", "MOUSE", "ODE", "CANADA", "TESTING", "BANANA", "RCMP"]

for word in word_list:
    line = PREFIX + word
    inputs = tokenizer(line, return_tensors="pt")

    outputs = model.generate(**inputs, num_return_sequences=10, temperature=0.7, do_sample = True)

    print(line)
    
    # choose the first sequence that doesn't contain the forbidden phrase
    for output in outputs:
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        if word.lower() not in decoded_output.lower():
            print(decoded_output)

    print()

'''

from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

#PREFIX = "Fill * for letters: "
PREFIX = ""

# specify the path to the saved weights
model_path = "results/word_to_clue/results_mtw/checkpoint-24000"

# load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

#word_list = ["AB*", "A**", "***", "**X**", "******", "CR*AM", "B**********", "******************"]
word_list = ["EDEN", "MATH", "SCIENCE", "RULE", "LEADER", "CUBE", "CAKE", "BUM", "INABOWL", "EXTREMELY", "FART", "PRETENSE", "POST", "MICHAELJFOX"]
# word_list = ["RAT", "MOUSE", "ODE", "CANADA", "TESTING", "BANANA", "RCMP"]

for word in word_list:
    line = PREFIX + word
    inputs = tokenizer(line, return_tensors="pt")


    '''
    outputs = model.generate(**inputs, num_return_sequences=10, temperature=0.95, do_sample = True)
    print(line)
    # choose the first sequence that doesn't contain the forbidden phrase
    for output in outputs:
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        if word.lower() not in decoded_output.lower():
            print(decoded_output)
    print()
    '''


    output = model.generate(inputs['input_ids'], max_length=30, num_return_sequences=1)
    decoded_output = tokenizer.decode(output[0])

    print(line)
    print(decoded_output)
    print()
