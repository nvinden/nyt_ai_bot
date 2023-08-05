from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

#PREFIX = "Fill * for letters: "
PREFIX = ""

# specify the path to the saved weights
model_path = "results/word_fill/t5-base-5-epochs-lower-lr-2/checkpoint-64101"

def main():
    # load the model and
    # tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    word_list = [''.join(["*"] * i) for i in range(10, 20)]

    for word in word_list:
        line = PREFIX + word
        inputs = tokenizer(line, return_tensors="pt")

        outputs = model.generate(**inputs, num_return_sequences=1000, temperature=0.95, do_sample = True)
        print(line)
        # choose the first sequence that doesn't contain the forbidden phrase
        for output in outputs:
            decoded_output = tokenizer.decode(output, skip_special_tokens=True)
            if word.lower() not in decoded_output.lower():
                print(decoded_output)
        print()

if __name__ == "__main__":
    main()