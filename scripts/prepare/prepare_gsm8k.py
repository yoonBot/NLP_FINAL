from datasets import load_dataset

dataset = load_dataset("gsm8k", "main")

# Small subset for experimentation
train_data = dataset["train"].select(range(5000))
held_out_data = dataset["train"].select(range(5000, 5200))


def write_train_dataset(data, path):

    with open(path, "w") as f:

        for i, ex in enumerate(data):

            question = ex["question"]
            answer = ex["answer"]

            text = f"""{i}

Question: {question}

Reasoning:
{answer}

<|endoftext|>

"""

            f.write(text)


def write_held_out_dataset(data, path):

    with open(path, "w") as f:

        for i, ex in enumerate(data):

            question = ex["question"]

            text = f"""{i}

Question: {question}

Reasoning:

"""

            f.write(text)


write_train_dataset(train_data, "data/gsm8k_small_train.txt")

write_held_out_dataset(
    held_out_data,
    "data/gsm8k_small_held_out.txt"
)

print("Finished creating GSM8K reasoning datasets.")
