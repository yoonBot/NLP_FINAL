import re
import random

random.seed(42)

# Read GSM8K
with open("data/gsm8k_small_train.txt", "r", encoding="utf-8") as f:
    gsm = f.read()

# Read addition dataset
with open("data/cot_integer_addition_train.txt", "r", encoding="utf-8") as f:
    add = f.read()

# Split addition examples
examples = re.split(r"\n(?=\d+\n\nQuestion:)", add)

# Keep only 2000 addition examples
examples = random.sample(examples, 2000)

with open("data/gsm8k_plus_cot_addition_train.txt", "w", encoding="utf-8") as f:
    f.write(gsm)
    f.write("\n\n")
    f.write("\n".join(examples))

with open("data/gsm8k_plus_cot_addition_train.txt") as f:
    text = f.read()

print("Questions:", text.count("Question:"))
