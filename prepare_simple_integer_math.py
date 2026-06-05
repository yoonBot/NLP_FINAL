import random
import os

random.seed(11711)

out_path = "data/simple_integer_math_train.txt"
os.makedirs("data", exist_ok=True)

examples = []
idx = 0

for _ in range(3000):
    a = random.randint(0, 100)
    b = random.randint(0, 100)
    ans = a + b
    examples.append((idx, f"What is {a} + {b}?", f"{a} + {b} = {ans}.", ans))
    idx += 1

for _ in range(3000):
    a = random.randint(0, 100)
    b = random.randint(0, 100)
    ans = a - b
    examples.append((idx, f"What is {a} - {b}?", f"{a} - {b} = {ans}.", ans))
    idx += 1

for _ in range(3000):
    a = random.randint(0, 20)
    b = random.randint(0, 20)
    ans = a * b
    examples.append((idx, f"What is {a} * {b}?", f"{a} * {b} = {ans}.", ans))
    idx += 1

for _ in range(1000):
    b = random.randint(1, 20)
    ans = random.randint(0, 20)
    a = b * ans
    examples.append((idx, f"What is {a} / {b}?", f"{a} / {b} = {ans}.", ans))
    idx += 1

random.shuffle(examples)

with open(out_path, "w", encoding="utf-8") as f:
    for new_idx, (_, question, reasoning, ans) in enumerate(examples):
        f.write(f"""{new_idx}

Question: {question}

Reasoning:
{reasoning}
#### {ans}<|endoftext|>

""")

print(f"Wrote {len(examples)} examples to {out_path}")