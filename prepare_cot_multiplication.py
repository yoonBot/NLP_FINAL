import os
import random

random.seed(42)
os.makedirs("data", exist_ok=True)

def make_example(i, a, b):
    result = a * b

    tens = a // 10
    ones = a % 10

    ones_product = ones * b
    ones_digit = ones_product % 10
    carry = ones_product // 10

    tens_product = tens * b + carry

    reasoning = f"""Ones: {ones} * {b} = {ones_product}, write {ones_digit} carry {carry}.
Tens: {tens} * {b} + carry {carry} = {tens_product}.
So {a} * {b} = {result}."""

    return f"""{i}

Question: What is {a} * {b}?

Reasoning:
{reasoning}
#### {result}<|endoftext|>

"""

examples = []

for i in range(10000):
    a = random.randint(10, 99)   # 2-digit number
    b = random.randint(2, 9)     # 1-digit multiplier
    examples.append(make_example(i, a, b))

with open("data/cot_multiplication_train.txt", "w", encoding="utf-8") as f:
    f.write("".join(examples))

print("Wrote data/cot_multiplication_train.txt")
print("Examples:", len(examples))
