import os
import random

random.seed(42)
os.makedirs("data", exist_ok=True)

def make_example(i, quotient, divisor):
    dividend = quotient * divisor

    reasoning = f"""{divisor} goes into {dividend} exactly {quotient} times, because {quotient} * {divisor} = {dividend}.
So {dividend} / {divisor} = {quotient}."""

    return f"""{i}

Question: What is {dividend} / {divisor}?

Reasoning:
{reasoning}
#### {quotient}<|endoftext|>

"""

examples = []

for i in range(10000):
    divisor = random.randint(2, 12)
    quotient = random.randint(2, 99)
    examples.append(make_example(i, quotient, divisor))

with open("data/cot_division_train.txt", "w", encoding="utf-8") as f:
    f.write("".join(examples))

print("Wrote data/cot_division_train.txt")
print("Examples:", len(examples))
