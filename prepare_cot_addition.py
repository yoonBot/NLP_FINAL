import random
import os

random.seed(11711)

out_path = "data/cot_integer_addition_train.txt"
os.makedirs("data", exist_ok=True)

def addition_cot(a, b):
    ans = a + b

    a_str = str(a).zfill(3)
    b_str = str(b).zfill(3)

    carry = 0
    lines = []

    ones_sum = int(a_str[2]) + int(b_str[2]) + carry
    ones_digit = ones_sum % 10
    carry = ones_sum // 10
    lines.append(f"Ones: {int(a_str[2])} + {int(b_str[2])} = {ones_sum}, write {ones_digit} carry {carry}.")

    tens_sum = int(a_str[1]) + int(b_str[1]) + carry
    tens_digit = tens_sum % 10
    carry = tens_sum // 10
    lines.append(f"Tens: {int(a_str[1])} + {int(b_str[1])} + carry = {tens_sum}, write {tens_digit} carry {carry}.")

    hundreds_sum = int(a_str[0]) + int(b_str[0]) + carry
    hundreds_digit = hundreds_sum % 10
    carry = hundreds_sum // 10
    lines.append(f"Hundreds: {int(a_str[0])} + {int(b_str[0])} + carry = {hundreds_sum}, write {hundreds_digit} carry {carry}.")

    if carry > 0:
        lines.append(f"Final carry is {carry}.")

    lines.append(f"So {a} + {b} = {ans}.")
    return "\n".join(lines), ans

examples = []

for i in range(10000):
    a = random.randint(0, 999)
    b = random.randint(0, 999)
    reasoning, ans = addition_cot(a, b)

    text = f"""{i}

Question: What is {a} + {b}?

Reasoning:
{reasoning}
#### {ans}<|endoftext|>

"""
    examples.append(text)

with open(out_path, "w", encoding="utf-8") as f:
    f.writelines(examples)

print(f"Wrote {len(examples)} examples to {out_path}")