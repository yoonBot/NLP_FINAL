import os
import random

random.seed(42)
os.makedirs("data", exist_ok=True)

def make_example(i, a, b):
    # Ensure non-negative result
    if b > a:
        a, b = b, a

    result = a - b

    ah, at, ao = a // 100, (a // 10) % 10, a % 10
    bh, bt, bo = b // 100, (b // 10) % 10, b % 10

    lines = []

    # Ones
    borrow_tens = 0
    if ao < bo:
        ones_value = ao + 10 - bo
        borrow_tens = 1
        lines.append(
            f"Ones: {ao} - {bo} needs borrow. Borrow 1 ten, so {ao + 10} - {bo} = {ones_value}."
        )
    else:
        ones_value = ao - bo
        lines.append(
            f"Ones: {ao} - {bo} = {ones_value}."
        )

    # Tens
    at_after = at - borrow_tens
    borrow_hundreds = 0
    if at_after
