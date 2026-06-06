import argparse
import os
import json
import csv
import sys

# Increase CSV field size limit just in case
csv.field_size_limit(sys.maxsize)

def get_args():
    parser = argparse.ArgumentParser(description="Prepare e-SNLI dataset from CSV files for SFT/DPO pipeline")
    parser.add_argument("--csv_dir", type=str, default="data/e-SNLI/dataset", help="Directory where e-SNLI CSV files are located")
    parser.add_argument("--train_size", type=int, default=20000, help="Number of training examples to select")
    parser.add_argument("--held_out_size", type=int, default=1000, help="Number of held-out examples to select")
    parser.add_argument("--dev_size", type=int, default=1000, help="Number of dev/eval examples to select")
    return parser.parse_args()

def load_csv_data(file_paths, limit=None):
    data = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"Warning: File {path} not found.")
            continue
        print(f"Reading {path}...")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Basic check to ensure valid row
                if not row.get("Sentence1") or not row.get("Sentence2") or not row.get("gold_label"):
                    continue
                # Some rows might have empty explanation
                explanation = row.get("Explanation_1", "").strip()
                if not explanation:
                    continue
                
                data.append({
                    "premise": row["Sentence1"].strip(),
                    "hypothesis": row["Sentence2"].strip(),
                    "explanation_1": explanation,
                    "label": row["gold_label"].strip().lower()
                })
                if limit and len(data) >= limit:
                    return data
    return data

def prepare_esnli_data():
    args = get_args()
    csv_dir = args.csv_dir

    train_paths = [
        os.path.join(csv_dir, "esnli_train_1.csv"),
        os.path.join(csv_dir, "esnli_train_2.csv")
    ]
    dev_paths = [
        os.path.join(csv_dir, "esnli_dev.csv")
    ]

    # Verify directory and files
    if not os.path.exists(csv_dir):
        print(f"Error: CSV directory '{csv_dir}' does not exist.")
        print("Please clone https://github.com/OanaMariaCamburu/e-SNLI and place")
        print("esnli_train_1.csv, esnli_train_2.csv, and esnli_dev.csv in the directory.")
        return

    print("Loading training data from CSVs...")
    # Load slightly more than train_size + held_out_size to account for boundary
    total_train_needed = args.train_size + args.held_out_size
    train_raw = load_csv_data(train_paths, limit=total_train_needed)
    
    if len(train_raw) == 0:
        print("Error: No training data could be loaded. Check if CSV files exist and are well-formatted.")
        return

    print("Loading dev data from CSV...")
    dev_raw = load_csv_data(dev_paths, limit=args.dev_size)
    if len(dev_raw) == 0:
        print("Error: No dev data could be loaded. Check if esnli_dev.csv exists.")
        return

    # Slice datasets
    train_size = min(args.train_size, len(train_raw) - args.held_out_size)
    held_out_size = args.held_out_size
    dev_size = min(args.dev_size, len(dev_raw))

    print(f"Loaded datasets: Train={train_size}, Held-out={held_out_size}, Dev={dev_size}")

    train_data = train_raw[:train_size]
    held_out_data = train_raw[train_size:train_size + held_out_size]
    dev_data = dev_raw[:dev_size]

    # Create outputs directory
    os.makedirs("data", exist_ok=True)

    # 1. Write SFT Training Dataset
    train_path = "data/esnli_small_train.txt"
    print(f"Writing SFT training dataset to {train_path}...")
    with open(train_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(train_data):
            premise = ex["premise"]
            hypothesis = ex["hypothesis"]
            explanation = ex["explanation_1"]
            label_name = ex["label"]

            text = f"""{i}

Premise: {premise}
Hypothesis: {hypothesis}

Explanation:
{explanation}
Therefore, the relationship is {label_name}.

<|endoftext|>

"""
            f.write(text)

    # 2. Write Held-out (Inference Prompt) Dataset
    held_out_path = "data/esnli_small_held_out.txt"
    print(f"Writing held-out dataset to {held_out_path}...")
    with open(held_out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(held_out_data):
            premise = ex["premise"]
            hypothesis = ex["hypothesis"]

            text = f"""{i}

Premise: {premise}
Hypothesis: {hypothesis}

Explanation:

"""
            f.write(text)

    # 3. Write Dev evaluation dataset (JSONL)
    dev_path = "data/esnli_dev.jsonl"
    print(f"Writing dev dataset to {dev_path}...")
    with open(dev_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(dev_data):
            premise = ex["premise"]
            hypothesis = ex["hypothesis"]
            explanation = ex["explanation_1"]
            label_name = ex["label"]

            record = {
                "id": i,
                "premise": premise,
                "hypothesis": hypothesis,
                "gold_explanation": explanation,
                "gold_label": label_name
            }
            f.write(json.dumps(record) + "\n")

    # 4. Write DPO Source dataset (JSONL)
    dpo_size = min(5000, train_size)
    dpo_source_path = "data/esnli_dpo_source.jsonl"
    print(f"Writing DPO source dataset to {dpo_source_path}...")
    with open(dpo_source_path, "w", encoding="utf-8") as f:
        for i in range(dpo_size):
            ex = train_data[i]
            premise = ex["premise"]
            hypothesis = ex["hypothesis"]
            explanation = ex["explanation_1"]
            label_name = ex["label"]

            record = {
                "id": i,
                "premise": premise,
                "hypothesis": hypothesis,
                "gold_explanation": explanation,
                "gold_label": label_name
            }
            f.write(json.dumps(record) + "\n")

    print("Finished creating e-SNLI datasets successfully!")

if __name__ == "__main__":
    prepare_esnli_data()
