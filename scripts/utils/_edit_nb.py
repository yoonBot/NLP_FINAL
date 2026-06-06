import json, re

PATH = "colab_pipeline_math.ipynb"
nb = json.load(open(PATH, encoding="utf-8"))

NEW_EXPERIMENTS = """EXPERIMENTS = [
    # ── Step 1: MultiArith reasoning base (already done → skipped; provides curriculum init) ──
    {'id': 'MA_plan_numaug',
     'init': 'arith', 'recipe': 'plan_numaug', 'rung': 'multiarith',
     'epochs': 40, 'lr': 1e-5, 'patience': 8, 'batch_size': 8},

    # ── Step 2: GSM8K core 2x2 (curriculum × mixed-data) ──
    # baseline: arith-init, gsm8k only
    {'id': 'G_A1_direct',
     'init': 'arith', 'recipe': 'gsm8k', 'rung': 'gsm8k',
     'epochs': 12, 'lr': 1e-5, 'patience': 4, 'batch_size': 16, 'eval_n': 150},
    # +curriculum+mix (KEY hypothesis): MA-tuned init, gsm8k+MA mixed
    {'id': 'G_B2_curriculum_mix',
     'init': 'ckpt:MA_plan_numaug', 'recipe': 'gsm8k_plus_ma', 'rung': 'gsm8k',
     'epochs': 12, 'lr': 5e-6, 'patience': 4, 'batch_size': 16, 'eval_n': 150},
    # +curriculum only: MA-tuned init, gsm8k only
    {'id': 'G_B1_curriculum',
     'init': 'ckpt:MA_plan_numaug', 'recipe': 'gsm8k', 'rung': 'gsm8k',
     'epochs': 12, 'lr': 5e-6, 'patience': 4, 'batch_size': 16, 'eval_n': 150},
    # +mix only: arith-init, gsm8k+MA mixed
    {'id': 'G_A2_mixed',
     'init': 'arith', 'recipe': 'gsm8k_plus_ma', 'rung': 'gsm8k',
     'epochs': 12, 'lr': 1e-5, 'patience': 4, 'batch_size': 16, 'eval_n': 150},

    # ── Step 3: Extended (run if time remains) ──
    {'id': 'G_B3_curriculum_1e5',
     'init': 'ckpt:MA_plan_numaug', 'recipe': 'gsm8k_plus_ma', 'rung': 'gsm8k',
     'epochs': 12, 'lr': 1e-5, 'patience': 4, 'batch_size': 16, 'eval_n': 150},
    {'id': 'G_C1_entity_mix',
     'init': 'ckpt:MA_plan_numaug', 'recipe': 'gsm8k_plus_entity', 'rung': 'gsm8k',
     'epochs': 12, 'lr': 5e-6, 'patience': 4, 'batch_size': 16, 'eval_n': 150},
]"""

# ---- Cell 1: swap the EXPERIMENTS = [ ... ] block ----
src1 = "".join(nb["cells"][1]["source"])
# Replace from 'EXPERIMENTS = [' through its matching closing '\n]'
pattern = re.compile(r"EXPERIMENTS = \[.*?\n\]", re.DOTALL)
assert pattern.search(src1), "EXPERIMENTS block not found in cell 1"
new_src1 = pattern.sub(lambda m: NEW_EXPERIMENTS, src1, count=1)
assert new_src1.count("EXPERIMENTS = [") == 1
nb["cells"][1]["source"] = new_src1.splitlines(keepends=True)

# ---- Cell 4: update cfg construction (batch_size + eval_n from exp) ----
src4 = "".join(nb["cells"][4]["source"])
old_cfg = (
    '        "patience":   exp.get("patience", 8),\n'
    '        "batch_size": 8,\n'
    '    }'
)
new_cfg = (
    '        "patience":   exp.get("patience", 8),\n'
    '        "batch_size": exp.get("batch_size", 8),\n'
    '        "eval_n":     exp.get("eval_n"),\n'
    '    }'
)
assert old_cfg in src4, "cfg block not found in cell 4"
new_src4 = src4.replace(old_cfg, new_cfg, 1)
nb["cells"][4]["source"] = new_src4.splitlines(keepends=True)

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("notebook updated")
