import ast
import csv
import json
import torch
from pathlib import Path

ROOT = Path.cwd()

FILES = {
    "E4 evaluator": (
        "scripts/eval_crosssource_v2_E4_baseline.py",
        ["make_model", "evaluate", "check_reproduction"],
    ),
    "Training pipeline": (
        "scripts/train_crosssource_v2.py",
        ["prepare_window", "evaluate_windows"],
    ),
    "Inference helper": (
        "experiments/crosssource_v2/legacy_helpers/gt4a2_exp1_train_eval.py",
        [
            "read_manifest",
            "shape_info_for",
            "load_frame",
            "keyframe_state",
            "transition_static",
            "one_fast_step",
        ],
    ),
    "GT2 helper": (
        "experiments/crosssource_v2/legacy_helpers/gt2_fixed_clip_repair_train.py",
        ["fast_step"],
    ),
}

def inspect_functions(label, filename, selected):
    path = ROOT / filename
    print(f"\n{'='*25} {label} {'='*25}")
    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)

    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    print("Available functions:")
    print([n.name for n in functions])

    names = set(selected)

    if label == "GT2 helper":
        names.update(
            n.name for n in functions
            if any(k in n.name.lower()
                   for k in ["load", "model", "depth", "state"])
        )

    for node in functions:
        if node.name not in names:
            continue

        print(f"\n--- {node.name} (L{node.lineno}-L{node.end_lineno}) ---")
        print("\n".join(lines[node.lineno-1:node.end_lineno]))

for label, (filename, selected) in FILES.items():
    inspect_functions(label, filename, selected)

print("\n========== PREFLIGHT CYCLE LOGIC ==========")
p = ROOT / "scripts/m05_crosssource_canary_preflight_nosha.py"
lines = p.read_text().splitlines()

for start, end in [(45, 90), (195, 240)]:
    print(f"\n--- Lines {start}-{end} ---")
    print("\n".join(lines[start-1:end]))

print("\n========== E4-C BEST CHECKPOINT ==========")

ckpt_path = ROOT / (
    "outputs/crosssource_v2_E4_20261104_C_s20261103_1000step/"
    "best_checkpoint.pt"
)

assert ckpt_path.is_file(), f"Missing: {ckpt_path}"

ck = torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint:", ckpt_path)
print("Keys:", list(ck.keys()))
print("Step:", ck.get("step"))
print("Best validation:", ck.get("best_val_loss"))
print("Head type:", type(ck.get("head_state")))

if isinstance(ck.get("head_state"), dict):
    print("Head tensors:", len(ck["head_state"]))
    print("Head parameters:",
          sum(v.numel() for v in ck["head_state"].values()
              if torch.is_tensor(v)))
    print("Head tensor shapes:")
    for k, v in ck["head_state"].items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape))

print("\n========== SELECTED BONN GT ROWS ==========")

pairs = {
    "crowd2": [433, 436, 440, 437, 444],
    "crowd3": [417, 420, 424, 421, 428],
    "balloon2": [225, 228, 232, 229, 236],
}

for scene, indices in pairs.items():
    path = ROOT / (
        f"outputs/bonn_pairs_20261102_230759/"
        f"gt4a2_exp1_{scene}_pairs_used.csv"
    )

    print(f"\n--- {scene} ---")

    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        for row in rows:
            if int(row["frame_index_zero_based"]) in indices:
                print(json.dumps(row, ensure_ascii=False))

print("\n========== AUDIT COMPLETE ==========")
