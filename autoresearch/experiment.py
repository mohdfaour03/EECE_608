"""Agent sandbox — experiment.py
Best approach so far: negative_loss scoring, budget=128 per seed, 3 seeds.
Point estimate tightness_ratio ~17%. Conservative still 0 — needs stronger attack."""
from __future__ import annotations
import random, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare
import torch
import torch.nn.functional as F

QUERY_BUDGET = 128
SEEDS = [401, 402, 403]


def compute_membership_scores(model, dataset, indices, device, batch_size=256):
    logits_list, labels_list = [], []
    with torch.no_grad():
        for s in range(0, len(indices), batch_size):
            bi = indices[s:s+batch_size]
            imgs, lbls = zip(*(dataset[i] for i in bi))
            x = torch.stack(imgs).to(device)
            y = torch.tensor(lbls, dtype=torch.long, device=device)
            logits_list.append(model(x))
            labels_list.append(y)
    return torch.cat(logits_list), torch.cat(labels_list)


def score_fn(logits, labels):
    """Negative loss: -CE(logits, label). Members have lower loss → higher score."""
    return (-F.cross_entropy(logits, labels, reduction="none")).cpu().tolist()


def run_audit():
    t0 = time.time()
    model, bundle, device, record = prepare.load_model_and_data()

    all_m, all_n = [], []
    for seed in SEEDS:
        rng = random.Random(seed)
        mi = rng.sample(range(len(bundle.train_dataset)), QUERY_BUDGET // 2)
        ni = rng.sample(range(len(bundle.eval_dataset)), QUERY_BUDGET // 2)
        ml, mlb = compute_membership_scores(model, bundle.train_dataset, mi, device)
        nl, nlb = compute_membership_scores(model, bundle.eval_dataset, ni, device)
        all_m.extend(score_fn(ml, mlb))
        all_n.extend(score_fn(nl, nlb))

    result = prepare.evaluate_audit(all_m, all_n)
    result.wall_seconds = time.time() - t0
    return result


if __name__ == "__main__":
    result = run_audit()
    prepare.print_results(result)
