"""Agent sandbox — THIS IS THE FILE THE AGENT MODIFIES.

The agent can change anything in this file to try different auditing strategies.
The goal is to maximize tightness_ratio (higher = better).

Current approach: passive audit with logit_margin scoring.
The agent should try new scoring functions, score combinations, query strategies,
calibration methods, or entirely new attack approaches.

Run: python autoresearch/experiment.py
Output: tightness_ratio: X.XXXXXX (parsed by the agent loop)
"""
from __future__ import annotations

import random
import time
import sys
from pathlib import Path

# Make sure prepare.py is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare

import torch
import torch.nn.functional as F


# ---- Knobs the agent can change ----
QUERY_BUDGET = 128  # total queries (half member, half non-member)
SEED = 401
# -------------------------------------


def compute_membership_scores(
    model, dataset, indices: list[int], device, batch_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect logits and labels for the given indices."""
    logits_list, labels_list = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            images, labels = [], []
            for idx in batch_idx:
                img, lbl = dataset[idx]
                images.append(img)
                labels.append(lbl)
            x = torch.stack(images).to(device)
            y = torch.tensor(labels, dtype=torch.long, device=device)
            logits_list.append(model(x))
            labels_list.append(y)
    return torch.cat(logits_list), torch.cat(labels_list)


def score_fn(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    """Membership score function — agent should experiment with this.

    Higher scores should indicate membership (training set).
    Current: logit_margin (gap between top-2 softmax probabilities).
    """
    probs = torch.softmax(logits, dim=1)
    top2 = probs.topk(k=2, dim=1).values
    scores = top2[:, 0] - top2[:, 1]
    return scores.cpu().tolist()


def run_audit() -> prepare.EvalResult:
    """Run one passive audit and return the evaluation result."""
    t0 = time.time()
    model, bundle, device, record = prepare.load_model_and_data()
    rng = random.Random(SEED)

    member_budget = QUERY_BUDGET // 2
    nonmember_budget = QUERY_BUDGET - member_budget

    member_indices = rng.sample(range(len(bundle.train_dataset)), member_budget)
    nonmember_indices = rng.sample(range(len(bundle.eval_dataset)), nonmember_budget)

    member_logits, member_labels = compute_membership_scores(
        model, bundle.train_dataset, member_indices, device
    )
    nonmember_logits, nonmember_labels = compute_membership_scores(
        model, bundle.eval_dataset, nonmember_indices, device
    )

    member_scores = score_fn(member_logits, member_labels)
    nonmember_scores = score_fn(nonmember_logits, nonmember_labels)

    result = prepare.evaluate_audit(member_scores, nonmember_scores)
    result.wall_seconds = time.time() - t0
    return result


if __name__ == "__main__":
    result = run_audit()
    prepare.print_results(result)
