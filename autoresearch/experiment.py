"""Agent sandbox — experiment.py
Best approach so far: negative_loss scoring, budget=128 per seed, 3 seeds.
Point estimate tightness_ratio ~17%. Conservative still 0 — needs stronger attack.

Now also reports GDP-based epsilon alongside Wilson CI for comparison."""
from __future__ import annotations
import random, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare
import torch
import torch.nn.functional as F

QUERY_BUDGET = 256
SEEDS = [401, 402, 403, 404, 405]


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

    # Wilson CI-based evaluation (existing)
    result = prepare.evaluate_audit(all_m, all_n)
    result.wall_seconds = time.time() - t0

    # GDP-based evaluation (new)
    from dp_audit_tightness.privacy.gdp_estimation import estimate_epsilon_gdp
    eps_upper = result.epsilon_upper
    gdp = estimate_epsilon_gdp(all_m, all_n, delta=prepare.DELTA)

    eps_gdp_point = _mu_to_eps_point(gdp, prepare.DELTA) if gdp.mu_point > 0 else 0.0
    gdp_point_tr = eps_gdp_point / eps_upper if eps_gdp_point > 0 else 0.0
    gdp_cons_tr = gdp.epsilon_lower / eps_upper if gdp.epsilon_lower > 0 else 0.0

    return result, gdp, gdp_cons_tr, gdp_point_tr


def _mu_to_eps_point(gdp, delta):
    """Get point estimate epsilon from GDP (using mu_point, not CI)."""
    from dp_audit_tightness.privacy.gdp_estimation import _mu_to_epsilon
    return _mu_to_epsilon(gdp.mu_point, delta)


def print_comparison(result, gdp, gdp_cons_tr, gdp_point_tr):
    """Print Wilson CI vs GDP side by side."""
    print("=" * 70)
    print("WILSON CI vs GDP ESTIMATION")
    print("=" * 70)
    print(f"  epsilon_upper:          {result.epsilon_upper:.4f}")
    print(f"  samples:                {result.num_member_samples} members, "
          f"{result.num_nonmember_samples} non-members")
    print()
    print(f"  --- Wilson CI (existing) ---")
    print(f"  epsilon_lower:          {result.epsilon_lower:.6f}")
    print(f"  tightness_ratio:        {result.tightness_ratio:.4%}")
    print(f"  TPR={result.selected_tpr}  FPR={result.selected_fpr}")
    print()
    eps_point = _mu_to_eps_point(gdp, prepare.DELTA) if gdp.mu_point > 0 else 0.0
    print(f"  --- GDP via AUC (new) ---")
    print(f"  AUC:                    {gdp.auc:.6f}")
    print(f"  mu_point:               {gdp.mu_point:.6f}")
    print(f"  mu_ci_lower (95%):      {gdp.mu_ci_lower:.6f}")
    print(f"  eps_point (from mu):    {eps_point:.6f}  "
          f"(tightness: {gdp_point_tr:.4%})")
    print(f"  eps_conservative (CI):  {gdp.epsilon_lower:.6f}  "
          f"(tightness: {gdp_cons_tr:.4%})")
    if gdp.warning:
        print(f"  warning: {gdp.warning}")
    print("=" * 70)


if __name__ == "__main__":
    result, gdp, gdp_cons_tr, gdp_point_tr = run_audit()
    prepare.print_results(result)
    print()
    print_comparison(result, gdp, gdp_cons_tr, gdp_point_tr)
