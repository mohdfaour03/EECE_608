"""GDP vs Wilson CI comparison — MNIST only, fast turnaround.

Reuses helpers from run_raw_lira_pilot.py but restricts to MNIST with a small
K ladder so it finishes in a few minutes on CPU. Reports Wilson vs GDP side
by side for both Raw LiRA and matched-negative-loss attacks.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

import run_raw_lira_pilot as pilot
from dp_audit_tightness.data.datasets import load_dataset_bundle
from dp_audit_tightness.models.io import load_model_for_inference
from dp_audit_tightness.utils.logging_utils import configure_logging


K_LADDER = [8, 16, 32]


def main() -> None:
    logger = configure_logging()
    logger.info("GDP vs Wilson comparison — MNIST only, K=%s", K_LADDER)

    config = pilot.make_train_config(
        dataset_name="mnist",
        data_dir=str(ROOT / "data" / "raw"),
        model_name="simple_mlp",
        input_dim=784,
        hidden_dim=64,
        num_classes=10,
        learning_rate=0.15,
        momentum=0.0,
        training_seed=123,
    )

    bundle = load_dataset_bundle(config.dataset, split_seed=config.run.split_seed)
    bundle = pilot.subset_bundle(bundle, train_limit=2048, eval_limit=512, seed=777)
    training_case = pilot.train_target_case("mnist", config, bundle, logger)

    eps_upper_rdp = training_case["record"].epsilon_upper_theory
    eps_upper_pld = training_case["record"].epsilon_upper_pld
    logger.info("Target trained. eps_upper_rdp=%.4f eps_upper_pld=%.4f", eps_upper_rdp, eps_upper_pld)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = load_model_for_inference(
        config.model, training_case["record"].model_artifact_path, device=device
    )

    qi_tr, qi_ev, per_seed_tr, per_seed_ev = pilot.sample_query_indices(
        train_size=len(bundle.train_dataset),
        eval_size=len(bundle.eval_dataset),
    )
    tgt_tr_losses = pilot.compute_loss_for_indices(target_model, bundle.train_dataset, qi_tr, device=device)
    tgt_ev_losses = pilot.compute_loss_for_indices(target_model, bundle.eval_dataset, qi_ev, device=device)

    t0 = time.time()
    sh_tr, sh_ev, sh_members = pilot.train_shadow_losses(
        training_case=training_case,
        dataset_name="mnist",
        query_train_indices=qi_tr,
        query_eval_indices=qi_ev,
        k_max=max(K_LADDER),
        logger=logger,
    )
    logger.info("Trained %d shadow models in %.1fs", max(K_LADDER), time.time() - t0)

    rows = []
    for k in K_LADDER:
        lira_m, lira_n = pilot.raw_lira_scores(
            query_train_indices=qi_tr,
            query_eval_indices=qi_ev,
            per_seed_train=per_seed_tr,
            per_seed_eval=per_seed_ev,
            target_train_losses=tgt_tr_losses,
            target_eval_losses=tgt_ev_losses,
            shadow_train_losses=sh_tr,
            shadow_eval_losses=sh_ev,
            shadow_member_sets=sh_members,
            k=k,
        )
        rows.append(pilot.build_result_row(
            dataset_name="mnist", attack_name="passive_raw_lira", k=k,
            training_case=training_case,
            member_scores=lira_m, nonmember_scores=lira_n,
            score_direction="lower",
        ))

        nl_m, nl_n = pilot.negative_loss_scores(
            query_train_indices=qi_tr,
            query_eval_indices=qi_ev,
            per_seed_train=per_seed_tr,
            per_seed_eval=per_seed_ev,
            target_train_losses=tgt_tr_losses,
            target_eval_losses=tgt_ev_losses,
        )
        rows.append(pilot.build_result_row(
            dataset_name="mnist", attack_name="passive_negative_loss", k=k,
            training_case=training_case,
            member_scores=nl_m, nonmember_scores=nl_n,
            score_direction="higher",
        ))

    print()
    print("=" * 100)
    print(f"GDP vs Wilson CI — MNIST (train=2048, eval=512, budget=512, 5 seeds)")
    print(f"eps_upper_rdp={eps_upper_rdp:.4f}   eps_upper_pld={eps_upper_pld:.4f}")
    print("=" * 100)
    hdr = f"{'attack':<26}{'K':>4}  {'wilson_cons':>12}{'wilson_pt':>11}{'gdp_cons':>11}{'gdp_pt':>10}  {'AUC':>6}  {'mu_pt':>7}  {'tr_wilson':>10}{'tr_gdp':>9}  {'gdp_ok':>6}"
    print(hdr)
    print("-" * 100)
    for r in rows:
        print(f"{r['attack']:<26}{r['k_shadows']:>4}  "
              f"{r['epsilon_lower_conservative']:>12.4f}{r['epsilon_lower_point']:>11.4f}"
              f"{r['epsilon_lower_gdp']:>11.4f}{r['epsilon_lower_gdp_point']:>10.4f}  "
              f"{r['gdp_auc']:>6.3f}  {r['gdp_mu_point']:>7.4f}  "
              f"{r['tightness_ratio_tighter'] or 0:>10.2%}{r['tightness_ratio_gdp']:>9.2%}  "
              f"{str(r['gdp_valid_lower_bound']):>6}")
    print("=" * 100)

    import json
    out_path = ROOT / "codex" / "results" / "gdp_comparison_mnist.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
