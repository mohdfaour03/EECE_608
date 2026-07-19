# E3 Kaggle artifact review — 2026-07-19

**Reviewer:** SOL  
**Run:** `mofaour/e3-mnist-raw-lira-saturation-v4`, Version 4  
**Status:** VERIFIED at artifact level, with provenance limitation noted below

## Retrieval and integrity

- Retrieved directly from the signed-in Kaggle Output page on 2026-07-19.
- Kaggle displayed `mnist_saturation_results_v4.zip` at 197.01 kB.
- Local archive: `codex/results/mnist_saturation_results_v4_20260717.zip`.
- SHA-256: `6ba57206ab9a36a8b9394fce5b68f6edb348b0cba6ff43d7d884a22b1c96b0ec`.
- ZIP CRC check: PASS; all seven payload files readable.
- Extracted without overwriting the older run under
  `codex/results/kaggle_e3_v4_20260717/mnist_saturation/`.

## Independent result checks

- Training: MNIST, train size 2048, eval size 2560, accuracy `0.468359375`.
- Accounting: epsilon RDP `2.242196097938645`; epsilon PLD `1.812994`;
  backend `google_dp_accounting` in every attack row.
- K ladder is exactly `{2,4,8,16,32,64,128,256,512}`.
- Every row uses audit seeds 401–405 and 512 queries per seed.
- Every attack row has `status=ok`, `validity=ok`, holdout threshold selection,
  and `epsilon_lower_conservative=0` with
  `conservative_zero_below_floor_or_no_signal` censoring.
- Persisted verdict is internally consistent: zero eligible ladder cells and
  `NO VERDICT -- fewer than 2 non-censored valid cells; cannot assess plateau`.
- AUC stays near chance (`0.5116`–`0.5275`) and score gap stays small
  (`0.0081`–`0.0224`) with no trend as K grows.
- Actual pooled samples are below the nominal 2560 for K=2, 4, and 8
  (2242, 2466, and 2554); the full 2560 is reached from K=16 onward.
- The run covers one target model/training seed (123, split seed 11). Five audit
  seeds quantify attack sampling variability, not target-model variability.

## Reproducibility comparison

An older local run, with training id dated 2026-07-07, was already present under
`codex/results/mnist_saturation/`. The July 17 Kaggle run reproduces every reported
attack metric and the verdict exactly. The summary JSON differs only in training
elapsed time (`4.145` versus `4.272` seconds); run metadata differs in timestamp,
platform paths (`/content` versus `/kaggle/working`), and training id. This is strong
deterministic-repeat evidence: an independent tensor-level comparison found the
checkpoint payloads identical as well. Retaining both artifact sets preserves provenance.

### Correction to commit `41185a3`

Commit `41185a3` is titled "Add E3 saturation v4 results from Kaggle run," but the
committed training row, config, run JSON, and checkpoint are identified as the
**2026-07-07 Colab run** (`train_codex_raw_lira_mnist_seed123_20260707T113503Z`,
paths rooted at `/content`). The archive retrieved directly from the July 17 Kaggle
Output page contains `train_codex_raw_lira_mnist_seed123_20260717T075636Z` with
`/kaggle/working` paths. Therefore `41185a3` committed the older download rather than
the July 17 archive. The attack rows and verdict are numerically identical, so this
does not change the scientific conclusion, but the artifact provenance label in that
commit is incorrect and must be corrected before the run is cited as a Kaggle rerun.

## Scientific conclusion

E3 does **not** establish auditor saturation. The admissible statement is that the
passive Raw-LiRA audit remains below the conservative detection floor at this query
budget, so E4 residual shares must retain the preregistered qualification
"residual >= partly weak-auditor." Increasing K beyond 512 is not motivated by this
flat ladder, but the diagnostics do not prove that the attack is not shadow-limited.
If another E3 run is funded, the informative intervention is a larger query/audit
budget or a stronger signal design, not merely more shadow models.

## Provenance limitation

The result ZIP contains result tables, training metadata/config, and a checkpoint,
but no code snapshot or manifest binding the output to the input bundle hash. Fable's
run log records the verified bundle SHA-256, and the Kaggle page records Version 4,
but the archive alone cannot prove that binding. Future runs should emit a manifest
with notebook version, input bundle SHA-256, code digest, configuration digest,
Kaggle run id, and output hashes.
