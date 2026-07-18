# Kaggle run log — 2026-07-17 (Fable, at Mohamad's direct order)

## E3 saturation K-ladder — LAUNCHED

- Notebook: kaggle.com/code/mofaour/e3-mnist-raw-lira-saturation-v4 (private), **Version 4**, GPU T4 ×2, Internet on.
- Dataset: `mofaour/dp-audit-bundle-v4` **v2** — refreshed with the 2026-07-11 `saturation_bundle_v4.zip`. VERIFIED: local sha256 = pinned `754fe0e9…a564`; the previous Kaggle copy (uploaded ~07-08, 76.16 kB vs 76.53 kB) was the stale pre-refresh build and was replaced.
- VERIFIED before leaving it to run: bundle staged, CUDA True (Tesla T4), pre-flight 1/2 "Ran 13 tests … OK → 13/13 PASS", pre-flight 2/2 fix-marker checks all PASS, shadow-model training epochs streaming. ETA ~1–1.5 h; results land in the notebook's Output tab as `mnist_saturation_results_v4.zip`.
- Governance note: D-006 puts Phase E runs last, but RUN_QUEUE lists E3 as "independent of E1; run any time", the notebook is self-gating, and Mohamad directly ordered the run (authority: user override).

## Three notebook bugs found & fixed (`notebooks/kaggle_saturation_v4.ipynb`, local edits, uncommitted)

1. **Kaggle changed dataset mount layout**: datasets now mount at `/kaggle/input/datasets/<owner>/<slug>`, not `/kaggle/input/<slug>`. Staging cell now auto-detects both (VERIFIED via interactive console: `os.walk('/kaggle/input')` → `datasets/mofaour/dp-audit-bundle-v4`). Versions 1–2 failed on this.
2. **`python -m unittest tests.test_empirical` is shadowed on Kaggle**: some installed package provides a regular `tests` package, which wins over our namespace-package `tests/` (no `__init__.py` in the bundle). Fixed by invoking the file directly (`python tests/test_empirical.py -v` — it self-inserts `src` into sys.path and has a `__main__` block). Version 3 failed on this.
3. **Final cell used `from google.colab import files`** in the "Kaggle edition" — would have crashed after the full run. Replaced with a print pointing to the Output tab.

## Follow-ups for Sol / next session

- ~~`notebooks/kaggle_decomposition_e4_v1.ipynb` needs the SAME three fixes~~ **DONE 2026-07-18 (Fable):** mount autodetect applied to both dataset slugs (`dp-audit-decomposition-v1`, `dp-audit-hpo-results`), test invocation switched to direct-file, no google.colab cell existed. E4 Kaggle notebook is launch-ready pending the `dp-audit-hpo-results` dataset upload (E1 zip from Mohamad's Drive).
- The three saturation-notebook edits are uncommitted (pending Task 0 baseline; this tree is still live per D-008a).
- When V4 finishes: pull `mnist_saturation_results_v4.zip` from the Output tab into `codex/results/` (or wherever the aggregation expects), and record the verdict (PLATEAUED / STILL CLIMBING / NO VERDICT) — it gates how E4 residual shares may be quoted.
- Failed versions 