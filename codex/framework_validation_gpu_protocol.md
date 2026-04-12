# GPU-Scale Framework Validation Protocol

## Purpose

This is the next-stage validation run after the CPU-scale sidecar pass.

Its job is not just to show that the framework executes. Its job is to test whether the strongest current findings remain stable when the audit support, shadow budget, dataset size, and training workload are all increased enough that CPU execution becomes impractical.

## Why This Is GPU-Scale

This run is intentionally sized so that it should only be launched on a machine with CUDA:

- full-size `mnist`
- full-size `cifar10`
- near-full `adult`
- multiple training epochs instead of `1`
- passive audit budget `2048` per seed
- `10` passive audit seeds
- Raw LiRA with `K=32` shadow models
- canary stress test with `128` canaries per seed

Even if a CPU could eventually finish, it is not the intended environment. The launcher enforces a CUDA requirement.

## Canonical GPU Matrix

### Datasets

- `mnist`
- `cifar10`
- `adult`

### Audit families

- `passive_negative_loss`
- `passive_negative_loss_matched`
- `passive_raw_lira`
- `canary_random`

### Support

- passive query budget per seed: `2048`
- passive audit seeds: `401-410`
- Raw LiRA shadow count: `K=32`
- canary audit seeds: `101-110`
- canaries per seed: `128`

## Training Scale

### MNIST

- full train / eval split
- `5` epochs
- hidden width `256`
- batch size `256`

### CIFAR-10

- full train / eval split
- `8` epochs
- CIFAR CNN
- batch size `256`

### Adult

- up to `30000` train and `10000` eval
- `5` epochs
- hidden width `128`
- batch size `512`

## Validation Goals

1. Re-check exact-PLD accounting at a larger training scale.
2. Re-check score-direction semantics under a stronger audit budget.
3. Test whether the current best provisional rows stay positive across a harder validation setting.
4. See whether the gap shrinks materially for `mnist` and `cifar10`.
5. Determine whether `adult + raw_lira` is still pathological at larger scale.

## Expected Deliverables

- GPU-scale validation summary JSON
- GPU-scale validation CSV
- GPU-scale validation checks JSON
- human-readable validation report

## Success Criteria

- framework checks still pass
- `mnist` and `cifar10` Raw LiRA remain non-pathological
- at least one strong row remains positive and stable enough to justify promotion from `provisional` to `trusted candidate`

## Failure Criteria

- exact PLD backend is not active
- score direction is missing or inconsistent
- matched baseline stops agreeing with the passive baseline
- the best provisional image-dataset rows collapse under scale
- pathological overshoot remains unresolved or spreads beyond `adult`
