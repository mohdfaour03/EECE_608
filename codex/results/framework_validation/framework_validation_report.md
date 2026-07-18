# Framework Validation Report

## Outcome

- framework execution validation: `pass with findings`
- exact-PLD accounting validation: `pass`
- attack semantics validation: `pass`
- scientific trust level: `still provisional overall`

## Counts

- training rows: `3`
- audit rows: `12`
- checks passed: `70`
- checks warned: `0`
- checks failed: `0`
- provisional rows: `4`
- exploratory rows: `6`
- invalidated rows: `1`
- not applicable rows: `1`

## What Passed

- the canonical matrix executed across all three datasets
- exact Google PLD was active for the training runs
- score direction is now explicit for every supported attack row
- the matched negative-loss baseline agrees with the canonical passive baseline
- the framework catches pathological overshoot instead of accepting it silently

## Main Findings

- `mnist + passive_raw_lira` (budget=512, seeds=5, K=16): `eps_lower_cons=0.894024`, `eps_upper=1.812994`, `trust=provisional`
- `cifar10 + passive_raw_lira` (budget=512, seeds=5, K=16): `eps_lower_cons=0.309951`, `eps_upper=2.635023`, `trust=provisional`
- `cifar10 + passive_negative_loss` (budget=512, seeds=5): `eps_lower_cons=0.118652`, `eps_upper=2.635023`, `trust=provisional`
- `cifar10 + passive_negative_loss_matched` (budget=512, seeds=5, K=16): `eps_lower_cons=0.118652`, `eps_upper=2.635023`, `trust=provisional`
- `mnist + passive_negative_loss` (budget=512, seeds=5): `eps_lower_cons=0.000000`, `eps_upper=1.812994`, `trust=exploratory`

## Remaining Problems

- `adult + passive_raw_lira` still overshoots the theoretical upper bound and remains invalidated
- most non-pathological rows are still provisional or exploratory rather than fully trusted
- canary validation is still only implemented for image datasets in this sidecar setup

## Interpretation

- the framework itself now validates as an end-to-end system
- the research story does not validate as final yet, because some rows remain tail-sensitive and one dataset-attack pair is still pathological
- the next phase should scale one canonical trustworthy line rather than expand breadth further
