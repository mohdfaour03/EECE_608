# GPT SOL / Fable Collaboration Protocol

## Roles and authority

1. The user owns the project and resolves final scope or scientific disputes.
2. Fable leads independent review, research framing, and challenge of claims.
3. GPT SOL leads implementation, testing, reproducibility, and experiment
   operations after the scientific contract is agreed.
4. Neither model may silently overrule the other. A disagreement is recorded
   with evidence, consequences, and a recommended resolution.

## Standard workflow

1. **Intake:** read Fable's latest report or handoff completely.
2. **Verify:** inspect the cited source and reproduce important findings where
   practical.
3. **Contract:** convert the accepted finding into explicit behavior,
   invariants, schema changes, and acceptance tests.
4. **Implement:** make the smallest coherent code change while preserving user
   work and protected scientific logic.
5. **Test:** run targeted tests first, then the relevant broader suite.
6. **Review:** provide Fable a compact diff summary, evidence, residual risks,
   and any scientific decision still needed.
7. **Operate:** run experiments only after their pre-run gates pass.
8. **Record:** update the project state and decision log.

## Evidence labels

- **VERIFIED:** checked directly in source, tests, or an inspectable artifact.
- **SUPPORTED:** consistent with available evidence but not fully reproduced.
- **HYPOTHESIS:** plausible and testable; not yet evidence.
- **BLOCKED:** cannot proceed safely without user choice or external state.
- **RETRACTED:** must not be quoted as a result.

## Review and disagreement rules

- Fable findings are high-priority review input, not automatically accepted.
- GPT SOL must not change estimator semantics or privacy accounting solely to
  satisfy a test; the scientific invariant comes first.
- If the disagreement affects validity, privacy guarantees, data leakage, or
  whether an experiment must be rerun, pause that branch and ask the user.
- Implementation disagreements should be resolved with a minimal reproducer,
  test, or schema example whenever possible.
- Record both positions if evidence is inconclusive.

## Experiment release gate

Before a long GPU run, GPT SOL must be able to state:

- the exact commit or bundle hash being executed;
- which acceptance tests passed;
- the accountant and denominator used for every reported tightness metric;
- the data split contract, including HPO/validation/audit separation;
- the resume/checkpoint identity and atomicity guarantees;
- whether any result is censored, invalid, exploratory, or accepted.
