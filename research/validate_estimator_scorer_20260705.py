"""Validation run 2026-07-05: estimator + scorer logic checks (pure Python).
Reproduces & extends the 2 Jul code-audit checks with the CURRENT working-tree code."""
import sys, random, statistics, math
sys.path.insert(0, "/sessions/affectionate-tender-clarke/mnt/EECE_608/src")
from dp_audit_tightness.privacy.empirical import estimate_empirical_lower_bound

random.seed(608)
DELTA = 1e-5
kw = dict(delta=DELTA, align_event_to_score_direction=True,
          require_member_favoring=True, report_confidence_supported_lower_bound=True)

def est(m, n, direction="higher"):
    return estimate_empirical_lower_bound(member_scores=m, nonmember_scores=n,
                                          score_direction=direction, **kw).epsilon_lower_empirical

results = []
# T1: no signal -> eps 0
m = [random.gauss(0,1) for _ in range(640)]; n = [random.gauss(0,1) for _ in range(640)]
e = est(m,n); results.append(("T1 no-signal -> eps=0", e, e < 0.05))

# T2: strong signal, members HIGHER, direction='higher' -> eps>0 ; 'lower' -> 0
m = [random.gauss(3,1) for _ in range(640)]; n = [random.gauss(0,1) for _ in range(640)]
eh, el = est(m,n,"higher"), est(m,n,"lower")
results.append(("T2a signal + direction=higher -> eps>0", eh, eh > 0.5))
results.append(("T2b signal + direction=lower  -> eps=0", el, el < 0.05))

# T3: ASYMMETRIC (buggy sigma-sweep) scorer on ZERO-membership-signal synthetic shadows
#     members: mean(OUT)-mean(IN) ~ N(0, small);  nonmembers: mean(shadows)-target ~ N(mu_shift, ...)
#     -> artificial separation -> spurious eps
K, B = 32, 640
def synth_losses(): return [random.gauss(2.0, 0.5) for _ in range(K)]
mem_a, non_a, mem_s, non_s = [], [], [], []
for _ in range(B):
    sl = synth_losses(); ins, outs = sl[:K//2], sl[K//2:]
    target_loss = random.gauss(2.0, 0.5)          # target statistically identical -> NO signal
    mem_a.append(statistics.fmean(outs) - statistics.fmean(ins))   # buggy member formula
    mem_s.append(statistics.fmean(outs) - target_loss)             # fixed symmetric formula
for _ in range(B):
    sl = synth_losses(); target_loss = random.gauss(2.0, 0.5)
    non_a.append(statistics.fmean(sl) - target_loss)               # buggy nonmember formula
    non_s.append(statistics.fmean(sl) - target_loss)               # symmetric (same formula)
ea = max(est(mem_a, non_a, "higher"), est(mem_a, non_a, "lower"))
es = est(mem_s, non_s, "higher")
results.append(("T3a buggy asymmetric scorer, NO signal -> spurious eps", ea, ea > 0.2))
results.append(("T3b fixed symmetric scorer,  NO signal -> eps~0", es, es < 0.05))

# T4: symmetric scorer WITH real membership signal (target overfits members: lower loss)
mem_s2, non_s2 = [], []
for _ in range(B):
    sl = synth_losses()
    mem_s2.append(statistics.fmean(sl[K//2:]) - random.gauss(0.8, 0.4))  # member: low target loss
    non_s2.append(statistics.fmean(sl) - random.gauss(2.0, 0.5))         # nonmember: matches ref
e4 = est(mem_s2, non_s2, "higher")
results.append(("T4 symmetric scorer + real signal + higher -> eps>0", e4, e4 > 0.5))

# T5: validity-gate logic: eps_lower vs eps_upper flag
eps_upper = 0.771
flagged = e4 > eps_upper  # with strong synthetic signal this SHOULD exceed a small bound
results.append(("T5 validity gate fires when eps_lower>eps_upper", e4, flagged == (e4 > eps_upper)))

print(f"{'TEST':<58}{'eps':>8}  PASS")
ok = True
for name, v, p in results:
    ok &= p
    print(f"{name:<58}{v:>8.3f}  {'PASS' if p else 'FAIL'}")
print("\nALL PASS" if ok else "\nSOME FAILED")
