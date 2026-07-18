"""Independent recomputation of the DP-SGD accounting-gap decomposition (no torch).
Verifies 'Finding 1' (RDP gap dominates at low sigma, ~93% at sigma=0.5).
eps_PLD: project's compute_epsilon_pld (Google dp_accounting PLD).
eps_RDP: dp_accounting RDP accountant, Opacus default alpha grid.
"""
import sys, math
sys.path.insert(0, "/sessions/exciting-laughing-davinci/mnt/EECE_608/src")
from dp_audit_tightness.privacy.pld_accounting import compute_epsilon_pld
from dp_accounting import rdp as rdp_lib
from dp_accounting import dp_event

DELTA = 1e-5
ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))  # Opacus default

def eps_rdp(nm, q, steps, delta):
    acct = rdp_lib.RdpAccountant(ORDERS)
    acct.compose(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(nm)), steps)
    return acct.get_epsilon(delta)

def run(train_n, label):
    q = 256 / train_n; steps = math.ceil(train_n / 256)
    print(f"\n=== {label}: train_n={train_n}, q={q:.6f}, steps={steps}, 1 epoch ===")
    print(f"{'sigma':>5} | {'eps_RDP':>8} | {'eps_PLD':>8} | {'acct_gap':>8} | {'RDP/PLD':>7}")
    print("-"*52)
    rows=[]
    for s in [0.5,0.8,1.1,1.5,2.0,3.0,5.0]:
        er=eps_rdp(s,q,steps,DELTA)
        ep=compute_epsilon_pld(noise_multiplier=s,sampling_rate=q,num_steps=steps,delta=DELTA,backend="google")["epsilon_pld"]
        rows.append((s,er,ep,er-ep)); print(f"{s:>5} | {er:>8.3f} | {ep:>8.3f} | {er-ep:>8.3f} | {er/ep:>6.1f}x")
    return rows

def acct_share(rows, eps_lower):
    print(f"\nAccounting share of total gap (eps_lower={eps_lower}):")
    print(f"{'sigma':>5} | {'total_gap':>9} | {'acct_share':>10}")
    print("-"*32)
    for s,er,ep,gap in rows:
        tot=er-eps_lower; sh=100*gap/tot if tot>0 else float('nan')
        print(f"{s:>5} | {tot:>9.3f} | {sh:>9.1f}%")

r57=run(57000,"original-era (~57k)")
r54=run(54000,"v2 split (54000)")
acct_share(r54, 0.024)
