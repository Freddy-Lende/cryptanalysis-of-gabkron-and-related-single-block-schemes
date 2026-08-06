"""
global_complexity.py  --  Reviewer point 3 (cost of not knowing rho).

The per-block width rho = max_i Colrk_q(X^(i)) is unknown to the attacker; only
t2 = floor((n2-k2)/2), a PUBLIC quantity, bounds it (rho in [1,t2]). We compare
three ways to price this, at each of the three linear-algebra exponents:

  (A) W_max  = max_{rho} W(rho)          -- the current table verdict (optimistic:
                                            assumes the best rho is known for free).
  (B) W_sum  = log2 sum_{rho} 2^{W(rho)} -- the reviewer's honest enumeration cost
                                            (worst case: try every rho).
  (C) W_univ = W(rho = t2)               -- GLOBAL, SEARCH-FREE: clear to the minimal
                                            n2 - t2 clean columns per block. This single
                                            clearing is valid for EVERY layout (rho<=t2),
                                            so no enumeration is ever performed. Because
                                            W(rho) is a sawtooth peaking at an interior
                                            rho, W_univ = W(t2) <= W_max always.

Key structural point: at rho = t2 one has p = n2 - t2 - k2 and k*p is the smallest
admissible, so r_max(t2) is the smallest r_max; when k*(n2-t2-k2)*n1 = n*lambda exactly,
r_max(t2) = lambda and the universal attack IS the proven (heuristic-free) regime.

Pure Python. Imports the exact 1/S1 term from gabkron_complexity_perblock.
"""
from math import log2
from gabkron_complexity_perblock import logW, log2_inv_S1, rstar, STR

def Wrho(neq, m, lam, r, omega, q=2):
    return logW(neq, m, lam, r, omega, q)

def logsumexp2(vals):
    a = max(vals)
    return a + log2(sum(2.0**(v-a) for v in vals))

def analyse(name, n1, k1, n2, k2, q, m, lam, claim):
    n, k = n1*n2, k1*k2
    t2 = (n2 - k2)//2
    rows = []                 # (rho, p, r_max, r_used_proven=lam)
    for rho in range(1, t2+1):
        p = n2 - rho - k2
        if p <= 0: continue
        rmax = rstar(k, p, n)
        if rmax < lam: continue
        neq = m*k*n1*p
        rows.append((rho, p, rmax, neq))
    out = {}
    for tag, omega in (("S",STR),("3",3.0),("237",2.37)):
        # accelerated: r = r_max(rho)
        Wacc = [Wrho(neq, m, lam, rmax, omega) for (rho,p,rmax,neq) in rows]
        # proven: r = lam (fixed), depends on rho only through neq
        Wpr  = [Wrho(neq, m, lam, lam, omega) for (rho,p,rmax,neq) in rows]
        rho_t2, p_t2, rmax_t2, neq_t2 = rows[-1]           # rho = t2 (last)
        out[tag] = dict(
            acc_max=max(Wacc), acc_sum=logsumexp2(Wacc), acc_univ=Wrho(neq_t2,m,lam,rmax_t2,omega),
            pr_max=max(Wpr),   pr_sum=logsumexp2(Wpr),   pr_univ=Wrho(neq_t2,m,lam,lam,omega),
            rmax_t2=rmax_t2, t2=t2)
    return name, claim, lam, out, rows[-1]

SETS = [("GabKron-128",2,2,24,12,2,48,3,128),
        ("GabKron-192",2,2,38,19,2,76,3,192),
        ("GabKron-256",2,2,52,26,2,104,3,256),
        ("new-GabKron-256",2,2,128,40,2,128,3,256)]

def mark(w, cl): return f"*{w:5.1f}*" if w < cl else f" {w:5.1f} "

print("="*118)
print(" Reviewer point 3: pricing the unknown rho.  A=max_rho (current) | B=sum_rho (honest enum) | C=W(t2) (global, search-free)")
print("="*118)
for name,n1,k1,n2,k2,q,m,lam,claim in SETS:
    nm,cl,l,out,(rho_t2,p_t2,rmax_t2,neq_t2) = analyse(name,n1,k1,n2,k2,q,m,lam,claim)
    proven_at_t2 = (rmax_t2 == lam)
    print(f"\n{name}   (claim {claim}, t2={out['S']['t2']}, r_max(t2)={rmax_t2}"
          f"{'  == lambda  => universal attack IS the proven regime' if proven_at_t2 else ''})")
    print(f"   {'exp':>5} | {'ACCELERATED  A_max   B_sum   C_univ(t2)':<42} | {'PROVEN r=lam  A_max   B_sum   C_univ(t2)':<42}")
    for tag in ("S","3","237"):
        o=out[tag]
        acc=f"{mark(o['acc_max'],cl)} {mark(o['acc_sum'],cl)} {mark(o['acc_univ'],cl)}"
        pr =f"{mark(o['pr_max'],cl)} {mark(o['pr_sum'],cl)} {mark(o['pr_univ'],cl)}"
        lbl={'S':'2.807','3':'3.0','237':'2.37'}[tag]
        print(f"   {lbl:>5} | {acc:<42} | {pr:<42}")
print("\n(* = below claim.  C_univ = W(t2): one universal clearing to n2-t2 columns, NO rho search.)")
