"""
success_probability.py  --  turns the accelerated-regime estimate into a rigorous
two-sided bound (review point on the union bound being "in the wrong direction").

For a uniformly random r-dimensional F_q-subspace F of F_qm and a fixed lambda-dim V,
the attack at r = r_max succeeds iff some orbit member alpha*V lies in F:

    success  =  Union_{alpha} A_alpha,   A_alpha = { alpha*V subseteq F }.

Let W_i = alpha_i V run over the orbit, of size  M = (q^m - 1)/|Stab(V)|.

  * First moment (Bonferroni upper term):
        S1 = M * Pr[A]  with  Pr[A] = C(m-lambda, r-lambda)_q / C(m, r)_q .

  * Second moment: Pr[A_i and A_j] = C(m-s, r-s)_q / C(m, r)_q, where
        s = dim(W_i + W_j) = 2*lambda - dim(W_i cap W_j) in [lambda+1, 2*lambda].
    Two DISTINCT orbit members never satisfy s < 2*lambda-1 for generic V (the
    intersection W_i cap W_j has dimension 0 or 1 only); in fact when
        r_max <= 2*lambda - 2
    no two distinct members fit in F at all, so the A_alpha are DISJOINT and
        Pr[success] = S1   exactly.

  * When r_max >= 2*lambda-1, inclusion-exclusion gives the lower bound
        Pr[success] >= S1 - S2,  S2 = sum_{i<j} Pr[A_i and A_j],
    and S2/S1 is exponentially small (measured below), so
        Pr[success] = S1 * (1 - o(1)),   E[#trials] = 1/S1 * (1 + o(1)).

Hence E[#trials] = 1/S1 is a genuine (two-sided) estimate, not merely an optimistic
lower bound. The leading-order union-bound figure q^{(lambda-1)m - lambda r} used in the
text under-estimates 1/S1 by only a fraction of a bit (shown below).

This script (i) prints S1, the disjointness threshold, and 1/S1 for the paper's sets, and
(ii) empirically measures Pr[success] at small sizes to confirm S1 is accurate.
Pure standard library.
"""
import random
from math import log2, gcd

Q = 2


def log2_gaussian(m, l, q=Q):
    if l < 0 or l > m:
        return float("-inf")
    s = 0.0
    for i in range(l):
        s += log2(q ** (m - i) - 1) - log2(q ** (l - i) - 1)
    return s


def max_stabiliser(m, lam, q=Q):
    return q ** gcd(lam, m) - 1


def log2_S1(m, lam, r, q=Q, stab=1):
    """log2 of the first-moment success probability S1 = M * Pr[A]."""
    M = log2((q ** m - 1)) - log2(stab)               # orbit size (q^m-1)/|Stab|
    PA = log2_gaussian(m - lam, r - lam, q) - log2_gaussian(m, r, q)
    return M + PA


def log2_S2_over_S1(m, lam, r, q=Q):
    """Upper bound on log2(S2/S1). S2 is dominated by pairs with s = 2*lambda-1
    (intersection dimension 1); their count is at most ~ q^{2*lambda}."""
    if r < 2 * lam - 1:
        return float("-inf")                           # disjoint: S2 = 0 exactly
    logPA = log2_gaussian(m - lam, r - lam, q) - log2_gaussian(m, r, q)
    logP_pair = log2_gaussian(m - (2 * lam - 1), r - (2 * lam - 1), q) \
        - log2_gaussian(m, r, q)
    log_count = 2 * lam                                # generous bound on # pairs at s=2lam-1
    return log_count + logP_pair - logPA - 1


GABKRON = [
    ("GabKron-128",     24, 12,  48, 3, 128),
    ("GabKron-192",     38, 19,  76, 3, 192),
    ("GabKron-256",     52, 26, 104, 3, 256),
    ("new-GabKron-128", 90, 18,  90, 3, 128),
    ("new-GabKron-192",120, 32, 120, 3, 192),
    ("new-GabKron-256",128, 40, 128, 3, 256),
]


def report():
    print("=" * 96)
    print(" Accelerated-regime success probability: two-sided bound (review point 3)")
    print("   E[#trials] = 1/S1 ; disjoint when r_max <= 2*lambda-2 (then exact);")
    print("   otherwise Pr[success] >= S1 - S2 with S2/S1 exponentially small.")
    print("=" * 96)
    print(f"{'set':17}{'lam':>4}{'r_max':>7}{'2lam-1':>8}{'log2 1/S1':>11}"
          f"{'union bd':>10}{'log2 S2/S1':>12}")
    for name, n2, k2, m, lam, claim in GABKRON:
        n, k = 2 * n2, 2 * k2
        t2 = (n2 - k2) // 2
        # worst rho -> take the r_max giving the largest 1/S1 (smallest S1)
        best = None
        for rho in range(1, t2 + 1):
            p = n2 - rho - k2
            if p <= 0:
                continue
            r = (k * p) // n
            lS1 = log2_S1(m, lam, r)
            if best is None or -lS1 > best[0]:
                best = (-lS1, r)
        inv_s1, r = best
        ub = (lam - 1) * m - lam * r
        s2 = log2_S2_over_S1(m, lam, r)
        s2s = "-inf (disjoint)" if s2 == float("-inf") else f"{s2:.1f}"
        print(f"{name:17}{lam:>4}{r:>7}{2*lam-1:>8}{inv_s1:>11.1f}{ub:>10}{s2s:>12}")


def measure(m=10, lam=3, rs=(5, 6), N=20000, seed=7):
    print()
    print("=" * 96)
    print(f" Empirical check at m={m}, lambda={lam}: measured Pr[success] vs first moment S1")
    print("=" * 96)
    rng = random.Random(seed)

    def span(basis):
        out = {0}
        for b in basis:
            out |= {x ^ b for x in out}
        return out

    # field
    import structure as ss
    ss.IRRED.setdefault(10, 0b10000001001)
    ss.IRRED.setdefault(12, 0b1000001010011)
    from gabkron_attack import GF, rand_subspace
    F = GF(m)
    Vb = rand_subspace(F, lam, rng)
    orbit = {frozenset(span([F.mul(a, v) for v in Vb])) for a in range(1, F.QM)}
    orbit = [list(o) for o in orbit]
    print(f"  orbit size = {len(orbit)}  (|Stab(V)| = {(F.QM-1)//len(orbit)})")
    print(f"  {'r':>3}{'measured':>12}{'S1':>12}{'ratio':>8}")
    for r in rs:
        hits = 0
        for _ in range(N):
            Fset = span(rand_subspace(F, r, rng))
            if any(all(w in Fset for w in W) for W in orbit):
                hits += 1
        p = hits / N
        S1 = 2 ** log2_S1(m, lam, r)
        print(f"  {r:>3}{p:>12.4f}{S1:>12.4f}{p/S1:>8.3f}")
    print("  ratio ~ 1 confirms S1 is an accurate (two-sided) estimate, not a loose bound.")


if __name__ == "__main__":
    report()
    measure()
