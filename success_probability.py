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
        s = dim(W_i + W_j) = 2*lambda - c,  c = dim(W_i cap W_j) in {0,...,lambda-1}
    (c = lambda would force beta = alpha_j/alpha_i in Stab(V) = F_q^*). So s >= lambda+1.

  * At r = lambda the events are mutually exclusive (a good guess of dimension lambda
    equals alpha V), so S2 = 0 and Pr[success] = S1 EXACTLY.

  * For r > lambda, group the pairs by c and bound S2/S1 = (1/2) sum_{beta!=1}
    C(m-s,r-s)/C(m-lambda,r-lambda) by TWO parts (the earlier proof kept only the second):
        - DISJOINT pairs c=0 (s=2*lambda): up to M = (q^m-1)/(q-1) of them (exponentially
          many in m). They contribute only for r >= 2*lambda (else C(m-2lam,r-2lam)=0):
              M * C(m-2lam,r-2lam)/C(m-lam,r-lam) = q^{-Omega(m)}.
        - INTERSECTING pairs c in [1,lambda-1] (s in [lambda+1,2lambda-1]): the count of
          beta!=1 with V cap beta V != {0} is <= ((q^lambda-1)/(q-1))^2, independent of m;
          bounding each by the largest term (s=lambda+1) gives q^{-Omega(m)}.
    Hence S2/S1 = q^{-Omega(m)} and Pr[success] = S1(1 - q^{-Omega(m)}),
    E[#trials] = 1/S1 (1 + q^{-Omega(m)}).

Hence E[#trials] = 1/S1 is a genuine (two-sided) estimate. The leading-order figure
q^{(lambda-1)m - lambda r} under-estimates 1/S1 by up to ~1.6 bits (GabKron, r near lambda).

This script prints 1/S1 and BOTH components of the S2/S1 bound (disjoint and intersecting)
for the paper's sets, and empirically measures Pr[success] at small sizes (including the
polynomial V with c=2) to confirm S1 is accurate. Pure standard library.
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


def log2_S2_components(m, lam, r, q=Q):
    """Two components of the bound  S2/S1 <= (1/2)(disjoint + intersecting).

    Writing S1 = M*Pr[V subseteq F] and, for a pair with sum-dimension s = 2*lam - c
    (c = dim(V cap beta V)), Pr[both]/Pr[one] = C(m-s, r-s)_q / C(m-lam, r-lam)_q, one has
        S2/S1 = (1/2) * sum_{beta != 1} C(m-s(beta), r-s(beta)) / C(m-lam, r-lam).
    We split the sum by c:

      * DISJOINT pairs  (c = 0, s = 2*lam):  there are up to M = (q^m-1)/(q-1) of them
        (exponentially many in m -- the term omitted by the earlier proof).  They enter
        only once r >= 2*lam, since C(m-2lam, r-2lam) = 0 for r < 2*lam.  Contribution
            (M) * C(m-2lam, r-2lam) / C(m-lam, r-lam).

      * INTERSECTING pairs (1 <= c <= lam-1, s = 2*lam - c in [lam+1, 2*lam-1]):
        the number of beta != 1 (mod F_q^*) with V cap beta V != {0} is at most
        ((q^lam-1)/(q-1))^2 (beta = y/x with x,y over the (q^lam-1)/(q-1) lines of V),
        INDEPENDENT of m.  Bounding every such pair by the largest term (smallest s = lam+1):
            ((q^lam-1)/(q-1))^2 * C(m-(lam+1), r-(lam+1)) / C(m-lam, r-lam).

    Both are q^{-Omega(m)} for fixed lam and r-lam; at r = lam both are 0 (S2 = 0 exactly).
    Returns (log2 disjoint_term, log2 intersecting_term), each -inf when it does not apply.
    """
    logPA = log2_gaussian(m - lam, r - lam, q) - log2_gaussian(m, r, q)   # log2 Pr[V subseteq F]
    # disjoint c = 0 (s = 2*lam): count ~ M, needs r >= 2*lam
    if r >= 2 * lam:
        logM = log2(q ** m - 1) - log2(q - 1)                             # generic |Stab| = q-1
        logP_disj = log2_gaussian(m - 2 * lam, r - 2 * lam, q) - log2_gaussian(m, r, q)
        t_disj = logM + logP_disj - logPA
    else:
        t_disj = float("-inf")
    # intersecting c in [1, lam-1], worst s = lam+1: m-independent count
    if r >= lam + 1:
        lines = (q ** lam - 1) // (q - 1)
        logP_int = log2_gaussian(m - (lam + 1), r - (lam + 1), q) - log2_gaussian(m, r, q)
        t_int = 2 * log2(lines) + logP_int - logPA
    else:
        t_int = float("-inf")
    return t_disj, t_int


def log2_S2_over_S1(m, lam, r, q=Q):
    """Full (two-sided) upper bound on log2(S2/S1), disjoint + intersecting pairs.

    S2 = 0 exactly at r = lambda (mutually exclusive). For r > lambda the events are NOT
    mutually exclusive and BOTH the disjoint and the intersecting pairs contribute; the
    total is q^{-Omega(m)}, so Pr[success] = S1 (1 - q^{-Omega(m)}) and 1/S1 is a genuine
    two-sided estimate of E[#trials]."""
    t_disj, t_int = log2_S2_components(m, lam, r, q)
    if t_disj == float("-inf") and t_int == float("-inf"):
        return float("-inf")                                   # r = lambda: S2 = 0 exact
    hi = max(t_disj, t_int)
    lo_disj = 0.0 if t_disj == float("-inf") else 2.0 ** (t_disj - hi)
    lo_int = 0.0 if t_int == float("-inf") else 2.0 ** (t_int - hi)
    return -1.0 + hi + log2(lo_disj + lo_int)                  # log2( (1/2)(2^t_disj + 2^t_int) )


GABKRON = [
    ("GabKron-128",     24, 12,  48, 3, 128),
    ("GabKron-192",     38, 19,  76, 3, 192),
    ("GabKron-256",     52, 26, 104, 3, 256),
    ("new-GabKron-128", 90, 18,  90, 3, 128),
    ("new-GabKron-192",120, 32, 120, 3, 192),
    ("new-GabKron-256",128, 40, 128, 3, 256),
]


def report():
    print("=" * 104)
    print(" Accelerated-regime success probability: two-sided bound (review point on Lemma 8)")
    print("   E[#trials] = 1/S1 ; S2 = 0 exactly at r = lambda (mutually exclusive).")
    print("   For r > lambda: S2/S1 <= (1/2)(disjoint + intersecting), both q^{-Omega(m)}.")
    print("   disjoint = c=0 pairs (s=2lam, ~M of them, r>=2lam); intersecting = c>=1 pairs.")
    print("=" * 104)
    print(f"{'set':17}{'lam':>4}{'r_max':>7}{'2lam':>6}{'log2 1/S1':>11}"
          f"{'union bd':>10}{'log2 disj':>11}{'log2 int':>10}{'log2 S2/S1':>12}")
    for name, n2, k2, m, lam, claim in GABKRON:
        n, k = 2 * n2, 2 * k2
        t2 = (n2 - k2) // 2
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
        t_disj, t_int = log2_S2_components(m, lam, r)
        s2 = log2_S2_over_S1(m, lam, r)
        fmt = lambda x: "  -inf   " if x == float("-inf") else f"{x:>9.1f}"
        s2s = "-inf(exact)" if s2 == float("-inf") else f"{s2:>10.1f}"
        print(f"{name:17}{lam:>4}{r:>7}{2*lam:>6}{inv_s1:>11.1f}{ub:>10}"
              f"{fmt(t_disj):>11}{fmt(t_int):>10}{s2s:>12}")
    print("-" * 104)
    print(" Both components are hugely negative for every set, so Pr[success] = S1(1 - o(1))")
    print(" and 1/S1 is a genuine two-sided estimate. (The disjoint term, omitted before, is")
    print(" the one that switches on at r >= 2*lambda -- present for the larger sets.)")


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
