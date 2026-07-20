#!/usr/bin/env python3
"""
proven_complexity.py -- work factors of the PROVEN regime r = lambda.

Reproduces Table `tab:proven` of the paper (Theorem "Heuristic-free recovery at
r = lambda"), i.e. equation (eq:Wrig):

    log2 W^pr = omega * log2(N_eq) + log2( |Stab(V)| * [m choose lambda]_q / (q^m - 1) )

with N_eq = m*k*n_1*p the number of F_q-equations of the recovery system, and
[m choose lambda]_q the Gaussian binomial coefficient.  Generically
|Stab(V)| = |F_q^*| = q - 1.

Contrast with the accelerated regime r = r_max used in the other tables, whose
guessing exponent is ((lambda-1)*m - lambda*r_max)*log2 q.  The proven regime
costs exactly lambda*(r_max - lambda)*log2 q extra bits, and no longer depends
on r_max: W^pr is therefore monotone decreasing in t_1 through the polynomial
factor alone, so the worst case over t_1 in [1, t_2] sits at t_1 = 1.

Pure standard library.  Run:  python3 proven_complexity.py
"""

from math import log2

Q = 2
OMEGAS = (2.37, 3.0)


# ----------------------------------------------------------------------
# Gaussian binomial and exact guess count
# ----------------------------------------------------------------------

def log2_gaussian_binomial(m, lam, q=Q):
    """log2 of [m choose lam]_q = prod_{i=0}^{lam-1} (q^{m-i}-1)/(q^{lam-i}-1)."""
    if lam < 0 or lam > m:
        raise ValueError("need 0 <= lam <= m")
    total = 0.0
    for i in range(lam):
        total += log2(q ** (m - i) - 1) - log2(q ** (lam - i) - 1)
    return total


def log2_trials_proven(m, lam, q=Q, stab=None):
    """log2 of the expected number of guesses at r = lambda.

    A good guess is an element of the orbit {alpha*V : alpha in F_qm^*}, of size
    (q^m - 1)/|Stab(V)| inside the Grassmannian Gr_lambda(q, m).  Hence

        E[#trials] = |Stab(V)| * [m choose lambda]_q / (q^m - 1).
    """
    if stab is None:
        stab = q - 1               # generic stabiliser F_q^*
    return log2(stab) + log2_gaussian_binomial(m, lam, q) - log2(q ** m - 1)


def log2_trials_accelerated(m, lam, r_max, q=Q):
    """log2 of the expected number of guesses at r = r_max (Lemma scalar_count)."""
    return ((lam - 1) * m - lam * r_max) * log2(q)


def work_factor(n_eq, log2_trials, omega):
    return omega * log2(n_eq) + log2_trials


# ----------------------------------------------------------------------
# Parameter sets
# ----------------------------------------------------------------------

# GabKron: (name, n1, k1, n2, k2, m, lambda, claimed, fixed_t1 or None)
GABKRON = [
    ("GabKron-128",     2, 2,  24, 12,  48, 3, 128, None),
    ("GabKron-192",     2, 2,  38, 19,  76, 3, 192, None),
    ("GabKron-256",     2, 2,  52, 26, 104, 3, 256, None),
    ("new-GabKron-128", 2, 2,  90, 18,  90, 3, 128, 6),
    ("new-GabKron-192", 2, 2, 120, 32, 120, 3, 192, 8),
    ("new-GabKron-256", 2, 2, 128, 40, 128, 3, 256, 8),
]

# Single block, LGRH / Modification II: (name, m, n, k, gamma, lambda, claimed)
SINGLE_L = [
    ("LGRH-128",   98,  89, 10, 11, 2, 128),
    ("LGRH-192",  165, 122, 14, 14, 2, 192),
    ("ModII-1",    88,  88, 48,  2, 2, 132),
    ("ModII-1",    88,  88, 48,  2, 3, 132),
    ("ModII-2",    98,  98, 52,  2, 2, 192),
    ("ModII-2",    98,  98, 52,  2, 3, 192),
    ("ModII-3",   129, 129, 65,  2, 2, 279),
    ("ModII-3",   129, 129, 65,  2, 3, 279),
]

# Modification I (subcode, k' = k - l): (name, m, n, k, l, lambda, claimed)
SINGLE_I = [
    ("ModI-1",  85,  85, 43, 2, 2, 136),
    ("ModI-1",  85,  85, 43, 2, 3, 136),
    ("ModI-2",  98,  98, 50, 3, 2, 203),
    ("ModI-2",  98,  98, 50, 3, 3, 203),
    ("ModI-3", 121, 121, 61, 4, 2, 276),
    ("ModI-3", 121, 121, 61, 4, 3, 276),
]


def verdict(w, claimed):
    return "BROKEN" if w < claimed else "not broken"


def report_gabkron():
    print("=" * 84)
    print("GabKron / new-GabKron -- proven regime r = lambda")
    print("  (original sets: worst case over t_1 in [1, t_2], attained at t_1 = 1)")
    print("=" * 84)
    header = f"{'set':17} {'claim':>5} {'t1':>3} {'r_max':>5} {'W1_acc':>7} " \
             f"{'W1_pr':>7} {'W2_pr':>7}  verdict (W1_pr)"
    print(header)
    for (name, n1, k1, n2, k2, m, lam, claimed, t_fixed) in GABKRON:
        n, k = n1 * n2, k1 * k2
        t2 = (n2 - k2) // 2
        candidates = [t_fixed] if t_fixed else list(range(1, t2 + 1))
        worst = None
        for t1 in candidates:
            p = n2 - t1 - k2
            if p <= 0:
                continue
            n_eq = m * k * n1 * p
            r_max = (k * p) // n
            w1_pr = work_factor(n_eq, log2_trials_proven(m, lam), 2.37)
            if worst is None or w1_pr > worst[0]:
                w2_pr = work_factor(n_eq, log2_trials_proven(m, lam), 3.0)
                w1_acc = work_factor(
                    n_eq, log2_trials_accelerated(m, lam, r_max), 2.37)
                worst = (w1_pr, w2_pr, w1_acc, t1, r_max)
        w1_pr, w2_pr, w1_acc, t1, r_max = worst
        print(f"{name:17} {claimed:>5} {t1:>3} {r_max:>5} {w1_acc:>7.1f} "
              f"{w1_pr:>7.1f} {w2_pr:>7.1f}  {verdict(w1_pr, claimed)}")


def report_single(rows, kind, title):
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)
    print(f"{'set':12} {'lam':>3} {'claim':>5} {'r_max':>5} {'W1_acc':>7} "
          f"{'W1_pr':>7} {'W2_pr':>7}  verdict (W1_pr)")
    for row in rows:
        if kind == "L":
            name, m, n, k, gamma, lam, claimed = row
            n_eq = m * k * (n - gamma - k)
            r_max = (k * (n - gamma - k)) // n
        else:                                   # Modification I
            name, m, n, k, ell, lam, claimed = row
            k_prime = k - ell
            n_eq = m * k_prime * (n - k)
            r_max = (k_prime * (n - k)) // n
        w1_acc = work_factor(n_eq, log2_trials_accelerated(m, lam, r_max), 2.37)
        w1_pr = work_factor(n_eq, log2_trials_proven(m, lam), 2.37)
        w2_pr = work_factor(n_eq, log2_trials_proven(m, lam), 3.0)
        print(f"{name:12} {lam:>3} {claimed:>5} {r_max:>5} {w1_acc:>7.1f} "
              f"{w1_pr:>7.1f} {w2_pr:>7.1f}  {verdict(w1_pr, claimed)}")


def sanity_checks():
    """The proven regime must cost exactly lambda*(r_max - lambda) extra bits."""
    print()
    print("=" * 84)
    print("Sanity check: (proven - accelerated) guessing exponent, in bits")
    print("  expected ~ lambda*(r_max - lambda) + O(1)")
    print("=" * 84)
    for (name, m, n, k, gamma, lam, claimed) in SINGLE_L:
        r_max = (k * (n - gamma - k)) // n
        delta = log2_trials_proven(m, lam) - log2_trials_accelerated(m, lam, r_max)
        predicted = lam * (r_max - lam)
        print(f"{name:12} lam={lam}  r_max={r_max:>3}  measured={delta:7.2f}  "
              f"predicted={predicted:>4}  gap={delta - predicted:+.2f}")


if __name__ == "__main__":
    report_gabkron()
    report_single(SINGLE_L, "L", "LGRH and Modification II -- proven regime r = lambda")
    report_single(SINGLE_I, "I", "Modification I -- proven regime r = lambda")
    sanity_checks()
