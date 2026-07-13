"""
apps_complexity.py
==================
Reproduces the complexity tables of the "Applications" section of the paper, all
as the single-block (n1=k1=1) specialisation of the per-block GabKron work factor.

Unified work factor (parity reference of p rows, D in V^{n x m}):
    N_eq   = m * k * p
    r*     = floor(k * p / n)     # OVER-determined Burle regime (n r <= k p), r* >= lambda
    log2 W = omega * log2(N_eq) + ((lambda-1)*m - lambda*r*) * log2(q)

  - LGRH  / Modification II : p = n - gamma - k     (gamma = ell)
  - Modification I (X=0)    : p = n - k , dimension k' = k - ell  (so N_eq = m*k'*(n-k))

omega = 2.37 is the value used in the paper tables (Alman-Vassilevska Williams 2024);
omega = 3 is printed alongside for reference (it appears only in prose).

No external dependencies (pure standard library).
"""
from math import log2, floor

OMEGAS = (2.37, 3.0)


def workfactor(m, n, k, p, lam, omega, q=2):
    """Single-block work factor, OVER-determined Burle regime r* = floor(k p / n)."""
    Neq = m * k * p
    r_star = floor(k * p / n)          # requires r_star >= lam (holds for all sets below)
    return omega * log2(Neq) + ((lam - 1) * m - lam * r_star) * log2(q), r_star


def verdict(val, claimed):
    return "BROKEN" if val < claimed else "secure"


# ----------------------------------------------------------------------
# 1. LGRH  (Nouet-Owa et al.) :  p = n - gamma - k
# ----------------------------------------------------------------------
# (name, m, n, k, gamma, q, lambda, claimed)
LGRH = [
    ("LGRH-128", 98, 89, 10, 11, 2, 2, 128),
    ("LGRH-192", 165, 122, 14, 14, 2, 2, 192),
]

# ----------------------------------------------------------------------
# 2. Modification II  :  p = n - ell - k   (ell = gamma) ; lambda in {2,3}
#    Reference figures reported by Nouet-Owa et al. (lambda = 2 only).
# ----------------------------------------------------------------------
MOD2 = [  # (m, n, k, ell, q, claimed, nouetowa_lambda2)
    (88, 88, 48, 2, 2, 132, 96),
    (98, 98, 52, 2, 2, 192, 102),
    (129, 129, 65, 2, 2, 279, 120),
]

# ----------------------------------------------------------------------
# 3. Modification I  (X = 0) :  p = n - k , k' = k - ell ; lambda in {2,3}
# ----------------------------------------------------------------------
MOD1 = [  # (m, n, k, ell, q, claimed)
    (85, 85, 43, 2, 2, 136),
    (98, 98, 50, 3, 2, 203),
    (121, 121, 61, 4, 2, 276),
]


def print_LGRH():
    print("=" * 72)
    print("LGRH  (single block, p = n - gamma - k)")
    print(f"{'set':<10}{'lam':>4}{'r*':>7}", end="")
    for w in OMEGAS:
        print(f"{'W(w='+str(w)+')':>13}", end="")
    print(f"{'claimed':>9}")
    for name, m, n, k, g, q, lam, claimed in LGRH:
        p = n - g - k
        _, r = workfactor(m, n, k, p, lam, OMEGAS[0])
        print(f"{name:<10}{lam:>4}{r:>7}", end="")
        for w in OMEGAS:
            val, _ = workfactor(m, n, k, p, lam, w)
            print(f"{val:>13.1f}", end="")
        print(f"{claimed:>9}")


def print_MOD2():
    print("=" * 86)
    print("Modification II  (single block, p = n - ell - k) ; vs Nouet-Owa (lambda=2)")
    print(f"{'(m,n,k,ell)':<16}{'lam':>4}{'r*':>7}", end="")
    for w in OMEGAS:
        print(f"{'W(w='+str(w)+')':>13}", end="")
    print(f"{'claimed':>9}{'NO25':>7}")
    for m, n, k, ell, q, claimed, no25 in MOD2:
        p = n - ell - k
        for lam in (2, 3):
            _, r = workfactor(m, n, k, p, lam, OMEGAS[0])
            tag = f"({m},{n},{k},{ell})" if lam == 2 else ""
            print(f"{tag:<16}{lam:>4}{r:>7}", end="")
            for w in OMEGAS:
                val, _ = workfactor(m, n, k, p, lam, w)
                print(f"{val:>13.1f}", end="")
            ref = f"{no25}" if lam == 2 else "---"
            print(f"{claimed:>9}{ref:>7}")


def print_MOD1():
    print("=" * 86)
    print("Modification I  (X=0, p = n - k, k' = k - ell)")
    print(f"{'(m,n,k,ell)':<16}{'lam':>4}{'r*':>7}", end="")
    for w in OMEGAS:
        print(f"{'W(w='+str(w)+')':>13}", end="")
    print(f"{'claimed':>9}")
    for m, n, k, ell, q, claimed in MOD1:
        kp = k - ell
        p = n - k
        for lam in (2, 3):
            # image rows are k' here, so N_eq = m*k'*(n-k); reuse workfactor with k<-k'
            _, r = workfactor(m, n, kp, p, lam, OMEGAS[0])
            tag = f"({m},{n},{k},{ell})" if lam == 2 else ""
            print(f"{tag:<16}{lam:>4}{r:>7}", end="")
            for w in OMEGAS:
                val, _ = workfactor(m, n, kp, p, lam, w)
                print(f"{val:>13.1f}", end="")
            print(f"{claimed:>9}")


if __name__ == "__main__":
    print_LGRH()
    print()
    print_MOD2()
    print()
    print_MOD1()
