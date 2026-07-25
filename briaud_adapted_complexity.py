"""
briaud_adapted_complexity.py
============================

Third complexity column W3 of the paper: the Briaud-Loidreau constrained system
(their Eq. (3), Prop. 3) adapted PER BLOCK, solved with the sparse-Wiedemann
polynomial factor at omega = 2, exactly as Nouetowa-Loidreau do.

Two facts anchor this column and are re-checked here:

 (1) VALIDATION.  On the Modification-II parameter sets, the Nouetowa-Loidreau
     "generalised Briaud" cost  P * q^{(lambda-1)m - mu*lambda}  with
         P  = m^3 (n-k+mu) [ (n-k-gamma)(n-k) + mu*n ]^2      (Wiedemann, omega=2)
         mu = floor( n R(1-R) - gamma R ),  R = k/n
     reproduces their PUBLISHED figures 96 / 102 / 120 (< 1 bit).  This is the
     cross-check that the whole W3 machinery is correct.

 (2) CASE SELECTION.
       * Single-block distorted schemes (LGRH, Mod. I, Mod. II) have gamma != 0:
         use the Case-2 formula above (this is the exact N-L formula).
       * GabKron clean blocks are, after per-block clearing (paper Prop. 'General
         per-block decomposition'), PURE masked Gabidulin codes (no residual
         distortion), i.e. the gamma = 0 case.  Then Briaud's own Case-1 lower
         bound applies per inner block [n2,k2]:
             P = m^3 (n2-k2)^5 ,  mu = floor( n2 R(1-R) ),  R = k2/n2 ,
         and, the n1 Kronecker blocks being solved independently (distinct
         parities, direct sum), the global cost adds log2(n1) bits.

W3 is a LOWER BOUND on the linear-algebra step under Wiedemann, exactly as Briaud
and Nouetowa-Loidreau state for their own figures -- not a claimed exact cost.

Why this column is NOT applied to OUR unilateral system: our system is built on the
dense public GENERATOR G_pub (primal form) and is measured to be ~50% dense
(see the sparsity check in the paper's reproducibility notes), so Wiedemann gives
no gain and omega = 2 is NOT justified for W1/W2.  The Briaud/N-L system is DUAL
(parity-based, with a sparse left factor V), hence sparse, hence omega = 2 there.

Pure standard-library Python.
"""
from math import log2, floor


def w3_case2(m, n, k, gamma, lam, q=2):
    """Nouetowa-Loidreau exact 'generalised Briaud' cost (gamma != 0, single block).

    Returns (log2_cost, mu) or (None, mu) if the guess system is not overdetermined.
    """
    R = k / n
    mu = floor(n * R * (1 - R) - gamma * R)
    if mu < lam:
        return None, mu
    inner = (n - k - gamma) * (n - k) + mu * n          # sparse-system size
    P = m ** 3 * (n - k + mu) * inner ** 2              # Wiedemann, omega = 2
    return log2(P) + ((lam - 1) * m - mu * lam) * log2(q), mu


def w3_case1(m, N, K, lam, n1=1, q=2):
    """Briaud Case-1 lower bound (gamma = 0, pure masked Gabidulin block).

    N, K are the INNER block length/dimension; n1 independent blocks add log2(n1).
    Returns (log2_cost, mu) or (None, mu).
    """
    R = K / N
    mu = floor(N * R * (1 - R))
    if mu < lam:
        return None, mu
    per = 3 * log2(m) + 5 * log2(N - K) + ((lam - 1) * m - lam * mu) * log2(q)
    return per + (log2(n1) if n1 > 1 else 0.0), mu


# --------------------------------------------------------------------------- #
#  Validation against the published Nouetowa-Loidreau figures
# --------------------------------------------------------------------------- #
def validate():
    print("=" * 78)
    print(" VALIDATION: reproduce Nouetowa-Loidreau published 'complexity of our attack'")
    print("=" * 78)
    published = [
        ("(88,88,48)", 88, 88, 48, 2, 2, 96),
        ("(98,98,52)", 98, 98, 52, 2, 2, 102),
        ("(129,129,65)", 129, 129, 65, 2, 2, 120),
    ]
    print(f"{'Mod II set':16}{'published':>10}{'our W3':>9}{'mu':>5}{'match?':>8}")
    ok = True
    for nm, m, n, k, gamma, lam, pub in published:
        w, mu = w3_case2(m, n, k, gamma, lam)
        good = abs(w - pub) < 2
        ok = ok and good
        print(f"{nm:16}{pub:>10}{w:>9.1f}{mu:>5}{('YES' if good else 'NO'):>8}")
    print(f"\n  overall: {'PASS' if ok else 'FAIL'}  (W3 machinery reproduces the published state of the art)")
    return ok


# --------------------------------------------------------------------------- #
#  The W3 column for every scheme in the paper
# --------------------------------------------------------------------------- #
def w3_table():
    print("\n" + "=" * 78)
    print(" W3 column (Briaud/N-L adapted, omega=2, sparse Wiedemann lower bound)")
    print("=" * 78)

    print("\n GabKron (Case 1, gamma=0 cleaned blocks, per inner [n2,k2], + log2 n1):")
    print(f"  {'set':16}{'claim':>6}{'W3':>9}{'mu':>5}")
    gab = [
        ("GabKron-128", 48, 2, 24, 12, 3, 128),
        ("GabKron-192", 76, 2, 38, 19, 3, 192),
        ("GabKron-256", 104, 2, 52, 26, 3, 256),
        ("new-GabKron-128", 90, 2, 90, 18, 3, 128),
        ("new-GabKron-192", 120, 2, 120, 32, 3, 192),
        ("new-GabKron-256", 128, 2, 128, 40, 3, 256),
    ]
    for nm, m, n1, n2, k2, lam, cl in gab:
        w, mu = w3_case1(m, n2, k2, lam, n1=n1)
        wt = f"{w:.1f}" if w is not None else "infeas."
        print(f"  {nm:16}{cl:>6}{wt:>9}{mu:>5}")

    print("\n Single-block distorted schemes (Case 2, gamma!=0, exact N-L formula):")
    print(f"  {'set':16}{'claim':>6}{'W3':>9}{'mu':>5}")
    sb = [
        ("LGRH-128", 98, 89, 10, 11, 2, 128),
        ("LGRH-192", 165, 122, 14, 14, 2, 192),
        ("ModII-1", 88, 88, 48, 2, 2, 132),
        ("ModII-2", 98, 98, 52, 2, 2, 192),
        ("ModII-3", 129, 129, 65, 2, 2, 279),
        ("ModI-1", 85, 85, 43, 2, 2, 136),
        ("ModI-2", 98, 98, 50, 3, 2, 203),
        ("ModI-3", 121, 121, 61, 4, 2, 276),
    ]
    for nm, m, n, k, gamma, lam, cl in sb:
        w, mu = w3_case2(m, n, k, gamma, lam)
        wt = f"{w:.1f}" if w is not None else "infeas."
        print(f"  {nm:16}{cl:>6}{wt:>9}{mu:>5}")


if __name__ == "__main__":
    validate()
    w3_table()
