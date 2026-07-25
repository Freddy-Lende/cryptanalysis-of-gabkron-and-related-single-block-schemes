"""
sparsity_check.py
=================

Justifies the linear-algebra exponents used in the complexity tables:

  * W1, W2 (our Burle-type recovery) use omega in {2.37, 3}, NOT omega=2.
  * W3 (Briaud/Nouetowa-Loidreau adapted) uses omega=2 (sparse Wiedemann).

The reason is structural and measured here: the two systems have opposite density.

  OUR system (primal).  Unknowns D, equation  G_pub D (I (x) H0)^T = 0  over F_q.
  It is built on the public GENERATOR G_pub, which is a masked-Gabidulin matrix with
  no zero structure, so each scalar equation couples about half of the unknowns.
  Measured density ~ 0.5 and nonzeros-per-row growing LINEARLY with the number of
  columns -> DENSE.  Wiedemann gives no asymptotic gain, so omega=2 is NOT justified.

  BRIAUD/N-L system (dual).  V Hhat_pub = Hhat_norm W, built on a PARITY-check matrix
  with a sparse left factor V.  Nouetowa-Loidreau report it as sparse and use Wiedemann
  (omega=2); our separate per-block experiment (briaud_perblock.py) measures density
  ~ 0.05-0.08 with a bounded number of nonzeros per row -> SPARSE.

This script measures the density of OUR unilateral system directly.
"""
import random
from structure import GF, moore
from gabkron_attack_common import red, gf2_rank_of


def our_system_density(m, n, p, r, n1=1, seed=1):
    """Build the F_2 matrix of our unilateral system and return (rows, cols, density,
    nonzeros-per-row, max-row).  Uses a random masked-generator-like G_pub, which is a
    density upper bound proxy for the real (also dense) public generator."""
    F = GF(m)
    rng = random.Random(seed)
    k = n // 2
    Gpub = [[rng.randrange(F.QM) for _ in range(n)] for _ in range(k)]
    h0 = [rng.randrange(1, F.QM) for _ in range(m)]
    while gf2_rank_of(F, h0) != m:
        h0 = [rng.randrange(1, F.QM) for _ in range(m)]
    H0 = moore(F, h0, p)
    Fbasis = []
    while len(Fbasis) < r:
        x = rng.randrange(1, F.QM)
        if gf2_rank_of(F, Fbasis + [x]) == len(Fbasis) + 1:
            Fbasis.append(x)
    ncolsD = n1 * m
    U = n * ncolsD * r

    def idx(i, c, l):
        return (i * ncolsD + c) * r + l

    rowmasks = []
    for a in range(k):
        for bb in range(n1):
            for bp in range(p):
                rb = [0] * m
                for i in range(n):
                    gai = Gpub[a][i]
                    if gai == 0:
                        continue
                    for cm in range(m):
                        h = H0[bp][cm]
                        if h == 0:
                            continue
                        base = F.mul(gai, h)
                        c = bb * m + cm
                        for l in range(r):
                            coeff = F.mul(base, Fbasis[l])
                            bit = 1 << idx(i, c, l)
                            while coeff:
                                rb[(coeff & -coeff).bit_length() - 1] ^= bit
                                coeff &= coeff - 1
                rowmasks.extend(rb)
    nnz = sum(bin(x).count('1') for x in rowmasks)
    nr = len(rowmasks)
    mx = max((bin(x).count('1') for x in rowmasks), default=0)
    return nr, U, nnz / (nr * U) if nr * U else 0.0, nnz / max(nr, 1), mx


if __name__ == "__main__":
    print("=" * 84)
    print(" Density of OUR unilateral system  G_pub D (I(x)H0)^T = 0  (primal, generator-based)")
    print("=" * 84)
    print(f"{'m':>3}{'n':>4}{'p':>3}{'r':>3} | {'rows':>6}{'cols':>7}{'density':>9}{'nnz/row':>9}{'nnz/row / cols':>15}")
    prev = None
    for m, n, p, r in [(6, 8, 3, 2), (6, 12, 4, 3), (6, 16, 5, 3), (6, 20, 6, 4)]:
        nr, U, dens, nzr, mx = our_system_density(m, n, p, r)
        print(f"{m:>3}{n:>4}{p:>3}{r:>3} | {nr:>6}{U:>7}{dens:>9.4f}{nzr:>9.1f}{nzr/U:>15.4f}")
    print("""
Reading: density stays ~0.5 and nonzeros-per-row grow LINEARLY with the number of
columns (nnz/row / cols ~ const).  The system is DENSE, so Wiedemann offers no gain and
omega=2 is not justified for W1/W2.  This is why the omega=2 column W3 uses the DUAL
Briaud/Nouetowa-Loidreau system (measured sparse, density ~0.05-0.08 in
briaud_perblock.py) and not our own.
""")
