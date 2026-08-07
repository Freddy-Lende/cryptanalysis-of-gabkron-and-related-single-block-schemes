"""
consistency_checks.py  --  resolves the computational parts of reviewer points 5, 6, formal.

Run:  python3 consistency_checks.py

(6a) Single-copy system: the attack solves ONE system G_pub Z H0^T = 0 (n1=1 -> m*k*p
     equations), not n1 copies (m*k*n1*p). Correct neq = m*k*p ; every work factor drops by
     omega*log2(n1). We print paper(n1-copy) vs single-copy at the three exponents.
(5)  Heuristic 1's ACTUAL claim  Supp_q(L_F) subseteq alpha V  (containment in a scalar
     multiple of V), verified directly -- not merely dim(support) <= lambda -- in the
     accelerated regime r=r_max>lambda. (Large-gap r_max-lambda~29 stays a heuristic
     extrapolation: unreachable in pure Python.)
(F)  Theorem 1 (full-rank extraction rk(G_pub D_F)=k) for EVERY r in [lambda, r_max], not
     only r=r_max, since Theorem 3 invokes it at r=lambda.
Pure standard library + the verified attack primitives.
"""
import random
import io, contextlib
from math import log2
with contextlib.redirect_stdout(io.StringIO()):
    from gabkron_complexity_perblock import log2_inv_S1, rstar, STR
from gabkron_attack import (build_instance, moore, extend_to, basis_extract,
                            gf2_rank_of, is_scalar_multiple_of_V)


def _logW(neq, m, lam, r, omega):
    return omega * log2(neq) + log2_inv_S1(m, lam, r)


def single_copy():
    print("=" * 92)
    print(" (6a) single-copy system: neq = m*k*p (NOT m*k*n1*p) -> W drops by omega*log2(n1)")
    print("=" * 92)
    SETS = [("GabKron-128", 2, 2, 24, 12, 48, 3, 128),
            ("GabKron-192", 2, 2, 38, 19, 76, 3, 192),
            ("GabKron-256", 2, 2, 52, 26, 104, 3, 256)]
    for tag, om in (("2.807", STR), ("3", 3.0), ("2.37", 2.37)):
        print(f"\n  omega={tag}: {'set':15}{'claim':>6}{'W_paper(wc)':>12}{'W_single(wc)':>13}{'drop':>6}")
        for name, n1, k1, n2, k2, m, lam, cl in SETS:
            n, k = n1 * n2, k1 * k2
            t2 = (n2 - k2) // 2
            # worst case over the public global rank t1 (decryption-safe), single-copy vs n1-copy
            Wp = Ws = float("-inf")
            for t1 in range(1, t2 + 1):
                p = n2 - t1 - k2
                if p <= 0:
                    continue
                r = rstar(k, p, n)
                if r < lam:
                    continue
                Wp = max(Wp, _logW(m * k * n1 * p, m, lam, r, om))
                Ws = max(Ws, _logW(m * k * p, m, lam, r, om))
            print(f"          {name:15}{cl:>6}{Wp:>12.1f}{Ws:>13.1f}{Wp - Ws:>6.1f}")
    print(f"\n  reduction = omega*log2(n1) = {STR:.2f}/3/2.37 bits for n1=2.")


def heuristic1_containment():
    print("\n" + "=" * 92)
    print(" (5) Heuristic 1 direct: Supp_q(L_F) subseteq alpha V  (containment, not just dim)")
    print("=" * 92)
    for (m, n1, k1, n2, k2, lam, t1, N) in [(16, 1, 1, 16, 6, 2, 1, 10),
                                            (16, 2, 2, 16, 6, 2, 1, 6),
                                            (18, 1, 1, 18, 8, 2, 1, 6)]:
        dimok = cont = 0; rmax = gap = None
        for s in range(11000, 11000 + N):
            I = build_instance(m, n1, k1, n2, k2, lam, s, t1=t1)
            F, k, n, p = I['F'], I['k'], I['n'], I['p']
            rmax = (k * p) // n; gap = rmax - lam
            rng = random.Random(s ^ 0x55)
            while True:
                h0 = [rng.randrange(1, F.QM) for _ in range(F.m)]
                if gf2_rank_of(F, h0) == F.m:
                    break
            kD, d, sup, ir = basis_extract(I, moore(F, h0, p), extend_to(F, I['Vb'], rmax, rng), h0)
            if kD is None:
                continue
            supp = [x for row in kD for x in row]
            if gf2_rank_of(F, supp) <= lam:
                dimok += 1
            if is_scalar_multiple_of_V(F, I['Vb'], supp):
                cont += 1
        print(f"  m={m} n1={n1} n2={n2} k2={k2} lam={lam} | r_max={rmax} gap={gap}: "
              f"dim<=lam {dimok}/{N}, Supp ⊆ alphaV {cont}/{N}")


def theorem1_all_r():
    print("\n" + "=" * 92)
    print(" (F) Theorem 1: full-rank extraction rk(G_pub D_F)=k for EVERY r in [lambda, r_max]")
    print("=" * 92)
    for (m, n1, k1, n2, k2, lam, t1, N) in [(16, 1, 1, 16, 6, 2, 1, 4),
                                            (18, 1, 1, 18, 8, 2, 1, 4)]:
        I0 = build_instance(m, n1, k1, n2, k2, lam, 0, t1=t1)
        k, n, p = I0['k'], I0['n'], I0['p']; rmax = (k * p) // n
        print(f"  m={m} n2={n2} k2={k2} lam={lam}: r in [{lam},{rmax}]")
        for r in range(lam, rmax + 1):
            ok = 0
            for s in range(12000, 12000 + N):
                I = build_instance(m, n1, k1, n2, k2, lam, s, t1=t1); F = I['F']
                rng = random.Random(s ^ r)
                while True:
                    h0 = [rng.randrange(1, F.QM) for _ in range(F.m)]
                    if gf2_rank_of(F, h0) == F.m:
                        break
                kD, d, sup, ir = basis_extract(I, moore(F, h0, p), extend_to(F, I['Vb'], r, rng), h0)
                if kD is not None and ir == k:
                    ok += 1
            print(f"    r={r} (r-lam={r - lam}): ir=k {ok}/{N}")


if __name__ == "__main__":
    single_copy()
    heuristic1_containment()
    theorem1_all_r()
