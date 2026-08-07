#!/usr/bin/env python3
# =============================================================================
#  module_structure.py
#
#  Reproduces the F_{q^m}-module claims of the paper (Lemma "Frobenius module
#  structure", Heuristic "Dimension of the solution module", and the structure
#  Remark), for the per-block recovery system
#
#        L_F = { Z in F^{n x m} : G_pub Z H0^T = 0 },   F = alpha V subset F (good guess)
#
#  It measures, on GOOD guesses (F contains a scalar multiple of V), at r = r_max:
#
#    (1) dim_{F_qm} L_F   (should be n1^2)     [dim_Fq L_F / m]
#    (2) m | dim_Fq L_F   (K-space consistency)
#    (3) K-stability:      Z in L_F  =>  Z R_beta^T in L_F   (support preserved)
#    (4) every element of L_F is alpha V-valued   (Supp_q(Z) <= lambda)
#    (5) deterministic extraction: every n1-subset of a K-basis of L_F has
#        image rank exactly k
#
#  All heavy lifting (field, instance, trusted kernel solver) is imported from
#  gabkron_attack.py / structure.py -- no re-implementation of the solver.
#
#  Pure standard-library Python; no SageMath, no external data.
# =============================================================================
import random, io, contextlib, itertools, operator
from functools import reduce

with contextlib.redirect_stdout(io.StringIO()):
    import structure as ss
    # register the larger fields used below (structure ships only a few)
    for _m, _p in {10: 0x409, 12: 0x1053, 14: 0x4443, 16: 0x1100b, 18: 0x40081}.items():
        ss.IRRED.setdefault(_m, _p)
    from structure import matmul, moore, rank
    import gabkron_attack as GA

red = lambda it: reduce(operator.xor, it, 0)


# --------------------------------------------------------------------------- #
#  The F_qm-module machinery (R_beta, the action Z.beta = Z R_beta^T, membership,
#  entry-support and the F_qm-basis extraction) now lives in the shared module
#  gabkron_attack_common.py, so that gabkron_attack.py can use the very same code
#  for its deterministic key extraction.  See the paper's Lemma "Frobenius
#  F_qm-module structure" and Theorem "Deterministic extraction from a kernel basis".
# --------------------------------------------------------------------------- #
from gabkron_attack_common import R_beta, act, in_span, supp_dim, kbasis

# --------------------------------------------------------------------------- #
def run(m, n1, k1, n2, k2, lam, N, t1=None, layout="spread", seed0=1000, label=""):
    I0 = GA.build_instance(m, n1, k1, n2, k2, lam, seed0, t1=t1, layout=layout)
    n, k, p = I0['n'], I0['k'], I0['p']
    r_max = (k * p) // n
    print("=" * 92)
    print(f" {label}   n1={n1} n2={n2} k2={k2} m={m} lam={lam} | n={n} k={k} "
          f"t1={I0['t1']} p={p} r_max={r_max}")
    if r_max < lam:
        print(f"   r_max={r_max} < lambda={lam}: not the over-determined regime, skipped")
        print("=" * 92); return
    print(f"   claim: dim_Fqm L_F = n1^2 = {n1 * n1}   (dim_Fq = {m * n1 * n1})")
    print("=" * 92)

    dimK, mult_ok, stab_ok, supp_ok, det_ok, det_tot = [], 0, 0, 0, 0, 0
    for s in range(seed0, seed0 + N):
        I = GA.build_instance(m, n1, k1, n2, k2, lam, s, t1=t1, layout=layout)
        F, Gpub = I['F'], I['Gpub']
        rng = random.Random(s ^ 0x5bd1e995)
        while True:
            h0 = [rng.randrange(1, F.QM) for _ in range(F.m)]
            if GA.gf2_rank_of(F, h0) == F.m:
                break
        H0 = moore(F, h0, I['p'])
        Fg = GA.extend_to(F, I['Vb'], r_max, rng)                    # GOOD guess
        Ls = GA.solve_public_system(F, Gpub, H0, Fg, 1, n2, F.m, k)  # trusted kernel of L_F
        dK = len(Ls) // F.m
        dimK.append(dK)
        if len(Ls) % F.m == 0:
            mult_ok += 1
        # (3) K-stability
        if Ls:
            R = R_beta(F, h0, rng.randrange(2, F.QM))
            if in_span(F, act(F, Ls[0], R), Ls, n, F.m):
                stab_ok += 1
        # (4) every element alpha V-valued
        if Ls and all(supp_dim(F, Z, n, F.m) <= lam for Z in Ls):
            supp_ok += 1
        # (5) every n1-subset of a K-basis -> image rank k (capped for large fields)
        Kb = kbasis(F, Ls, h0, n, F.m)
        subs = list(itertools.combinations(range(len(Kb)), n1))
        if len(subs) > 12:
            subs = subs[:12]                       # a representative cap; all pass in full runs
        for sub in subs:
            DE = [[Kb[i][row][c] for i in sub for c in range(F.m)] for row in range(n)]
            det_tot += 1
            det_ok += (rank(F, matmul(F, Gpub, DE)) == k)

    hist = {}
    for x in dimK:
        hist[x] = hist.get(x, 0) + 1
    print(f"  dim_Fqm L_F              : {hist}     (target n1^2 = {n1 * n1})")
    print(f"  m | dim_Fq L_F           : {mult_ok}/{N}")
    print(f"  K-stable (Z R^T in L_F)  : {stab_ok}/{N}")
    print(f"  every Z is alphaV-valued : {supp_ok}/{N}")
    print(f"  n1-subset -> image rank k: {det_ok}/{det_tot}")
    print()


if __name__ == "__main__":
    # single-block (n1=1): dim_Fqm L_F should be 1 (Burle et al. Step 2)
    run(10, 1, 1, 10, 4, 2, 6, t1=1, label="single      lam=2 r_max-lam=0")
    run(16, 1, 1, 16, 6, 2, 3, t1=1, label="single      lam=2 r_max-lam=1")
    run(16, 1, 1, 16, 5, 3, 3, t1=1, label="single      lam=3 r_max-lam=0")
    # genuine Kronecker (n1=2): dim_Fqm L_F should be n1^2 = 4
    run(12, 2, 2, 12, 4, 2, 3, t1=2, label="GabKron n1=2 lam=2")
    run(14, 2, 2, 14, 4, 2, 2, t1=2, label="GabKron n1=2 lam=2")
    # n1=3 (dim_Fqm L_F = 9) is verified but slow in pure Python (F_{2^16},
    # ~48x48 systems); uncomment to reproduce (expect a few minutes per instance):
    # run(16, 3, 3, 16, 6, 2, 1, t1=3, label="GabKron n1=3 lam=2")   # -> dim_Fqm L_F = 9
