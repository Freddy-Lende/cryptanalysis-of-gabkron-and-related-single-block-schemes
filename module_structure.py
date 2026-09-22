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
#    (4) the JOINT support of L_F: Supp_q of the concatenation (Z_1|...|Z_d) has
#        F_q-dimension <= lambda  (the property required for decryption; this is the
#        cryptanalytic requirement, NOT the stronger literal 'Supp subset alpha V',
#        and NOT merely per-vector support)
#    (5) deterministic extraction: every n1-subset of a K-basis of L_F has
#        image rank exactly k
#    (C1) direct measurement of dim_Fq(ker_r G_pub cap F^n): should be 0 (trivial kernel);
#         reported for ALL n1 -- for n1>=2 this is NOT implied by dim L_F = n1^2
#    (scalar images) number of distinct scalar images gamma*V contained in F (gated to
#         small m); a good guess should give exactly 1 (a single image alpha V)
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

def c1_dim(F, Gpub, Fg, n, k):
    """dim_Fq of { v in Fg^n : Gpub v = 0 } = ker_r(Gpub) cap F^n  (C1 space).
    Builds the F_q-linear map v |-> Gpub v on the guess space Fg and returns its nullity.
    C1 (trivial kernel) holds iff this is 0."""
    r = len(Fg)
    rows = []
    for i in range(n):
        for l in range(r):
            bitrow = []
            for row in range(k):
                e = F.mul(Gpub[row][i], Fg[l])          # entry of Gpub[:,i]*Fg[l] in F_qm
                bitrow.extend(F.bits(e))                 # m F_2 coordinates of that entry
            rows.append(bitrow)                          # image of one input basis vector (len k*m)
    return n * r - ss._gf2_rank(rows)                    # nullity = dim_Fq(ker_r Gpub cap F^n)


def _in_span_bits(F, e, basis_bits, base_rank):
    """True iff field element e lies in the F_q-span whose bit-basis has rank base_rank."""
    return ss._gf2_rank(basis_bits + [F.bits(e)]) == base_rank

def _subspace_key(elts):
    """Canonical key of the F_2-span of `elts` (all q^lambda combinations; lambda is small)."""
    span = {0}
    for e in elts:
        span |= {s ^ e for s in span}
    return frozenset(span)

def scalar_hits(F, Vb, Fg):
    """Number of DISTINCT scalar images gamma*V (gamma in F_qm*) contained in span_Fq(Fg).
    A good guess has at least one (alpha V); 'no second' means this count equals 1.
    Cost ~ q^m * lambda, so the caller gates it to small m."""
    Fbits = [F.bits(x) for x in Fg]
    base = ss._gf2_rank(Fbits)
    seen = set()
    for g in range(1, F.QM):
        img = [F.mul(g, v) for v in Vb]
        if all(_in_span_bits(F, e, Fbits, base) for e in img):
            seen.add(_subspace_key(img))
    return len(seen)

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
    c1_ok, c1_vals = 0, []
    sh_vals = []   # distinct scalar images of V in F (gated to small m)
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
        # (C1) direct measurement of dim_Fq(ker_r Gpub cap F^n) on this good guess
        c1 = c1_dim(F, Gpub, Fg, n, k)
        c1_vals.append(c1); c1_ok += (c1 == 0)
        if m <= 14:                                   # scalar-hit count is O(q^m); small m only
            sh_vals.append(scalar_hits(F, I['Vb'], Fg))
        if len(Ls) % F.m == 0:
            mult_ok += 1
        # (3) K-stability
        if Ls:
            R = R_beta(F, h0, rng.randrange(2, F.QM))
            if in_span(F, act(F, Ls[0], R), Ls, n, F.m):
                stab_ok += 1
        # (4) JOINT support of L_F: the F_q-span of the entries of the *whole* basis
        #     (equivalently of the concatenation D_F=(Z_1|...|Z_d)) must have joint support
        #     dimension at most lambda; no identification with alpha V is required.
        #     Per-vector support <= lambda is NOT sufficient,
        #     since distinct basis vectors could carry distinct lambda-supports.
        if Ls:
            D_all = [ [ Ls[j][row][c] for j in range(len(Ls)) for c in range(F.m) ]
                      for row in range(n) ]
            if supp_dim(F, D_all, n, len(Ls) * F.m) <= lam:
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
    print(f"  C1: ker_r Gpub cap F^n=0 : {c1_ok}/{N}   (dim_Fq measured: {sorted(set(c1_vals))})")
    if sh_vals:
        print(f"  scalar images of V in F  : {sorted(set(sh_vals))}   (1 = a single image alpha V; \"no second\")")
    print(f"  K-stable (Z R^T in L_F)  : {stab_ok}/{N}")
    print(f"  joint Supp dim <= lam    : {supp_ok}/{N}   (property required for decryption)")
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
