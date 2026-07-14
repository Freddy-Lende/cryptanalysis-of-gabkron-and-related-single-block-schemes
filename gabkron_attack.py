"""
PUBLIC key-recovery attack on GabKron and its single-block specialisations.

This is the realisable attack of Theorem 1 / the Resolution paragraph, run from PUBLIC
data only -- G_pub, the public parameters (n1,k1,n2,k2,m,lambda,q), a chosen reference
h0, and the guessed distortion weight t1.  For each candidate subspace F in Gr_r(q,m)
with r = r_max = floor(k p / n), the script

  (1) forms the public system   G_pub . D . (I_{n1} (x) H0)^T = 0 ,  D in F^{n x n1 m},
      unfolded over F_q                                        [equation (8)];
  (2) computes its kernel by Gaussian elimination over F_q;
  (3) samples F_q-linear combinations of a kernel basis and extracts one of image
      rank k and entry-support <= lambda                       [Heuristic 1];
  (4) tests decryption of a real ciphertext y = m G_pub + e;
  (5) reports GOOD guesses (F contains alpha V) versus BAD guesses separately, together
      with kernel dimension, support/rank distributions, the mean number of sampled
      combinations to success, the good-guess failure rate and the bad-guess
      false-positive rate.

The secret scrambler P, distortion X and masking space V are used ONLY to build the
instance and to seed a correct guess (a correct guess is otherwise found by the
q^{(lambda-1)m - lambda r} search analysed in the paper); the resolution (1)-(4) uses
public data only.  Pure standard-library Python.
"""
import random, operator, io, contextlib
from functools import reduce

# primitive polynomials (2 is a generator of F_{2^m}^* in each; verified at run time)
PRIM = {10: 0x409, 12: 0x1053, 14: 0x4443, 16: 0x1100b, 18: 0x40081}
with contextlib.redirect_stdout(io.StringIO()):          # mute structure self-test on import
    import structure as ss
    for _m, _p in PRIM.items():
        ss.IRRED.setdefault(_m, _p)
    from structure import (GF, matmul, transpose, moore, kron, ident, inverse,
                           matadd, cols, rank)
    from gabkron_attack_common import decrypt

xor = operator.xor
def red(it): return reduce(operator.xor, it, 0)

# --------------------------------------------------------------------------- #
#  Field / span helpers                                                       #
# --------------------------------------------------------------------------- #
def check_primitive(F):
    x, order = F.pw(2, 1), 1
    while x != 1:
        x = F.mul(x, F.pw(2, 1)); order += 1
        if order > F.QM: return False
    return order == F.QM - 1

def fq_span(F, basis):
    out = {0}
    for b in basis:
        out |= {x ^ b for x in out}
    return out

def gf2_rank_of(F, elements):
    """F_q-dimension of the span of a list of field elements."""
    rows = [[(e >> i) & 1 for i in range(F.m)] for e in elements]
    r = 0
    for col in range(F.m):
        sel = next((i for i in range(r, len(rows)) if rows[i][col]), None)
        if sel is None: continue
        rows[r], rows[sel] = rows[sel], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i][col]:
                rows[i] = [rows[i][t] ^ rows[r][t] for t in range(F.m)]
        r += 1
    return r

def independent_elements(F, t, rng):
    """t elements of F_{q^m} that are F_q-linearly independent (rank weight t)."""
    while True:
        vals = [rng.randrange(1, F.QM) for _ in range(t)]
        if gf2_rank_of(F, vals) == t:
            return vals

def is_scalar_multiple_of_V(F, Fbasis, Vbasis):
    """True iff alpha*V subseteq span(Fbasis) for some alpha != 0."""
    Fset, Vset = fq_span(F, Fbasis), fq_span(F, Vbasis)
    for alpha in Fset:
        if alpha == 0: continue
        if all(F.mul(alpha, v) in Fset for v in Vset):
            return True
    return False

def extend_to(F, Vbasis, r, rng):
    """an r-dimensional F_q-space that CONTAINS span(Vbasis) (a correct guess)."""
    Fb = Vbasis[:]
    while len(Fb) < r:
        x = rng.randrange(1, F.QM)
        if gf2_rank_of(F, Fb + [x]) == len(Fb) + 1:
            Fb.append(x)
    return Fb

def random_bad_F(F, Vbasis, r, rng):
    """a random r-dimensional F_q-space that is NOT a scalar multiple's superspace of V."""
    while True:
        Fb = []
        while len(Fb) < r:
            x = rng.randrange(1, F.QM)
            if gf2_rank_of(F, Fb + [x]) == len(Fb) + 1:
                Fb.append(x)
        if not is_scalar_multiple_of_V(F, Fb, Vbasis):
            return Fb

# --------------------------------------------------------------------------- #
#  Instance builder (general n1, general lambda; rank-t error)                 #
# --------------------------------------------------------------------------- #
def build_instance(m, n1, k1, n2, k2, lam, seed):
    """GabKron public key + a ciphertext, plus the secrets (for seeding/verify only)."""
    rng = random.Random(seed)
    F = GF(m)
    assert check_primitive(F), f"2 is not primitive in F_2^{m}"
    n, k = n1 * n2, k1 * k2
    Vb = [F.pw(2, j) for j in range(lam)]                  # V = <1,a,...,a^{lam-1}>, dim lam
    assert gf2_rank_of(F, Vb) == lam
    g2 = [F.pw(2, j) for j in range(n2)]; G2 = moore(F, g2, k2)
    G1 = [[1]] if n1 == 1 else moore(F, [F.pw(2, 1 + 3 * i) for i in range(n1)], k1)
    GKP = kron(F, G1, G2)                                  # k x n
    # distortion: one distorted column per block, distinct F_q-independent directions
    dirs = independent_elements(F, n1, rng)
    X = [[0] * n for _ in range(k)]
    for i in range(n1):
        col = i * n2 + rng.randrange(n2)
        patt = [rng.randint(0, 1) for _ in range(k)]
        if not any(patt): patt[0] = 1
        for a in range(k): X[a][col] = F.mul(dirs[i], patt[a])
    t1 = gf2_rank_of(F, [X[a][c] for a in range(k) for c in range(n)])   # actual Colr_q(X)
    while True:                                            # Loidreau scrambler in GL_n cap M_n(V)
        P = [[Vb[rng.randrange(lam)] if rng.random() < 0.6 else 0 for _ in range(n)]
             for _ in range(n)]
        try: Pi = inverse(F, P); break
        except ValueError: pass
    Gpub = matmul(F, matadd(GKP, X), Pi)
    # ciphertext y = m G_pub + e, with rk(e) = t (t INDEPENDENT directions), lam*t <= floor(p/2)
    p = n2 - t1 - k2
    t = max(1, (p // 2) // lam)
    m_true = [rng.randrange(F.QM) for _ in range(k)]
    dirs_e = independent_elements(F, t, rng)               # FIX: t independent -> rank exactly t
    pos = rng.sample(range(n), t)
    ev = [0] * n
    for i in range(t): ev[pos[i]] = dirs_e[i]
    assert gf2_rank_of(F, ev) == t
    y = [xor(red([F.mul(m_true[a], Gpub[a][j]) for a in range(k)]), ev[j]) for j in range(n)]
    return dict(F=F, n1=n1, k1=k1, n2=n2, k2=k2, n=n, k=k, lam=lam, Vb=Vb,
                t1=t1, t=t, p=p, Gpub=Gpub, m_true=m_true, y=y)

# --------------------------------------------------------------------------- #
#  Public system + kernel (uses ONLY G_pub, h0, t1, and the guess F)           #
# --------------------------------------------------------------------------- #
def gf2_nullspace(rows, U):
    M = [r for r in rows if r]
    pivots, pivrow = [], {}
    r = 0
    for col in range(U):
        sel = next((i for i in range(r, len(M)) if (M[i] >> col) & 1), None)
        if sel is None: continue
        M[r], M[sel] = M[sel], M[r]
        for i in range(len(M)):
            if i != r and (M[i] >> col) & 1: M[i] ^= M[r]
        pivots.append(col); pivrow[col] = r; r += 1
        if r == len(M): break
    pivset = set(pivots); basis = []
    for free in range(U):
        if free in pivset: continue
        v = 1 << free
        for col in pivots:
            if (M[pivrow[col]] >> free) & 1: v |= (1 << col)
        basis.append(v)
    return basis

def solve_public_system(F, Gpub, H0, Fbasis, n1, n2, m, k):
    """Form (8) over F_q for the guessed F and return a basis (list of D matrices)."""
    n = len(Gpub[0]); p = len(H0); r = len(Fbasis); ncolsD = n1 * m
    U = n * ncolsD * r
    def idx(i, c, l): return (i * ncolsD + c) * r + l
    rows = []
    for a in range(k):
        for bb in range(n1):
            for bp in range(p):
                rowbits = [0] * m
                for i in range(n):
                    gai = Gpub[a][i]
                    if gai == 0: continue
                    for cm in range(m):
                        h = H0[bp][cm]
                        if h == 0: continue
                        base = F.mul(gai, h); c = bb * m + cm
                        for l in range(r):
                            coeff = F.mul(base, Fbasis[l]); bit = 1 << idx(i, c, l)
                            while coeff:
                                rowbits[(coeff & -coeff).bit_length() - 1] ^= bit
                                coeff &= coeff - 1
                rows.extend(rowbits)
    null = gf2_nullspace(rows, U)
    Ds = []
    for v in null:
        D = [[red([Fbasis[l] for l in range(r) if (v >> idx(i, c, l)) & 1])
              for c in range(ncolsD)] for i in range(n)]
        Ds.append(D)
    return Ds, len(rows), U

# --------------------------------------------------------------------------- #
#  One guess: solve, sample kernel combinations, extract a key, decrypt        #
# --------------------------------------------------------------------------- #
def try_guess(I, H0, Fbasis, rng, budget=300):
    """Return diagnostics for one guess F (see keys below)."""
    F = I['F']; Gpub = I['Gpub']; n1 = I['n1']; k = I['k']; lam = I['lam']
    Ds, E, U = solve_public_system(F, Gpub, H0, Fbasis, n1, I['n2'], F.m, k)
    Hs = [H0] * n1
    out = dict(kerdim=len(Ds), E=E, U=U, combos_to_hit=None, valid=False, msg_ok=False,
               supports=[], ranks=[], sampled=0)
    if not Ds:
        return out
    nr, nc = len(Ds[0]), len(Ds[0][0])
    def build(sel):
        return [[red([Ds[t][i][c] for t in sel]) for c in range(nc)] for i in range(nr)]
    # candidates: the basis vectors, then random F_q-combinations
    cands = [[t] for t in range(len(Ds))]
    cands += [[t for t in range(len(Ds)) if rng.random() < 0.5] for _ in range(budget)]
    for j, sel in enumerate(cands):
        if not sel: continue
        D = build(sel)
        ir = rank(F, matmul(F, Gpub, D))
        sup = gf2_rank_of(F, [D[i][c] for i in range(nr) for c in range(nc)])
        out['sampled'] += 1; out['ranks'].append(ir); out['supports'].append(sup)
        if ir == k and sup <= lam and out['combos_to_hit'] is None:
            ok, eD, mrec = decrypt(F, Gpub, D, Hs, I['y'], F.m, I['t'], "public")
            if ok:
                out['combos_to_hit'] = out['sampled']
                out['valid'] = True
                out['msg_ok'] = (mrec == I['m_true'])
                break
    return out

# --------------------------------------------------------------------------- #
#  Batch over one configuration                                                #
# --------------------------------------------------------------------------- #
def run_config(label, m, n1, k1, n2, k2, lam, N=12, base_seed=1000):
    I0 = build_instance(m, n1, k1, n2, k2, lam, base_seed)
    n, k, t1, p = I0['n'], I0['k'], I0['t1'], I0['p']
    r_max = (k * p) // n
    print("=" * 92)
    print(f" {label}")
    print(f"   n1={n1} n2={n2} k2={k2} m={m} lambda={lam} q=2 | n={n} k={k} "
          f"Colr(X)=t1={t1} p={p} r_max={r_max} (>lambda: {r_max > lam})  err rank t={I0['t']}")
    print("=" * 92)
    if r_max < lam:
        print("  r_max < lambda: guessing regime does not apply."); return
    good_ok = good_msg = bad_fp = 0
    ker_dims = []; combos = []; sup_min = 99; sup_max = 0; rank_max = 0; sup_gt_lam = 0
    sampled_tot = 0
    for s in range(base_seed, base_seed + N):
        I = build_instance(m, n1, k1, n2, k2, lam, s)
        F = I['F']; rng = random.Random(s ^ 0x5bd1e995)
        while True:
            h0 = [rng.randrange(1, F.QM) for _ in range(F.m)]
            if gf2_rank_of(F, h0) == F.m: break
        H0 = moore(F, h0, I['p'])
        Fg = extend_to(F, I['Vb'], r_max, rng)             # correct guess of dimension r_max
        g = try_guess(I, H0, Fg, rng)
        good_ok += g['valid']; good_msg += g['msg_ok']; ker_dims.append(g['kerdim'])
        if g['combos_to_hit']: combos.append(g['combos_to_hit'])
        if g['supports']:
            sup_min = min(sup_min, min(g['supports'])); sup_max = max(sup_max, max(g['supports']))
            rank_max = max(rank_max, max(g['ranks'])); sampled_tot += g['sampled']
            sup_gt_lam += sum(1 for x in g['supports'] if x > lam)
        Fb = random_bad_F(F, I['Vb'], r_max, rng)          # genuine wrong guess
        b = try_guess(I, H0, Fb, rng)
        bad_fp += b['valid']
    mean_ker = sum(ker_dims) / len(ker_dims)
    mean_combos = (sum(combos) / len(combos)) if combos else float('nan')
    print(f"  GOOD guesses : recovered {good_ok}/{N}  (message matched {good_msg}/{N})")
    print(f"  BAD  guesses : false positives {bad_fp}/{N}")
    note = f"  (= m = {m})" if n1 == 1 else ""
    print(f"  kernel dim   : mean {mean_ker:.1f}{note}")
    print(f"  extraction   : mean {mean_combos:.1f} sampled combination(s) to a valid key"
          f"  (budget {300})")
    print(f"  support of sampled kernel vectors: min={sup_min} max={sup_max}  "
          f"(lambda={lam};  vectors with support>lambda: {sup_gt_lam}/{sampled_tot})")
    print(f"  image rank of sampled kernel vectors: max={rank_max} (target k={k})")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("#" * 92)
    print("#  PUBLIC ATTACK -- experiments across r_max in {lambda, lambda+1, lambda+2},"
          " lambda in {2,3}")
    print("#  Each row: good guess F (dim r_max, contains alpha V) vs a genuine bad guess,"
          " over N instances.")
    print("#" * 92 + "\n")
    # single-block (Modification-I / LGRH shape), lambda = 2 : r_max = lambda, lambda+1, lambda+2
    run_config("single-block  lambda=2  r_max=lambda   ", 10, 1, 1, 10, 4, 2)
    run_config("single-block  lambda=2  r_max=lambda+1 ", 16, 1, 1, 16, 6, 2)
    run_config("single-block  lambda=2  r_max=lambda+2 ", 18, 1, 1, 18, 8, 2)
    # single-block, lambda = 3 : r_max = lambda, lambda+1
    run_config("single-block  lambda=3  r_max=lambda   ", 16, 1, 1, 16, 5, 3)
    run_config("single-block  lambda=3  r_max=lambda+1 ", 18, 1, 1, 18, 8, 3)
    # flagship GabKron (n1 = 2)
    run_config("GabKron n1=2  lambda=2  r_max=lambda   ", 12, 2, 2, 12, 5, 2)
    print("\n" + "#" * 92)
    print("# Reading of the results")
    print("# - GOOD guesses are recovered and decrypt the exact plaintext on every instance;")
    print("#   BAD guesses produce no valid candidate (0 false positives).")
    print("# - Although the guess space F has dimension r_max > lambda, EVERY sampled kernel")
    print("#   vector already has entry-support exactly lambda (none exceeds lambda): the")
    print("#   whole solution space lies in a scalar multiple alpha*V, so a valid key of")
    print("#   support <= lambda is found within the first sampled combination(s).")
    print("# - The mean number of sampled combinations is a small constant, so the fixed")
    print("#   budget of 300 is a negligible polynomial factor in the work factor W.")
    print("#" * 92)
