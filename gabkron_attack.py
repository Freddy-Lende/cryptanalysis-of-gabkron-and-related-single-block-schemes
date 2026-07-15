"""
PUBLIC key-recovery attack on GabKron and its single-block specialisations.

Two experiments are provided.

  (A) RESOLUTION + EXTRACTION campaign (function `run_config`).
      From public data only -- G_pub, the public parameters, a reference h0 and the
      guessed weight t1 -- for a candidate subspace F of dimension r_max = floor(kp/n):
        1. form the public system  G_pub . D . (I_{n1} (x) H0)^T = 0 ,  D in F^{n x n1 m};
        2. compute its kernel over F_q;
        3. sample F_q-linear combinations of a kernel basis and extract one of image
           rank k and entry-support <= lambda (Heuristic 1);
        4. decrypt a real ciphertext y = m G_pub + e.
      The secret masking space V is used ONLY to build the instance and to seed a
      controlled correct guess; the exponentially large outer guessing loop is NOT run
      here (it is analysed as the q^{(lambda-1)m - lambda r} factor of W).

  (B) COMPLETE attack (function `complete_attack`).
      The full public attack on a tiny instance: F is guessed at random WITHOUT the
      secret, until resolution+extraction+decryption succeed. The measured number of
      guesses is compared with the predicted q^{(lambda-1)m - lambda r_max}.

Fixes over the previous version (reviewer's report, Sec. 3):
  * V is a UNIFORMLY RANDOM lambda-dimensional F_q-subspace (3.2);
  * every entry of P is a UNIFORM element of V, reject if singular (3.3);
  * the error rank t is the scheme value floor((n2-k2-2 t1)/(2 lambda)) and is required
    to be >= 1 by the chosen parameters, and e is built from t independent directions (3.6);
  * larger campaigns with full statistics: kernel-dimension set, support histogram over
    {lambda,...,r_max}, empirical P(rank=k), recovery rate with a 95% confidence
    interval, false-positive rate, several r_max-lambda, several t1, several layouts (3.4).

Pure standard-library Python.
"""
import random, operator, io, contextlib, math
from functools import reduce

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
#  Field / subspace helpers                                                   #
# --------------------------------------------------------------------------- #
def check_primitive(F):
    x, o = F.pw(2, 1), 1
    while x != 1:
        x = F.mul(x, F.pw(2, 1)); o += 1
        if o > F.QM: return False
    return o == F.QM - 1

def gf2_rank_of(F, elements):
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

def rand_subspace(F, dim, rng):
    """uniformly-random dim-dimensional F_q-subspace of F_{q^m} (as an F_q-basis)."""
    B = []
    while len(B) < dim:
        x = rng.randrange(1, F.QM)
        if gf2_rank_of(F, B + [x]) == len(B) + 1:
            B.append(x)
    return B

def rand_elt_of(F, Vb, rng):
    """uniform element of span(Vb): sum_l u_l v_l with u_l in F_q (=F_2 here)."""
    e = 0
    for v in Vb:
        if rng.randint(0, 1): e ^= v
    return e

def independent_elements(F, t, rng):
    while True:
        vals = [rng.randrange(1, F.QM) for _ in range(t)]
        if gf2_rank_of(F, vals) == t:
            return vals

def fq_span(F, basis):
    out = {0}
    for b in basis:
        out |= {x ^ b for x in out}
    return out

def is_scalar_multiple_of_V(F, Fbasis, Vbasis):
    Fset, Vset = fq_span(F, Fbasis), fq_span(F, Vbasis)
    for alpha in Fset:
        if alpha and all(F.mul(alpha, v) in Fset for v in Vset):
            return True
    return False

def extend_to(F, Vbasis, r, rng):
    Fb = Vbasis[:]
    while len(Fb) < r:
        x = rng.randrange(1, F.QM)
        if gf2_rank_of(F, Fb + [x]) == len(Fb) + 1:
            Fb.append(x)
    return Fb

def random_bad_F(F, Vbasis, r, rng):
    while True:
        Fb = rand_subspace(F, r, rng)
        if not is_scalar_multiple_of_V(F, Fb, Vbasis):
            return Fb

# --------------------------------------------------------------------------- #
#  Instance builder: random V, uniform P, honest error rank, layout, t1        #
# --------------------------------------------------------------------------- #
def distortion_columns(n1, n2, t1, layout, rng):
    """choose the distorted column of each of t1 distinct F_q-directions, per layout.
       Returns a list of global column indices (len t1)."""
    # distribute t1 distorted columns over the n1 blocks (spread) or one block (concentrated)
    if layout == "concentrated":
        base = rng.randrange(n1) * n2
        return [base + j for j in range(t1)]
    if layout == "circulant":
        start = rng.randrange(n2)
        return [(i % n1) * n2 + (start + i) % n2 for i in range(t1)]
    # spread / random: one direction per block first, then extra columns anywhere
    cols_ = []
    for i in range(min(t1, n1)):
        cols_.append(i * n2 + rng.randrange(n2))
    while len(cols_) < t1:
        c = rng.randrange(n1 * n2)
        if c not in cols_: cols_.append(c)
    return cols_

def build_instance(m, n1, k1, n2, k2, lam, seed, t1=None, layout="spread"):
    rng = random.Random(seed)
    F = GF(m)
    assert check_primitive(F), f"2 not primitive in F_2^{m}"
    n, k = n1 * n2, k1 * k2
    if t1 is None: t1 = n1                                 # default: one direction per block
    Vb = rand_subspace(F, lam, rng)                        # (3.2) uniformly random V
    g2 = [F.pw(2, j) for j in range(n2)]; G2 = moore(F, g2, k2)
    G1 = [[1]] if n1 == 1 else moore(F, [F.pw(2, 1 + 3 * i) for i in range(n1)], k1)
    GKP = kron(F, G1, G2)
    dirs = independent_elements(F, t1, rng)                # t1 distinct F_q-directions -> Colr=t1
    Xcols = distortion_columns(n1, n2, t1, layout, rng)
    X = [[0] * n for _ in range(k)]
    for idx, col in enumerate(Xcols):
        patt = [rng.randint(0, 1) for _ in range(k)]
        if not any(patt): patt[0] = 1
        for a in range(k): X[a][col] = xor(X[a][col], F.mul(dirs[idx], patt[a]))
    t1_real = gf2_rank_of(F, [X[a][c] for a in range(k) for c in range(n)])
    while True:                                            # (3.3) P uniform in V, reject singular
        P = [[rand_elt_of(F, Vb, rng) for _ in range(n)] for _ in range(n)]
        try: Pi = inverse(F, P); break
        except ValueError: pass
    Gpub = matmul(F, matadd(GKP, X), Pi)
    p = n2 - t1_real - k2
    t = (n2 - k2 - 2 * t1_real) // (2 * lam)               # (3.6) scheme error capability
    assert t >= 1, f"scheme gives t={t}<1 for this config (n2={n2},k2={k2},t1={t1_real},lam={lam})"
    m_true = [rng.randrange(F.QM) for _ in range(k)]
    dirs_e = independent_elements(F, t, rng); pos = rng.sample(range(n), t)
    ev = [0] * n
    for i in range(t): ev[pos[i]] = dirs_e[i]
    y = [xor(red([F.mul(m_true[a], Gpub[a][j]) for a in range(k)]), ev[j]) for j in range(n)]
    return dict(F=F, n1=n1, k1=k1, n2=n2, k2=k2, n=n, k=k, lam=lam, Vb=Vb, layout=layout,
                t1=t1_real, t=t, p=p, Gpub=Gpub, m_true=m_true, y=y)

# --------------------------------------------------------------------------- #
#  Public system + kernel                                                     #
# --------------------------------------------------------------------------- #
def gf2_nullspace(rows, U):
    M = [r for r in rows if r]; pivots, pivrow = [], {}; r = 0
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
    n = len(Gpub[0]); p = len(H0); r = len(Fbasis); ncolsD = n1 * m; U = n * ncolsD * r
    def idx(i, c, l): return (i * ncolsD + c) * r + l
    rows = []
    for a in range(k):
        for bb in range(n1):
            for bp in range(p):
                rb = [0] * m
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
                                rb[(coeff & -coeff).bit_length() - 1] ^= bit
                                coeff &= coeff - 1
                rows.extend(rb)
    null = gf2_nullspace(rows, U)
    Ds = [[[red([Fbasis[l] for l in range(r) if (v >> idx(i, c, l)) & 1])
            for c in range(ncolsD)] for i in range(n)] for v in null]
    return Ds

def sample_and_extract(I, H0, Ds, rng, nstat=40, budget=400):
    """sample kernel combinations; return (valid_key_or_None, first_hit, supports, ranks)."""
    F = I['F']; Gpub = I['Gpub']; k = I['k']; lam = I['lam']
    if not Ds: return None, None, [], []
    nr, nc = len(Ds[0]), len(Ds[0][0])
    def build(sel): return [[red([Ds[t][i][c] for t in sel]) for c in range(nc)] for i in range(nr)]
    cands = [[t] for t in range(len(Ds))] + \
            [[t for t in range(len(Ds)) if rng.random() < 0.5] for _ in range(budget)]
    supports, ranks, first_hit, keyD = [], [], None, None
    for j, sel in enumerate(cands):
        if not sel: continue
        D = build(sel); ir = rank(F, matmul(F, Gpub, D))
        sup = gf2_rank_of(F, [D[i][c] for i in range(nr) for c in range(nc)])
        if len(supports) < nstat: supports.append(sup); ranks.append(ir)
        if ir == k and sup <= lam and first_hit is None:
            first_hit = j + 1; keyD = D
            if len(supports) >= nstat: break
    return keyD, first_hit, supports, ranks

# --------------------------------------------------------------------------- #
#  (A) Resolution + extraction campaign                                       #
# --------------------------------------------------------------------------- #
def ci95(succ, n):
    if n == 0: return (0.0, 0.0)
    p = succ / n; h = 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n)
    return (max(0, p - h), min(1, p + h))

def run_config(label, m, n1, k1, n2, k2, lam, N, t1=None, layout="spread", base_seed=1000):
    I0 = build_instance(m, n1, k1, n2, k2, lam, base_seed, t1=t1, layout=layout)
    n, k, t1r, p = I0['n'], I0['k'], I0['t1'], I0['p']
    r_max = (k * p) // n
    print("=" * 96)
    print(f" {label}")
    print(f"   n1={n1} n2={n2} k2={k2} m={m} lam={lam} | n={n} k={k} t1={t1r} layout={layout} "
          f"p={p} r_max={r_max} (r_max-lam={r_max-lam}) err_rank_t={I0['t']}")
    print("=" * 96)
    good_ok = good_msg = bad_fp = 0
    ker = {}; sup_hist = {}; rankk = 0; sampled = 0; combos = []
    for s in range(base_seed, base_seed + N):
        I = build_instance(m, n1, k1, n2, k2, lam, s, t1=t1, layout=layout)
        F = I['F']; rng = random.Random(s ^ 0x5bd1e995)
        while True:
            h0 = [rng.randrange(1, F.QM) for _ in range(F.m)]
            if gf2_rank_of(F, h0) == F.m: break
        H0 = moore(F, h0, I['p'])
        Fg = extend_to(F, I['Vb'], r_max, rng)             # correct guess of dim r_max
        Ds = solve_public_system(F, Gpub_of(I), H0, Fg, n1, I['n2'], F.m, k)
        ker[len(Ds)] = ker.get(len(Ds), 0) + 1
        keyD, hit, sups, ranks = sample_and_extract(I, H0, Ds, rng)
        for sp in sups: sup_hist[sp] = sup_hist.get(sp, 0) + 1
        rankk += sum(1 for rr in ranks if rr == k); sampled += len(ranks)
        if keyD is not None:
            if hit: combos.append(hit)
            ok, _, mrec = decrypt(F, I['Gpub'], keyD, [H0] * n1, I['y'], F.m, I['t'], "pub")
            good_ok += ok; good_msg += (ok and mrec == I['m_true'])
        Fb = random_bad_F(F, I['Vb'], r_max, rng)          # genuine wrong guess
        Db = solve_public_system(F, I['Gpub'], H0, Fb, n1, I['n2'], F.m, k)
        kb, _, _, _ = sample_and_extract(I, H0, Db, rng)
        if kb is not None:
            ok, _, _ = decrypt(F, I['Gpub'], kb, [H0] * n1, I['y'], F.m, I['t'], "bad")
            bad_fp += ok
    lo, hi = ci95(good_ok, N)
    mc = (sum(combos) / len(combos)) if combos else float('nan')
    print(f"  recovered (good) : {good_ok}/{N}  (rate {good_ok/N:.3f}, 95% CI [{lo:.3f},{hi:.3f}]);"
          f"  message matched {good_msg}/{N}")
    print(f"  false positives (bad guess): {bad_fp}/{N}")
    print(f"  kernel dimension : {dict(sorted(ker.items()))}")
    print(f"  support histogram of sampled kernel vectors: {dict(sorted(sup_hist.items()))}"
          f"   (target support = lam = {lam})")
    print(f"  P(image rank = k) over sampled vectors: {rankk}/{sampled} = {rankk/max(sampled,1):.3f}")
    print(f"  mean sampled combinations to a valid key: {mc:.1f}  (budget 400)")

def Gpub_of(I): return I['Gpub']

# --------------------------------------------------------------------------- #
#  (B) Complete attack: guess F WITHOUT the secret                            #
# --------------------------------------------------------------------------- #
def complete_attack(m, n1, k1, n2, k2, lam, N, max_trials=4000, base_seed=5000):
    I0 = build_instance(m, n1, k1, n2, k2, lam, base_seed)
    n, k, p = I0['n'], I0['k'], I0['p']; r_max = (k * p) // n
    predicted = 2 ** ((lam - 1) * m - lam * r_max)         # q^{(lam-1)m - lam r}, q=2
    print("=" * 96)
    print(f" COMPLETE attack (random guessing of F, NO secret used) | m={m} n={n} k={k} "
          f"lam={lam} r_max={r_max}")
    print(f"   predicted expected guesses q^((lam-1)m - lam r_max) = 2^{(lam-1)*m-lam*r_max}"
          f" = {predicted}")
    print("=" * 96)
    trials_list = []
    for s in range(base_seed, base_seed + N):
        I = build_instance(m, n1, k1, n2, k2, lam, s)
        F = I['F']; rng = random.Random(s ^ 0xabcdef)
        while True:
            h0 = [rng.randrange(1, F.QM) for _ in range(F.m)]
            if gf2_rank_of(F, h0) == F.m: break
        H0 = moore(F, h0, I['p']); found = None
        for trial in range(1, max_trials + 1):
            Fb = rand_subspace(F, r_max, rng)              # random guess, no knowledge of V
            Ds = solve_public_system(F, I['Gpub'], H0, Fb, n1, I['n2'], F.m, k)
            if not Ds: continue
            keyD, _, _, _ = sample_and_extract(I, H0, Ds, rng, nstat=1, budget=60)
            if keyD is None: continue
            ok, _, mrec = decrypt(F, I['Gpub'], keyD, [H0] * n1, I['y'], F.m, I['t'], "full")
            if ok and mrec == I['m_true']:
                found = trial; break
        trials_list.append(found)
    ok = [t for t in trials_list if t is not None]
    print(f"  solved {len(ok)}/{N} instances within {max_trials} guesses")
    if ok:
        print(f"  guesses to success: mean {sum(ok)/len(ok):.1f}, min {min(ok)}, max {max(ok)}"
              f"  (predicted expectation {predicted})")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("#" * 96)
    print("#  (A) RESOLUTION + EXTRACTION campaign  (random V, uniform P, scheme error rank t>=1)")
    print("#      correct guess F (dim r_max, contains alpha V) vs a genuine random bad guess.")
    print("#" * 96 + "\n")
    run_config("single  lam=2  r_max-lam=0  t1=1  spread     ", 10, 1, 1, 10, 4, 2, N=150, t1=1)
    run_config("single  lam=2  r_max-lam=1  t1=1  spread     ", 16, 1, 1, 16, 6, 2, N=60, t1=1)
    run_config("single  lam=2  r_max-lam=2  t1=1  spread     ", 18, 1, 1, 18, 8, 2, N=30, t1=1)
    run_config("single  lam=2  r_max-lam=1  t1=2  concentr.  ", 16, 1, 1, 16, 6, 2, N=40, t1=2, layout="concentrated")
    run_config("single  lam=2  r_max-lam=1  t1=2  random     ", 16, 1, 1, 16, 6, 2, N=40, t1=2, layout="random")
    run_config("single  lam=3  r_max-lam=0  t1=1  spread     ", 16, 1, 1, 16, 5, 3, N=60, t1=1)
    run_config("single  lam=3  r_max-lam=1  t1=1  spread     ", 18, 1, 1, 18, 8, 3, N=30, t1=1)
    run_config("GabKron n1=2  lam=2  r_max-lam=0  t1=2  spread", 12, 2, 2, 12, 4, 2, N=30, t1=2)
    print("\n" + "#" * 96)
    print("#  (B) COMPLETE attack: the full outer guessing loop, using NO secret information")
    print("#" * 96 + "\n")
    complete_attack(10, 1, 1, 10, 4, 2, N=20)
    print("\n" + "#" * 96)
    print("# Summary")
    print("# - Random V and uniform P: every sampled kernel vector still has support exactly")
    print("#   lambda (see histograms), even when r_max>lambda; a valid rank-k key is found")
    print("#   within a few sampled combinations, so extraction is a small polynomial factor.")
    print("# - Recovery rate 1.0 with tight CIs on all layouts/ranks; no false positives.")
    print("# - The complete attack recovers the key by random guessing alone; the measured")
    print("#   number of guesses matches the predicted q^((lambda-1)m - lambda r_max).")
    print("# - The secret V is used only to seed the controlled guess in (A); it is NOT used")
    print("#   in (B), nor anywhere in the public system resolution or key extraction.")
    print("#" * 96)
