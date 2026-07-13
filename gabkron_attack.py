"""
PUBLIC key-recovery attack on GabKron and its single-block specialisations.

This is the realisable attack of Theorem 1 / the Resolution paragraph. In contrast
with the structural witness (which uses the secret P, X to *build* the key), here the
alternative key is recovered from PUBLIC data only:
    G_pub, the public parameters (n1,k1,n2,k2,m,lambda,q), a chosen reference h0,
    and the guessed distortion weight t1.

For each candidate subspace F in Gr_r(q,m) with r = r_max = floor(k p / n), the script

  (1) forms the public system   G_pub . D . (I_{n1} (x) H0)^T = 0 ,  D in F^{n x n1 m},
      unfolded over F_q                                        [equation (8)];
  (2) computes its kernel by Gaussian elimination over F_q     [step (3) of the review];
  (3) extracts a non-zero solution D, tests entry-support <= lambda and image rank k
                                                               [step (4)];
  (4) tests decryption of a real ciphertext y = m G_pub + e    [step (5)];
  (5) reports GOOD guesses (F = alpha V) versus BAD guesses    [step (6)].

The secret scrambler P, distortion X and masking space V are used ONLY to build the
instance and to seed a correct guess (a correct guess is otherwise found by the
q^{(lambda-1)m - lambda r} search analysed in the paper); the resolution (1)-(4) uses
public data only. Pure standard-library Python.
"""
import random, operator, io, contextlib
from functools import reduce

import structure as ss
# seed primitive polynomials for the fields we use (2 is a generator in each)
ss.IRRED.setdefault(10, 0b10000001001)          # x^10 + x^3 + 1
ss.IRRED.setdefault(12, 0b1000001010011)         # x^12 + x^6 + x^4 + x + 1
with contextlib.redirect_stdout(io.StringIO()):  # mute structure self-test
    from structure import (GF, matmul, transpose, moore, kron, ident, inverse,
                                matadd, cols, rank, right_kernel)
    from gabkron_attack_common import decrypt

xor = operator.xor
def red(it): return reduce(operator.xor, it, 0)

# --------------------------------------------------------------------------- #
#  Field / span helpers                                                       #
# --------------------------------------------------------------------------- #
def check_primitive(F):
    """2 must be a generator of F_{q^m}^*: order = q^m - 1."""
    x, order = F.pw(2, 1), 1
    while x != 1:
        x = F.mul(x, F.pw(2, 1)); order += 1
        if order > F.QM: return False
    return order == F.QM - 1

def fq_span(F, basis):
    """the 2^len(basis) elements of the F_q-span of `basis` (q = 2)."""
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

def is_scalar_multiple_of_V(F, Fbasis, Vbasis):
    """True iff alpha*V subseteq span(Fbasis) for some alpha != 0 (dims equal => = )."""
    Fset, Vset = fq_span(F, Fbasis), fq_span(F, Vbasis)
    for alpha in Fset:
        if alpha == 0: continue
        if all(F.mul(alpha, v) in Fset for v in Vset):
            return True
    return False

# --------------------------------------------------------------------------- #
#  Instance builder (general n1); controlled column rank t1                    #
# --------------------------------------------------------------------------- #
def build_instance(m, n1, k1, n2, k2, lam, seed):
    """Return a GabKron public key + a ciphertext, plus the secrets (for seeding/verify)."""
    rng = random.Random(seed)
    F = GF(m)
    assert check_primitive(F), f"2 is not primitive in F_2^{m}"
    n, k = n1 * n2, k1 * k2
    Vb = [1, F.pw(2, 1)][:lam]                              # V = <1, a, ...>_{F_q}, dim lam
    g2 = [F.pw(2, j) for j in range(n2)]; G2 = moore(F, g2, k2)
    if n1 == 1:
        G1 = [[1]]
    else:
        g1 = [F.pw(2, 1 + 3 * i) for i in range(n1)]; G1 = moore(F, g1, k1)
    GKP = kron(F, G1, G2)                                   # k x n
    # distortion: one distorted column per block, distinct F_q-independent directions
    dirs = [F.pw(2, 2 + 5 * i) for i in range(n1)]          # n1 independent directions
    X = [[0] * n for _ in range(k)]
    for i in range(n1):
        col = i * n2 + rng.randrange(n2)                    # distorted column of block i
        patt = [rng.randint(0, 1) for _ in range(k)]
        if not any(patt): patt[0] = 1
        for a in range(k): X[a][col] = F.mul(dirs[i], patt[a])
    t1 = gf2_rank_of(F, [X[a][c] for a in range(k) for c in range(n)])  # actual Colr_q(X)
    # Loidreau scrambler P in GL_n cap M_n(V)
    while True:
        P = [[Vb[rng.randrange(lam)] if rng.random() < 0.6 else 0 for _ in range(n)]
             for _ in range(n)]
        try: Pi = inverse(F, P); break
        except ValueError: pass
    Gpub = matmul(F, matadd(GKP, X), Pi)
    # ciphertext y = m G_pub + e, rk(e) = t, with lam*t <= floor(p/2)
    p = n2 - t1 - k2
    t = max(1, (p // 2) // lam)
    m_true = [rng.randrange(F.QM) for _ in range(k)]
    while True:
        cdir = rng.randrange(1, F.QM); pos = rng.sample(range(n), t)
        ev = [cdir if j in pos else 0 for j in range(n)]
        if gf2_rank_of(F, ev) == t: break
    y = [xor(red([F.mul(m_true[a], Gpub[a][j]) for a in range(k)]), ev[j]) for j in range(n)]
    return dict(F=F, n1=n1, k1=k1, n2=n2, k2=k2, n=n, k=k, lam=lam, Vb=Vb,
                t1=t1, t=t, p=p, Gpub=Gpub, m_true=m_true, y=y)

# --------------------------------------------------------------------------- #
#  Public system + kernel (uses ONLY G_pub, h0, t1, and the guess F)           #
# --------------------------------------------------------------------------- #
def gf2_nullspace(rows, U):
    """basis of {x in F_2^U : A x = 0}, rows given as U-bit integer masks."""
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
        for col in pivots:                       # read fully-reduced pivot rows
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
                        base = F.mul(gai, h)
                        c = bb * m + cm
                        for l in range(r):
                            coeff = F.mul(base, Fbasis[l])
                            bit = 1 << idx(i, c, l)
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
#  One guess: solve, extract a valid key, decrypt                              #
# --------------------------------------------------------------------------- #
def try_guess(I, H0, Fbasis, rng, combo_budget=300):
    F = I['F']; Gpub = I['Gpub']; n1 = I['n1']; k = I['k']
    Ds, E, U = solve_public_system(F, Gpub, H0, Fbasis, n1, I['n2'], F.m, k)
    Hs = [H0] * n1
    best = dict(kerdim=len(Ds), E=E, U=U, imgrank=0, support=None,
                decrypt=False, msg_ok=False)
    if not Ds:
        return best
    nr, nc = len(Ds[0]), len(Ds[0][0])
    def build(sel):                              # F_q-combination of kernel basis vectors
        return [[red([Ds[t][i][c] for t in sel]) for c in range(nc)] for i in range(nr)]
    # every kernel element already has entries in F (support <= r = lambda); we only
    # need image rank k. Try the basis vectors first, then random combinations.
    cands = ([[t] for t in range(len(Ds))] +
             [[t for t in range(len(Ds)) if rng.random() < 0.5] for _ in range(combo_budget)])
    for sel in cands:
        if not sel: continue
        D = build(sel)
        ir = rank(F, matmul(F, Gpub, D))
        if ir > best['imgrank']: best['imgrank'] = ir
        if ir == k:
            sup = gf2_rank_of(F, [D[i][c] for i in range(nr) for c in range(nc)])
            best['support'] = sup
            if sup <= I['lam']:
                ok, eD, mrec = decrypt(F, Gpub, D, Hs, I['y'], F.m, I['t'], "public")
                if ok:
                    best['decrypt'] = True
                    best['msg_ok'] = (mrec == I['m_true'])
                    return best
    return best

# --------------------------------------------------------------------------- #
#  Driver                                                                      #
# --------------------------------------------------------------------------- #
def run(m, n1, k1, n2, k2, lam, seed, n_bad=4):
    I = build_instance(m, n1, k1, n2, k2, lam, seed)
    F = I['F']; n, k, t1, p = I['n'], I['k'], I['t1'], I['p']
    r_max = (k * p) // n
    print("=" * 78)
    print(f" PUBLIC ATTACK  |  n1={n1} k1={k1} n2={n2} k2={k2} m={m} lambda={lam} q=2")
    print(f"   n={n} k={k}  Colr_q(X)=t1={t1}  p=n2-t1-k2={p}  r_max=floor(kp/n)={r_max}"
          f"  (>= lambda: {r_max >= lam})")
    print(f"   ciphertext error rank t={I['t']}  (lambda*t={lam*I['t']} <= floor(p/2)={p//2})")
    print("=" * 78)
    if r_max < lam:
        print("  r_max < lambda: guessing regime does not apply to this size."); return
    # public reference h0 (rank m), H0 = Moore(h0, p)
    rng = random.Random(seed ^ 0x9e3779b9)
    while True:
        h0 = [rng.randrange(1, F.QM) for _ in range(F.m)]
        if gf2_rank_of(F, h0) == F.m: break
    H0 = moore(F, h0, p)

    print("  columns:  ker = dim of the public-system kernel over F_q  |  img rk ="
          " image rank rk(Gpub.D), target k")
    print("            supp = F_q-dim of the key's entry support (<= lambda)  |  "
          "decrypt / msg=m? = key decrypts y / recovers the exact plaintext")
    print(f"\n{'guess F':<22}{'kind':<7}{'ker':>4}{'img rk':>8}{'supp':>6}"
          f"{'decrypt':>9}{'msg=m?':>8}")
    print("-" * 78)
    # GOOD guesses: F = alpha V  (alpha = 1, then random alphas); dim = r_max = lambda
    goods = [("F = V", I['Vb'])]
    for al in (F.pw(2, 3), F.pw(2, 7)):
        goods.append((f"F = a^{'?'} V", [F.mul(al, v) for v in I['Vb']]))
    for name, Fb in goods:
        assert is_scalar_multiple_of_V(F, Fb, I['Vb'])
        b = try_guess(I, H0, Fb, rng)
        print(f"{name:<22}{'GOOD':<7}{b['kerdim']:>4}{b['imgrank']:>8}"
              f"{str(b['support']):>6}{('YES' if b['decrypt'] else 'no'):>9}"
              f"{('YES' if b['msg_ok'] else '-'):>8}")
    # BAD guesses: random r_max-dim spaces that are NOT alpha V
    made = 0
    while made < n_bad:
        Fb = []
        while len(Fb) < r_max:
            x = rng.randrange(1, F.QM)
            if gf2_rank_of(F, Fb + [x]) == len(Fb) + 1: Fb.append(x)
        if is_scalar_multiple_of_V(F, Fb, I['Vb']): continue
        b = try_guess(I, H0, Fb, rng)
        print(f"{'F = <random>':<22}{'bad':<7}{b['kerdim']:>4}{b['imgrank']:>8}"
              f"{str(b['support']):>6}{('YES' if b['decrypt'] else 'no'):>9}"
              f"{('YES' if b['msg_ok'] else '-'):>8}")
        made += 1
    print("-" * 78)
    print("  Reading: every kernel element already has entries in the guessed F (support")
    print("  <= r = lambda). A GOOD guess (F = alpha V) yields a non-trivial kernel from")
    print("  which a combination of image rank k is found and DECRYPTS to the plaintext;")
    print("  a BAD guess yields a trivial (or spurious) kernel with no such key.")


def batch(m, n1, k1, n2, k2, lam, N=24, base_seed=1000):
    """Run the PUBLIC attack (good guess F = V, plus one bad guess) on N instances."""
    ok_dec = ok_msg = bad_rej = 0
    for s in range(base_seed, base_seed + N):
        I = build_instance(m, n1, k1, n2, k2, lam, s)
        F = I['F']; rng = random.Random(s ^ 0x5bd1e995)
        while True:
            h0 = [rng.randrange(1, F.QM) for _ in range(F.m)]
            if gf2_rank_of(F, h0) == F.m: break
        H0 = moore(F, h0, I['p'])
        g = try_guess(I, H0, I['Vb'], rng)                 # good guess F = V
        ok_dec += g['decrypt']; ok_msg += g['msg_ok']
        while True:                                        # one genuine bad guess
            Fb = []
            while len(Fb) < (I['k'] * I['p']) // I['n']:
                x = rng.randrange(1, F.QM)
                if gf2_rank_of(F, Fb + [x]) == len(Fb) + 1: Fb.append(x)
            if not is_scalar_multiple_of_V(F, Fb, I['Vb']): break
        bmark = try_guess(I, H0, Fb, rng)
        bad_rej += (not bmark['decrypt'])
    tag = f"n1={n1} n2={n2} k2={k2} m={m}"
    print(f"  [{tag:<22}]  decrypted {ok_dec}/{N}  (message matched {ok_msg}/{N});"
          f"  bad guesses rejected {bad_rej}/{N}")


if __name__ == "__main__":
    # single-block (Modification-I / LGRH shape): n1 = k1 = 1
    run(m=10, n1=1, k1=1, n2=10, k2=4, lam=2, seed=1)
    print()
    # flagship GabKron: n1 = k1 = 2
    run(m=12, n1=2, k1=2, n2=12, k2=5, lam=2, seed=2)
    print("\n" + "=" * 78)
    print(" BATCH over random instances (public attack, good guess F = V)")
    print("=" * 78)
    batch(m=10, n1=1, k1=1, n2=10, k2=4, lam=2, N=24)
    batch(m=12, n1=2, k1=2, n2=12, k2=5, lam=2, N=24)
