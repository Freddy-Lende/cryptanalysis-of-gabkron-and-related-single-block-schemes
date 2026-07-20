"""
gabkron_attack_common.py  --  shared machinery for the two GabKron attack scripts.

Provides:
  build_instance(...)        one random spread-distortion GabKron instance + ciphertext
  recover_perblock(I,h0)     per-block recovery: UNIFORM clearing to exactly
                             n2-t1 clean columns per block, single parity
                             p = n2 - t1 - k2, per-block full-column-rank T_{2,i};
                             matches Theorem "Alternative key recovery".
  decrypt(...)               full message recovery from a recovered key (no residual).

Pure Python (python3, NOT Sage). Needs structure.py.
"""
import io, contextlib, random, functools, operator
xor = operator.xor
red = lambda l: functools.reduce(xor, l, 0)

def fmt(F, x):
    if x == 0: return "0"
    for e in range(F.QM - 1):
        if F.pw(2, e) == x: return "1" if e == 0 else ("a" if e == 1 else f"a^{e}")
    return "+".join((("a^%d" % i) if i > 1 else ("a" if i == 1 else "1"))
                    for i in range(F.m) if (x >> i) & 1)
def fmt_vec(F, v): return "[" + ", ".join(fmt(F, x) for x in v) + "]"

with contextlib.redirect_stdout(io.StringIO()):
    from structure import (GF, matmul, transpose, is_zero, rank,
                                right_kernel, gf2_kernel, moore, stab_basis)
    import structure as ss
    from structure import (ident, matadd, cols, kron, inverse,
                                solve_xA_eq_b, _xrank)

def gf2_rank(vecs):
    A = [v[:] for v in vecs]; r = 0
    if not A: return 0
    nv = len(A[0])
    for c in range(nv):
        s = next((i for i in range(r, len(A)) if A[i][c]), None)
        if s is None: continue
        A[r], A[s] = A[s], A[r]
        for i in range(len(A)):
            if i != r and A[i][c]: A[i] = [xor(A[i][t], A[r][t]) for t in range(nv)]
        r += 1
        if r == len(A): break
    return r

def fq_kernel(F, X):
    kk, nn = len(X), len(X[0])
    rows = [[F.bits(X[i][j])[b] for j in range(nn)] for i in range(kk) for b in range(F.m)]
    return gf2_kernel(rows)

def clear_block(F, Xi, n2, rng):
    """T in GL_{n2}(F_q): the l_i F_q-dependent columns to the LEFT, clean to the RIGHT."""
    K = fq_kernel(F, Xi); rng.shuffle(K); right = []
    for v in K:
        if gf2_rank(right + [v]) > len(right): right.append(v)
    left = []
    for i in range(n2):
        e = [1 if t == i else 0 for t in range(n2)]
        if gf2_rank(right + left + [e]) > len(right) + len(left): left.append(e)
        if len(left) + len(right) == n2: break
    allc = left + right
    T = [[allc[c][r] for c in range(n2)] for r in range(n2)]
    inverse(F, T); return T, len(left)

def build_T2(F, Gp, H0i, Lc):
    """full-col-rank T2 in F_q^{m x Lc} with  Gp T2^T H0i^T = 0  (H0i T2 a parity of Gp)."""
    m = F.m; hr = len(H0i); rows = []
    for p in range(len(Gp)):
        for c in range(hr):
            for b in range(m):
                rows.append([F.bits(F.mul(Gp[p][l], H0i[c][a]))[b] for a in range(m) for l in range(Lc)])
    ker = gf2_kernel(rows)
    def rs(v): return [[v[a * Lc + l] for l in range(Lc)] for a in range(m)]
    for v in ker:
        Tt = rs(v)
        if gf2_rank([[Tt[a][l] for a in range(m)] for l in range(Lc)]) == Lc: return Tt
    r = random.Random(0)
    for _ in range(8000):
        sub = [v for v in ker if r.randint(0, 1)]
        if not sub: continue
        vv = [red([x[i] for x in sub]) for i in range(m * Lc)]; Tt = rs(vv)
        if gf2_rank([[Tt[a][l] for a in range(m)] for l in range(Lc)]) == Lc: return Tt
    return None

def gf_solve(F, A, b):
    """one solution x of A x = b over F_qm (A: r x c, b: r), or None if inconsistent."""
    r = len(A); c = len(A[0]); M = [A[i][:] + [b[i]] for i in range(r)]; piv = []; row = 0
    for col in range(c):
        sel = next((i for i in range(row, r) if M[i][col] != 0), None)
        if sel is None: continue
        M[row], M[sel] = M[sel], M[row]
        inv = F.inv(M[row][col]); M[row] = [F.mul(inv, v) for v in M[row]]
        for i in range(r):
            if i != row and M[i][col] != 0:
                f = M[i][col]; M[i] = [xor(M[i][t], F.mul(f, M[row][t])) for t in range(c + 1)]
        piv.append(col); row += 1
        if row == r: break
    for i in range(row, r):
        if M[i][c] != 0 and not any(M[i][:c]): return None
    x = [0] * c
    for i, col in enumerate(piv): x[col] = M[i][c]
    return x

def root_space(F, sigma):
    """basis over F_q of {x : sum_u sigma_u x^{[u]} = 0} (sigma linearized, len nu+1)."""
    m = F.m
    img = [red([F.mul(sigma[u], F.frob(1 << b, u)) for u in range(len(sigma))]) for b in range(m)]
    M = [[(img[bcol] >> brow) & 1 for bcol in range(m)] for brow in range(m)]
    return [red([(1 << b) for b in range(m) if v[b]]) for v in gf2_kernel(M)]

def gab_decode_block(F, h0, d, S, tau):
    """Gabidulin syndrome decoding from PARITY support h0 (Moore(h0,d) the parity):
       recover the length-m error vector e with e Moore(h0,d)^T = S, rk(e) <= tau.
       Key equation (char 2): sum_{u=0}^{nu} sigma_u S_{j-u}^{[u]} = 0, sigma_nu = 1."""
    m = F.m; L = len(h0)
    if not any(S): return [0] * L            # zero syndrome => zero error in this block
    for nu in range(1, tau + 1):
        if d - nu < nu: break
        A = [[F.frob(S[j - u], u) for u in range(nu)] for j in range(nu, d)]
        b = [F.frob(S[j - nu], nu) for j in range(nu, d)]
        sol = gf_solve(F, A, b)
        if sol is None: continue
        sigma = sol + [1]
        Ve = root_space(F, sigma)
        if len(Ve) != nu: continue
        rows = []; rhs = []
        for j in range(d):
            for brow in range(m):
                rc = []
                for i in range(L):
                    for l in range(nu):
                        rc.append((F.mul(Ve[l], F.frob(h0[i], j)) >> brow) & 1)
                rows.append(rc); rhs.append((S[j] >> brow) & 1)
        nun = L * nu; A2 = [rows[i] + [rhs[i]] for i in range(len(rows))]; piv = []; r = 0
        for col in range(nun):
            sel = next((i for i in range(r, len(A2)) if A2[i][col]), None)
            if sel is None: continue
            A2[r], A2[sel] = A2[sel], A2[r]
            for i in range(len(A2)):
                if i != r and A2[i][col]: A2[i] = [xor(A2[i][t], A2[r][t]) for t in range(nun + 1)]
            piv.append(col); r += 1
            if r == len(A2): break
        if any(A2[i][nun] == 1 and not any(A2[i][:nun]) for i in range(r, len(A2))): continue
        y = [0] * nun
        for i, col in enumerate(piv): y[col] = A2[i][nun]
        return [red([F.mul(Ve[l], y[i * nu + l]) for l in range(nu)]) for i in range(L)]
    return None

def build_N(F, Gpub, D, Hs, m):
    """N = D . blockdiag(Hs_i^T): for block i, D[:, i*m:(i+1)*m] (.) Hs[i]^T."""
    n = len(D); n1 = len(Hs)
    blocks = []
    for i in range(n1):
        Hi = Hs[i]; ri = len(Hi)
        for c in range(ri):
            blocks.append([red([F.mul(D[row][i * m + j], Hi[c][j]) for j in range(m)]) for row in range(n)])
    # blocks is list of columns (each length n); transpose to N (n x sum ri)
    return [[blocks[c][row] for c in range(len(blocks))] for row in range(n)]

def decrypt(F, Gpub, D, Hs, y, m, t, label):
    """Theorem (full recovery): z = yD = m(G_pub D) + eD; decode each length-m block
       in the dual Gabidulin code D_i = ker(H_i) to recover (eD)_i by syndrome, then
       solve m from m(G_pub D) = z - eD using rk(G_pub D)=k. Returns (ok, eD, mrec)."""
    n = len(D); ncols = len(D[0]); n1 = len(Hs)
    z = [red([F.mul(y[i], D[i][c]) for i in range(n)]) for c in range(ncols)]
    eD = [0] * ncols
    for i in range(n1):
        Hi = Hs[i]; d = len(Hi); h0 = Hi[0]                 # parity support = row 0 of Moore
        zi = z[i * m:(i + 1) * m]
        Si = [red([F.mul(zi[j], Hi[r][j]) for j in range(m)]) for r in range(d)]
        ei = gab_decode_block(F, h0, d, Si, d // 2)
        if ei is None: return False, None, None
        for j in range(m): eD[i * m + j] = ei[j]
    codeword = [xor(z[c], eD[c]) for c in range(ncols)]      # = m (G_pub D)
    GD = matmul(F, Gpub, D)
    mrec = solve_xA_eq_b(F, GD, codeword)
    ok = mrec is not None and all(
        red([F.mul(mrec[i], GD[i][c]) for i in range(len(GD))]) == codeword[c] for c in range(ncols))
    return ok, eD, mrec

# ----------------------------------------------------------------------
def build_instance(seed, m, n2, k2, lblocks):
    rng = random.Random(seed)
    ss.IRRED.setdefault(m, {4: 0b10011, 6: 0b1000011, 8: 0b100011011, 10: 0b10000001001}[m]); F = GF(m)
    n1 = k1 = 2; n, k = n1 * n2, k1 * k2; lam = 2; Vb = [1, 2]
    t1 = sum(lblocks); t = max((n2 - k2 - 2 * t1) // (2 * lam), 1)
    v2 = [F.pw(2, j) for j in range(n2)]; G2 = moore(F, v2, k2); H2 = right_kernel(F, G2)
    v1 = [F.pw(2, 1), F.pw(2, 4)]; G1 = moore(F, v1, k1); GKP = kron(F, G1, G2); e0 = F.pw(2, 2)

    def colrank_block(li):
        while True:
            B = [[0] * n2 for _ in range(k)]
            dirs = [[F.mul(e0, rng.randint(0, 1)) for _ in range(k)] for _ in range(li)]
            for p in range(li):
                w = [rng.randint(0, 1) for _ in range(n2)]
                while all(x == 0 for x in w): w = [rng.randint(0, 1) for _ in range(n2)]
                for i in range(k):
                    for j in range(n2): B[i][j] ^= F.mul(dirs[p][i], w[j])
            _, lic = clear_block(F, B, n2, rng)
            if lic == li: return B
    while True:
        Xb = [colrank_block(lblocks[i]) for i in range(n1)]
        X = [Xb[0][i] + Xb[1][i] for i in range(k)]
        if not is_zero(cols(X, list(range(0, n2)))) and not is_zero(cols(X, list(range(n2, 2 * n2)))):
            break  # spread: both Kronecker blocks non-zero
    while True:
        P = [[Vb[rng.randint(0, 1)] for _ in range(n)] for _ in range(n)]
        try: Pi = inverse(F, P); break
        except ValueError: pass
    Gpub = matmul(F, matadd(GKP, X), Pi)
    # ciphertext shared by both formulations
    m_true = [rng.randint(0, F.QM - 1) for _ in range(k)]
    while True:
        cdir = rng.randint(1, F.QM - 1); uv = [0] * n
        for p in rng.sample(range(n), t): uv[p] = 1
        ev = [F.mul(cdir, uv[j]) for j in range(n)]
        if _xrank(F, [ev]) == t: break
    y = [xor(red([F.mul(m_true[i], Gpub[i][j]) for i in range(k)]), ev[j]) for j in range(n)]
    return dict(F=F, rng=rng, n1=n1, k1=k1, n2=n2, k2=k2, n=n, k=k, lam=lam, Vb=Vb,
                t1=t1, t=t, G2=G2, G1=G1, GKP=GKP, Xb=Xb, X=X, P=P, Pi=Pi, Gpub=Gpub,
                m_true=m_true, y=y, lblocks=lblocks)

def recover_perblock(I, h0):
    """Per-block recovery: UNIFORM clearing to exactly n2-t1 clean columns per block.
       Single parity p = n2-t1-k2 (same for every block); per-block T_{2,i}.
       Returns D, Hs (=[H0]*n1), Vset, p.  Matches Theorem (alternative key recovery)."""
    F = I['F']; Gpub = I['Gpub']; P = I['P']; n = I['n']; n1 = I['n1']
    n2 = I['n2']; k2 = I['k2']; m = F.m; k = I['k']; Vb = I['Vb']; lam = I['lam']
    rng = I['rng']; t1 = I['t1']
    Vset = [red([Vb[s] for s in range(lam) if (c >> s) & 1]) for c in range(2 ** lam)]
    p = n2 - t1 - k2                      # uniform parity row count
    H0 = moore(F, h0, p); Hs = [H0] * n1
    Ts = []; Icl_i = []
    for i in range(n1):
        Ti, li = clear_block(F, I['Xb'][i], n2, rng)      # li distorted cols to the left
        Ts.append(Ti)
        clean = list(range(li, n2))                        # clean cols (in T-coords)
        keep = clean[:n2 - t1]                             # EXACTLY n2-t1 of them
        Icl_i.append([i * n2 + c for c in keep])
    T = ident(F, n)
    for i in range(n1):
        for a in range(n2):
            for b in range(n2): T[i * n2 + a][i * n2 + b] = Ts[i][a][b]
    Q = matmul(F, P, T)
    GQ = matmul(F, Gpub, Q)
    T2s = []
    for i in range(n1):
        Gi = cols(GQ, Icl_i[i])                            # k x (n2-t1) clean block
        T2 = build_T2(F, Gi, H0, n2 - t1); T2s.append(T2)
        if T2 is None: return None, Hs, Vset, p
    D = [[0] * (n1 * m) for _ in range(n)]
    for i in range(n1):
        QI = cols(Q, Icl_i[i])                             # n x (n2-t1)
        for row in range(n):
            for a in range(m):
                D[row][i * m + a] = red([F.mul(QI[row][l], T2s[i][a][l])
                                         for l in range(n2 - t1)])
    return D, Hs, Vset, p

def _blk(F, Hs, m, n1):
    """blockdiag(Hs_1,...,Hs_n1): (sum rows) x (n1*m)."""
    rows = sum(len(H) for H in Hs); P = [[0] * (n1 * m) for _ in range(rows)]; off = 0
    for i in range(n1):
        for r in range(len(Hs[i])):
            for a in range(m): P[off + r][i * m + a] = Hs[i][r][a]
        off += len(Hs[i])
    return P


# =========================================================================== #
#  F_qm-MODULE MACHINERY  (paper: Lemma "Frobenius F_qm-module structure of the
#  solution space" and Theorem "Deterministic extraction from a kernel basis")
#
#  L_F = { Z in F^{n x m} : G_pub Z H0^T = 0 } is an F_qm-vector space under
#  Z . beta = Z R_beta^T, where R_beta is multiplication by beta in the basis h0.
#  The action preserves the entry-support, and concatenating a whole F_qm-basis
#  of L_F yields a key of image rank k -- whatever dim_{F_qm} L_F happens to be.
# =========================================================================== #

def gf2_rank_of(F, elements):
    """F_q-dimension of the span of a list of F_qm elements (q=2 bit layout)."""
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

# --------------------------------------------------------------------------- #
#  R_beta : matrix over F_q of multiplication by beta in the basis h0
#           (h0 R_beta = beta * h0), used for the F_qm-action Z . beta = Z R_beta^T
# --------------------------------------------------------------------------- #
def R_beta(F, h0, beta):
    m = F.m
    A = [[(h0[j] >> b) & 1 for j in range(m)] for b in range(m)]      # bits of h0[j]

    def coords(x):                                                    # solve A c = bits(x)
        M = [A[b][:] + [(x >> b) & 1] for b in range(m)]
        piv, r = [], 0
        for c in range(m):
            s = next((i for i in range(r, m) if M[i][c]), None)
            if s is None:
                continue
            M[r], M[s] = M[s], M[r]
            for i in range(m):
                if i != r and M[i][c]:
                    M[i] = [M[i][t] ^ M[r][t] for t in range(m + 1)]
            piv.append(c); r += 1
        out = [0] * m
        for i, c in enumerate(piv):
            out[c] = M[i][m]
        return out

    R = [[0] * m for _ in range(m)]
    for c in range(m):
        cc = coords(F.mul(beta, h0[c]))
        for j in range(m):
            R[j][c] = cc[j]
    return R


def act(F, Z, R):
    """Z . R^T  (F_q-linear recombination of columns of Z)."""
    m = len(R)
    return [[red([F.mul(Z[i][a], R[c][a]) for a in range(m) if R[c][a]])
             for c in range(m)] for i in range(len(Z))]


def in_span(F, Z, basis_vecs, n, m):
    """is vec(Z) in the F_2-span of basis_vecs (list of matrices, vectorised)?"""
    def vec(M):
        return [b for i in range(n) for c in range(m) for b in F.bits(M[i][c])]
    M = [vec(B) for B in basis_vecs]
    t = vec(Z)
    L = len(t); r = 0; piv = []
    for c in range(L):
        s = next((i for i in range(r, len(M)) if M[i][c]), None)
        if s is None:
            continue
        M[r], M[s] = M[s], M[r]
        for i in range(len(M)):
            if i != r and M[i][c]:
                M[i] = [M[i][x] ^ M[r][x] for x in range(L)]
        piv.append(c); r += 1
    for i, c in enumerate(piv):
        if t[c]:
            t = [t[x] ^ M[i][x] for x in range(L)]
    return not any(t)


def supp_dim(F, Z, n, m):
    return gf2_rank_of(F, [Z[i][c] for i in range(n) for c in range(m)])


def kbasis(F, Ls, h0, n, m, n1cap=None):
    """extract an F_qm-basis of L_F = span_Fq(Ls) via the R_beta action."""
    gens = [F.pw(2, j) for j in range(m)]
    Rg = [R_beta(F, h0, g) for g in gens]
    Kb, span = [], []
    for Z in Ls:
        if not span or not in_span(F, Z, span, n, m):
            Kb.append(Z)
            span += [act(F, Z, R) for R in Rg]
            if n1cap and len(Kb) >= n1cap:
                break
    return Kb
