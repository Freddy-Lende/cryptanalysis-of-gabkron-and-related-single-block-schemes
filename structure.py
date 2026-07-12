"""
What is Stab_R(Gab2) when n2 < m ?  (Sun24: m=2n2 ; Lau24: m=n2).
Determines dim_Fq(S), and whether S is a field (commutative + all nonzero invertible).
Sage-safe (operator.xor; runs identically under python3 and Sage load()).
"""
import operator, itertools
xor = operator.xor

# ---- parametric GF(2^m) ----------------------------------------------------
IRRED = {2:0b111, 4:0b10011, 6:0b1000011, 8:0b100011011}   # primitive polys
class GF:
    def __init__(self, m):
        self.m = m; self.MOD = IRRED[m]; self.QM = 1 << m; self.N = self.QM
    def mul(self, x, y):
        r = 0
        while y:
            if y & 1: r = xor(r, x)
            y >>= 1; x <<= 1
            if x & self.QM: x = xor(x, self.MOD)
        return r
    def pw(self, x, e):
        r, b = 1, x
        while e:
            if e & 1: r = self.mul(r, b)
            b = self.mul(b, b); e >>= 1
        return r
    def inv(self, x): return self.pw(x, self.QM - 2)
    def frob(self, x, i):
        for _ in range(i): x = self.mul(x, x)
        return x
    def bits(self, x): return [(x >> i) & 1 for i in range(self.m)]

def matmul(F, A, B):
    r, s, c = len(A), len(B), len(B[0]); out = [[0]*c for _ in range(r)]
    for i in range(r):
        for k in range(s):
            a = A[i][k]
            if a:
                for j in range(c): out[i][j] = xor(out[i][j], F.mul(a, B[k][j]))
    return out
def transpose(A): return [list(c) for c in zip(*A)]
def is_zero(A): return all(all(x == 0 for x in row) for row in A)

def rref(F, A):
    A = [row[:] for row in A]; rows = len(A); c = len(A[0]); piv = []; r = 0
    for col in range(c):
        sel = next((i for i in range(r, rows) if A[i][col]), None)
        if sel is None: continue
        A[r], A[sel] = A[sel], A[r]
        iv = F.inv(A[r][col]); A[r] = [F.mul(iv, x) for x in A[r]]
        for i in range(rows):
            if i != r and A[i][col]:
                f = A[i][col]; A[i] = [xor(A[i][j], F.mul(f, A[r][j])) for j in range(c)]
        piv.append(col); r += 1
        if r == rows: break
    return A, piv
def rank(F, A): return len(rref(F, A)[1]) if A and A[0] else 0
def right_kernel(F, A):
    R, piv = rref(F, A); c = len(A[0]); ps = set(piv); basis = []
    for f in [j for j in range(c) if j not in ps]:
        x = [0]*c; x[f] = 1
        for ri, pc in enumerate(piv): x[pc] = R[ri][f]
        basis.append(x)
    return basis
def gf2_kernel(rows):
    A = [r[:] for r in rows]
    if not A: return []
    nv = len(A[0]); piv = []; r = 0
    for col in range(nv):
        sel = next((i for i in range(r, len(A)) if A[i][col]), None)
        if sel is None: continue
        A[r], A[sel] = A[sel], A[r]
        for i in range(len(A)):
            if i != r and A[i][col]: A[i] = [xor(A[i][j], A[r][j]) for j in range(nv)]
        piv.append(col); r += 1
        if r == len(A): break
    ps = set(piv); basis = []
    for f in [j for j in range(nv) if j not in ps]:
        x = [0]*nv; x[f] = 1
        for ri, pc in enumerate(piv): x[pc] = A[ri][f]
        basis.append(x)
    return basis
def moore(F, v, rrows): return [[F.frob(v[j], i) for j in range(len(v))] for i in range(rrows)]

# --- generic matrix utilities over GF(q^m) (moved here; self-contained) ------
def ident(F, n):  return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
def matadd(A, B): return [[A[i][j] ^ B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
def cols(A, idx): return [[A[i][j] for j in idx] for i in range(len(A))]

def kron(F, A, B):
    ra, ca, rb, cb = len(A), len(A[0]), len(B), len(B[0])
    out = [[0]*(ca*cb) for _ in range(ra*rb)]
    for i in range(ra):
        for j in range(ca):
            a = A[i][j]
            if a:
                for p in range(rb):
                    for q in range(cb):
                        out[i*rb+p][j*cb+q] = F.mul(a, B[p][q])
    return out

def inverse(F, A):
    nn = len(A); aug = [A[i][:] + ident(F, nn)[i] for i in range(nn)]; r = 0
    for col in range(nn):
        sel = next((i for i in range(r, nn) if aug[i][col]), None)
        if sel is None: raise ValueError("singular")
        aug[r], aug[sel] = aug[sel], aug[r]
        iv = F.inv(aug[r][col]); aug[r] = [F.mul(iv, x) for x in aug[r]]
        for i in range(nn):
            if i != r and aug[i][col]:
                f = aug[i][col]
                aug[i] = [aug[i][j] ^ F.mul(f, aug[r][j]) for j in range(2*nn)]
        r += 1
    return [row[nn:] for row in aug]

def solve_xA_eq_b(F, A, b):
    """particular x (len rows(A)) with x A = b."""
    r, c = len(A), len(A[0])
    M = [[A[i][j] for i in range(r)] + [b[j]] for j in range(c)]
    R, piv = rref(F, M)
    x = [0]*r
    for ri, pc in enumerate(piv):
        if pc == r: return None
        x[pc] = R[ri][r]
    return x

def _gf2_rank(rows):
    A = [row[:] for row in rows]
    if not A: return 0
    nv = len(A[0]); r = 0
    for col in range(nv):
        sel = next((i for i in range(r, len(A)) if A[i][col]), None)
        if sel is None: continue
        A[r], A[sel] = A[sel], A[r]
        for i in range(len(A)):
            if i != r and A[i][col]:
                A[i] = [A[i][j] ^ A[r][j] for j in range(nv)]
        r += 1
        if r == len(A): break
    return r

def _xrank(F, M):
    return _gf2_rank([F.bits(x) for row in M for x in row])

def stab_basis(F, G2, H2, n2):
    k2 = len(G2); nk = len(H2)
    eqs = []
    for p in range(k2):
        for r in range(nk):
            coeff = [F.mul(G2[p][l], H2[r][j]) for l in range(n2) for j in range(n2)]
            for b in range(F.m):
                eqs.append([F.bits(coeff[idx])[b] for idx in range(n2*n2)])
    ker = gf2_kernel(eqs)
    return [[[v[l*n2 + j] for j in range(n2)] for l in range(n2)] for v in ker]

def analyze(F, n2, k2, label):
    v2 = [F.pw(2, j) for j in range(n2)]            # gv2 = (1, a, a^2, ..., a^{n2-1})
    G2 = moore(F, v2, k2); H2 = right_kernel(F, G2)
    assert is_zero(matmul(F, G2, transpose(H2)))
    S = stab_basis(F, G2, H2, n2)
    d = len(S)
    # field tests on the algebra spanned by S (over F2): commutative? all nonzero invertible?
    def lc(coeffs):
        out = [[0]*n2 for _ in range(n2)]
        for c, M in zip(coeffs, S):
            if c: out = [[xor(out[i][j], M[i][j]) for j in range(n2)] for i in range(n2)]
        return out
    elems = [lc(c) for c in itertools.product(range(2), repeat=d)]   # all q^d elements
    # closed under product?
    Sset = set(tuple(x for row in M for x in row) for M in elems)
    closed = all(tuple(x for row in matmul(F, A, B) for x in row) in Sset
                 for A in elems for B in elems)
    # commutative?
    comm = all(matmul(F, A, B) == matmul(F, B, A) for A in elems for B in elems)
    # every nonzero invertible (rank n2)?  -> field
    nonzero = [M for M in elems if any(any(x for x in row) for row in M)]
    allinv = all(rank(F, M) == n2 for M in nonzero)
    print(f"  {label}: q=2, m={F.m}, [n2={n2}, k2={k2}]  ->  "
          f"dim_F2(S)={d}  |S|=2^{d}={2**d}")
    print(f"     closed under product: {closed} | commutative: {comm} | "
          f"every nonzero invertible: {allinv}  => "
          f"{'FIELD GF(2^%d)'%d if (closed and comm and allinv) else 'NOT a field'}")
    return d, (closed and comm and allinv)

print("=== Lau24 regime: m = n2 (full-length inner code) ===")
analyze(GF(2), 2, 1, "Lau")
analyze(GF(4), 4, 2, "Lau")
analyze(GF(4), 4, 1, "Lau")
analyze(GF(4), 4, 3, "Lau")

print("\n=== Sun24 regime: m = 2*n2 (half-length inner code) ===")
analyze(GF(4), 2, 1, "Sun")          # m=4, n2=2
analyze(GF(6), 3, 1, "Sun")          # m=6, n2=3
analyze(GF(6), 3, 2, "Sun")          # m=6, n2=3, k2=2

print("\n=== intermediate n2 < m (other ratios), to see the pattern ===")
analyze(GF(6), 2, 1, "n2=2<m=6")
analyze(GF(8), 3, 1, "n2=3<m=8")
analyze(GF(8), 4, 2, "n2=4<m=8")
analyze(GF(8), 5, 2, "n2=5<m=8")
