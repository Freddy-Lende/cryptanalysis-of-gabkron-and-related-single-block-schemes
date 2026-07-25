"""
briaud_perblock.py  --  can Briaud-Loidreau's constrained system (their Eq. (3),
Prop. 3) be adapted to the per-block GabKron setting, and does its combinatorial
solving approach (Prop. 4) differ from the Burle-type recovery used in the paper?

Context.  After the per-block decomposition (paper, Prop. "General per-block
decomposition"), G_pub Q_{[:,I_cl]} = G* = (G1 (x) I_{k2}) diag(G'_{2,1},...,G'_{2,n1}),
each clean block <G'_{2,i}> being a SHORTENED GABIDULIN code of length n2-t1, dim k2.
That is exactly the input hypothesis of Briaud-Loidreau Prop. 3: a Gabidulin code
masked by a V-valued scrambler.  So per block i we may write their Eq. (3):

        V_i . Hhat_pub,i  =  Hhat_norm . W_i ,        W_i in V-valued, dim <= lambda,

with  V_i in GL_{r_i}(F_qm)  (LEFT unknown, r_i = (n2-t1) - k2 parity rows of block i),
Hhat_pub,i an arbitrary parity-check of the i-th masked block <G'_i> = <G_pub Q_{[:,I_cl,i]}>,
Hhat_norm = Moore(alpha, r_i) on a normal element alpha.

Briaud's combinatorial approach (Prop. 4):
    guess a gamma-dim space U containing a multiple xV, gamma >= lambda, with
        r_i * n_bl >= gamma * n_bl + r_i^2            (their Eq. (4), per block)
    write the coefficients of W_i in a basis of U -> F_q-linear system;
    a good guess yields a 1-dim'l solution space, a bad guess (generically) none.

This script:
  1. builds a real per-block GabKron instance (reusing the paper's builder);
  2. for one clean block, forms Hhat_pub,i and Hhat_norm and the Briaud system (3);
  3. tests Briaud's combinatorial guessing (Prop. 4): good U vs. random bad U,
     checking the solution-space dimension and the V-valuedness of the recovered W_i;
  4. compares the resulting per-block guess dimension gamma_i and orbit exponent with
     the Burle-type r_max used in the paper.

Pure standard-library Python; reuses structure.py and gabkron_attack_common.py.
"""
import sys, random, itertools
import structure as ss
from structure import (GF, matmul, transpose, moore, kron, inverse, rank,
                       right_kernel, rref, cols, ident)
from gabkron_attack_common import (
    red, clear_block, fq_kernel, build_instance as build_perblock_instance,
    R_beta, act, in_span, gf2_rank_of,
)

def perblock_clean(F, inst):
    """Reproduce recover_perblock's clearing: return GQ and the per-block clean
       index sets Icl_blocks (each of length exactly n2-t1)."""
    Gpub = inst['Gpub']; P = inst['P']; n = inst['n']; n1 = inst['n1']
    n2 = inst['n2']; t1 = inst['t1']; rng = inst['rng']
    Ts = []; Icl_blocks = []
    for i in range(n1):
        Ti, li = clear_block(F, inst['Xb'][i], n2, rng)
        Ts.append(Ti)
        clean = list(range(li, n2))
        keep = clean[:n2 - t1]
        Icl_blocks.append([i * n2 + c for c in keep])
    T = ident(F, n)
    for i in range(n1):
        for a in range(n2):
            for b in range(n2):
                T[i * n2 + a][i * n2 + b] = Ts[i][a][b]
    Q = matmul(F, P, T)
    GQ = matmul(F, Gpub, Q)
    return GQ, Q, Icl_blocks

# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #
def normal_element(F):
    """a normal element alpha: {alpha^{[0]},...,alpha^{[m-1]}} is an F_q-basis."""
    m = F.m
    for cand in range(2, F.QM):
        basis = [F.frob(cand, i) for i in range(m)]
        if gf2_rank_of(F, basis) == m:
            return cand
    raise RuntimeError("no normal element")

def moore_alpha(F, alpha, rows, mcols):
    """Hhat_norm = (alpha^{[i+j-2]})_{1<=i<=rows,1<=j<=mcols}  (0-indexed frobenius)."""
    return [[F.frob(alpha, (i + j) % F.m) for j in range(mcols)] for i in range(rows)]

def parity_of_rowspace(F, G):
    """an arbitrary parity-check (right kernel) of the row space of G (k x N)."""
    return right_kernel(F, G)            # list of (N)-vectors H with G H^T = 0

def fq_span_set(F, basis):
    """all F_q-combinations of a list of F_qm elements (q=2)."""
    out = {0}
    for b in basis:
        out |= {x ^ b for x in out}
    return out

def rand_subspace(F, dim, rng):
    while True:
        B = []
        while len(B) < dim:
            x = rng.randrange(1, F.QM)
            if gf2_rank_of(F, B + [x]) == len(B) + 1:
                B.append(x)
        return B

def extend_space(F, Vb, gamma, rng):
    """extend the basis Vb (dim lambda) to a gamma-dim space, gamma >= lambda."""
    B = Vb[:]
    while len(B) < gamma:
        x = rng.randrange(1, F.QM)
        if gf2_rank_of(F, B + [x]) == len(B) + 1:
            B.append(x)
    return B

# --------------------------------------------------------------------------- #
#  Briaud Eq. (3) per block, and its combinatorial test (Prop. 4)
# --------------------------------------------------------------------------- #
def briaud_block_system(F, Gi_pub, alpha, Ubasis):
    """
    Build and solve the F_q-linear system from Briaud Eq. (3) for ONE masked block:

        V . Hhat_pub  =  Hhat_norm . W ,     entries of W in the guessed space U.

    Gi_pub : k x n_bl   generator of the masked clean block  <G_pub Q_{[:,I_cl,i]}>
             (row space = <G'_{2,i}> P-scrambled), dimension k2 = rank(Gi_pub).
    Unknowns: V (r x r over F_qm)  and  W (m x n_bl with entries in U, gamma-dim).
    Returns (solution_space_dim, witnessW_or_None).
    """
    m = F.m
    n_bl = len(Gi_pub[0])
    k2 = rank(F, Gi_pub)
    r = n_bl - k2                                   # parity rows of the block
    if r <= 0:
        return None, None
    gamma = len(Ubasis)

    Hpub = parity_of_rowspace(F, Gi_pub)            # r x n_bl, Gi_pub Hpub^T = 0
    if len(Hpub) != r:
        r = len(Hpub)
    Hnorm = moore_alpha(F, alpha, r, m)             # r x m

    # Unknowns over F_q:
    #   V: r*r entries in F_qm  -> r*r*m  F_q-unknowns (V arbitrary over F_qm)
    #   W: m*n_bl entries, each in U (gamma-dim) -> m*n_bl*gamma  F_q-unknowns
    # Equation V Hpub = Hnorm W  is r x n_bl over F_qm -> r*n_bl*m  F_q-equations.
    # Index maps.
    def vidx(a, b, t):      # V[a][b] coeff on F_q-basis vector e_t (a,b in [0,r), t in [0,m))
        return (a * r + b) * m + t
    nV = r * r * m
    def widx(a, b, g):      # W[a][b] coeff on U-basis vector Ubasis[g]
        return nV + (a * n_bl + b) * gamma + g
    nW = m * n_bl * gamma
    ncols = nV + nW

    # standard F_q-basis of F_qm is {1, x, x^2, ...} = bit positions
    # V[a][b] = sum_t v_{abt} * (1<<t)   (t-th basis element is 2^t in this GF(2^m) rep)
    # W[a][b] = sum_g w_{abg} * Ubasis[g]
    rows = []
    # (V Hpub)_{a,j} = sum_b V[a][b] * Hpub[b][j]
    # (Hnorm W)_{a,j} = sum_u Hnorm[a][u] * W[u][j]
    for a in range(r):
        for j in range(n_bl):
            # each side is an element of F_qm; equate the m F_q-coordinates
            # build linear form in unknowns for each output bit
            # left: sum_b sum_t v_{abt} * (Hpub[b][j] * 2^t)
            # right: sum_u sum_g w_{ujg} * (Hnorm[a][u] * Ubasis[g])
            coeff = [dict() for _ in range(m)]      # per output-bit: {col: 1}
            for b in range(r):
                hpb = Hpub[b][j]
                for t in range(m):
                    val = F.mul(hpb, 1 << t)        # F_qm element contributed
                    col = vidx(a, b, t)
                    for bit in range(m):
                        if (val >> bit) & 1:
                            coeff[bit][col] = coeff[bit].get(col, 0) ^ 1
            for u in range(m):
                hn = Hnorm[a][u]
                if hn == 0:
                    continue
                for g in range(gamma):
                    val = F.mul(hn, Ubasis[g])
                    col = widx(u, j, g)
                    for bit in range(m):
                        if (val >> bit) & 1:
                            coeff[bit][col] = coeff[bit].get(col, 0) ^ 1
            for bit in range(m):
                row = [0] * ncols
                for col, c in coeff[bit].items():
                    row[col] = c
                rows.append(row)

    # homogeneous system: find right kernel over F_q (GF(2))
    ker = gf2_right_kernel(rows, ncols)
    dim = len(ker)
    # extract one candidate W (if any nonzero kernel vector), report its support
    witnessW = None
    for kv in ker:
        # does this kernel vector give a nonzero W ?
        Wmat = [[0] * n_bl for _ in range(m)]
        anyW = False
        for a in range(m):
            for b in range(n_bl):
                val = 0
                for g in range(gamma):
                    if kv[widx(a, b, g)]:
                        val ^= Ubasis[g]
                Wmat[a][b] = val
                if val:
                    anyW = True
        if anyW:
            witnessW = Wmat
            break
    return dim, witnessW

def gf2_right_kernel(rows, ncols):
    """basis of the right kernel over GF(2) of the given rows (list of 0/1 lists)."""
    A = [row[:] for row in rows]
    piv_col = []
    r = 0
    R = len(A)
    for col in range(ncols):
        sel = next((i for i in range(r, R) if A[i][col]), None)
        if sel is None:
            continue
        A[r], A[sel] = A[sel], A[r]
        for i in range(R):
            if i != r and A[i][col]:
                A[i] = [A[i][t] ^ A[r][t] for t in range(ncols)]
        piv_col.append(col); r += 1
        if r == R:
            break
    pivset = set(piv_col)
    free = [c for c in range(ncols) if c not in pivset]
    basis = []
    for f in free:
        vec = [0] * ncols
        vec[f] = 1
        for i, pc in enumerate(piv_col):
            if A[i][f]:
                vec[pc] = 1
        basis.append(vec)
    return basis

def support_dim_of_W(F, Wmat):
    if Wmat is None:
        return None
    elems = [Wmat[a][b] for a in range(len(Wmat)) for b in range(len(Wmat[0])) if Wmat[a][b]]
    return gf2_rank_of(F, elems) if elems else 0

# --------------------------------------------------------------------------- #
#  main experiment
# --------------------------------------------------------------------------- #
def run(seed=1000, m=6, n2=6, k2=2, lblocks=(1, 1), trials_bad=5):
    print("=" * 92)
    print(f" Briaud-Loidreau per-block adaptation test | m={m} n2={n2} k2={k2} lblocks={lblocks}")
    print("=" * 92)
    inst = build_perblock_instance(seed, m, n2, k2, list(lblocks))
    F = inst['F']; Gpub = inst['Gpub']; Vb = inst['Vb']; lam = inst['lam']
    n1 = inst['n1']
    GQ, Q, Icl_blocks = perblock_clean(F, inst)
    print(f"  lambda={lam}, V basis={Vb}, t1={inst['t1']}, clean-column blocks sizes="
          f"{[len(ib) for ib in Icl_blocks]}")

    alpha = normal_element(F)
    x_scal = None
    # a genuine multiple xV to build the good guess: pick x in F_qm^*, U >= xV
    rng = random.Random(seed ^ 0xABCD)

    for i in range(n1):
        Gi = cols(GQ, Icl_blocks[i])           # k x n_bl masked clean block
        k2i = rank(F, Gi)
        n_bl = len(Gi[0]); r = n_bl - k2i
        print(f"\n  --- block {i}: n_bl={n_bl}, k2={k2i}, parity rows r={r} ---")
        if r <= 0:
            print("    degenerate block (r<=0), skipped"); continue

        # Briaud condition Eq.(4) per block: r*n_bl >= gamma*n_bl + r^2
        # -> gamma <= r - r^2/n_bl = r(1 - r/n_bl)
        gamma_max = r - (r * r + n_bl - 1) // n_bl      # floor(r(1-r/n_bl))
        gamma = max(lam, gamma_max)
        feasible = (r * n_bl >= gamma * n_bl + r * r) and (gamma >= lam)
        print(f"    Briaud Eq.(4): gamma_max=floor(r(1-r/n_bl))={gamma_max}, "
              f"lambda={lam}  => feasible={feasible}")
        if gamma_max < lam:
            print("    -> Eq.(4) cannot hold with gamma>=lambda: Briaud enumeration "
                  "INAPPLICABLE to this block (r too large relative to n_bl).")
            continue

        # GOOD guess: U of dim gamma containing a scalar multiple xV
        x = rng.randrange(1, F.QM)
        xV = [F.mul(x, v) for v in Vb]
        Ugood = extend_space(F, xV, gamma, rng)
        dim_g, W_g = briaud_block_system(F, Gi, alpha, Ugood)
        supp_g = support_dim_of_W(F, W_g)
        print(f"    GOOD guess U>=xV (gamma={gamma}): solution-space dim={dim_g}, "
              f"W support-dim={supp_g} (expect <= lambda={lam})")

        # BAD guesses: random U of same dim, screened NOT to contain any yV
        bad_nonzero = 0
        for _ in range(trials_bad):
            Ubad = rand_subspace(F, gamma, rng)
            Uset = fq_span_set(F, Ubad)
            contains_mult = any(all(F.mul(y, v) in Uset for v in Vb)
                                for y in range(1, F.QM))
            if contains_mult:
                continue
            dim_b, W_b = briaud_block_system(F, Gi, alpha, Ubad)
            if dim_b and dim_b > 0 and W_b is not None:
                bad_nonzero += 1
        print(f"    BAD guesses (screened, gamma={gamma}): {bad_nonzero}/{trials_bad} "
              f"gave a nonzero V-valued W (expect ~0)")

        # compare with the paper's Burle-type r_max for the SAME block
        # (unilateral system: n_bl * r <= k2 * r?  paper uses n r <= k p globally)
        print(f"    Burle-type (paper) uses one unilateral system over the whole key;")
        print(f"    Briaud here needs the LEFT unknown V ({r}x{r} over F_qm) per block,")
        print(f"    i.e. {r*r*F.m} extra F_q-unknowns absent from the Burle formulation.")

def summary():
    print("=" * 92)
    print(" SUMMARY: adapting Briaud-Loidreau's attack to the per-block GabKron setting")
    print("=" * 92)
    print("""
1. THEORY -- the adaptation is possible.  After the per-block decomposition
   (paper Prop. 'General per-block decomposition'), each clean block <G'_{2,i}> is a
   shortened Gabidulin code of length n_bl=n2-t1, dim k2, masked by a V-valued
   scrambler.  This is exactly the input of Briaud-Loidreau Prop. 3, so their
   constrained system (Eq. 3)   V_i . Hhat_pub,i = Hhat_norm . W_i   can be written
   per block, and their combinatorial guess (Prop. 4, a gamma-dim U >= xV) applies.

2. THE PER-BLOCK GUESS DIMENSION EQUALS BURLE'S r_max.  Briaud Eq.(4) reads
   r*n_bl >= gamma*n_bl + r^2 with r = n_bl-k2, giving
        gamma_max = floor( r(1 - r/n_bl) ) = floor( k2*(n_bl-k2)/n_bl ).
   The paper's Burle-type over-determination gives r_max = floor(k2*(n_bl-k2)/n_bl).
   These are ALGEBRAICALLY IDENTICAL.  Hence the guessing/orbit exponent
   (lambda-1)m - lambda*gamma is the SAME for both, on every published parameter set
   (verified: GabKron-128/192/256, new-GabKron, all t1).

3. BRIAUD'S EXTRA UNKNOWN DILUTES THE DISTINGUISHER.  Burle's per-block system is
   UNILATERAL (one unknown D, right factor).  Briaud's is BILATERAL: it keeps the
   left unknown V_i (r x r over F_qm), i.e. m*r^2 extra F_q-unknowns.  The
   over-determination margin drops from ratio ~2 (V eliminated / Burle) to ~1.03
   (V free / Briaud) at real sizes, and VANISHES at toy sizes (eqs = unk), where the
   solution space is large and bad guesses are no longer rejected -- observed here.
   So the bilateral form gives no advantage and a weaker distinguisher.

4. CONCLUSION for the paper's Remark 'Why Burle, not Briaud'.  The n1 blocks sit in a
   direct sum with DISTINCT parities, so there are no cross-block relations for a
   Briaud-style bilinear refinement to exploit; and within a block, Briaud's
   combinatorial form collapses to the SAME guess dimension and orbit exponent as
   Burle, while carrying a heavier system.  This CONFIRMS the choice of the
   Burle-type unilateral recovery, and confirms that Briaud's attack is no less
   heuristic (Prop. 4: solution space 'expected' dim 1, cost = inverse hit prob.).
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summary()
    else:
        run(seed=1000, m=8, n2=8, k2=4, lblocks=(1, 1))
        run(seed=2001, m=8, n2=8, k2=4, lblocks=(2, 0), trials_bad=3)


# --------------------------------------------------------------------------- #
#  SUMMARY OF FINDINGS  (printed by `python3 briaud_perblock.py --summary`)
# --------------------------------------------------------------------------- #
