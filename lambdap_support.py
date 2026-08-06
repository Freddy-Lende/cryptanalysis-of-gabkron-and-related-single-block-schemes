"""
lambdap_support.py  --  Reviewer point 1 (new-GabKron / Lau et al., arXiv:2410.06849).

Lau's improved GabKron uses, per column-block i of the scrambler P, a subspace
U_i = <Gamma_{i,1},...,Gamma_{i,lambda'}> of dimension lambda' <= lambda inside V, so that
    rk_q(e P_{C_i}) <= lambda' * t   (NOT lambda * t),
and correctness needs  lambda' t + t1 <= t2 = floor((n2-k2)/2).  For the three published
sets (lambda=3, t=12,14,14, t1=6,8,8) this forces lambda' = 2, and then
    lambda t  = 36,42,42  >  floor(p/2) = 33,40,40  >=  lambda' t = 24,28,28,
so a key that only respects the GLOBAL lambda=3 support does NOT decode published ciphertexts,
while one that respects the LOCAL lambda'=2 support does.

QUESTION (reviewer option 1): does the attack's extracted key D_F inherit the local lambda'
support per block?  We test it directly: build a faithful lambda'-structured instance at the
PUBLISHED regime (lambda t > floor(p/2) >= lambda' t) and measure, per module-block, the
F_q-rank of the error contribution (e D_F)_j.  If max_j rk <= lambda' t <= floor(p/2), the key
is decodable (option 1 works); if it reaches lambda t, it is not.

Model: q=2, n1=k1=2, V of dim lambda, U_i a random dim-lambda' subset of a fixed V-basis
(Lau sec.4.4). We use dense (non-circulant) blocks valued in U_i -- the circulant structure
of Lau is a key-size optimisation and does not change the support of e P_{C_i}.
Pure Python; reuses the verified attack primitives.
"""
import random
from gabkron_attack import (GF, check_primitive, red, moore, kron, matmul, matadd, inverse,
                            rand_subspace, rand_elt_of, independent_elements, extend_to,
                            gf2_rank_of, solve_public_system, basis_extract, gf2_nullspace)
import operator
xor = operator.xor


def rand_subspace_of_basis(F, Vbasis, dimp, rng):
    """U = span of a random size-dimp subset of the V-basis (Lau sec.4.4: Gamma_i chosen
    from {gamma_1,...,gamma_lambda})."""
    return list(rng.sample(list(Vbasis), dimp))


def build_lambdap_instance(m, n1, k1, n2, k2, lam, lamp, t1, t, seed):
    rng = random.Random(seed)
    F = GF(m); assert check_primitive(F)
    n, k = n1 * n2, k1 * k2
    Vb = rand_subspace(F, lam, rng)                                 # global V, dim lambda
    Ui = [rand_subspace_of_basis(F, Vb, lamp, rng) for _ in range(n1)]  # per-block U_i, dim lambda'
    g2 = [F.pw(2, j) for j in range(n2)]; G2 = moore(F, g2, k2)
    G1 = moore(F, [F.pw(2, 1 + 3 * i) for i in range(n1)], k1)
    GKP = kron(F, G1, G2)
    # distortion X: t1 F_q-independent directions, one distorted column per block (colrk t1 per block)
    dirs = independent_elements(F, t1, rng)
    X = [[0] * n for _ in range(k)]
    for i in range(n1):
        cols_i = rng.sample(range(n2), t1)
        for d_idx, col in enumerate(cols_i):
            gcol = i * n2 + col
            patt = [rng.randint(0, 1) for _ in range(k)]
            if not any(patt): patt[0] = 1
            for a in range(k):
                X[a][gcol] = xor(X[a][gcol], F.mul(dirs[d_idx], patt[a]))
    # P: column j (block i=j//n2) valued in U_i  (Lau sec.4.4)
    while True:
        P = [[rand_elt_of(F, Ui[j // n2], rng) for j in range(n)] for _ in range(n)]
        try:
            Pi = inverse(F, P); break
        except ValueError:
            pass
    Gpub = matmul(F, matadd(GKP, X), Pi)
    p = n2 - t1 - k2
    m_true = [rng.randrange(F.QM) for _ in range(k)]
    dirs_e = independent_elements(F, t, rng); pos = rng.sample(range(n), t)
    ev = [0] * n
    for i in range(t):
        ev[pos[i]] = dirs_e[i]
    y = [xor(red([F.mul(m_true[a], Gpub[a][j]) for a in range(k)]), ev[j]) for j in range(n)]
    return dict(F=F, n1=n1, k1=k1, n2=n2, k2=k2, n=n, k=k, lam=lam, lamp=lamp, Vb=Vb, Ui=Ui,
                t1=t1, t=t, p=p, Gpub=Gpub, m_true=m_true, y=y, e=ev, P=P)


def measure(I, seed):
    """Recover with the correct global guess (dim r_max), then measure the per-module-block
    F_q-rank of the error contribution (e D_F)_j. Returns (max_rank, d, glob_support, radius)."""
    F, k, n, n2, k2, t1, lam = I['F'], I['k'], I['n'], I['n2'], I['k2'], I['t1'], I['lam']
    p = I['p']; r_max = (k * p) // n
    rng = random.Random(seed ^ 0x9e3779b9)
    while True:
        h0 = [rng.randrange(1, F.QM) for _ in range(F.m)]
        if gf2_rank_of(F, h0) == F.m: break
    H0 = moore(F, h0, p)
    Fg = extend_to(F, I['Vb'], r_max, rng)                         # correct GLOBAL guess, dim r_max
    keyD, d, sup, ir = basis_extract(I, H0, Fg, h0)
    if keyD is None:
        return None
    ncols = len(keyD[0])
    eD = [red([F.mul(I['e'][i], keyD[i][c]) for i in range(n)]) for c in range(ncols)]
    per_block = [gf2_rank_of(F, eD[j * F.m:(j + 1) * F.m]) for j in range(d)]
    return dict(maxrank=max(per_block), d=d, glob_sup=sup, radius=p // 2,
                ir_eq_k=(ir == k), per_block=per_block)


def enumerate_lambdap_subspaces(F, Vb, lamp):
    """All lambda'-dim F_q-subspaces of span(V) (the option-1 search space). Returns bases."""
    from itertools import combinations
    from gabkron_attack import fq_span
    elts = [e for e in fq_span(F, Vb) if e != 0]
    seen, out = set(), []
    for combo in combinations(elts, lamp):
        if gf2_rank_of(F, list(combo)) != lamp:
            continue
        key = frozenset(fq_span(F, list(combo)))
        if key in seen:
            continue
        seen.add(key); out.append(list(combo))
    return out


def option1_test(m, n1, k1, n2, k2, lam, lamp, t1, t, N, seed0=9000):
    """Option 1 (reviewer): can exploiting the U_i give a key that is BOTH full-rank and
    lambda'-locally-supported (hence decodable at the published weight)? We try (a) the
    global dim-lambda guess and (b) EVERY dim-lambda' subspace of V (the search an attacker
    who knows V would run). Reports full-rank and decodability for each."""
    print(f"\n[Option 1] m={m} n2={n2} k2={k2} t1={t1} t={t} lam={lam} lam'={lamp}: "
          f"lam'*t={lamp*t}, lam*t={lam*t}")
    gfr = gdec = d2fr = d2dec = 0
    ncand = None
    for s in range(seed0, seed0 + N):
        I = build_lambdap_instance(m, n1, k1, n2, k2, lam, lamp, t1, t, s)
        F, n, k, p = I['F'], I['n'], I['k'], I['p']
        radius, r_max = p // 2, (k * p) // n
        rng = random.Random(s ^ 0x1234)
        while True:
            h0 = [rng.randrange(1, F.QM) for _ in range(F.m)]
            if gf2_rank_of(F, h0) == F.m:
                break
        H0 = moore(F, h0, p)
        Fg = extend_to(F, I['Vb'], r_max, rng)
        kD, dm, sup, ir = basis_extract(I, H0, Fg, h0)
        if kD is not None:
            gfr += 1
            eD = [red([F.mul(I['e'][i], kD[i][c]) for i in range(n)]) for c in range(len(kD[0]))]
            if max(gf2_rank_of(F, eD[j * F.m:(j + 1) * F.m]) for j in range(dm)) <= radius:
                gdec += 1
        cands = enumerate_lambdap_subspaces(F, I['Vb'], lamp); ncand = len(cands)
        hit = False
        for Ub in cands:
            kD2, dm2, sup2, ir2 = basis_extract(I, H0, Ub, h0)
            if kD2 is not None:
                d2fr += 1
                eD = [red([F.mul(I['e'][i], kD2[i][c]) for i in range(n)]) for c in range(len(kD2[0]))]
                if max(gf2_rank_of(F, eD[j * F.m:(j + 1) * F.m]) for j in range(dm2)) <= radius:
                    hit = True; break
        if hit:
            d2dec += 1
    print(f"   #dim-{lamp} subspaces of V = {ncand} per instance (the option-1 search cost)")
    print(f"   global dim-{lam} guess : full-rank {gfr}/{N}, decodable at published t {gdec}/{N}")
    print(f"   dim-{lamp} guesses      : full-rank {d2fr}/{N * ncand if ncand else 0} tries, "
          f"some subspace decodable {d2dec}/{N}")
    print(f"   => option 1 {'WORKS' if d2dec else 'FAILS'}: no key is both full-rank and "
          f"lambda'-supported (dim-{lam} decodes never, dim-{lamp} full-rank never).")


def main():
    # n1=k1=2, lambda=3, lambda'=2. Published regime: lambda t > floor(p/2) >= lambda' t.
    # m=16,n2=16,k2=6,t1=1,t=2: p=9, floor(p/2)=4 ; lambda' t=4 (fits), lambda t=6 (fails).
    cfgs = [
        dict(m=16, n1=2, k1=2, n2=16, k2=6, lam=3, lamp=2, t1=1, t=2, N=8),
        dict(m=18, n1=2, k1=2, n2=18, k2=6, lam=3, lamp=2, t1=1, t=2, N=6),
    ]
    print("=" * 100)
    print(" new-GabKron local-support test (reviewer pt 1): does the extracted key respect U_i (dim lambda')?")
    print("   published regime: lambda*t > floor(p/2) >= lambda'*t  =>  a global-lambda key CANNOT decode")
    print("=" * 100)
    for c in cfgs:
        lam, lamp, t = c['lam'], c['lamp'], c['t']
        print(f"\n cfg m={c['m']} n1={c['n1']} n2={c['n2']} k2={c['k2']} t1={c['t1']} t={t} "
              f"lambda={lam} lambda'={lamp}")
        print(f"   radius floor(p/2), lambda'*t={lamp*t}, lambda*t={lam*t}")
        hist = {}
        okcount = 0
        for s in range(7000, 7000 + c['N']):
            I = build_lambdap_instance(c['m'], c['n1'], c['k1'], c['n2'], c['k2'],
                                       lam, lamp, c['t1'], t, s)
            r = measure(I, s)
            if r is None:
                print("   (no key extracted)"); continue
            hist[r['maxrank']] = hist.get(r['maxrank'], 0) + 1
            okcount += (r['maxrank'] <= r['radius'])
            rad = r['radius']
        print(f"   max per-block rk(e D_F) histogram: {dict(sorted(hist.items()))}")
        print(f"   radius = {rad}; decodable (maxrank<=radius): {okcount}/{c['N']}")
        print(f"   -> lambda'*t={lamp*t} would be decodable; lambda*t={lam*t} would not.")
        reaches_lamt = sum(v for kk, v in hist.items() if kk > rad)
        if reaches_lamt:
            print(f"   VERDICT: {reaches_lamt}/{c['N']} instances reach rk>{rad} (up to lambda*t): the GLOBAL")
            print(f"            key does NOT preserve local lambda' support -> published ciphertexts do")
            print(f"            not decode from the global recovery alone.")


    print("\n" + "=" * 100)
    print(" OPTION 1 test: exploit U_i explicitly -- enumerate dim-lambda' subspaces of V")
    print("=" * 100)
    option1_test(16, 2, 2, 16, 6, 3, 2, 1, 2, N=6)


if __name__ == "__main__":
    main()
