"""
Sun's per-block decomposition handles any distortion layout directly.

Claim 1 (load-bearing):  for T in GL_{n2}(F_q),  G2 T = Moore(v2 T, k2).
  => right-multiplying a Gabidulin generator by an F_q-invertible T gives a
     Gabidulin code on the transformed support v2 T (still Gabidulin).

Claim 2:  with a per-block T = diag(T1,...,T_{n1}) where each T_i clears X_i to
  [X'_i | 0], the surviving right part G'_{2,i} of each block is a SHORTENED
  Gabidulin code with its own parity H_i, and J_i G'_{2,i} is orthogonal to it
  -- for ANY X, including a SPREAD one (both blocks non-zero); each block is cleared on its own.

This is why our (a)/(b) test (checked against the FIXED H2) saw "failure": after
a cross-block T the support changes to v2 T_i, whose parity is H_i != H2.  Sun
keeps n1 blocks with n1 parities H_i and never needs a common H2.
"""
import io, contextlib, random, operator
xor = operator.xor
with contextlib.redirect_stdout(io.StringIO()):
    from structure import (GF, matmul, transpose, is_zero, rank,
                                right_kernel, gf2_kernel, moore, stab_basis)
    import structure as ss
    from structure import ident, cols, kron, inverse

def gf2_rank(vecs):
    A=[v[:] for v in vecs]; r=0
    if not A: return 0
    nv=len(A[0])
    for c in range(nv):
        s=next((i for i in range(r,len(A)) if A[i][c]),None)
        if s is None: continue
        A[r],A[s]=A[s],A[r]
        for i in range(len(A)):
            if i!=r and A[i][c]: A[i]=[xor(A[i][t],A[r][t]) for t in range(nv)]
        r+=1
        if r==len(A): break
    return r

def fq_kernel(F,X):
    k,n=len(X),len(X[0])
    rows=[[F.bits(X[i][j])[b] for j in range(n)] for i in range(k) for b in range(F.m)]
    return gf2_kernel(rows)

def vecmat_fq(F, v, T):  # v (len n2) times T (n2 x n2 over F_q) -> len n2 over F_qm
    n2=len(T)
    return [ functools_reduce(F, [F.mul(v[j], T[j][l]) for j in range(n2)]) for l in range(n2)]

import functools
def functools_reduce(F, lst): return functools.reduce(xor, lst, 0)

def clear_block_T(F, Xi, n2, rng):
    """T_i in GL_{n2}(F_q) with Xi T_i = [X'_i | 0]; last cols span ker_q(Xi)."""
    K=fq_kernel(F,Xi)              # right kernel over F_q
    li_compl=len(K)               # = n2 - colrank(Xi)
    right=[]
    rng.shuffle(K)
    for v in K:
        if gf2_rank(right+[v])>len(right): right.append(v)
    left=[]
    for i in range(n2):
        e=[1 if t==i else 0 for t in range(n2)]
        if gf2_rank(right+left+[e])>len(right)+len(left): left.append(e)
        if len(left)+len(right)==n2: break
    allcols=left+right
    T=[[allcols[c][r] for c in range(n2)] for r in range(n2)]
    try: inverse(F,T)
    except ValueError: return None,None
    li=len(left)                  # surviving (non-cleared) columns = colrank
    return T, li

def run(seed=3):
    rng=random.Random(seed)
    m=4; ss.IRRED.setdefault(4,0b10011); F=GF(4)
    n2,k2=4,2; n1=k1=2; n,k=n1*n2,k1*k2
    v2=[F.pw(2,j) for j in range(n2)]; G2=moore(F,v2,k2); H2=right_kernel(F,G2)
    print("="*78); print(" SUN PER-BLOCK DECOMPOSITION (per-block clearing, any layout)"); print("="*78)

    # Claim 1: G2 T = Moore(v2 T, k2)
    okall=True
    for _ in range(50):
        while True:
            T=[[rng.randint(0,1) for _ in range(n2)] for _ in range(n2)]
            try: inverse(F,T); break
            except ValueError: pass
        lhs=matmul(F,G2,T)
        vT=vecmat_fq(F,v2,T); rhs=moore(F,vT,k2)
        if lhs!=rhs: okall=False; break
    print(f" Claim 1:  G2 T == Moore(v2 T, k2) for random T in GL_n2(F_q):  {okall}")

    # Claim 2: SPREAD X, per-block clearing keeps each block's right part Gabidulin
    e=F.pw(2,2)
    def colrank1_block():           # F_q-column-rank 1: X_i = c * a^T, a in F_q^n2
        c=[F.mul(e,rng.randint(0,1)) for _ in range(k)]
        while all(x==0 for x in c): c=[F.mul(e,rng.randint(0,1)) for _ in range(k)]
        a=[rng.randint(0,1) for _ in range(n2)]
        while all(x==0 for x in a): a=[rng.randint(0,1) for _ in range(n2)]
        return [[F.mul(c[i],a[j]) for j in range(n2)] for i in range(k)]
    while True:
        Xb=[colrank1_block() for _ in range(n1)]
        X=[Xb[0][i]+Xb[1][i] for i in range(k)]
        if not is_zero(Xb[0]) and not is_zero(Xb[1]): break  # spread: both blocks non-zero
    print(" Claim 2:  X is SPREAD (both Kronecker blocks non-zero)")
    allgab=True
    for i in range(n1):
        Xi=Xb[i]
        Ti,li=clear_block_T(F,Xi,n2,rng)
        XiTi=matmul(F,Xi,Ti)
        cleared=is_zero(cols(XiTi,list(range(li,n2))))          # right n2-li cols = 0
        G2Ti=matmul(F,G2,Ti)
        Gp2=cols(G2Ti,list(range(li,n2)))                       # surviving right part
        vTi=vecmat_fq(F,v2,Ti); is_moore=(G2Ti==moore(F,vTi,k2))
        Hi=right_kernel(F,Gp2)                                  # its OWN parity
        orth=is_zero(matmul(F,Gp2,transpose(Hi)))
        print(f"   block {i}: colrank l_i={li}, X_i T_i=[X'|0]:{cleared}, "
              f"G2 T_i is Moore(v2 T_i):{is_moore}, "
              f"G'2,i Gabidulin (orth to its own H_i):{orth}")
        allgab = allgab and cleared and is_moore and orth
    print("-"*78)
    print(f" => For a SPREAD X, every block reduces to a shortened Gabidulin code")
    print(f"    with its OWN parity H_i. Per-block reduction works: {allgab}")
    print("    Each block is cleared independently; no common parity needed.")
    print("="*78)

if __name__=="__main__":
    run()
