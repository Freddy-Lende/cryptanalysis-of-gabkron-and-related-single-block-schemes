"""
Does a SPREAD distortion really resist?  Decide via the residual dimension.

The recovered trapdoor has image rank rk(G_pub D) = rk(G*), where G* = clean
columns of G_KP T (a DIRECT structural rank, not a degenerate decryption test).
The residual RSD code <D'> has dimension  k - rk(G*),  and the residual cost is
   R = q^{ t*ceil(m*(resdim+1)/n) - m }.

Three scenarios (n1=k1=2):
  (A) SPREAD X, per-block T=diag(T1,T2) clearing each block's l_i columns
      -> clean part keeps ALL n1 blocks (loses only t1=sum l_i columns).
  (B) MONO-BLOCK X, paper consolidation T=diag(T1,I): clean = the OTHER full
      block only (the supported block is discarded) -> rk = (n1-1)k2.
  (C) MONO-BLOCK X, per-block T: clean keeps both blocks' survivors.

Prediction to test: per-block (A,C) give rk(G*)=k  =>  residual 0  =>  FULL
recovery for EVERY distortion at cost W; (B) is the suboptimal paper route.
Pure-Python, direct linear algebra.
"""
import io, contextlib, random, operator, math
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

def fq_colrank(F,X):
    k,n=len(X),len(X[0])
    return gf2_rank([[bit for i in range(k) for bit in F.bits(X[i][j])] for j in range(n)])

def clear_to_left(F,Xi,n2,rng):
    """T_i in GL_n2(F_q) with Xi T_i = [X'_i | 0]; right n2-l_i cols cleared."""
    K=fq_kernel(F,Xi); rng.shuffle(K)
    right=[]
    for v in K:
        if gf2_rank(right+[v])>len(right): right.append(v)
    left=[]
    for i in range(n2):
        e=[1 if t==i else 0 for t in range(n2)]
        if gf2_rank(right+left+[e])>len(right)+len(left): left.append(e)
        if len(left)+len(right)==n2: break
    allc=left+right; T=[[allc[c][r] for c in range(n2)] for r in range(n2)]
    inverse(F,T)              # raises if singular
    return T, len(left)       # l_i = #left (distorted) columns

def Rcost(q,t,m,resdim,n):
    if resdim==0: return 0,"resdim 0: FULL recovery, NO residual"
    expo=t*math.ceil(m*(resdim+1)/n)-m
    return max(expo,0),f"log2 R = {max(expo,0)}"

def block(X,s,n2): return cols(X,list(range(s*n2,(s+1)*n2)))

def run(seed=4, m=6, n2=6, k2=2, t1A=(1,1), t1mono=2):
    rng=random.Random(seed)
    ss.IRRED.setdefault(m,{4:0b10011,6:0b1000011,8:0b100011011}[m]); F=GF(m)
    n1=k1=2; n,k=n1*n2,k1*k2; lam=3; q=2
    t=max((n2-k2-2*max(sum(t1A),t1mono))//(2*lam),1)
    v2=[F.pw(2,j) for j in range(n2)]; G2=moore(F,v2,k2); H2=right_kernel(F,G2)
    v1=[F.pw(2,1),F.pw(2,3)]; G1=moore(F,v1,k1); GKP=kron(F,G1,G2)
    e=F.pw(2,2)
    print("="*82)
    print(f" RESIDUAL DIMENSION:  spread (per-block) vs mono-block (paper)   m={m},n2={n2},k2={k2},k={k}")
    print("="*82)

    def colrank_block(li):
        # F_q-column-rank exactly li : sum of li rank-1 layers c_a * w_a^T
        B=[[0]*n2 for _ in range(k)]
        for _ in range(li):
            c=[F.mul(e,rng.randint(0,1)) for _ in range(k)]
            w=[rng.randint(0,1) for _ in range(n2)]
            for i in range(k):
                for j in range(n2): B[i][j]^=F.mul(c[i],w[j])
        return B

    # ---------- (A) SPREAD X, per-block ----------
    while True:
        Xb=[colrank_block(t1A[0]), colrank_block(t1A[1])]
        X=[Xb[0][i]+Xb[1][i] for i in range(k)]
        if not is_zero(block(X,0,n2)) and not is_zero(block(X,1,n2)): break  # spread: both blocks non-zero
    Ts=[]; cleanidx=[]
    for i in range(n1):
        Ti,li=clear_to_left(F,Xb[i],n2,rng); Ts.append(Ti)
        cleanidx += [i*n2+c for c in range(li,n2)]      # surviving cols of block i
    T=ident(F,n)
    for i in range(n1):
        for a in range(n2):
            for b in range(n2): T[i*n2+a][i*n2+b]=Ts[i][a][b]
    XT=matmul(F,X,T); GKPT=matmul(F,GKP,T)
    cleanXT=is_zero(cols(XT,cleanidx))
    Gstar=cols(GKPT,cleanidx); rkA=rank(F,Gstar); resA=k-rkA
    rA,msgA=Rcost(q,t,m,resA,n)
    print(f"(A) SPREAD (both blocks distorted), per-block: l=({t1A[0]},{t1A[1]}), clean cols={len(cleanidx)}/{n}")
    print(f"    X clean cols zero:{cleanXT}  rk(G*)={rkA}/{k}  residual dim={resA}  {msgA}")

    # ---------- (B) MONO-BLOCK X, paper consolidation (clean = block 2 only) ----------
    X1=colrank_block(t1mono); Xmono=[X1[i]+[0]*n2 for i in range(k)]
    T1,l1=clear_to_left(F,[row[:n2] for row in Xmono],n2,rng)
    Tb=ident(F,n)
    for a in range(n2):
        for b in range(n2): Tb[a][b]=T1[a][b]
    GKPTb=matmul(F,GKP,Tb)
    cleanB=list(range(n2,n))                              # block 2 (the paper's clean part)
    GstarB=cols(GKPTb,cleanB); rkB=rank(F,GstarB); resB=k-rkB
    rB,msgB=Rcost(q,t,m,resB,n)
    print(f"(B) MONO  paper route (discard supported block): clean cols={len(cleanB)}/{n}")
    print(f"    rk(G*)={rkB}/{k}  residual dim={resB}  {msgB}")

    # ---------- (C) MONO-BLOCK X, per-block ----------
    cleanidxC=[c for c in range(l1,n2)]+list(range(n2,n))  # block1 survivors + all block2
    GstarC=cols(GKPTb,cleanidxC); rkC=rank(F,GstarC); resC=k-rkC
    rC,msgC=Rcost(q,t,m,resC,n)
    print(f"(C) MONO  per-block (keep block-1 survivors too): clean cols={len(cleanidxC)}/{n}")
    print(f"    rk(G*)={rkC}/{k}  residual dim={resC}  {msgC}")
    print("-"*82)
    print(" Per-block keeps n - t1 clean columns (loses only the distorted ones);")
    print(" the paper's mono route discards a whole block (n2 columns).")
    print("="*82)

if __name__=="__main__":
    for sd in range(5):
        run(seed=sd)
