# Generate concrete LaTeX matrices for the per-block (spread) example.
import io, contextlib, random, operator
xor=operator.xor
with contextlib.redirect_stdout(io.StringIO()):
    from structure import GF, matmul, transpose, is_zero, rank, right_kernel, gf2_kernel, moore, stab_basis
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
    k,n=len(X),len(X[0]); rows=[[F.bits(X[i][j])[b] for j in range(n)] for i in range(k) for b in range(F.m)]
    return gf2_kernel(rows)
def clear_to_left(F,Xi,n2,rng):
    K=fq_kernel(F,Xi); rng.shuffle(K); right=[]
    for v in K:
        if gf2_rank(right+[v])>len(right): right.append(v)
    left=[]
    for i in range(n2):
        e=[1 if t==i else 0 for t in range(n2)]
        if gf2_rank(right+left+[e])>len(right)+len(left): left.append(e)
        if len(left)+len(right)==n2: break
    allc=left+right; T=[[allc[c][r] for c in range(n2)] for r in range(n2)]
    inverse(F,T); return T,len(left)

# field GF(16), a^4+a+1 (same as mono example)
m=4; ss.IRRED.setdefault(4,0b10011); F=GF(4)
n2,k2=4,2; n1=k1=2; n,k=n1*n2,k1*k2
# but t1<=1 here -> spread impossible; use n2=6,k2=2,m=6 instead for spread
m=6; ss.IRRED.setdefault(6,0b1000011); F=GF(6)
n2,k2=6,2; n1=k1=2; n,k=n1*n2,k1*k2
v2=[F.pw(2,j) for j in range(n2)]; G2=moore(F,v2,k2)
v1=[F.pw(2,1),F.pw(2,4)]; G1=moore(F,v1,k1); GKP=kron(F,G1,G2)
def show(x):
    if x==0:return '0'
    for e in range(F.QM-1):
        if F.pw(2,e)==x:return 'a^{%d}'%e if e>1 else ('a' if e==1 else '1')
    return str(x)
def mat(M): return '\\\\\n'.join(' & '.join(show(x) for x in r) for r in M)

rng=random.Random(11)
e=F.pw(2,2)
def colrank1():
    c=[F.mul(e,rng.randint(0,1)) for _ in range(k)]
    while all(x==0 for x in c): c=[F.mul(e,rng.randint(0,1)) for _ in range(k)]
    w=[rng.randint(0,1) for _ in range(n2)]
    while all(x==0 for x in w): w=[rng.randint(0,1) for _ in range(n2)]
    return [[F.mul(c[i],w[j]) for j in range(n2)] for i in range(k)]
while True:
    Xb=[colrank1(),colrank1()]; X=[Xb[0][i]+Xb[1][i] for i in range(k)]
    if not is_zero(cols(X,list(range(0,n2)))) and not is_zero(cols(X,list(range(n2,2*n2)))): break  # genuinely spread: both Kronecker blocks non-zero
Ts=[];clean=[]
for i in range(n1):
    Ti,li=clear_to_left(F,Xb[i],n2,rng); Ts.append((Ti,li)); clean+=[i*n2+c for c in range(li,n2)]
T=ident(F,n)
for i in range(n1):
    Ti=Ts[i][0]
    for a in range(n2):
        for b in range(n2): T[i*n2+a][i*n2+b]=Ti[a][b]
GKPT=matmul(F,GKP,T); Gstar=cols(GKPT,clean); XT=matmul(F,X,T)   # X.T = (X'' | 0) per block
# distorted (non-clean) columns = the l_i left columns of each block; X'' = their gathered content
distorted=[i*n2+c for i in range(n1) for c in range(Ts[i][1])]
Xpp=cols(XT,distorted)                                            # X'' : the non-zero gathered part
print("l =",[Ts[0][1],Ts[1][1]],"clean cols:",len(clean),"of",n)
print("X clean zero:",is_zero(cols(XT,clean)),"  rk(G*) =",rank(F,Gstar),"/",k)
print("=== G1 ==="); print(mat(G1))
print("=== G2 ==="); print(mat(G2))
print("=== X^(1) ==="); print(mat(Xb[0]))
print("=== X^(2) ==="); print(mat(Xb[1]))
print("=== T1 ==="); print(mat(Ts[0][0]))
print("=== T2 ==="); print(mat(Ts[1][0]))
print("=== X.T = (X'' | 0) per block  (clean columns are zero) ==="); print(mat(XT))
print("=== X'' (gathered non-zero columns, one per block) ==="); print(mat(Xpp))
print("=== Gstar (rk above) ==="); print(mat(Gstar))
