"""
Corrected work factors in the OVER-DETERMINED Burle regime (reviewer's central point).

For a guess F in Gr_r(q,m), the recovery system has
    U = m*n*r    unknowns over F_q   (D in M_{n,m}(F), n the code length)
    E = m*k*p    equations over F_q  (p = parity row count)
The Burle distinguishing mechanism needs MORE equations than unknowns, U <= E, i.e.
    n*r <= k*p   =>   r* = floor(k*p / n)      (largest r keeping it over-determined),
subject to r* >= lambda.  The trial exponent is ((lambda-1)m - lambda r*)*log2(q):

    log2 W = omega*log2(m*k*n1*p) + ((lambda-1)*m - lambda*r*) * log2(q).

(GabKron: n=n1*n2, k=k1*k2, p=n2-t1-k2.)  This is the previous cost model with
r* = floor(kp/n) instead of floor(kp/n)+1, adding lambda*log2(q) bits and restoring
the over-determined regime.  Pure Python (python3).
"""
from math import log2

def rstar(k, p, n):
    return (k * p) // n

def logW(neq, m, lam, r, omega, q=2):
    return omega * log2(neq) + ((lam - 1) * m - lam * r) * log2(q)

def gabrow(name, n1, k1, n2, k2, q, m, lam, cl, t1):
    n, k = n1 * n2, k1 * k2; p = n2 - t1 - k2
    r = rstar(k, p, n); neq = m * k * n1 * p
    return name, cl, t1, (n2 - k2)//2, r, logW(neq,m,lam,r,2.37,q), logW(neq,m,lam,r,3.0,q), r>=lam

def singlerow(name, m, n, k, gamma, q, lam, cl, kp=None):
    kk = kp if kp is not None else k; p = n - gamma - k
    r = rstar(kk, p, n); neq = m * kk * p
    return name, cl, lam, r, logW(neq,m,lam,r,2.37,q), logW(neq,m,lam,r,3.0,q), r>=lam

def b(w, cl): return f"*{w:.1f}*" if w < cl else f"{w:.1f}"

def scan_original(name, n1,k1,n2,k2,q,m,lam,cl):
    """Scan all t1 in [1,t2]; return (name,cl,t2, (t1,r,W2.37,W3) at min and at max of W)."""
    n,k=n1*n2,k1*k2; t2=(n2-k2)//2; rows=[]
    for t1 in range(1,t2+1):
        p=n2-t1-k2; r=rstar(k,p,n); neq=m*k*n1*p
        rows.append((t1,r,logW(neq,m,lam,r,2.37,q),logW(neq,m,lam,r,3.0,q)))
    return name,cl,t2,min(rows,key=lambda x:x[3]),max(rows,key=lambda x:x[3])

ORIG = [("GabKron-128",2,2,24,12,2,48,3,128),
        ("GabKron-192",2,2,38,19,2,76,3,192),
        ("GabKron-256",2,2,52,26,2,104,3,256)]
NEW  = [("new-GabKron-128",2,2,90,18,2,90,3,128,6),
        ("new-GabKron-192",2,2,120,32,2,120,3,192,8),
        ("new-GabKron-256",2,2,128,40,2,128,3,256,8)]

print("="*96)
print(" GabKron per-block W(t1)  --- OVER-DETERMINED Burle regime  r_max = floor(kp/n)")
print(" Original sets: t1 is NOT fixed and W(t1) is non-monotone, so we scan every t1 in")
print(" [1,t2] and report the MIN and MAX of W with the weight t1 attaining each.")
print("="*96)
print(f"{'scheme':<16}{'':<5}{'claimed':>8}{'t2':>4}{'t1':>4}{'r_max':>6}{'W(2.37)':>10}{'W(3)':>9}")
print("-"*96)
for name,cl,t2,lo,hi in [scan_original(*o) for o in ORIG]:
    for tag,(t1,r,w237,w3) in (("min",lo),("max",hi)):
        head = name if tag=="min" else ""
        print(f"{head:<16}{tag:<5}{(cl if tag=='min' else ''):>8}{(t2 if tag=='min' else ''):>4}"
              f"{t1:>4}{r:>6}{b(w237,cl):>10}{b(w3,cl):>9}")
print("-"*96)
for (name,n1,k1,n2,k2,q,m,lam,cl,t1) in NEW:
    n,k=n1*n2,k1*k2; p=n2-t1-k2; r=rstar(k,p,n); neq=m*k*n1*p; t2=(n2-k2)//2
    print(f"{name:<16}{'':<5}{cl:>8}{t2:>4}{t1:>4}{r:>6}"
          f"{b(logW(neq,m,lam,r,2.37,q),cl):>10}{b(logW(neq,m,lam,r,3.0,q),cl):>9}")
print("-"*96)
print(" (*..* below claimed;  original rows give min/max of W over t1 with the weight t1)")

print("\n"+"="*92); print(" LGRH   r*=floor(k(n-g-k)/n)   lambda=2"); print("="*92)
print(f"{'set':<12}{'claimed':>8}{'r*':>4}{'W(2.37)':>10}{'W(3)':>9}  verdict")
print("-"*92)
for row in [singlerow("LGRH-128",98,89,10,11,2,2,128), singlerow("LGRH-192",165,122,14,14,2,2,192)]:
    name,cl,lam,r,w237,w3,ok = row
    print(f"{name:<12}{cl:>8}{r:>4}{b(w237,cl):>10}{b(w3,cl):>9}  "
          f"{'2.37:brk' if w237<cl else '2.37:sec'}, {'3:brk' if w3<cl else '3:sec'}")

print("\n"+"="*92); print(" Modification II  (gamma=2)"); print("="*92)
for (nm,m,n,k,cl) in [("ModII-132",88,88,48,132),("ModII-192",98,98,52,192),("ModII-279",129,129,65,279)]:
    for lam in (2,3):
        name,c,l,r,w237,w3,ok = singlerow(nm,m,n,k,2,2,lam,cl)
        print(f"  {nm:<10} lam={lam}  r*={r:>3}  W(2.37)={b(w237,cl):>8}  W(3)={b(w3,cl):>8}")

print("\n"+"="*92); print(" Modification I  (subcode k'=k-l, parity n-k)"); print("="*92)
for (nm,m,n,k,l,cl) in [("ModI-136",85,85,43,2,136),("ModI-203",98,98,50,3,203),("ModI-276",121,121,61,4,276)]:
    kp=k-l
    for lam in (2,3):
        p=n-k; r=rstar(kp,p,n); neq=m*kp*p
        print(f"  {nm:<9} lam={lam}  r*={r:>3}  W(2.37)={b(logW(neq,m,lam,r,2.37),cl):>8}  W(3)={b(logW(neq,m,lam,r,3.0),cl):>8}")

print("\n CROSS-CHECK vs reviewer's corrected values:")
_,_,_,_,g128hi = scan_original("GabKron-128",2,2,24,12,2,48,3,128)
print(f"   GabKron-128 max W(3)={g128hi[3]:.1f} at t1={g128hi[0]} (rev 128.9)")
_n,_k=2*128,2*40; _p=128-8-40; _r=rstar(_n and _k,_p,_n)
print(f"   new-GabKron-256 W(2.37)={logW(128*_k*2*_p,128,3,rstar(_k,_p,_n),2.37):.1f} (rev 229.9),"
      f" W(3)={logW(128*_k*2*_p,128,3,rstar(_k,_p,_n),3.0):.1f} (rev 242.9)")
r=singlerow("LGRH-128",98,89,10,11,2,2,128); print(f"   LGRH-128 W(2.37)={r[4]:.1f} (rev 122.0)")
r=singlerow("LGRH-192",165,122,14,14,2,2,192); print(f"   LGRH-192 W(2.37)={r[4]:.1f} (rev 187.0)")
