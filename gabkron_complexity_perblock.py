"""
For a guess F in Gr_r(q,m), the recovery system has
    U = m*n*r    unknowns over F_q   (D in M_{n,m}(F), n the code length)
    E = m*k*p    equations over F_q  (p = parity row count)
The Burle distinguishing mechanism needs MORE equations than unknowns, U <= E, i.e.
    n*r <= k*p   =>   r* = floor(k*p / n)      (largest r keeping it over-determined),
subject to r* >= lambda.  The trial exponent is ((lambda-1)m - lambda r*)*log2(q):

    log2 W = omega*log2(m*k*n1*p) + ((lambda-1)*m - lambda*r*) * log2(q).

(GabKron: n=n1*n2, k=k1*k2, p=n2-t1-k2.)

THREE linear-algebra exponents are reported, in decreasing order of realism:
    omega = 2.8074  (STRASSEN = log2 7): the smallest exponent achieved by an algorithm
                     one can actually run at these sizes (Strassen / Strassen-Winograd,
                     and M4RI-style routines over F_2).  This is the OPERATIONAL column:
                     a value below the claimed level here is a concrete break.
    omega = 3       (schoolbook / plain Gaussian elimination): conservative proxy.
    omega = 2.37    (Alman et al. 2025, laser method on Coppersmith-Winograd powers):
                     an ASYMPTOTIC bound with no implementation at any realisable
                     dimension.  Reported for reference only; a break that appears solely
                     in this column is flagged, never counted as operational.
Pure Python (python3).
"""
from math import log2

STR = 2.8074  # Strassen exponent = log2(7), the operational floor over F_2

def rstar(k, p, n):
    return (k * p) // n

def _log2_gauss(m, l, q=2):
    if l < 0 or l > m:
        return float("-inf")
    s = 0.0
    for i in range(l):
        s += log2(q ** (m - i) - 1) - log2(q ** (l - i) - 1)
    return s

def log2_inv_S1(m, lam, r, q=2):
    """log2 of the exact expected guess count 1/S1 (Lemma: two-sided count).
    Leading order is ((lam-1)m - lam r) but the Gaussian form is exact and makes the
    accelerated regime coincide with the proven one at r = lambda."""
    return -(log2(q ** m - 1) - log2(q - 1)
             + _log2_gauss(m - lam, r - lam, q) - _log2_gauss(m, r, q))

def logW(neq, m, lam, r, omega, q=2):
    # exact 1/S1 guessing term (was: ((lam-1)*m - lam*r)*log2(q), leading order only)
    return omega * log2(neq) + log2_inv_S1(m, lam, r, q)

def triple(neq, m, lam, r, q=2):
    """(W@Strassen 2.807, W@3, W@2.37) for one recovery system."""
    return (logW(neq, m, lam, r, STR, q),
            logW(neq, m, lam, r, 3.0, q),
            logW(neq, m, lam, r, 2.37, q))

def gabrow(name, n1, k1, n2, k2, q, m, lam, cl, t1):
    n, k = n1 * n2, k1 * k2; p = n2 - t1 - k2
    r = rstar(k, p, n); neq = m * k * p          # single-copy (one system), was m*k*n1*p
    ws, w3, w237 = triple(neq, m, lam, r, q)
    return name, cl, t1, (n2 - k2)//2, r, ws, w3, w237, r>=lam

def singlerow(name, m, n, k, gamma, q, lam, cl, kp=None):
    kk = kp if kp is not None else k; p = n - gamma - k
    r = rstar(kk, p, n); neq = m * kk * p
    ws, w3, w237 = triple(neq, m, lam, r, q)
    return name, cl, lam, r, ws, w3, w237, r>=lam

def b(w, cl): return f"*{w:.1f}*" if w < cl else f"{w:.1f}"

def scan_original(name, n1,k1,n2,k2,q,m,lam,cl):
    """Scan all t1 in [1,t2]; return min and max of W over t1 (max ranked by the
    conservative omega=3 column, the worst case for the attacker)."""
    n,k=n1*n2,k1*k2; t2=(n2-k2)//2; rows=[]
    for t1 in range(1,t2+1):
        p=n2-t1-k2; r=rstar(k,p,n); neq=m*k*p
        ws,w3,w237=triple(neq,m,lam,r,q)
        rows.append((t1,r,ws,w3,w237))
    return name,cl,t2,min(rows,key=lambda x:x[3]),max(rows,key=lambda x:x[3])

ORIG = [("GabKron-128",2,2,24,12,2,48,3,128),
        ("GabKron-192",2,2,38,19,2,76,3,192),
        ("GabKron-256",2,2,52,26,2,104,3,256)]
NEW  = [("new-GabKron-128",2,2,90,18,2,90,3,128,6),
        ("new-GabKron-192",2,2,120,32,2,120,3,192,8),
        ("new-GabKron-256",2,2,128,40,2,128,3,256,8)]

print("="*104)
print(" GabKron per-block W(t1)  --- OVER-DETERMINED Burle regime  r_max = floor(kp/n)")
print(" Columns: W(2.807)=Strassen OPERATIONAL | W(3)=conservative | W(2.37)=asymptotic ref")
print(" Original sets: t1 not fixed, W(t1) non-monotone; MIN and MAX over t1 (max ranked by W@3).")
print("="*104)
print(f"{'scheme':<16}{'':<5}{'claimed':>8}{'t2':>4}{'t1':>4}{'r_max':>6}"
      f"{'W(2.807)':>10}{'W(3)':>9}{'W(2.37)':>10}")
print("-"*104)
for name,cl,t2,lo,hi in [scan_original(*o) for o in ORIG]:
    for tag,(t1,r,ws,w3,w237) in (("min",lo),("max",hi)):
        head = name if tag=="min" else ""
        print(f"{head:<16}{tag:<5}{(cl if tag=='min' else ''):>8}{(t2 if tag=='min' else ''):>4}"
              f"{t1:>4}{r:>6}{b(ws,cl):>10}{b(w3,cl):>9}{b(w237,cl):>10}")
print("-"*104)
for (name,n1,k1,n2,k2,q,m,lam,cl,t1) in NEW:
    n,k=n1*n2,k1*k2; p=n2-t1-k2; r=rstar(k,p,n); neq=m*k*p; t2=(n2-k2)//2
    ws,w3,w237=triple(neq,m,lam,r,q)
    print(f"{name:<16}{'':<5}{cl:>8}{t2:>4}{t1:>4}{r:>6}"
          f"{b(ws,cl):>10}{b(w3,cl):>9}{b(w237,cl):>10}")
print("-"*104)
print(" (*..* below claimed. OPERATIONAL verdict = W(2.807); W(2.37) below-claim alone = asymptotic only.)")

print("\n"+"="*104); print(" LGRH   r*=floor(k(n-g-k)/n)   lambda=2"); print("="*104)
print(f"{'set':<12}{'claimed':>8}{'r*':>4}{'W(2.807)':>10}{'W(3)':>9}{'W(2.37)':>10}  verdict (operational = 2.807)")
print("-"*104)
for row in [singlerow("LGRH-128",98,89,10,11,2,2,128), singlerow("LGRH-192",165,122,14,14,2,2,192)]:
    name,cl,lam,r,ws,w3,w237,ok = row
    if ws<cl:      v="OPERATIONAL break (Strassen)"
    elif w3<cl:    v="break at omega<=2.807"
    elif w237<cl:  v="ASYMPTOTIC ONLY (omega=2.37)"
    else:          v="not broken"
    print(f"{name:<12}{cl:>8}{r:>4}{b(ws,cl):>10}{b(w3,cl):>9}{b(w237,cl):>10}  {v}")

print("\n"+"="*104); print(" Modification II  (gamma=2)"); print("="*104)
print(f"  {'set':<10}{'lam':>4}{'r*':>4}{'W(2.807)':>10}{'W(3)':>9}{'W(2.37)':>10}")
for (nm,m,n,k,cl) in [("ModII-132",88,88,48,132),("ModII-192",98,98,52,192),("ModII-279",129,129,65,279)]:
    for lam in (2,3):
        name,c,l,r,ws,w3,w237,ok = singlerow(nm,m,n,k,2,2,lam,cl)
        print(f"  {nm:<10}{lam:>4}{r:>4}{b(ws,cl):>10}{b(w3,cl):>9}{b(w237,cl):>10}")

print("\n"+"="*104); print(" Modification I  (subcode k'=k-l, parity n-k)"); print("="*104)
print(f"  {'set':<10}{'lam':>4}{'r*':>4}{'W(2.807)':>10}{'W(3)':>9}{'W(2.37)':>10}")
for (nm,m,n,k,l,cl) in [("ModI-136",85,85,43,2,136),("ModI-203",98,98,50,3,203),("ModI-276",121,121,61,4,276)]:
    kp=k-l
    for lam in (2,3):
        p=n-k; r=rstar(kp,p,n); neq=m*kp*p
        ws,w3,w237=triple(neq,m,lam,r)
        print(f"  {nm:<10}{lam:>4}{r:>4}{b(ws,cl):>10}{b(w3,cl):>9}{b(w237,cl):>10}")

print("\n  (omega=2.37 and omega=3 must be unchanged):")
_,_,_,_,g128hi = scan_original("GabKron-128",2,2,24,12,2,48,3,128)
print(f"   GabKron-128 max: W(2.807)={g128hi[2]:.1f}  W(3)={g128hi[3]:.1f}  W(2.37)={g128hi[4]:.1f}  at t1={g128hi[0]}")
_k=2*40; _n=2*128; _p=128-8-40; _r=rstar(_k,_p,_n)
_ws,_w3,_w237=triple(128*_k*_p,128,3,_r)
print(f"   new-GabKron-256: W(2.807)={_ws:.1f}  W(3)={_w3:.1f} (was 242.9)  W(2.37)={_w237:.1f} (was 229.9)")
r=singlerow("LGRH-128",98,89,10,11,2,2,128); print(f"   LGRH-128: W(2.807)={r[4]:.1f}  W(3)={r[5]:.1f}  W(2.37)={r[6]:.1f} (was 122.0)")
r=singlerow("LGRH-192",165,122,14,14,2,2,192); print(f"   LGRH-192: W(2.807)={r[4]:.1f}  W(3)={r[5]:.1f}  W(2.37)={r[6]:.1f} (was 187.0)")

# Rep-GabKron (Lau et al.): resists at every width.
print("   --- Rep-GabKron (resists) ---")
for nm, n2, k2, m, t1, claim in [("Rep-GabKron-128",105,35,211,7,128),
                                 ("Rep-GabKron-192",150,50,307,10,192),
                                 ("Rep-GabKron-256",165,55,331,11,256)]:
    n, k = 2*n2, 2*k2
    p = n2 - t1 - k2; r = rstar(k, p, n); neq = m*k*p
    ws,w3,w237 = triple(neq, m, 3, r)
    print(f"   {nm}: W(2.807)={ws:.1f} W(3)={w3:.1f} W(2.37)={w237:.1f} at rho=t1={t1} (claim {claim}) -> resists")
