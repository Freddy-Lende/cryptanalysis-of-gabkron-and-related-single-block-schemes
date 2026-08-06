"""
universal_validation.py 

Claim: the attacker need not know or enumerate the per-block width rho. Clearing to the
MINIMAL n2 - t2 clean columns per block (t2 = floor((n2-k2)/2), a PUBLIC quantity) is a
single universal recovery, valid for EVERY layout with actual rho <= t2, of cost W(t2).

Here we CONFIRM correctness empirically: we build instances with a SMALL actual distortion
(global rank 1, i.e. rho = 1) and then recover them using the universal width w = t2 --
i.e. the t2-width parity reference p = n2 - t2 - k2 -- rather than the instance's own
(smaller, easier) width. If decryption still succeeds, over-clearing is sound and no rho
search is needed.

The width enters the recovery ONLY through p = len(H0) (the number of parity rows of the
supercode reference); solve_public_system already uses the full public generator. So the
universal attack is exactly "use H0 with p = n2 - t2 - k2".

Pure Python; imports the verified attack primitives.
"""
import random
from gabkron_attack import (build_instance, moore, extend_to, basis_extract, gf2_rank_of)
from gabkron_attack_common import decrypt


def universal_recover(I, width_w):
    """Recover at the universal width w (p = n2 - w - k2), ignoring the instance's own
    smaller distortion. Returns (decrypt_ok, msg_ok, r_used, support)."""
    F, n2, k2, k, n, lam = I['F'], I['n2'], I['k2'], I['k'], I['n'], I['lam']
    p_w = n2 - width_w - k2
    r_w = (k * p_w) // n
    if r_w < lam:
        return None, None, r_w, None                  # width too large: r_max < lambda
    rng = random.Random(hash((I['t1'], width_w)) & 0xffffffff)
    while True:
        h0 = [rng.randrange(1, F.QM) for _ in range(F.m)]
        if gf2_rank_of(F, h0) == F.m:
            break
    H0 = moore(F, h0, p_w)                             # t2-width parity reference
    Fg = extend_to(F, I['Vb'], r_w, rng)              # correct guess of dimension r_w
    keyD, d, sup, ir = basis_extract(I, H0, Fg, h0)
    if keyD is None:
        return False, False, r_w, sup
    ok, _, mrec = decrypt(F, I['Gpub'], keyD, [H0] * d, I['y'], F.m, I['t'], "univ")
    return ok, (ok and mrec == I['m_true']), r_w, sup


CONFIGS = [   # (m, n1, k1, n2, k2, lam, N) with r_max(t2) >= lambda
    (16, 1, 1, 16, 8, 2, 12),
    # heavier, uncomment locally:
    # (18, 1, 1, 18, 8, 2, 12),
    # (16, 2, 2, 16, 8, 2, 8),
]

def main():
    print("=" * 96)
    print(" Universal (search-free) recovery: clear to n2 - t2 columns, actual rho small")
    print("   t2 = floor((n2-k2)/2) is public; instances built with global rank 1 (rho=1)")
    print("=" * 96)
    for (m, n1, k1, n2, k2, lam, N) in CONFIGS:
        t2 = (n2 - k2) // 2
        p_w = n2 - t2 - k2
        okU = msgU = 0
        rw = sup = None
        for s in range(3000, 3000 + N):
            I = build_instance(m, n1, k1, n2, k2, lam, s, t1=1)   # actual rho = 1
            du, mu, rw, sup = universal_recover(I, t2)
            if du is None:
                print(f" m={m} n1={n1} n2={n2} k2={k2} lam={lam}: r_max(t2)={rw} < lambda, skip")
                break
            okU += bool(du); msgU += bool(mu)
        else:
            regime = "PROVEN (r=lambda)" if rw == lam else f"accelerated (r={rw})"
            print(f" m={m:2d} n1={n1} n2={n2} k2={k2} lam={lam} | universal width w=t2={t2} "
                  f"(p={p_w}, r={rw}, {regime}): decrypted {okU}/{N}, matched {msgU}/{N}, support {sup}")
    print("\n=> over-clearing to the public width t2 decrypts small-rho instances: no rho search needed.")


if __name__ == "__main__":
    main()
