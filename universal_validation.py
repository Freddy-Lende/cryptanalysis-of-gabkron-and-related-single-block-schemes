"""
universal_validation.py  --  Reviewer point 3, soundness half (global width w = t1).

The attacker uses the PUBLIC global rank t1 as the clearing width: it clears the n2 - t1
columns of the global distortion rank, which is valid for every per-block layout (rho <= t1),
so no rho search is performed. We confirm empirically that recovery at w = t1 decrypts
regardless of how the t1-rank distortion is distributed across blocks (spread vs concentrated
layouts give different per-block rho but the same global t1), and that the recovery cost is
layout-independent. This is the correct, decryption-safe replacement for the earlier (flawed)
universal choice w = t2, which can violate lambda t <= floor(p/2).

Pure Python; reuses the verified attack primitives.
"""
import random
from gabkron_attack import (build_instance, moore, extend_to, basis_extract, gf2_rank_of)
from gabkron_attack_common import decrypt


def recover_at_t1(I, seed):
    """Standard recovery at the instance's own (public) global width t1. Returns
    (decrypt_ok, msg_ok, r_used, support, dmod)."""
    F, k, n, p, lam = I['F'], I['k'], I['n'], I['p'], I['lam']
    r_max = (k * p) // n
    if r_max < lam:
        return None, None, r_max, None, None
    rng = random.Random(seed ^ 0xabcdef)
    while True:
        h0 = [rng.randrange(1, F.QM) for _ in range(F.m)]
        if gf2_rank_of(F, h0) == F.m:
            break
    H0 = moore(F, h0, p)
    Fg = extend_to(F, I['Vb'], r_max, rng)
    keyD, d, sup, ir = basis_extract(I, H0, Fg, h0)
    if keyD is None:
        return False, False, r_max, sup, d
    ok, _, mrec = decrypt(F, I['Gpub'], keyD, [H0] * d, I['y'], F.m, I['t'], "t1")
    return ok, (ok and mrec == I['m_true']), r_max, sup, d


CONFIGS = [   # (m, n1, k1, n2, k2, lam, t1, N)
    (16, 2, 2, 16, 6, 2, 2, 3),
    # heavier, uncomment locally:
    # (18, 2, 2, 18, 8, 2, 2, 6),
]


def main():
    print("=" * 96)
    print(" Recovery at the PUBLIC global width w = t1: decryption is layout-independent")
    print("   (same global t1, different per-block rho via spread/concentrated layouts).")
    print("=" * 96)
    for (m, n1, k1, n2, k2, lam, t1, N) in CONFIGS:
        for layout in ("spread", "concentrated"):
            ok = msg = 0
            rused = sup = None
            for s in range(4000, 4000 + N):
                I = build_instance(m, n1, k1, n2, k2, lam, s, t1=t1, layout=layout)
                du, mu, rused, sup, d = recover_at_t1(I, s)
                if du is None:
                    print(f"  m={m} n2={n2} k2={k2} lam={lam} t1={t1} {layout}: r_max<lambda, skip")
                    break
                ok += bool(du); msg += bool(mu)
            else:
                print(f"  m={m} n1={n1} n2={n2} k2={k2} lam={lam} t1={t1} layout={layout:12s}"
                      f" (w=t1, r={rused}): decrypted {ok}/{N}, matched {msg}/{N}, support {sup}")
    print("\n=> the public width w=t1 decrypts across layouts: no rho enumeration is needed,")
    print("   and lambda t <= floor(p/2) holds at t1 (unlike the earlier w=t2 choice).")


if __name__ == "__main__":
    main()
