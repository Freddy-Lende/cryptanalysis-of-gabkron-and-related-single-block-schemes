"""
residual_lambdap.py  --  new-GabKron (Lau et al.) at the PUBLISHED error weight.

After the structural recovery (full-rank key D_F valued in a scalar multiple alpha V of the
masking subspace, dim lambda), applying D_F to a published ciphertext gives, per block, a
projected error  f = e D_F  of rank  w <= lambda*t, while the supercode ker(H0) uniquely
corrects only  tau = floor(p/2),  p = n2 - t1 - k2.  The excess is  Delta = w - tau.

WHY THE PUBLISHED WEIGHT IS NOT BROKEN (two routes, both above the security level):

(1) GENERIC error-erasure decoding beyond the unique rank radius. To correct a rank-w error
    with a radius-tau code, guess s support dimensions to treat as erasures; the decoder
    succeeds iff  2(w-s)+s <= p, i.e.  s >= 2w - p = 2*Delta.  Guessing an s-dim subspace
    INSIDE the (secret) rank-w error support of F_q^m costs about  [m,s]_q / [w,s]_q  trials.
    For the published sets this is >= 312 bits -- above the claimed level. (The earlier
    manuscript figure delta*(m-delta)=252 was wrong: it used s=Delta instead of s>=2*Delta,
    and even log2[128,2]_2 = 253.4 already exceeds 252.)

(2) STRUCTURED decoding. Writing alpha V = <v_1,...,v_lambda> (known after recovery),
    D_F = sum_a v_a D^{(a)} with D^{(a)} over F_q, so  f = e D_F = sum_a v_a g_a  with
    g_a = e D^{(a)} all supported on E = Supp(e) (dim t, secret). Recovering f amounts to
    recovering E, i.e. the rank-t error e -- the scheme's OWN rank-decoding problem, tuned
    to ~2^{claim}. Only one combined syndrome is available (not lambda independent ones), so
    a Rank-Support-Learning speed-up does not directly apply; a dedicated structured attack
    would have to be described and proven, and is not established here.

Conclusion: no residual route drops below the security level. new-GabKron is broken only at
the WEAKENED weight t_weak = floor((n2-k2-2 t1)/(2 lambda)) (where lambda*t_weak <= floor(p/2),
so the recovered key decodes directly); at the PUBLISHED weight its status is open (resistant
to the generic residual). Pure standard library.
"""
import math


def log2_gauss(a, b, q=2):
    if b < 0 or b > a:
        return float("-inf")
    return sum(math.log2((q ** (a - i) - 1) / (q ** (i + 1) - 1)) for i in range(b))


SETS = [  # name, n2, k2, t1, t_pub, lam, m, claim
    ("new-GabKron-128", 90, 18, 6, 12, 3, 90, 128),
    ("new-GabKron-192", 120, 32, 8, 14, 3, 120, 192),
    ("new-GabKron-256", 128, 40, 8, 14, 3, 128, 256),
]


def main():
    print("=" * 104)
    print(" new-GabKron at the PUBLISHED weight: residual decoding cost (correct error-erasure model)")
    print("   generic: guess s >= 2*Delta support dims; cost ~ [m,s]_q / [w,s]_q")
    print("=" * 104)
    print(f"{'set':17}{'m':>4}{'w=lam*t':>8}{'p':>5}{'tau':>5}{'Delta':>6}{'s>=2D':>6}"
          f"{'log2 cost':>10}{'t_weak':>8}{'claim':>6}  verdict@published")
    for nm, n2, k2, t1, t, lam, m, claim in SETS:
        w = lam * t
        p = n2 - t1 - k2
        tau = p // 2
        Delta = w - tau
        s = 2 * Delta
        cost = log2_gauss(m, s) - log2_gauss(w, s)
        t_weak = (n2 - k2 - 2 * t1) // (2 * lam)
        verdict = "BROKEN" if cost < claim else "NOT broken (residual >= claim)"
        print(f"{nm:17}{m:>4}{w:>8}{p:>5}{tau:>5}{Delta:>6}{s:>6}{cost:>10.1f}"
              f"{t_weak:>8}{claim:>6}  {verdict}")
    print("-" * 104)
    print(" Structured route (f = sum_a v_a g_a, g_a supported on E=Supp(e)) reduces to rank-t")
    print(" decoding, i.e. the scheme's own RSD (~2^claim); one combined syndrome only, so no")
    print(" direct RSL speed-up. Not established below the level.")
    print(" => Only the WEAKENED weight t_weak is broken; the published weight is OPEN.")


if __name__ == "__main__":
    main()
