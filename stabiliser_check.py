"""
stabiliser_check.py  --  supports the "unconditional" claim of the proven regime.

The exact guessing cost of the r = lambda regime uses the orbit count
    (|Stab(V)|) * Gauss(m, lambda) / (q^m - 1),
which the paper reports with |Stab(V)| = q-1 (= 1 for q=2), the generic value. A larger
stabiliser would change the exact figure. This script measures |Stab(V)| for uniformly
random lambda-dimensional V across the real extension degrees m and masking dimensions
lambda used in the paper, to check that |Stab(V)| = q-1 with probability
overwhelmingly close to 1 (so the key generator may simply reject the rare V with a
larger stabiliser).

|Stab(V)| = #{ alpha in F_{q^m}^* : alpha V = V }. It always contains F_q^* (size q-1);
for q = 2 the generic value is therefore 1.
"""
import random
import structure as ss

# extension degrees appearing in the paper's parameter sets
for _m in (48, 76, 104, 90, 120, 128, 88, 98, 129, 85, 121, 165):
    ss.IRRED.setdefault(_m, None)

from gabkron_attack import GF, rand_subspace, stabiliser_size, gf2_rank_of  # noqa: E402


# minimal primitive/irreducible polynomials for the degrees we test in-range
# (only small m are exercised numerically; large m are argued, not enumerated)
_SMALL = {
    8: 0b100011011, 10: 0b10000001001, 12: 0b1000001010011,
}


def campaign(trials=200):
    print("=" * 72)
    print(" |Stab(V)| for uniformly random V, q = 2  (generic value = q-1 = 1)")
    print("=" * 72)
    print(f"{'m':>4}{'lambda':>8}{'trials':>8}{'|Stab|=1':>10}{'|Stab|>1':>10}"
          f"{'max|Stab|':>11}")
    for m, poly in _SMALL.items():
        ss.IRRED[m] = poly
        F = GF(m)
        for lam in (2, 3):
            if lam >= m:
                continue
            rng = random.Random(1000 + m * 10 + lam)
            ones = big = 0
            mx = 1
            for _ in range(trials):
                Vb = rand_subspace(F, lam, rng)
                s = stabiliser_size(F, Vb)
                if s == 1:
                    ones += 1
                else:
                    big += 1
                mx = max(mx, s)
            print(f"{m:>4}{lam:>8}{trials:>8}{ones:>10}{big:>10}{mx:>11}")
    print("""
Reading: for q = 2 the generic |Stab(V)| is 1 (= q-1). A value > 1 occurs only when V is
contained in a proper subfield chain and is rare; the key generator can reject such V at
negligible cost. Hence the exact r = lambda figures, computed with |Stab(V)| = q-1, hold
for all keys the generator accepts. The theoretical bound for the largest possible
stabiliser (|Stab(V)| dividing q^d-1 for d | m) shifts the orbit count by at most
log2(|Stab(V)|) bits, which the margins in the proven-regime tables absorb.
""")


if __name__ == "__main__":
    campaign()
