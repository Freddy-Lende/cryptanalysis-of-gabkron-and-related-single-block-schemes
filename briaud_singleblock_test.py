"""
briaud_singleblock_test.py  --  direct solvability test of Briaud-Loidreau Eq.(3)
on ONE masked Gabidulin block, at a size where their Eq.(4) admits gamma>=lambda.

This isolates the question "does the Briaud constrained system (3) solve per block,
with a V-valued W, exactly as in their Prop. 4?" from the Kronecker plumbing.  A
GabKron clean block is precisely such a masked Gabidulin block, so a positive answer
here is what the per-block adaptation would rely on.

We build:  a [n_bl,k2] Gabidulin code G', mask it as Gpub = G' P^-1 with P in V-valued
GL, V of dimension lambda; then run Briaud's combinatorial guess (a gamma-dim U>=xV)
and solve (3) over F_q.  Good guess -> nonzero V-valued W; screened bad guess -> none.
"""
import random
import structure as ss
from structure import GF, matmul, moore, right_kernel, rank, inverse, cols
from gabkron_attack_common import red, gf2_rank_of
from briaud_perblock import (briaud_block_system, support_dim_of_W, normal_element,
                             extend_space, rand_subspace, fq_span_set)

def masked_gab_block(F, n_bl, k2, lam, rng):
    """Gpub = G' P^{-1}, G' a [n_bl,k2] Gabidulin code, P in GL_{n_bl}(V-valued)."""
    v = [F.pw(2, j) for j in range(n_bl)]
    G = moore(F, v, k2)
    Vb = []
    while len(Vb) < lam:
        x = rng.randrange(1, F.QM)
        if gf2_rank_of(F, Vb + [x]) == len(Vb) + 1:
            Vb.append(x)
    while True:
        P = [[Vb[rng.randint(0, len(Vb) - 1)] for _ in range(n_bl)] for _ in range(n_bl)]
        try:
            Pi = inverse(F, P); break
        except ValueError:
            pass
    Gpub = matmul(F, G, Pi)
    return Gpub, Vb

def run(m=8, n_bl=8, k2=4, lam=2, trials_bad=5, seed=7):
    print("=" * 92)
    print(f" Briaud Eq.(3) single-block solvability | m={m} n_bl={n_bl} k2={k2} lam={lam}")
    print("=" * 92)
    ss.IRRED.setdefault(m, {8: 0b100011011, 10: 0b10000001001, 12: 0b1000001010011}[m])
    F = GF(m); rng = random.Random(seed)
    r = n_bl - k2
    gamma = int(r * (1 - r / n_bl))
    print(f"  r={r}, Briaud Eq.(4) gamma_max=floor(r(1-r/n_bl))={gamma}, "
          f"feasible (gamma>=lam)={gamma >= lam}")
    if gamma < lam:
        print("  Eq.(4) infeasible at this size; pick larger n_bl or smaller k2."); return

    Gpub, Vb = masked_gab_block(F, n_bl, k2, lam, rng)
    alpha = normal_element(F)

    # GOOD guess: U of dim gamma containing a scalar multiple xV
    x = rng.randrange(1, F.QM)
    xV = [F.mul(x, v) for v in Vb]
    Ugood = extend_space(F, xV, gamma, rng)
    dim_g, W_g = briaud_block_system(F, Gpub, alpha, Ugood)
    supp_g = support_dim_of_W(F, W_g)
    print(f"  GOOD guess U>=xV (dim {gamma}): solution-space dim over F_q = {dim_g}, "
          f"recovered W support-dim = {supp_g}  (Briaud Prop.4 predicts dim>=1, supp<=lam={lam})")

    # BAD guesses screened to contain no multiple of V
    bad_hits = 0; bad_run = 0
    for _ in range(trials_bad):
        Ubad = rand_subspace(F, gamma, rng)
        Uset = fq_span_set(F, Ubad)
        if any(all(F.mul(y, v) in Uset for v in Vb) for y in range(1, F.QM)):
            continue
        bad_run += 1
        dim_b, W_b = briaud_block_system(F, Gpub, alpha, Ubad)
        if dim_b and W_b is not None and support_dim_of_W(F, W_b) is not None:
            # a genuinely nonzero V-valued solution from a bad guess would be a false positive
            if support_dim_of_W(F, W_b) and support_dim_of_W(F, W_b) <= lam:
                bad_hits += 1
    print(f"  BAD guesses (screened): {bad_hits}/{bad_run} produced a nonzero small-support W "
          f"(Briaud: 'no non-zero solution with overwhelming probability')")

if __name__ == "__main__":
    run(m=8, n_bl=8, k2=4, lam=2, seed=7)
    run(m=10, n_bl=10, k2=6, lam=2, seed=11)
    run(m=10, n_bl=10, k2=7, lam=2, seed=13)
