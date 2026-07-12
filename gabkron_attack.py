"""
gabkron_attack.py  --  end-to-end GabKron key recovery, PER-BLOCK version.

This is the attack of the paper WITHOUT the uniform W_A variant: it uses only the
per-block recovery with parity p = n2 - t1 - k2 (Theorem "Alternative key recovery",
work factor W(t1)).  For each random spread-distortion instance it

  1. builds G_pub = (G_KP + X) P^{-1}, X of column rank t1, and a ciphertext
     y = m G_pub + e with rk(e) = t;
  2. clears each Kronecker block UNIFORMLY to exactly n2-t1 clean columns, builds a
     per-block full-column-rank T_{2,i} so that H0 = Moore(h0, n2-t1-k2) is a parity
     reference of every clean block, and assembles the V-valued key
        D = Q[:,Icl] blockdiag(T_{2,1}^T, ..., T_{2,n1}^T);
  3. prints the certificate  D in V ,  G_pub D blockdiag(H0^T) = 0 ,  rk(G_pub D)=k;
  4. DECRYPTS y with D (no residual rank-decoding step) and checks m_rec == m_true.

Run:  python3 gabkron_attack.py
Needs: gabkron_attack_common.py, structure.py (same folder).
"""
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import gabkron_attack_common as C
from gabkron_attack_common import (matmul, transpose, is_zero, rank, fmt_vec, _blk)


def run(seed=1, m=10, n2=10, k2=2, lblocks=(1, 1)):
    I = C.build_instance(seed, m, n2, k2, lblocks)
    F = I['F']; n1 = I['n1']; k = I['k']; t = I['t']; mm = F.m
    h0 = [F.pw(2, j) for j in range(mm)]
    print("=" * 78)
    print(f" GabKron per-block attack   q=2 m={mm} "
          f"(n1,k1,n2,k2)=({n1},{n1},{n2},{k2}) lam={I['lam']} t1={I['t1']} t={t}")
    print(f" spread distortion,  lambda*t={I['lam']*t}")
    print("=" * 78)

    D, Hs, Vset, p = C.recover_perblock(I, h0)
    if D is None:
        print(" recovery FAILED (no full-column-rank T_2 found)"); return False
    GD = matmul(F, I['Gpub'], D)
    inV = all(D[i][j] in Vset for i in range(I['n']) for j in range(n1 * mm))
    orth = is_zero(matmul(F, GD, transpose(_blk(F, Hs, mm, n1))))
    rk = rank(F, GD)
    print(f" recovery (p = n2-t1-k2 = {p} rows, uniform; per-block T_2i):")
    print(f"   certificate:  D in V = {inV} ,  G_pub D . blockdiag(H0^T) = 0 = {orth} ,"
          f"  rk(G_pub D) = {rk} == {k} -> {rk == k}")
    okd, eD, mrec = C.decrypt(F, I['Gpub'], D, Hs, I['y'], mm, t, "perblock")
    ok = (mrec == I['m_true'])
    print(f"   decryption :  e recovered = {okd} ,  m_rec == m_true = {ok}")
    print("=" * 78)
    return (rk == k) and inV and orth and ok


if __name__ == "__main__":
    results = []
    for sd in range(1, 25):
        results.append(run(seed=sd, m=10, n2=10, k2=2, lblocks=(1, 1)))
    print(f"\n SUMMARY (per-block, no W_A):  {sum(results)}/{len(results)} instances "
          f"recovered + decrypted.")
