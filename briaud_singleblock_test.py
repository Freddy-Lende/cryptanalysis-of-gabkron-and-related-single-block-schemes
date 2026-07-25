"""
briaud_singleblock_test.py  --  direct test of the Briaud-Loidreau constrained system
(Eq. 3) on ONE masked Gabidulin block, checking the distinguisher of their Prop. 4.

A GabKron clean block is a masked shortened Gabidulin code, so this isolates the
question "does the per-block Briaud system distinguish a good guess from a bad one?"
from the Kronecker plumbing. The correct distinguisher (Prop. 4) is the DIMENSION GAP:
a good guess U >= xV yields a solution space exactly m larger (one extra F_qm-dimension,
the recovered (xV, xW) pair) than a screened bad guess. This requires a genuine
Gabidulin block, i.e. n <= m.

This is a thin wrapper over briaud_perblock.run, which implements the corrected system.
"""
import structure as ss
ss.IRRED.setdefault(8, 0b100011011)
ss.IRRED.setdefault(10, 0b10000001001)

from briaud_perblock import run

if __name__ == "__main__":
    # Feasible Gabidulin blocks (n <= m). Both separate cleanly (diff = m on every trial).
    # The borderline equality case rn = gamma n + r^2 may fail to separate, so these use
    # configs where the distinguisher is robust.
    run(seed=7, m=10, n=8, k2=4, lam=2, trials=6)
    run(seed=17, m=10, n=9, k2=5, lam=2, trials=4)
