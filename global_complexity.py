"""
global_complexity.py  --  Reviewer point 3 (pricing the unknown per-block width rho).

The per-block width rho = max_i Colr_q(X^(i)) is unknown to the attacker; the table must not
be read as max_rho W(rho) (that is not the cost of an enumeration). The clean fix uses the
GLOBAL rank t1 = Colr_q(X), which is a PUBLIC design parameter (the total distortion rank):

  * Since rho <= t1, clearing to the n2 - t1 columns of the *global* rank is valid for every
    per-block layout (Proposition 2 holds at width w = t1), so NO rho search is performed.
  * Decryption is preserved: at w = t1 the parity length is p = n2 - t1 - k2, and the
    designers' weight t = floor((n2-k2-2 t1)/(2 lambda)) gives
        lambda t <= (n2-k2-2 t1)/2 = p/2 = radius,
    so the recovered key decodes directly. (The earlier universal choice w = t2 does NOT
    preserve this: e.g. GabKron-256 with t1=1 has t=4, lambda t=12, but at w=t2=13 the
    radius is floor(13/2)=6 < 12 -- decryption fails. t1 is the correct public width.)

The reported work factor is then W(t1) at the public t1, and, since the manuscript is
conservative over all admissible t1 in [1,t2], the tabulated verdict is the WORST CASE over
public t1, i.e. max_{t1} W(t1). All costs use the single-copy count N_eq = m*k*(n2-t1-k2).
Pure Python; imports the exact 1/S1 term from gabkron_complexity_perblock.
"""
from math import log2
from gabkron_complexity_perblock import logW, log2_inv_S1, rstar, STR


def logsumexp2(vals):
    a = max(vals)
    return a + log2(sum(2.0 ** (v - a) for v in vals))


def analyse(name, n1, k1, n2, k2, q, m, lam, claim):
    n, k = n1 * n2, k1 * k2
    t2 = (n2 - k2) // 2
    rows = []                                  # (t1, p, r_max, decodes?)
    for t1 in range(1, t2 + 1):
        p = n2 - t1 - k2
        if p <= 0:
            continue
        r = rstar(k, p, n)
        if r < lam:
            continue
        t_des = (n2 - k2 - 2 * t1) // (2 * lam)          # designers' weight at this t1
        decodes = (lam * t_des <= p // 2)
        rows.append((t1, p, r, decodes))
    out = {}
    for tag, om in (("S", STR), ("3", 3.0), ("237", 2.37)):
        Wacc = [logW(m * k * p, m, lam, r, om) for (t1, p, r, dec) in rows]   # single-copy
        Wpr = [logW(m * k * p, m, lam, lam, om) for (t1, p, r, dec) in rows]
        out[tag] = dict(acc_worst=max(Wacc), acc_sum=logsumexp2(Wacc),
                        pr_worst=max(Wpr), pr_sum=logsumexp2(Wpr))
    all_decode = all(dec for (_, _, _, dec) in rows)
    return name, claim, out, all_decode, t2


SETS = [("GabKron-128", 2, 2, 24, 12, 2, 48, 3, 128),
        ("GabKron-192", 2, 2, 38, 19, 2, 76, 3, 192),
        ("GabKron-256", 2, 2, 52, 26, 2, 104, 3, 256)]


def mark(w, cl):
    return f"*{w:5.1f}*" if w < cl else f" {w:5.1f} "


def main():
    print("=" * 104)
    print(" Reviewer point 3: global width w = t1 (public), no rho search, decryption-safe.")
    print("   verdict = worst case over public t1 = max_{t1} W(t1) ; single-copy N_eq=m*k*(n2-t1-k2).")
    print("   (max_{t1} W = the table's 'max' column; the Sum column is the enumeration cost if t1 is hidden.)")
    print("=" * 104)
    for name, n1, k1, n2, k2, q, m, lam, cl in SETS:
        nm, claim, out, all_dec, t2 = analyse(name, n1, k1, n2, k2, q, m, lam, cl)
        print(f"\n{name}  (claim {claim}, t2={t2}, decryption-safe at every public t1: {all_dec})")
        print(f"   {'exp':>5} | {'ACCEL  worst(=max_t1)   Sum_t1':<32} | {'PROVEN  worst   Sum_t1':<28}")
        for tag in ("S", "3", "237"):
            o = out[tag]
            lbl = {'S': '2.807', '3': '3.0', '237': '2.37'}[tag]
            acc = f"{mark(o['acc_worst'], cl)}   {mark(o['acc_sum'], cl)}"
            pr = f"{mark(o['pr_worst'], cl)}   {mark(o['pr_sum'], cl)}"
            print(f"   {lbl:>5} | {acc:<32} | {pr:<28}")
    print("\n(* = below claim.  worst(=max_t1) is the value the tables report; it is decryption-safe")
    print(" because w=t1 keeps lambda t <= floor(p/2).  No universal w=t2 is used.)")


if __name__ == "__main__":
    main()
