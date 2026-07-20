# Structural Cryptanalysis of Loidreau-Masked Gabidulin–Kronecker and Related Single-Block Schemes

Reference implementation accompanying the paper

> **Structural Cryptanalysis of Loidreau-Masked Gabidulin–Kronecker and Related Single-Block Schemes**

It provides Python implementations of the structural attacks presented in the paper,
together with all scripts required to reproduce the experimental results, complexity
estimates, and worked examples.

**Requirements:** Python 3.x — standard library only. No SageMath or external
dependencies are required.

---

## Repository contents

```
.
├── structure.py
├── gabkron_attack_common.py
│
├── verify_redthread_F32.py
├── sun_perblock.py
├── residual_perblock.py
├── gabkron_perblock_example.py
│
├── gabkron_attack.py
│
├── module_structure.py
│
├── gabkron_complexity_perblock.py
├── apps_complexity.py
├── proven_complexity.py
│
└── README.md
```

---

## Running the scripts

Each script is self-contained and is executed directly with

```bash
python3 <script_name>.py
```

No command-line arguments or external datasets are required.

---

## Helper modules

### `structure.py`
Finite-field arithmetic over `GF(q^m)` (field elements, Frobenius maps) together with
the generic matrix and linear-algebra routines shared across the scripts: Moore
matrices, kernels, ranks, Kronecker product, matrix inverse, and column operations.

### `gabkron_attack_common.py`
Shared implementation of:

- GabKron instance generation,
- per-block decomposition,
- alternative-key recovery,
- plaintext decryption.

---

## Structural reduction

### `verify_redthread_F32.py`
Verifies the per-block decomposition

```
Gpub · Q[:, Icl]  =  G*  =  (G1 ⊗ I_k2) · diag(G'_{2,i})
```

on the running `F_{2^5}` instance.

### `sun_perblock.py`
Independent verification of the same structural decomposition.

### `residual_perblock.py`
Checks the full-rank property `rk(G*) = k` across parameter sets and both spread and
block-supported distortion layouts (confirming that recovery is full-rank, with no
residual rank-decoding phase).

### `gabkron_perblock_example.py`
Complete worked example over `F_{2^6}`. The script prints:

- the public key,
- the block decomposition,
- the identity `X · T = (X'' | 0)`,
- the recovered clean block,
- all rank verifications.

---

## Key-recovery attack

### `gabkron_attack.py`
Complete **public** key-recovery attack, using only the public generator `G_pub`, the
public parameters, a chosen reference `h0` and the guessed distortion weight `t1`. It
runs two experiments:

- **(A) Accelerated regime `r = r_max`.** For a candidate subspace `F` of dimension
  `r_max = floor(k p / n)` it forms the **per-block** system `G_pub Z H0^T = 0`, computes
  the module `L_F`, extracts an `F_qm`-basis of it via the `R_beta` action and
  **concatenates the whole basis** into `D_F`. This is the deterministic extraction of the
  paper's Theorem *Deterministic extraction from a kernel basis*: no sampling, no subset
  enumeration, and no assumption on `d = dim_{F_qm} L_F`. The only remaining check is the
  entry-support `<= lambda`. The campaign uses a **uniformly random** masking space `V`,
  entries of `P` **uniform in `V`**, and the scheme's error rank
  `t = floor((n2-k2-2 t1)/(2 lambda)) >= 1`. Over eight configurations
  (`r_max - lambda in {0,1,2}`, `lambda in {2,3}`, several `t1`, layouts
  spread/concentrated/random, single-block and Kronecker) it reports the recovery rate
  with a 95% confidence interval, the false-positive rate on genuine random bad guesses,
  the `F_q`- and `F_qm`-dimensions of the solution space, the support of `D_F` and whether
  its image rank equals `k`. The secret `V` is used only to seed the controlled correct
  guess; the exponential outer guessing loop is not run here.

- **(A') Proven regime `r = lambda`.** The same pipeline with a guess of dimension exactly
  `lambda`. A good guess of that dimension containing `alpha V` must **equal** `alpha V`,
  so every element of `L_F` is `alpha V`-valued by construction and the support test cannot
  fail: this campaign validates the paper's Theorem *Heuristic-free recovery at
  r = lambda* end to end, with **no heuristic involved**.

- **(B) Complete attack.** The full public attack on a tiny instance, where `F` is
  guessed **at random without any secret** until resolution + extraction + decryption
  succeed; the measured number of guesses is compared with the predicted
  `q^{(lambda-1)m - lambda r_max}`.

Empirically, even when `r_max > lambda`, every sampled kernel vector already has
entry-support exactly `lambda`, so a valid key is found within the first sampled
combination(s). The full campaign takes on the order of 15--20 minutes; reduce the
per-configuration `N` in `__main__` for a quicker run.

### `module_structure.py`
Reproduces the **F_{q^m}-module structure** of the per-block solution space
`L_F = { Z : G_pub Z H0^T = 0 }` (paper: Lemma *Frobenius module structure*, the
extraction Heuristic, and the structure Remark). Using the trusted kernel solver of
`gabkron_attack.py` (whose `R_beta`, `act`, `in_span`, `supp_dim` and `kbasis` now live
in `gabkron_attack_common.py` and are shared by both scripts), it measures on good
guesses at `r = r_max`:

- `dim_{F_qm} L_F` (found equal to `n1^2`), and the consistency `m | dim_{F_q} L_F`;
- **K-stability**: `Z in L_F  =>  Z R_beta^T in L_F`, where `R_beta` is multiplication
  by `beta` in the basis `h0` (so the action preserves the entry-support);
- that **every** element of `L_F` is `alpha V`-valued (support `<= lambda`);
- **deterministic extraction**: concatenating an `F_qm`-basis of `L_F` yields an image
  of rank exactly `k` — no random sampling, and no need to know `dim L_F`.

Note that four of the five reported cases have `r_max = lambda`, hence `F = alpha V`
exactly: there the `alpha V`-valuedness is *forced* rather than observed (paper:
Theorem *Heuristic-free recovery at r = lambda*), so the measured `n1^2` is intrinsic.

The default suite covers `n1 in {1,2}`, `lambda in {2,3}` and runs in a couple of
minutes; the `n1 = 3` case (verified to give `dim_{F_qm} L_F = 9`) is included as a
commented, slower run.

## Complexity evaluation

### `gabkron_complexity_perblock.py`
Reproduces the complexity table for the GabKron cryptosystem (original and
new-GabKron sets, at representative distortion weights).

### `apps_complexity.py`
Computes the work factors for the single-block schemes:

- the Loidreau–Gabidulin–Rashwan–Honary (LGRH) scheme,
- Modification I,
- Modification II,

and compares them with the complexities previously reported in the literature.

Both scripts evaluate the per-block work factor

```
log2 W = omega * log2(m*k*n1*p) + ((lambda-1)*m - lambda*r_star) * log2(q)
```

with the over-determined guess dimension

```
r_star = floor(k*p / n)
```

and print the result for both `omega = 2.37` and `omega = 3`.

---

## Reproducibility

Every quantitative result reported in the paper can be reproduced with the scripts in
this repository:

- structural decompositions,
- alternative secret-key recovery,
- plaintext decryption,
- complexity estimates,
- all numerical tables appearing in the paper.

---

## Citation

If this software contributes to your research, please cite the accompanying paper:

```bibtex
@unpublished{Metouke2026,
  title  = {Structural Cryptanalysis of Loidreau-Masked Gabidulin--Kronecker and Related Single-Block Schemes},
  author = {Freddy Lende Metouke, ...},
  year   = {2026},
  note   = {Preprint}
}
```

---

## License

Released under the MIT License.

### `proven_complexity.py`
Work factors of the **proven regime `r = lambda`** (paper: Theorem *Heuristic-free
recovery at r = lambda*, equation `eq:Wrig`, Table `tab:proven`). At `r = lambda` a good
guess forces `F = alpha V`, so the support bound holds by construction and the attack
uses **no heuristic at all**.

The guessing cost is computed with the **exact** Gaussian binomial,

```
E[#trials] = |Stab(V)| * [m choose lambda]_q / (q^m - 1),   |Stab(V)| = q-1 generically,
```

rather than the leading-order estimate used in the accelerated regime. The script prints,
for all four schemes and both `omega in {2.37, 3}`, the accelerated figure `W1_acc`
alongside the proven `W1_pr`, `W2_pr`, and ends with a sanity check confirming that the
proven regime costs exactly `lambda*(r_max - lambda)` extra bits (up to the `O(1)` gap
between the exact count and its estimate).

Since `W^pr` does not depend on `r_max`, it decreases monotonically in `t_1` through the
polynomial factor alone: for the original GabKron sets the worst case over
`t_1 in [1, t_2]` is at `t_1 = 1`, and that is what the table reports.

```
python3 proven_complexity.py
```
