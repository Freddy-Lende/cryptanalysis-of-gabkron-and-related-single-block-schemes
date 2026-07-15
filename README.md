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
├── gabkron_witness.py
│
├── gabkron_attack.py
│
├── module_structure.py
│
├── gabkron_complexity_perblock.py
├── apps_complexity.py
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

### `gabkron_witness.py`

Validates the structural witness predicted by the theory and the corresponding decryption once the witness has been constructed from the secret information. This script serves as a correctness check of the theoretical results rather than a public-key implementation of the attack.

---

## Key-recovery attack

### `gabkron_attack.py`
Complete **public** key-recovery attack, using only the public generator `G_pub`, the
public parameters, a chosen reference `h0` and the guessed distortion weight `t1`. It
runs two experiments:

- **(A) Resolution + extraction campaign.** For a candidate subspace `F` of dimension
  `r_max = floor(k p / n)` it forms the public system, computes its kernel, samples
  `F_q`-linear combinations and extracts a key of image rank `k` and support `<= lambda`,
  then decrypts. The campaign uses a **uniformly random** masking space `V`, entries of
  `P` **uniform in `V`**, and the scheme's error rank `t = floor((n2-k2-2 t1)/(2 lambda)) >= 1`.
  It reports, over many instances and several regimes (`r_max - lambda in {0,1,2}`,
  `lambda in {2,3}`, several `t1`, layouts spread/concentrated/random, single-block and
  Kronecker), the recovery rate with a 95% confidence interval, the false-positive rate
  on genuine random bad guesses, the kernel-dimension set, the support histogram of the
  sampled kernel vectors, the empirical `P(rank = k)`, and the mean number of sampled
  combinations to a valid key. The secret `V` is used only to seed the controlled
  correct guess; the exponential outer guessing loop is not run here.

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
`gabkron_attack.py`, it measures on good guesses at `r = r_max`:

- `dim_{F_qm} L_F` (found equal to `n1^2`), and the consistency `m | dim_{F_q} L_F`;
- **K-stability**: `Z in L_F  =>  Z R_beta^T in L_F`, where `R_beta` is multiplication
  by `beta` in the basis `h0` (so the action preserves the entry-support);
- that **every** element of `L_F` is `alpha V`-valued (support `<= lambda`);
- **deterministic extraction**: every `n1`-subset of an `F_qm`-basis of `L_F` yields an
  image of rank exactly `k` — no random sampling.

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
  author = {Freddy Lende Metouke,...},
  year   = {2026},
  note   = {Preprint}
}
```

---

## License

Released under the MIT License.
