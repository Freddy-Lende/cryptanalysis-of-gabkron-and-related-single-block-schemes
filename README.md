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
├── stab_structure.py
├── gabkron_attack_common.py
│
├── verify_redthread_F32.py
├── sun_perblock.py
├── residual_perblock.py
├── gabkron_perblock_example.py
│
├── gabkron_attack_with_secret.py
├── gabkron_attack.py
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

### 'gabkron_attack_with_secret.py'
Validates the structural witness predicted by the theory and the corresponding decryption once the witness has been constructed from the secret information. This script is intended as a correctness check of the theoretical results rather than a public-key implementation of the attack.
---

## Key-recovery attack

### `gabkron_attack.py`
Complete **public** key-recovery attack, using only the public generator `G_pub`, the
public parameters, a chosen reference `h0`, and the guessed distortion weight `t1`
(the secret scrambler, distortion, and masking subspace are used only to build the
instance and to seed a correct guess). For each candidate subspace `F` the script:

1. forms the public system `G_pub . D . (I (x) H0)^T = 0` over `F_q`;
2. computes its kernel by Gaussian elimination;
3. extracts a solution of image rank `k` (every kernel element already has entries in
   `F`, i.e. support `<= lambda`);
4. decrypts a real ciphertext `y = m G_pub + e`;
5. reports GOOD guesses (`F = alpha V`) versus BAD guesses separately.

It runs a worked single-block and a GabKron (`n1 = 2`) instance, then a batch that
recovers and decrypts **24 / 24** random instances of each while rejecting **24 / 24**
bad guesses.

---

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
