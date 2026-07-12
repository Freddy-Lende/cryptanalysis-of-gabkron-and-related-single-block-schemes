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

---

## Key-recovery attack

### `gabkron_attack.py`
Complete implementation of the structural attack. The script:

1. generates a GabKron instance;
2. computes the structural reduction;
3. solves the Burle linear system in the over-determined regime;
4. recovers an alternative secret key;
5. decrypts the ciphertext.

For the experimental setup reported in the paper, it successfully recovers and
decrypts **24 out of 24** randomly generated spread-distortion instances.

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
