# Corrected and Optimized Chaotic Substitution Boxes for Lightweight IoT Cryptography

Code and computed results underlying the manuscript *"Corrected and Optimized Chaotic
Substitution Boxes for Lightweight IoT Cryptography: A Critical Reappraisal of the DAC
S-Box and a Provably Optimal Replacement"* (Scientific Reports, submission
`edb262da-cfe7-4f35-a687-5726cb36e6d3`).

Every number printed in the manuscript is produced by this code and is checked
mechanically against the result files it writes. Nothing in the paper is typed by hand.

## Reproducing the results

Python 3.12 with the pinned dependencies in `requirements.txt`:

```
pip install -r requirements.txt
python main.py            # writes output/results.json  and the seven paper figures
python run_optimized.py   # writes output/results_optimized.json and the comparison figures
python verify_latex.py    # compares every number in paper.tex against those two files
python -m pytest tests -q # 298 tests
```

Both runners are deterministic: two runs produce byte-identical result files apart from
their `timings` block. The optimizer reproduces the frozen tables `DAC-OPT` and
`DAC-OPT-GOLD` bit for bit from seed 0.5.

`verify_latex.py` compares 333 printed values and exits non-zero on any disagreement.
Six rows of the software-benchmark table are wall-clock measurements and therefore
depend on the machine; they are the only values that may differ, and they must be
whitelisted explicitly, one at a time:

```
python verify_latex.py \
  --allow tab:benchmarks/software/s_box_lookup_throughput \
  --allow tab:benchmarks/software/s_box_lookup_latency \
  --allow tab:benchmarks/software/cipher_encryption_throughput \
  --allow tab:benchmarks/software/cipher_decryption_throughput \
  --allow tab:benchmarks/software/encryption_latency_per_block \
  --allow tab:benchmarks/software/key_schedule_generation
```

The deterministic companion row in the same table, the cipher avalanche ratio, is *not*
whitelisted and reproduces exactly.

## What is here

| Path | Contents |
|---|---|
| `src/security_analysis.py` | Metric engine: Walsh spectrum, LAT, DDT, BCT, SAC/BIC, algebraic degree and immunity |
| `src/boolean_function.py` | Truth tables, ANF via the Möbius transform, component analysis |
| `src/sbox_tables.py` | The frozen tables (DAC, E-DAC, DAC-OPT, DAC-OPT-GOLD) and their regeneration |
| `src/sbox_optimizer.py` | Chaotic RNG, GF(2⁵) arithmetic, the affine-equivalence search and the annealer |
| `src/chaotic_maps.py` | Logistic, tent and tent–logistic maps; Lyapunov exponents; orbit entropy |
| `src/block_cipher.py` | The 32-bit proof-of-concept SPN and its key schedule |
| `src/cipher_analysis.py` | Branch numbers and the MILP minimum-active-S-box bounds; test vectors |
| `src/comparative_analysis.py` | Competitor S-boxes, the comparison tables and the figures |
| `src/benchmarks.py` | Timing harness (illustrative only; see the manuscript) |
| `src/literature.py` | Values quoted from the base paper, kept in one place |
| `hdl/` | Verilog lookup tables, self-checking testbenches, and their generator |
| `tests/` | 298 tests, including brute-force references for every metric |
| `output/results*.json` | Written by the two runners; the result files every table in the paper is generated from |
| `verify_latex.py` | The gate that fails if any printed value disagrees with those files (needs the manuscript source, which is not part of this repository while the paper is under review) |
| `set_code_doi.py` | Inserts the archive DOI into the manuscript once this archive is deposited |

## A note on the hardware claims

No synthesis, place-and-route, simulation or hardware measurement was performed in this
work, and none is claimed. The Verilog is a lookup-table description with a self-checking
testbench verified against the Python model; no simulator was available. The gate-equivalent
figures that appear in the manuscript's comparison table are quoted from the literature and
are labelled as such.

## Not included

The referee reports and the base paper PDF are deliberately excluded: the former are
confidential, the latter is the copyrighted work of another group. The Springer Nature
class and style files are omitted for the same reason and are available from the
publisher. The manuscript source and the response to reviewers are not committed while
the paper is under review. Generated output (figures and result files) is not committed
either; `python main.py` and `python run_optimized.py` recreate all of it.
