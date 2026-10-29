# Reappraisal of the dynamic Airy chaotic substitution box and affine class selection of power map representatives

Code, data and computed results underlying the manuscript *"Reappraisal of the dynamic
Airy chaotic substitution box and affine class selection of power map representatives with
ideal avalanche"* (Scientific Reports, submission `edb262da-cfe7-4f35-a687-5726cb36e6d3`).
Release **v1.0.0** is the version used for the printed tables.

Every number printed in the manuscript is produced by this code and is checked
mechanically against the result files it writes. Nothing in the paper is typed by hand.

## Reproducing the results

Python 3.12 with the pinned dependencies in `requirements.txt`:

```
pip install -r requirements.txt
python main.py            # output/results.json, the paper figures, MILP + exact trail search (~25 s)
python run_optimized.py   # output/results_optimized.json and the comparison figures (~5 s)
python run_studies.py     # output/results_studies.json and output/tables/*.csv (~5 min)
python run_hardware.py    # output/results_hardware.json and output/netlists/ (~4 min)
python verify_results.py --hardware   # tiered verification report + MANIFEST.json
python verify_latex.py    # compares every number in paper.tex against the four files
python -m pytest tests -q # the full test suite
```

All four runners are deterministic: two runs produce identical result files apart from
their `timings` block. The optimizer reproduces the frozen tables `DAC-OPT` and
`DAC-OPT-GOLD` bit for bit from seed 0.5, and so does the independent executable
specification of the paper's Algorithms 2 and 3 (`src/algorithm_spec.py`).

Seeds: chaotic seed 0.5 for the selection; x0 in {0.05, ..., 0.50} for the annealing
study; chaotic k/64, PCG64 0..31 and Weyl k/32 for the generator ablation; PCG64 2026 for
the affine-class statistics and the class-area study; `--seed 42` (default) for the
benchmarks and test vectors of `main.py`. The MILP bounds and the exact trail search of
the paper are produced by `main.py`.

### Verification report

`python verify_results.py` writes `output/verification_report.md`, its JSON twin and
`output/MANIFEST.json` (commit, dependency versions, seeds, SHA-256 of every result file).
The report separates four tiers:

| Tier | What is reproduced | How |
|---|---|---|
| A | every S-box metric, the DDT/LAT/BCT CSV files, the identities, the exact ideal-SAC count, Algorithm 2 | self-contained enumeration in `verify_results.py` (no code shared with `src/`) plus `src/algorithm_spec.py` |
| B | MILP bounds and exact trail search | `src.cipher_analysis.analyze_cipher` re-run and compared leaf by leaf |
| C | logic synthesis (`--hardware`) | Yosys/ABC re-run on both libraries, netlists re-verified |
| D | the manuscript (when `paper.tex` is present) | `verify_latex.py` |

### Automatic consistency checks

Every runner recomputes, for every S-box and with independent routines, the identities

* `W_{f_b}(a) = 2 LAT[a][b]` (Walsh spectrum versus the signed LAT count),
* `C_max = 2 LAP(S)` (maximal absolute correlation),
* `NL(S) = 2^(n-1) - 2^n LAP(S) = 2^(n-1) - max|W|/2` (for n = 5: `NL = 16 - 32 LAP`),
* `delta(S) = 2^n DP(S)`,

and refuses to write any result file if one of them fails
(`src/security_analysis.check_metric_identities`). `verify_latex.py` additionally checks
every row of the manuscript's comparison table against the same identities, requires the
manuscript's equations to state them in this form, and enforces the editor's title rules.

### Machine-dependent rows

`verify_latex.py` exits non-zero on any disagreement. Six rows of the software-benchmark
table are wall-clock measurements and therefore depend on the machine; on another machine
they are the only values that may differ, and they must be whitelisted explicitly, one at
a time:

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
| `src/security_analysis.py` | Metric engine: Walsh spectrum, LAT, DDT, BCT, SAC/BIC, algebraic degree and immunity, fixed points, the metric identities |
| `src/boolean_function.py` | Truth tables, ANF via the Möbius transform, component analysis |
| `src/sbox_tables.py` | The frozen tables (DAC, E-DAC, DAC-OPT, DAC-OPT-GOLD) and their regeneration |
| `src/sbox_optimizer.py` | Chaotic RNG, GF(2⁵) arithmetic, the affine-class selection and the annealer (with validated parameter domains, including N = 1 and T0 = 0) |
| `src/algorithm_spec.py` | Executable specification of the paper's Algorithms 2 and 3: plain Python, no code shared with the optimizer, reproduces both tables and every search statistic |
| `src/search_studies.py` | Generator ablation (chaotic / PCG64 / Weyl / lexicographic), invariant-versus-variable criteria over random affine representatives, the exact count of ideal-SAC representatives (Moebius inversion over the subspace lattice, orbits), annealing budget and temperature study |
| `src/hardware_synthesis.py` | Common-condition synthesis (Yosys/ABC, NanGate45 and SKY130 Liberty files fetched from a pinned commit and SHA-256 checked), gate-level verification, area, depth, delay, energy, leakage |
| `src/chaotic_maps.py` | Logistic, tent and tent–logistic maps; Lyapunov exponents; orbit entropy |
| `src/block_cipher.py` | The 32-bit proof-of-concept SPN and its key schedule |
| `src/cipher_analysis.py` | Branch numbers, MILP minimum-active-S-box bounds, exact Matsui-style trail search over the real DDT/LAT, test vectors |
| `src/comparative_analysis.py` | Competitor S-boxes, the comparison tables and the figures |
| `src/benchmarks.py` | Timing harness (illustrative only; see the manuscript) |
| `src/literature.py` | Values quoted from the base paper, kept in one place |
| `hdl/` | Verilog lookup tables, self-checking testbenches, and their generator |
| `tests/` | The test suite, including brute-force references for every metric and an independent re-derivation of the exact trail search |
| `output/results*.json` | Written by the four runners; the result files every table in the paper is generated from |
| `output/tables/` | Lookup table, DDT, LAT and BCT of every analysed S-box as CSV |
| `output/netlists/` | Mapped gate-level netlists (Yosys JSON) of every S-box on both libraries |
| `output/verification_report.*`, `output/MANIFEST.json` | Written by `verify_results.py` |
| `verify_latex.py` | The gate that fails if any printed value disagrees with those files (needs the manuscript source, which is not part of this repository while the paper is under review) |
| `verify_results.py`, `run_hardware.py` | Tiered verification report; hardware runner |
| `set_code_doi.py` | Inserts the archive DOI into the manuscript and the response letters once this archive is deposited |
| `CITATION.cff`, `.zenodo.json` | Citation and archive metadata for the permanent deposit |

## Hardware figures

`run_hardware.py` synthesises every S-box from the same 32-entry lookup description with
Yosys 0.70 / ABC (YoWASP build, pinned in `requirements.txt`) on the NanGate FreePDK45 and
SkyWater SKY130 HD libraries, verifies each mapped netlist on all 32 inputs, and computes
area (um^2 and NAND2 gate equivalents), logic depth, critical-path delay (NLDM tables),
energy per evaluation (zero-delay activity over all 1024 input pairs) and leakage. These
are pre-layout, single-corner figures that compare the S-boxes under identical conditions;
they are not sign-off values. The Liberty files are downloaded on first use from
OpenROAD-flow-scripts at a pinned commit, checked against SHA-256 digests, cached under
`hdl/synth/libs/` and not redistributed.

## License

Released under the MIT License; see [`LICENSE`](LICENSE).

## Not included

The referee reports and the base paper PDF are deliberately excluded: the former are
confidential, the latter is the copyrighted work of another group. The Springer Nature
class and style files are omitted for the same reason and are available from the
publisher. The manuscript source and the response to reviewers are not committed while
the paper is under review. The figures are not committed; the runners recreate them.
