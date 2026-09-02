"""Software performance benchmarks for the DAC S-box and the SPN block cipher.

**These timings are illustrative only, not a performance claim.** Everything
here is single-threaded, interpreted CPython: the numbers measure the cost of
a Python ``for`` loop and a dict/list index far more than they measure the
S-box or the cipher, and they are two to three orders of magnitude away from
what the same algorithm costs in optimised C or in hardware. They are
reported so that the software side of the paper is reproducible and honestly
labelled, and they must not be compared against C, assembly or ASIC/FPGA
figures from the literature. The hardware discussion is the credible
performance story; no synthesis is performed anywhere in this repository.

Every benchmark takes an S-box table (default: the paper's DAC S-box), warms
up once, times ``repeat`` independent runs with :func:`timeit.repeat` and
reports both the minimum (the usual microbenchmark estimate) and the median.
All functions return plain dicts and print nothing; :func:`run_all_benchmarks`
is the only reporting entry point.

Measured:
- S-box lookup: raw ``table[x]``, the :meth:`DACSBox.substitute` method call
  (which adds range validation and CPython call overhead) and a numpy
  fancy-index batch lookup. The cost of an empty Python loop is reported
  alongside so the pure lookup cost can be separated from interpreter overhead.
- Block cipher encryption/decryption throughput (blocks/s and MB/s, 4-byte blocks).
- Key schedule generation.
- Avalanche quality of the cipher over seeded random plaintexts.
- Memory footprint of the cipher state (computed from the table size).

Reproducibility metadata
------------------------
Every record returned by a ``benchmark_*`` function (and by
:func:`estimate_memory_footprint`) carries a ``"methodology"`` sub-dict built
by :func:`benchmark_methodology`: the machine (CPU brand string, core counts,
OS and version), the software stack (Python build, numpy version), the exact
number of operations timed, the amount of data processed, the number of
:func:`timeit.repeat` repetitions, the timer used and its resolution, and the
statement that both the minimum *and* the median are reported (rates are
derived from the minimum). The pre-existing result keys are unchanged, so
consumers of ``output/results.json`` keep working.
"""

from __future__ import annotations

import os
import platform
import random
import statistics
import time
import timeit
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np

from .block_cipher import BLOCK_BYTES, BLOCK_SIZE, KEY_SIZE, NUM_ROUNDS, DACBlockCipher
from .sbox import DAC_SBOX, SBOX_BITS, SBOX_SIZE, DACSBox


DEFAULT_KEY = 0x0123456789ABCDEF0123
DEFAULT_SEED = 20250101

#: ``(n_lookups, n_blocks, n_keys, n_avalanche_plaintexts)`` of the two
#: benchmark scales.  This is the single source of the benchmark sample
#: sizes: :func:`run_all_benchmarks` selects from it and
#: :func:`src.comparative_analysis.collect_benchmarks` imports and
#: re-exports it, so "quick" means the same amount of work whichever entry
#: point is used.  ``quick`` is about 10x less work than ``standard``.
BENCHMARK_SIZES: dict[str, tuple[int, int, int, int]] = {
    "standard": (100_000, 5_000, 1_000, 100),
    "quick": (10_000, 500, 100, 20),
}

#: Timing repeats used by both entry points.
BENCHMARK_REPEAT = 5

# Hardware figures quoted by the base paper (Table 4 / Section 5). They are
# literature claims, not measurements made by this code; the audit found the
# 11.25 GE / 6.225 ns numbers are not backed by a synthesis report.
PAPER_REPORTED_GATE_EQUIVALENTS = {
    "DAC": 11.25,
    "Thakor_et_al": 12.54,
    "PRESENT": 22.5,
    "SKINNY": 12.0,
    "Hummingbird-2": 20.0,
    "RECTANGLE": 18.0,
}
PAPER_REPORTED_HW_LATENCY_NS = 6.225   # device EP4CE10E22C8 (a Cyclone IV FPGA; the paper says "ASIC")


# ============================================================
# Reproducibility metadata (reviewer 2, additional comment 4)
# ============================================================
#: Registry value that holds the real CPU brand string on Windows.
CPU_REGISTRY_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
CPU_REGISTRY_VALUE = "ProcessorNameString"

#: Wording attached to every timing record; see the module docstring.
ILLUSTRATIVE_NOTE = (
    "Illustrative only: single-threaded interpreted CPython. The figure is "
    "dominated by interpreter overhead, is not a performance claim, and must "
    "not be compared with optimised C, assembly or hardware implementations."
)


def _cpu_model() -> tuple[str, str]:
    """``(brand string, where it came from)`` for the CPU running the benchmarks.

    ``platform.processor()`` returns a useless family/model/stepping string on
    Windows (e.g. ``"AMD64 Family 25 ..."``), so the real brand is read from
    the registry value :data:`CPU_REGISTRY_VALUE` under
    :data:`CPU_REGISTRY_KEY`.  Falls back to ``platform.processor()`` and then
    to ``"unknown"`` on any platform or permission failure, so the metadata is
    never a hard dependency on Windows.
    """
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, CPU_REGISTRY_KEY) as key:
                name = str(winreg.QueryValueEx(key, CPU_REGISTRY_VALUE)[0]).strip()
            if name:
                return name, f"winreg HKLM\\{CPU_REGISTRY_KEY}\\{CPU_REGISTRY_VALUE}"
        except (ImportError, OSError, ValueError):
            pass
    for source, value in (("/proc/cpuinfo", _cpu_model_from_proc()),
                          ("platform.processor()", platform.processor())):
        if value:
            return value.strip(), source
    return "unknown", "unavailable"


def _cpu_model_from_proc() -> Optional[str]:
    """CPU brand from ``/proc/cpuinfo`` (Linux); ``None`` anywhere else."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _physical_cores() -> tuple[Optional[int], str]:
    """``(physical core count, source)``; ``(None, "unavailable")`` if unknown.

    Uses :mod:`psutil` when it happens to be installed (it is not a declared
    dependency of this project) and reports the count as unknown otherwise
    rather than guessing from the logical count.
    """
    try:
        import psutil
    except ImportError:
        return None, "unavailable"
    try:
        cores = psutil.cpu_count(logical=False)
    except Exception:                                   # pragma: no cover
        return None, "unavailable"
    return (int(cores), "psutil.cpu_count(logical=False)") if cores else (None, "unavailable")


@lru_cache(maxsize=1)
def _platform_metadata() -> tuple[tuple[str, Any], ...]:
    """Cached key/value pairs of :func:`platform_metadata` (probed once)."""
    cpu, cpu_source = _cpu_model()
    physical, physical_source = _physical_cores()
    perf_counter = time.get_clock_info("perf_counter")
    return tuple({
        "cpu_model": cpu,
        "cpu_model_source": cpu_source,
        "cpu_architecture": platform.machine(),
        "logical_cores": os.cpu_count(),
        "physical_cores": physical,
        "physical_cores_source": physical_source,
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "python_build": " ".join(platform.python_build()),
        "numpy_version": np.__version__,
        "timer": "time.perf_counter (via timeit.repeat)",
        "timer_resolution_ns": perf_counter.resolution * 1e9,
        "timer_is_monotonic": perf_counter.monotonic,
    }.items())


def platform_metadata() -> Dict[str, Any]:
    """Machine, OS, interpreter and timer description of this benchmark run.

    A fresh dict on every call (the probe itself is cached), holding the CPU
    brand string, the logical and physical core counts, the operating system
    and its version, the Python build, the numpy version and the timer used
    with its resolution in nanoseconds.  Values that cannot be determined on
    the running platform are ``None`` with a ``*_source`` of ``"unavailable"``
    -- never guessed.
    """
    return dict(_platform_metadata())


def benchmark_methodology(operation: str, n_operations: int, unit: str,
                          repeat: Optional[int] = None,
                          bytes_processed: Optional[int] = None,
                          timed: bool = True) -> Dict[str, Any]:
    """The reproducibility record attached to one benchmark result.

    Args:
        operation: What a single timed operation is, in words.
        n_operations: How many of them one timed run performs.
        unit: The unit the reported per-operation figures are in.
        repeat: Number of :func:`timeit.repeat` repetitions (``None`` for a
            record that is not a timing).
        bytes_processed: Amount of data one timed run processes, when the
            benchmark has a natural data volume.
        timed: ``False`` for derived, non-timed records (memory footprint,
            avalanche statistics), which are exact and repeat-independent.

    Returns:
        A dict with the sample size, the repetitions, the statistics that are
        reported, the ``illustrative_only`` flag and the full
        :func:`platform_metadata`.
    """
    record: Dict[str, Any] = {
        "operation": operation,
        "n_operations_timed": int(n_operations),
        "unit": unit,
        "timed": bool(timed),
        "platform": platform_metadata(),
    }
    if bytes_processed is not None:
        record["bytes_processed"] = int(bytes_processed)
    if timed:
        record.update({
            "repetitions": None if repeat is None else int(repeat),
            "warmup_runs": 1,
            "timing_method": ("timeit.repeat(func, number=1, repeat=R) around a "
                              "Python loop over all n_operations_timed inputs"),
            "statistics_reported": ("both the minimum and the median of the R "
                                    "repetitions, each divided by "
                                    "n_operations_timed"),
            "rates_derived_from": "minimum",
            "single_threaded": True,
            "illustrative_only": True,
            "note": ILLUSTRATIVE_NOTE,
        })
    return record


def _time_calls(func: Callable[[], object], repeat: int) -> list[float]:
    """Warm up ``func`` once, then return ``repeat`` wall-clock timings of single calls."""
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    func()
    return timeit.repeat(func, number=1, repeat=repeat)


def _per_item_stats(times: Sequence[float], n_items: int, unit: float = 1e9) -> dict[str, float]:
    """min/median per-item cost (default in ns) and the per-second rate from the minimum."""
    best = min(times) / n_items
    return {
        "min": best * unit,
        "median": statistics.median(times) / n_items * unit,
        "per_sec": 1.0 / best,
    }


def _random_blocks(n: int, rng: random.Random) -> list[int]:
    return [rng.getrandbits(BLOCK_SIZE) for _ in range(n)]


def benchmark_sbox_lookup(sbox_table: Sequence[int] | np.ndarray = DAC_SBOX,
                          n_lookups: int = 1_000_000, repeat: int = 5,
                          seed: int = DEFAULT_SEED) -> dict:
    """Time three ways of applying the S-box to ``n_lookups`` random 5-bit inputs.

    Returns a dict with ``loop_overhead`` (empty ``for`` loop over the inputs),
    ``table_lookup`` (``table[x]`` in a loop), ``method_call``
    (:meth:`DACSBox.substitute` in a loop) and ``numpy_batch`` (one fancy-index
    of a uint8 array), each holding ``ns_per_lookup_min``,
    ``ns_per_lookup_median`` and ``lookups_per_sec``. ``table_lookup_net_ns``
    is the raw lookup cost with the loop overhead subtracted, and
    ``methodology`` is the :func:`benchmark_methodology` record.

    The figures are illustrative only (see the module docstring): a raw
    ``table[x]`` in CPython costs tens of nanoseconds almost entirely in the
    interpreter, which is why the empty-loop cost is measured alongside it.
    """
    sbox = DACSBox(sbox_table)
    table = sbox.table
    substitute = sbox.substitute
    rng = random.Random(seed)
    inputs = [rng.getrandbits(SBOX_BITS) for _ in range(n_lookups)]
    inputs_np = np.array(inputs, dtype=np.uint8)

    def loop_overhead() -> None:
        for _ in inputs:
            pass

    def table_lookup() -> None:
        for x in inputs:
            table[x]

    def method_call() -> None:
        for x in inputs:
            substitute(x)

    def numpy_batch() -> None:
        sbox.substitute_array(inputs_np)

    results: dict = {"n_lookups": n_lookups, "repeat": repeat}
    for name, func in (("loop_overhead", loop_overhead), ("table_lookup", table_lookup),
                       ("method_call", method_call), ("numpy_batch", numpy_batch)):
        stats = _per_item_stats(_time_calls(func, repeat), n_lookups)
        results[name] = {
            "ns_per_lookup_min": stats["min"],
            "ns_per_lookup_median": stats["median"],
            "lookups_per_sec": stats["per_sec"],
        }
    results["table_lookup_net_ns"] = max(
        0.0, results["table_lookup"]["ns_per_lookup_min"]
        - results["loop_overhead"]["ns_per_lookup_min"])
    results["methodology"] = benchmark_methodology(
        operation=f"one {SBOX_BITS}-bit S-box substitution of a random input",
        n_operations=n_lookups, unit="nanoseconds per lookup", repeat=repeat,
        bytes_processed=None)
    return results


def benchmark_block_cipher(sbox_table: Sequence[int] | np.ndarray = DAC_SBOX,
                           n_blocks: int = 100_000, repeat: int = 5,
                           key: int = DEFAULT_KEY, seed: int = DEFAULT_SEED,
                           full_sbox_layer: bool = False) -> dict:
    """Encryption and decryption throughput over ``n_blocks`` random 32-bit blocks.

    Both directions are timed with the same ``for`` loop shape. The decrypted
    blocks are checked against the plaintexts once during warm-up.

    Returns ``enc``/``dec`` sub-dicts with ``us_per_block_min``,
    ``us_per_block_median``, ``blocks_per_sec`` and ``mb_per_sec`` (10^6
    bytes/s, 4 bytes per block; both rates from the minimum time), plus a
    ``methodology`` record naming the machine, the ``n_blocks * 4`` bytes
    processed per timed run and the number of repetitions.

    The throughput is illustrative only (see the module docstring): a
    ``NUM_ROUNDS``-round SPN written in pure Python is interpreter-bound and
    the MB/s figure is not comparable with any C or hardware measurement.

    Raises:
        RuntimeError: if a block does not round-trip.
    """
    cipher = DACBlockCipher(key, sbox_table, full_sbox_layer=full_sbox_layer)
    blocks = _random_blocks(n_blocks, random.Random(seed))
    encrypt = cipher.encrypt_block
    decrypt = cipher.decrypt_block
    ciphertexts = [encrypt(b) for b in blocks]
    if [decrypt(c) for c in ciphertexts] != blocks:
        raise RuntimeError("block cipher round-trip failed")

    def encrypt_all() -> None:
        for b in blocks:
            encrypt(b)

    def decrypt_all() -> None:
        for c in ciphertexts:
            decrypt(c)

    results: dict = {"n_blocks": n_blocks, "repeat": repeat,
                     "full_sbox_layer": full_sbox_layer}
    for name, func in (("enc", encrypt_all), ("dec", decrypt_all)):
        stats = _per_item_stats(_time_calls(func, repeat), n_blocks, unit=1e6)
        results[name] = {
            "us_per_block_min": stats["min"],
            "us_per_block_median": stats["median"],
            "blocks_per_sec": stats["per_sec"],
            "mb_per_sec": stats["per_sec"] * BLOCK_BYTES / 1e6,
        }
    results["methodology"] = benchmark_methodology(
        operation=f"one {BLOCK_SIZE}-bit block through all {NUM_ROUNDS} rounds "
                  f"(encryption and decryption timed separately)",
        n_operations=n_blocks, unit="microseconds per block", repeat=repeat,
        bytes_processed=n_blocks * BLOCK_BYTES)
    return results


def benchmark_key_schedule(sbox_table: Sequence[int] | np.ndarray = DAC_SBOX,
                           n_keys: int = 10_000, repeat: int = 5,
                           seed: int = DEFAULT_SEED) -> dict:
    """Time cipher construction (17-round key schedule) for ``n_keys`` random 80-bit keys.

    The S-box object is built once and shared, so the timing covers the key
    schedule and not the table validation.  The record carries the usual
    ``methodology`` sub-dict; the figures are illustrative only (see the
    module docstring).
    """
    sbox = DACSBox(sbox_table)
    rng = random.Random(seed)
    keys = [rng.getrandbits(KEY_SIZE) for _ in range(n_keys)]

    def schedule_all() -> None:
        for key in keys:
            DACBlockCipher(key, sbox)

    stats = _per_item_stats(_time_calls(schedule_all, repeat), n_keys, unit=1e6)
    return {
        "n_keys": n_keys,
        "repeat": repeat,
        "us_per_key_schedule_min": stats["min"],
        "us_per_key_schedule_median": stats["median"],
        "key_schedules_per_sec": stats["per_sec"],
        "methodology": benchmark_methodology(
            operation=f"one {NUM_ROUNDS + 1}-subkey schedule from a random "
                      f"{KEY_SIZE}-bit key",
            n_operations=n_keys, unit="microseconds per key schedule",
            repeat=repeat, bytes_processed=n_keys * (KEY_SIZE // 8)),
    }


def benchmark_avalanche_quality(sbox_table: Sequence[int] | np.ndarray = DAC_SBOX,
                                n_tests: int = 1000, key: int = DEFAULT_KEY,
                                seed: int = DEFAULT_SEED,
                                full_sbox_layer: bool = False) -> dict:
    """Average fraction of ciphertext bits that change per single plaintext-bit flip.

    For each of ``n_tests`` seeded random plaintexts every one of the 32 input
    bits is flipped, giving ``32 * n_tests`` comparisons. Also returns the
    per-input-bit averages (index = flipped bit position) and the spread of the
    per-comparison ratio; the ideal value is 0.5.

    This record is a statistic, not a timing: it is exact and reproducible for
    a given seed, and its ``methodology`` sub-dict is marked ``timed: False``.
    """
    if n_tests < 1:
        raise ValueError(f"n_tests must be >= 1, got {n_tests}")
    cipher = DACBlockCipher(key, sbox_table, full_sbox_layer=full_sbox_layer)
    encrypt = cipher.encrypt_block
    plaintexts = _random_blocks(n_tests, random.Random(seed))
    changed = np.empty((n_tests, BLOCK_SIZE), dtype=np.int64)
    for row, pt in enumerate(plaintexts):
        ct = encrypt(pt)
        changed[row] = [(ct ^ encrypt(pt ^ (1 << bit))).bit_count() for bit in range(BLOCK_SIZE)]
    ratios = changed / BLOCK_SIZE
    avg = float(ratios.mean())
    return {
        "n_plaintexts": n_tests,
        "total_comparisons": n_tests * BLOCK_SIZE,
        "avg_avalanche_ratio": avg,
        "std_avalanche_ratio": float(ratios.std()),
        "min_avalanche_ratio": float(ratios.min()),
        "max_avalanche_ratio": float(ratios.max()),
        "per_input_bit_ratio": ratios.mean(axis=0).tolist(),
        "ideal": 0.5,
        "deviation_from_ideal": abs(avg - 0.5),
        "full_sbox_layer": full_sbox_layer,
        "methodology": benchmark_methodology(
            operation=f"one single-bit-flip encryption pair over a {BLOCK_SIZE}-bit block",
            n_operations=n_tests * BLOCK_SIZE, unit="fraction of output bits changed",
            bytes_processed=n_tests * (BLOCK_SIZE + 1) * BLOCK_BYTES, timed=False),
    }


def estimate_memory_footprint(sbox_table: Sequence[int] | np.ndarray = DAC_SBOX) -> dict:
    """Bit-level memory footprint of the cipher state (tables, round keys, key register).

    The gate-equivalent and latency figures are the base paper's reported
    values, returned under ``paper_reported`` for reference only; nothing here
    is synthesised.
    """
    DACSBox(sbox_table)  # validate
    sbox_table_bits = SBOX_SIZE * SBOX_BITS
    round_keys_bits = (NUM_ROUNDS + 1) * BLOCK_SIZE
    total_bits = 2 * sbox_table_bits + round_keys_bits + KEY_SIZE
    return {
        "sbox_table_bits": sbox_table_bits,
        "sbox_table_bytes": sbox_table_bits / 8,
        "inv_table_bits": sbox_table_bits,
        "round_keys_bits": round_keys_bits,
        "key_register_bits": KEY_SIZE,
        "total_bits": total_bits,
        "total_bytes": total_bits / 8,
        "paper_reported": {
            "gate_equivalents": dict(PAPER_REPORTED_GATE_EQUIVALENTS),
            "hw_latency_ns": PAPER_REPORTED_HW_LATENCY_NS,
            "hw_platform": "EP4CE10E22C8 (Cyclone IV FPGA; called ASIC in the paper)",
        },
        "methodology": benchmark_methodology(
            operation="bit count of the cipher state, derived from the table sizes",
            n_operations=1, unit="bits", timed=False),
    }


def run_all_benchmarks(quick: bool = False,
                       sbox_table: Sequence[int] | np.ndarray = DAC_SBOX,
                       seed: int = DEFAULT_SEED) -> dict:
    """Run and print the complete benchmark suite; returns all result dicts.

    The sample sizes come from :data:`BENCHMARK_SIZES`, the same table
    :func:`src.comparative_analysis.collect_benchmarks` uses, so ``quick``
    means the same thing from either entry point: ``quick=True`` selects the
    ``"quick"`` scale, which is about 10x less work than the ``"standard"``
    scale used otherwise.  Both paths use :data:`BENCHMARK_REPEAT` timing
    repeats.  ``seed`` drives the random lookups, plaintexts and keys of
    every benchmark.

    The printed banner names the machine, the OS, the interpreter and the
    timing methodology, and repeats that every timing here is illustrative
    only (see the module docstring).  The returned section keys are unchanged
    (``sbox``, ``cipher``, ``key_schedule``, ``memory``, ``avalanche``); each
    of those records carries its own ``"methodology"`` sub-dict, and
    :func:`platform_metadata` gives the same machine description on its own
    for callers that want it once rather than per record.
    """
    print("\n" + "=" * 70)
    print("  SOFTWARE PERFORMANCE BENCHMARKS")
    print("=" * 70)

    n_lookups, n_blocks, n_keys, n_aval = BENCHMARK_SIZES["quick" if quick else "standard"]
    repeat = BENCHMARK_REPEAT
    meta = platform_metadata()
    print(f"\n  CPU: {meta['cpu_model']} ({meta['cpu_architecture']}, "
          f"{meta['logical_cores']} logical / "
          f"{meta['physical_cores'] if meta['physical_cores'] is not None else 'unknown'}"
          f" physical cores)")
    print(f"  OS: {meta['os']} {meta['os_release']} (version {meta['os_version']})")
    print(f"  Python {meta['python_version']} ({meta['python_implementation']}, "
          f"{meta['python_compiler']}), numpy {meta['numpy_version']}")
    print(f"  Timing: {meta['timer']}, resolution "
          f"{meta['timer_resolution_ns']:.1f} ns; {repeat} repetitions after one "
          f"warm-up run; both min and median reported, rates from the minimum")
    print(f"  {ILLUSTRATIVE_NOTE}")

    print("\n--- S-Box Lookup Performance ---")
    sbox_bench = benchmark_sbox_lookup(sbox_table, n_lookups, repeat, seed=seed)
    print(f"  {n_lookups:,} random 5-bit inputs, {repeat} repeats (min / median)")
    for name, label in (("loop_overhead", "empty Python loop"),
                        ("table_lookup", "raw table[x]"),
                        ("method_call", "DACSBox.substitute()"),
                        ("numpy_batch", "numpy batch lookup")):
        row = sbox_bench[name]
        print(f"  {label:22s}: {row['ns_per_lookup_min']:7.1f} / "
              f"{row['ns_per_lookup_median']:7.1f} ns/lookup "
              f"({row['lookups_per_sec']:,.0f} lookups/s)")
    print(f"  raw lookup net of loop: {sbox_bench['table_lookup_net_ns']:.1f} ns")

    print("\n--- Block Cipher Performance ---")
    cipher_bench = benchmark_block_cipher(sbox_table, n_blocks, repeat, seed=seed)
    print(f"  {n_blocks:,} random blocks, {repeat} repeats (min / median)")
    for name, label in (("enc", "Encryption"), ("dec", "Decryption")):
        row = cipher_bench[name]
        print(f"  {label}: {row['us_per_block_min']:.2f} / {row['us_per_block_median']:.2f} "
              f"us/block, {row['blocks_per_sec']:,.0f} blocks/s, {row['mb_per_sec']:.3f} MB/s")

    print("\n--- Key Schedule Performance ---")
    ks_bench = benchmark_key_schedule(sbox_table, n_keys, repeat, seed=seed)
    print(f"  {ks_bench['us_per_key_schedule_min']:.2f} / "
          f"{ks_bench['us_per_key_schedule_median']:.2f} us per key schedule "
          f"(min / median over {n_keys:,} keys)")

    print("\n--- Memory Footprint ---")
    mem = estimate_memory_footprint(sbox_table)
    print(f"  S-box table: {mem['sbox_table_bits']} bits ({mem['sbox_table_bytes']:.0f} bytes)")
    print(f"  Total cipher state: {mem['total_bits']} bits ({mem['total_bytes']:.0f} bytes)")
    reported = mem["paper_reported"]
    print("  Gate equivalents as reported in the paper (literature values, not synthesised):")
    for name, ge in reported["gate_equivalents"].items():
        print(f"    {name:15s}: {ge:5.2f} GE  {'#' * int(ge * 2)}")

    print("\n--- Avalanche Quality (statistical) ---")
    aval_bench = benchmark_avalanche_quality(sbox_table, n_aval, seed=seed)
    print(f"  {aval_bench['total_comparisons']:,} single-bit flips over "
          f"{n_aval:,} seeded random plaintexts")
    print(f"  Average avalanche ratio: {aval_bench['avg_avalanche_ratio']:.6f} (ideal: 0.5), "
          f"std {aval_bench['std_avalanche_ratio']:.4f}, "
          f"range [{aval_bench['min_avalanche_ratio']:.4f}, {aval_bench['max_avalanche_ratio']:.4f}]")
    print(f"  Deviation from ideal: {aval_bench['deviation_from_ideal']:.6f}")

    print("\n--- Paper-reported hardware latency vs software ---")
    sw_ns = sbox_bench["table_lookup"]["ns_per_lookup_min"]
    print(f"  Paper-reported ({reported['hw_platform']}; not synthesised here): "
          f"{reported['hw_latency_ns']} ns/lookup")
    print(f"  Software (CPython, raw table[x] incl. loop): {sw_ns:.1f} ns/lookup")
    print(f"  Ratio: {sw_ns / reported['hw_latency_ns']:.1f}x")
    print("  (The ratio compares an interpreted Python loop with a literature "
          "hardware claim; it is illustrative, not a like-for-like measurement.)")

    return {
        "sbox": sbox_bench,
        "cipher": cipher_bench,
        "key_schedule": ks_bench,
        "memory": mem,
        "avalanche": aval_bench,
    }
