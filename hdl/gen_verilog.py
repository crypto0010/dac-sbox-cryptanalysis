#!/usr/bin/env python
"""Verilog generator for the frozen 5-bit S-box tables of the DAC study.

Usage (from any working directory)::

    python hdl/gen_verilog.py NAME [--out-dir DIR] [--force]
    python hdl/gen_verilog.py NAME --check [--out-dir DIR]
    python hdl/gen_verilog.py --list

``NAME`` is a key of :data:`src.sbox_tables.TABLES` (``DAC``, ``E-DAC``,
``DAC-OPT``, ``DAC-OPT-GOLD``).  Two files are written next to this script
(or into ``--out-dir``):

``<prefix>_sbox.v``
    ``module <prefix>_sbox`` (``S(x)`` as a fully specified ``case`` lookup)
    and ``module <prefix>_sbox_inv`` (``S^-1``), 5-bit in / 5-bit out,
    purely combinational.
``<prefix>_sbox_tb.v``
    A self-checking testbench: every input against the forward and inverse
    tables, ``S^-1(S(x)) == x`` through a chained instance and the
    distinctness of the 32 outputs.  It prints a PASS/FAIL summary and ends
    with ``$fatal`` (non-zero simulator exit status) on any mismatch::

        iverilog -g2012 -o sim hdl/<prefix>_sbox.v hdl/<prefix>_sbox_tb.v && vvp sim

``<prefix>`` is the lower-cased name with every run of non-alphanumerics
replaced by ``_`` (``DAC-OPT`` -> ``dac_opt``).  Existing files are never
overwritten without ``--force``; ``hdl/dac_sbox.v`` / ``hdl/dac_sbox_tb.v``
are hand-maintained (they also carry the substitution, permutation and
round layers of the SPN cipher) and are only *checked* by this script.

The output is deterministic (no timestamps), so ``tests/test_hdl.py`` can
verify that the committed files are up to date; :func:`parse_case_table`
and :func:`parse_vector_array` read the tables back out of any of the files.
No area or timing figures are emitted: those require synthesis with a
specific cell library or device and cannot be derived from a lookup-table
description.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HDL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HDL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.security_analysis import analyze_sbox  # noqa: E402
from src.sbox_tables import TABLES, FrozenSBox, inverse_table  # noqa: E402

N_BITS = 5
SIZE = 1 << N_BITS
RULE = "// " + "=" * 76
THIN_RULE = "// " + "-" * 76

#: Honest replacement for the base paper's unsupported "11.25 GE / 6.225 ns".
AREA_NOTE = (
    "Area / timing: no gate-equivalent or delay figures are given for this "
    "file.  Such numbers require synthesis with a specific cell library or "
    "device and are not derived from the HDL; none has been performed."
)

_CASE_ENTRY = re.compile(r"5'd(\d+)\s*:\s*data_out\s*=\s*5'd(\d+)\s*;")
_VECTOR_ENTRY = re.compile(
    r"\b(\w+)\[(\d+)\]\s*=\s*(\d+)'([bBdDhH])([0-9a-fA-F_]+)\s*;")


# ============================================================
# Naming and rendering
# ============================================================
def module_prefix(name: str) -> str:
    """Verilog identifier prefix of a table name (``DAC-OPT`` -> ``dac_opt``)."""
    prefix = re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")
    if not prefix or prefix[0].isdigit():
        raise ValueError(f"cannot derive a Verilog identifier from {name!r}")
    return prefix


def _comment(text: str, width: int = 76) -> List[str]:
    return ["// " + line for line in textwrap.wrap(text, width)]


def render_lut_module(module_name: str, table: Sequence[int], title: str) -> str:
    """One combinational ``case`` lookup module for a 32-entry table."""
    lines = [THIN_RULE, f"// {title}", THIN_RULE,
             f"module {module_name} (",
             "    input  wire [4:0] data_in,",
             "    output reg  [4:0] data_out",
             ");", "",
             "    always @(*) begin",
             "        case (data_in)"]
    for x, y in enumerate(table):
        label = f"5'd{x}:"
        lines.append(f"            {label:<7s}data_out = 5'd{y};")
    lines += ["            default: data_out = 5'd0;",
              "        endcase", "    end", "", "endmodule", ""]
    return "\n".join(lines)


def render_header(record: FrozenSBox, prefix: str) -> str:
    """File header: provenance, computed metrics and the area/timing note."""
    metrics = analyze_sbox(record.table, record.name)
    if record.strategy is None:
        provenance = "static table; no search to rerun"
    else:
        provenance = (f"src.sbox_optimizer.optimize_dac_sbox({record.strategy!r}, "
                      f"seed={record.seed}, iterations={record.iterations})")
    metric_text = (
        f"NL(S) {metrics['nl_sbox']} (coordinates "
        f"{'/'.join(str(v) for v in metrics['nl_per_bit'])}), max LAT bias "
        f"{metrics['max_bias']} (LAP {metrics['lap']:.4f}), differential uniformity "
        f"{metrics['differential_uniformity']}, boomerang uniformity "
        f"{metrics['boomerang_uniformity']}, algebraic degree "
        f"{metrics['min_degree']}..{metrics['max_degree']}, SAC {metrics['sac_avg']:.4f} "
        f"in [{metrics['sac_min']:.3f}, {metrics['sac_max']:.3f}], BIC-SAC "
        f"{metrics['bic_sac_avg']:.4f}; all {metrics['balanced_components']} nonzero "
        f"components balanced.")
    lines = [RULE,
             f"// {record.name} 5-bit substitution box - Verilog HDL (generated, do not edit)",
             RULE,
             f"// Generated by hdl/gen_verilog.py from src.sbox_tables.TABLES[{record.name!r}].",
             f"// Regenerate with:  python hdl/gen_verilog.py {record.name} --force",
             "//",
             *_comment(f"Table: {record.description}"),
             *_comment(f"Provenance: {provenance}"),
             "// S(x), x = 0..31 (hex): " + " ".join(f"{v:02X}" for v in record.table),
             "//",
             *_comment("Computed by src.security_analysis.analyze_sbox: " + metric_text),
             "//",
             f"// Modules: {prefix}_sbox (S), {prefix}_sbox_inv (S^-1); testbench {prefix}_sbox_tb.v.",
             "//",
             *_comment(AREA_NOTE),
             RULE, ""]
    return "\n".join(lines)


def render_sbox_file(record: FrozenSBox, prefix: Optional[str] = None) -> str:
    """Complete ``<prefix>_sbox.v``: header, forward and inverse modules."""
    prefix = module_prefix(record.name) if prefix is None else prefix
    return "\n".join([
        render_header(record, prefix),
        render_lut_module(f"{prefix}_sbox", record.table,
                          f"{record.name} S-box: data_out = S(data_in)"),
        "",
        render_lut_module(f"{prefix}_sbox_inv", record.inverse,
                          f"{record.name} inverse S-box: data_out = S^-1(data_in)"),
    ])


def _array_init(name: str, values: Sequence[int], width: int, per_line: int) -> List[str]:
    """``name[i] = width'dV;`` assignments, ``per_line`` per source line."""
    items = [f"{name}[{i}]".ljust(len(name) + 4) + f" = {width}'d{v};"
             for i, v in enumerate(values)]
    widest = max(len(item) for item in items)
    return ["        " + " ".join(item.ljust(widest) for item in items[k:k + per_line]).rstrip()
            for k in range(0, len(items), per_line)]


def render_testbench(record: FrozenSBox, prefix: Optional[str] = None) -> str:
    """Self-checking testbench for the modules of :func:`render_sbox_file`."""
    prefix = module_prefix(record.name) if prefix is None else prefix
    tb = f"{prefix}_sbox_tb"
    lines = [
        "`timescale 1ns / 1ps",
        RULE,
        f"// Self-checking testbench for {prefix}_sbox.v (generated by hdl/gen_verilog.py)",
        RULE,
        f"// Table: src.sbox_tables.TABLES[{record.name!r}].  Checks every input against",
        "// the forward and inverse tables, S^-1(S(x)) == x through a chained instance",
        "// and the distinctness of the 32 outputs.  Prints a PASS/FAIL summary; any",
        "// mismatch ends the simulation with $fatal (non-zero exit status).",
        f"// Run:  iverilog -g2012 -o sim {prefix}_sbox.v {prefix}_sbox_tb.v && vvp sim",
        RULE,
        f"module {tb};", "",
        "    reg  [4:0]  x;",
        f"    wire [4:0]  s_x;          // {prefix}_sbox(x)",
        f"    wire [4:0]  inv_x;        // {prefix}_sbox_inv(x)",
        f"    wire [4:0]  round_trip;   // {prefix}_sbox_inv({prefix}_sbox(x))",
        "    reg  [4:0]  expected_sbox [0:31];",
        "    reg  [4:0]  expected_inv  [0:31];",
        "    reg  [31:0] seen;",
        "    integer i;",
        "    integer errors_sbox, errors_inv, errors_round_trip, errors_bijective, errors;",
        "",
        f"    {prefix}_sbox     dut_sbox  (.data_in(x),   .data_out(s_x));",
        f"    {prefix}_sbox_inv dut_inv   (.data_in(x),   .data_out(inv_x));",
        f"    {prefix}_sbox_inv dut_chain (.data_in(s_x), .data_out(round_trip));",
        "",
        "    initial begin",
        f"        // S(x) and S^-1(x), x = 0..31, from src.sbox_tables.TABLES[{record.name!r}]",
        *_array_init("expected_sbox", record.table, 5, 4),
        *_array_init("expected_inv", record.inverse, 5, 4),
        "",
        "        errors_sbox = 0; errors_inv = 0; errors_round_trip = 0; errors_bijective = 0;",
        "        seen = 32'd0;",
        "        x = 5'd0;",
        "",
        "        for (i = 0; i < 32; i = i + 1) begin",
        "            x = i[4:0];",
        "            #10;",
        "            if (s_x !== expected_sbox[i]) begin",
        "                errors_sbox = errors_sbox + 1;",
        f"                $display(\"  FAIL {prefix}_sbox(%0d) = %0d, expected %0d\", i, s_x, expected_sbox[i]);",
        "            end",
        "            if (inv_x !== expected_inv[i]) begin",
        "                errors_inv = errors_inv + 1;",
        f"                $display(\"  FAIL {prefix}_sbox_inv(%0d) = %0d, expected %0d\", i, inv_x, expected_inv[i]);",
        "            end",
        "            if (round_trip !== x) begin",
        "                errors_round_trip = errors_round_trip + 1;",
        "                $display(\"  FAIL S^-1(S(%0d)) = %0d\", i, round_trip);",
        "            end",
        "            seen = seen | (32'd1 << s_x);",
        "        end",
        "        if (seen !== 32'hFFFFFFFF) begin",
        "            errors_bijective = 1;",
        f"            $display(\"  FAIL {prefix}_sbox outputs are not distinct (mask %h)\", seen);",
        "        end",
        "",
        "        errors = errors_sbox + errors_inv + errors_round_trip + errors_bijective;",
        f"        $display(\"{tb} summary: sbox %0d/32, inverse %0d/32, S^-1(S(x)) %0d/32 mismatches\",",
        "                 errors_sbox, errors_inv, errors_round_trip);",
        "        if (errors_bijective == 0)",
        f"            $display(\"{tb} bijectivity: PASS\");",
        "        else",
        f"            $display(\"{tb} bijectivity: FAIL\");",
        "        if (errors != 0) begin",
        f"            $display(\"{tb}: FAIL (%0d mismatches)\", errors);",
        f"            $fatal(1, \"{tb} failed\");",
        "        end",
        f"        $display(\"{tb}: PASS\");",
        "        $finish;",
        "    end", "",
        "endmodule", "",
    ]
    return "\n".join(lines)


def render_files(name: str) -> Dict[str, str]:
    """``{file name: content}`` for the table ``name`` (``KeyError`` if unknown)."""
    record = TABLES[name]
    prefix = module_prefix(name)
    return {f"{prefix}_sbox.v": render_sbox_file(record, prefix),
            f"{prefix}_sbox_tb.v": render_testbench(record, prefix)}


# ============================================================
# Reading tables back out of Verilog
# ============================================================
def module_body(text: str, module_name: str) -> str:
    """Source text between ``module <module_name>`` and its ``endmodule``."""
    pattern = rf"^[ \t]*module[ \t]+{re.escape(module_name)}\b(.*?)^[ \t]*endmodule\b"
    match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    if match is None:
        raise ValueError(f"module {module_name!r} not found")
    return match.group(1)


def parse_case_table(text: str, module_name: str) -> List[int]:
    """The 32-entry table encoded by the ``case`` statement of ``module_name``.

    Raises ``ValueError`` if the module is missing, an input is listed twice
    or not at all, a value is out of range or the table is not a permutation.
    """
    table: Dict[int, int] = {}
    for x_text, y_text in _CASE_ENTRY.findall(module_body(text, module_name)):
        x, y = int(x_text), int(y_text)
        if x in table:
            raise ValueError(f"{module_name}: input {x} is listed twice")
        if not (0 <= x < SIZE and 0 <= y < SIZE):
            raise ValueError(f"{module_name}: entry 5'd{x} -> 5'd{y} is out of range")
        table[x] = y
    if len(table) != SIZE:
        raise ValueError(f"{module_name}: {len(table)} of {SIZE} case entries found")
    values = [table[x] for x in range(SIZE)]
    if sorted(values) != list(range(SIZE)):
        raise ValueError(f"{module_name}: the table is not a permutation of 0..{SIZE - 1}")
    return values


def parse_vector_array(text: str, array_name: str) -> List[int]:
    """Values of the ``array_name[i] = <width>'<base><digits>;`` assignments.

    Indices must form ``0 .. len - 1`` without gaps or repeats, and every
    literal must fit its declared width.
    """
    values: Dict[int, int] = {}
    for name, index_text, width, base, digits in _VECTOR_ENTRY.findall(text):
        if name != array_name:
            continue
        index = int(index_text)
        value = int(digits.replace("_", ""), {"b": 2, "d": 10, "h": 16}[base.lower()])
        if value >= 1 << int(width):
            raise ValueError(f"{array_name}[{index}]: {width}'{base}{digits} exceeds {width} bits")
        if index in values:
            raise ValueError(f"{array_name}[{index}] is assigned twice")
        values[index] = value
    if not values:
        raise ValueError(f"array {array_name!r} not found")
    if sorted(values) != list(range(len(values))):
        raise ValueError(f"{array_name}: indices are not contiguous from 0")
    return [values[i] for i in range(len(values))]


# ============================================================
# Files
# ============================================================
def write_files(name: str, out_dir: Path | str = HDL_DIR, force: bool = False) -> List[Path]:
    """Write the two files of :func:`render_files` into ``out_dir``.

    Raises ``FileExistsError`` if a target exists and ``force`` is false.
    """
    files = render_files(name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / file_name for file_name in files]
    existing = [str(p) for p in paths if p.exists()]
    if existing and not force:
        raise FileExistsError("refusing to overwrite without --force: " + ", ".join(existing))
    for path, content in zip(paths, files.values()):
        path.write_text(content, encoding="ascii", newline="\n")
    return paths


def check_files(name: str, out_dir: Path | str = HDL_DIR) -> List[str]:
    """Problems found in ``<prefix>_sbox.v`` / ``<prefix>_sbox_tb.v`` (empty = consistent).

    The forward and inverse ``case`` tables and the testbench's expected
    arrays are parsed back and compared with the frozen table.  This works
    for the hand-maintained ``dac_sbox.v`` as well as for generated files.
    """
    record = TABLES[name]
    prefix = module_prefix(name)
    out_dir = Path(out_dir)
    expected = {
        (f"{prefix}_sbox.v", "module", f"{prefix}_sbox"): record.table,
        (f"{prefix}_sbox.v", "module", f"{prefix}_sbox_inv"): inverse_table(record.table),
        (f"{prefix}_sbox_tb.v", "array", "expected_sbox"): record.table,
        (f"{prefix}_sbox_tb.v", "array", "expected_inv"): inverse_table(record.table),
    }
    problems: List[str] = []
    for (file_name, kind, target), table in expected.items():
        path = out_dir / file_name
        if not path.exists():
            problems.append(f"{path}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        try:
            found = (parse_case_table(text, target) if kind == "module"
                     else parse_vector_array(text, target))
        except ValueError as exc:
            problems.append(f"{path}: {exc}")
            continue
        if found != list(table):
            problems.append(f"{path}: {kind} {target} does not match TABLES[{name!r}]")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate (or check) Verilog for a frozen DAC-study S-box table.")
    parser.add_argument("name", nargs="?", help="table name: " + ", ".join(TABLES))
    parser.add_argument("--out-dir", default=str(HDL_DIR),
                        help="output directory (default: the hdl/ directory)")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("--check", action="store_true",
                        help="parse the existing files back and compare with the table")
    parser.add_argument("--list", action="store_true", help="list the table names and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name, record in TABLES.items():
            print(f"{name:14s} {module_prefix(name)}_sbox.v  {record.description}")
        return 0
    if args.name is None:
        parser.error("NAME is required (use --list to see the choices)")
    if args.name not in TABLES:
        parser.error(f"unknown table {args.name!r}; choose from {', '.join(TABLES)}")

    if args.check:
        problems = check_files(args.name, args.out_dir)
        for problem in problems:
            print(f"MISMATCH {problem}")
        if problems:
            return 1
        print(f"OK {args.name}: Verilog tables in {args.out_dir} match src.sbox_tables")
        return 0

    try:
        paths = write_files(args.name, args.out_dir, args.force)
    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 1
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
