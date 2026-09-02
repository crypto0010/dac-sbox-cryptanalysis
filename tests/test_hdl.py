"""Tests of the Verilog sources in ``hdl/`` against the Python reference model.

No Verilog simulator is required (none is installed on the reference
machine): the ``case`` lookup tables, the permutation wiring and every
expected vector of the self-checking testbenches are parsed back out of the
``.v`` files with :mod:`hdl.gen_verilog` and compared with
:mod:`src.sbox` / :mod:`src.block_cipher` / :mod:`src.sbox_tables`.  The
generator itself is exercised end to end, including its CLI from a foreign
working directory, and the committed generated files are checked to be
byte-for-byte reproductions of the current tables.
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HDL_DIR = ROOT / "hdl"
sys.path.insert(0, str(ROOT))

import pytest

from src.block_cipher import PERM_TABLE, DACBlockCipher, permute_bits
from src.sbox import DAC_SBOX, DAC_SBOX_INV, DACSBox
from src.sbox_tables import (
    DAC_OPT_GOLD_SBOX, DAC_OPT_SBOX, DAC_OPT_SBOX_INV, TABLES, inverse_table,
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_verilog", HDL_DIR / "gen_verilog.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


def _read(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


DAC_SBOX_V = _read(HDL_DIR / "dac_sbox.v")
DAC_SBOX_TB_V = _read(HDL_DIR / "dac_sbox_tb.v")


# ------------------------------------------------------------
# Hand-written dac_sbox.v against the Python model
# ------------------------------------------------------------
def test_dac_sbox_case_tables_match_python():
    assert gen.parse_case_table(DAC_SBOX_V, "dac_sbox") == DAC_SBOX
    assert gen.parse_case_table(DAC_SBOX_V, "dac_sbox_inv") == DAC_SBOX_INV


def test_dac_opt_sbox_case_tables_match_python():
    text = _read(HDL_DIR / "dac_opt_sbox.v")
    assert gen.parse_case_table(text, "dac_opt_sbox") == DAC_OPT_SBOX
    assert gen.parse_case_table(text, "dac_opt_sbox_inv") == DAC_OPT_SBOX_INV
    assert DAC_OPT_SBOX_INV == inverse_table(DAC_OPT_SBOX)


def test_dac_permutation_layer_matches_perm_table():
    body = gen.module_body(DAC_SBOX_V, "dac_permutation_layer")
    assignments = re.findall(
        r"assign\s+data_out\[(\d+)\]\s*=\s*data_in\[(\d+)\]\s*;", body)
    assert len(assignments) == 32
    # data_in[b] is MSB-first index 31 - b; PERM_TABLE maps source index to target.
    perm = {31 - int(source): 31 - int(target) for target, source in assignments}
    assert [perm[i] for i in range(32)] == list(PERM_TABLE)


def test_dac_substitution_layer_wiring():
    body = gen.module_body(DAC_SBOX_V, "dac_substitution_layer")
    slices = [(int(hi), int(lo)) for hi, lo in
              re.findall(r"data_in\[(\d+):(\d+)\]", body)]
    # Six aligned 5-bit chunks, MSB first (paper Fig. 3 layout).
    assert slices == [(31, 27), (26, 22), (21, 17), (16, 12), (11, 7), (6, 2)]
    # The two remainder bits are complemented, not substituted.
    assert re.search(r"\{\s*~data_in\[1\]\s*,\s*~data_in\[0\]\s*\}", body)

    # Pin the output positions (no simulator available): the instance fed by
    # input chunk k must drive output chunk k of the final concatenation,
    # matching DACSBox.substitute_block32 (a swapped concatenation would agree
    # with every slice check above but scramble the layer).
    instances = re.findall(
        r"dac_sbox\s+\w+\s*\(\s*\.data_in\s*\(\s*data_in\[(\d+):(\d+)\]\s*\)\s*,"
        r"\s*\.data_out\s*\(\s*(\w+)\s*\)\s*\)\s*;", body)
    assert len(instances) == 6
    slice_of_wire = {wire: (int(hi), int(lo)) for hi, lo, wire in instances}
    assert len(slice_of_wire) == 6                     # distinct output wires
    concat = re.search(r"assign\s+data_out\s*=\s*\{([^;]*)\}\s*;", body)
    assert concat is not None
    parts = [part.strip() for part in concat.group(1).split(",")]
    assert parts[-1] == "remainder" and len(parts) == 7
    for chunk, wire in enumerate(parts[:6]):
        assert slice_of_wire[wire] == (31 - 5 * chunk, 27 - 5 * chunk), wire


def test_dac_testbench_vectors_match_python_model():
    text = DAC_SBOX_TB_V
    assert gen.parse_vector_array(text, "expected_sbox") == DAC_SBOX
    assert gen.parse_vector_array(text, "expected_inv") == DAC_SBOX_INV
    assert gen.parse_vector_array(text, "perm_pos") == list(PERM_TABLE)

    perm_in = gen.parse_vector_array(text, "perm_vec_in")
    perm_out = gen.parse_vector_array(text, "perm_vec_out")
    assert len(perm_in) == len(perm_out) >= 4
    for data, expected in zip(perm_in, perm_out):
        assert permute_bits(data) == expected

    round_in = gen.parse_vector_array(text, "round_vec_in")
    round_key = gen.parse_vector_array(text, "round_vec_key")
    round_skip = gen.parse_vector_array(text, "round_vec_skip")
    round_out = gen.parse_vector_array(text, "round_vec_out")
    assert len(round_in) == len(round_key) == len(round_skip) == len(round_out) >= 6
    assert set(round_skip) == {0, 1}          # both sides of the skip_perm mux
    assert any(skip == 1 and key == 0 for skip, key in zip(round_skip, round_key))
    sbox = DACSBox()
    for data, key, skip, expected in zip(round_in, round_key, round_skip, round_out):
        substituted = sbox.substitute_block32(data ^ key)
        model = substituted if skip else DACBlockCipher.permutation_layer(substituted)
        assert model == expected

    # The localparam vector counts agree with the array lengths.
    params = dict(re.findall(r"localparam\s+(\w+)\s*=\s*(\d+)\s*;", text))
    assert int(params["N_PERM"]) == len(perm_in)
    assert int(params["N_ROUND"]) == len(round_in)


@pytest.mark.parametrize("file_name", ["dac_sbox_tb.v", "dac_opt_sbox_tb.v"])
def test_testbenches_are_self_checking(file_name):
    text = _read(HDL_DIR / file_name)
    assert "$fatal" in text               # non-zero exit status on failure
    assert "PASS" in text and "FAIL" in text
    assert text.count("!==") >= 3         # case inequality catches X/Z outputs


def test_hdl_headers_make_no_unsupported_hardware_claims():
    """The old header presented the paper's 11.25 GE / 6.225 ns as results of
    this HDL; the honest note must have replaced them in every .v file."""
    for path in sorted(HDL_DIR.glob("*.v")):
        text = _read(path)
        assert "11.25" not in text, path.name
        assert "6.225" not in text, path.name
    for file_name in ("dac_sbox.v", "dac_opt_sbox.v"):
        assert "not derived from the HDL" in _read(HDL_DIR / file_name)


# ------------------------------------------------------------
# Generator: committed files, rendering, parsing, CLI
# ------------------------------------------------------------
def test_committed_generated_files_are_up_to_date():
    for file_name, content in gen.render_files("DAC-OPT").items():
        assert _read(HDL_DIR / file_name) == content, file_name
    assert gen.check_files("DAC-OPT", HDL_DIR) == []
    # The hand-maintained DAC files pass the same consistency check.
    assert gen.check_files("DAC", HDL_DIR) == []


def test_generated_tables_round_trip_for_every_frozen_table(tmp_path):
    for name, record in TABLES.items():
        prefix = gen.module_prefix(name)
        paths = gen.write_files(name, tmp_path)
        assert [path.name for path in paths] == [f"{prefix}_sbox.v", f"{prefix}_sbox_tb.v"]
        text = _read(tmp_path / f"{prefix}_sbox.v")
        assert gen.parse_case_table(text, f"{prefix}_sbox") == record.table
        assert gen.parse_case_table(text, f"{prefix}_sbox_inv") == record.inverse
        tb_text = _read(tmp_path / f"{prefix}_sbox_tb.v")
        assert gen.parse_vector_array(tb_text, "expected_sbox") == record.table
        assert gen.parse_vector_array(tb_text, "expected_inv") == record.inverse
        assert gen.check_files(name, tmp_path) == []
    # DAC generated afresh agrees with the hand-written module.
    generated = _read(tmp_path / "dac_sbox.v")
    assert (gen.parse_case_table(generated, "dac_sbox")
            == gen.parse_case_table(DAC_SBOX_V, "dac_sbox"))


def test_generator_refuses_to_overwrite_without_force(tmp_path):
    gen.write_files("DAC-OPT-GOLD", tmp_path)
    with pytest.raises(FileExistsError):
        gen.write_files("DAC-OPT-GOLD", tmp_path)
    gen.write_files("DAC-OPT-GOLD", tmp_path, force=True)   # explicit overwrite works
    assert gen.check_files("DAC-OPT-GOLD", tmp_path) == []


def test_module_prefix_naming():
    assert gen.module_prefix("DAC") == "dac"
    assert gen.module_prefix("DAC-OPT") == "dac_opt"
    assert gen.module_prefix("DAC-OPT-GOLD") == "dac_opt_gold"
    assert gen.module_prefix("E-DAC") == "e_dac"
    with pytest.raises(ValueError):
        gen.module_prefix("---")


def test_parsers_reject_broken_sources():
    with pytest.raises(ValueError):
        gen.module_body(DAC_SBOX_V, "no_such_module")
    complete = gen.render_lut_module("broken", DAC_SBOX, "t")
    with pytest.raises(ValueError):                     # a missing entry
        gen.parse_case_table(complete.replace("5'd7:  data_out = 5'd22;", "", 1), "broken")
    with pytest.raises(ValueError):                     # a duplicated input
        gen.parse_case_table(complete.replace(
            "5'd7:  data_out = 5'd22;",
            "5'd6:  data_out = 5'd22;", 1), "broken")
    constant = gen.render_lut_module("broken", [0] * 32, "t")
    with pytest.raises(ValueError):                     # not a permutation
        gen.parse_case_table(constant, "broken")
    with pytest.raises(ValueError):
        gen.parse_vector_array("module m; endmodule", "expected_sbox")
    with pytest.raises(ValueError):                     # literal wider than declared
        gen.parse_vector_array("a[0] = 5'd32;", "a")


def test_generator_cli_runs_from_any_cwd(tmp_path):
    script = str(HDL_DIR / "gen_verilog.py")
    out_dir = tmp_path / "generated"
    result = subprocess.run(
        [sys.executable, script, "DAC-OPT-GOLD", "--out-dir", str(out_dir)],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = _read(out_dir / "dac_opt_gold_sbox.v")
    assert gen.parse_case_table(text, "dac_opt_gold_sbox") == DAC_OPT_GOLD_SBOX
    check = subprocess.run(
        [sys.executable, script, "DAC-OPT-GOLD", "--check", "--out-dir", str(out_dir)],
        cwd=tmp_path, capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    again = subprocess.run(
        [sys.executable, script, "DAC-OPT-GOLD", "--out-dir", str(out_dir)],
        cwd=tmp_path, capture_output=True, text=True)
    assert again.returncode == 1 and "--force" in again.stderr
    listing = subprocess.run(
        [sys.executable, script, "--list"],
        cwd=tmp_path, capture_output=True, text=True)
    assert listing.returncode == 0
    assert all(name in listing.stdout for name in TABLES)
