"""Common-condition logic synthesis of 5-bit S-boxes on two open cell libraries.

Every S-box is described in the same way -- a 32-entry ``case`` lookup, the
form in which a designer receives a table -- and run through the same open
tool flow:

1. Yosys (YoWASP build, version pinned in ``requirements.txt``)::

       read_liberty -lib <lib>; read_verilog sbox.v; synth -flatten -top sbox
       abc -liberty <lib> [-dont_use ...]; opt_clean; stat -liberty <lib>
       write_json sbox.json

   ``abc -liberty`` is ABC's standard delay-oriented mapping with area
   recovery (``&dch -f; &nf``) onto the cells of the library.
2. The mapped netlist is read back and evaluated gate by gate with the
   Boolean ``function`` of every cell from the Liberty file; all 32 inputs
   must reproduce the table (``functionally_verified``), otherwise the run
   fails.
3. From the same Liberty data (non-linear delay and power models):

   * **area** -- sum of cell areas, also in gate equivalents (GE) of the
     library's two-input NAND (``NAND2_X1`` / ``nand2_1``);
   * **logic depth** -- cells on the longest input-to-output path;
   * **critical-path delay** -- arc-based static timing, worst of rise and
     fall on every arc, bilinear table interpolation; each input driven by
     the reference inverter, each output loaded by one reference-inverter
     input;
   * **dynamic energy per evaluation** -- zero-delay switching activity
     averaged over all 1024 ordered pairs of consecutive inputs (exact for
     uniformly random inputs, glitches excluded): output-pin internal energy
     interpolated at the actual slew and load, input-pin internal energy
     where the library provides it, and ``C V^2 / 2`` for every switched
     net capacitance;
   * **leakage power** -- sum of ``cell_leakage_power``.

These are pre-layout estimates (no wires, no placement, no parasitics, one
corner); they rank the S-boxes under one fixed flow and are not sign-off
figures.  The libraries are downloaded once from a pinned commit of the
OpenROAD-flow-scripts repository and checked against SHA-256 digests; they
are not redistributed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

N_BITS = 5
SIZE = 1 << N_BITS

#: Pinned source of the Liberty files (OpenROAD-flow-scripts, tag v3.0).
ORFS_COMMIT = "181e9133776117ea1b9f74dbacbfdaadff8c331b"
ORFS_RAW = f"https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/{ORFS_COMMIT}/"


@dataclass(frozen=True)
class Library:
    """A standard-cell library and the reference cells used by the flow."""

    key: str
    label: str
    path: str                       #: path inside the ORFS repository
    sha256: str
    nand2: str                      #: GE reference cell
    inverter: Tuple[str, str, str]  #: reference driver / load: (cell, input pin, output pin)
    dont_use: Tuple[str, ...] = ()

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    @property
    def url(self) -> str:
        return ORFS_RAW + self.path


LIBRARIES: Dict[str, Library] = {
    "nangate45": Library(
        key="nangate45",
        label="NanGate FreePDK45 Open Cell Library, typical corner (1.1 V, 25 C)",
        path="flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib",
        sha256="8d540a4d4cf6d09d27c87ad067857a9c0c2eeb023ab7a56e058cd3113db4e9b1",
        nand2="NAND2_X1", inverter=("INV_X1", "A", "ZN"),
        # never use the tie, filler, antenna and tristate cells for logic
        dont_use=("ANTENNA*", "FILLCELL*", "LOGIC0*", "LOGIC1*", "TBUF*", "TINV*", "CLKGATE*")),
    "sky130hd": Library(
        key="sky130hd",
        label="SkyWater SKY130 high-density library, typical corner (1.8 V, 25 C)",
        path="flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib",
        sha256="ec0e1067a35c8bf20b11e58d1e8ac53326067e4dac84a125cc1b917a3518d0d9",
        nand2="sky130_fd_sc_hd__nand2_1", inverter=("sky130_fd_sc_hd__inv_1", "A", "Y"),
        # the OpenROAD sky130hd don't-use set: power-gating, probe, clock and delay cells
        dont_use=("*lpflow*", "*probe*", "*clk*", "*dly*", "*conb*", "*ebuf*", "*einv*")),
}

#: Hand-written structural descriptions of two reference S-boxes, from their
#: specifications, synthesised with the same flow as a sanity check of how
#: far the lookup-table description is from a designer's circuit.
STRUCTURAL_REFERENCES: Dict[str, str] = {
    "ASCON": """
    wire x0 = x[4] ^ x[0];
    wire x4 = x[0] ^ x[1];
    wire x2 = x[2] ^ x[3];
    wire x1 = x[3];
    wire x3 = x[1];
    wire t0 = ~x0 & x1, t1 = ~x1 & x2, t2 = ~x2 & x3, t3 = ~x3 & x4, t4 = ~x4 & x0;
    wire y0 = x0 ^ t1, y1 = x1 ^ t2, y2 = x2 ^ t3, y3 = x3 ^ t4, y4 = x4 ^ t0;
    assign y = {y0 ^ y4, y1 ^ y0, ~y2, y3 ^ y2, y4};
""",
    "Keccak-chi": """
    assign y = {x[4] ^ (~x[0] & x[1]), x[3] ^ (~x[4] & x[0]), x[2] ^ (~x[3] & x[4]),
                x[1] ^ (~x[2] & x[3]), x[0] ^ (~x[1] & x[2])};
""",
}


# ------------------------------------------------------------------
# Library download
# ------------------------------------------------------------------
def default_cache_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "hdl", "synth", "libs")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_library(lib: Library, cache_dir: Optional[str] = None) -> str:
    """Path of the verified Liberty file, downloading it once if needed."""
    cache_dir = cache_dir or default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, lib.filename)
    if not os.path.exists(path) or _sha256(path) != lib.sha256:
        tmp = path + ".part"
        with urllib.request.urlopen(lib.url, timeout=120) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
        if _sha256(tmp) != lib.sha256:
            os.remove(tmp)
            raise RuntimeError(f"{lib.filename}: SHA-256 mismatch after download from {lib.url}")
        os.replace(tmp, path)
    return path


# ------------------------------------------------------------------
# Liberty parsing (the subset the flow needs)
# ------------------------------------------------------------------
_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[{}();:,]|[^\s{}();:,"]+')


@dataclass
class Group:
    name: str
    args: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)
    groups: List["Group"] = field(default_factory=list)

    def all(self, name: str) -> List["Group"]:
        return [g for g in self.groups if g.name == name]


def _clean(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return text.replace("\\\r\n", " ").replace("\\\n", " ")


def _unquote(token: str) -> str:
    return token[1:-1] if token.startswith('"') else token


def _parse_group(tokens: List[str], pos: int) -> Tuple[Group, int]:
    name = tokens[pos]
    assert tokens[pos + 1] == "(", f"expected '(' after {name}"
    pos += 2
    args: List[str] = []
    while tokens[pos] != ")":
        if tokens[pos] != ",":
            args.append(_unquote(tokens[pos]))
        pos += 1
    pos += 1
    assert tokens[pos] == "{", f"expected '{{' in group {name}"
    pos += 1
    group = Group(name, args)
    while tokens[pos] != "}":
        key = tokens[pos]
        if tokens[pos + 1] == ":":
            value = []
            pos += 2
            while tokens[pos] != ";" and tokens[pos] != "}":
                value.append(_unquote(tokens[pos]))
                pos += 1
            if tokens[pos] == ";":
                pos += 1
            group.attrs[key] = " ".join(value)
        elif tokens[pos + 1] == "(":
            # complex attribute ``key (args);`` or nested group ``key (args) {...}``
            close = pos + 2
            while tokens[close] != ")":
                close += 1
            if tokens[close + 1] == "{":
                child, pos = _parse_group(tokens, pos)
                group.groups.append(child)
            else:
                values = [_unquote(t) for t in tokens[pos + 2:close] if t != ","]
                group.attrs.setdefault(key, values)
                pos = close + 1
                if tokens[pos] == ";":
                    pos += 1
        else:
            pos += 1
    return group, pos + 1


def _parse(text: str) -> Group:
    tokens = _TOKEN.findall(_clean(text))
    group, _ = _parse_group(tokens, 0)
    return group


def _floats(values: Any) -> List[float]:
    if isinstance(values, str):
        values = [values]
    out: List[float] = []
    for value in values:
        out.extend(float(v) for v in str(value).replace('"', "").split(",") if v.strip())
    return out


@dataclass
class Table:
    """A 1-D or 2-D Liberty lookup table with named axes."""

    variables: Tuple[str, ...]
    index: Tuple[List[float], ...]
    values: List[List[float]]

    def lookup(self, slew: float, load: float) -> float:
        coords = []
        for var in self.variables:
            coords.append(load if "capacitance" in var else slew)
        if len(self.index) == 1:
            return _interp1(self.index[0], self.values[0], coords[0])
        rows = [_interp1(self.index[1], row, coords[1]) for row in self.values]
        return _interp1(self.index[0], rows, coords[0])


def _interp1(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """Piecewise-linear interpolation, linear extrapolation at both ends."""
    if len(xs) == 1:
        return ys[0]
    i = 0
    while i < len(xs) - 2 and x > xs[i + 1]:
        i += 1
    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class Cell:
    name: str
    area: float
    leakage: float
    inputs: Dict[str, float]                       #: pin -> capacitance
    output: str
    function: str
    delay: Dict[str, List[Table]]                  #: related pin -> [cell_rise, cell_fall]
    transition: Dict[str, List[Table]]             #: related pin -> [rise_tr, fall_tr]
    out_power: Dict[str, List[Tuple[Table, Table]]]  #: related pin -> [(rise, fall)]
    in_power: Dict[str, List[Tuple[Table, Table]]]   #: input pin -> [(rise, fall)]


class LibertyLibrary:
    """Units, templates and lazily parsed cells of one Liberty file."""

    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            self.text = handle.read()
        head = self.text[:200_000]
        cap = re.search(r"capacitive_load_unit\s*\(\s*([\d.eE+-]+)\s*,\s*\"?(\w+)\"?\s*\)", head)
        scale = float(cap.group(1)) * {"ff": 1.0, "pf": 1e3}[cap.group(2).lower()]
        self.cap_to_ff = scale                 #: capacitance unit -> fF
        self.energy_to_fj = scale              #: internal energy unit (cap unit * V^2) -> fJ
        self.time_to_ps = 1e3                  #: both libraries use 1 ns
        self.voltage = float(re.search(r"nom_voltage\s*:\s*([\d.]+)", head).group(1))
        self.leakage_to_nw = 1.0               #: both libraries use 1 nW
        self.templates: Dict[str, Tuple[str, ...]] = {}
        self.template_index: Dict[str, Dict[int, List[float]]] = {}
        for match in re.finditer(r"(lu_table_template|power_lut_template)\s*\(\s*\"?([\w.]+)\"?"
                                 r"\s*\)\s*\{([^}]*)\}", self.text):
            body = match.group(3)
            variables = tuple(v for v in re.findall(r"variable_\d\s*:\s*\"?(\w+)\"?", body))
            self.templates[match.group(2)] = variables
            self.template_index[match.group(2)] = {
                int(k): _floats(v) for k, v in re.findall(r"index_(\d)\s*\(\s*\"([^\"]*)\"", body)}
        self._cells: Dict[str, Cell] = {}

    def _cell_text(self, name: str) -> str:
        match = re.search(r"cell\s*\(\s*\"?" + re.escape(name) + r"\"?\s*\)\s*\{", self.text)
        if match is None:
            raise KeyError(f"cell {name} not in library")
        depth, index = 0, match.end() - 1
        while True:
            char = self.text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return self.text[match.start():index + 1]
            index += 1

    def _table(self, group: Group) -> Table:
        template = group.args[0] if group.args else ""
        variables = self.templates.get(template, ())
        if not variables or template in ("scalar",):
            return Table(("input_net_transition",), ([0.0],), [[_floats(group.attrs["values"])[0]]])
        index = []
        for k in range(1, len(variables) + 1):
            values = group.attrs.get(f"index_{k}")
            index.append(_floats(values) if values is not None
                         else self.template_index[template][k])
        flat = _floats(group.attrs["values"])
        if len(index) == 1:
            values = [flat]
        else:
            width = len(index[1])
            values = [flat[i:i + width] for i in range(0, len(flat), width)]
        return Table(variables, tuple(index), values)

    def cell(self, name: str) -> Cell:
        if name in self._cells:
            return self._cells[name]
        group = _parse(_clean(self._cell_text(name)))
        leakage = group.attrs.get("cell_leakage_power")
        if leakage is None:
            values = [float(g.attrs["value"]) for g in group.all("leakage_power")]
            leakage = sum(values) / len(values) if values else 0.0
        inputs: Dict[str, float] = {}
        in_power: Dict[str, List[Tuple[Table, Table]]] = {}
        output, function = "", ""
        delay: Dict[str, List[Table]] = {}
        transition: Dict[str, List[Table]] = {}
        out_power: Dict[str, List[Tuple[Table, Table]]] = {}
        for pin in group.all("pin"):
            pin_name = pin.args[0]
            direction = pin.attrs.get("direction", "")
            if direction == "input":
                inputs[pin_name] = float(pin.attrs.get("capacitance", 0.0))
                for power in pin.all("internal_power"):
                    rise, fall = power.all("rise_power"), power.all("fall_power")
                    if rise and fall:
                        in_power.setdefault(pin_name, []).append(
                            (self._table(rise[0]), self._table(fall[0])))
            elif direction == "output":
                output, function = pin_name, pin.attrs.get("function", "")
                for timing in pin.all("timing"):
                    related = timing.attrs.get("related_pin", "")
                    tables = {g.name: self._table(g) for g in timing.groups}
                    if "cell_rise" in tables and "cell_fall" in tables:
                        delay.setdefault(related, []).extend(
                            [tables["cell_rise"], tables["cell_fall"]])
                        transition.setdefault(related, []).extend(
                            [tables["rise_transition"], tables["fall_transition"]])
                for power in pin.all("internal_power"):
                    related = power.attrs.get("related_pin", "")
                    rise, fall = power.all("rise_power"), power.all("fall_power")
                    if rise and fall:
                        out_power.setdefault(related, []).append(
                            (self._table(rise[0]), self._table(fall[0])))
        cell = Cell(name, float(group.attrs["area"]), float(leakage), inputs, output,
                    function, delay, transition, out_power, in_power)
        self._cells[name] = cell
        return cell


# ------------------------------------------------------------------
# Liberty Boolean functions
# ------------------------------------------------------------------
_FTOKEN = re.compile(r"\s*([A-Za-z_][\w\[\]]*|[01]|[!'&*|+^()])")


def compile_function(expr: str):
    """Liberty ``function`` -> Python callable on a dict of 0/1 pin values.

    Precedence (Liberty): ``!`` and postfix ``'``, then ``^``, then AND
    (``&``, ``*`` or juxtaposition), then OR (``|``, ``+``).
    """
    tokens = [t for t in _FTOKEN.findall(expr)]
    pos = 0

    def peek() -> Optional[str]:
        return tokens[pos] if pos < len(tokens) else None

    def take() -> str:
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def primary():
        tok = take()
        if tok == "!":
            inner = primary()
            node = lambda v, f=inner: 1 - f(v)  # noqa: E731
        elif tok == "(":
            node = expr_or()
            assert take() == ")"
        elif tok in ("0", "1"):
            node = lambda v, c=int(tok): c  # noqa: E731
        else:
            node = lambda v, n=tok: v[n]  # noqa: E731
        while peek() == "'":
            take()
            node = lambda v, f=node: 1 - f(v)  # noqa: E731
        return node

    def expr_xor():
        node = primary()
        while peek() == "^":
            take()
            right = primary()
            node = lambda v, a=node, b=right: a(v) ^ b(v)  # noqa: E731
        return node

    def expr_and():
        node = expr_xor()
        while peek() is not None and (peek() in ("&", "*") or peek() == "!" or peek() == "("
                                      or re.match(r"[A-Za-z_01]", peek())):
            if peek() in ("&", "*"):
                take()
            right = expr_xor()
            node = lambda v, a=node, b=right: a(v) & b(v)  # noqa: E731
        return node

    def expr_or():
        node = expr_and()
        while peek() in ("|", "+"):
            take()
            right = expr_and()
            node = lambda v, a=node, b=right: a(v) | b(v)  # noqa: E731
        return node

    result = expr_or()
    if pos != len(tokens):
        raise ValueError(f"unparsed function {expr!r}")
    return result


# ------------------------------------------------------------------
# Yosys
# ------------------------------------------------------------------
def yosys_command() -> List[str]:
    exe = shutil.which("yowasp-yosys")
    if exe is None:
        scripts = os.path.join(os.path.dirname(sys.executable), "Scripts")
        for candidate in ("yowasp-yosys.exe", "yowasp-yosys"):
            if os.path.exists(os.path.join(scripts, candidate)):
                return [os.path.join(scripts, candidate)]
        raise RuntimeError("yowasp-yosys not found; pip install -r requirements.txt")
    return [exe]


def yosys_version() -> str:
    try:
        from importlib.metadata import version
        return "yowasp-yosys " + version("yowasp-yosys")
    except Exception:  # pragma: no cover - metadata missing
        return "yowasp-yosys (unknown version)"


def lookup_verilog(table: Sequence[int]) -> str:
    lines = ["module sbox (input [4:0] x, output reg [4:0] y);", "  always @(*) begin",
             "    case (x)"]
    lines += [f"      5'd{x}: y = 5'd{int(v)};" for x, v in enumerate(table)]
    lines += ["    endcase", "  end", "endmodule", ""]
    return "\n".join(lines)


def structural_verilog(body: str) -> str:
    return f"module sbox (input [4:0] x, output [4:0] y);\n{body}\nendmodule\n"


def run_yosys(verilog: str, lib: Library, lib_path: str, tag: str) -> Dict[str, Any]:
    """Synthesize ``module sbox``; returns the parsed JSON netlist and the log."""
    work = os.path.dirname(lib_path)        # YoWASP only sees the working directory
    os.makedirs(os.path.join(work, "work"), exist_ok=True)
    stem = f"work/{tag}_{lib.key}"
    with open(os.path.join(work, stem + ".v"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(verilog)
    dont_use = " ".join(f"-dont_use {pattern}" for pattern in lib.dont_use)
    script = (f"read_liberty -lib {lib.filename}; read_verilog {stem}.v; "
              f"synth -flatten -top sbox; abc -liberty {lib.filename} {dont_use}; "
              f"opt_clean; stat -liberty {lib.filename}; write_json {stem}.json")
    proc = subprocess.run(yosys_command() + ["-q", "-l", stem + ".log", "-p", script],
                          cwd=work, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"yosys failed for {tag}/{lib.key}: {proc.stderr[-2000:]}")
    with open(os.path.join(work, stem + ".json"), "r", encoding="utf-8") as handle:
        netlist = json.load(handle)
    with open(os.path.join(work, stem + ".log"), "r", encoding="utf-8") as handle:
        log = handle.read()
    area = re.findall(r"Chip area for module '\\sbox': ([\d.]+)", log)
    return {"netlist": netlist["modules"]["sbox"], "yosys_area": float(area[-1]),
            "script": script}


# ------------------------------------------------------------------
# Netlist evaluation
# ------------------------------------------------------------------
@dataclass
class Instance:
    name: str
    cell: Cell
    pins: Dict[str, int]       #: input pin -> net id (>= 2) or constant 0 / 1
    out: int
    func: Any


def _bit(bit: Any) -> int:
    """Yosys JSON bit: net ids are integers >= 2, constants the strings "0"/"1"."""
    if isinstance(bit, str):
        if bit not in ("0", "1"):
            raise RuntimeError(f"undefined constant {bit!r} in the mapped netlist")
        return int(bit)
    return int(bit)


def _netlist(module: Mapping[str, Any], library: LibertyLibrary):
    in_bits = [_bit(b) for b in module["ports"]["x"]["bits"]]     # LSB first
    out_bits = [_bit(b) for b in module["ports"]["y"]["bits"]]
    instances: List[Instance] = []
    for name, cell in sorted(module["cells"].items()):
        model = library.cell(cell["type"])
        conns = cell["connections"]
        pins = {pin: _bit(conns[pin][0]) for pin in model.inputs}
        instances.append(Instance(name, model, pins, _bit(conns[model.output][0]),
                                  compile_function(model.function)))
    driven = {inst.out: inst for inst in instances}
    order: List[Instance] = []
    state: Dict[str, int] = {}

    def visit(inst: Instance) -> None:
        if state.get(inst.name) == 2:
            return
        if state.get(inst.name) == 1:
            raise RuntimeError("combinational loop in the mapped netlist")
        state[inst.name] = 1
        for net in inst.pins.values():
            if net in driven:
                visit(driven[net])
        state[inst.name] = 2
        order.append(inst)

    for inst in instances:
        visit(inst)
    return in_bits, out_bits, order


def _simulate(x: int, in_bits, out_bits, order) -> Tuple[int, Dict[int, int]]:
    values: Dict[int, int] = {0: 0, 1: 1}
    for k, net in enumerate(in_bits):
        values[net] = (x >> k) & 1
    for inst in order:
        values[inst.out] = inst.func({pin: values[net] for pin, net in inst.pins.items()})
    y = 0
    for k, net in enumerate(out_bits):
        y |= values[net] << k
    return y, values


def evaluate(table: Sequence[int], module: Mapping[str, Any], library: LibertyLibrary,
             lib: Library) -> Dict[str, Any]:
    """Functional check, area, depth, delay, energy and leakage of a mapped netlist."""
    in_bits, out_bits, order = _netlist(module, library)
    states = []
    for x in range(SIZE):
        y, values = _simulate(x, in_bits, out_bits, order)
        if y != int(table[x]):
            raise RuntimeError(f"mapped netlist computes S({x}) = {y}, table says {table[x]}")
        states.append(values)

    ref = library.cell(lib.inverter[0])
    ref_cap = ref.inputs[lib.inverter[1]]
    # capacitive load of every net (fanout input pins + reference load on outputs)
    load: Dict[int, float] = {}
    for inst in order:
        for pin, net in inst.pins.items():
            if net >= 2:
                load[net] = load.get(net, 0.0) + inst.cell.inputs[pin]
    for net in out_bits:
        if net >= 2:
            load[net] = load.get(net, 0.0) + ref_cap
    # input slew: the reference inverter driving the input net
    first = ref.transition[lib.inverter[1]][0]
    base_slew = min(next(axis for var, axis in zip(first.variables, first.index)
                         if "transition" in var))
    slew: Dict[int, float] = {}
    arrival: Dict[int, float] = {}
    level: Dict[int, int] = {}
    for net in in_bits:
        tr = ref.transition[lib.inverter[1]]
        slew[net] = max(t.lookup(base_slew, load.get(net, 0.0)) for t in tr)
        arrival[net] = 0.0
        level[net] = 0
    for inst in order:
        c_load = load.get(inst.out, 0.0)
        best_arrival, best_slew, best_level = 0.0, 0.0, 0
        for pin, net in inst.pins.items():
            if net < 2 or net not in arrival:
                continue
            tables = inst.cell.delay.get(pin)
            if not tables:
                continue
            d = max(t.lookup(slew[net], c_load) for t in tables)
            s = max(t.lookup(slew[net], c_load) for t in inst.cell.transition[pin])
            best_arrival = max(best_arrival, arrival[net] + d)
            best_slew = max(best_slew, s)
            best_level = max(best_level, level[net])
        arrival[inst.out] = best_arrival
        slew[inst.out] = best_slew
        level[inst.out] = best_level + 1
    out_nets = [n for n in out_bits if n >= 2]
    delay_ps = max(arrival.get(n, 0.0) for n in out_nets) * library.time_to_ps
    depth = max(level.get(n, 0) for n in out_nets)

    # dynamic energy over all ordered input pairs (zero-delay activity)
    v2 = library.voltage ** 2
    total = 0.0
    for x0 in range(SIZE):
        for x1 in range(SIZE):
            a, b = states[x0], states[x1]
            if x0 == x1:
                continue
            energy = 0.0
            for net in set(load):
                if a.get(net) != b.get(net):
                    energy += 0.5 * load[net] * library.cap_to_ff * v2
            for inst in order:
                toggled = [pin for pin, net in inst.pins.items() if a[net] != b[net]]
                for pin in toggled:
                    rising = b[inst.pins[pin]] == 1
                    for rise, fall in inst.cell.in_power.get(pin, []):
                        table = rise if rising else fall
                        energy += table.lookup(slew[inst.pins[pin]], 0.0) * library.energy_to_fj
                if a.get(inst.out) != b.get(inst.out) and toggled:
                    rising = b[inst.out] == 1
                    c_load = load.get(inst.out, 0.0)
                    share = 0.0
                    for pin in toggled:
                        groups = inst.cell.out_power.get(pin, [])
                        if groups:
                            share += sum((r if rising else f).lookup(slew[inst.pins[pin]], c_load)
                                         for r, f in groups) / len(groups)
                    energy += share / len(toggled) * library.energy_to_fj
            total += energy
    energy_fj = total / (SIZE * SIZE)

    nand2 = library.cell(lib.nand2).area
    area = sum(inst.cell.area for inst in order)
    histogram: Dict[str, int] = {}
    for inst in order:
        histogram[inst.cell.name] = histogram.get(inst.cell.name, 0) + 1
    return {
        "functionally_verified": True,
        "cells": len(order),
        "area_um2": round(area, 4),
        "area_ge": round(area / nand2, 2),
        "logic_depth": int(depth),
        "delay_ps": round(delay_ps, 1),
        "energy_fj_per_evaluation": round(energy_fj, 2),
        "leakage_nw": round(sum(inst.cell.leakage for inst in order) * library.leakage_to_nw, 4),
        "cell_histogram": dict(sorted(histogram.items())),
    }


def synthesize(table: Sequence[int], lib_key: str, tag: str,
               structural: Optional[str] = None,
               cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """Synthesize one S-box on one library and evaluate the netlist."""
    lib = LIBRARIES[lib_key]
    path = fetch_library(lib, cache_dir)
    verilog = structural_verilog(structural) if structural else lookup_verilog(table)
    run = run_yosys(verilog, lib, path, tag)
    library = _library_cache(path)
    result = evaluate(table, run["netlist"], library, lib)
    if abs(result["area_um2"] - run["yosys_area"]) > 1e-3:
        raise RuntimeError(f"area disagrees with yosys stat: {result['area_um2']} vs "
                           f"{run['yosys_area']}")
    result["description"] = "structural" if structural else "lookup table"
    return result


_LIB_CACHE: Dict[str, LibertyLibrary] = {}


def _library_cache(path: str) -> LibertyLibrary:
    if path not in _LIB_CACHE:
        _LIB_CACHE[path] = LibertyLibrary(path)
    return _LIB_CACHE[path]


def library_record(lib_key: str) -> Dict[str, Any]:
    lib = LIBRARIES[lib_key]
    library = _library_cache(fetch_library(lib))
    return {
        "label": lib.label, "source": lib.url, "sha256": lib.sha256,
        "ge_reference_cell": lib.nand2,
        "ge_reference_area_um2": library.cell(lib.nand2).area,
        "reference_inverter": lib.inverter[0],
        "voltage_v": library.voltage,
        "dont_use": list(lib.dont_use),
    }
