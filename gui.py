#!/usr/bin/env python3
"""
gui.py — TOML editor and simulation launcher.
No dependency on sim/* modules — interfaces only with simulation.toml and main.py.
Usage: python gui.py [path/to/simulation.toml]
"""
import base64
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import tomllib
from misc.gen_icon import build_icon as _build_icon

_ICON_B64 = base64.b64encode(_build_icon()).decode()


# ── Form schema ──────────────────────────────────────────────────────────────
#
# Tabs/Sections/Fields are plain data; a Tab is rendered onto a scrollable
# Frame by `_render_tab()` and read/written by `_populate_from_schema()` /
# `_collect_from_schema()` (defined further down once their helpers exist).
# Step 1 ports just the General tab — Sweep, Output, Amplifier, and Carriers
# still use the hand-coded methods.

@dataclass(frozen=True)
class Field:
    """One labelled input bound to a TOML path.

    ``path`` is the nested-key tuple inside the cfg dict (e.g.
    ``("simulation", "seed")`` → ``cfg["simulation"]["seed"]``).  The
    internal var key is ``path`` joined by ``"."``.  ``type`` is one of:

      "int"               StringVar + Entry, parsed as int.
      "float"             StringVar + Entry, parsed as float.
      "float_optional"    StringVar + Entry; blank → omitted from cfg.
      "str" / "path"      StringVar + Entry; "path" adds a Browse… button.
      "bool"              BooleanVar + Checkbutton (label is button text).
      "float_list"        StringVar + Entry of comma-separated floats.
      "float_list_text"   tk.Text widget (label above, multi-line capable).

    ``default`` is used by populate when the cfg value is missing; for
    "float_optional" a missing / None value maps to an empty input instead.
    """
    path:    tuple[str, ...]
    label:   str
    type:    str
    default: object = None
    tip:     str    = ""
    width:   int    = 20

    @property
    def key(self) -> str:
        return ".".join(self.path)


@dataclass(frozen=True)
class Section:
    """A titled group of fields.

    ``description`` renders as a gray paragraph between the title and the
    fields.  ``separator`` toggles the horizontal rule under the title:
    True for the General / Sweep style, False for the Amplifier style
    (bold title only, no rule).
    """
    title:       str
    fields:      tuple[Field, ...]
    columns:     int  = 2     # 1 = stacked, 2 = label/entry pairs side by side
    description: str  = ""
    separator:   bool = True


@dataclass(frozen=True)
class Tab:
    name:     str
    sections: tuple[Section, ...]


# ── TOML serializer ───────────────────────────────────────────────────────────

def _lit(v) -> str:
    if isinstance(v, bool):  return "true" if v else "false"
    if isinstance(v, str):   return f'"{v}"'
    if isinstance(v, float): return f"{int(v):_}" if v == int(v) else f"{v:g}"
    if isinstance(v, int):   return f"{v:_}"
    return str(v)

def _arr(lst) -> str:
    return "[" + ", ".join(_lit(x) for x in lst) + "]"

def build_toml(cfg: dict) -> str:
    L = []
    def ln(s=""): L.append(s)
    def kv(k, v, w=0): L.append(f"{k:<{w}} = {_lit(v)}")
    def kva(k, v, w=0): L.append(f"{k:<{w}} = {_arr(v)}")

    sim = cfg["simulation"]
    ln("[simulation]")
    kv("seed                  ", sim["seed"])
    kv("max_block_size_samples", sim["max_block_size_samples"])
    kv("target_ci_half_width  ", sim["target_ci_half_width"])
    if sim.get("target_ci_relative") is not None:
        kv("target_ci_relative    ", sim["target_ci_relative"])
    kv("confidence            ", sim["confidence"])
    kv("min_errors            ", sim["min_errors"])
    kv("max_iterations        ", sim["max_iterations"])
    ln()

    sw = cfg["sweep"]
    ln("[sweep]")
    kv("sample_rate       ", sw["sample_rate"])
    kva("ibo_db            ", sw.get("ibo_db", []))
    kva("noise_density_dbfs", sw.get("noise_density_dbfs", []))
    ln()

    amp = cfg["amplifier"]
    ln("[amplifier.am_am]")
    kva("input ", amp["am_am"]["input"]);  kva("output", amp["am_am"]["output"]);  ln()
    ln("[amplifier.am_pm]")
    kva("input    ", amp["am_pm"]["input"]);  kva("phase_deg", amp["am_pm"]["phase_deg"]);  ln()

    ln("[ola]")
    kv("filter_span", cfg["ola"]["filter_span"], 12)
    kv("block_size ", cfg["ola"]["block_size"],  12)
    ln()

    o = cfg.get("output", {})
    ln("[output]")
    kv("output_dir", o.get("output_dir", "."), 10)
    kv("plots     ", bool(o.get("plots", True)))
    ln()

    for carr in cfg.get("carrier", []):
        ln("[[carrier]]")
        for k in ("name", "modulation", "symbol_rate", "sps", "rolloff", "filter_span",
                  "power_db", "freq", "enabled", "sweep_demod"):
            if k in carr:
                kv(f"{k:12}", carr[k])
        cod = carr.get("coding")
        if cod:
            ln();  ln("[carrier.coding]")
            for k, v in cod.items():
                kv(f"{k:14}", v)
        ch = carr.get("channel")
        if ch:
            ln();  ln("[carrier.channel]")
            for k, v in ch.items():
                (kva if isinstance(v, list) else kv)(f"{k:22}", v)
        pn = carr.get("phase_noise")
        if pn:
            ln();  ln("[carrier.phase_noise]")
            for k, v in pn.items():
                (kva if isinstance(v, list) else kv)(f"{k:11}", v)
        ln()

    return "\n".join(L)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_float_list(text: str) -> list[float]:
    cleaned = text.strip().strip("[]")
    return [float(x) for x in cleaned.split(",") if x.strip()] if cleaned else []

def _fmt(v) -> str:
    if isinstance(v, float): return str(int(v)) if v == int(v) else f"{v:g}"
    return str(v) if v is not None else ""

class _Tip:
    """Lightweight hover tooltip attached to any widget."""
    def __init__(self, widget: tk.Widget, text: str):
        self._w   = widget
        self._txt = text
        self._win: tk.Toplevel | None = None
        widget.bind("<Enter>",   self._show, add="+")
        widget.bind("<Leave>",   self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _show(self, _=None):
        if self._win or not self._txt:
            return
        x = self._w.winfo_rootx() + self._w.winfo_width() + 6
        y = self._w.winfo_rooty() + 2
        self._win = tw = tk.Toplevel(self._w)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self._txt, justify="left",
                 background="#ffffc0", foreground="#1a1a1a",
                 relief="solid", borderwidth=1,
                 wraplength=380, font=("", 8),
                 padx=5, pady=3).pack()

    def _hide(self, _=None):
        if self._win:
            self._win.destroy()
            self._win = None


def _lf(parent, text, row, col, **kw):
    kw.setdefault("padx", (0, 4))
    kw.setdefault("pady", 2)
    ttk.Label(parent, text=text).grid(row=row, column=col, sticky="w", **kw)

def _ent(parent, var, row, col, width=18, tip="", **kw):
    e = ttk.Entry(parent, textvariable=var, width=width)
    e.grid(row=row, column=col, sticky="w", pady=2, **kw)
    if tip:
        _Tip(e, tip)
    return e

def _scrollable(parent) -> ttk.Frame:
    """Wrap a Frame in a Canvas+Scrollbar; return the inner Frame."""
    canvas = tk.Canvas(parent, highlightthickness=0)
    vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas, padding=12)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    win = canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
    def _wheel(e): canvas.yview_scroll(-1 * (e.delta // 120), "units")
    canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
    canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
    return inner


# ── Schema-driven rendering and (de)serialisation ────────────────────────────

_L_PAD_RIGHT = (16, 4)   # padx on right-column labels (separates the two groups)


def _make_browse_cb(var: tk.StringVar):
    """Standard directory-picker callback for "path"-typed fields."""
    def _cb():
        path = filedialog.askdirectory(initialdir=var.get() or ".")
        if path:
            var.set(path)
    return _cb


def _render_field(parent, fld: Field, row: int,
                  variables: dict[str, tk.Variable],
                  texts: dict[str, tk.Text]) -> int:
    """Create the widget(s) for one Field, register the binding, and return
    the next free row.  Variables/Texts dicts are mutated.

    Used by the columns=1 layout in `_render_section`; the columns=2 layout
    is restricted to single-row scalar fields and inlines its own creation.
    """
    if fld.type == "bool":
        var = tk.BooleanVar(value=bool(fld.default))
        variables[fld.key] = var
        cb = ttk.Checkbutton(parent, text=fld.label, variable=var)
        cb.grid(row=row, column=0, columnspan=4, sticky="w", pady=(6, 0))
        if fld.tip: _Tip(cb, fld.tip)
        return row + 1

    if fld.type == "float_list_text":
        ttk.Label(parent, text=fld.label + ":", foreground="gray").grid(
            row=row, column=0, columnspan=4, sticky="w")
        t = tk.Text(parent, height=1, width=64, wrap="word",
                    font=("Consolas", 9))
        t.grid(row=row + 1, column=0, columnspan=4, sticky="ew", pady=2)
        texts[fld.key] = t
        if fld.tip: _Tip(t, fld.tip)
        return row + 2

    # All remaining types (int/float/float_optional/str/path/float_list)
    # use a StringVar + Entry.  "path" adds a Browse… button to the right.
    var = tk.StringVar()
    variables[fld.key] = var
    if fld.type == "path":
        _lf(parent, fld.label + ":", row, 0)
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=row, column=1, sticky="w")
        _ent(row_frame, var, 0, 0, width=fld.width, tip=fld.tip)
        ttk.Button(row_frame, text="Browse…", width=8,
                   command=_make_browse_cb(var)).grid(row=0, column=1, padx=4)
        return row + 1

    _lf(parent, fld.label + ":", row, 0)
    _ent(parent, var, row, 1, width=fld.width, tip=fld.tip)
    return row + 1


def _render_section(parent, sec: Section, section_row: int,
                    variables: dict[str, tk.Variable],
                    texts: dict[str, tk.Text]) -> int:
    """Render one Section onto `parent` starting at `section_row`; return
    the next free row.  Section title / separator / description are drawn
    by `_render_tab`; this only handles the fields."""
    if sec.columns == 2:
        # Pair scalar fields into (label, entry) (label, entry) on one row.
        # This layout only accepts single-row scalar types; verified at the
        # schema layer by the field type set used in _GENERAL_TAB.
        for i in range(0, len(sec.fields), 2):
            left = sec.fields[i]
            var_l = tk.StringVar()
            variables[left.key] = var_l
            _lf(parent, left.label + ":", section_row, 0)
            _ent(parent, var_l, section_row, 1, width=left.width, tip=left.tip)
            if i + 1 < len(sec.fields):
                right = sec.fields[i + 1]
                var_r = tk.StringVar()
                variables[right.key] = var_r
                _lf(parent, right.label + ":", section_row, 2, padx=_L_PAD_RIGHT)
                _ent(parent, var_r, section_row, 3,
                     width=right.width, tip=right.tip)
            section_row += 1
        return section_row
    # columns=1: dispatch each field to _render_field, which knows how to
    # lay out every supported type (including multi-row text widgets).
    for fld in sec.fields:
        section_row = _render_field(parent, fld, section_row, variables, texts)
    return section_row


def _render_tab(nb, schema: Tab,
                variables: dict[str, tk.Variable],
                texts: dict[str, tk.Text]) -> None:
    """Build a Tab from its schema onto a Notebook.  Each section gets a
    bold title, optional horizontal separator, optional gray description,
    then its fields.  Columns 1 and 3 expand if extra width is available."""
    frame = ttk.Frame(nb); nb.add(frame, text=schema.name)
    inner = _scrollable(frame)
    r = 0
    for sec in schema.sections:
        title_pady = (12, 0) if sec.separator else (10, 2)
        ttk.Label(inner, text=sec.title, font=("", 10, "bold")).grid(
            row=r, column=0, columnspan=4, sticky="w", pady=title_pady)
        r += 1
        if sec.separator:
            ttk.Separator(inner, orient="horizontal").grid(
                row=r, column=0, columnspan=4, sticky="ew", pady=(0, 4))
            r += 1
        if sec.description:
            ttk.Label(inner, text=sec.description, foreground="gray",
                      justify="left").grid(
                row=r, column=0, columnspan=3, sticky="w")
            r += 1
        r = _render_section(inner, sec, r, variables, texts)
    inner.columnconfigure(1, weight=1)
    if any(s.columns == 2 for s in schema.sections):
        inner.columnconfigure(3, weight=1)


def _walk_fields(schema: Tab):
    """Yield every Field in a Tab schema, flattening across sections."""
    for sec in schema.sections:
        yield from sec.fields


def _cfg_get(cfg: dict, path: tuple[str, ...]):
    """Navigate cfg by path; return None if any intermediate key is missing."""
    d = cfg
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _cfg_set(cfg: dict, path: tuple[str, ...], value) -> None:
    """Insert `value` at `path` in `cfg`, creating intermediate dicts."""
    d = cfg
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def _populate_from_schema(schema: Tab, cfg: dict,
                          variables: dict[str, tk.Variable],
                          texts: dict[str, tk.Text]) -> None:
    """Read each field's cfg value and push the formatted form into its
    binding (StringVar / BooleanVar / Text widget) — dispatched by type.

    For ``float_optional``: a missing / None cfg value → empty string in the
    var (which collect interprets as "omit").  For every other type, a
    missing value substitutes the field's ``default``.
    """
    for fld in _walk_fields(schema):
        raw = _cfg_get(cfg, fld.path)

        if fld.type == "float_list_text":
            t = texts[fld.key]
            t.delete("1.0", "end")
            if raw:
                t.insert("1.0", ", ".join(_fmt(x) for x in raw))
            continue
        if fld.type == "bool":
            variables[fld.key].set(bool(raw) if raw is not None
                                    else bool(fld.default))
            continue
        if fld.type == "float_list":
            variables[fld.key].set(
                ", ".join(_fmt(x) for x in (raw or [])))
            continue

        var = variables[fld.key]
        if fld.type == "float_optional":
            var.set(_fmt(raw) if raw is not None else "")
        elif raw is None:
            d = fld.default
            var.set(_fmt(d) if isinstance(d, (int, float)) else str(d))
        elif isinstance(raw, (int, float)):
            var.set(_fmt(raw))
        else:
            var.set(str(raw))


def _collect_from_schema(schema: Tab,
                         variables: dict[str, tk.Variable],
                         texts: dict[str, tk.Text],
                         cfg: dict) -> None:
    """Read each binding, parse per the field's type, and write into `cfg`.

    Empty ``float_optional`` values are omitted from `cfg` so the caller's
    TOML writer doesn't emit a key the user left blank.
    """
    for fld in _walk_fields(schema):
        if fld.type == "float_list_text":
            _cfg_set(cfg, fld.path,
                     _parse_float_list(texts[fld.key].get("1.0", "end")))
            continue
        if fld.type == "bool":
            _cfg_set(cfg, fld.path, bool(variables[fld.key].get()))
            continue

        raw = str(variables[fld.key].get()).strip()
        if fld.type == "float_optional":
            if raw:
                _cfg_set(cfg, fld.path, float(raw))
            continue
        if fld.type == "float_list":
            _cfg_set(cfg, fld.path, _parse_float_list(raw))
            continue
        if fld.type == "int":
            value = int(float(raw))
        elif fld.type == "float":
            value = float(raw)
        else:
            value = raw   # "str" or "path"
        _cfg_set(cfg, fld.path, value)


# ── General-tab schema ───────────────────────────────────────────────────────

_GENERAL_TAB = Tab(
    name="General",
    sections=(
        Section("Simulation", columns=2, fields=(
            Field(("simulation", "seed"), "Seed", "int", default=42,
                  tip="Random seed for reproducible simulations (integer)."),
            Field(("sweep", "sample_rate"), "Sample Rate (MHz)", "float",
                  default=16.0,
                  tip="Composite wideband sample rate in MHz. "
                      "Must be at least 2x the highest carrier edge frequency."),
        )),
        Section("Adaptive BER measurement", columns=2, fields=(
            Field(("simulation", "max_block_size_samples"),
                  "Max Block Size (samples)", "int", default=16_777_216,
                  tip="Per-carrier native-rate buffer cap, in samples, for ONE "
                      "iteration. num_symbols (uncoded) or num_frames (coded) "
                      "are derived from this so the largest per-carrier buffer "
                      "never exceeds it. Increase to reduce iteration count at "
                      "low BER; decrease to fit smaller machines. Memory ≈ "
                      "this × 16 bytes per active demod carrier."),
            Field(("simulation", "max_iterations"), "Max Iterations", "int",
                  default=100,
                  tip="Safety cap on the number of full sim runs per (IBO, noise) "
                      "point. Each iteration processes one Max-Block-Size buffer "
                      "per carrier; iterations that hit this cap without "
                      "converging are flagged in report.md with an asterisk on "
                      "the iteration count."),
            Field(("simulation", "target_ci_half_width"),
                  "Target CI Half-Width", "float", default=2e-3,
                  tip="Absolute half-width on BER at the chosen confidence level. "
                      "Iterations accumulate at each (IBO, noise) point until "
                      "the Wilson interval is at most ±this around the estimate. "
                      "Example: 2e-3 means BER ± 0.002 at 95% confidence."),
            Field(("simulation", "target_ci_relative"),
                  "Target CI Relative", "float_optional",
                  tip="Optional relative half-width on BER, expressed as a "
                      "fraction of BER itself (e.g. 0.01 = ±1% of BER). When set, "
                      "convergence is declared as soon as EITHER the absolute or "
                      "relative target is met. Lets high-BER points exit quickly "
                      "without forcing a tiny absolute interval that would only "
                      "matter at low BER. Leave blank to use the absolute target "
                      "only."),
            Field(("simulation", "min_errors"), "Min Errors", "int", default=50,
                  tip="Minimum cumulative bit errors required before convergence "
                      "can be declared at a sweep point. Prevents premature "
                      "stops when the CI is tight but jittery from too few "
                      "errors. Typical value: 50."),
            Field(("simulation", "confidence"), "Confidence", "float",
                  default=0.95,
                  tip="Two-sided confidence level for the Wilson interval, in "
                      "(0, 1). Typical value: 0.95. Used for both the CI stop "
                      "criterion and the rule-of-three upper bound reported "
                      "when zero errors are observed."),
        )),
        Section("Overlap-Add (OLA) Filter", columns=2, fields=(
            Field(("ola", "filter_span"), "Filter Span", "int", default=16,
                  tip="Half-span of the OLA resampling filter in symbols. "
                      "Longer span = better stopband rejection, higher latency."),
            Field(("ola", "block_size"), "Block Size", "int", default=4096,
                  tip="FFT block size for the overlap-add resampler (samples). "
                      "Must be a power of two; larger = more efficient for "
                      "long filters."),
        )),
    ),
)


_SWEEP_OUTPUT_TAB = Tab(
    name="Sweep & Output",
    sections=(
        Section(
            title="Parameter Sweep",
            description=("The sweep runs at every (IBO, noise) combination. "
                         "Each list must contain at least one value;\n"
                         "a list of one value pins that axis. The first point "
                         "feeds the wideband PSD plot."),
            columns=1,
            fields=(
                Field(("sweep", "ibo_db"), "IBO values (dB)", "float_list",
                      width=44,
                      tip="Comma-separated IBO values to sweep (dB). "
                          "Example: 0.0, 1.5, 3.0, 4.5, 6.0"),
                Field(("sweep", "noise_density_dbfs"), "Noise values (dBFS/Hz)",
                      "float_list", width=44,
                      tip="Comma-separated noise density values to sweep "
                          "(dBFS/Hz). Example: -140.0, -130.0, -120.0"),
            ),
        ),
        Section(
            title="Output",
            columns=1,
            fields=(
                Field(("output", "output_dir"), "Output Directory", "path",
                      default=".", width=28),
                Field(("output", "plots"), "Generate plots", "bool",
                      default=True,
                      tip="When checked, the simulation writes wideband.png, "
                          "amplifier.png, <carrier>_detector.png for each "
                          "detector carrier, and <carrier>_channel.png for "
                          "each carrier with channel impairments. report.md "
                          "is always written."),
            ),
        ),
    ),
)


_AMPLIFIER_TAB = Tab(
    name="Amplifier",
    sections=(
        Section(
            title="AM-AM Table", columns=1, separator=False,
            fields=(
                Field(("amplifier", "am_am", "input"),
                      "Input amplitude (comma-separated)", "float_list_text"),
                Field(("amplifier", "am_am", "output"),
                      "Output (comma-separated)", "float_list_text"),
            ),
        ),
        Section(
            title="AM-PM Table", columns=1, separator=False,
            fields=(
                Field(("amplifier", "am_pm", "input"),
                      "Input amplitude (comma-separated)", "float_list_text"),
                Field(("amplifier", "am_pm", "phase_deg"),
                      "Phase (°) (comma-separated)", "float_list_text"),
            ),
        ),
    ),
)


# ── CarrierFrame ──────────────────────────────────────────────────────────────

class CarrierFrame(ttk.LabelFrame):
    _MAIN = [
        ("name",        "Name",             "str",   "carrier",
         "Unique identifier for this carrier. Used in output reports and seeker results."),
        ("modulation",  "Modulation",       "str",   "BPSK",
         "Modulation scheme: BPSK, DBPSK, MSK, QPSK, OQPSK, 8PSK, 16QAM, 16APSK, 32APSK"),
        ("symbol_rate", "Symbol Rate (MHz)", "float", "1",
         "Symbol rate in MHz (megabaud). Occupied bandwidth ≈ symbol_rate × (1 + rolloff)."),
        ("sps",         "SPS",              "int",   "4",
         "Samples per symbol at the wideband composite sample rate. Integer ≥ 2; typical value: 4."),
        ("rolloff",     "Roll-off",         "float", "0.35",
         "RRC filter roll-off factor α (0 – 1). Higher = wider occupied bandwidth, lower peak ISI."),
        ("filter_span", "Filter Span",      "int",   "8",
         "RRC filter half-span in symbols. Total taps = filter_span × sps + 1."),
        ("power_db",    "Power (dB)",       "float", "0.0",
         "Carrier power in dBFS relative to the wideband composite full-scale."),
        ("freq",        "Freq (MHz)",       "float", "0.0",
         "Carrier centre frequency offset from DC (MHz). Negative = below centre frequency."),
    ]
    _CODING_SCHEMES = ["convolutional", "concatenated", "turbo", "ldpc"]
    _CH = [
        ("ripple_db",         "Ripple (dB)",      "float", "0.5",
         "Peak-to-peak amplitude ripple across the carrier bandwidth (dB)."),
        ("ripple_cycles",     "Ripple Cycles",    "float", "2.0",
         "Number of full ripple cycles across the carrier bandwidth."),
        ("max_phase_dev_deg", "Max Phase (°)",    "float", "5.0",
         "Maximum deviation from linear phase across the carrier bandwidth (degrees)."),
        ("phase_poly_order",  "Phase Poly Order", "int",   "2",
         "Order of the polynomial used to model the phase-vs-frequency distortion."),
    ]

    def __init__(self, parent, on_remove, data: dict, **kw):
        super().__init__(parent, text=data.get("name", "carrier"), padding=6, **kw)
        self._on_remove = on_remove
        self._vars:        dict[str, tk.Variable] = {}
        self._ch_vars:     dict[str, tk.Variable] = {}
        self._coding_vars: dict[str, tk.Variable] = {}
        self._pn_texts:    dict[str, tk.Text]     = {}
        # Caches hold the last-known values for each optional sub-section so
        # toggling its checkbox off and back on restores them instead of
        # collapsing back to defaults.  Seeded from the initial data.
        self._coding_cache: dict = dict(data.get("coding") or {})
        self._ch_cache:     dict = dict(data.get("channel") or {})
        self._pn_cache:     dict = dict(data.get("phase_noise") or {})
        self._enabled     = tk.BooleanVar(value=data.get("enabled", True))
        self._sweep_demod = tk.BooleanVar(value=data.get("sweep_demod", False))
        self._has_coding  = tk.BooleanVar(value=bool(data.get("coding")))
        ch = data.get("channel", {})
        self._has_ch = tk.BooleanVar(value=bool(ch) and ch.get("enabled", True))
        pn = data.get("phase_noise", {})
        self._has_pn = tk.BooleanVar(value=bool(pn) and pn.get("enabled", True))
        self._build(data)

    @property
    def carrier_name(self) -> str:
        return self._vars.get("name", tk.StringVar()).get() or "carrier"

    def _build(self, d: dict):
        ttk.Button(self, text="Remove", command=self._on_remove,
                   width=8).grid(row=0, column=3, sticky="ne", padx=2)

        # Main parameter fields (2-column grid)
        _MODS = ["BPSK", "DBPSK", "MSK", "QPSK", "OQPSK", "8PSK", "16QAM", "16APSK", "32APSK"]
        for i, (key, label, _, dflt, tip) in enumerate(self._MAIN):
            raw = d.get(key, dflt)
            var = tk.StringVar(value=_fmt(raw) if isinstance(raw, (int, float)) else str(raw))
            self._vars[key] = var
            r, c = (i // 2) + 1, (i % 2) * 2
            _lf(self, label + ":", r, c)
            if key == "modulation":
                cb = ttk.Combobox(self, textvariable=var, values=_MODS,
                                  state="readonly", width=12)
                cb.grid(row=r, column=c + 1, sticky="w", pady=2)
                if tip:
                    _Tip(cb, tip)
            else:
                _ent(self, var, r, c + 1, width=14, tip=tip)
            if key == "name":
                var.trace_add("write",
                              lambda *_, v=var: self.configure(text=v.get() or "carrier"))

        n_main_rows = (len(self._MAIN) + 1) // 2  # ceil(9/2) = 5
        check_row   = n_main_rows + 1              # row 6
        cod_row     = check_row + 1                # row 7
        ch_row      = cod_row + 2                  # row 9
        pn_row      = ch_row + 2                   # row 11

        # ── Enable checkboxes ────────────────────────────────────────────────
        ttk.Checkbutton(self, text="Include in wideband",
                        variable=self._enabled).grid(
            row=check_row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(self, text="Enable detector model",
                        variable=self._sweep_demod).grid(
            row=check_row, column=2, columnspan=2, sticky="w", pady=(8, 0))

        # ── FEC coding ────────────────────────────────────────────────────────
        ttk.Checkbutton(self, text="FEC coding", variable=self._has_coding,
                        command=self._toggle_coding).grid(
            row=cod_row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._coding_frame = ttk.LabelFrame(self, text="FEC Parameters", padding=4)
        self._coding_frame.grid(row=cod_row + 1, column=0, columnspan=4,
                                 sticky="ew", padx=(14, 0), pady=(2, 0))
        if d.get("coding"):
            self._populate_coding(d["coding"])
        else:
            self._coding_frame.grid_remove()

        # ── Channel impairments ──────────────────────────────────────────────
        ttk.Checkbutton(self, text="Channel impairments", variable=self._has_ch,
                        command=self._toggle_ch).grid(
            row=ch_row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._ch_frame = ttk.Frame(self, padding=(14, 0, 0, 0))
        self._ch_frame.grid(row=ch_row + 1, column=0, columnspan=4, sticky="ew")
        if self._has_ch.get():
            self._populate_ch(self._ch_cache)
        else:
            self._ch_frame.grid_remove()

        # ── Phase noise ──────────────────────────────────────────────────────
        ttk.Checkbutton(self, text="Phase noise", variable=self._has_pn,
                        command=self._toggle_pn).grid(
            row=pn_row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._pn_frame = ttk.Frame(self, padding=(14, 0, 0, 0))
        self._pn_frame.grid(row=pn_row + 1, column=0, columnspan=4, sticky="ew")
        if self._has_pn.get():
            self._populate_pn(self._pn_cache)
        else:
            self._pn_frame.grid_remove()

    def _toggle_coding(self):
        if self._has_coding.get():
            self._populate_coding(self._coding_cache)
            self._coding_frame.grid()
        else:
            self._coding_cache = self._snapshot_coding()
            for w in self._coding_frame.winfo_children(): w.destroy()
            self._coding_vars.clear()
            self._coding_frame.grid_remove()

    def _snapshot_coding(self) -> dict:
        """Read current FEC widget values into a dict for cache."""
        out: dict = {}
        scheme = self._coding_vars.get("scheme", tk.StringVar()).get().strip()
        if scheme:
            out["scheme"] = scheme
        raw = self._coding_vars.get("block_length", tk.StringVar()).get().strip()
        if raw:
            try:               out["block_length"] = int(float(raw))
            except ValueError: pass
        matrix = self._coding_vars.get("matrix", tk.StringVar()).get().strip()
        if matrix:
            out["matrix"] = matrix
        return out

    def _populate_coding(self, cod: dict):
        for w in self._coding_frame.winfo_children(): w.destroy()
        self._coding_vars.clear()
        _lf(self._coding_frame, "Scheme:", 0, 0)
        scheme_var = tk.StringVar(value=cod.get("scheme", "convolutional"))
        self._coding_vars["scheme"] = scheme_var
        cb = ttk.Combobox(self._coding_frame, textvariable=scheme_var,
                          values=self._CODING_SCHEMES, state="readonly", width=14)
        cb.grid(row=0, column=1, sticky="w", pady=2)
        _Tip(cb, "FEC scheme: convolutional, concatenated, turbo, or ldpc.")
        _lf(self._coding_frame, "Block Length:", 0, 2)
        bl_var = tk.StringVar(value=_fmt(cod.get("block_length", 1024)))
        self._coding_vars["block_length"] = bl_var
        _ent(self._coding_frame, bl_var, 0, 3, width=10,
             tip="Data bits per frame (convolutional and turbo). Ignored for concatenated/ldpc.")

        # LDPC matrix row: only shown when scheme == "ldpc".
        self._ldpc_label = ttk.Label(self._coding_frame, text="LDPC Matrix:")
        self._ldpc_label.grid(row=1, column=0, sticky="w", padx=(0, 4), pady=2)
        lm_var = tk.StringVar(value=cod.get("matrix", ""))
        self._coding_vars["matrix"] = lm_var
        self._ldpc_entry = _ent(self._coding_frame, lm_var, 1, 1, width=32,
            tip="Path to .alist file for LDPC code. "
                "Leave blank to use the bundled default (data/ldpc/mackay_13298.alist).")
        cb.bind("<<ComboboxSelected>>",
                lambda _e: self._update_ldpc_visibility(), add="+")
        self._update_ldpc_visibility()

    def _update_ldpc_visibility(self):
        scheme = self._coding_vars.get("scheme", tk.StringVar()).get()
        if scheme == "ldpc":
            self._ldpc_label.grid()
            self._ldpc_entry.grid()
        else:
            self._ldpc_label.grid_remove()
            self._ldpc_entry.grid_remove()

    def _toggle_ch(self):
        if self._has_ch.get():
            self._populate_ch(self._ch_cache)
            self._ch_frame.grid()
        else:
            self._ch_cache = self._snapshot_ch()
            for w in self._ch_frame.winfo_children(): w.destroy()
            self._ch_vars.clear()
            self._ch_frame.grid_remove()

    def _snapshot_ch(self) -> dict:
        """Read current channel-impairment widget values into a dict."""
        out: dict = {}
        for key, _, typ, _, _ in self._CH:
            if key not in self._ch_vars: continue
            raw = self._ch_vars[key].get().strip()
            if not raw: continue
            try:
                out[key] = (int(float(raw)) if typ == "int"
                            else float(raw) if typ == "float" else raw)
            except ValueError:
                out[key] = raw       # keep bad value verbatim for re-display
        return out

    def _populate_ch(self, ch: dict):
        for w in self._ch_frame.winfo_children(): w.destroy()
        self._ch_vars.clear()
        for i, (key, label, _, dflt, tip) in enumerate(self._CH):
            raw = ch.get(key, dflt)
            var = tk.StringVar(value=_fmt(raw) if isinstance(raw, (int, float)) else str(raw))
            self._ch_vars[key] = var
            r, c = (i // 2) + 1, (i % 2) * 2
            _lf(self._ch_frame, label + ":", r, c)
            _ent(self._ch_frame, var, r, c + 1, width=14, tip=tip)

    def _toggle_pn(self):
        if self._has_pn.get():
            self._populate_pn(self._pn_cache)
            self._pn_frame.grid()
        else:
            self._pn_cache = self._snapshot_pn()
            for w in self._pn_frame.winfo_children(): w.destroy()
            self._pn_texts.clear()
            self._pn_frame.grid_remove()

    def _snapshot_pn(self) -> dict:
        """Read current phase-noise mask values into a dict."""
        out: dict = {}
        if "offset_hz" in self._pn_texts:
            offs = _parse_float_list(self._pn_texts["offset_hz"].get("1.0", "end"))
            if offs: out["offset_hz"] = offs
        if "dbc_per_hz" in self._pn_texts:
            dbcs = _parse_float_list(self._pn_texts["dbc_per_hz"].get("1.0", "end"))
            if dbcs: out["dbc_per_hz"] = dbcs
        return out

    def _populate_pn(self, pn: dict):
        """Two text widgets — offset_hz and dbc_per_hz — for the per-carrier
        oscillator mask.  Same units and conventions as the previous
        global section."""
        for w in self._pn_frame.winfo_children(): w.destroy()
        self._pn_texts.clear()
        for r, (key, label, default) in enumerate((
            ("offset_hz",  "Offset (Hz, comma-separated):",     pn.get("offset_hz", [])),
            ("dbc_per_hz", "L(f) (dBc/Hz, comma-separated):",   pn.get("dbc_per_hz", [])),
        )):
            ttk.Label(self._pn_frame, text=label, foreground="gray").grid(
                row=2 * r, column=0, columnspan=4, sticky="w")
            t = tk.Text(self._pn_frame, height=1, width=64, wrap="word",
                         font=("Consolas", 9))
            t.grid(row=2 * r + 1, column=0, columnspan=4, sticky="ew", pady=2)
            if default:
                t.insert("1.0", ", ".join(_fmt(x) for x in default))
            self._pn_texts[key] = t

    def to_dict(self) -> dict:
        d = {}
        for key, _, typ, _, _ in self._MAIN:
            raw = self._vars[key].get().strip()
            d[key] = (int(float(raw)) if typ == "int"
                      else float(raw) if typ == "float" else raw)
        d["enabled"]     = bool(self._enabled.get())
        d["sweep_demod"] = bool(self._sweep_demod.get())

        if self._has_coding.get() and self._coding_vars:
            cod: dict = {}
            scheme = self._coding_vars.get("scheme", tk.StringVar()).get().strip()
            if scheme:
                cod["scheme"] = scheme
            raw = self._coding_vars.get("block_length", tk.StringVar()).get().strip()
            try:
                cod["block_length"] = int(float(raw))
            except ValueError:
                pass
            matrix = self._coding_vars.get("matrix", tk.StringVar()).get().strip()
            if matrix:
                cod["matrix"] = matrix
            if cod:
                d["coding"] = cod

        if self._has_ch.get() and self._ch_vars:
            ch: dict = {}
            for key, _, typ, _, _ in self._CH:
                if key not in self._ch_vars: continue
                raw = self._ch_vars[key].get().strip()
                ch[key] = (int(float(raw)) if typ == "int"
                           else float(raw) if typ == "float" else raw)
            d["channel"] = ch

        if self._has_pn.get() and self._pn_texts:
            offs = _parse_float_list(self._pn_texts["offset_hz"].get("1.0", "end"))
            dbcs = _parse_float_list(self._pn_texts["dbc_per_hz"].get("1.0", "end"))
            if offs or dbcs:
                d["phase_noise"] = {
                    "enabled":    True,
                    "offset_hz":  offs,
                    "dbc_per_hz": dbcs,
                }
        return d


# ── Main application ──────────────────────────────────────────────────────────

_PCT_RE   = re.compile(r'^\[\s*(\d+)%\]')
# Matches the per-iteration in-place status lines emitted from sim.sweep:
#   "  iter  3/100: chunk 5/12"          (during chunk processing)
#   "  iter  3/100 done: c1   bits=..."  (cumulative tally after each iter)
# Both forms overwrite the previous status line in the GUI log and in
# simulation.log so a sweep point shows as a single rolling line.
_CHUNK_RE = re.compile(r'^\s+iter\s+\d+/\d+')


class App:
    def __init__(self, root: tk.Tk, path: Path):
        self.root      = root
        self.path      = path
        self._carriers: list[CarrierFrame] = []
        self._vars:     dict[str, tk.Variable] = {}
        self._texts:    dict[str, tk.Text] = {}
        self._proc     = None
        self._log_file = None
        self._running  = False
        self._last_progress_time: float = 0.0
        self._last_progress_line: str = ""
        self._slow_warned:        bool = False
        self._last_line_was_chunk: bool = False
        self._log_file_chunk_pos: int | None = None
        root.title("SO-WAT")
        root.minsize(760, 580)
        _icon = tk.PhotoImage(data=_ICON_B64)
        root.wm_iconphoto(True, _icon)
        self._icon_ref = _icon   # prevent GC
        self._build_ui()
        self._load(path)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header band ──────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, background="#0a0e1c", height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        icon_display = tk.PhotoImage(data=_ICON_B64).subsample(2)  # 64×64
        self._hdr_icon = icon_display  # prevent GC
        tk.Label(hdr, image=icon_display,
                 background="#0a0e1c").place(relx=1.0, rely=0.5,
                                             anchor="e", x=-10)
        tk.Label(hdr, text="SO-WAT",
                 font=("Consolas", 20, "bold"),
                 foreground="#00dcc3",
                 background="#0a0e1c").place(relx=0.5, rely=0.32, anchor="center")
        tk.Label(hdr, text="Simulation Orchestrator  ·  Waveform Analysis Tool",
                 font=("Consolas", 8),
                 foreground="#3d5a6e",
                 background="#0a0e1c").place(relx=0.5, rely=0.72, anchor="center")

        # ── Toolbar ──────────────────────────────────────────────────────────
        tb = ttk.Frame(self.root, padding=(8, 6))
        tb.pack(fill="x")
        self._path_var = tk.StringVar()
        ttk.Label(tb, text="File:").pack(side="left")
        ttk.Entry(tb, textvariable=self._path_var, width=36,
                  state="readonly").pack(side="left", padx=4)
        ttk.Button(tb, text="Open…",    command=self._open_file).pack(side="left", padx=2)
        ttk.Button(tb, text="Save",     command=self._save).pack(side="left", padx=2)
        ttk.Button(tb, text="Save As…", command=self._save_as).pack(side="left", padx=2)
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=10, pady=2)
        self._run_btn = ttk.Button(tb, text="▶  Launch Simulation", command=self._launch)
        self._run_btn.pack(side="left", padx=2)
        self._stop_btn = ttk.Button(tb, text="■  Stop", command=self._stop, state="disabled")
        self._stop_btn.pack(side="left", padx=2)

        ttk.Separator(self.root, orient="horizontal").pack(fill="x")

        # ── Bottom area (packed before notebook so it claims bottom space) ───
        self._status = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self._status, anchor="w",
                  foreground="gray").pack(side="bottom", fill="x", padx=8, pady=(0, 4))

        ttk.Separator(self.root, orient="horizontal").pack(side="bottom", fill="x")

        prog_frame = ttk.Frame(self.root, padding=(6, 4))
        prog_frame.pack(side="bottom", fill="x")

        self._progress = ttk.Progressbar(prog_frame, orient="horizontal",
                                          mode="determinate", maximum=100)
        self._progress.pack(fill="x", pady=(0, 3))

        log_outer = ttk.Frame(prog_frame)
        log_outer.pack(fill="x")
        log_vsb = ttk.Scrollbar(log_outer, orient="vertical")
        self._log_text = tk.Text(
            log_outer, height=12, wrap="word", font=("Consolas", 8),
            state="disabled", background="#1e1e1e", foreground="#d4d4d4",
            yscrollcommand=log_vsb.set,
        )
        log_vsb.config(command=self._log_text.yview)
        log_vsb.pack(side="right", fill="y")
        self._log_text.pack(side="left", fill="x", expand=True)

        ttk.Separator(self.root, orient="horizontal").pack(side="bottom", fill="x")

        # ── Notebook ─────────────────────────────────────────────────────────
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=6, pady=4)
        self._build_general_tab(nb)
        self._build_amplifier_tab(nb)
        self._build_sweep_output_tab(nb)
        self._build_carriers_tab(nb)

    def _sv(self, key, default="") -> tk.StringVar:
        v = tk.StringVar(value=str(default))
        self._vars[key] = v
        return v

    def _text_widget(self, parent, key, row, height=1) -> tk.Text:
        t = tk.Text(parent, height=height, width=64, wrap="word",
                    font=("Consolas", 9))
        t.grid(row=row, column=0, columnspan=4, sticky="ew", pady=2)
        self._texts[key] = t
        return t

    def _section(self, parent, title, row) -> int:
        ttk.Label(parent, text=title, font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=4, sticky="w", pady=(12, 0))
        ttk.Separator(parent, orient="horizontal").grid(
            row=row + 1, column=0, columnspan=4, sticky="ew", pady=(0, 4))
        return row + 2

    def _build_general_tab(self, nb):
        _render_tab(nb, _GENERAL_TAB, self._vars, self._texts)

    def _build_amplifier_tab(self, nb):
        _render_tab(nb, _AMPLIFIER_TAB, self._vars, self._texts)

    def _build_sweep_output_tab(self, nb):
        _render_tab(nb, _SWEEP_OUTPUT_TAB, self._vars, self._texts)

    def _build_carriers_tab(self, nb):
        tab = ttk.Frame(nb);  nb.add(tab, text="Carriers")
        tb = ttk.Frame(tab, padding=(8, 4))
        tb.pack(fill="x")
        ttk.Button(tb, text="+ Add Carrier", command=self._add_carrier).pack(side="left")

        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=8, pady=2)
        ttk.Label(tb, text="View:").pack(side="left")
        self._focus_var = tk.StringVar(value="All")
        self._focus_combo = ttk.Combobox(tb, textvariable=self._focus_var,
                                          values=["All"], width=18, state="readonly")
        self._focus_combo.pack(side="left", padx=4)
        self._focus_combo.bind("<<ComboboxSelected>>", self._apply_focus)

        self._carr_inner = _scrollable(tab)

    # ── Carrier management ────────────────────────────────────────────────────

    def _add_carrier(self, data: dict | None = None):
        ref: list[CarrierFrame | None] = [None]

        def remove():
            assert ref[0] is not None
            ref[0].destroy()
            self._carriers.remove(ref[0])
            self._refresh_focus_options()

        cf = CarrierFrame(self._carr_inner, on_remove=remove, data=data or {})
        cf.pack(fill="x", pady=4, padx=2)
        ref[0] = cf
        self._carriers.append(cf)
        self._refresh_focus_options()

    def _refresh_focus_options(self):
        names = ["All"] + [cf.carrier_name for cf in self._carriers]
        self._focus_combo["values"] = names
        if self._focus_var.get() not in names:
            self._focus_var.set("All")
        self._apply_focus()

    def _apply_focus(self, *_):
        sel = self._focus_var.get()
        for cf in self._carriers:
            if sel == "All" or cf.carrier_name == sel:
                cf.pack(fill="x", pady=4, padx=2)
            else:
                cf.pack_forget()

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _load(self, path: Path):
        if not path.exists():
            self._status.set(f"File not found: {path} — using defaults.")
            return
        try:
            with open(path, "rb") as f:
                cfg = tomllib.load(f)
        except Exception as e:  # pylint: disable=broad-exception-caught
            messagebox.showerror("Load error", str(e));  return
        self.path = path
        self._path_var.set(str(path))
        self._populate(cfg)
        self._status.set(f"Loaded: {path}")

    def _populate(self, cfg: dict):
        # General / Sweep & Output / Amplifier tabs are schema-driven;
        # carriers are still hand-coded.
        for schema in (_GENERAL_TAB, _SWEEP_OUTPUT_TAB, _AMPLIFIER_TAB):
            _populate_from_schema(schema, cfg, self._vars, self._texts)

        for cf in self._carriers: cf.destroy()
        self._carriers.clear()
        for carr in cfg.get("carrier", []):
            self._add_carrier(carr)

    def _collect(self) -> dict:
        cfg: dict = {}
        for schema in (_GENERAL_TAB, _SWEEP_OUTPUT_TAB, _AMPLIFIER_TAB):
            _collect_from_schema(schema, self._vars, self._texts, cfg)
        # Mirror the legacy behavior: empty output_dir → "."
        if not cfg.get("output", {}).get("output_dir"):
            cfg.setdefault("output", {})["output_dir"] = "."
        cfg["carrier"] = [cf.to_dict() for cf in self._carriers]
        return cfg

    def _save(self):
        try:
            cfg = self._collect()
            self.path.write_text(build_toml(cfg), encoding="utf-8")
            self._status.set(f"Saved: {self.path}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            messagebox.showerror("Save error", str(e))

    def _save_as(self):
        p = filedialog.asksaveasfilename(
            initialfile=self.path.name,
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")])
        if p:
            self.path = Path(p)
            self._path_var.set(str(self.path))
            self._save()

    def _open_file(self):
        p = filedialog.askopenfilename(
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")])
        if p:
            self._load(Path(p))

    # ── Subprocess monitoring ─────────────────────────────────────────────────

    def _set_running(self, running: bool):
        self._running = running
        self._run_btn.configure(state="disabled" if running else "normal")
        self._stop_btn.configure(state="normal" if running else "disabled")

    def _log_clear(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")
        self._last_line_was_chunk = False

    def _log_append(self, msg: str):
        is_chunk = bool(_CHUNK_RE.match(msg))
        self._log_text.configure(state="normal")
        if self._last_line_was_chunk:
            # end-1c is the phantom newline; end-2c is the \n ending the real last line
            self._log_text.delete("end-2c linestart", "end-2c")
            self._log_text.insert("end-2c linestart", msg)
        else:
            self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")
        self._last_line_was_chunk = is_chunk

    def _stop(self):
        if self._proc and self._running:
            self._proc.terminate()
            self._log_append("[GUI] Simulation stopped by user.")
            self._status.set("Stopped.")
            self._set_running(False)

    def _launch(self):
        if self._running:
            return
        try:
            self._save()
        except Exception as e:  # pylint: disable=broad-exception-caught
            messagebox.showerror("Save error", str(e))
            return

        self._log_clear()
        self._progress["value"] = 0
        self._last_progress_time = time.monotonic()
        self._last_progress_line = ""
        self._slow_warned = False
        self._set_running(True)
        self._status.set("Running simulation...")

        out_dir = Path(self._vars["output.output_dir"].get().strip() or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._log_file_chunk_pos = None
        try:
            self._log_file = open(out_dir / "simulation.log", "wb")
        except OSError:
            self._log_file = None

        main_py = Path(__file__).parent / "main.py"
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", str(main_py), str(self.path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(__file__).parent),
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            messagebox.showerror("Launch error", str(e))
            self._set_running(False)
            return

        self._queue = queue.Queue()
        threading.Thread(target=self._read_output, daemon=True).start()
        self.root.after(100, self._poll_proc)

    def _log_file_write(self, line: str):
        if self._log_file is None:
            return
        is_chunk = bool(_CHUNK_RE.match(line))
        if self._log_file_chunk_pos is not None:
            self._log_file.seek(self._log_file_chunk_pos)
            self._log_file.truncate()
        if is_chunk:
            self._log_file_chunk_pos = self._log_file.tell()
        else:
            self._log_file_chunk_pos = None
        self._log_file.write((line + "\n").encode("utf-8"))
        self._log_file.flush()

    def _read_output(self):
        if self._proc is None or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            stripped = line.rstrip()
            self._queue.put(stripped)
            self._log_file_write(stripped)
        self._proc.wait()
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            self._log_file_chunk_pos = None
        self._queue.put(None)

    def _poll_proc(self):
        try:
            while True:
                line = self._queue.get_nowait()
                if line is None:
                    self._on_run_complete()
                    return
                self._log_append(line)
                m = _PCT_RE.match(line)
                if m:
                    self._progress["value"] = int(m.group(1))
                    self._last_progress_time = time.monotonic()
                    self._last_progress_line = line
                    self._slow_warned = False
        except queue.Empty:
            pass
        self.root.after(100, self._poll_proc)

    def _on_run_complete(self):
        rc = self._proc.returncode if self._proc else -1
        if rc == 0:
            self._progress["value"] = 100
            self._status.set("Simulation complete.")
        else:
            self._status.set(f"Simulation exited with code {rc}.")
        self._set_running(False)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    toml_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("simulation.toml")
    tk_root = tk.Tk()
    try:
        tk_root.tk.call("tk", "scaling", 1.25)
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    App(tk_root, toml_path)
    tk_root.mainloop()
