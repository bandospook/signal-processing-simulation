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
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import tomllib

from tkconfig import (
    Field, Section, Tab,
    render_tab, render_section,
    populate_from_schema, collect_from_schema, walk_fields,
    cfg_get, emit_table, scrollable, parse_float_list,
)
from misc.gen_icon import build_icon as _build_icon

_ICON_B64 = base64.b64encode(_build_icon()).decode()


# ── TOML serializer ─────────────────────────────────────
#
# build_toml stays here because the table order and the [[carrier]] sub-table
# layout are specific to this project; the generic emission primitives
# (emit_table / lit / arr) live in tkconfig.toml_writer.

# Top-level table emission order: (TOML header, cfg path).  Tables absent
# from cfg are skipped.
_TOML_TABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("simulation",      ("simulation",)),
    ("sweep",           ("sweep",)),
    ("amplifier.am_am", ("amplifier", "am_am")),
    ("amplifier.am_pm", ("amplifier", "am_pm")),
    ("ola",             ("ola",)),
    ("output",          ("output",)),
)

# Carrier scalar keys, in emission order, followed by the optional sub-tables.
_CARRIER_MAIN_KEYS: tuple[str, ...] = (
    "name", "modulation", "symbol_rate", "sps", "rolloff",
    "filter_span", "power_db", "freq", "enabled", "sweep_demod",
)
_CARRIER_SUB_TABLES: tuple[str, ...] = ("coding", "channel", "phase_noise")


def build_toml(cfg: dict) -> str:
    """Serialise a config dict to TOML text with aligned, sectioned output.

    Emits the fixed top-level tables (in `_TOML_TABLES` order) followed by
    one `[[carrier]]` block per carrier, each with its optional
    `[carrier.coding]` / `[carrier.channel]` / `[carrier.phase_noise]`
    sub-tables.  Generic key alignment is delegated to tkconfig.emit_table.
    """
    L: list[str] = []
    for header, path in _TOML_TABLES:
        section = cfg_get(cfg, path)
        if isinstance(section, dict):
            emit_table(L, f"[{header}]", section)
            L.append("")

    for carr in cfg.get("carrier", []):
        emit_table(L, "[[carrier]]", carr, _CARRIER_MAIN_KEYS)
        for sub in _CARRIER_SUB_TABLES:
            sub_d = carr.get(sub)
            if isinstance(sub_d, dict) and sub_d:
                L.append("")
                emit_table(L, f"[carrier.{sub}]", sub_d)
        L.append("")

    return "\n".join(L)


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


# ── Carrier sub-schemas ──────────────────────────────────────────────────────
#
# CarrierFrame is one tk.LabelFrame per [[carrier]] block in the TOML.  Its
# main scalar fields and the optional Channel / Phase-noise sub-sections all
# flow through the schema helpers; only the FEC sub-section is still
# hand-coded (the LDPC-matrix field's dynamic visibility doesn't fit the
# schema's flat Field model yet).
#
# Field paths are single-element because they live inside the carrier dict
# (e.g. the carrier name is at ``carr_dict["name"]``, not at any deeper path).

_MODULATIONS: tuple[str, ...] = (
    "BPSK", "DBPSK", "MSK", "QPSK", "OQPSK",
    "8PSK", "16QAM", "16APSK", "32APSK",
)

_CARRIER_MAIN_FIELDS = Section(
    title="(carrier main fields — title is set per-instance)",
    columns=2, separator=False, right_col_padx=(0, 4),
    fields=(
        Field(("name",), "Name", "str", default="carrier", width=14,
              tip="Unique identifier for this carrier. "
                  "Used in output reports and seeker results."),
        Field(("modulation",), "Modulation", "str_enum", default="BPSK", width=12,
              options=_MODULATIONS,
              tip="Modulation scheme: BPSK, DBPSK, MSK, QPSK, OQPSK, "
                  "8PSK, 16QAM, 16APSK, 32APSK"),
        Field(("symbol_rate",), "Symbol Rate (MHz)", "float", default=1.0, width=14,
              tip="Symbol rate in MHz (megabaud). "
                  "Occupied bandwidth ≈ symbol_rate × (1 + rolloff)."),
        Field(("sps",), "SPS", "int", default=4, width=14,
              tip="Samples per symbol at the wideband composite sample rate. "
                  "Integer ≥ 2; typical value: 4."),
        Field(("rolloff",), "Roll-off", "float", default=0.35, width=14,
              tip="RRC filter roll-off factor α (0 – 1). "
                  "Higher = wider occupied bandwidth, lower peak ISI."),
        Field(("filter_span",), "Filter Span", "int", default=8, width=14,
              tip="RRC filter half-span in symbols. "
                  "Total taps = filter_span × sps + 1."),
        Field(("power_db",), "Power (dB)", "float", default=0.0, width=14,
              tip="Carrier power in dBFS relative to the wideband composite "
                  "full-scale."),
        Field(("freq",), "Freq (MHz)", "float", default=0.0, width=14,
              tip="Carrier centre frequency offset from DC (MHz). "
                  "Negative = below centre frequency."),
    ),
)

_CARRIER_CHANNEL_FIELDS = Section(
    title="(channel impairments)",
    columns=2, separator=False, right_col_padx=(0, 4),
    fields=(
        Field(("ripple_db",), "Ripple (dB)", "float", default=0.5, width=14,
              tip="Peak-to-peak amplitude ripple across the carrier "
                  "bandwidth (dB)."),
        Field(("ripple_cycles",), "Ripple Cycles", "float", default=2.0, width=14,
              tip="Number of full ripple cycles across the carrier bandwidth."),
        Field(("max_phase_dev_deg",), "Max Phase (°)", "float",
              default=5.0, width=14,
              tip="Maximum deviation from linear phase across the carrier "
                  "bandwidth (degrees)."),
        Field(("phase_poly_order",), "Phase Poly Order", "int",
              default=2, width=14,
              tip="Order of the polynomial used to model the phase-vs-"
                  "frequency distortion."),
    ),
)

_CARRIER_PHASE_NOISE_FIELDS = Section(
    title="(phase noise mask)",
    columns=1, separator=False,
    fields=(
        Field(("offset_hz",), "Offset (Hz, comma-separated)",
              "float_list_text"),
        Field(("dbc_per_hz",), "L(f) (dBc/Hz, comma-separated)",
              "float_list_text"),
    ),
)

_CODING_SCHEMES: tuple[str, ...] = ("convolutional", "concatenated", "turbo", "ldpc")

_CARRIER_CODING_FIELDS = Section(
    title="(FEC parameters)",
    columns=2, separator=False, right_col_padx=(0, 4),
    fields=(
        Field(("scheme",), "Scheme", "str_enum", default="convolutional",
              width=14, options=_CODING_SCHEMES,
              tip="FEC scheme: convolutional, concatenated, turbo, or ldpc."),
        Field(("block_length",), "Block Length", "int", default=1024, width=10,
              tip="Data bits per frame (convolutional and turbo). "
                  "Ignored for concatenated/ldpc."),
        Field(("matrix",), "LDPC Matrix", "str", default="", width=32,
              visible_when=(("scheme",), ("ldpc",)),
              tip="Path to .alist file for LDPC code.  Leave blank to use "
                  "the bundled default (data/ldpc/mackay_13298.alist)."),
    ),
)


# ── CarrierFrame ──────────────────────────────────────────────────────────────

class CarrierFrame(ttk.LabelFrame):
    """One [[carrier]] block, rendered as a tk.LabelFrame.

    Main scalar fields and the optional Channel / Phase-noise sub-sections
    are driven by the schemas at module scope; FEC coding is still
    hand-coded because the LDPC-matrix field's dynamic visibility doesn't
    fit the schema's flat Field model.
    """

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

        # Main parameter fields — schema-driven 2-column grid starting at row 1.
        # No float_list_text fields here, so the texts dict slot stays empty.
        next_row = render_section(self, _CARRIER_MAIN_FIELDS, 1, self._vars, {})
        populate_from_schema(_CARRIER_MAIN_FIELDS, d, self._vars, {})
        # Carrier-specific behavior: name field updates the LabelFrame title.
        name_var = self._vars["name"]
        name_var.trace_add("write",
                            lambda *_, v=name_var: self.configure(
                                text=str(v.get()) or "carrier"))

        check_row = next_row + 1                   # blank row above checkboxes
        cod_row   = check_row + 1
        ch_row    = cod_row + 2
        pn_row    = ch_row + 2

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
        """Read current FEC widget values into a dict for cache.

        Lenient: empty or unparseable fields are skipped so toggling
        off mid-edit can never crash.
        """
        out: dict = {}
        for fld in walk_fields(_CARRIER_CODING_FIELDS):
            if fld.key not in self._coding_vars: continue
            raw = str(self._coding_vars[fld.key].get()).strip()
            if not raw: continue
            try:
                out[fld.path[0]] = (int(float(raw)) if fld.type == "int"
                                     else float(raw) if fld.type == "float"
                                     else raw)
            except ValueError:
                out[fld.path[0]] = raw   # keep bad value verbatim for re-display
        return out

    def _populate_coding(self, cod: dict):
        for w in self._coding_frame.winfo_children(): w.destroy()
        self._coding_vars.clear()
        # The LDPC-matrix field's visible_when=(("scheme",), ("ldpc",))
        # is wired automatically by render_section.
        render_section(self._coding_frame, _CARRIER_CODING_FIELDS, 0,
                         self._coding_vars, {})
        populate_from_schema(_CARRIER_CODING_FIELDS, cod,
                              self._coding_vars, {})

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
        """Read current channel-impairment widget values into a dict.

        Lenient: empty or unparseable fields are skipped so toggling the
        section off mid-edit can never crash.
        """
        out: dict = {}
        for fld in walk_fields(_CARRIER_CHANNEL_FIELDS):
            if fld.key not in self._ch_vars: continue
            raw = str(self._ch_vars[fld.key].get()).strip()
            if not raw: continue
            try:
                out[fld.path[0]] = (int(float(raw)) if fld.type == "int"
                                     else float(raw) if fld.type == "float"
                                     else raw)
            except ValueError:
                out[fld.path[0]] = raw     # keep bad value verbatim for re-display
        return out

    def _populate_ch(self, ch: dict):
        for w in self._ch_frame.winfo_children(): w.destroy()
        self._ch_vars.clear()
        # Schema-driven render starting at row 1 (matches the original
        # one-row top padding inside _ch_frame).
        render_section(self._ch_frame, _CARRIER_CHANNEL_FIELDS, 1,
                         self._ch_vars, {})
        populate_from_schema(_CARRIER_CHANNEL_FIELDS, ch, self._ch_vars, {})

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
        for fld in walk_fields(_CARRIER_PHASE_NOISE_FIELDS):
            if fld.key not in self._pn_texts: continue
            val = parse_float_list(self._pn_texts[fld.key].get("1.0", "end"))
            if val: out[fld.path[0]] = val
        return out

    def _populate_pn(self, pn: dict):
        """Two text widgets — offset_hz and dbc_per_hz — for the per-carrier
        oscillator mask.  Same units and conventions as the previous
        global section."""
        for w in self._pn_frame.winfo_children(): w.destroy()
        self._pn_texts.clear()
        render_section(self._pn_frame, _CARRIER_PHASE_NOISE_FIELDS, 0,
                         {}, self._pn_texts)
        populate_from_schema(_CARRIER_PHASE_NOISE_FIELDS, pn,
                              {}, self._pn_texts)

    def to_dict(self) -> dict:
        d: dict = {}
        collect_from_schema(_CARRIER_MAIN_FIELDS, self._vars, {}, d)
        d["enabled"]     = bool(self._enabled.get())
        d["sweep_demod"] = bool(self._sweep_demod.get())

        if self._has_coding.get() and self._coding_vars:
            cod: dict = {}
            collect_from_schema(_CARRIER_CODING_FIELDS,
                                  self._coding_vars, {}, cod)
            # Don't write an empty `matrix` key — the simulator's loader treats
            # absence as "use the bundled default", which is the desired
            # behavior when the user leaves the field blank or the scheme
            # isn't ldpc (in which case the field isn't shown at all).
            if not cod.get("matrix"):
                cod.pop("matrix", None)
            if cod:
                d["coding"] = cod

        if self._has_ch.get() and self._ch_vars:
            ch: dict = {}
            collect_from_schema(_CARRIER_CHANNEL_FIELDS, self._ch_vars, {}, ch)
            d["channel"] = ch

        if self._has_pn.get() and self._pn_texts:
            pn: dict = {}
            collect_from_schema(_CARRIER_PHASE_NOISE_FIELDS,
                                  {}, self._pn_texts, pn)
            if pn.get("offset_hz") or pn.get("dbc_per_hz"):
                d["phase_noise"] = {"enabled": True, **pn}
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
        render_tab(nb, _GENERAL_TAB, self._vars, self._texts)

    def _build_amplifier_tab(self, nb):
        render_tab(nb, _AMPLIFIER_TAB, self._vars, self._texts)

    def _build_sweep_output_tab(self, nb):
        render_tab(nb, _SWEEP_OUTPUT_TAB, self._vars, self._texts)

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

        self._carr_inner = scrollable(tab)

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
            populate_from_schema(schema, cfg, self._vars, self._texts)

        for cf in self._carriers: cf.destroy()
        self._carriers.clear()
        for carr in cfg.get("carrier", []):
            self._add_carrier(carr)

    def _collect(self) -> dict:
        cfg: dict = {}
        for schema in (_GENERAL_TAB, _SWEEP_OUTPUT_TAB, _AMPLIFIER_TAB):
            collect_from_schema(schema, self._vars, self._texts, cfg)
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
