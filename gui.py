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
from misc.gen_icon import build_icon as _build_icon

_ICON_B64 = base64.b64encode(_build_icon()).decode()


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

    ln("[simulation]");  kv("seed", cfg["simulation"]["seed"]);  ln()

    wb = cfg["wideband"]
    ln("[wideband]");  kv("sample_rate       ", wb["sample_rate"])
    if wb.get("noise_density_dbfs") is not None:
        kv("noise_density_dbfs", wb["noise_density_dbfs"])
    ln()

    amp = cfg["amplifier"]
    ln("[amplifier]");  kv("input_backoff_db", amp["input_backoff_db"]);  ln()
    ln("[amplifier.am_am]")
    kva("input ", amp["am_am"]["input"]);  kva("output", amp["am_am"]["output"]);  ln()
    ln("[amplifier.am_pm]")
    kva("input    ", amp["am_pm"]["input"]);  kva("phase_deg", amp["am_pm"]["phase_deg"]);  ln()

    ln("[ola]")
    kv("filter_span", cfg["ola"]["filter_span"], 12)
    kv("block_size ", cfg["ola"]["block_size"],  12)
    ln()

    o = cfg.get("output", {})
    ln("[output]");  kv("output_dir", o.get("output_dir", "."), 10)
    for k in ("wideband", "nl_tables", "sweep", "sweep_table", "detector_results"):
        if o.get(k): kv(k, o[k], 18)
    ln()

    sw = cfg.get("sweep", {})
    ibo, nsw = sw.get("ibo_db", []), sw.get("noise_density_dbfs", [])
    if ibo or nsw:
        ln("[sweep]")
        if ibo: kva("ibo_db            ", ibo)
        if nsw: kva("noise_density_dbfs", nsw)
        ln()

    for carr in cfg.get("carrier", []):
        ln("[[carrier]]")
        for k in ("name", "modulation", "symbol_rate", "sps", "rolloff", "filter_span",
                  "num_symbols", "power_db", "freq", "enabled", "sweep_demod", "use_seeker"):
            if k in carr:
                kv(f"{k:12}", carr[k])
        sk = carr.get("seeker")
        if sk:
            ln();  ln("[carrier.seeker]")
            for k, v in sk.items():
                kv(f"{k:14}", v)
        ch = carr.get("channel")
        if ch:
            ln();  ln("[carrier.channel]")
            for k, v in ch.items():
                (kva if isinstance(v, list) else kv)(f"{k:22}", v)
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
                 wraplength=280, font=("", 8),
                 padx=5, pady=3).pack()

    def _hide(self, _=None):
        if self._win:
            self._win.destroy()
            self._win = None


def _lf(parent, text, row, col, **kw):
    ttk.Label(parent, text=text).grid(row=row, column=col, sticky="w",
                                      padx=(0, 4), pady=2, **kw)

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


# ── CarrierFrame ──────────────────────────────────────────────────────────────

class CarrierFrame(ttk.LabelFrame):
    _MAIN = [
        ("name",        "Name",             "str",   "carrier",
         "Unique identifier for this carrier. Used in output reports and seeker results."),
        ("modulation",  "Modulation",       "str",   "BPSK",
         "Modulation scheme: BPSK, DBPSK, QPSK, OQPSK, 8PSK, 16QAM, 16APSK, 32APSK"),
        ("symbol_rate", "Symbol Rate (MHz)", "float", "1",
         "Symbol rate in MHz (megabaud). Occupied bandwidth ≈ symbol_rate × (1 + rolloff)."),
        ("sps",         "SPS",              "int",   "4",
         "Samples per symbol at the wideband composite sample rate. Integer ≥ 2; typical value: 4."),
        ("rolloff",     "Roll-off",         "float", "0.35",
         "RRC filter roll-off factor α (0 – 1). Higher = wider occupied bandwidth, lower peak ISI."),
        ("filter_span", "Filter Span",      "int",   "8",
         "RRC filter half-span in symbols. Total taps = filter_span × sps + 1."),
        ("num_symbols", "Num Symbols",      "int",   "1000",
         "Symbols simulated per run. Higher count improves BER statistical accuracy."),
        ("power_db",    "Power (dB)",       "float", "0.0",
         "Carrier power in dBFS relative to the wideband composite full-scale."),
        ("freq",        "Freq (MHz)",       "float", "0.0",
         "Carrier centre frequency offset from DC (MHz). Negative = below centre frequency."),
    ]
    _SEEKER = [
        ("target_ber",    "Target BER",          "float", "0.001",
         "Target bit-error ratio the seeker converges to (e.g. 0.001 = 10⁻³)."),
        ("confidence",    "Confidence",           "float", "0.95",
         "Statistical confidence level for the final BER estimate (e.g. 0.95 = 95%)."),
        ("ber_accuracy",  "BER Accuracy",         "float", "0.0005",
         "Acceptable half-width of the BER confidence interval at the converged noise level."),
        ("noise_lo_dbfs", "Noise Lo (dBFS/Hz)",   "float", "-160.0",
         "Lower bound of the bisection search. Must produce a BER below the target (dBFS/Hz)."),
        ("noise_hi_dbfs", "Noise Hi (dBFS/Hz)",   "float", "-80.0",
         "Upper bound of the bisection search. Must produce a BER above the target (dBFS/Hz)."),
    ]
    _CH = [
        ("ripple_db",         "Ripple (dB)",      "float", "0.5",
         "Peak-to-peak amplitude ripple across the carrier bandwidth (dB)."),
        ("ripple_cycles",     "Ripple Cycles",    "float", "2.0",
         "Number of full ripple cycles across the carrier bandwidth."),
        ("max_phase_dev_deg", "Max Phase (°)",    "float", "5.0",
         "Maximum deviation from linear phase across the carrier bandwidth (degrees)."),
        ("phase_poly_order",  "Phase Poly Order", "int",   "2",
         "Order of the polynomial used to model the phase-vs-frequency distortion."),
        ("plot",              "Plot Filename",    "str",   "",
         "Filename for the channel response plot. Leave blank to skip."),
    ]

    def __init__(self, parent, on_remove, data: dict, **kw):
        super().__init__(parent, text=data.get("name", "carrier"), padding=6, **kw)
        self._on_remove = on_remove
        self._vars:    dict[str, tk.Variable] = {}
        self._ch_vars: dict[str, tk.Variable] = {}
        self._sk_vars: dict[str, tk.Variable] = {}
        self._enabled     = tk.BooleanVar(value=data.get("enabled", True))
        self._sweep_demod = tk.BooleanVar(value=data.get("sweep_demod", False))
        # Use IntVar for radio: 0=fixed, 1=seeker
        self._use_seeker  = tk.IntVar(value=1 if data.get("use_seeker", False) else 0)
        ch = data.get("channel", {})
        self._has_ch = tk.BooleanVar(value=bool(ch) and ch.get("enabled", True))
        self._build(data)

    @property
    def carrier_name(self) -> str:
        return self._vars.get("name", tk.StringVar()).get() or "carrier"

    def _build(self, d: dict):
        ttk.Button(self, text="Remove", command=self._on_remove,
                   width=8).grid(row=0, column=3, sticky="ne", padx=2)

        # Main parameter fields (2-column grid)
        _MODS = ["BPSK", "DBPSK", "QPSK", "OQPSK", "8PSK", "16QAM", "16APSK", "32APSK"]
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
        det_row     = check_row + 1                # row 7
        sk_row      = det_row + 1                  # row 8
        ch_row      = sk_row + 1                   # row 9

        # ── Enable checkboxes ────────────────────────────────────────────────
        ttk.Checkbutton(self, text="Include in wideband",
                        variable=self._enabled,
                        command=self._update_visibility).grid(
            row=check_row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(self, text="Enable detector model",
                        variable=self._sweep_demod,
                        command=self._update_visibility).grid(
            row=check_row, column=2, columnspan=2, sticky="w", pady=(8, 0))

        # ── Mode radio buttons (shown only when both enables are on) ─────────
        self._radio_frame = ttk.Frame(self)
        self._radio_frame.grid(row=det_row, column=0, columnspan=4, sticky="w",
                                pady=(2, 0))
        ttk.Label(self._radio_frame, text="Mode:").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(self._radio_frame, text="Fixed noise level",
                        variable=self._use_seeker, value=0,
                        command=self._update_visibility).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(self._radio_frame, text="BER seeker",
                        variable=self._use_seeker, value=1,
                        command=self._update_visibility).pack(side="left")

        # ── Seeker parameter frame (shown only in seeker mode) ───────────────
        self._seeker_frame = ttk.LabelFrame(self, text="BER Seeker Parameters", padding=4)
        self._seeker_frame.grid(row=sk_row, column=0, columnspan=4, sticky="ew",
                                 padx=(14, 0), pady=(2, 0))
        sk = d.get("seeker", {})
        for i, (key, label, _, dflt, tip) in enumerate(self._SEEKER):
            raw = sk.get(key, dflt)
            var = tk.StringVar(value=_fmt(raw) if isinstance(raw, (int, float)) else str(raw))
            self._sk_vars[key] = var
            r, c = i // 2, (i % 2) * 2
            _lf(self._seeker_frame, label + ":", r, c)
            _ent(self._seeker_frame, var, r, c + 1, width=12, tip=tip)

        # ── Channel impairments ──────────────────────────────────────────────
        ttk.Checkbutton(self, text="Channel impairments", variable=self._has_ch,
                        command=self._toggle_ch).grid(
            row=ch_row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._ch_frame = ttk.Frame(self, padding=(14, 0, 0, 0))
        self._ch_frame.grid(row=ch_row + 1, column=0, columnspan=4, sticky="ew")
        if "channel" in d:
            self._populate_ch(d["channel"])

        self._update_visibility()

    def _update_visibility(self):
        both_on = self._enabled.get() and self._sweep_demod.get()
        if both_on:
            self._radio_frame.grid()
            if self._use_seeker.get():
                self._seeker_frame.grid()
            else:
                self._seeker_frame.grid_remove()
        else:
            self._radio_frame.grid_remove()
            self._seeker_frame.grid_remove()

    def _toggle_ch(self):
        if self._has_ch.get():
            self._populate_ch({})
        else:
            for w in self._ch_frame.winfo_children(): w.destroy()
            self._ch_vars.clear()

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

    def to_dict(self) -> dict:
        d = {}
        for key, _, typ, _, _ in self._MAIN:
            raw = self._vars[key].get().strip()
            d[key] = (int(float(raw)) if typ == "int"
                      else float(raw) if typ == "float" else raw)
        d["enabled"]     = bool(self._enabled.get())
        d["sweep_demod"] = bool(self._sweep_demod.get())
        d["use_seeker"]  = bool(self._use_seeker.get())

        if d["enabled"] and d["sweep_demod"] and d["use_seeker"]:
            sk: dict = {}
            for key, _, _, _, _ in self._SEEKER:
                raw = self._sk_vars[key].get().strip()
                try:
                    sk[key] = float(raw)
                except ValueError:
                    pass
            if sk:
                d["seeker"] = sk

        if self._has_ch.get() and self._ch_vars:
            ch: dict = {}
            for key, _, typ, _, _ in self._CH:
                if key not in self._ch_vars: continue
                raw = self._ch_vars[key].get().strip()
                ch[key] = (int(float(raw)) if typ == "int"
                           else float(raw) if typ == "float" else raw)
            d["channel"] = ch
        return d


# ── Main application ──────────────────────────────────────────────────────────

_PCT_RE   = re.compile(r'^\[\s*(\d+)%\]')
_CHUNK_RE = re.compile(r'^\s+chunk \d+/\d+')


class App:
    def __init__(self, root: tk.Tk, path: Path):
        self.root      = root
        self.path      = path
        self._carriers: list[CarrierFrame] = []
        self._vars:     dict[str, tk.StringVar] = {}
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

    def _text_widget(self, parent, key, row, height=2) -> tk.Text:
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
        tab = ttk.Frame(nb);  nb.add(tab, text="General")
        f = _scrollable(tab)
        r = self._section(f, "Simulation", 0)
        _lf(f, "Seed:", r, 0)
        _ent(f, self._sv("sim.seed"), r, 1,
             tip="Random seed for reproducible simulations (integer)."); r += 1

        r = self._section(f, "Wideband", r)
        _lf(f, "Sample Rate (MHz):", r, 0)
        _ent(f, self._sv("wb.sample_rate"), r, 1, width=20,
             tip="Composite wideband sample rate in MHz.\n"
                 "Must be at least 2x the highest carrier edge frequency."); r += 1
        _lf(f, "Noise Density (dBFS/Hz):", r, 0)
        _ent(f, self._sv("wb.noise"), r, 1,
             tip="AWGN noise power spectral density added after the amplifier (dBFS/Hz).\n"
                 "Leave blank to disable noise entirely."); r += 1
        ttk.Label(f, text="Leave blank to disable AWGN noise.",
                  foreground="gray").grid(row=r, column=1, sticky="w");  r += 1

        r = self._section(f, "Overlap-Add (OLA) Filter", r)
        _lf(f, "Filter Span:", r, 0)
        _ent(f, self._sv("ola.filter_span"), r, 1,
             tip="Half-span of the OLA resampling filter in symbols.\n"
                 "Longer span = better stopband rejection, higher latency."); r += 1
        _lf(f, "Block Size:", r, 0)
        _ent(f, self._sv("ola.block_size"), r, 1,
             tip="FFT block size for the overlap-add resampler (samples).\n"
                 "Must be a power of two; larger = more efficient for long filters."); r += 1
        f.columnconfigure(1, weight=1)

    def _build_amplifier_tab(self, nb):
        tab = ttk.Frame(nb);  nb.add(tab, text="Amplifier")
        f = _scrollable(tab)
        r = 0
        _lf(f, "Input Backoff (dB):", r, 0)
        _ent(f, self._sv("amp.ibo"), r, 1,
             tip="Input back-off relative to amplifier saturation (dB).\n"
                 "Higher IBO = more linear operation, lower output power.\n"
                 "0 dB = driven to saturation."); r += 2

        for title, ik, ok, olabel in (
            ("AM-AM Table", "amp.am_am.in", "amp.am_am.out", "Output"),
            ("AM-PM Table", "amp.am_pm.in", "amp.am_pm.phase", "Phase (°)"),
        ):
            ttk.Label(f, text=title, font=("", 10, "bold")).grid(
                row=r, column=0, columnspan=4, sticky="w", pady=(10, 2));  r += 1
            ttk.Label(f, text="Input amplitude (comma-separated):",
                      foreground="gray").grid(row=r, column=0, columnspan=4, sticky="w");  r += 1
            self._text_widget(f, ik, r);  r += 1
            ttk.Label(f, text=f"{olabel} (comma-separated):",
                      foreground="gray").grid(row=r, column=0, columnspan=4, sticky="w");  r += 1
            self._text_widget(f, ok, r);  r += 1
        f.columnconfigure(0, weight=1)

    def _build_sweep_output_tab(self, nb):
        tab = ttk.Frame(nb);  nb.add(tab, text="Sweep & Output")
        f = _scrollable(tab)
        r = self._section(f, "Parameter Sweep", 0)
        ttk.Label(f, text="Leave both fields blank to skip the sweep.",
                  foreground="gray").grid(row=r, column=0, columnspan=3, sticky="w");  r += 1
        _lf(f, "IBO values (dB):", r, 0)
        _ent(f, self._sv("sweep.ibo"), r, 1, width=44,
             tip="Comma-separated IBO values to sweep (dB).\n"
                 "Example: 0.0, 1.5, 3.0, 4.5, 6.0"); r += 1
        _lf(f, "Noise values (dBFS/Hz):", r, 0)
        _ent(f, self._sv("sweep.noise"), r, 1, width=44,
             tip="Comma-separated noise density values to sweep (dBFS/Hz).\n"
                 "Example: -140.0, -130.0, -120.0"); r += 1
        ttk.Label(f, text="Example: 0.0, 1.5, 3.0, 4.5, 6.0",
                  foreground="gray").grid(row=r, column=1, sticky="w");  r += 1

        r = self._section(f, "Output Files", r)
        _lf(f, "Output Directory:", r, 0)
        row_frame = ttk.Frame(f)
        row_frame.grid(row=r, column=1, sticky="w");  r += 1
        _ent(row_frame, self._sv("out.dir"), 0, 0, width=28)
        ttk.Button(row_frame, text="Browse…", command=self._browse_out_dir,
                   width=8).grid(row=0, column=1, padx=4)
        for label, key, tip in (
            ("Wideband plot:",    "out.wideband",
             "PNG filename for the wideband composite PSD plot."),
            ("NL tables plot:",  "out.nl_tables",
             "PNG filename for the AM-AM / AM-PM nonlinearity table plots."),
            ("Sweep plot:",      "out.sweep",
             "PNG filename for the 2D IBO × noise BER sweep heatmap."),
            ("Sweep table:",     "out.sweep_table",
             "Markdown filename for the numeric sweep results table."),
            ("Detector results:", "out.detector_results",
             "Markdown filename for per-carrier BER / EVM / IL report."),
        ):
            _lf(f, label, r, 0)
            _ent(f, self._sv(key), r, 1, tip=tip);  r += 1
        f.columnconfigure(1, weight=1)

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
        sim = cfg.get("simulation", {})
        self._vars["sim.seed"].set(str(sim.get("seed", 42)))

        wb = cfg.get("wideband", {})
        self._vars["wb.sample_rate"].set(_fmt(wb.get("sample_rate", 16)))
        nd = wb.get("noise_density_dbfs")
        self._vars["wb.noise"].set(_fmt(nd) if nd is not None else "")

        ola = cfg.get("ola", {})
        self._vars["ola.filter_span"].set(str(ola.get("filter_span", 16)))
        self._vars["ola.block_size"].set(str(ola.get("block_size", 4096)))

        amp = cfg.get("amplifier", {})
        self._vars["amp.ibo"].set(_fmt(amp.get("input_backoff_db", 3.0)))

        def set_text(key, lst):
            t = self._texts[key]
            t.delete("1.0", "end")
            t.insert("1.0", ", ".join(_fmt(x) for x in lst))

        am_am = amp.get("am_am", {})
        set_text("amp.am_am.in",  am_am.get("input", []))
        set_text("amp.am_am.out", am_am.get("output", []))
        am_pm = amp.get("am_pm", {})
        set_text("amp.am_pm.in",    am_pm.get("input", []))
        set_text("amp.am_pm.phase", am_pm.get("phase_deg", []))

        sw = cfg.get("sweep", {})
        self._vars["sweep.ibo"].set(  ", ".join(_fmt(x) for x in sw.get("ibo_db", [])))
        self._vars["sweep.noise"].set(", ".join(_fmt(x) for x in sw.get("noise_density_dbfs", [])))

        o = cfg.get("output", {})
        self._vars["out.dir"].set(o.get("output_dir", "."))
        self._vars["out.wideband"].set(o.get("wideband", ""))
        self._vars["out.nl_tables"].set(o.get("nl_tables", ""))
        self._vars["out.sweep"].set(o.get("sweep", ""))
        self._vars["out.sweep_table"].set(o.get("sweep_table", ""))
        self._vars["out.detector_results"].set(o.get("detector_results", ""))

        for cf in self._carriers: cf.destroy()
        self._carriers.clear()
        for carr in cfg.get("carrier", []):
            self._add_carrier(carr)

    def _collect(self) -> dict:
        def sv(key): return self._vars[key].get().strip()
        def fv(key): return float(sv(key))
        def iv(key): return int(float(sv(key)))
        def tv(key): return _parse_float_list(self._texts[key].get("1.0", "end"))

        cfg: dict = {
            "simulation": {"seed": iv("sim.seed")},
            "wideband":   {"sample_rate": fv("wb.sample_rate")},
            "amplifier": {
                "input_backoff_db": fv("amp.ibo"),
                "am_am": {"input": tv("amp.am_am.in"), "output": tv("amp.am_am.out")},
                "am_pm": {"input": tv("amp.am_pm.in"), "phase_deg": tv("amp.am_pm.phase")},
            },
            "ola":    {"filter_span": iv("ola.filter_span"), "block_size": iv("ola.block_size")},
            "output": {"output_dir": sv("out.dir") or "."},
        }

        noise_raw = sv("wb.noise")
        if noise_raw:
            cfg["wideband"]["noise_density_dbfs"] = float(noise_raw)

        for k, vk in (("wideband",          "out.wideband"),
                      ("nl_tables",          "out.nl_tables"),
                      ("sweep",              "out.sweep"),
                      ("sweep_table",        "out.sweep_table"),
                      ("detector_results",   "out.detector_results")):
            val = sv(vk)
            if val: cfg["output"][k] = val

        ibo_list = _parse_float_list(sv("sweep.ibo"))
        nsw_list = _parse_float_list(sv("sweep.noise"))
        if ibo_list or nsw_list:
            cfg["sweep"] = {"ibo_db": ibo_list, "noise_density_dbfs": nsw_list}

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

    def _browse_out_dir(self):
        d = filedialog.askdirectory(initialdir=self._vars["out.dir"].get() or ".")
        if d: self._vars["out.dir"].set(d)

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

        out_dir = Path(self._vars["out.dir"].get().strip() or ".")
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
        if self._running and not self._slow_warned:
            elapsed = time.monotonic() - self._last_progress_time
            if elapsed >= 30.0:
                label = self._last_progress_line or "(simulation start)"
                self._log_append(
                    f"[GUI] Still working — no progress update for {int(elapsed)}s "
                    f"(last step: {label})"
                )
                self._slow_warned = True
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
