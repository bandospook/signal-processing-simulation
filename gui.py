#!/usr/bin/env python3
"""
gui.py — TOML editor and simulation launcher.
No dependency on sim/* modules — interfaces only with simulation.toml and main.py.
Usage: python gui.py [path/to/simulation.toml]
"""
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import tomllib

_ICON_B64 = (
    "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"
    "wkePAHzCx/8ANlyElpgctQIAAAAASUVORK5CYII="
)


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

def _lf(parent, text, row, col, **kw):
    ttk.Label(parent, text=text).grid(row=row, column=col, sticky="w",
                                      padx=(0, 4), pady=2, **kw)

def _ent(parent, var, row, col, width=18, **kw):
    e = ttk.Entry(parent, textvariable=var, width=width)
    e.grid(row=row, column=col, sticky="w", pady=2, **kw)
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
        ("name",        "Name",             "str",   "carrier"),
        ("modulation",  "Modulation",       "str",   "BPSK"),
        ("symbol_rate", "Symbol Rate (Hz)", "float", "1e6"),
        ("sps",         "SPS",              "int",   "4"),
        ("rolloff",     "Roll-off",         "float", "0.35"),
        ("filter_span", "Filter Span",      "int",   "8"),
        ("num_symbols", "Num Symbols",      "int",   "1000"),
        ("power_db",    "Power (dB)",       "float", "0.0"),
        ("freq",        "Freq (Hz)",        "float", "0.0"),
    ]
    _SEEKER = [
        ("target_ber",    "Target BER",          "float", "0.001"),
        ("confidence",    "Confidence",           "float", "0.95"),
        ("ber_accuracy",  "BER Accuracy",         "float", "0.0005"),
        ("noise_lo_dbfs", "Noise Lo (dBFS/Hz)",   "float", "-160.0"),
        ("noise_hi_dbfs", "Noise Hi (dBFS/Hz)",   "float", "-80.0"),
    ]
    _CH = [
        ("ripple_db",         "Ripple (dB)",      "float", "0.5"),
        ("ripple_cycles",     "Ripple Cycles",    "float", "2.0"),
        ("max_phase_dev_deg", "Max Phase (°)",    "float", "5.0"),
        ("phase_poly_order",  "Phase Poly Order", "int",   "2"),
        ("plot",              "Plot Filename",    "str",   ""),
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

    def _build(self, d: dict):
        ttk.Button(self, text="Remove", command=self._on_remove,
                   width=8).grid(row=0, column=3, sticky="ne", padx=2)

        # Main parameter fields (2-column grid)
        for i, (key, label, typ, dflt) in enumerate(self._MAIN):
            raw = d.get(key, dflt)
            var = tk.StringVar(value=_fmt(raw) if isinstance(raw, (int, float)) else str(raw))
            self._vars[key] = var
            r, c = (i // 2) + 1, (i % 2) * 2
            _lf(self, label + ":", r, c)
            _ent(self, var, r, c + 1, width=14)
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
        for i, (key, label, typ, dflt) in enumerate(self._SEEKER):
            raw = sk.get(key, dflt)
            var = tk.StringVar(value=_fmt(raw) if isinstance(raw, (int, float)) else str(raw))
            self._sk_vars[key] = var
            r, c = i // 2, (i % 2) * 2
            _lf(self._seeker_frame, label + ":", r, c)
            _ent(self._seeker_frame, var, r, c + 1, width=12)

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
        for i, (key, label, typ, dflt) in enumerate(self._CH):
            raw = ch.get(key, dflt)
            var = tk.StringVar(value=_fmt(raw) if isinstance(raw, (int, float)) else str(raw))
            self._ch_vars[key] = var
            r, c = (i // 2) + 1, (i % 2) * 2
            _lf(self._ch_frame, label + ":", r, c)
            _ent(self._ch_frame, var, r, c + 1, width=14)

    def to_dict(self) -> dict:
        d = {}
        for key, _, typ, _ in self._MAIN:
            raw = self._vars[key].get().strip()
            d[key] = (int(float(raw)) if typ == "int"
                      else float(raw) if typ == "float" else raw)
        d["enabled"]     = bool(self._enabled.get())
        d["sweep_demod"] = bool(self._sweep_demod.get())
        d["use_seeker"]  = bool(self._use_seeker.get())

        if d["enabled"] and d["sweep_demod"] and d["use_seeker"]:
            sk: dict = {}
            for key, _, _, _ in self._SEEKER:
                raw = self._sk_vars[key].get().strip()
                try:
                    sk[key] = float(raw)
                except ValueError:
                    pass
            if sk:
                d["seeker"] = sk

        if self._has_ch.get() and self._ch_vars:
            ch: dict = {}
            for key, _, typ, _ in self._CH:
                if key not in self._ch_vars: continue
                raw = self._ch_vars[key].get().strip()
                ch[key] = (int(float(raw)) if typ == "int"
                           else float(raw) if typ == "float" else raw)
            d["channel"] = ch
        return d


# ── Main application ──────────────────────────────────────────────────────────

_PCT_RE = re.compile(r'^\[\s*(\d+)%\]')


class App:
    def __init__(self, root: tk.Tk, path: Path):
        self.root      = root
        self.path      = path
        self._carriers: list[CarrierFrame] = []
        self._vars:     dict[str, tk.StringVar] = {}
        self._texts:    dict[str, tk.Text] = {}
        self._proc     = None
        self._running  = False
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
            log_outer, height=4, wrap="word", font=("Consolas", 8),
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
        _lf(f, "Seed:", r, 0);  _ent(f, self._sv("sim.seed"), r, 1);  r += 1

        r = self._section(f, "Wideband", r)
        _lf(f, "Sample Rate (Hz):", r, 0)
        _ent(f, self._sv("wb.sample_rate"), r, 1, width=20);  r += 1
        _lf(f, "Noise Density (dBFS/Hz):", r, 0)
        _ent(f, self._sv("wb.noise"), r, 1);  r += 1
        ttk.Label(f, text="Leave blank to disable AWGN noise.",
                  foreground="gray").grid(row=r, column=1, sticky="w");  r += 1

        r = self._section(f, "Overlap-Add (OLA) Filter", r)
        _lf(f, "Filter Span:", r, 0);  _ent(f, self._sv("ola.filter_span"), r, 1);  r += 1
        _lf(f, "Block Size:", r, 0);   _ent(f, self._sv("ola.block_size"), r, 1);   r += 1
        f.columnconfigure(1, weight=1)

    def _build_amplifier_tab(self, nb):
        tab = ttk.Frame(nb);  nb.add(tab, text="Amplifier")
        f = _scrollable(tab)
        r = 0
        _lf(f, "Input Backoff (dB):", r, 0)
        _ent(f, self._sv("amp.ibo"), r, 1);  r += 2

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
        _ent(f, self._sv("sweep.ibo"), r, 1, width=44);  r += 1
        _lf(f, "Noise values (dBFS/Hz):", r, 0)
        _ent(f, self._sv("sweep.noise"), r, 1, width=44);  r += 1
        ttk.Label(f, text="Example: 0.0, 1.5, 3.0, 4.5, 6.0",
                  foreground="gray").grid(row=r, column=1, sticky="w");  r += 1

        r = self._section(f, "Output Files", r)
        _lf(f, "Output Directory:", r, 0)
        row_frame = ttk.Frame(f)
        row_frame.grid(row=r, column=1, sticky="w");  r += 1
        _ent(row_frame, self._sv("out.dir"), 0, 0, width=28)
        ttk.Button(row_frame, text="Browse…", command=self._browse_out_dir,
                   width=8).grid(row=0, column=1, padx=4)
        for label, key in (
            ("Wideband plot:",         "out.wideband"),
            ("NL tables plot:",        "out.nl_tables"),
            ("Sweep plot:",            "out.sweep"),
            ("Sweep table:",           "out.sweep_table"),
            ("Detector results:",      "out.detector_results"),
        ):
            _lf(f, label, r, 0)
            _ent(f, self._sv(key), r, 1);  r += 1
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
        names = ["All"] + [cf._vars["name"].get() or "carrier" for cf in self._carriers]
        self._focus_combo["values"] = names
        if self._focus_var.get() not in names:
            self._focus_var.set("All")
        self._apply_focus()

    def _apply_focus(self, *_):
        sel = self._focus_var.get()
        for cf in self._carriers:
            cf_name = cf._vars.get("name", tk.StringVar()).get()
            if sel == "All" or cf_name == sel:
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
        except Exception as e:
            messagebox.showerror("Load error", str(e));  return
        self.path = path
        self._path_var.set(str(path))
        self._populate(cfg)
        self._status.set(f"Loaded: {path}")

    def _populate(self, cfg: dict):
        sim = cfg.get("simulation", {})
        self._vars["sim.seed"].set(str(sim.get("seed", 42)))

        wb = cfg.get("wideband", {})
        self._vars["wb.sample_rate"].set(_fmt(wb.get("sample_rate", 16e6)))
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
        except Exception as e:
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

    def _log_clear(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def _log_append(self, msg: str):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _launch(self):
        if self._running:
            return
        try:
            self._save()
        except Exception as e:
            messagebox.showerror("Save error", str(e))
            return

        self._log_clear()
        self._progress["value"] = 0
        self._set_running(True)
        self._status.set("Running simulation...")

        main_py = Path(__file__).parent / "main.py"
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", str(main_py), str(self.path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(__file__).parent),
            )
        except Exception as e:
            messagebox.showerror("Launch error", str(e))
            self._set_running(False)
            return

        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._read_output, daemon=True).start()
        self.root.after(100, self._poll_proc)

    def _read_output(self):
        if self._proc is None or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            self._queue.put(line.rstrip())
        self._proc.wait()
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
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass
    App(root, toml_path)
    root.mainloop()
