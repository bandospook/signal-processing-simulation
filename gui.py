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
        self._has_ch      = tk.BooleanVar(value="channel" in data)
        self._ch_enabled  = tk.BooleanVar(value=data.get("channel", {}).get("enabled", True))
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
        self._ch_enabled.set(ch.get("enabled", True))
        ttk.Checkbutton(self._ch_frame, text="Enabled",
                        variable=self._ch_enabled).grid(row=0, column=0,
                                                         columnspan=4, sticky="w")
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
            ch: dict = {"enabled": bool(self._ch_enabled.get())}
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
        root.title("Simulation Config Editor")
        root.minsize(760, 580)
        self._build_ui()
        self._load(path)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
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
        ref: list[CarrierFrame] = [None]

        def remove():
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
