"""Reusable tkinter widget helpers — tooltips, scrollable frames, the.

labelled-entry primitive, value formatting, and a directory-picker callback.

None of these know about the schema; they are the low-level building blocks
that ``tkconfig.render`` composes.
"""
import tkinter as tk
from tkinter import ttk, filedialog


def fmt(v) -> str:
    """Format a scalar for display: integral floats lose the trailing .0,.

    None becomes an empty string.
    """
    if isinstance(v, float): return str(int(v)) if v == int(v) else f"{v:g}"
    return str(v) if v is not None else ""


def parse_float_list(text: str) -> list[float]:
    """Parse a comma-separated (optionally bracketed) string into floats."""
    cleaned = text.strip().strip("[]")
    return [float(x) for x in cleaned.split(",") if x.strip()] if cleaned else []


class Tip:
    """Lightweight hover tooltip attached to any widget."""

    def __init__(self, widget: tk.Widget, text: str):
        """Attach a tooltip showing `text` to `widget` (no-op if `text` is empty)."""
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


def labeled(parent, text, row, col, **kw):
    """Grid a left-aligned Label at (row, col); return the Label widget."""
    kw.setdefault("padx", (0, 4))
    kw.setdefault("pady", 2)
    lbl = ttk.Label(parent, text=text)
    lbl.grid(row=row, column=col, sticky="w", **kw)
    return lbl


def entry(parent, var, row, col, width=18, tip="", **kw):
    """Grid an Entry bound to `var` at (row, col); return the Entry widget."""
    e = ttk.Entry(parent, textvariable=var, width=width)
    e.grid(row=row, column=col, sticky="w", pady=2, **kw)
    if tip:
        Tip(e, tip)
    return e


def scrollable(parent) -> ttk.Frame:
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


def make_browse_cb(var: tk.StringVar):
    """Directory-picker callback that writes the chosen path into `var`."""
    def _cb():
        path = filedialog.askdirectory(initialdir=var.get() or ".")
        if path:
            var.set(path)
    return _cb
