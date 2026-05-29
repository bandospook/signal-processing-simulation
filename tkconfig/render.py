"""Render a schema onto tkinter widgets.

``render_tab`` builds a whole Tab onto a Notebook; ``render_section`` builds
a single Section (used directly for sub-sections like per-item groups).
Both create tk Variables / Text widgets keyed by ``Field.key`` and register
them into the caller-supplied dicts, which ``tkconfig.serde`` then reads and
writes.
"""
import tkinter as tk
from tkinter import ttk

from .schema import Field, Section, Tab
from .serde import walk_fields
from .widgets import Tip, entry, labeled, make_browse_cb, scrollable


def _render_scalar(parent, fld: Field, row: int, col: int,
                   variables: dict[str, tk.Variable],
                   *, right_col_padx: tuple[int, int] = (16, 4),
                   tracker: dict[str, tuple[tk.Widget, ...]] | None = None) -> None:
    """Render one single-row label+widget pair at (row, col..col+1).

    Supports every "scalar" field type — int / float / float_optional / str /
    path / str_enum / float_list (all use a StringVar).  If ``tracker`` is
    given, the (label, input-widget) pair is stored at ``tracker[fld.key]``
    so ``_wire_visibility`` can show/hide them.
    """
    padx_kw = {} if col == 0 else {"padx": right_col_padx}
    lbl = labeled(parent, fld.label + ":", row, col, **padx_kw)
    var = tk.StringVar()
    variables[fld.key] = var
    if fld.type == "str_enum":
        cb = ttk.Combobox(parent, textvariable=var, values=list(fld.options),
                           state="readonly", width=fld.width)
        cb.grid(row=row, column=col + 1, sticky="w", pady=2)
        if fld.tip: Tip(cb, fld.tip)
        if tracker is not None: tracker[fld.key] = (lbl, cb)
        return
    if fld.type == "path":
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=row, column=col + 1, sticky="w")
        entry(row_frame, var, 0, 0, width=fld.width, tip=fld.tip)
        ttk.Button(row_frame, text="Browse…", width=8,
                   command=make_browse_cb(var)).grid(row=0, column=1, padx=4)
        if tracker is not None: tracker[fld.key] = (lbl, row_frame)
        return
    ent = entry(parent, var, row, col + 1, width=fld.width, tip=fld.tip)
    if tracker is not None: tracker[fld.key] = (lbl, ent)


def _render_field(parent, fld: Field, row: int,
                  variables: dict[str, tk.Variable],
                  texts: dict[str, tk.Text]) -> int:
    """Create the widget(s) for one Field in a columns=1 layout; return the
    next free row."""
    if fld.type == "bool":
        var = tk.BooleanVar(value=bool(fld.default))
        variables[fld.key] = var
        cb = ttk.Checkbutton(parent, text=fld.label, variable=var)
        cb.grid(row=row, column=0, columnspan=4, sticky="w", pady=(6, 0))
        if fld.tip: Tip(cb, fld.tip)
        return row + 1

    if fld.type == "float_list_text":
        ttk.Label(parent, text=fld.label + ":", foreground="gray").grid(
            row=row, column=0, columnspan=4, sticky="w")
        t = tk.Text(parent, height=1, width=64, wrap="word",
                    font=("Consolas", 9))
        t.grid(row=row + 1, column=0, columnspan=4, sticky="ew", pady=2)
        texts[fld.key] = t
        if fld.tip: Tip(t, fld.tip)
        return row + 2

    _render_scalar(parent, fld, row, 0, variables)
    return row + 1


def _wire_visibility(sec: Section, variables: dict[str, tk.Variable],
                     tracker: dict[str, tuple[tk.Widget, ...]]) -> None:
    """For each gated field, trace the controlling var so the field's widgets
    show/hide live as the controller's value changes."""
    for fld in walk_fields(sec):
        if fld.visible_when is None:
            continue
        ctrl_path, allowed = fld.visible_when
        ctrl_key = ".".join(ctrl_path)
        widgets = tracker.get(fld.key)
        if ctrl_key not in variables or not widgets:
            continue
        def _update(*_, _ws=widgets, _ctrl=ctrl_key, _allowed=allowed):
            value = str(variables[_ctrl].get())
            for w in _ws:
                if value in _allowed: w.grid()
                else:                 w.grid_remove()
        variables[ctrl_key].trace_add("write", _update)
        _update()    # apply initial visibility


def render_section(parent, sec: Section, section_row: int,
                   variables: dict[str, tk.Variable],
                   texts: dict[str, tk.Text]) -> int:
    """Render one Section's fields onto `parent` starting at `section_row`;
    return the next free row.  Section title / separator / description are
    drawn by `render_tab`; this handles fields and visible_when gating."""
    tracker: dict[str, tuple[tk.Widget, ...]] = {}
    if sec.columns == 2:
        for i in range(0, len(sec.fields), 2):
            _render_scalar(parent, sec.fields[i], section_row, 0, variables,
                           right_col_padx=sec.right_col_padx, tracker=tracker)
            if i + 1 < len(sec.fields):
                _render_scalar(parent, sec.fields[i + 1], section_row, 2,
                               variables, right_col_padx=sec.right_col_padx,
                               tracker=tracker)
            section_row += 1
        _wire_visibility(sec, variables, tracker)
        return section_row
    for fld in sec.fields:
        section_row = _render_field(parent, fld, section_row, variables, texts)
    _wire_visibility(sec, variables, tracker)
    return section_row


def render_tab(nb, schema: Tab,
               variables: dict[str, tk.Variable],
               texts: dict[str, tk.Text]) -> None:
    """Build a Tab from its schema onto a Notebook.  Each section gets a bold
    title, optional separator, optional gray description, then its fields.
    Columns 1 and 3 expand if extra width is available."""
    frame = ttk.Frame(nb); nb.add(frame, text=schema.name)
    inner = scrollable(frame)
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
        r = render_section(inner, sec, r, variables, texts)
    inner.columnconfigure(1, weight=1)
    if any(s.columns == 2 for s in schema.sections):
        inner.columnconfigure(3, weight=1)
