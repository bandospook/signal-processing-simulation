"""Read/write form values between a cfg dict and the tk variables/widgets.

``populate_from_schema`` pushes cfg values into the bindings created by the
renderer; ``collect_from_schema`` reads them back out.  Both walk the same
schema, so adding a field needs no change here.
"""
import tkinter as tk

from .widgets import fmt, parse_float_list


def walk_fields(schema):
    """Yield every Field in a Tab or Section schema.

    Accepting either lets the same populate / collect helpers walk a full
    tab (Tab → many sections) and a single sub-section (Section → its fields).
    """
    if hasattr(schema, "sections"):
        for sec in schema.sections:
            yield from sec.fields
    else:
        yield from schema.fields


def cfg_get(cfg: dict, path: tuple[str, ...]):
    """Navigate cfg by path; return None if any intermediate key is missing."""
    d = cfg
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def cfg_set(cfg: dict, path: tuple[str, ...], value) -> None:
    """Insert `value` at `path` in `cfg`, creating intermediate dicts."""
    d = cfg
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def populate_from_schema(schema, cfg: dict,
                         variables: dict[str, tk.Variable],
                         texts: dict[str, tk.Text]) -> None:
    """Read each field's cfg value and push the formatted form into its.

    binding (StringVar / BooleanVar / Text widget) — dispatched by type.

    For ``float_optional``: a missing / None cfg value → empty string in the
    var (which collect interprets as "omit").  For every other type, a
    missing value substitutes the field's ``default``.
    """
    for fld in walk_fields(schema):
        raw = cfg_get(cfg, fld.path)

        if fld.type == "float_list_text":
            t = texts[fld.key]
            t.delete("1.0", "end")
            if raw:
                t.insert("1.0", ", ".join(fmt(x) for x in raw))
            continue
        if fld.type == "bool":
            variables[fld.key].set(bool(raw) if raw is not None
                                    else bool(fld.default))
            continue
        if fld.type == "float_list":
            variables[fld.key].set(", ".join(fmt(x) for x in (raw or [])))
            continue

        var = variables[fld.key]
        if fld.type == "float_optional":
            var.set(fmt(raw) if raw is not None else "")
        elif raw is None:
            d = fld.default
            var.set(fmt(d) if isinstance(d, (int, float)) else str(d))
        elif isinstance(raw, (int, float)):
            var.set(fmt(raw))
        else:
            var.set(str(raw))


def collect_from_schema(schema,
                        variables: dict[str, tk.Variable],
                        texts: dict[str, tk.Text],
                        cfg: dict) -> None:
    """Read each binding, parse per the field's type, and write into `cfg`.

    Empty ``float_optional`` values are omitted from `cfg` so the caller's
    TOML writer doesn't emit a key the user left blank.  A gated field whose
    ``visible_when`` controller doesn't currently match is skipped entirely —
    a hidden field never contributes to the output.
    """
    for fld in walk_fields(schema):
        if fld.visible_when is not None:
            ctrl_path, allowed = fld.visible_when
            ctrl_key = ".".join(ctrl_path)
            if ctrl_key in variables and str(variables[ctrl_key].get()) not in allowed:
                continue
        if fld.type == "float_list_text":
            cfg_set(cfg, fld.path,
                    parse_float_list(texts[fld.key].get("1.0", "end")))
            continue
        if fld.type == "bool":
            cfg_set(cfg, fld.path, bool(variables[fld.key].get()))
            continue

        raw = str(variables[fld.key].get()).strip()
        if fld.type == "float_optional":
            if raw:
                cfg_set(cfg, fld.path, float(raw))
            continue
        if fld.type == "float_list":
            cfg_set(cfg, fld.path, parse_float_list(raw))
            continue
        if fld.type == "int":
            value = int(float(raw))
        elif fld.type == "float":
            value = float(raw)
        else:
            value = raw   # "str" / "path" / "str_enum"
        cfg_set(cfg, fld.path, value)
