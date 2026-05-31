"""Minimal aligned-TOML serialisation primitives.

``lit`` / ``arr`` format scalars and lists; ``emit_table`` appends a table
header plus aligned ``key = value`` lines.  The caller decides table order
and grouping — these helpers carry no application-specific knowledge.
"""


def lit(v) -> str:
    """Format a scalar as a TOML literal."""
    if isinstance(v, bool):  return "true" if v else "false"
    if isinstance(v, str):   return f'"{v}"'
    if isinstance(v, float): return f"{int(v):_}" if v == int(v) else f"{v:g}"
    if isinstance(v, int):   return f"{v:_}"
    return str(v)


def arr(lst) -> str:
    """Format a list as a TOML inline array."""
    return "[" + ", ".join(lit(x) for x in lst) + "]"


def emit_table(L: list[str], header: str, d: dict,
               keys: tuple[str, ...] | None = None) -> None:
    """Append a `header` line followed by ``key = value`` lines, with keys.

    padded so the ``=`` signs align within the table.  None-valued keys are
    skipped; `keys` (when given) restricts and orders the emitted keys,
    otherwise the dict's own key order is used.  List values render as inline
    arrays.
    """
    chosen = [k for k in (keys or tuple(d.keys()))
              if k in d and d[k] is not None]
    L.append(header)
    width = max((len(k) for k in chosen), default=0)
    for k in chosen:
        v = d[k]
        rhs = arr(v) if isinstance(v, list) else lit(v)
        L.append(f"{k:<{width}} = {rhs}")
