"""Declarative form schema: Field / Section / Tab.

These are plain frozen dataclasses describing a TOML-backed form.  A `Tab`
is rendered onto a tkinter Notebook by ``tkconfig.render``; values flow
between the cfg dict and the widgets via ``tkconfig.serde``.  Nothing here
is specific to any one application — the meaning of each field comes entirely
from its ``path`` into the cfg dict.
"""
from dataclasses import dataclass


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
      "str_enum"          StringVar + Combobox limited to ``options``.
      "bool"              BooleanVar + Checkbutton (label is button text).
      "float_list"        StringVar + Entry of comma-separated floats.
      "float_list_text"   tk.Text widget (label above, multi-line capable).

    ``options`` is the allowed value list for ``"str_enum"`` (ignored for
    other types).  ``default`` is used by populate when the cfg value is
    missing; for "float_optional" a missing / None value maps to an empty
    input instead.

    ``visible_when`` gates rendering on another field's current value.
    Format: ``(controller_path, allowed_values)`` — e.g.
    ``(("scheme",), ("ldpc",))`` means "show only when the field at path
    ``('scheme',)`` in the same section holds the value ``'ldpc'``".  The
    renderer wires a trace on the controller var so the gated widgets
    show/hide live.  ``None`` (the default) means always visible.
    """
    path:    tuple[str, ...]
    label:   str
    type:    str
    default: object                                                       = None
    tip:     str                                                          = ""
    width:   int                                                          = 20
    options: tuple[str, ...]                                              = ()
    visible_when: tuple[tuple[str, ...], tuple[str, ...]] | None          = None

    @property
    def key(self) -> str:
        return ".".join(self.path)


@dataclass(frozen=True)
class Section:
    """A titled group of fields.

    ``description`` renders as a gray paragraph between the title and the
    fields.  ``separator`` toggles the horizontal rule under the title.
    ``right_col_padx`` overrides the padx applied to the right-column label
    in two-column layouts; (16, 4) gives a wider gap between groups, (0, 4)
    a tighter look.
    """
    title:          str
    fields:         tuple[Field, ...]
    columns:        int             = 2
    description:    str             = ""
    separator:      bool            = True
    right_col_padx: tuple[int, int] = (16, 4)


@dataclass(frozen=True)
class Tab:
    name:     str
    sections: tuple[Section, ...]
