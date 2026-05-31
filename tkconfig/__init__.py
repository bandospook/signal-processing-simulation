"""tkconfig — a small, project-agnostic toolkit for building tkinter TOML.

configuration editors from a declarative schema.

Define a form as `Tab`/`Section`/`Field` data, then:

  - ``render_tab(notebook, tab, variables, texts)`` builds the widgets,
  - ``populate_from_schema(tab, cfg, variables, texts)`` loads cfg → widgets,
  - ``collect_from_schema(tab, variables, texts, cfg)`` reads widgets → cfg,
  - ``emit_table(...)`` / ``lit`` / ``arr`` help serialise cfg back to TOML.

No application-specific knowledge lives here; the meaning of every field is
carried by its ``path`` into the cfg dict.
"""
from .schema import Field, Section, Tab
from .widgets import (Tip, entry, fmt, labeled, make_browse_cb,
                      parse_float_list, scrollable)
from .serde import (cfg_get, cfg_set, collect_from_schema,
                    populate_from_schema, walk_fields)
from .render import render_section, render_tab
from .toml_writer import arr, emit_table, lit

__all__ = [
    "Field", "Section", "Tab",
    "Tip", "entry", "fmt", "labeled", "make_browse_cb",
    "parse_float_list", "scrollable",
    "cfg_get", "cfg_set", "collect_from_schema",
    "populate_from_schema", "walk_fields",
    "render_section", "render_tab",
    "arr", "emit_table", "lit",
]
