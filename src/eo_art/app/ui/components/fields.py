"""One form field per ``FieldSpec``: shared by the config form and the
prepare-chain / overlay editors."""

from collections.abc import Callable
from typing import Any

import flet as ft

from eo_art.app.config_forms import FieldSpec, parse_number
from eo_art.app.ui.widgets import dropdown

_ZERO = {"bool": False, "int": 0, "float": 0.0, "str": ""}


def placeholder(spec: FieldSpec) -> Any:
    """Value an optional field starts from when it gets switched on."""
    if spec.default is not None:
        return spec.default
    if spec.kind == "tuple":
        return [_ZERO.get(spec.item or "float", 0)] * spec.size
    return _ZERO.get(spec.kind, spec.choices[0] if spec.choices else None)


def number_value(spec: FieldSpec, value: Any) -> Any:
    """Number fields hold the typed text while editing (so "2" doesn't turn
    into "2.0" mid-typing); parse it for saving."""
    return parse_number(spec.kind, value)


def _number_error(kind: str, value: Any) -> str | None:
    if not isinstance(parse_number(kind, value), str):
        return None
    return f"not a{'n integer' if kind == 'int' else ' number'}"


def field(
    spec: FieldSpec,
    value: Any,
    set_value: Callable[[Any], None],
    help_text: str | None = None,
    options: list[str] | None = None,
) -> ft.Control:
    """``value`` None = unset (shows the schema default). ``options`` turns a
    text field into a dropdown over those values (plus the current one)."""
    label = spec.label + (" *" if spec.required else "")
    enabled = not spec.optional or value is not None
    shown = value if value is not None else placeholder(spec)

    if spec.kind == "str" and options:
        choices = list(dict.fromkeys([*([shown] if shown else []), *options]))
        control: ft.Control = dropdown(
            label, [(o, o) for o in choices], shown or None, set_value, expand=True
        )
    elif spec.kind == "bool":
        control = ft.Checkbox(
            label=label,
            value=bool(shown),
            on_change=lambda e: set_value(e.control.value),
        )
    elif spec.kind == "enum":
        control = dropdown(
            label, [(c, c) for c in spec.choices], shown, set_value, expand=True
        )
    elif spec.kind in ("int", "float"):
        control = ft.TextField(
            label=label,
            value=shown if isinstance(shown, str) else str(shown),
            keyboard_type=ft.KeyboardType.NUMBER,
            error=_number_error(spec.kind, value) if value is not None else None,
            on_change=lambda e: set_value(e.control.value),
            dense=True,
        )
    elif spec.kind == "tuple":
        items = list(shown)

        def set_item(index: int, text: str) -> None:
            set_value([text if i == index else v for i, v in enumerate(items)])

        kind = spec.item or "float"
        control = ft.Row(
            [
                ft.TextField(
                    label=f"{label} {i + 1}",
                    value=item if isinstance(item, str) else str(item),
                    keyboard_type=ft.KeyboardType.NUMBER,
                    error=_number_error(kind, item),
                    on_change=lambda e, i=i: set_item(i, e.control.value),
                    dense=True,
                    expand=True,
                )
                for i, item in enumerate(items)
            ],
            spacing=4,
        )
    else:
        control = ft.TextField(
            label=label,
            value="" if shown is None else str(shown),
            # An untouched empty required field stays MISSING instead of "".
            on_change=lambda e: set_value(
                None if e.control.value == "" and not spec.optional else e.control.value
            ),
            dense=True,
        )
    control.disabled = not enabled
    # Hovering the field (its name) or the info icon explains it.
    control.tooltip = f"{help_text}\n\n{spec.path}" if help_text else spec.path
    if help_text:
        info = ft.Icon(
            ft.Icons.INFO_OUTLINE,
            size=16,
            color=ft.Colors.ON_SURFACE_VARIANT,
            tooltip=help_text,
        )
        fill = spec.kind != "bool"
        control = ft.Row(
            [ft.Container(control, expand=fill), info],
            spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    if spec.optional:
        toggle = ft.Checkbox(
            value=enabled,
            tooltip=f"set {spec.path}",
            on_change=lambda e: set_value(
                placeholder(spec) if e.control.value else None
            ),
        )
        control = ft.Row([toggle, ft.Container(control, expand=True)], spacing=0)
    # Four or more inputs side by side need a row of their own.
    wide = spec.kind == "tuple" and spec.size >= 4
    return ft.Container(control, col=12 if wide else {"sm": 12, "md": 6, "xl": 4})
