"""Editors for the render config's lists: a prepare chain (DEM or overlay)
and the overlay list. They hold no state of their own: the config form owns
the lists and hands in an ``update`` that applies a change to the latest
version of one list."""

from collections.abc import Callable
from typing import Any

import flet as ft

from eo_art.app.config_forms import (
    CHAIN_KEY,
    op_fields,
    overlay_fields,
    prep_ops,
)
from eo_art.app.field_help import OP_HELP, OVERLAY_HELP
from eo_art.app.ui.components.fields import field
from eo_art.app.ui.widgets import caption

Items = list[dict[str, Any]]
# Applies a function to the current list (functional update, so fast typing
# never works on a stale copy).
Update = Callable[[Callable[[Items], Items]], None]


def _set_key(update: Update, index: int, key: str, value: Any) -> None:
    update(
        lambda items: [
            {**it, key: value} if i == index else it for i, it in enumerate(items)
        ]
    )


def _move(update: Update, index: int, step: int) -> None:
    def move(items: Items) -> Items:
        target = index + step
        if not 0 <= target < len(items):
            return items
        moved = list(items)
        moved[index], moved[target] = moved[target], moved[index]
        return moved

    update(move)


def _remove(update: Update, index: int) -> None:
    update(lambda items: [it for i, it in enumerate(items) if i != index])


def _item_actions(update: Update, index: int, count: int, what: str) -> ft.Control:
    return ft.Row(
        [
            ft.IconButton(
                ft.Icons.ARROW_UPWARD,
                icon_size=18,
                tooltip=f"Move {what} up",
                disabled=index == 0,
                on_click=lambda: _move(update, index, -1),
            ),
            ft.IconButton(
                ft.Icons.ARROW_DOWNWARD,
                icon_size=18,
                tooltip=f"Move {what} down",
                disabled=index == count - 1,
                on_click=lambda: _move(update, index, 1),
            ),
            ft.IconButton(
                ft.Icons.DELETE_OUTLINE,
                icon_size=18,
                tooltip=f"Remove {what}",
                on_click=lambda: _remove(update, index),
            ),
        ],
        spacing=0,
    )


def _block(controls: list[ft.Control]) -> ft.Control:
    """A list item: indented, with a line down its left edge."""
    return ft.Container(
        ft.Column(controls, spacing=8),
        padding=ft.Padding.only(left=12, top=4, bottom=8),
        border=ft.Border.only(left=ft.BorderSide(2, ft.Colors.OUTLINE_VARIANT)),
    )


def _appender(update: Update, op: str) -> Callable[[], None]:
    """Click handler adding ``op``. Not ``lambda op=op: ...``: Flet counts
    defaulted parameters too and would pass the click event in as ``op``."""
    return lambda: update(lambda items: [*items, {"op": op}])


def chain_editor(chain: Items, update: Update) -> ft.Control:
    """Ordered prep ops: each with its parameters, reorderable, removable."""
    rows: list[ft.Control] = []
    for index, entry in enumerate(chain):
        name = entry["op"]
        description, param_help = OP_HELP.get(name, ("", {}))
        header = ft.Row(
            [
                ft.Text(f"{index + 1}. {name}", weight=ft.FontWeight.W_500),
                caption(description),
                ft.Container(expand=True),
                _item_actions(update, index, len(chain), "op"),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        specs = op_fields(name)
        fields = [
            field(
                spec,
                entry.get(spec.path),
                lambda value, i=index, key=spec.path: _set_key(update, i, key, value),
                param_help.get(spec.path),
            )
            for spec in specs
        ]
        rows.append(
            _block(
                [header, ft.ResponsiveRow(fields, run_spacing=8)]
                if fields
                else [header]
            )
        )

    add = ft.PopupMenuButton(
        content=ft.Row(
            [ft.Icon(ft.Icons.ADD, size=18), ft.Text("Add op")],
            spacing=4,
            tight=True,
        ),
        tooltip="Append a prep op to the chain",
        items=[
            ft.PopupMenuItem(
                content=ft.Text(f"{op} — {OP_HELP.get(op, ('', {}))[0]}"),
                on_click=_appender(update, op),
            )
            for op in prep_ops()
        ],
    )
    if not chain:
        rows.append(caption("No ops: the raster is used as it is."))
    return ft.Column([*rows, add], spacing=8)


def overlays_editor(overlays: Items, update: Update, rasters: list[str]) -> ft.Control:
    """Every overlay with its fields and its own prepare chain."""
    specs = overlay_fields()
    blocks: list[ft.Control] = []
    for index, overlay in enumerate(overlays):

        def update_chain(change: Callable[[Items], Items], i: int = index) -> None:
            update(
                lambda items: [
                    {**it, CHAIN_KEY: change(list(it.get(CHAIN_KEY) or []))}
                    if j == i
                    else it
                    for j, it in enumerate(items)
                ]
            )

        header = ft.Row(
            [
                ft.Text(
                    overlay.get("name") or f"overlay {index + 1}",
                    weight=ft.FontWeight.BOLD,
                ),
                caption(str(overlay.get("path") or "no raster chosen")),
                ft.Container(expand=True),
                _item_actions(update, index, len(overlays), "overlay"),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        fields = [
            field(
                spec,
                overlay.get(spec.path),
                lambda value, i=index, key=spec.path: _set_key(update, i, key, value),
                OVERLAY_HELP.get(spec.path),
                options=rasters if spec.path == "path" else None,
            )
            for spec in specs
        ]
        blocks.append(
            _block(
                [
                    header,
                    ft.ResponsiveRow(fields, run_spacing=8),
                    ft.Text("prepare", size=13, weight=ft.FontWeight.W_500),
                    chain_editor(list(overlay.get(CHAIN_KEY) or []), update_chain),
                ]
            )
        )

    def add_overlay() -> None:
        def add(items: Items) -> Items:
            return [
                *items,
                {
                    "name": f"overlay_{len(items) + 1}",
                    "path": rasters[0] if rasters else None,
                },
            ]

        update(add)

    if not overlays:
        blocks.append(caption("No overlays: the terrain is rendered without imagery."))
    blocks.append(
        ft.Row(
            [ft.OutlinedButton("Add overlay", icon=ft.Icons.ADD, on_click=add_overlay)]
        )
    )
    return ft.Column(blocks, spacing=12)
