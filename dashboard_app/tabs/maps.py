"""Render the detailed maps tab."""

from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


def render_maps_tab(ctx: dict) -> None:
    """Render the detailed maps tab."""
    globals().update(ctx)
    render_section_title(9, "Cartographie detaillee des cas et foyers")
    render_tab_narrative("cartographie")
    render_detailed_maps_tab(
        df_f=df_f,
        show_maps=show_maps,
        idsr_mode=IDSR_MODE,
    )


