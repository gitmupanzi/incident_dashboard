"""Affiche l'onglet de cartographie détaillée."""

from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


def render_maps_tab(ctx: dict) -> None:
    """Affiche l'onglet de cartographie détaillée."""
    globals().update(ctx)
    render_section_title(8, "Cartographie detaillee des cas et foyers")
    render_tab_narrative("cartographie")
    render_detailed_maps_tab(
        df_f=df_f,
        show_maps=show_maps,
        idsr_mode=IDSR_MODE,
    )


