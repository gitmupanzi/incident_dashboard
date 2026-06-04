"""Affiche l'onglet de détail de la vue d'ensemble."""

from dashboard_app.advanced import st
from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


def render_overview_detail_tab(ctx: dict) -> None:
    """Affiche l'onglet de détail de la vue d'ensemble."""
    globals().update(ctx)
    st.markdown(
        """
        <div class="cousp-detail-empty">
            <strong>Vue d’ensemble active</strong>
            La synthèse principale est affichée plus haut dans la page. Utilisez les filtres latéraux pour mettre à jour les KPI, cartes et graphiques, puis ouvrez un onglet détaillé pour approfondir l’analyse.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Les options avancées de visualisation sont regroupées dans la barre latérale. "
        "La cartographie détaillée est disponible dans son onglet dédié dès qu’elle est activée dans la sidebar."
    )

