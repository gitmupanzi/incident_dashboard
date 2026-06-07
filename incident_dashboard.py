from dashboard_app.runtime_support import build_runtime_context, inject_runtime_support
from dashboard_app.app_loader import (
    LINE_LIST_BUNDLE_LABEL,
    LINE_LIST_DIR,
    build_postgresql_query,
    get_excel_sheet_names_from_path,
    get_line_list_bundle_caption,
    guess_preferred_included_file,
    list_available_line_list_files,
    read_dhis2_tracker_file,
    read_postgresql_file,
)
from dashboard_app.column_mapping import (
    AUTO_APPLY_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DERIVED_COLUMNS,
    SOURCE_COLUMNS,
    STANDARD_COLUMNS,
    add_derived_columns_after_mapping,
    apply_auto_prefill_to_selection_state,
    auto_map_columns,
    build_auto_applied_mapping,
    build_mapping_preview_table,
    build_mapping_quality_report,
    build_mapping_warnings,
    dataframe_to_standardized_excel_bytes,
    list_mapping_profiles,
    load_mapping_profile,
    rename_dataframe_to_standard,
    save_mapping_profile,
    validate_mapping,
)
from dashboard_app.tabs.overview_detail import render_overview_detail_tab
from dashboard_app.overview import *
from dashboard_app.domain import *
from dashboard_app.tabs.statistics import render_statistics_tab
from dashboard_app.tabs.surveillance import render_surveillance_tab
from dashboard_app.tabs.profile import render_profile_tab
from dashboard_app.tabs.quality import render_quality_tab
from dashboard_app.tabs.cousp import render_cousp_tab
from dashboard_app.tabs.idsr import render_idsr_tab
from dashboard_app.tabs.irep import render_irep_tab
from dashboard_app.tabs.maps import render_maps_tab
from dashboard_app.tabs.methodology import render_methodology_tab
from dashboard_app.domain import     _resolve_map_filter_value

try:
    from dashboard_app.overview import format_range_label_for_display, split_geo_pair_label
except ImportError:
    def format_range_label_for_display(value):
        if value is None:
            return "-"
        text = str(value).strip()
        if not text:
            return "-"
        return re.sub(r"\s*->\s*", " -> ", text).replace(" -> ", " → ")

    def split_geo_pair_label(label_value):
        if label_value is None:
            return None, None
        text = str(label_value).strip()
        if not text:
            return None, None
        if " / " not in text:
            return None, text
        province_txt, zone_txt = [part.strip() for part in text.split(" / ", 1)]
        return (province_txt or None), (zone_txt or None)


inject_runtime_support(globals())

st.set_page_config(page_title="LL RDC - Dashboard", layout="wide")

inject_professional_dashboard_css()
render_professional_header()


# =========================
# SIDEBAR: INPUT
# =========================
st.sidebar.header("Source des données")

# ✅ Choix maladie (pour renommer/standardiser correctement)
disease_key = st.sidebar.selectbox(
    "Maladie / type de line list",
    options=list(DISEASE_SPECS.keys()),
    format_func=lambda k: DISEASE_SPECS.get(k, {}).get("label", k),
    index=0,
)

line_list_source = "upload"
upl = None
sheet_upl = ""
selected_local_name = ""
selected_local_path = None
postgres_host = "localhost"
postgres_port = 5432
postgres_database = ""
postgres_user = ""
postgres_password = ""
postgres_query_mode = "Table"
postgres_table_name = ""
postgres_sql_query = ""
dhis2_url = ""
dhis2_username = ""
dhis2_password = ""
dhis2_format_sortie = "json"
dhis2_connect_timeout = 30
dhis2_read_timeout = 900
dhis2_max_retries = 2
dhis2_ajouter_localisation_notification = True
dhis2_renommer_variable = True
dhis2_variables_brute = False

# Par défaut: feuille selon la maladie (modifiable)
default_sheet = DISEASE_SPECS.get(disease_key, DISEASE_SPECS["cholera"]).get("default_sheet", "")
disease_enabled = is_disease_enabled(disease_key)

if not disease_enabled:
    disabled_label = DISEASE_SPECS.get(disease_key, {}).get("label", disease_key)
    st.sidebar.warning(
        f"{disabled_label} : maladie désactivée. "
        "Le fichier peut être téléversé, mais aucune analyse ne sera exécutée."
    )

if disease_key != "idsr":
    source_options = {
        "T\u00e9l\u00e9verser un fichier": "upload",
        "Charger un fichier inclus": "local",
        "Se connecter \u00e0 DHIS2 Tracker": "dhis2",
        "Se connecter \u00e0 PostgreSQL": "postgres",
    }
    source_label = st.sidebar.selectbox(
        "Source de donn\u00e9es",
        options=list(source_options.keys()),
        index=0,
        key="ll_source_mode",
    )
    line_list_source = source_options[source_label]

    if line_list_source == "upload":
        upl = st.sidebar.file_uploader(
            "Fichier line list",
            type=["xlsx", "xls", "csv"],
            key="ll_upload"
        )
        sheet_upl = st.sidebar.text_input("Nom feuille (si Excel upload)", value=default_sheet)
    elif line_list_source == "local":
        available_line_list_files = list_available_line_list_files()
        if available_line_list_files:
            preferred_local_path = guess_preferred_included_file(
                available_line_list_files,
                disease_key=disease_key,
                default_sheet=default_sheet,
            )
            previous_disease_key = st.session_state.get("_ll_local_prev_disease_key")
            current_local_name = st.session_state.get("ll_local_file")
            available_local_names = [p.name for p in available_line_list_files]
            if (
                previous_disease_key != disease_key
                or current_local_name not in available_local_names
            ):
                st.session_state["ll_local_file"] = (
                    preferred_local_path.name if preferred_local_path is not None else available_local_names[0]
                )
            st.session_state["_ll_local_prev_disease_key"] = disease_key

            selected_local_name = st.sidebar.selectbox(
                "Fichier inclus",
                options=available_local_names,
                key="ll_local_file",
            )
            selected_local_path = next((p for p in available_line_list_files if p.name == selected_local_name), None)
            if selected_local_path is not None:
                st.sidebar.caption(get_line_list_bundle_caption())
                if selected_local_path.suffix.lower() in {".xlsx", ".xls"}:
                    local_sheets = get_excel_sheet_names_from_path(selected_local_path)
                    local_default_sheet = default_sheet if default_sheet in local_sheets else local_sheets[0]
                    previous_sheet_file = st.session_state.get("_ll_local_sheet_file")
                    current_sheet_value = st.session_state.get("ll_local_sheet")
                    if (
                        previous_disease_key != disease_key
                        or previous_sheet_file != selected_local_path.name
                        or current_sheet_value not in local_sheets
                    ):
                        st.session_state["ll_local_sheet"] = local_default_sheet
                    st.session_state["_ll_local_sheet_file"] = selected_local_path.name
                    sheet_upl = st.sidebar.text_input(
                        "Nom feuille (si Excel local)",
                        key="ll_local_sheet",
                    )
        else:
            st.sidebar.warning("Aucun fichier `.xlsx`, `.xls` ou `.csv` inclus dans l'application n'a \u00e9t\u00e9 trouv\u00e9.")
    elif line_list_source == "dhis2":
        st.sidebar.caption("Connexion à DHIS2 Tracker")
        dhis2_url = st.sidebar.text_area(
            "URL DHIS2 Tracker",
            value="",
            height=160,
            key="ll_dhis2_url",
            help="Colle ici l'URL analytics/enrollments .json ou .csv.",
        )
        dhis2_username = st.sidebar.text_input("Utilisateur DHIS2", value="", key="ll_dhis2_user")
        dhis2_password = st.sidebar.text_input("Mot de passe DHIS2", value="", type="password", key="ll_dhis2_password")
        dhis2_format_sortie = st.sidebar.radio(
            "Format de sortie DHIS2",
            ["json", "csv"],
            index=0,
            key="ll_dhis2_format",
            horizontal=True,
        )
        dhis2_connect_timeout = st.sidebar.number_input(
            "Timeout connexion (s)",
            min_value=1,
            max_value=300,
            value=30,
            step=1,
            key="ll_dhis2_connect_timeout",
        )
        dhis2_read_timeout = st.sidebar.number_input(
            "Timeout lecture (s)",
            min_value=1,
            max_value=3600,
            value=900,
            step=30,
            key="ll_dhis2_read_timeout",
        )
        dhis2_max_retries = st.sidebar.number_input(
            "Nombre de tentatives",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            key="ll_dhis2_max_retries",
        )
        dhis2_ajouter_localisation_notification = st.sidebar.checkbox(
            "Ajouter la localisation de notification",
            value=True,
            key="ll_dhis2_add_notification_location",
        )
        dhis2_renommer_variable = st.sidebar.checkbox(
            "Renommer / standardiser les variables",
            value=True,
            key="ll_dhis2_rename_variables",
            help="Utilise par défaut `data/Rename_columns.xlsx` comme référence.",
        )
        dhis2_variables_brute = st.sidebar.checkbox(
            "Conserver les variables brutes DHIS2",
            value=False,
            key="ll_dhis2_keep_raw_variables",
        )
    else:
        st.sidebar.caption("Connexion à une base PostgreSQL")
        postgres_host = st.sidebar.text_input("Hôte PostgreSQL", value="localhost", key="ll_pg_host")
        postgres_port = st.sidebar.number_input(
            "Port PostgreSQL",
            min_value=1,
            max_value=65535,
            value=5432,
            step=1,
            key="ll_pg_port",
        )
        postgres_database = st.sidebar.text_input("Base de données", value="", key="ll_pg_database")
        postgres_user = st.sidebar.text_input("Utilisateur", value="", key="ll_pg_user")
        postgres_password = st.sidebar.text_input("Mot de passe", value="", type="password", key="ll_pg_password")
        postgres_query_mode = st.sidebar.radio(
            "Mode de lecture",
            ["Table", "Requête SQL"],
            index=0,
            key="ll_pg_query_mode",
        )
        if postgres_query_mode == "Table":
            postgres_table_name = st.sidebar.text_input("Nom de la table", value="", key="ll_pg_table")
        else:
            postgres_sql_query = st.sidebar.text_area(
                "Requête SQL",
                value="SELECT * FROM public.line_list",
                height=120,
                key="ll_pg_sql",
            )
else:
    st.sidebar.info(
        "Mode **IDSR agrégé (hebdo)** : le chargement du fichier et les analyses se font "
        "uniquement dans l’onglet **IDSR**."
    )

    # En mode IDSR, on ne force pas une line list
    sheet_upl = default_sheet


supp_doublons = st.sidebar.checkbox("Supprimer les doublons (simple)", value=False)
show_maps = st.sidebar.checkbox(
    "Activer l’onglet cartographie détaillée",
    value=False,
    key="show_maps",
    help="Affiche les cartes détaillées dans l’onglet Cartographie.",
)

st.sidebar.header("Période")
year_filter_slot = st.sidebar.container()
use_week_filter = st.sidebar.checkbox(
    "Filtrer sur la semaine épidémiologique",
    value=True,
    help="Activé par défaut. La fenêtre initiale couvre toute l'année épidémiologique (semaine 1 à 53).",
)
week_min = st.sidebar.number_input(
    "Semaine min",
    min_value=1,
    max_value=53,
    value=1,
    step=1,
    disabled=not use_week_filter,
)
week_max = st.sidebar.number_input(
    "Semaine max",
    min_value=1,
    max_value=53,
    value=53,
    step=1,
    disabled=not use_week_filter,
)

st.sidebar.header("Seuil timeliness")
seuil_jours = st.sidebar.number_input("Seuil (jours) pour % sous seuil", min_value=0, max_value=30, value=2, step=1)

def _reset_display_options() -> None:
    st.session_state["show_maps"] = False
    st.session_state["use_custom_viz"] = True
    st.session_state["annot_vals"] = False
    st.session_state["pas_x"] = 1
    st.session_state["seuil_min_count"] = 0
    st.session_state["show_sidebar_summary"] = True
    st.session_state["overview_pyramid_age_mode"] = "5yr"

st.sidebar.header("Visualisations")
if "overview_pyramid_age_mode" not in st.session_state:
    st.session_state["overview_pyramid_age_mode"] = "5yr"
with st.sidebar.expander("Paramètres avancés des visualisations", expanded=False):
    use_custom_viz = st.checkbox(
        "Utiliser visualisations custom (dataminsante)",
        value=True,
        key="use_custom_viz",
        help="Les fonctions custom sont intégrées dans le package applicatif du dashboard."
    )
    pyramid_age_mode = st.selectbox(
        "Decoupage age (tous les graphiques)",
        options=list(AGE_PYRAMID_MODE_LABELS.keys()),
        index=list(AGE_PYRAMID_MODE_LABELS.keys()).index(st.session_state.get("overview_pyramid_age_mode", "5yr")),
        key="overview_pyramid_age_mode",
        format_func=lambda key: AGE_PYRAMID_MODE_LABELS.get(key, key),
        help="Applique le meme decoupage a la pyramide age-sexe et aux graphiques lies a l'age.",
    )
    annot_vals = st.checkbox("Afficher annotations (valeurs)", value=False, key="annot_vals")
    pas_x = st.number_input("Pas X (ticks)", min_value=1, max_value=10, value=1, step=1, key="pas_x")
    seuil_min_count = st.number_input("Seuil minimal (filtrer petits groupes)", min_value=0, max_value=100, value=0, step=1, key="seuil_min_count")
    st.button("Réinitialiser les options d’affichage", key="reset_display_options", on_click=_reset_display_options)

pyramid_age_mode = str(st.session_state.get("overview_pyramid_age_mode", "5yr"))



show_sidebar_summary = st.sidebar.checkbox(
    "Afficher le résumé des filtres actifs",
    value=False,
    key="show_sidebar_summary",
    help="Affiche dans la barre latérale un résumé du périmètre courant et des filtres appliqués.",
)


# =========================
# LOAD
# =========================
IDSR_MODE = (disease_key == "idsr")

def build_overview_age_sex_pyramid_figure(
    df_: pd.DataFrame,
    pyramid_age_mode: str,
    use_custom_viz_flag: bool,
) -> tuple[object | None, str | None]:
    df_pyr = df_.copy()
    pyramid_age_col = get_age_pyramid_group_column_name(pyramid_age_mode)
    df_pyr[pyramid_age_col] = derive_age_pyramid_generic(df_pyr, pyramid_age_mode)

    has_age = pyramid_age_col in df_pyr.columns and df_pyr[pyramid_age_col].notna().any()
    has_sex = COL_SEX in df_pyr.columns and df_pyr[COL_SEX].notna().any()
    if not (has_age and has_sex):
        return None, "data_missing"

    if use_custom_viz_flag and HAS_CUSTOM_VIZ:
        fig_pyr = plot_pyramide_symetrique(
            df=df_pyr,
            col_categorie=pyramid_age_col,
            col_groupe=COL_SEX,
            valeurs_neg=["Masculin", "Homme", "M"],
            titre=None,
            seuil_min=0,
            croissant=True,
            afficher_signe_negatif_dans_label=False,
        )
        if fig_pyr is None:
            return None, "render_failed"
        fig_pyr.update_layout(
            height=430,
            margin=dict(t=18, b=44, l=72, r=56),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            uniformtext_minsize=8,
            uniformtext_mode="hide",
            barmode="relative",
        )
        fig_pyr.update_xaxes(automargin=True)
        fig_pyr.update_yaxes(automargin=True)
        return fig_pyr, None

    sex_display_map = {
        "feminin": "Feminin",
        "féminin": "Feminin",
        "f": "Feminin",
        "female": "Feminin",
        "femme": "Feminin",
        "masculin": "Masculin",
        "m": "Masculin",
        "male": "Masculin",
        "homme": "Masculin",
    }
    work = df_pyr[[pyramid_age_col, COL_SEX]].dropna().copy()
    work["_Sexe_dashboard"] = work[COL_SEX].apply(
        lambda value: sex_display_map.get(str(value).strip().lower(), str(value).strip())
    )
    age_order = get_age_pyramid_category_order(pyramid_age_mode)
    extra_age_values = [
        value for value in work[pyramid_age_col].astype(str).unique().tolist()
        if value not in age_order
    ]
    age_order = age_order + extra_age_values
    work[pyramid_age_col] = pd.Categorical(work[pyramid_age_col], categories=age_order, ordered=True)
    counts = (
        work.groupby([pyramid_age_col, "_Sexe_dashboard"], observed=True)
        .size()
        .reset_index(name="n")
    )
    pivot = (
        counts.pivot_table(
            index=pyramid_age_col,
            columns="_Sexe_dashboard",
            values="n",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reindex(age_order)
        .fillna(0)
    )

    fig_pyr = go.Figure()
    for group_name, direction in [("Masculin", -1), ("Feminin", 1)]:
        if group_name not in pivot.columns:
            continue
        values = pd.to_numeric(pivot[group_name], errors="coerce").fillna(0)
        fig_pyr.add_trace(
            go.Bar(
                y=age_order,
                x=(values * direction).tolist(),
                orientation="h",
                name=group_name,
                marker=dict(color=SEX_COLOR_MAP.get(group_name, "#4c78a8")),
                text=values.astype(int).astype(str).tolist(),
                texttemplate="%{text}",
                textposition="outside",
                customdata=values.astype(int).tolist(),
                hovertemplate=f"Sexe: {group_name}<br>Tranche d'âge: %{{y}}<br>Nombre de cas: %{{customdata}}<extra></extra>",
                cliponaxis=False,
            )
        )

    max_val = int(max(abs(v) for tr in fig_pyr.data for v in tr.x)) if fig_pyr.data else 0
    axis_max = max(1, int(np.ceil(max_val * 1.08)))
    fig_pyr.update_layout(
        height=430,
        margin=dict(t=18, b=44, l=72, r=56),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        barmode="relative",
        template="plotly_white",
        xaxis=dict(
            tickvals=[-max_val, 0, max_val],
            ticktext=[str(max_val), "0", str(max_val)],
            automargin=True,
            zeroline=True,
            zerolinewidth=1.2,
            zerolinecolor="rgba(26,30,43,0.28)",
            range=[-axis_max, axis_max],
            title="Nombre de cas",
        ),
        yaxis=dict(categoryorder="array", categoryarray=age_order, title="Tranche d'âge"),
    )
    fig_pyr.update_xaxes(automargin=True)
    fig_pyr.update_yaxes(automargin=True)
    return fig_pyr, "fallback"

def render_overview_dashboard_v2(
    df_: pd.DataFrame,
    files_used: list[str],
    disease_key: str,
    use_custom_viz_flag: bool,
    pyramid_age_mode: str,
    annotate_values_flag: bool,
    x_tick_step: int,
) -> None:
    """Version enrichie de la page d'accueil inspirée d'un briefing institutionnel."""
    if df_.empty:
        st.info("Aucune donnée filtrée n'est disponible pour la synthèse d'accueil.")
        return

    payload = build_dashboard_kpi_payload(df_)
    render_context_row(files_used, disease_key, df_, payload)
    render_dashboard_kpis(payload)
    render_standards_note()

    weekly = payload.get("weekly", pd.DataFrame())
    map_state_key = "overview_v2_show_maps"
    if map_state_key not in st.session_state:
        st.session_state[map_state_key] = False
    show_overview_maps = bool(st.session_state.get(map_state_key, False))
    overview_province_map_mode = "Statique"
    overview_map_mode_label = list(MAP_ANNOTATION_MODE_OPTIONS.keys())[0]
    overview_map_threshold = 1

    with st.expander("Options des cartes de synthèse", expanded=False):
        st.caption("Par défaut, les cartes ne sont pas chargées pour accélérer l'ouverture du tableau de bord.")
        action_col1, action_col2 = st.columns([0.8, 1.2])
        with action_col1:
            if show_overview_maps:
                if st.button("Masquer les cartes", key="overview_hide_maps_v2"):
                    st.session_state[map_state_key] = False
                    st.rerun()
            else:
                if st.button("Afficher les cartes", key="overview_show_maps_v2"):
                    st.session_state[map_state_key] = True
                    st.rerun()
        with action_col2:
            st.write(f"État actuel : **{'cartes chargées' if show_overview_maps else 'cartes masquées'}**")

        if show_overview_maps:
            overview_province_map_mode = st.radio(
                "Carte province de synthèse",
                ["Statique", "Interactive"],
                index=0,
                horizontal=True,
                key="overview_province_map_mode_v2",
            )
            overview_map_mode_label = st.selectbox(
                "Annotations sur les cartes de synthèse",
                options=list(MAP_ANNOTATION_MODE_OPTIONS.keys()),
                index=0,
                key="overview_map_annotation_mode_v2",
            )
            overview_map_threshold = st.number_input(
                "Seuil d'affichage des annotations (valeur >)",
                min_value=0,
                max_value=100000,
                value=1,
                step=1,
                key="overview_map_annotation_threshold_v2",
            )

    overview_map_mode = MAP_ANNOTATION_MODE_OPTIONS[overview_map_mode_label]
    fig_map_prov = None
    gdf_map_prov = None
    df_match_prov = None
    note_map_prov = "Cartographie non chargée."
    value_col_prov = None
    group_col_prov = None
    fig_map_zs = None
    note_map_zs = "Cartographie non chargée."
    if show_overview_maps:
        province_map_payload = prepare_overview_map_data(df_, level="province", match_threshold=0.90)
        gdf_map_prov, df_match_prov, note_map_prov, value_col_prov, group_col_prov, _ = province_map_payload
        if overview_province_map_mode == "Statique":
            fig_map_prov, note_map_prov = build_static_map_overview(
                df_,
                level="province",
                annotation_mode=overview_map_mode,
                annotation_threshold=float(overview_map_threshold),
            )
        fig_map_zs, note_map_zs = build_static_map_overview(
            df_,
            level="zone",
            annotation_mode=overview_map_mode,
            annotation_threshold=float(overview_map_threshold),
        )

    c1, c2, c3 = st.columns([1.05, 1.35, 1.35])
    with c1:
        st.markdown("<div class='cousp-panel-title'>Indicateurs clés de la semaine</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='cousp-summary-box'><div class='summary-lead'>Bloc de synthèse opérationnelle</div></div>",
            unsafe_allow_html=True,
        )
        w1, w2 = st.columns(2)
        latest = payload.get("latest", {})
        previous = payload.get("previous", {})
        with w1:
            st.metric(
                "Cas semaine",
                format_metric_value(latest.get("Cas", np.nan)),
                format_pct_delta(latest.get("Cas", np.nan), previous.get("Cas", np.nan)),
            )
            st.metric(
                "CFR semaine",
                format_metric_value(latest.get("Létalité (%)", np.nan), decimals=2, suffix="%"),
                format_pct_delta(latest.get("Létalité (%)", np.nan), previous.get("Létalité (%)", np.nan)),
            )
        with w2:
            st.metric(
                "Décès semaine",
                format_metric_value(latest.get("Décès", np.nan)),
                format_pct_delta(latest.get("Décès", np.nan), previous.get("Décès", np.nan)),
            )
            st.metric(
                f"Admission <= {get_session_int('seuil_jours', 2)} jours",
                format_metric_value(payload.get("promptitude_pct"), decimals=1, suffix="%"),
                f"n={payload.get('promptitude_n', 0)}",
            )

        st.write(f"Province la plus notifiée : **{payload.get('top_province', 'non disponible')}**")
        st.write(f"Zone de santé la plus notifiée : **{payload.get('top_zs', 'non disponible')}**")
        st.write(
            f"Fenêtre couverte : **{format_range_label_for_display(payload.get('week_span', '-'))}** avec **{format_metric_value(payload.get('cases', 0))}** cas analysés."
        )
    with c2:
        if not show_overview_maps:
            st.info("Cartes non chargées. Utilisez `Options des cartes de synthèse` pour les afficher.")
        elif overview_province_map_mode == "Interactive":
            render_interactive_map_overview(
                "Carte interactive par province",
                gdf_join=gdf_map_prov,
                df_map=df_match_prov,
                note=note_map_prov,
                value_col=value_col_prov,
                source_df=df_,
                source_label_col=group_col_prov,
                chart_key="overview_province_map_v2",
                clicked_state_key="map_clicked_province",
                filter_state_key="prov_sel",
                height=540,
            )
        else:
            render_static_map_overview("Carte statique par province", fig_map_prov, note_map_prov)

    with c3:
        if not show_overview_maps:
            st.info("Cartes non chargées. Utilisez `Options des cartes de synthèse` pour les afficher.")
        else:
            render_static_map_overview("Carte statique par zone de santé", fig_map_zs, note_map_zs)

    render_delay_snapshot_panel(payload)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("<div class='cousp-panel-title'>Surveillance temporelle hebdomadaire</div>", unsafe_allow_html=True)
        if weekly.empty:
            st.info("Série hebdomadaire indisponible.")
        else:
            fig_surveillance = build_weekly_cases_deaths_combo(
                weekly_df=weekly,
                x_col="label",
                cases_col="Cas",
                deaths_col="Décès",
                titre=" ",
                x_titre="Semaine épidémiologique",
                y_titre_cas="Nombre de cas",
                y_titre_deces="Nombre de décès",
                rotation=0,
                annot_bars=annotate_values_flag,
                annot_line=annotate_values_flag,
            )
            if fig_surveillance is not None and x_tick_step > 1 and len(weekly) > x_tick_step:
                fig_surveillance.update_xaxes(
                    tickmode="array",
                    tickvals=weekly["label"].iloc[:: max(int(x_tick_step), 1)],
                    ticktext=weekly["label"].iloc[:: max(int(x_tick_step), 1)],
                )
            st_plot(fig_surveillance, key="overview_weekly_surveillance_v2", annotate_values=annotate_values_flag)

    with d2:
        st.markdown("<div class='cousp-panel-title'>Tendance hebdomadaire des cas et de la létalité observée</div>", unsafe_allow_html=True)
        if weekly.empty:
            st.info("Tendance hebdomadaire indisponible.")
        else:
            week_col = resolve_week_column(df_)
            fig_combo = build_weekly_cases_cfr_combo(
                df=df_,
                week_col=week_col,
                death_col="is_death",
                titre="",
                rotation=45,
                annot_bars=annotate_values_flag,
                annot_line=annotate_values_flag,
                pas_x=int(x_tick_step) if week_col in [COL_WNUM, "YW"] else None,
                taille_fig=(1400, 550),
            )
            st_plot(fig_combo, key="overview_weekly_combo_v2", annotate_values=annotate_values_flag)

    p1, p2, p3 = st.columns([1.3, 0.95, 1.15])
    with p1:
        st.markdown("<div class='cousp-panel-title'>Distribution géographique des notifications</div>", unsafe_allow_html=True)
        geo_col = COL_PROV if COL_PROV in df_.columns and df_[COL_PROV].notna().any() else COL_ZS
        if geo_col in df_.columns and df_[geo_col].notna().any():
            geo_tbl = build_frequency_table(df_, geo_col, top_n=10).sort_values("n", ascending=True)
            fig_geo = px.bar(
                geo_tbl,
                x="n",
                y=geo_col,
                orientation="h",
                text="n" if annotate_values_flag else None,
                color="n",
                color_continuous_scale=["#dbe8f9", "#2b74ca"],
                labels={geo_col: "Lieu", "n": "Nombre de cas"},
            )
            fig_geo.update_layout(coloraxis_showscale=False, title=" ")
            st_plot(fig_geo, key="overview_geo_distribution_v2", annotate_values=annotate_values_flag)
        else:
            st.info("Aucune variable géographique exploitable n'a été détectée.")

    with p2:
        st.markdown("<div class='cousp-panel-title'>Répartition par sexe</div>", unsafe_allow_html=True)
        if COL_SEX in df_.columns and df_[COL_SEX].notna().any():
            sex_tbl = build_frequency_table(df_, COL_SEX)
            fig_sex = px.pie(
                sex_tbl,
                names=COL_SEX,
                values="n",
                hole=0.62,
                color=COL_SEX,
                color_discrete_map=SEX_COLOR_MAP,
            )
            st_plot(fig_sex, key="overview_sex_pie_v2", annotate_values=annotate_values_flag)
        else:
            st.info("La variable Sexe est absente ou vide.")

    with p3:
        st.markdown("<div class='cousp-panel-title'>Courbe hebdomadaire des cas.</div>", unsafe_allow_html=True)
        if weekly.empty or "label" not in weekly.columns or "Cas" not in weekly.columns:
            st.info("Courbe hebdomadaire indisponible.")
        else:
            fig_age = px.line(
                weekly,
                x="label",
                y="Cas",
                markers=True,
                color_discrete_sequence=["#2d7d46"],
                title=" ",
                labels={"label": "Semaine épidémiologique", "Cas": "Nombre de cas"},
            )
            fig_age.update_traces(line=dict(width=3))
            fig_age.update_layout(xaxis_tickangle=-35)
            if x_tick_step > 1 and len(weekly) > x_tick_step:
                fig_age.update_xaxes(
                    tickmode="array",
                    tickvals=weekly["label"].iloc[:: max(int(x_tick_step), 1)],
                    ticktext=weekly["label"].iloc[:: max(int(x_tick_step), 1)],
                )
            st_plot(fig_age, key="overview_age_hist_v2", annotate_values=annotate_values_flag)

    p4, p5 = st.columns([1.55, 1.0])
    with p4:
        st.markdown("<div class='cousp-panel-title'>Pyramide âge-sexe</div>", unsafe_allow_html=True)
        fig_pyr, pyramid_status = build_overview_age_sex_pyramid_figure(
            df_=df_,
            pyramid_age_mode=pyramid_age_mode,
            use_custom_viz_flag=use_custom_viz_flag,
        )
        if fig_pyr is not None:
            if pyramid_status == "fallback":
                st.caption("Rendu standard actif : les visualisations custom sont desactivees.")
            st_plot(fig_pyr, key="overview_pyramid_v2", height=430, annotate_values=False)
        elif pyramid_status == "data_missing":
            st.info("Pyramide indisponible : variables Age/Sexe insuffisantes.")
        else:
            st.info("Pyramide indisponible : le rendu du graphique a echoue.")

    with p5:
        st.markdown("<div class='cousp-panel-title'>Distribution par tranche d'âge</div>", unsafe_allow_html=True)
        df_age_group = df_.copy()
        age_group_col = get_age_pyramid_group_column_name(pyramid_age_mode)
        df_age_group[age_group_col] = derive_age_pyramid_generic(df_age_group, pyramid_age_mode)

        if age_group_col in df_age_group.columns and df_age_group[age_group_col].notna().any():
            age_group_tbl = build_frequency_table(df_age_group, age_group_col)
            age_order = get_age_pyramid_category_order(pyramid_age_mode)
            extra_age_values = [value for value in age_group_tbl[age_group_col].astype(str).tolist() if value not in age_order]
            age_group_tbl[age_group_col] = pd.Categorical(
                age_group_tbl[age_group_col],
                categories=age_order + extra_age_values,
                ordered=True,
            )
            age_group_tbl = age_group_tbl.sort_values(age_group_col)
            fig_age_group = px.bar(
                age_group_tbl,
                x=age_group_col,
                y="n",
                text="n" if annotate_values_flag else None,
                color_discrete_sequence=["#d97b16"],
                title=" ",
            )
            fig_age_group.update_layout(xaxis_tickangle=-35)
            st_plot(fig_age_group, key="overview_age_group_v2", annotate_values=annotate_values_flag)
        else:
            st.info("Les classes d'âge ne sont pas disponibles.")

if not IDSR_MODE:
    source_ready = False
    source_message = ""
    if line_list_source == "upload":
        source_ready = upl is not None
        source_message = "Veuillez téléverser un fichier de données (`.xlsx` ou `.csv`) pour démarrer l’analyse de surveillance."
    elif line_list_source == "local":
        source_ready = selected_local_path is not None and selected_local_path.exists()
        source_message = "Aucun fichier inclus exploitable n'est disponible dans l'application."
    elif line_list_source == "dhis2":
        source_ready = bool(str(dhis2_url).strip())
        source_message = "Renseigne l'URL DHIS2 Tracker pour charger les données de surveillance."
    else:
        source_ready = all([
            str(postgres_host).strip(),
            str(postgres_database).strip(),
            str(postgres_user).strip(),
        ])
        source_message = "Renseigne les paramètres PostgreSQL pour charger les données de surveillance."

    if not source_ready:
        st.info(source_message)

        st.markdown(
            "<div class='cousp-panel-title'>Visualisations disponibles</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="cousp-detail-empty">
                <strong>Visualisations disponibles</strong>
                Une fois le fichier chargé, le tableau de bord permet d’explorer la situation
                épidémiologique sous plusieurs angles complémentaires : synthèse, évolution
                temporelle, distribution géographique et tableaux analytiques.
            </div>
            """,
            unsafe_allow_html=True,
        )

        welcome_cards = [
            ("Situation globale", "Vue synthétique des cas, des décès et des indicateurs clés de surveillance."),
            ("Évolution hebdomadaire", "Suivi de la dynamique des cas et lecture rapide des tendances."),
            ("Létalité par semaine", "Analyse de la gravité au fil du temps à partir des décès rapportés."),
            ("Répartition provinciale", "Comparaison rapide des provinces les plus touchées."),
            ("Analyse par zone de santé", "Lecture plus fine de la distribution spatiale des notifications."),
            ("Province × semaine", "Tableaux croisés pour relier dimensions géographique et temporelle."),
            ("Cartographie des cas", "Affichage disponible si les fichiers géographiques nécessaires sont présents."),
        ]

        for start in range(0, len(welcome_cards), 3):
            cols = st.columns(3)
            for col, (title, description) in zip(cols, welcome_cards[start:start + 3]):
                with col:
                    st.markdown(
                        f"""
                        <div class="cousp-context-chip" style="min-height: 132px; margin-bottom: 0.85rem;">
                            <div class="label">{title}</div>
                            <div class="value" style="font-size:0.95rem; font-weight:700; margin-top:0.45rem; line-height:1.35;">
                                {description}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.stop()

    try:
        if line_list_source == "upload":
            _bytes = upl.getvalue() if hasattr(upl, "getvalue") else None
            _md5 = hashlib.md5(_bytes).hexdigest() if _bytes is not None else None
            _cache_key = (
                "upload",
                upl.name,
                getattr(upl, "size", None),
                _md5,
                str(sheet_upl).strip() if sheet_upl is not None else "",
            )

            if st.session_state.get("_ll_cache_key") == _cache_key and isinstance(st.session_state.get("_ll_cache_raw"), pd.DataFrame):
                raw = st.session_state["_ll_cache_raw"]
            else:
                if upl.name.lower().endswith(".csv"):
                    raw = pd.read_csv(upl)
                else:
                    sh = sheet_upl.strip() if isinstance(sheet_upl, str) else ""
                    raw = pd.read_excel(upl, sheet_name=sh if sh else 0)
                st.session_state["_ll_cache_key"] = _cache_key
                st.session_state["_ll_cache_raw"] = raw

            files_used = [f"upload:{upl.name}"]
        elif line_list_source == "local":
            sh = sheet_upl.strip() if isinstance(sheet_upl, str) else ""
            _cache_key = (
                "local",
                str(selected_local_path.resolve()),
                selected_local_path.stat().st_mtime_ns,
                sh,
            )

            if st.session_state.get("_ll_cache_key") == _cache_key and isinstance(st.session_state.get("_ll_cache_raw"), pd.DataFrame):
                raw = st.session_state["_ll_cache_raw"]
            else:
                if selected_local_path.suffix.lower() == ".csv":
                    raw = pd.read_csv(selected_local_path)
                else:
                    raw = pd.read_excel(selected_local_path, sheet_name=sh if sh else 0)
                st.session_state["_ll_cache_key"] = _cache_key
                st.session_state["_ll_cache_raw"] = raw

            files_used = [f"bundle:{selected_local_path.name}"]
        elif line_list_source == "dhis2":
            _cache_key = (
                "dhis2",
                dhis2_url.strip(),
                dhis2_username.strip(),
                hashlib.sha256(str(dhis2_password).encode("utf-8")).hexdigest(),
                dhis2_format_sortie,
                int(dhis2_connect_timeout),
                int(dhis2_read_timeout),
                int(dhis2_max_retries),
                bool(dhis2_ajouter_localisation_notification),
                bool(dhis2_renommer_variable),
                bool(dhis2_variables_brute),
                disease_key,
            )

            if st.session_state.get("_ll_cache_key") == _cache_key and isinstance(st.session_state.get("_ll_cache_raw"), pd.DataFrame):
                raw = st.session_state["_ll_cache_raw"]
            else:
                raw = read_dhis2_tracker_file(
                    url=dhis2_url.strip(),
                    username=dhis2_username.strip(),
                    password=dhis2_password,
                    format_sortie=dhis2_format_sortie,
                    connect_timeout=int(dhis2_connect_timeout),
                    read_timeout=int(dhis2_read_timeout),
                    max_retries=int(dhis2_max_retries),
                    ajouter_localisation_notification=bool(dhis2_ajouter_localisation_notification),
                    renommer_variable=bool(dhis2_renommer_variable),
                    variables_brute=bool(dhis2_variables_brute),
                    disease_key=disease_key,
                )
                st.session_state["_ll_cache_key"] = _cache_key
                st.session_state["_ll_cache_raw"] = raw

            files_used = [f"dhis2:{dhis2_url.strip()}"]
        else:
            query = build_postgresql_query(postgres_query_mode, postgres_table_name, postgres_sql_query)
            _cache_key = (
                "postgres",
                postgres_host.strip(),
                int(postgres_port),
                postgres_database.strip(),
                postgres_user.strip(),
                hashlib.sha256(str(postgres_password).encode("utf-8")).hexdigest(),
                query,
            )

            if st.session_state.get("_ll_cache_key") == _cache_key and isinstance(st.session_state.get("_ll_cache_raw"), pd.DataFrame):
                raw = st.session_state["_ll_cache_raw"]
            else:
                raw = read_postgresql_file(
                    postgres_host.strip(),
                    int(postgres_port),
                    postgres_database.strip(),
                    postgres_user.strip(),
                    postgres_password,
                    query,
                )
                st.session_state["_ll_cache_key"] = _cache_key
                st.session_state["_ll_cache_raw"] = raw

            files_used = [f"postgres:{postgres_database.strip()}"]

    except Exception as e:
        if line_list_source == "upload":
            st.error(f"Impossible de lire le fichier téléversé : {e}")
        elif line_list_source == "local":
            st.error(f"Impossible de lire le fichier inclus sélectionné : {e}")
        elif line_list_source == "dhis2":
            st.error(f"Impossible de charger les données DHIS2 Tracker : {e}")
        else:
            st.error(f"Impossible de charger les données PostgreSQL : {e}")
        st.stop()

    if not disease_enabled:
        disease_label = DISEASE_SPECS.get(disease_key, {}).get("label", disease_key)
        st.warning(
            f"La maladie sélectionnée ({disease_label}) est actuellement désactivée. "
            "Le fichier a bien été téléversé, mais aucune analyse ne sera exécutée."
        )
        st.stop()

    if disease_key == "autre":
        mapping_placeholder = "-- Non associee --"
        profile_placeholder = "-- Aucun profil --"
        raw_column_options = list(raw.columns)
        mapping_threshold = AUTO_APPLY_CONFIDENCE_THRESHOLD
        mapping_cache_key = ("autre", _cache_key)
        mapping_token = hashlib.md5(repr(mapping_cache_key).encode("utf-8")).hexdigest()[:12]
        mapping_cache_state_key = "_ll_autre_mapping_cache_key"
        mapping_state_key = "_ll_autre_mapping"
        mapping_valid_state_key = "_ll_autre_mapping_validated"
        profile_name_state_key = "_ll_autre_mapping_profile_name"
        auto_mapping, auto_metadata = auto_map_columns(raw_column_options, threshold=mapping_threshold)
        auto_prefill_mapping = build_auto_applied_mapping(
            auto_metadata,
            threshold=mapping_threshold,
            include_derived=False,
        )

        if st.session_state.get(mapping_cache_state_key) != mapping_cache_key:
            st.session_state[mapping_cache_state_key] = mapping_cache_key
            st.session_state[mapping_state_key] = {}
            st.session_state[mapping_valid_state_key] = False
            st.session_state[profile_name_state_key] = ""
            for standard_name in SOURCE_COLUMNS:
                select_key = f"autre_map_{mapping_token}_{standard_name}"
                st.session_state[select_key] = auto_prefill_mapping.get(standard_name, mapping_placeholder)
            for standard_name in DERIVED_COLUMNS:
                select_key = f"autre_map_{mapping_token}_{standard_name}"
                st.session_state[select_key] = mapping_placeholder
        else:
            current_source_state = {
                standard_name: st.session_state.get(f"autre_map_{mapping_token}_{standard_name}", mapping_placeholder)
                for standard_name in SOURCE_COLUMNS
            }
            updated_source_state = apply_auto_prefill_to_selection_state(
                current_source_state,
                auto_prefill_mapping,
                mapping_placeholder,
            )
            for standard_name, selected_value in updated_source_state.items():
                st.session_state[f"autre_map_{mapping_token}_{standard_name}"] = selected_value

        with st.expander("Correspondance des colonnes pour 'Autre'", expanded=True):
            st.caption(
                "Le dashboard propose une correspondance automatique des colonnes. "
                "Valide ou corrige les associations avant de lancer l'analyse."
            )
            st.dataframe(
                pd.DataFrame({"Colonnes detectees dans le fichier": [str(col) for col in raw_column_options]}),
                width="stretch",
                hide_index=True,
            )
            available_profiles = [profile_placeholder, *list_mapping_profiles()]
            profile_col1, profile_col2 = st.columns([3, 1])
            selected_profile = profile_col1.selectbox(
                "Charger un profil de mapping",
                options=available_profiles,
                key=f"autre_mapping_profile_select_{mapping_token}",
            )
            load_profile_clicked = profile_col2.button(
                "Charger",
                key=f"autre_mapping_profile_load_{mapping_token}",
            )
            if load_profile_clicked and selected_profile != profile_placeholder:
                loaded_profile = load_mapping_profile(selected_profile)
                loaded_mapping = loaded_profile.get("mapping", {}) or {}
                missing_source_columns = []
                for standard_name in STANDARD_COLUMNS:
                    select_key = f"autre_map_{mapping_token}_{standard_name}"
                    profile_source = loaded_mapping.get(standard_name)
                    if profile_source in raw_column_options:
                        st.session_state[select_key] = profile_source
                    else:
                        if standard_name in SOURCE_COLUMNS:
                            st.session_state[select_key] = auto_prefill_mapping.get(standard_name, mapping_placeholder)
                        else:
                            st.session_state[select_key] = mapping_placeholder
                        if profile_source:
                            missing_source_columns.append(f"{standard_name}: {profile_source}")
                st.session_state[mapping_state_key] = {}
                st.session_state[mapping_valid_state_key] = False
                st.session_state[profile_name_state_key] = loaded_profile.get("profile_name", "")
                if missing_source_columns:
                    st.warning(
                        "Certaines colonnes du profil charge ne sont pas presentes dans le fichier courant : "
                        + "; ".join(missing_source_columns[:8])
                    )
                else:
                    st.success("Profil de mapping charge. Verifie puis valide les associations.")

            selectable_columns = [mapping_placeholder, *raw_column_options]
            st.markdown("**Variables sources a associer**")
            for standard_name, meta in SOURCE_COLUMNS.items():
                select_key = f"autre_map_{mapping_token}_{standard_name}"
                if select_key not in st.session_state:
                    st.session_state[select_key] = auto_prefill_mapping.get(standard_name, mapping_placeholder)
                suggestion_meta = auto_metadata.get(standard_name, {})
                suggestion_label = (
                    f"Suggestion: {suggestion_meta.get('source_column')} "
                    f"({suggestion_meta.get('method')}, score={suggestion_meta.get('confidence')})"
                    if suggestion_meta.get("source_column") is not None
                    else "Aucune suggestion automatique"
                )
                st.selectbox(
                    f"{standard_name}{' *' if meta.get('required') else ''}",
                    options=selectable_columns,
                    key=select_key,
                    help=f"{meta.get('role', '')}. {suggestion_label}",
                )

            st.markdown("**Variables derivees ou de secours**")
            st.caption(
                "Ces colonnes peuvent etre mappees si elles existent deja dans le fichier, "
                "mais elles sont aussi calculables automatiquement quand les colonnes sources sont disponibles."
            )
            for standard_name, meta in DERIVED_COLUMNS.items():
                select_key = f"autre_map_{mapping_token}_{standard_name}"
                if select_key not in st.session_state:
                    st.session_state[select_key] = mapping_placeholder
                suggestion_meta = auto_metadata.get(standard_name, {})
                suggestion_label = (
                    f"Suggestion: {suggestion_meta.get('source_column')} "
                    f"({suggestion_meta.get('method')}, score={suggestion_meta.get('confidence')})"
                    if suggestion_meta.get("source_column") is not None
                    else "Calculable automatiquement"
                )
                st.selectbox(
                    f"{standard_name} (optionnel)",
                    options=selectable_columns,
                    key=select_key,
                    help=f"{meta.get('role', '')}. {suggestion_label}",
                )

            pending_mapping = {}
            for standard_name in STANDARD_COLUMNS:
                select_key = f"autre_map_{mapping_token}_{standard_name}"
                selected_value = st.session_state.get(select_key, mapping_placeholder)
                pending_mapping[standard_name] = None if selected_value == mapping_placeholder else selected_value

            if (
                st.session_state.get(mapping_valid_state_key)
                and pending_mapping != st.session_state.get(mapping_state_key, {})
            ):
                st.session_state[mapping_valid_state_key] = False

            preview_df = build_mapping_preview_table(
                pending_mapping,
                auto_metadata,
                threshold=mapping_threshold,
            )
            st.markdown("**Apercu du mapping avant validation**")
            st.dataframe(
                preview_df[
                    [
                        "Variable standard",
                        "Type de variable",
                        "Colonne source proposée",
                        "Méthode de détection",
                        "Score de confiance",
                        "Statut",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

            for warning_msg in build_mapping_warnings(pending_mapping):
                st.warning(warning_msg)

            mapping_submitted = st.button(
                "Valider la correspondance",
                key=f"autre_mapping_validate_{mapping_token}",
            )
            validation_errors = []
            if mapping_submitted:
                is_valid_mapping, validation_errors = validate_mapping(pending_mapping)
                st.session_state[mapping_state_key] = pending_mapping
                st.session_state[mapping_valid_state_key] = is_valid_mapping
                if validation_errors:
                    for err in validation_errors:
                        st.error(err)
                else:
                    st.success("Correspondance validee. L'analyse peut maintenant continuer.")
            elif st.session_state.get(mapping_valid_state_key):
                st.success("Correspondance deja validee pour le fichier courant.")

            current_valid_mapping = st.session_state.get(mapping_state_key, {})
            standardized_preview_df = None
            derived_info = {}
            quality_report = None
            if st.session_state.get(mapping_valid_state_key) and current_valid_mapping:
                standardized_preview_df = rename_dataframe_to_standard(
                    raw,
                    current_valid_mapping,
                    keep_unmapped_columns=True,
                )
                standardized_preview_df, derived_info = add_derived_columns_after_mapping(
                    standardized_preview_df,
                    return_info=True,
                )
                derived_info["original_columns"] = raw_column_options
                quality_report = build_mapping_quality_report(
                    standardized_preview_df,
                    current_valid_mapping,
                    derived_info=derived_info,
                )

                st.markdown("**Rapport qualite apres mapping**")
                st.write(f"Nombre de lignes : **{quality_report.get('Nombre de lignes', 0):,}**".replace(",", " "))
                st.write(
                    f"Colonnes reconnues : **{quality_report.get('Nombre de colonnes standards reconnues', 0)}**"
                )
                st.write(
                    f"Colonnes non reconnues : **{quality_report.get('Nombre de colonnes non reconnues', 0)}**"
                )
                dates_valides = quality_report.get("Dates valides", {})
                st.write(
                    f"Dates valides : **{dates_valides.get('valid', 0)} / {dates_valides.get('total', 0)}**"
                )
                ages_valides = quality_report.get("Âges valides", {})
                st.write(
                    f"Âges valides : **{ages_valides.get('valid', 0)} / {ages_valides.get('total', 0)}**"
                )
                st.write(
                    f"Semaines epidemiologiques calculees : **{quality_report.get('Semaines épidémiologiques calculées', 0)}**"
                )
                st.write(
                    f"Tranches d'age calculees : **{quality_report.get('Tranches d’âge calculées', 0)}**"
                )

                profile_name = st.text_input(
                    "Nom du profil de mapping a sauvegarder (optionnel)",
                    key=profile_name_state_key,
                    placeholder="autre_cholera_labo",
                )
                if st.button(
                    "Sauvegarder ce profil",
                    key=f"autre_mapping_save_{mapping_token}",
                ):
                    if not str(profile_name).strip():
                        st.warning("Renseigne un nom de profil avant la sauvegarde.")
                    else:
                        save_path = save_mapping_profile(
                            current_valid_mapping,
                            profile_name=profile_name,
                            metadata={
                                "disease_key": disease_key,
                                "source": "manual_validation",
                                "columns_count": len(raw_column_options),
                            },
                        )
                        st.success(f"Profil sauvegarde : {save_path.name}")

                st.download_button(
                    "Telecharger le fichier standardise",
                    data=dataframe_to_standardized_excel_bytes(standardized_preview_df),
                    file_name="liste_lineaire_standardisee.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"autre_mapping_download_{mapping_token}",
                )

        if not st.session_state.get(mapping_valid_state_key):
            st.info("Valide la correspondance des colonnes pour poursuivre l'analyse de l'option 'Autre'.")
            st.stop()

        raw = rename_dataframe_to_standard(
            raw,
            st.session_state.get(mapping_state_key, pending_mapping),
            keep_unmapped_columns=True,
        )
        raw = add_derived_columns_after_mapping(raw)

    # ✅ 1) Standardisation commune (Rougeole/Choléra/…)
    raw = standardize_ll_by_disease(raw, disease_key)
    df_quality_source = raw.copy()

    # ✅ 2) Standardisation spécifique choléra (les indicateurs/timeliness/etc.)
    df = standardize_df(raw)

    # Filtre Année (sidebar > Période)
    years_selected_main = []
    if COL_YEAR in df.columns:
        years_available_main = (
            pd.to_numeric(df[COL_YEAR], errors="coerce")
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )
        if years_available_main:
            # Initialiser une seule fois avec toutes les années, puis conserver l'état utilisateur.
            if "year_sel_main" not in st.session_state:
                st.session_state["year_sel_main"] = years_available_main.copy()
            else:
                # Nettoyer l'état si la liste des années disponibles change.
                st.session_state["year_sel_main"] = [
                    y for y in st.session_state["year_sel_main"] if y in years_available_main
                ]

            years_selected_main = year_filter_slot.multiselect(
                "Année",
                options=years_available_main,
                key="year_sel_main",
                placeholder="Toutes les années",
                help="Si aucune année n'est sélectionnée, le filtre Année n'est pas appliqué."
            )
            if years_selected_main:
                df = df[pd.to_numeric(df[COL_YEAR], errors="coerce").isin(years_selected_main)]
        else:
            year_filter_slot.info("Aucune année exploitable trouvée.")

    # Filtre semaine
    if use_week_filter and COL_WNUM in df.columns:
        week_values = pd.to_numeric(df[COL_WNUM], errors="coerce").dropna()
        if not week_values.empty:
            available_week_min = int(week_values.min())
            available_week_max = int(week_values.max())
            selected_week_min = max(int(week_min), available_week_min)
            selected_week_max = min(int(week_max), available_week_max)

            if selected_week_min > selected_week_max:
                st.warning(
                    f"Filtre semaine invalide pour les données courantes. Plage disponible : "
                    f"{available_week_min} à {available_week_max}."
                )
                df = df.iloc[0:0]
            else:
                df = df[df[COL_WNUM].between(selected_week_min, selected_week_max)]

    # Doublons (simple)
    if supp_doublons:
        key_cols = [c for c in ["Semaine_epid","Nom_complet",COL_SEX,COL_AGE,COL_PROV, COL_ZS,COL_UNIT, "Profession"] if c in df.columns]
        if key_cols:
            df = df.drop_duplicates(subset=key_cols, keep="first")

    df = df.copy()
    age_col_auto = pick_age_col(df)

    with st.expander("Source analytique et fichiers utilisés", expanded=False):
        st.write(files_used[:200])
        disease_label = DISEASE_SPECS.get(disease_key, {}).get("label", disease_key)
        st.write(f"Périmètre courant : **{disease_label}**")
        st.write(
            "Les KPI et visuels de la page d'accueil sont calculés après application des filtres temporels et géographiques."
        )
    # =========================
    # FILTERS (UI) - MULTISELECT DÉPENDANTS AVEC "Toutes" PAR DÉFAUT
    # =========================
    st.sidebar.header("Filtres géographiques")

    clicked_province = st.session_state.pop("map_clicked_province", None)
    clicked_zone = st.session_state.pop("map_clicked_zone", None)

    # ---- Init state ----
    if "prov_sel" not in st.session_state:
        st.session_state["prov_sel"] = ["Toutes"]
    if "zs_sel" not in st.session_state:
        st.session_state["zs_sel"] = ["Toutes"]
    if "as_sel" not in st.session_state:
        st.session_state["as_sel"] = ["Toutes"]
    if "class_sel" not in st.session_state:
        st.session_state["class_sel"] = ["Toutes"]

    if clicked_province and COL_PROV in df.columns:
        selected_prov = _resolve_map_filter_value(clicked_province, df[COL_PROV].dropna().unique().tolist())
        if selected_prov:
            st.session_state["prov_sel"] = [selected_prov]
            st.session_state["zs_sel"] = ["Toutes"]
            st.session_state["as_sel"] = ["Toutes"]

    if clicked_zone and COL_ZS in df.columns:
        selected_prov = None
        zone_label = clicked_zone
        if COL_PROV in df.columns:
            clicked_prov_label, clicked_zone_label = split_geo_pair_label(clicked_zone)
            if clicked_zone_label:
                zone_label = clicked_zone_label
            if clicked_prov_label:
                selected_prov = _resolve_map_filter_value(clicked_prov_label, df[COL_PROV].dropna().unique().tolist())

        selected_zone = _resolve_map_filter_value(zone_label, df[COL_ZS].dropna().unique().tolist())
        if selected_zone:
            st.session_state["zs_sel"] = [selected_zone]
            st.session_state["as_sel"] = ["Toutes"]
            if selected_prov:
                st.session_state["prov_sel"] = [selected_prov]
            elif COL_PROV in df.columns:
                zone_key = _norm_key(selected_zone)
                province_candidates = (
                    df.loc[
                        df[COL_ZS].astype(str).map(_norm_key) == zone_key,
                        COL_PROV,
                    ]
                    .dropna()
                    .astype(str)
                    .tolist()
                )
                province_candidates = [p for p in province_candidates if p]
                if len(set(province_candidates)) == 1:
                    st.session_state["prov_sel"] = [province_candidates[0]]
                else:
                    st.session_state["prov_sel"] = ["Toutes"]

    # ---- Bouton reset ----
    if st.sidebar.button("Réinitialiser les filtres géographiques"):
        st.session_state["prov_sel"] = ["Toutes"]
        st.session_state["zs_sel"] = ["Toutes"]
        st.session_state["as_sel"] = ["Toutes"]
        st.session_state["class_sel"] = ["Toutes"]
        st.rerun()

    df0 = df.copy()  # base (non filtré)

    def normalize_sel(state_key: str, options: list[str]):
        """
        - Garde seulement les valeurs valides
        - Si l'utilisateur a des choix spécifiques -> enlève "Toutes"
        - Si vide -> remet ["Toutes"]
        """
        sel = st.session_state.get(state_key, ["Toutes"])
        sel = [x for x in sel if x in options]

        if any(x != "Toutes" for x in sel):
            sel = [x for x in sel if x != "Toutes"]

        if len(sel) == 0:
            sel = ["Toutes"]

        st.session_state[state_key] = sel
        return sel

    # =========================
    # Province (multiselect)
    # =========================
    df1 = df0.copy()
    if COL_PROV in df0.columns:
        prov_options = ["Toutes"] + sorted([x for x in df0[COL_PROV].dropna().unique().tolist() if x])
        normalize_sel("prov_sel", prov_options)

        prov_sel = st.sidebar.multiselect(
            "Province (notification)",
            options=prov_options,
            key="prov_sel",
        )

        if prov_sel and ("Toutes" not in prov_sel):
            df1 = df1[df1[COL_PROV].isin(prov_sel)]

    # =========================
    # Zone de santé (multiselect, dépend de Province)
    # =========================
    df2 = df1.copy()
    if COL_ZS in df1.columns:
        zs_options = ["Toutes"] + sorted([x for x in df1[COL_ZS].dropna().unique().tolist() if x])
        normalize_sel("zs_sel", zs_options)

        zs_sel = st.sidebar.multiselect(
            "Zone de santé (notification)",
            options=zs_options,
            key="zs_sel",
        )

        if zs_sel and ("Toutes" not in zs_sel):
            df2 = df2[df2[COL_ZS].isin(zs_sel)]

    # =========================
    # Aire de santé (multiselect, dépend de Province + ZS)
    # =========================
    df3 = df2.copy()
    if COL_AS in df2.columns:
        as_options = ["Toutes"] + sorted([x for x in df2[COL_AS].dropna().unique().tolist() if x])
        normalize_sel("as_sel", as_options)

        as_sel = st.sidebar.multiselect(
            "Aire de santé (notification)",
            options=as_options,
            key="as_sel",
        )

        if as_sel and ("Toutes" not in as_sel):
            df3 = df3[df3[COL_AS].isin(as_sel)]

    # df_f = dataframe filtré géographiquement
    df_f = df3

    # =========================
    # Classification finale (multiselect, "Toutes" par défaut, dépend de df_f)
    # =========================
    if COL_CLASS in df_f.columns:
        class_values = sorted([x for x in df_f[COL_CLASS].dropna().unique().tolist() if x])
        class_options = ["Toutes"] + class_values
        normalize_sel("class_sel", class_options)

        class_sel = st.sidebar.multiselect(
            "Classification finale",
            options=class_options,
            key="class_sel",
        )

        if class_sel and ("Toutes" not in class_sel):
            df_f = df_f[df_f[COL_CLASS].isin(class_sel)]

    # =========================
    # Résultat final labo (multiselect, "Toutes" par défaut, dépend de df_f)
    # =========================
    lab_result_filter_col = None
    if "Resultat_labo" in df_f.columns and df_f["Resultat_labo"].notna().any():
        lab_result_filter_col = "Resultat_labo"
    elif COL_TDRR in df_f.columns and df_f[COL_TDRR].notna().any():
        lab_result_filter_col = COL_TDRR

    if lab_result_filter_col is not None:
        lab_result_values = sorted(
            [x for x in df_f[lab_result_filter_col].dropna().unique().tolist() if x]
        )
        lab_result_options = ["Toutes"] + lab_result_values
        normalize_sel("lab_result_sel", lab_result_options)

        lab_result_sel = st.sidebar.multiselect(
            "Résultat final labo",
            options=lab_result_options,
            key="lab_result_sel",
        )

        if lab_result_sel and ("Toutes" not in lab_result_sel):
            df_f = df_f[df_f[lab_result_filter_col].isin(lab_result_sel)]

    df_f_source = (
        df_quality_source.loc[df_f.index].copy()
        if "df_quality_source" in locals() and isinstance(df_quality_source, pd.DataFrame)
        else df_f.copy()
    )

    age_col = pick_age_col(df_f)

    if show_sidebar_summary:
        with st.sidebar.expander("Résumé des filtres actifs", expanded=True):
            st.caption(f"Lignes analysées après filtres : {len(df_f):,}")
            disease_label = DISEASE_SPECS.get(disease_key, {}).get("label", disease_key)
            st.write(f"Maladie / source : **{disease_label}**")

            years_summary = "Toutes"
            if years_selected_main:
                years_summary = ", ".join(str(y) for y in years_selected_main[:6])
                if len(years_selected_main) > 6:
                    years_summary += " ..."
            st.write(f"Année : **{years_summary}**")

            if use_week_filter and COL_WNUM in df.columns:
                st.write(f"Semaines : **{int(week_min)} → {int(week_max)}**")
            else:
                st.write("Semaines : **toutes**")

            prov_summary = ", ".join(st.session_state.get("prov_sel", ["Toutes"])[:4])
            zs_summary = ", ".join(st.session_state.get("zs_sel", ["Toutes"])[:4])
            class_summary = ", ".join(st.session_state.get("class_sel", ["Toutes"])[:4])
            lab_result_summary = ", ".join(st.session_state.get("lab_result_sel", ["Toutes"])[:4])
            st.write(f"Province : **{prov_summary}**")
            st.write(f"Zone de santé : **{zs_summary}**")
            st.write(f"Classification : **{class_summary}**")
            st.write(f"Résultat final labo : **{lab_result_summary}**")
            st.write(f"Cartographie détaillée : **{'activée' if show_maps else 'désactivée'}**")
else:
    # Mode IDSR: on ne charge pas de line list ici. Les analyses IDSR sont dans l'onglet 9.
    raw = pd.DataFrame()
    df = pd.DataFrame()
    df_f = pd.DataFrame()
    df_quality_source = pd.DataFrame()
    df_f_source = pd.DataFrame()
    files_used = []
    st.info(
        "Mode **IDSR agrégé hebdomadaire** : utilisez l’onglet **IDSR** pour téléverser, "
        "filtrer et analyser le fichier agrégé."
    )
    if show_sidebar_summary:
        with st.sidebar.expander("Résumé des filtres actifs", expanded=True):
            st.write("Mode : **IDSR agrégé**")
            st.write("Chargement fichier : **onglet IDSR**")
            st.write(f"Cartographie détaillée : **{'activée' if show_maps else 'désactivée'}**")

# =========================
# TABS
# =========================
def _legacy_render_overview_dashboard(
    df_: pd.DataFrame,
    files_used: list[str],
    disease_key: str,
    use_custom_viz_flag: bool,
    annotate_values_flag: bool,
    x_tick_step: int,
    pyramid_age_mode: str = "5yr",
) -> None:
    """Assemble la page d'accueil institutionnelle avant les onglets détaillés."""
    if df_.empty:
        st.info("Aucune donnée filtrée n'est disponible pour la synthèse d'accueil.")
        return

    payload = build_dashboard_kpi_payload(df_)
    render_context_row(files_used, disease_key, df_, payload)
    render_dashboard_kpis(payload)
    render_standards_note()

    weekly = payload.get("weekly", pd.DataFrame())
    with st.expander("Options des cartes de synthèse", expanded=False):
        overview_province_map_mode = st.radio(
            "Carte province de synthèse",
            ["Statique", "Interactive"],
            index=0,
            horizontal=True,
            key="overview_province_map_mode",
        )
        overview_map_mode_label = st.selectbox(
            "Annotations sur les cartes de synthèse",
            options=list(MAP_ANNOTATION_MODE_OPTIONS.keys()),
            index=0,
            key="overview_map_annotation_mode",
        )
        overview_map_threshold = st.number_input(
            "Seuil d'affichage des annotations (valeur >)",
            min_value=0,
            max_value=100000,
            value=1,
            step=1,
            key="overview_map_annotation_threshold",
        )

    overview_map_mode = MAP_ANNOTATION_MODE_OPTIONS[overview_map_mode_label]
    fig_map_prov = None
    province_map_payload = prepare_overview_map_data(df_, level="province", match_threshold=0.90)
    gdf_map_prov, df_match_prov, note_map_prov, value_col_prov, group_col_prov, _ = province_map_payload
    if overview_province_map_mode == "Statique":
        fig_map_prov, note_map_prov = build_static_map_overview(
            df_,
            level="province",
            annotation_mode=overview_map_mode,
            annotation_threshold=float(overview_map_threshold),
        )
    fig_map_zs, note_map_zs = build_static_map_overview(
        df_,
        level="zone",
        annotation_mode=overview_map_mode,
        annotation_threshold=float(overview_map_threshold),
    )

    c1, c2, c3 = st.columns([1.05, 1.35, 1.35])
    with c1:
        st.markdown("<div class='cousp-panel-title'>Indicateurs clés de la semaine</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='cousp-summary-box'><div class='summary-lead'>Bloc de synthèse opérationnelle</div></div>",
            unsafe_allow_html=True,
        )
        w1, w2 = st.columns(2)
        latest = payload.get("latest", {})
        previous = payload.get("previous", {})
        with w1:
            st.metric(
                "Cas semaine",
                format_metric_value(latest.get("Cas", np.nan)),
                format_pct_delta(latest.get("Cas", np.nan), previous.get("Cas", np.nan)),
            )
            st.metric(
                "CFR semaine",
                format_metric_value(latest.get("Létalité (%)", np.nan), decimals=2, suffix="%"),
                format_pct_delta(latest.get("Létalité (%)", np.nan), previous.get("Létalité (%)", np.nan)),
            )
        with w2:
            st.metric(
                "Décès semaine",
                format_metric_value(latest.get("Décès", np.nan)),
                format_pct_delta(latest.get("Décès", np.nan), previous.get("Décès", np.nan)),
            )
            st.metric(
                f"Admission <= {get_session_int('seuil_jours', 2)} jours",
                format_metric_value(payload.get("promptitude_pct"), decimals=1, suffix="%"),
                f"n={payload.get('promptitude_n', 0)}",
            )

        st.write(f"Province la plus notifiée : **{payload.get('top_province', 'non disponible')}**")
        st.write(f"Zone de santé la plus notifiée : **{payload.get('top_zs', 'non disponible')}**")
        st.write(
            f"Fenêtre couverte : **{format_range_label_for_display(payload.get('week_span', '-'))}** avec **{format_metric_value(payload.get('cases', 0))}** cas analysés."
        )

    with c2:
        if overview_province_map_mode == "Interactive":
            render_interactive_map_overview(
                "Carte interactive par province",
                gdf_join=gdf_map_prov,
                df_map=df_match_prov,
                note=note_map_prov,
                value_col=value_col_prov,
                source_df=df_,
                source_label_col=group_col_prov,
                chart_key="overview_province_map",
                clicked_state_key="map_clicked_province",
                filter_state_key="prov_sel",
                height=540,
            )
        else:
            render_static_map_overview("Carte statique par province", fig_map_prov, note_map_prov)

    with c3:
        render_static_map_overview("Carte statique par zone de santé", fig_map_zs, note_map_zs)

    render_delay_snapshot_panel(payload)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("<div class='cousp-panel-title'>Surveillance temporelle hebdomadaire</div>", unsafe_allow_html=True)
        if weekly.empty:
            st.info("Série hebdomadaire indisponible.")
        else:
            fig_surveillance = build_weekly_cases_deaths_combo(
                weekly_df=weekly,
                x_col="label",
                cases_col="Cas",
                deaths_col="Décès",
                titre=" ",
                x_titre="Semaine épidémiologique",
                y_titre_cas="Nombre de cas",
                y_titre_deces="Nombre de décès",
                rotation=0,
                annot_bars=annotate_values_flag,
                annot_line=annotate_values_flag,
            )
            if fig_surveillance is not None and x_tick_step > 1 and len(weekly) > x_tick_step:
                fig_surveillance.update_xaxes(
                    tickmode="array",
                    tickvals=weekly["label"].iloc[:: max(int(x_tick_step), 1)],
                    ticktext=weekly["label"].iloc[:: max(int(x_tick_step), 1)],
                )
            st_plot(fig_surveillance, key="overview_weekly_surveillance", annotate_values=annotate_values_flag)

    with d2:
        st.markdown("<div class='cousp-panel-title'>Tendance hebdomadaire des cas et de la létalité observée</div>", unsafe_allow_html=True)
        if weekly.empty:
            st.info("Tendance hebdomadaire indisponible.")
        else:
            week_col = resolve_week_column(df_)
            fig_combo = build_weekly_cases_cfr_combo(
                df=df_,
                week_col=week_col,
                death_col="is_death",
                titre="",
                rotation=45,
                annot_bars=annotate_values_flag,
                annot_line=annotate_values_flag,
                pas_x=int(x_tick_step) if week_col in [COL_WNUM, "YW"] else None,
                taille_fig=(1400, 550),
            )
            st_plot(fig_combo, key="overview_weekly_combo", annotate_values=annotate_values_flag)

    p1, p2, p3 = st.columns([1.3, 0.95, 1.15])
    with p1:
        st.markdown("<div class='cousp-panel-title'>Distribution géographique des notifications</div>", unsafe_allow_html=True)
        geo_col = COL_PROV if COL_PROV in df_.columns and df_[COL_PROV].notna().any() else COL_ZS
        if geo_col in df_.columns and df_[geo_col].notna().any():
            geo_tbl = build_frequency_table(df_, geo_col, top_n=10).sort_values("n", ascending=True)
            fig_geo = px.bar(
                geo_tbl,
                x="n",
                y=geo_col,
                orientation="h",
                text="n" if annotate_values_flag else None,
                color="n",
                color_continuous_scale=["#dbe8f9", "#2b74ca"],
                labels={geo_col: "Lieu", "n": "Nombre de cas"},
            )
            fig_geo.update_layout(coloraxis_showscale=False, title=" ")
            st_plot(fig_geo, key="overview_geo_distribution", annotate_values=annotate_values_flag)
        else:
            st.info("Aucune variable géographique exploitable n'a été détectée.")

    with p2:
        st.markdown("<div class='cousp-panel-title'>Répartition par sexe</div>", unsafe_allow_html=True)
        if COL_SEX in df_.columns and df_[COL_SEX].notna().any():
            sex_tbl = build_frequency_table(df_, COL_SEX)
            fig_sex = px.pie(
                sex_tbl,
                names=COL_SEX,
                values="n",
                hole=0.62,
                color=COL_SEX,
                color_discrete_map=SEX_COLOR_MAP,
            )
            st_plot(fig_sex, key="overview_sex_pie", annotate_values=annotate_values_flag)
        else:
            st.info("La variable Sexe est absente ou vide.")

    with p3:
        st.markdown("<div class='cousp-panel-title'>Courbe hebdomadaire des cas.</div>", unsafe_allow_html=True)
        if weekly.empty or "label" not in weekly.columns or "Cas" not in weekly.columns:
            st.info("Courbe hebdomadaire indisponible.")
        else:
            fig_age = px.line(
                weekly,
                x="label",
                y="Cas",
                markers=True,
                color_discrete_sequence=["#2d7d46"],
                title=" ",
                labels={"label": "Semaine épidémiologique", "Cas": "Nombre de cas"},
            )
            fig_age.update_traces(line=dict(width=3))
            fig_age.update_layout(xaxis_tickangle=-35)
            if x_tick_step > 1 and len(weekly) > x_tick_step:
                fig_age.update_xaxes(
                    tickmode="array",
                    tickvals=weekly["label"].iloc[:: max(int(x_tick_step), 1)],
                    ticktext=weekly["label"].iloc[:: max(int(x_tick_step), 1)],
                )
            st_plot(fig_age, key="overview_age_hist", annotate_values=annotate_values_flag)

    p4, p5 = st.columns([1.55, 1.0])
    with p4:
        st.markdown("<div class='cousp-panel-title'>Pyramide age-sexe</div>", unsafe_allow_html=True)
        st.caption(f"Decoupage actif : {AGE_PYRAMID_MODE_LABELS.get(pyramid_age_mode, pyramid_age_mode)}")
        fig_pyr, pyramid_status = build_overview_age_sex_pyramid_figure(
            df_=df_,
            pyramid_age_mode=pyramid_age_mode,
            use_custom_viz_flag=use_custom_viz_flag,
        )
        if fig_pyr is not None:
            if pyramid_status == "fallback":
                st.caption("Rendu standard actif : les visualisations custom sont desactivees.")
            st_plot(fig_pyr, key="overview_pyramid", height=430, annotate_values=False)
        elif pyramid_status == "data_missing":
            st.info("Pyramide indisponible : variables Age/Sexe insuffisantes.")
        else:
            st.info("Pyramide indisponible : le rendu du graphique a echoue.")

    with p5:
        st.markdown("<div class='cousp-panel-title'>Distribution par tranche d'âge</div>", unsafe_allow_html=True)
        df_age_group = df_.copy()
        age_group_col = get_age_pyramid_group_column_name(pyramid_age_mode)
        df_age_group[age_group_col] = derive_age_pyramid_generic(df_age_group, pyramid_age_mode)

        if age_group_col in df_age_group.columns and df_age_group[age_group_col].notna().any():
            age_group_tbl = build_frequency_table(df_age_group, age_group_col)
            age_order = get_age_pyramid_category_order(pyramid_age_mode)
            extra_age_values = [value for value in age_group_tbl[age_group_col].astype(str).tolist() if value not in age_order]
            age_group_tbl[age_group_col] = pd.Categorical(
                age_group_tbl[age_group_col],
                categories=age_order + extra_age_values,
                ordered=True,
            )
            age_group_tbl = age_group_tbl.sort_values(age_group_col)
            fig_age_group = px.bar(
                age_group_tbl,
                x=age_group_col,
                y="n",
                text="n" if annotate_values_flag else None,
                color_discrete_sequence=["#d97b16"],
                title=" ",
            )
            fig_age_group.update_layout(xaxis_tickangle=-35)
            st_plot(fig_age_group, key="overview_age_group", annotate_values=annotate_values_flag)
        else:
            st.info("Les classes d'âge ne sont pas disponibles.")

if not IDSR_MODE:
    render_overview_dashboard_v2(
        df_=df_f,
        files_used=files_used,
        disease_key=disease_key,
        use_custom_viz_flag=use_custom_viz,
        pyramid_age_mode=pyramid_age_mode,
        annotate_values_flag=annot_vals,
        x_tick_step=int(pas_x),
    )
    st.markdown("<div class='cousp-panel-title'>Analyses détaillées par onglet</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='cousp-panel-title'>Espaces analytiques détaillés</div>", unsafe_allow_html=True)

st.caption("Sélectionnez un onglet détaillé ci-dessous. Le contenu s'affiche en pleine largeur sans navigation compacte par boutons.")

(
    tab_overview_detail,
    tab_statistics,
    tab_methodology,
    tab_surveillance,
    tab_profil,
    tab_qualite,
    tab_irep,
    tab_cousp,
    tab_maps,
    tab_idsr,
) = st.tabs(
    [
        "Vue d’ensemble",
        "Notions statistiques",
        "Méthodologie",
        "Surveillance",
        "Profil",
        "Qualité et export",
        "IREP",
        "COUSP",
        "Cartographie",
        "IDSR",
    ]
)

with tab_overview_detail:
    render_overview_detail_tab(build_runtime_context(**globals()))

with tab_statistics:
    render_statistics_tab(build_runtime_context(**globals()))

# =========================
# ONGLET DÉTAILLÉ : MÉTHODOLOGIE
# =========================
with tab_methodology:
    render_methodology_tab(build_runtime_context(**globals()))

with tab_surveillance:
    render_surveillance_tab(build_runtime_context(**globals()))

with tab_profil:
    render_profile_tab(build_runtime_context(**globals()))

with tab_qualite:
    render_quality_tab(build_runtime_context(**globals()))
    
with tab_irep:
    render_irep_tab(build_runtime_context(**globals()))

with tab_cousp:
    render_cousp_tab(build_runtime_context(**globals()))

with tab_maps:
    render_maps_tab(build_runtime_context(**globals()))

with tab_idsr:
    render_idsr_tab(build_runtime_context(**globals()))

render_footer()
