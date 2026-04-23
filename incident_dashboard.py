from dashboard_app.runtime_support import build_runtime_context, inject_runtime_support
from dashboard_app.app_loader import (
    LINE_LIST_BUNDLE_LABEL,
    LINE_LIST_DIR,
    build_postgresql_query,
    get_excel_sheet_names_from_path,
    get_line_list_bundle_caption,
    guess_preferred_included_file,
    list_available_line_list_files,
    read_postgresql_file,
)
from dashboard_app.tabs.overview_detail import render_overview_detail_tab
from dashboard_app.tabs.surveillance import render_surveillance_tab
from dashboard_app.tabs.profile import render_profile_tab
from dashboard_app.tabs.quality import render_quality_tab
from dashboard_app.tabs.sitrep import render_sitrep_tab
from dashboard_app.tabs.idsr import render_idsr_tab
from dashboard_app.tabs.irep import render_irep_tab
from dashboard_app.tabs.maps import render_maps_tab
from dashboard_app.tabs.methodology import render_methodology_tab

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

st.sidebar.header("Visualisations")
with st.sidebar.expander("Paramètres avancés des visualisations", expanded=False):
    use_custom_viz = st.checkbox(
        "Utiliser visualisations custom (dataminsante)",
        value=True,
        key="use_custom_viz",
        help="Les fonctions custom sont intégrées dans le package applicatif du dashboard."
    )
    annot_vals = st.checkbox("Afficher annotations (valeurs)", value=False, key="annot_vals")
    pas_x = st.number_input("Pas X (ticks)", min_value=1, max_value=10, value=1, step=1, key="pas_x")
    seuil_min_count = st.number_input("Seuil minimal (filtrer petits groupes)", min_value=0, max_value=100, value=0, step=1, key="seuil_min_count")
    st.button("Réinitialiser les options d’affichage", key="reset_display_options", on_click=_reset_display_options)

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

if not IDSR_MODE:
    source_ready = False
    source_message = ""
    if line_list_source == "upload":
        source_ready = upl is not None
        source_message = "Veuillez téléverser un fichier de données (`.xlsx` ou `.csv`) pour démarrer l’analyse de surveillance."
    elif line_list_source == "local":
        source_ready = selected_local_path is not None and selected_local_path.exists()
        source_message = "Aucun fichier inclus exploitable n'est disponible dans l'application."
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

    # ✅ 1) Standardisation commune (Rougeole/Choléra/…)
    raw = standardize_ll_by_disease(raw, disease_key)

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
        selected_zone = _resolve_map_filter_value(clicked_zone, df[COL_ZS].dropna().unique().tolist())
        if selected_zone:
            st.session_state["zs_sel"] = [selected_zone]
            st.session_state["as_sel"] = ["Toutes"]
            if COL_PROV in df.columns:
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
            st.write(f"Province : **{prov_summary}**")
            st.write(f"Zone de santé : **{zs_summary}**")
            st.write(f"Classification : **{class_summary}**")
            st.write(f"Cartographie détaillée : **{'activée' if show_maps else 'désactivée'}**")
else:
    # Mode IDSR: on ne charge pas de line list ici. Les analyses IDSR sont dans l'onglet 9.
    raw = pd.DataFrame()
    df = pd.DataFrame()
    df_f = pd.DataFrame()
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
def render_overview_dashboard(
    df_: pd.DataFrame,
    files_used: list[str],
    disease_key: str,
    use_custom_viz_flag: bool,
    annotate_values_flag: bool,
    x_tick_step: int,
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
            f"Fenêtre couverte : **{payload.get('week_span', '-')}** avec **{format_metric_value(payload.get('cases', 0))}** cas analysés."
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
        st.markdown("<div class='cousp-panel-title'>Répartition par âge</div>", unsafe_allow_html=True)
        years = infer_age_years_generic(df_)
        if years.notna().any():
            age_hist = pd.DataFrame({"Age_en_ans": years.dropna()})
            fig_age = px.histogram(
                age_hist,
                x="Age_en_ans",
                nbins=18,
                color_discrete_sequence=["#2d7d46"],
                title=" ",
            )
            st_plot(fig_age, key="overview_age_hist", annotate_values=annotate_values_flag)
        else:
            age_col = pick_age_col(df_)
            if age_col is None:
                st.info("Aucune information d'âge exploitable n'est disponible.")
            else:
                age_tbl = build_frequency_table(df_, age_col)
                fig_age = px.bar(age_tbl, x=age_col, y="n", color_discrete_sequence=["#2d7d46"])
                fig_age.update_layout(xaxis_tickangle=-35)
                st_plot(fig_age, key="overview_age_bar", annotate_values=annotate_values_flag)

    p4, p5 = st.columns([1.55, 1.0])
    with p4:
        st.markdown("<div class='cousp-panel-title'>Pyramide age-sexe</div>", unsafe_allow_html=True)
        df_pyr = df_.copy()
        df_pyr["Tranche_age_5ans_dashboard"] = derive_age_5yr_generic(df_pyr)
        if use_custom_viz_flag and HAS_CUSTOM_VIZ and COL_SEX in df_pyr.columns and df_pyr["Tranche_age_5ans_dashboard"].notna().any():
            fig_pyr = plot_pyramide_symetrique(
                df=df_pyr,
                col_categorie="Tranche_age_5ans_dashboard",
                col_groupe=COL_SEX,
                valeurs_neg=["Masculin", "Homme", "M"],
                titre=None,
                seuil_min=0,
                croissant=True,
                afficher_signe_negatif_dans_label=False,
            )
            if fig_pyr is not None:
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
            st_plot(fig_pyr, key="overview_pyramid", height=430, annotate_values=False)
        else:
            st.info("Pyramide indisponible : variables Age/Sexe insuffisantes.")

    with p5:
        st.markdown("<div class='cousp-panel-title'>Distribution par tranche d'âge</div>", unsafe_allow_html=True)
        df_age_group = df_.copy()
        age_group_col = pick_age_col(df_age_group)
        if age_group_col is None:
            df_age_group["Tranche_age_4cat_dashboard"] = derive_age_4cat_generic(df_age_group)
            age_group_col = "Tranche_age_4cat_dashboard"

        if age_group_col in df_age_group.columns and df_age_group[age_group_col].notna().any():
            age_group_tbl = build_frequency_table(df_age_group, age_group_col)
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
    render_overview_dashboard(
        df_=df_f,
        files_used=files_used,
        disease_key=disease_key,
        use_custom_viz_flag=use_custom_viz,
        annotate_values_flag=annot_vals,
        x_tick_step=int(pas_x),
    )
    st.markdown("<div class='cousp-panel-title'>Analyses détaillées par onglet</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='cousp-panel-title'>Espaces analytiques détaillés</div>", unsafe_allow_html=True)

st.caption("Sélectionnez un onglet détaillé ci-dessous. Le contenu s'affiche en pleine largeur sans navigation compacte par boutons.")

(
    tab_overview_detail,
    tab_methodology,
    tab_surveillance,
    tab_profil,
    tab_qualite,
    tab_maps,
    tab_sitrep,
    tab_idsr,
    tab_irep,
) = st.tabs(
    [
        "Vue d’ensemble",
        "Méthodologie",
        "\U0001F4C8 Surveillance",
        "\U0001F465 Profil",
        "\U0001F5C2\ufe0f Qualité & export",
        "\U0001F5FA\ufe0f Cartographie",
        "\U0001F4DD SITREP",
        "\U0001F4DA IDSR",
        "\U0001F4CC IREP",
    ]
)

with tab_overview_detail:
    render_overview_detail_tab(build_runtime_context(**globals()))

# =========================
# TAB 1: DYNAMIQUE ÉPIDÉMIOLOGIQUE ET PROMPTITUDE
# =========================
with tab_surveillance:
    render_surveillance_tab(build_runtime_context(**globals()))

with tab_profil:
    render_profile_tab(build_runtime_context(**globals()))

with tab_qualite:
    render_quality_tab(build_runtime_context(**globals()))

with tab_sitrep:
    render_sitrep_tab(build_runtime_context(**globals()))

with tab_idsr:
    render_idsr_tab(build_runtime_context(**globals()))

with tab_irep:
    render_irep_tab(build_runtime_context(**globals()))

with tab_maps:
    render_maps_tab(build_runtime_context(**globals()))

with tab_methodology:
    render_methodology_tab(build_runtime_context(**globals()))

render_footer()
