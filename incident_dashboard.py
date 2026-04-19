from dashboard_app.core import *
from dashboard_app.core import _normalize_metric_alias_columns

st.set_page_config(page_title="LL RDC - Dashboard", layout="wide")

from dashboard_app.domain import *
from dashboard_app.domain import _is_yes_series, _norm_key, _resolve_map_filter_value, _tdr_result_norm
from dashboard_app.overview import *
from dashboard_app.advanced import *

try:
    from sqlalchemy import create_engine, text

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    create_engine = None
    text = None

try:
    import psycopg2

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None

inject_professional_dashboard_css()
render_professional_header()

LINE_LIST_DIR = Path(__file__).resolve().parent / "line_list"
LINE_LIST_BUNDLE_LABEL = "line_list/"


def list_available_line_list_files() -> list[Path]:
    if not LINE_LIST_DIR.exists():
        return []
    return sorted(
        [
            path
            for path in LINE_LIST_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".csv"}
        ],
        key=lambda p: p.name.lower(),
    )


def get_line_list_bundle_caption() -> str:
    return f"Fichiers inclus dans l'application : `{LINE_LIST_BUNDLE_LABEL}`"


def get_excel_sheet_names_from_path(path: Path) -> list[str]:
    try:
        return pd.ExcelFile(path).sheet_names
    except Exception as exc:
        st.error(f"Impossible de lire le fichier Excel local : {exc}")
        st.stop()


def validate_table_identifier(identifier: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\\.]*", str(identifier).strip()))


def build_postgresql_query(query_mode: str, table_name: str, sql_query: str) -> str:
    if query_mode == "Table":
        clean_table_name = str(table_name).strip()
        if not clean_table_name:
            st.error("Renseigne le nom de la table PostgreSQL.")
            st.stop()
        if not validate_table_identifier(clean_table_name):
            st.error("Le nom de table contient des caractères non autorisés.")
            st.stop()
        return f"SELECT * FROM {clean_table_name}"

    clean_query = str(sql_query).strip()
    if not clean_query:
        st.error("Renseigne une requête SQL PostgreSQL.")
        st.stop()
    return clean_query


def read_postgresql_file(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    query: str,
) -> pd.DataFrame:
    if not SQLALCHEMY_AVAILABLE and not PSYCOPG2_AVAILABLE:
        st.error("Le connecteur PostgreSQL n'est pas installé. Ajoute `sqlalchemy` et `psycopg2-binary`.")
        st.stop()

    try:
        if SQLALCHEMY_AVAILABLE:
            engine = create_engine(
                f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
                pool_pre_ping=True,
            )
            with engine.connect() as connection:
                df_loaded = pd.read_sql_query(text(query), connection)
            engine.dispose()
        else:
            with psycopg2.connect(
                host=host,
                port=port,
                dbname=database,
                user=user,
                password=password,
            ) as connection:
                df_loaded = pd.read_sql_query(query, connection)
    except Exception as exc:
        st.error(f"Impossible de charger les données PostgreSQL : {exc}")
        st.stop()

    df_loaded.columns = df_loaded.columns.astype(str).str.strip()
    return df_loaded

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
        "Téléverser un fichier": "upload",
        "Charger un fichier inclus": "local",
        "Se connecter à PostgreSQL": "postgres",
    }
    source_label = st.sidebar.selectbox(
        "Téléverser une line list (xlsx/csv)",
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
            selected_local_name = st.sidebar.selectbox(
                "Fichier inclus",
                options=[p.name for p in available_line_list_files],
                key="ll_local_file",
            )
            selected_local_path = next((p for p in available_line_list_files if p.name == selected_local_name), None)
            if selected_local_path is not None:
                st.sidebar.caption(get_line_list_bundle_caption())
                if selected_local_path.suffix.lower() in {".xlsx", ".xls"}:
                    local_sheets = get_excel_sheet_names_from_path(selected_local_path)
                    local_default_sheet = default_sheet if default_sheet in local_sheets else local_sheets[0]
                    sheet_upl = st.sidebar.text_input(
                        "Nom feuille (si Excel local)",
                        value=local_default_sheet,
                        key="ll_local_sheet",
                    )
        else:
            st.sidebar.warning("Aucun fichier `.xlsx`, `.xls` ou `.csv` inclus dans l'application n'a été trouvé.")
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
        help="Ici, les fonctions custom sont intégrées dans ce fichier (autonome)."
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
            default=st.session_state["prov_sel"],
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
            default=st.session_state["zs_sel"],
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
            default=st.session_state["as_sel"],
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
            default=st.session_state["class_sel"],
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

tab_overview_detail, tab_surveillance, tab_profil, tab_qualite, tab_maps, tab_sitrep, tab_idsr, tab_irep = st.tabs(
    [
        "Vue d’ensemble",
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


def _prepare_surveillance_period_scope(df_scope: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit un repère temporel robuste pour les fenêtres de surveillance."""
    if df_scope is None or df_scope.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["order", "label"])

    scoped = df_scope.copy()
    scoped["_surv_order"] = np.nan
    scoped["_surv_label"] = pd.NA

    if COL_WNUM in scoped.columns and pd.to_numeric(scoped[COL_WNUM], errors="coerce").notna().any():
        week_num = pd.to_numeric(scoped[COL_WNUM], errors="coerce")
        if COL_YEAR in scoped.columns and pd.to_numeric(scoped[COL_YEAR], errors="coerce").notna().any():
            year_num = pd.to_numeric(scoped[COL_YEAR], errors="coerce")
            mask = week_num.notna() & year_num.notna()
            scoped.loc[mask, "_surv_order"] = (year_num[mask] * 100) + week_num[mask]
            scoped.loc[mask, "_surv_label"] = [
                f"SE{int(w):02d}-{int(y)}"
                for y, w in zip(year_num[mask], week_num[mask])
            ]
        else:
            mask = week_num.notna()
            scoped.loc[mask, "_surv_order"] = week_num[mask]
            scoped.loc[mask, "_surv_label"] = [f"SE{int(w):02d}" for w in week_num[mask]]
    elif "YW" in scoped.columns and scoped["YW"].notna().any():
        yw_text = scoped["YW"].astype("string").fillna("").str.strip()
        parsed = yw_text.str.extract(r"(?P<year>\d{4}).*?(?P<week>\d{1,2})")
        year_num = pd.to_numeric(parsed["year"], errors="coerce")
        week_num = pd.to_numeric(parsed["week"], errors="coerce")
        mask = year_num.notna() & week_num.notna()
        if mask.any():
            scoped.loc[mask, "_surv_order"] = (year_num[mask] * 100) + week_num[mask]
            scoped.loc[mask, "_surv_label"] = [
                f"SE{int(w):02d}-{int(y)}"
                for y, w in zip(year_num[mask], week_num[mask])
            ]
        else:
            valid_mask = yw_text != ""
            scoped.loc[valid_mask, "_surv_label"] = yw_text[valid_mask]
            categories = pd.Categorical(scoped.loc[valid_mask, "_surv_label"])
            scoped.loc[valid_mask, "_surv_order"] = categories.codes

    reference = (
        scoped.loc[scoped["_surv_order"].notna() & scoped["_surv_label"].notna(), ["_surv_order", "_surv_label"]]
        .drop_duplicates()
        .rename(columns={"_surv_order": "order", "_surv_label": "label"})
        .sort_values("order")
        .reset_index(drop=True)
    )
    return scoped, reference


def _build_surveillance_top_table(
    df_scope: pd.DataFrame,
    group_cols: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """Construit un top géographique avec cas, décès et létalité."""
    valid_group_cols = [c for c in group_cols if c in df_scope.columns]
    if df_scope is None or df_scope.empty or not valid_group_cols:
        return pd.DataFrame()

    tmp = df_scope.copy()
    tmp["_cas_"] = 1
    tmp["_deces_"] = (
        pd.to_numeric(tmp["is_death"], errors="coerce").fillna(0).astype(int)
        if "is_death" in tmp.columns
        else 0
    )

    for col in valid_group_cols:
        tmp[col] = (
            tmp[col]
            .fillna("Inconnu")
            .astype(str)
            .str.strip()
            .replace("", "Inconnu")
        )

    grouped = (
        tmp.groupby(valid_group_cols, as_index=False)
        .agg(Cas=("_cas_", "sum"), Décès=("_deces_", "sum"))
        .sort_values(["Cas", "Décès"], ascending=[False, False])
        .head(int(top_n))
        .copy()
    )
    grouped["Létalité (%)"] = np.where(
        grouped["Cas"] > 0,
        grouped["Décès"] / grouped["Cas"] * 100.0,
        np.nan,
    ).round(2)
    grouped.insert(0, "Rang", range(1, len(grouped) + 1))
    return grouped


def _describe_surveillance_top_table(
    table: pd.DataFrame,
    label_cols: list[str],
    total_cases: int,
    empty_label: str,
) -> str:
    """Génère une lecture rapide du top 5 géographique."""
    if table.empty or total_cases <= 0:
        return empty_label

    leader = table.iloc[0]
    leader_parts = [str(leader[col]) for col in label_cols if col in table.columns and pd.notna(leader[col])]
    leader_label = " / ".join([part for part in leader_parts if part and part != "nan"]) or "Non renseigné"
    leader_share = float(leader["Cas"]) / float(total_cases) * 100.0
    top5_share = float(table["Cas"].sum()) / float(total_cases) * 100.0
    return (
        f"{leader_label} concentre {leader_share:.1f}% des cas de cette période ; "
        f"le top 5 en concentre {top5_share:.1f}%."
    )


def _surveillance_scope_kpis(df_scope: pd.DataFrame) -> tuple[int, int, float]:
    """Retourne cas, décès et létalité pour une fenêtre donnée."""
    if df_scope is None or df_scope.empty:
        return 0, 0, np.nan
    total_cases = int(len(df_scope))
    total_deaths = int(
        pd.to_numeric(df_scope["is_death"], errors="coerce").fillna(0).sum()
    ) if "is_death" in df_scope.columns else 0
    cfr = safe_pct(total_deaths, total_cases)
    return total_cases, total_deaths, cfr


def _surveillance_clean_text_series(series: pd.Series) -> pd.Series:
    """Nettoie une série texte pour les comptages métier."""
    if series is None:
        return pd.Series(dtype="object")
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned[cleaned != ""]


def _surveillance_period_bounds(df_scope: pd.DataFrame) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[str]]:
    """Déduit la meilleure période lisible à partir des dates disponibles."""
    for col in [DATE_NOTIF, DATE_ONSET, DATE_ADM, DATE_INV]:
        if col in df_scope.columns:
            dt = pd.to_datetime(df_scope[col], errors="coerce")
            if dt.notna().any():
                return dt.min(), dt.max(), col
    return None, None, None


def _format_surveillance_date(dt_value: Optional[pd.Timestamp]) -> str:
    """Formate une date de résumé si disponible."""
    if dt_value is None or pd.isna(dt_value):
        return "-"
    try:
        return pd.Timestamp(dt_value).strftime("%d/%m/%Y")
    except Exception:
        return str(dt_value)


def _normalize_severity_label(value: Any) -> str:
    """Harmonise les modalités de sévérité/déshydratation."""
    if value is None or pd.isna(value):
        return MISSING_LABEL
    raw = str(value).strip()
    if not raw:
        return MISSING_LABEL
    norm = "".join(
        c for c in unicodedata.normalize("NFD", raw.lower())
        if unicodedata.category(c) != "Mn"
    )
    if "sever" in norm:
        return "Sévère"
    if "moder" in norm:
        return "Modérée"
    if "leger" in norm or "legere" in norm or "mild" in norm:
        return "Légère"
    if "inconnu" in norm or "non renseigne" in norm or norm == "na":
        return MISSING_LABEL
    return raw


def _build_severity_summary_text(df_scope: pd.DataFrame) -> Optional[str]:
    """Construit une phrase courte sur la sévérité si la variable est disponible."""
    if COL_DEHY not in df_scope.columns:
        return None
    severity = _surveillance_clean_text_series(df_scope[COL_DEHY])
    if severity.empty:
        return None
    counts = (
        severity.map(_normalize_severity_label)
        .value_counts()
        .rename_axis("Modalité")
        .reset_index(name="n")
    )
    preferred_order = {"Légère": 0, "Modérée": 1, "Sévère": 2, MISSING_LABEL: 3}
    counts["_order"] = counts["Modalité"].map(lambda x: preferred_order.get(x, 99))
    counts = counts.sort_values(["_order", "n"], ascending=[True, False])
    parts = [f"{row['Modalité']} : {int(row['n'])}" for _, row in counts.iterrows()]
    if not parts:
        return None
    return "Degré de sévérité selon la variable disponible : " + " | ".join(parts)


def _build_investigation_summary_text(df_scope: pd.DataFrame) -> Optional[str]:
    """Construit un résumé standard des cas investigués."""
    if DATE_INV not in df_scope.columns:
        return None
    total_cases = int(len(df_scope))
    if total_cases <= 0:
        return None
    investigated = int(pd.to_datetime(df_scope[DATE_INV], errors="coerce").notna().sum())
    return (
        f"Cas investigués : {format_metric_value(investigated)} "
        f"({format_metric_value(safe_pct(investigated, total_cases), decimals=1)}%)."
    )


def _build_tdr_summary_text(df_scope: pd.DataFrame) -> Optional[str]:
    """Construit un résumé standard des TDR réalisés et de la positivité."""
    result_col = None
    if COL_TDRR in df_scope.columns and df_scope[COL_TDRR].notna().any():
        result_col = COL_TDRR
    elif "Resultat_labo" in df_scope.columns and df_scope["Resultat_labo"].notna().any():
        result_col = "Resultat_labo"

    if COL_TDR in df_scope.columns:
        tdr_yes = _is_yes_series(df_scope[COL_TDR])
        n_tdr = int(tdr_yes.sum())
        if n_tdr <= 0:
            return "TDR réalisés : 0."
        if result_col is None:
            return f"TDR réalisés : {format_metric_value(n_tdr)}."
        label_prefix = "TDR réalisés"
    else:
        if result_col is None:
            return None
        tdr_yes = pd.Series(True, index=df_scope.index)
        n_tdr = int(df_scope[result_col].notna().sum())
        label_prefix = "Résultats labo renseignés"

    res_n = _tdr_result_norm(df_scope[result_col])
    valid_mask = tdr_yes & res_n.isin(TDR_POS_SET.union(TDR_NEG_SET))
    n_valid = int(valid_mask.sum())
    n_pos = int((tdr_yes & res_n.isin(TDR_POS_SET)).sum())
    if n_valid > 0:
        positivity = safe_pct(n_pos, n_valid)
        return (
            f"{label_prefix} : {format_metric_value(n_tdr)} "
            f"(positivité : {format_metric_value(positivity, decimals=1)}%; "
            f"{format_metric_value(n_pos)} positifs sur {format_metric_value(n_valid)} résultats interprétables)."
        )
    return f"{label_prefix} : {format_metric_value(n_tdr)} (positivité non calculable : résultats interprétables absents)."


def _build_comparison_sentence(
    current_df: pd.DataFrame,
    previous_df: Optional[pd.DataFrame],
    current_label: Optional[str] = None,
    previous_label: Optional[str] = None,
) -> Optional[str]:
    """Produit une phrase standard de tendance par rapport à une période de référence."""
    if previous_df is None or previous_df.empty:
        return None

    cur_cases, cur_deaths, cur_cfr = _surveillance_scope_kpis(current_df)
    prev_cases, prev_deaths, prev_cfr = _surveillance_scope_kpis(previous_df)
    if cur_cases <= 0:
        return None

    cur_label_txt = current_label or "la période courante"
    prev_label_txt = previous_label or "la période précédente"

    if prev_cases <= 0:
        return (
            f"{cur_label_txt} : {format_metric_value(cur_cases)} cas et {format_metric_value(cur_deaths)} décès "
            f"(létalité {format_metric_value(cur_cfr, decimals=1)}%). "
            f"Aucun cas n’avait été rapporté durant {prev_label_txt}."
        )

    delta = pct_change_safe(cur_cases, prev_cases)
    if pd.isna(delta):
        trend_text = "La variation par rapport à la période précédente n’est pas calculable."
    elif delta > 0:
        trend_text = f"Une hausse de {abs(delta):.1f}% est observée"
    elif delta < 0:
        trend_text = f"Une baisse de {abs(delta):.1f}% est observée"
    else:
        trend_text = "La situation est stable"

    return (
        f"{cur_label_txt} : {format_metric_value(cur_cases)} cas et {format_metric_value(cur_deaths)} décès "
        f"(létalité {format_metric_value(cur_cfr, decimals=1)}%). "
        f"{trend_text} par rapport à {prev_label_txt} "
        f"({format_metric_value(prev_cases)} cas, {format_metric_value(prev_deaths)} décès, "
        f"létalité {format_metric_value(prev_cfr, decimals=1)}%)."
    )


def _build_top_province_summary_text(df_scope: pd.DataFrame) -> Optional[str]:
    """Construit une phrase standard sur la concentration géographique provinciale."""
    total_cases, _, _ = _surveillance_scope_kpis(df_scope)
    top_prov = _build_surveillance_top_table(df_scope, [COL_PROV], top_n=5)
    if top_prov.empty or total_cases <= 0:
        return None

    top5_share = safe_pct(top_prov["Cas"].sum(), total_cases)
    province_bits = []
    for _, row in top_prov.iterrows():
        province_bits.append(
            f"{row[COL_PROV]} ({safe_pct(row['Cas'], total_cases):.1f}%)"
        )
    return (
        f"La majorité des cas ({top5_share:.1f}%) est concentrée dans les provinces suivantes : "
        + "; ".join(province_bits)
        + "."
    )


def _build_scope_overview_text(
    df_scope: pd.DataFrame,
    scope_kind: str,
    latest_week_df: Optional[pd.DataFrame] = None,
    latest_label: Optional[str] = None,
) -> Optional[str]:
    """Construit le paragraphe d'ouverture adapté à la fenêtre courante."""
    total_cases, total_deaths, cfr = _surveillance_scope_kpis(df_scope)
    if total_cases <= 0:
        return None

    n_prov = (
        _surveillance_clean_text_series(df_scope[COL_PROV]).nunique()
        if COL_PROV in df_scope.columns else 0
    )
    n_zs = (
        _surveillance_clean_text_series(df_scope[COL_ZS]).nunique()
        if COL_ZS in df_scope.columns else 0
    )
    dmin, dmax, _ = _surveillance_period_bounds(df_scope)
    date_span = None
    if dmin is not None and dmax is not None:
        date_span = f"du {_format_surveillance_date(dmin)} au {_format_surveillance_date(dmax)}"

    if scope_kind == "weekly":
        prefix = f"Durant {latest_label or 'la semaine la plus récente'}"
    elif scope_kind == "recent4":
        prefix = "Au cours des 4 dernières semaines"
    else:
        prefix = f"Sur le cumul {date_span}" if date_span else "Sur l’ensemble de la fenêtre sélectionnée"

    geo_text = ""
    if n_zs > 0 and n_prov > 0:
        geo_text = f"{format_metric_value(n_zs)} ZS dans {format_metric_value(n_prov)} provinces"
    elif n_zs > 0:
        geo_text = f"{format_metric_value(n_zs)} ZS"
    elif n_prov > 0:
        geo_text = f"{format_metric_value(n_prov)} provinces"

    base = (
        f"{prefix}, {format_metric_value(total_cases)} cas suspects et {format_metric_value(total_deaths)} décès "
        f"ont été enregistrés (létalité : {format_metric_value(cfr, decimals=1)}%)."
    )

    if geo_text:
        base += f" Les notifications proviennent de {geo_text}."

    if scope_kind == "cumulative" and latest_week_df is not None and not latest_week_df.empty:
        wk_cases, wk_deaths, wk_cfr = _surveillance_scope_kpis(latest_week_df)
        if latest_label:
            base += (
                f" La semaine la plus récente ({latest_label}) totalise "
                f"{format_metric_value(wk_cases)} cas, {format_metric_value(wk_deaths)} décès "
                f"et une létalité de {format_metric_value(wk_cfr, decimals=1)}%."
            )
    return base


def _build_province_interpretations(
    df_scope: pd.DataFrame,
    comparison_df: Optional[pd.DataFrame] = None,
    comparison_label: Optional[str] = None,
    top_n: int = 5,
) -> list[str]:
    """Construit un résumé narratif standard par province pour le top 5."""
    if df_scope is None or df_scope.empty or COL_PROV not in df_scope.columns:
        return []

    top_prov = _build_surveillance_top_table(df_scope, [COL_PROV], top_n=top_n)
    if top_prov.empty:
        return []

    summaries: list[str] = []
    for _, row in top_prov.iterrows():
        province_name = row.get(COL_PROV, MISSING_LABEL)
        prov_mask = df_scope[COL_PROV].fillna("").astype(str).str.strip() == str(province_name).strip()
        prov_df = df_scope.loc[prov_mask].copy()
        if prov_df.empty:
            continue

        prov_cases, prov_deaths, prov_cfr = _surveillance_scope_kpis(prov_df)
        sentence = (
            f"{province_name} : {format_metric_value(prov_cases)} cas suspects, "
            f"dont {format_metric_value(prov_deaths)} décès "
            f"(létalité : {format_metric_value(prov_cfr, decimals=1)}%)."
        )

        if comparison_df is not None and not comparison_df.empty and COL_PROV in comparison_df.columns:
            prev_mask = comparison_df[COL_PROV].fillna("").astype(str).str.strip() == str(province_name).strip()
            prev_df = comparison_df.loc[prev_mask].copy()
            if not prev_df.empty:
                prev_cases, prev_deaths, prev_cfr = _surveillance_scope_kpis(prev_df)
                if prev_cases > 0:
                    delta = pct_change_safe(prov_cases, prev_cases)
                    if not pd.isna(delta):
                        if delta > 0:
                            trend = f"Une hausse de {abs(delta):.1f}% est observée"
                        elif delta < 0:
                            trend = f"Une baisse de {abs(delta):.1f}% est observée"
                        else:
                            trend = "La situation est stable"
                        sentence += (
                            f" {trend} par rapport à {comparison_label or 'la période précédente'} "
                            f"({format_metric_value(prev_cases)} cas, {format_metric_value(prev_deaths)} décès, "
                            f"létalité {format_metric_value(prev_cfr, decimals=1)}%)."
                        )
                else:
                    sentence += f" Aucun cas n’avait été rapporté durant {comparison_label or 'la période précédente'}."

        if COL_ZS in prov_df.columns:
            zs_series = _surveillance_clean_text_series(prov_df[COL_ZS])
            n_zs_reporting = int(zs_series.nunique())
            if n_zs_reporting > 0:
                mean_cases = prov_cases / n_zs_reporting
                sentence += (
                    f" Une moyenne de {mean_cases:.1f} cas par ZS est observée parmi "
                    f"{format_metric_value(n_zs_reporting)} ZS ayant notifié des cas."
                )

                top_zs = _build_surveillance_top_table(prov_df, [COL_ZS], top_n=3)
                if not top_zs.empty:
                    top_zs_bits = [
                        f"{zs_row[COL_ZS]} ({format_metric_value(zs_row['Cas'])} cas)"
                        for _, zs_row in top_zs.iterrows()
                    ]
                    sentence += " Les ZS les plus touchées sont : " + ", ".join(top_zs_bits) + "."

        severity_text = _build_severity_summary_text(prov_df)
        if severity_text is not None:
            sentence += f" {severity_text}."

        summaries.append(sentence)

    return summaries


def _build_surveillance_summary_lines(
    df_scope: pd.DataFrame,
    scope_kind: str,
    current_label: Optional[str] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    comparison_label: Optional[str] = None,
    latest_week_df: Optional[pd.DataFrame] = None,
    latest_label: Optional[str] = None,
) -> list[str]:
    """Assemble le résumé automatique standard à afficher dans chaque situation."""
    lines: list[str] = []

    overview_text = _build_scope_overview_text(
        df_scope,
        scope_kind=scope_kind,
        latest_week_df=latest_week_df,
        latest_label=latest_label,
    )
    if overview_text:
        lines.append(overview_text)

    comparison_text = _build_comparison_sentence(
        current_df=df_scope,
        previous_df=comparison_df,
        current_label=current_label,
        previous_label=comparison_label,
    )
    if comparison_text:
        lines.append(comparison_text)

    top_prov_text = _build_top_province_summary_text(df_scope)
    if top_prov_text:
        lines.append(top_prov_text)

    severity_text = _build_severity_summary_text(df_scope)
    if severity_text:
        lines.append(severity_text + ".")

    investigation_text = _build_investigation_summary_text(df_scope)
    if investigation_text:
        lines.append(investigation_text)

    tdr_text = _build_tdr_summary_text(df_scope)
    if tdr_text:
        lines.append(tdr_text)

    return lines


def _render_surveillance_window(
    title: str,
    df_scope: pd.DataFrame,
    description: str,
    empty_message: str,
    narrative_context: Optional[dict[str, Any]] = None,
) -> None:
    """Affiche une fenêtre standardisée de surveillance."""
    st.markdown(f"### {title}")
    st.caption(description)

    if df_scope.empty:
        st.info(empty_message)
        return

    total_cases = int(len(df_scope))
    total_deaths = int(
        pd.to_numeric(df_scope["is_death"], errors="coerce").fillna(0).sum()
    ) if "is_death" in df_scope.columns else 0
    cfr = (total_deaths / total_cases * 100.0) if total_cases > 0 else np.nan

    k1, k2, k3 = st.columns(3)
    k1.metric("Cas", format_metric_value(total_cases))
    k2.metric("Décès", format_metric_value(total_deaths))
    k3.metric("Létalité (%)", format_metric_value(cfr, decimals=2))

    narrative_context = narrative_context or {}
    summary_lines = _build_surveillance_summary_lines(
        df_scope=df_scope,
        scope_kind=narrative_context.get("scope_kind", "cumulative"),
        current_label=narrative_context.get("current_label"),
        comparison_df=narrative_context.get("comparison_df"),
        comparison_label=narrative_context.get("comparison_label"),
        latest_week_df=narrative_context.get("latest_week_df"),
        latest_label=narrative_context.get("latest_label"),
    )

    if summary_lines:
        st.markdown("**Résumé automatique**")
        st.markdown("\n".join([f"- {line}" for line in summary_lines]))

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Analyse des 5 provinces les plus touchées**")
        top_prov = _build_surveillance_top_table(df_scope, [COL_PROV], top_n=5)
        if top_prov.empty:
            st.info("L’analyse provinciale est indisponible : variable Province absente ou non renseignée.")
        else:
            st.caption(
                _describe_surveillance_top_table(
                    top_prov,
                    [COL_PROV],
                    total_cases,
                    "Aucune province exploitable dans cette fenêtre.",
                )
            )
            st.dataframe(top_prov, width="stretch", hide_index=True)

    with g2:
        st.markdown("**Analyse des 5 zones de santé les plus touchées**")
        zs_group_cols = [c for c in [COL_PROV, COL_ZS] if c in df_scope.columns]
        top_zs = _build_surveillance_top_table(df_scope, zs_group_cols, top_n=5)
        if top_zs.empty:
            st.info("L’analyse des zones de santé est indisponible : variable ZS absente ou non renseignée.")
        else:
            st.caption(
                _describe_surveillance_top_table(
                    top_zs,
                    zs_group_cols,
                    total_cases,
                    "Aucune zone de santé exploitable dans cette fenêtre.",
                )
            )
            st.dataframe(top_zs, width="stretch", hide_index=True)

    province_interpretations = _build_province_interpretations(
        df_scope,
        comparison_df=narrative_context.get("comparison_df"),
        comparison_label=narrative_context.get("comparison_label"),
        top_n=5,
    )
    if province_interpretations:
        with st.expander("Interprétation automatique par province (top 5)", expanded=False):
            for province_text in province_interpretations:
                st.markdown(f"- {province_text}")


def _build_sitrep_week_label(se: Any, annee: Any) -> str:
    """Formate un libellé standard de semaine pour le SITREP."""
    try:
        return f"SE{int(float(se)):02d}/{int(float(annee))}"
    except Exception:
        return f"SE{se}/{annee}"


def _format_sitrep_metric_delta(current: Any, previous: Any) -> Optional[str]:
    """Retourne un delta court et lisible pour les KPI du SITREP."""
    if previous is None or pd.isna(previous):
        return None
    try:
        cur_val = float(current)
        prev_val = float(previous)
    except Exception:
        return None

    if prev_val > 0:
        delta = pct_change_safe(cur_val, prev_val)
        if pd.isna(delta):
            return None
        return f"{delta:+.1f}%"
    if cur_val > 0:
        return f"+{int(cur_val)} vs 0"
    return "stable"


def _build_sitrep_alert_summary(alert_df: pd.DataFrame) -> Optional[str]:
    """Construit une phrase courte sur les signaux statistiques disponibles."""
    if alert_df is None or not isinstance(alert_df, pd.DataFrame) or alert_df.empty:
        return None

    signal_col = next(
        (col for col in ["signal", "Signal", "alerte", "Alerte"] if col in alert_df.columns),
        None,
    )
    if signal_col is not None:
        signals = _surveillance_clean_text_series(alert_df[signal_col])
        if not signals.empty:
            counts = signals.value_counts().head(3)
            details = [f"{label} ({int(count)})" for label, count in counts.items()]
            return "Signaux statistiques à vérifier sur la dernière semaine disponible : " + ", ".join(details) + "."

    return (
        f"{format_metric_value(len(alert_df))} signaux statistiques sont remontés sur la dernière "
        "semaine disponible et nécessitent une vérification."
    )


def _build_sitrep_critical_cfr_summary(payload: dict[str, Any]) -> Optional[str]:
    """Résume les unités géographiques à létalité élevée pour le SITREP."""
    provcrit = payload.get("prov_cfr_crit")
    zscrit = payload.get("zs_cfr_crit")
    parts: list[str] = []

    if isinstance(provcrit, pd.DataFrame) and not provcrit.empty and COL_PROV in provcrit.columns:
        focus = [
            f"{row[COL_PROV]} ({format_metric_value(row['CFR_%'], decimals=1)}%)"
            for _, row in provcrit.head(3).iterrows()
        ]
        if focus:
            parts.append("provinces " + ", ".join(focus))

    if isinstance(zscrit, pd.DataFrame) and not zscrit.empty and COL_ZS in zscrit.columns:
        focus = []
        for _, row in zscrit.head(5).iterrows():
            if COL_PROV in zscrit.columns:
                label = f"{row[COL_PROV]} / {row[COL_ZS]}"
            else:
                label = str(row[COL_ZS])
            focus.append(f"{label} ({format_metric_value(row['CFR_%'], decimals=1)}%)")
        if focus:
            parts.append("zones de santé " + ", ".join(focus))

    if not parts:
        return None

    return "La létalité doit être vérifiée en priorité dans les " + " ; ".join(parts) + "."


def _build_sitrep_summary_lines(payload: dict[str, Any]) -> list[str]:
    """Assemble les messages-clés du SITREP à partir du périmètre sélectionné."""
    lines: list[str] = []
    current_df = payload.get("selected_df", pd.DataFrame())
    cumulative_df = payload.get("cumulative_df", pd.DataFrame())
    previous_df = payload.get("previous_df", pd.DataFrame())
    current_label = payload.get("selected_week_label")
    previous_label = payload.get("previous_week_label")

    if isinstance(current_df, pd.DataFrame) and not current_df.empty:
        overview_text = _build_scope_overview_text(
            current_df,
            scope_kind="weekly",
            latest_week_df=current_df,
            latest_label=current_label,
        )
        if overview_text:
            lines.append(overview_text)

        comparison_text = _build_comparison_sentence(
            current_df=current_df,
            previous_df=previous_df if isinstance(previous_df, pd.DataFrame) and not previous_df.empty else None,
            current_label=current_label,
            previous_label=previous_label,
        )
        if comparison_text:
            lines.append(comparison_text)

        top_prov_text = _build_top_province_summary_text(current_df)
        if top_prov_text:
            lines.append(top_prov_text)

        investigation_text = _build_investigation_summary_text(current_df)
        if investigation_text:
            lines.append(investigation_text)

        tdr_text = _build_tdr_summary_text(current_df)
        if tdr_text:
            lines.append(tdr_text)
    else:
        lines.append(
            f"Aucun cas n’est rapporté pour {current_label or 'la semaine sélectionnée'} dans le périmètre filtré."
        )

    if isinstance(cumulative_df, pd.DataFrame) and not cumulative_df.empty:
        cumulative_text = _build_scope_overview_text(
            cumulative_df,
            scope_kind="cumulative",
            latest_week_df=current_df if isinstance(current_df, pd.DataFrame) and not current_df.empty else None,
            latest_label=current_label,
        )
        if cumulative_text:
            lines.append(cumulative_text)

    critical_text = _build_sitrep_critical_cfr_summary(payload)
    if critical_text:
        lines.append(critical_text)

    alert_text = _build_sitrep_alert_summary(payload.get("alertes_last"))
    if alert_text:
        lines.append(alert_text)

    return lines


def _build_sitrep_action_lines(payload: dict[str, Any]) -> list[str]:
    """Construit des priorités opérationnelles courtes pour le SITREP."""
    current_df = payload.get("selected_df", pd.DataFrame())
    current_label = payload.get("selected_week_label", "la semaine sélectionnée")
    actions: list[str] = []

    if not isinstance(current_df, pd.DataFrame) or current_df.empty:
        return [
            f"Vérifier si l’absence de cas en {current_label} reflète réellement la situation ou un retard de notification."
        ]

    total_cases = int(len(current_df))
    top_prov = payload.get("top_prov_focus")
    if isinstance(top_prov, pd.DataFrame) and not top_prov.empty and total_cases > 0:
        leader = top_prov.iloc[0]
        leader_share = safe_pct(leader["Cas"], total_cases)
        actions.append(
            f"Cibler en priorité {leader[COL_PROV]}, qui concentre {leader_share:.1f}% des cas de {current_label}."
        )

    critical_text = _build_sitrep_critical_cfr_summary(payload)
    if critical_text:
        actions.append(critical_text)

    alert_text = _build_sitrep_alert_summary(payload.get("alertes_last"))
    if alert_text:
        actions.append(alert_text)

    if COL_TDR in current_df.columns:
        n_tdr = int(_is_yes_series(current_df[COL_TDR]).sum())
        if n_tdr <= 0:
            actions.append(
                "Aucun TDR n’est documenté sur la semaine sélectionnée : renforcer la confirmation biologique si elle est attendue."
            )

    if DATE_INV in current_df.columns:
        investigated = int(pd.to_datetime(current_df[DATE_INV], errors="coerce").notna().sum())
        if investigated < total_cases:
            actions.append(
                f"Compléter l’investigation de {format_metric_value(total_cases - investigated)} cas pour consolider la lecture opérationnelle."
            )

    return actions[:4]


# =========================
# NAVIGATION COMPACTE
# - Surveillance & promptitude = anciens onglets 1 + 2 + 3
# - Profil descriptif = anciens onglets 4 + 4b
# - Données, complétude & qualité = anciens onglets 5 + 6 + 7
# - Cartographie, SITREP, IDSR et IREP restent dédiés
# =========================

# =========================
# TAB 1: DYNAMIQUE ÉPIDÉMIOLOGIQUE ET PROMPTITUDE
# =========================
with tab_surveillance:
    if IDSR_MODE:
        st.info("Mode **IDSR agrégé hebdomadaire** : les analyses de liste linéaire sont désactivées dans cet espace. Veuillez utiliser l’onglet **IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : suivre la situation selon trois niveaux de lecture complémentaires pour distinguer l’hebdomadaire, la tendance récente et le cumul.

            **📖 Logique de lecture**
            - La **situation hebdomadaire** reflète la semaine la plus récente visible dans la plage filtrée.
            - La **situation des 4 dernières semaines** permet de lire rapidement la tendance récente.
            - La **situation cumulée** consolide l’ensemble de la fenêtre active pour orienter la réponse.

            **⚠️ Point d’attention**
            - Les indicateurs dépendent directement des filtres actifs de semaines, de géographie et de classification.
            """,
            expanded=False
        )

        render_section_title(1, "Surveillance épidémiologique")
        st.caption(
            "Cette organisation permet une lecture progressive de la situation à partir de la plage de semaines active dans la barre latérale."
        )

        df_surv_scope, surv_reference = _prepare_surveillance_period_scope(df_f)

        if not surv_reference.empty:
            latest_order = surv_reference["order"].iloc[-1]
            latest_label = str(surv_reference["label"].iloc[-1])
            last4_reference = surv_reference.tail(4).copy()
            last4_orders = last4_reference["order"].tolist()
            last4_labels = last4_reference["label"].astype(str).tolist()
            first_label = str(surv_reference["label"].iloc[0])
            previous_week_reference = surv_reference.tail(2).head(1).copy() if len(surv_reference) >= 2 else pd.DataFrame()
            previous_week_df = pd.DataFrame()
            previous_week_label = None
            if not previous_week_reference.empty:
                previous_week_label = str(previous_week_reference["label"].iloc[0])
                previous_week_df = df_surv_scope[
                    df_surv_scope["_surv_order"] == previous_week_reference["order"].iloc[0]
                ].copy()

            prev4_reference = (
                surv_reference.iloc[max(len(surv_reference) - 8, 0): max(len(surv_reference) - 4, 0)].copy()
                if len(surv_reference) > 4 else pd.DataFrame()
            )
            prev4_df = pd.DataFrame()
            prev4_label = None
            if not prev4_reference.empty:
                prev4_orders = prev4_reference["order"].tolist()
                prev4_df = df_surv_scope[df_surv_scope["_surv_order"].isin(prev4_orders)].copy()
                prev4_labels = prev4_reference["label"].astype(str).tolist()
                if prev4_labels:
                    prev4_label = ", ".join(prev4_labels)

            df_latest_week = df_surv_scope[df_surv_scope["_surv_order"] == latest_order].copy()
            df_last4_weeks = df_surv_scope[df_surv_scope["_surv_order"].isin(last4_orders)].copy()

            _render_surveillance_window(
                "1. Situation hebdomadaire",
                df_latest_week,
                f"Semaine la plus récente dans la fenêtre filtrée : {latest_label}.",
                "Aucune donnée n’est disponible pour la semaine hebdomadaire sélectionnée.",
                narrative_context={
                    "scope_kind": "weekly",
                    "current_label": latest_label,
                    "comparison_df": previous_week_df if not previous_week_df.empty else None,
                    "comparison_label": previous_week_label,
                    "latest_week_df": df_latest_week,
                    "latest_label": latest_label,
                },
            )

            st.divider()

            _render_surveillance_window(
                "2. Situation des 4 dernières semaines",
                df_last4_weeks,
                f"Lecture glissante sur les {len(last4_labels)} semaines les plus récentes de la sélection : {', '.join(last4_labels)}.",
                "Aucune donnée n’est disponible pour construire la tendance des 4 dernières semaines.",
                narrative_context={
                    "scope_kind": "recent4",
                    "current_label": "Les 4 dernières semaines",
                    "comparison_df": prev4_df if not prev4_df.empty else None,
                    "comparison_label": "les 4 semaines précédentes" if prev4_label else None,
                    "latest_week_df": df_latest_week,
                    "latest_label": latest_label,
                },
            )

            st.divider()

            _render_surveillance_window(
                "3. Situation cumulée",
                df_surv_scope,
                f"Cumul de toute la fenêtre active : {first_label} à {latest_label}.",
                "Aucune donnée n’est disponible pour la situation cumulée.",
                narrative_context={
                    "scope_kind": "cumulative",
                    "current_label": f"{first_label} à {latest_label}",
                    "comparison_df": None,
                    "comparison_label": None,
                    "latest_week_df": df_latest_week,
                    "latest_label": latest_label,
                },
            )


            if (
                COL_PROV in df_surv_scope.columns
                and df_surv_scope[COL_PROV].notna().any()
                and "_surv_label" in df_surv_scope.columns
                and df_surv_scope["_surv_label"].notna().any()
            ):
                st.divider()
                st.markdown("### Courbe épidémiologique des cas par province")
                prov_totals = df_surv_scope[[COL_PROV]].dropna().copy()
                prov_totals["_prov"] = prov_totals[COL_PROV].astype(str).str.strip()
                prov_totals = prov_totals[prov_totals["_prov"] != ""]
                prov_options = prov_totals["_prov"].value_counts().index.tolist()
                default_provs = prov_options if len(prov_options) <= 10 else prov_options[:10]
                selected_curve_provs = st.multiselect(
                    "Provinces à afficher",
                    options=prov_options,
                    default=default_provs,
                    key="surveillance_multi_curve_provinces",
                    help="Tu peux aussi cliquer sur la légende du graphique pour masquer ou afficher une province.",
                )
                if selected_curve_provs:
                    fig_multi_prov = build_weekly_multiline_by_group(
                        df=df_surv_scope,
                        week_col="_surv_label",
                        group_col=COL_PROV,
                        selected_groups=selected_curve_provs,
                        titre=" ",
                        x_titre="Semaine épidémiologique",
                        y_titre="Nombre de cas",
                        rotation=45,
                        pas_x=int(pas_x),
                        annot=annot_vals,
                        taille_fig=(1500, 700),
                    )
                    if fig_multi_prov is not None:
                        st.plotly_chart(
                            fig_multi_prov,
                            width="stretch",
                            key="surveillance_multi_curve_province",
                        )
                        st.caption(
                            "Astuce : clique sur une province dans la légende pour masquer ou afficher sa courbe. "
                            "Double-clique pour isoler une province."
                        )
                    else:
                        st.info("Aucune courbe exploitable n'a pu être construite pour les provinces sélectionnées.")
                else:
                    st.info("Sélectionne au moins une province pour afficher la courbe épidémiologique multi-provinces.")
        else:
            st.info("Variable semaine indisponible : aucune colonne temporelle exploitable n’a été détectée pour organiser la surveillance par fenêtres.")
    # Section suivante : promptitude. Les indicateurs de performance et de létalité déjà présentés plus haut ne sont pas répétés ici afin d’éviter les redondances.

with tab_surveillance:
    st.divider()
    render_section_title(2, "Promptitude de notification, investigation et prise en charge")
    if IDSR_MODE:
        st.info("Mode **IDSR agrégé hebdomadaire** : les analyses de liste linéaire sont désactivées dans cet espace. Veuillez utiliser l’onglet **IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            f"""
            **🎯 Objectif** : Mesurer la rapidité de détection et d’accès aux soins.
        
            **📖 Indicateurs**
            - Délai **début maladie → admission**
            - Délai **début maladie → prélèvement**
            - **% ≤ {seuil_jours} jours** : proportion de cas pris en charge rapidement.
        
            **⚠️ Points d’attention**
            - Des délais longs augmentent le risque de transmission communautaire.
            - Des délais négatifs ou extrêmes = erreurs de saisie ou dates incorrectes.
            """,
            expanded=False
        )
        
        st.subheader("Analyse de la promptitude des principales étapes du parcours du cas et de la notification")
        
        delais_cols = [c for c in ["delai_onset_to_adm", "delai_onset_to_prel"] if c in df_f.columns]
        
        if not delais_cols:
            st.info("Les analyses de délais sont indisponibles : les dates nécessaires sont absentes ou non exploitables.")
        else:
            df_del = df_f.copy()
            for c in delais_cols:
                df_del.loc[df_del[c] < 0, c] = np.nan

            delay_summary_std = build_standard_delay_summary(df_del)
            available_delay_pairs = list_available_standard_delays(df_del)
            seuil_val = float(seuil_jours)
            seuil_lab = int(seuil_val) if seuil_val.is_integer() else round(seuil_val, 1)

            st.markdown("**Lecture opérationnelle prioritaire**")
            st.caption(
                "Les indicateurs et classements ci-dessous sont plus utiles pour l’action immédiate que la distribution graphique brute."
            )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                p1, n1 = pct_under_threshold(df_del.get("delai_onset_to_adm"), seuil_jours)
                st.metric("Admission ≤ seuil (%)", "-" if np.isnan(p1) else f"{p1:.1f}", help=f"n = {n1}")
            with c2:
                p2, n2 = pct_under_threshold(df_del.get("delai_onset_to_prel"), seuil_jours)
                st.metric("Prélèvement ≤ seuil (%)", "-" if np.isnan(p2) else f"{p2:.1f}", help=f"n = {n2}")
            with c3:
                st.metric("Délais admission documentés", format_metric_value(n1))
            with c4:
                st.metric("Délais prélèvement documentés", format_metric_value(n2))

            if not delay_summary_std.empty:
                st.markdown("**Résumé standard des délais disponibles**")
                st_dataframe_safe(delay_summary_std, height=320)

            if COL_PROV in df_del.columns:
                st.markdown("**Provinces à surveiller en priorité**")
                ranking_specs = []
                if "delai_onset_to_adm" in delais_cols:
                    ranking_specs.append(("delai_onset_to_adm", "Début maladie → admission"))
                if "delai_onset_to_prel" in delais_cols:
                    ranking_specs.append(("delai_onset_to_prel", "Début maladie → prélèvement"))

                if ranking_specs:
                    rank_cols = st.columns(len(ranking_specs))
                    for col_ui, (delay_col, delay_label) in zip(rank_cols, ranking_specs):
                        delay_group_tbl = build_delay_group_summary(
                            df_del,
                            delay_col=delay_col,
                            group_col=COL_PROV,
                            threshold=seuil_jours,
                        )
                        pct_col = f"% <= {seuil_lab} j"
                        if not delay_group_tbl.empty and pct_col in delay_group_tbl.columns:
                            delay_group_tbl = delay_group_tbl[
                                pd.to_numeric(delay_group_tbl["n"], errors="coerce").fillna(0) > 0
                            ].copy()
                            delay_group_tbl = (
                                delay_group_tbl
                                .sort_values([pct_col, "Mediane_j", "n"], ascending=[True, False, False], na_position="last")
                                .head(10)
                                .copy()
                            )
                        with col_ui:
                            st.markdown(f"**{delay_label}**")
                            if delay_group_tbl.empty:
                                st.info("Pas assez de données exploitables pour établir un classement provincial.")
                            else:
                                st.dataframe(delay_group_tbl, width="stretch", height=360, hide_index=True)

            with st.expander("Distribution détaillée des délais observés (graphique)", expanded=False):
                if use_custom_viz and HAS_CUSTOM_VIZ:
                    fig = plot_boxplot_delais_plotly(
                        df=df_del,
                        colonnes_delais=delais_cols,
                        col_groupe=COL_PROV if COL_PROV in df_del.columns else None,
                        titre=" ",
                        taille_fig=(1500, 600),
                        rotation=45
                    )
                    st_plot(fig, key="boxplot_delais_custom")
                else:
                    long = df_del.melt(value_vars=delais_cols, var_name="Type_delai", value_name="Jours").dropna()
                    fig = px.box(long, x="Type_delai", y="Jours", points="outliers", title="Boxplot des délais (global)")
                    fig = apply_plotly_value_annotations(fig, annot_vals)
                    st.plotly_chart(fig, width="stretch")

            if available_delay_pairs:
                st.divider()
                st.markdown("**Analyse détaillée d'un délai standard**")
                delay_label_to_col = {label: col for col, label in available_delay_pairs}
                delay_focus_label = st.selectbox(
                    "Délai standard à profiler",
                    options=list(delay_label_to_col.keys()),
                    key="timeliness_delay_focus",
                )

                group_candidates = []
                for c in [COL_PROV, COL_ZS, pick_age_col(df_del), COL_SEX, COL_CLASS]:
                    if c and c in df_del.columns and df_del[c].notna().any() and c not in group_candidates:
                        group_candidates.append(c)

                if group_candidates:
                    g1, g2, g3 = st.columns([1.15, 1.15, 0.9])
                    with g1:
                        delay_group_focus = st.selectbox(
                            "Variable de regroupement",
                            options=group_candidates,
                            key="timeliness_group_focus",
                        )
                    with g2:
                        delay_metric_focus = st.selectbox(
                            "Indicateur a classer",
                            options=["Mediane (jours)", f"% <= {seuil_jours} jours"],
                            key="timeliness_metric_focus",
                        )
                    with g3:
                        delay_topn = st.slider(
                            "Top groupes",
                            min_value=5,
                            max_value=30,
                            value=15,
                            step=1,
                            key="timeliness_group_topn",
                        )

                    delay_focus_col = delay_label_to_col[delay_focus_label]
                    delay_group_tbl = build_delay_group_summary(
                        df_del,
                        delay_col=delay_focus_col,
                        group_col=delay_group_focus,
                        threshold=seuil_jours,
                    )

                    if not delay_group_tbl.empty:
                        pct_col = f"% <= {seuil_lab} j"
                        sort_col = "Mediane_j" if delay_metric_focus.startswith("Mediane") else pct_col
                        ascending = bool(delay_metric_focus.startswith("Mediane"))
                        delay_group_view = (
                            delay_group_tbl.sort_values(sort_col, ascending=ascending, na_position="last")
                            .head(int(delay_topn))
                            .copy()
                        )

                        t1, t2 = st.columns([1.05, 1.35])
                        with t1:
                            st.dataframe(delay_group_view, width="stretch", height=420, hide_index=True)
                        with t2:
                            plot_df = delay_group_view.sort_values(sort_col, ascending=True, na_position="last")
                            fig_delay_focus = px.bar(
                                plot_df,
                                x=sort_col,
                                y=delay_group_focus,
                                orientation="h",
                                text=sort_col,
                                title=f"{delay_focus_label} par {delay_group_focus}",
                                color=sort_col,
                                color_continuous_scale=["#dbe8f9", "#2b74ca"],
                            )
                            fig_delay_focus.update_layout(
                                coloraxis_showscale=False,
                                xaxis_title=sort_col,
                                yaxis_title=delay_group_focus,
                            )
                            fig_delay_focus = apply_plotly_value_annotations(fig_delay_focus, annot_vals)
                            st.plotly_chart(fig_delay_focus, width="stretch", key="timeliness_delay_focus_chart")
                    else:
                        st.info("Le délai sélectionné ne dispose pas d'assez de données exploitables pour ce regroupement.")
                else:
                    st.info("Aucune variable standard de regroupement n'est disponible pour profiler les délais.")

    # =========================
    # TAB 4: Démographie
    # =========================
with tab_profil:
    if IDSR_MODE:
        st.info("Mode **IDSR agrégé hebdomadaire** : les analyses de liste linéaire sont désactivées dans cet espace. Veuillez utiliser l’onglet **IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Identifier les groupes les plus touchés.
        
            **📖 Interprétation**
            - Répartition **sexe** : différences d’exposition ou d’accès aux soins.
            - Répartition **âge** : identifie les groupes vulnérables/à risque.
            - **Pyramide âge/sexe** : profil de transmission (domicile, école, activités, etc.).
        
            **⚠️ Points d’attention**
            - Vérifier la complétude de l’âge et du sexe : beaucoup de “Inconnu” biaise la lecture.
            """,
            expanded=False
        )
        
      
        st.divider()

        st.subheader("Contrôle qualité des variables d’âge")

        # --- Indicateurs rapides ---
        n_total = len(df_f)

        # Manquants âge: on considère Age OU une tranche (Tranche_age/Tranche_age_en_ans)
        has_age_num = (COL_AGE in df_f.columns)
        has_tr4 = (COL_AGEG2 in df_f.columns)
        has_tr5 = (COL_AGEG in df_f.columns)

        age_num_na = df_f[COL_AGE].isna() if has_age_num else pd.Series([True]*n_total, index=df_f.index)
        tr4_na = df_f[COL_AGEG2].isna() if has_tr4 else pd.Series([True]*n_total, index=df_f.index)
        tr5_na = df_f[COL_AGEG].isna() if has_tr5 else pd.Series([True]*n_total, index=df_f.index)

        missing_age_mask = age_num_na & tr4_na & tr5_na
        pct_age_missing = float(missing_age_mask.mean() * 100.0) if n_total else 0.0

        # Unité incohérente
        incoh_mask = pd.Series([False]*n_total, index=df_f.index)
        if COL_UNIT in df_f.columns and df_f[COL_UNIT].notna().any():
            u = df_f[COL_UNIT].astype("string").str.lower().str.strip()
            ok = (
                u.str.contains(AGE_UNIT_YEAR_PATTERN, na=False)
                | u.str.contains(AGE_UNIT_MONTH_PATTERN, na=False)
                | u.str.contains(AGE_UNIT_WEEK_PATTERN, na=False)
                | u.str.contains(AGE_UNIT_DAY_PATTERN, na=False)
            )
            incoh_mask = u.notna() & (~ok)
        pct_unit_incoh = float(incoh_mask.mean() * 100.0) if n_total else 0.0

        # Âges extrêmes (convertis en années quand possible)
        years = infer_age_years_generic(df_f) if has_age_num else pd.Series([np.nan]*n_total, index=df_f.index)
        extreme_mask = years.notna() & ((years < 0) | (years > 110))
        pct_extreme = float(extreme_mask.mean() * 100.0) if n_total else 0.0

        # --- Affichage KPI ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Âge manquant", f"{pct_age_missing:.1f}%")
        m2.metric("Unité âge incohérente", f"{pct_unit_incoh:.1f}%")
        m3.metric("Âges extrêmes (<0 ou >110 ans)", f"{pct_extreme:.1f}%")
        m4.metric("N (après filtres)", f"{n_total:,}".replace(",", " "))

        with st.expander("Détails qualité (unités, âges extrêmes)"):
            if COL_UNIT in df_f.columns:
                unit_dist = (
                    df_f[COL_UNIT].astype("string").fillna("NA").str.lower().str.strip()
                    .value_counts().reset_index()
                )
                unit_dist.columns = ["Unite_age (valeur)", "N"]
                st.dataframe(unit_dist, width="stretch", height=260)
            else:
                st.info("La variable Unite_age est absente du fichier analysé.")

            if extreme_mask.any():
                show_cols = [c for c in [COL_PROV, COL_ZS, COL_AGE, COL_UNIT, DATE_ONSET, DATE_ADM, DATE_NOTIF] if c in df_f.columns]
                df_ext = df_f.loc[extreme_mask, show_cols].copy().head(50)
                df_ext.insert(0, "Age_en_ans_estime", years.loc[extreme_mask].head(50).round(2).values)
                st.warning("Exemples de valeurs extrêmes (maximum 50) à vérifier et corriger si nécessaire.")
                st.dataframe(df_ext, width="stretch", height=320)
            else:
                st.success("Aucune valeur d’âge extrême n’a été détectée selon les règles en vigueur.")

      
    # =========================
    # TAB 4B: Analyse descriptive standard
    # =========================
with tab_profil:
    st.divider()
    render_section_title(3, "Analyse descriptive selon le modèle Temps-Lieu-Personne")
    if IDSR_MODE:
        st.info("Mode **IDSR agrégé hebdomadaire** : les analyses de liste linéaire sont désactivées dans cet espace. Veuillez utiliser l’onglet **IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : présenter une analyse descriptive conforme à une logique standard.

            **📖 Structure **
            - Vue d'ensemble
            - Personne
            - Lieu
            - Laboratoire
            - Tableaux descriptifs complémentaires
            """,
            expanded=False
        )

        st.subheader("Résumé automatisé conforme au langage de surveillance")
        st.info(build_who_narrative_summary(df_f))

        st.subheader("1. Situation générale")
        st_dataframe_safe(build_global_summary_table(df_f))
                
        st.divider()
        st.subheader("2. Dimension personne — tableaux détaillés et structure avancée")
        st.caption("Les visuels rapides sexe, âge et pyramide de synthèse sont regroupés sur la page d’accueil. Ici, l’accent est mis sur les tableaux analytiques détaillés.")
        a1, a2 = st.columns(2)
        with a1:
            if COL_SEX in df_f.columns:
                sex_tbl = build_frequency_table(df_f, COL_SEX)
                st.markdown("**Table de fréquence par sexe**")
                st_dataframe_safe(sex_tbl)
            else:
                st.info("La variable Sexe est absente du fichier analysé.")
        with a2:
            age_display_col = None
            if COL_AGEG2 in df_f.columns and df_f[COL_AGEG2].notna().any():
                age_display_col = COL_AGEG2
            elif COL_AGEG in df_f.columns and df_f[COL_AGEG].notna().any():
                age_display_col = COL_AGEG
            if age_display_col is not None:
                age_tbl = build_frequency_table(df_f, age_display_col)
                st.markdown(f"**Table de fréquence par {age_display_col}**")
                st_dataframe_safe(age_tbl)
            else:
                years = infer_age_years_generic(df_f)
                if years.notna().any():
                    age_num = pd.DataFrame({'Age_en_ans': years.dropna()})
                    st.markdown("**Résumé statistique de l’âge en années**")
                    st.dataframe(age_num.describe().T, width='stretch')
                else:
                    st.info("Aucune information d’âge exploitable n’a été détectée.")

        df_desc = df_f.copy()
        df_desc['Tranche_age_4cat_std'] = derive_age_4cat_generic(df_desc)
        df_desc['Tranche_age_5ans_std'] = derive_age_5yr_generic(df_desc)
        if use_custom_viz and HAS_CUSTOM_VIZ and age_col and COL_SEX in df_desc.columns and COL_PROV in df_desc.columns:
            st.markdown("**Structure âge-sexe détaillée par province**")
            fig = graphique_pyramide_age(
                df=df_desc,
                col_tranche=age_col,
                col_sexe=COL_SEX,
                col_valeur=COL_UNIT if COL_UNIT in df_desc.columns else COL_SEX,
                valeurs_neg=['Masculin', 'Homme', 'M'],
                titre='Pyramides âge-sexe par province',
                seuil_min=10,
                croissant=False,
                afficher_signe_negatif_dans_label=False,
                facette_col=COL_PROV,
                annot=annot_vals,
                taille_fig=(1500, 900),
                return_fig=True,
                couleur_contour_facette="#777772"
            )
            st_plot(fig, key='oms_pyr_faceted_prov')
        else:
            st.info("La structure âge-sexe détaillée par province n’est pas disponible : Province, Sexe et une variable de tranche d’âge sont requis.")

        st.divider()
        st.subheader("3. Dimension lieu — répartition par province, zone de santé et aire de santé")
        geo_cols = [c for c in [COL_PROV, COL_ZS, COL_AS] if c in df_f.columns]
        if geo_cols:
            geo_choice = st.selectbox('Niveau géographique d’analyse', geo_cols, key='oms_geo_choice')
            top_n_geo = st.slider('Nombre de catégories à afficher', 5, 30, 15, key='oms_top_geo')
            geo_tbl = build_frequency_table(df_f, geo_choice, top_n=top_n_geo)
            fig = px.bar(geo_tbl, x=geo_choice, y='n', title=f'Répartition des cas par {geo_choice}')
            fig.update_layout(xaxis_tickangle=-45)
            fig = apply_plotly_value_annotations(fig, annot_vals)
            st.plotly_chart(fig, width='stretch', key='oms_geo_bar')
            st_dataframe_safe(geo_tbl)
        else:
            st.info("Aucune variable géographique standard n’a été détectée.")

        st.divider()
        st.subheader("4. Composante laboratoire — résumé opérationnel")
        lab_tbl = build_simple_lab_table(df_f)
        if not lab_tbl.empty:
            l1, l2 = st.columns([1, 1])
            with l1:
                st_dataframe_safe(lab_tbl)
            with l2:
                fig = px.bar(lab_tbl, x='Indicateur labo', y='n', title='Résumé des indicateurs de laboratoire')
                fig.update_layout(xaxis_tickangle=-45)
                fig = apply_plotly_value_annotations(fig, annot_vals)
                st.plotly_chart(fig, width='stretch', key='oms_lab_bar')

            weekly_lab = build_weekly_lab_summary(df_f)
            if not weekly_lab.empty:
                st.markdown("**Suivi hebdomadaire des tests valides, tests positifs et taux de positivité**")
                fig_lab_combo = go.Figure()
                fig_lab_combo.add_trace(
                    go.Bar(
                        x=weekly_lab["Semaine"],
                        y=weekly_lab["Tests valides"],
                        name="Tests valides",
                        marker_color="#4f81bd",
                    )
                )
                fig_lab_combo.add_trace(
                    go.Bar(
                        x=weekly_lab["Semaine"],
                        y=weekly_lab["Tests positifs"],
                        name="Tests positifs",
                        marker_color="#d97b16",
                    )
                )
                fig_lab_combo.add_trace(
                    go.Scatter(
                        x=weekly_lab["Semaine"],
                        y=weekly_lab["Positivité (%)"],
                        name="Positivité (%)",
                        mode="lines+markers",
                        line=dict(color="#b9353f", width=3),
                        marker=dict(size=8),
                        yaxis="y2",
                    )
                )
                fig_lab_combo.update_layout(
                    title=" ",
                    barmode="group",
                    xaxis_title="Semaine épidémiologique",
                    yaxis_title="Nombre de tests",
                    yaxis2=dict(
                        title="Positivité (%)",
                        overlaying="y",
                        side="right",
                        rangemode="tozero",
                    ),
                )
                st_plot(fig_lab_combo, key="lab_weekly_combo", annotate_values=False)
                with st.expander("Afficher la table hebdomadaire des indicateurs laboratoire", expanded=False):
                    st_dataframe_safe(weekly_lab, height=320)

            has_tdr_chain = COL_TDR in df_f.columns and df_f[COL_TDR].notna().any()
            coverage_label = "TDR réalisé (%)" if has_tdr_chain else "Tests documentés (%)"
            positivity_label = "Positivité TDR (%)" if has_tdr_chain else "Positivité labo (%)"

            if COL_PROV in df_f.columns:
                st.markdown("**Tableau provincial consolidé des indicateurs clés de surveillance**")
                prov_kpi = compute_group_indicators(df_f, COL_PROV).sort_values("Cas", ascending=False).head(15).copy()
                prov_kpi = _normalize_metric_alias_columns(prov_kpi)
                prov_kpi = prov_kpi.rename(
                    columns={
                        "Décès": "Décès",
                        "CFR_%": "CFR (%)",
                        "Prélèvement_%": "Prélèvement (%)",
                        "Hospitalisation_%": "Hospitalisation (%)",
                        "TDR_réalisé_%": coverage_label,
                        "Positivité_TDR_%": positivity_label,
                    }
                )
                st_dataframe_safe(prov_kpi, height=420)
        else:
            st.info("Aucune variable laboratoire simple n’a été détectée (prélèvement, TDR ou résultat).")

        st.divider()
        st.subheader("5. Indicateurs standards stratifiés")
        st.caption(
            "Vue transversale standard des cas, décès, CFR et indicateurs de surveillance, "
            "applicable à toute line list standardisée."
        )

        strat_age_col = pick_age_col(df_f)
        strat_candidates = []
        for c in [COL_SEX, strat_age_col, COL_PROV, COL_ZS, COL_AS, COL_CLASS]:
            if c and c in df_f.columns and df_f[c].notna().any() and c not in strat_candidates:
                strat_candidates.append(c)

        if strat_candidates:
            metric_map = {
                "Cas": "Cas",
                "Décès": "Décès",
                "CFR (%)": "CFR (%)",
                "Prélèvement (%)": "Prélèvement (%)",
                "Hospitalisation (%)": "Hospitalisation (%)",
                coverage_label: coverage_label,
                positivity_label: positivity_label,
            }

            s_cfg1, s_cfg2, s_cfg3 = st.columns([1.15, 1.15, 0.9])
            with s_cfg1:
                strat_choice = st.selectbox(
                    "Variable de stratification",
                    options=strat_candidates,
                    key="std_strat_choice",
                )
            with s_cfg2:
                strat_metric_label = st.selectbox(
                    "Indicateur à classer",
                    options=list(metric_map.keys()),
                    index=0,
                    key="std_strat_metric",
                )
            with s_cfg3:
                strat_topn = st.slider(
                    "Top modalités",
                    min_value=5,
                    max_value=30,
                    value=15,
                    step=1,
                    key="std_strat_topn",
                )

            strat_tbl = compute_group_indicators(df_f, strat_choice).copy()
            strat_tbl = _normalize_metric_alias_columns(strat_tbl)
            strat_tbl = strat_tbl.rename(
                columns={
                    "Décès": "Décès",
                    "CFR_%": "CFR (%)",
                    "Prélèvement_%": "Prélèvement (%)",
                    "Hospitalisation_%": "Hospitalisation (%)",
                    "TDR_réalisé_%": coverage_label,
                    "Positivité_TDR_%": positivity_label,
                }
            )

            if not strat_tbl.empty:
                total_cases_strat = pd.to_numeric(strat_tbl["Cas"], errors="coerce").sum()
                strat_tbl["Part des cas (%)"] = np.where(
                    total_cases_strat > 0,
                    (pd.to_numeric(strat_tbl["Cas"], errors="coerce") / total_cases_strat) * 100.0,
                    np.nan,
                ).round(1)

                sort_col = metric_map[strat_metric_label]
                strat_view = (
                    strat_tbl.sort_values(sort_col, ascending=False, na_position="last")
                    .head(int(strat_topn))
                    .copy()
                )

                s_tbl, s_fig = st.columns([1.05, 1.35])
                with s_tbl:
                    st.dataframe(strat_view, width="stretch", height=430, hide_index=True)
                with s_fig:
                    plot_df = strat_view.sort_values(sort_col, ascending=True, na_position="last")
                    fig_strat = px.bar(
                        plot_df,
                        x=sort_col,
                        y=strat_choice,
                        orientation="h",
                        text=sort_col,
                        title=f"{strat_metric_label} par {strat_choice}",
                        color=sort_col,
                        color_continuous_scale=["#e7f1df", "#2d7d46"],
                    )
                    fig_strat.update_layout(
                        coloraxis_showscale=False,
                        xaxis_title=strat_metric_label,
                        yaxis_title=strat_choice,
                    )
                    fig_strat = apply_plotly_value_annotations(fig_strat, annot_vals)
                    st.plotly_chart(fig_strat, width="stretch", key="std_strat_chart")
            else:
                st.info("Les indicateurs standards sont indisponibles pour la variable de stratification sélectionnée.")
        else:
            st.info("Aucune variable standard exploitable n'est disponible pour une stratification transversale.")

        st.divider()
        st.subheader("6. Tableaux descriptifs des variables catégorielles")
        st.caption("Les analyses de délais sont centralisées dans l’onglet Surveillance afin d’éviter leur répétition ici.")
        default_cat_candidates = [COL_SEX, COL_PROV, COL_ZS, COL_AS, COL_AGEG2, COL_AGEG, COL_ISSUE, COL_PREL, COL_TDR, COL_TDRR, COL_HOSP, COL_DEHY, COL_CLASS]
        cat_candidates = [c for c in default_cat_candidates if c in df_f.columns]
        extra_candidates = [c for c in df_f.columns if (not is_numeric_dtype(df_f[c])) and c not in cat_candidates]
        cat_options = cat_candidates + extra_candidates[:20]
        if cat_options:
            cat_choice = st.multiselect('Variables catégorielles à décrire', cat_options, default=cat_candidates[:4], key='oms_cat_choice')
            for c in cat_choice:
                with st.expander(f'Fréquences — {c}', expanded=False):
                    if c == COL_ZS and COL_PROV in df_f.columns:
                        tbl = (
                            df_f.assign(
                                _province=df_f[COL_PROV].fillna('Inconnu').astype(str).str.strip().replace('', 'Inconnu'),
                                _modalite=df_f[COL_ZS].fillna('Inconnu').astype(str).str.strip().replace('', 'Inconnu'),
                            )
                            .groupby(['_province', '_modalite'], dropna=False)
                            .size()
                            .reset_index(name='n')
                            .rename(columns={
                                '_province': 'Province de notification',
                                '_modalite': 'Zone_de_sante_notification',
                            })
                        )
                        tbl['%'] = (tbl['n'] / max(len(df_f), 1) * 100).round(1)
                        tbl = tbl.sort_values(['n', 'Province de notification', 'Zone_de_sante_notification'], ascending=[False, True, True])
                        st_dataframe_safe(tbl)
                    elif c == COL_AS and COL_PROV in df_f.columns and COL_ZS in df_f.columns:
                        tbl = (
                            df_f.assign(
                                _province=df_f[COL_PROV].fillna('Inconnu').astype(str).str.strip().replace('', 'Inconnu'),
                                _zone=df_f[COL_ZS].fillna('Inconnu').astype(str).str.strip().replace('', 'Inconnu'),
                                _modalite=df_f[COL_AS].fillna('Inconnu').astype(str).str.strip().replace('', 'Inconnu'),
                            )
                            .groupby(['_province', '_zone', '_modalite'], dropna=False)
                            .size()
                            .reset_index(name='n')
                            .rename(columns={
                                '_province': 'Province de notification',
                                '_zone': 'Zone de notification',
                                '_modalite': 'Aire_de_sante_notification',
                            })
                        )
                        tbl['%'] = (tbl['n'] / max(len(df_f), 1) * 100).round(1)
                        tbl = tbl.sort_values(['n', 'Province de notification', 'Zone de notification', 'Aire_de_sante_notification'], ascending=[False, True, True, True])
                        st_dataframe_safe(tbl)
                    else:
                        st_dataframe_safe(build_frequency_table(df_f, c))
        else:
            st.info("Aucune variable catégorielle exploitable n’a été détectée.")

    # =========================
    # TAB 5: Complétude
    # =========================
with tab_qualite:
    render_section_title(4, "Complétude des données et couverture des rapports")
    if IDSR_MODE:
        st.info("Mode **IDSR agrégé hebdomadaire** : les analyses de liste linéaire sont désactivées dans cet espace. Veuillez utiliser l’onglet **IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Vérifier si les provinces attendues notifient (complétude géographique).
        
            **📖 Interprétation**
            - **Manquantes** : silence épidémiologique ou problème de remontée/rapportage.
            - Le tableau croisé aide à repérer les zones/provinces dominantes ou sous-notifiantes.
        
            **⚠️ Points d’attention**
            - Une province silencieuse pendant une épidémie = signal d’alerte système à investiguer.
            """,
            expanded=False
        )     
        
        with st.expander("Définir les provinces en épidémie (attendues dans la line list)", expanded=False):

            # ---- Init states ----
            if "epidemie_state_tab5" not in st.session_state:
                st.session_state.epidemie_state_tab5 = EPIDEMIE.copy()

            if "epid_version_tab5" not in st.session_state:
                st.session_state.epid_version_tab5 = 0

            # ---- Callbacks (ne modifient PAS les keys existantes) ----
            def _apply_bulk_tab5(value: bool):
                # Met à jour le dict + change de version -> recrée les checkboxes
                for p in EPIDEMIE.keys():
                    st.session_state.epidemie_state_tab5[p] = value
                st.session_state.epid_version_tab5 += 1

            def _reset_defaults_tab5():
                st.session_state.epidemie_state_tab5 = EPIDEMIE.copy()
                st.session_state.epid_version_tab5 += 1

            def _sync_one_tab5(prov: str, widget_key: str):
                # Synchronise le dict à partir de l'état réel du widget
                st.session_state.epidemie_state_tab5[prov] = bool(st.session_state.get(widget_key, False))

            st.markdown("✅ **Coche** = province considérée **en épidémie** (attendue dans la line list)")

            provs = sorted(list(EPIDEMIE.keys()))
            cols = st.columns(3)

            # ---- Checkboxes (keys versionnées) ----
            v = st.session_state.epid_version_tab5
            for i, prov in enumerate(provs):
                with cols[i % 3]:
                    wkey = f"chk_epid_{prov}_v{v}"  # <-- clé versionnée
                    st.checkbox(
                        prov,
                        value=st.session_state.epidemie_state_tab5.get(prov, False),
                        key=wkey,
                        on_change=_sync_one_tab5,
                        args=(prov, wkey),
                    )

            # ---- Boutons (on_click = safe) ----
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.button("Sélectionner toutes les provinces", key="tab5_all", on_click=_apply_bulk_tab5, args=(True,))
            with c2:
                st.button("Désélectionner toutes les provinces", key="tab5_none", on_click=_apply_bulk_tab5, args=(False,))
            with c3:
                st.button("Réinitialiser selon les paramètres par défaut du script", key="tab5_reset", on_click=_reset_defaults_tab5)

            # ✅ Provinces attendues (UI Tab5)
            PROVINCES_EPID = [p for p, ok in st.session_state.epidemie_state_tab5.items() if ok]
       
        st.subheader("Suivi de la complétude de notification : provinces attendues versus provinces effectivement rapportées")
        
        if COL_PROV not in df_f.columns:
            st.info("La variable Province_notification est absente du fichier analysé.")
        else:
            if COL_WNUM in df_f.columns and df_f[COL_WNUM].notna().any():
                last_w = int(df_f[COL_WNUM].max())
                present = sorted(df_f.loc[df_f[COL_WNUM] == last_w, COL_PROV].dropna().unique().tolist())
                st.caption(f"Calcul sur la semaine max filtrée: SE{last_w:02d}")
            else:
                present = sorted(df_f[COL_PROV].dropna().unique().tolist())
                st.caption("Calcul sur l’ensemble filtré (pas de Num_semaine_epid exploitable).")
        
            missing = [p for p in PROVINCES_EPID if p not in present]
            nb_att = len(PROVINCES_EPID)
            nb_rec = len([p for p in PROVINCES_EPID if p in present])
            compl = (nb_rec / nb_att * 100) if nb_att > 0 else np.nan
        
            c1, c2, c3 = st.columns(3)
            c1.metric("Provinces attendues", str(nb_att))
            c2.metric("Provinces trouvées", str(nb_rec))
            c3.metric("Complétude (%)", f"{compl:.1f}")
            if missing:
                st.warning("Provinces attendues non reçues : " + ", ".join(missing))
        
            with st.expander("Tableau provinces attendues vs reçues"):
                df_comp = pd.DataFrame({
                    "Province attendue": PROVINCES_EPID,
                    "Présente": [p in present for p in PROVINCES_EPID],
                    "Manquante": [p if p in missing else "" for p in PROVINCES_EPID],
                })
                st_dataframe_safe(df_comp)
        
            with st.expander("Cas par province (complétude / volume)", expanded=True):
                prov_counts = df_f[COL_PROV].fillna("Inconnu").value_counts().reset_index()
                prov_counts.columns = [COL_PROV, "Cas"]
                figp = px.bar(prov_counts, x=COL_PROV, y="Cas", title=" ")
                figp.update_layout(xaxis_tickangle=-45)
                figp = apply_plotly_value_annotations(figp, annot_vals)
                st.plotly_chart(figp, width="stretch")
        
            # TCD
            with st.expander("Tableau croisé dynamique – occurrences", expanded=False):
                # --- Scope: même logique que ton calcul "semaine max filtrée"
                scope_last_week = st.checkbox(
                    "Calculer uniquement sur la semaine max filtrée (même scope que la complétude)",
                    value=True,
                    key="ct_scope_last_week"
                )
                df_scope = df_f.copy()
                if scope_last_week and (COL_WNUM in df_scope.columns) and df_scope[COL_WNUM].notna().any():
                    last_w = int(df_scope[COL_WNUM].max())
                    df_scope = df_scope.loc[df_scope[COL_WNUM] == last_w].copy()
                    st.caption(f"Scope: SE{last_w:02d}")
                else:
                    st.caption("Scope: ensemble filtré")
        
                # --- Outils UX (global)
                cUX1, cUX2, cUX3, cUX4 = st.columns([1.1, 1.1, 1.1, 0.9])
                with cUX1:
                    show_pct = st.checkbox("Afficher les pourcentages", value=False, key="ct_show_pct")
                with cUX2:
                    show_bar = st.checkbox("Afficher les barres dans le tableau", value=True, key="ct_show_bar")
                with cUX3:
                    tbl_height = st.number_input("Hauteur du tableau", min_value=250, max_value=1200, value=520, step=50, key="ct_tbl_height")
                with cUX4:
                    do_download = st.checkbox("Activer l’export", value=True, key="ct_export_on")
        
                # --- Choix du niveau d’agrégation (on maintient les 3 options)
                level = st.radio(
                    "Niveau d’agrégation",
                    ["Province (occurrences)", "Province + Zone de santé", "Tableau croisé Province × Zone"],
                    index=0,
                    horizontal=True,
                    key="ct_level"
                )
        
                # Helper: affiche tableau + option export
                def _show_table(df_to_show: pd.DataFrame, name: str):
                    st.dataframe(
                        df_to_show, width='stretch', height=int(tbl_height),
                        hide_index=False,
                        column_config=None
                    )
                    if do_download:
                        csv = df_to_show.to_csv(index=True).encode("utf-8")
                        st.download_button(
                            f"Télécharger {name} (CSV)",
                            data=csv,
                            file_name=f"{name}.csv".replace(" ", "_").lower(),
                            mime="text/csv",
                            key=f"dl_{name}"
                        )
        
                # 1) Province (occurrences)
                if level == "Province (occurrences)":
                    if COL_PROV not in df_scope.columns:
                        st.info("La variable Province_notification est absente du fichier analysé.")
                    else:
                        piv = (
                            df_scope.assign(_prov=df_scope[COL_PROV].fillna("Inconnu"))
                            .groupby("_prov", dropna=False)
                            .size()
                            .reset_index(name="Occurrences")
                            .sort_values("Occurrences", ascending=False)
                            .rename(columns={"_prov": COL_PROV})
                        )
        
                        if show_pct:
                            total = int(piv["Occurrences"].sum()) if len(piv) else 0
                            piv["%"] = (piv["Occurrences"] / total * 100).round(1) if total > 0 else 0.0
        
                        if show_bar:
                            st.dataframe(
                                piv, width='stretch', height=int(tbl_height),
                                column_config={
                                    "Occurrences": st.column_config.ProgressColumn(
                                        "Occurrences",
                                        help="Occurrences (barres)",
                                        format="%d",
                                        min_value=0,
                                        max_value=int(piv["Occurrences"].max()) if len(piv) else 1,
                                    )
                                },
                            )
                            if do_download:
                                csv = df_to_csv_bytes(piv)
                                st.download_button(
                                    "Télécharger province_occurrences (CSV)",
                                    data=csv,
                                    file_name="province_occurrences.csv",
                                    mime="text/csv",
                                    key="dl_prov_occ"
                                )
                        else:
                            _show_table(piv, "province_occurrences")
        
                    with st.expander("Graphique (top provinces)"):
                        topk = st.number_input("Nombre de provinces à afficher", min_value=5, max_value=30, value=15, step=1, key="ct_topk_prov")
                        figp = px.bar(piv.head(int(topk)), x=COL_PROV, y="Occurrences", title="Top provinces – occurrences")
                        figp.update_layout(xaxis_tickangle=-45)
                        figp = apply_plotly_value_annotations(figp, annot_vals)
                        st.plotly_chart(figp, width="stretch")
        
                # 2) Province + Zone de santé
                elif level == "Province + Zone de santé":
                    if (COL_PROV not in df_scope.columns) or (COL_ZS not in df_scope.columns):
                        st.info("Les variables Province_notification et/ou Zone_de_sante_notification sont absentes.")
                    else:
                        colA, colB, colC = st.columns([1.2, 1.2, 1.6])
                        with colA:
                            view_mode = st.radio(
                                "Vue",
                                ["Top N (table longue)", "Déroulable Province → Zone"],
                                index=1,
                                horizontal=True,
                                key="ct_view_mode_pz"
                            )
                        with colB:
                            limit_zones = st.checkbox("Limiter le nombre de zones de santé (performance)", value=True, key="ct_limit_zones_pz")
                        with colC:
                            top_z = st.number_input("Nombre maximum de zones de santé", min_value=10, max_value=2000, value=250, step=25, key="ct_top_z_pz")
        
                        df_scope2 = df_scope.copy()
                        if limit_zones:
                            zones_top = (
                                df_scope2[COL_ZS].fillna("Inconnu")
                                .value_counts()
                                .head(int(top_z))
                                .index.tolist()
                            )
                            df_scope2 = df_scope2[df_scope2[COL_ZS].fillna("Inconnu").isin(zones_top)].copy()
        
                        piv = (
                            df_scope2.assign(
                                _prov=df_scope2[COL_PROV].fillna("Inconnu"),
                                _zs=df_scope2[COL_ZS].fillna("Inconnu"),
                            )
                            .groupby(["_prov", "_zs"], dropna=False)
                            .size()
                            .reset_index(name="Occurrences")
                            .sort_values("Occurrences", ascending=False)
                            .rename(columns={"_prov": COL_PROV, "_zs": COL_ZS})
                        )
        
                        if show_pct:
                            tot_prov = piv.groupby(COL_PROV, as_index=False)["Occurrences"].sum().rename(columns={"Occurrences": "Total_province"})
                            piv = piv.merge(tot_prov, on=COL_PROV, how="left")
                            piv["%_dans_province"] = (piv["Occurrences"] / piv["Total_province"] * 100).round(1)
                            piv = piv.drop(columns=["Total_province"])
        
                        tot_prov = (
                            piv.groupby(COL_PROV, as_index=False)["Occurrences"].sum()
                            .sort_values("Occurrences", ascending=False)
                        )
        
                        if view_mode == "Top N (table longue)":
                            top_n = st.number_input("Nombre maximum de lignes à afficher", min_value=10, max_value=20000, value=500, step=50, key="ct_topn_long")
                            df_show = piv.head(int(top_n)).copy()
        
                            if show_bar:
                                st.dataframe(
                                    df_show, width='stretch', height=int(tbl_height),
                                    column_config={
                                        "Occurrences": st.column_config.ProgressColumn(
                                            "Occurrences",
                                            format="%d",
                                            min_value=0,
                                            max_value=int(piv["Occurrences"].max()) if len(piv) else 1,
                                        )
                                    },
                                )
                            else:
                                _show_table(df_show, "province_zone_topN")
        
                        else:
                            tcd = (
                                piv.set_index([COL_PROV, COL_ZS])[["Occurrences"]]
                                .sort_values("Occurrences", ascending=False)
                            )
                            tcd = tcd.reindex(tot_prov[COL_PROV].tolist(), level=0)
        
                            st.caption("Clique sur les triangles à gauche pour dérouler/replier Province → Zone.")
                            st.dataframe(tcd, width='stretch', height=int(tbl_height))
        
                            if do_download:
                                csv = tcd.reset_index().to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Télécharger province_zone_deroulable (CSV)",
                                    data=csv,
                                    file_name="province_zone_deroulable.csv",
                                    mime="text/csv",
                                    key="dl_pz_deroulable"
                                )
        
                        with st.expander("Totaux par province (somme des zones)"):
                            if show_bar:
                                st.dataframe(
                                    tot_prov, width='stretch', height=450,
                                    column_config={
                                        "Occurrences": st.column_config.ProgressColumn(
                                            "Occurrences",
                                            format="%d",
                                            min_value=0,
                                            max_value=int(tot_prov["Occurrences"].max()) if len(tot_prov) else 1,
                                        )
                                    },
                                )
                            else:
                                st_dataframe_safe(tot_prov)
        
                        with st.expander("Graphique (top provinces)"):
                            topk = st.number_input("Nombre de provinces à afficher", min_value=5, max_value=30, value=15, step=1, key="ct_topk_pz")
                            figp = px.bar(tot_prov.head(int(topk)), x=COL_PROV, y="Occurrences", title="Top provinces – occurrences (scope)")
                            figp.update_layout(xaxis_tickangle=-45)
                            figp = apply_plotly_value_annotations(figp, annot_vals)
                            st.plotly_chart(figp, width="stretch")
        
                # 3) Tableau croisé Province × Zone
                else:
                    if (COL_PROV not in df_scope.columns) or (COL_ZS not in df_scope.columns):
                        st.info("Les variables Province_notification et/ou Zone_de_sante_notification sont absentes.")
                    else:
                        cA, cB, cC = st.columns([1.1, 1.3, 1.6])
                        with cA:
                            limit_zones = st.checkbox("Limiter aux zones les plus fréquentes", value=True, key="ct_limit_zones_wide")
                        with cB:
                            top_z = st.number_input("Top zones", min_value=10, max_value=1500, value=120, step=10, key="ct_topz_wide")
                        with cC:
                            show_heatmap = st.checkbox("Afficher en heatmap", value=False, key="ct_show_heatmap")
        
                        if limit_zones:
                            zones_top = (
                                df_scope[COL_ZS].fillna("Inconnu")
                                .value_counts()
                                .head(int(top_z))
                                .index.tolist()
                            )
                            df_ct = df_scope[df_scope[COL_ZS].fillna("Inconnu").isin(zones_top)].copy()
                        else:
                            df_ct = df_scope.copy()
        
                        ct = pd.crosstab(
                            index=df_ct[COL_PROV].fillna("Inconnu"),
                            columns=df_ct[COL_ZS].fillna("Inconnu"),
                            margins=True,
                            margins_name="Total",
                            dropna=False
                        )
        
                        sort_totals = st.checkbox("Trier par total décroissant", value=True, key="ct_sort_totals")
                        if sort_totals and "Total" in ct.columns and "Total" in ct.index:
                            rows = ct.drop(index="Total", errors="ignore").sort_values("Total", ascending=False)
                            cols_tot = ct.drop(columns="Total", errors="ignore").loc["Total"].sort_values(ascending=False).index.tolist() \
                                if "Total" in ct.index else ct.drop(columns="Total", errors="ignore").columns.tolist()
                            ct = rows[cols_tot]
                            ct.loc["Total"] = ct.sum(axis=0)
                            ct["Total"] = ct.sum(axis=1)
                            ct = ct.fillna(0).astype(int)
        
                        st.dataframe(ct, width='stretch', height=int(tbl_height))
        
                        if do_download:
                            csv = ct.to_csv(index=True).encode("utf-8")
                            st.download_button(
                                "Télécharger province_x_zone (CSV)",
                                data=csv,
                                file_name="province_x_zone.csv",
                                mime="text/csv",
                                key="dl_ct_wide"
                            )
        
                        if show_heatmap:
                            ct_heat = ct.drop(index="Total", errors="ignore").drop(columns="Total", errors="ignore")
                            fig_hm = px.imshow(
                                ct_heat,
                                aspect="auto",
                                labels=dict(x="Zone de santé", y="Province", color="Occurrences"),
                                title="Heatmap – Occurrences Province × Zone"
                            )
                            fig_hm.update_layout(height=700)
                            st.plotly_chart(fig_hm, width="stretch")
        
    # =========================
    # TAB 6: DATA & EXPORT
    # =========================
with tab_qualite:
    st.divider()
    render_section_title(5, "Extraction, revue et export des données")
    if IDSR_MODE:
        st.info("Mode **IDSR agrégé hebdomadaire** : les analyses de liste linéaire sont désactivées dans cet espace. Veuillez utiliser l’onglet **IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Consulter et exporter les données filtrées pour analyses/partage.
        
            **📖 Utilisation**
            - Exportation **CSV/Excel** pour analyses complémentaires (R/Python/DHIS2).
            - Vérifier les filtres actifs avant export.
        
            **⚠️ Points d’attention**
            - Les exports reflètent exactement le périmètre filtré (province/ZS/AS/semaine/classification).
            """,
            expanded=False
        )
        
        st.subheader("Extraction des données filtrées, traçabilité et options d’export")

        def _build_export_traceability_table(df_scope: pd.DataFrame) -> pd.DataFrame:
            """Construit une table simple de traçabilité pour les exports."""
            week_min_val = "-"
            week_max_val = "-"
            year_min_val = "-"
            year_max_val = "-"

            if COL_WNUM in df_scope.columns and df_scope[COL_WNUM].notna().any():
                week_num = pd.to_numeric(df_scope[COL_WNUM], errors="coerce").dropna()
                if not week_num.empty:
                    week_min_val = int(week_num.min())
                    week_max_val = int(week_num.max())

            if COL_YEAR in df_scope.columns and df_scope[COL_YEAR].notna().any():
                year_num = pd.to_numeric(df_scope[COL_YEAR], errors="coerce").dropna()
                if not year_num.empty:
                    year_min_val = int(year_num.min())
                    year_max_val = int(year_num.max())

            rows = [
                {"Paramètre": "Date export", "Valeur": str(date.today())},
                {"Paramètre": "Maladie / line list", "Valeur": DISEASE_SPECS.get(disease_key, {}).get("label", disease_key)},
                {"Paramètre": "Sources chargées", "Valeur": ", ".join(files_used) if isinstance(files_used, list) and files_used else "Non documenté"},
                {"Paramètre": "Cas exportés", "Valeur": int(len(df_scope))},
                {"Paramètre": "Colonnes exportées", "Valeur": int(len(df_scope.columns))},
                {"Paramètre": "SE min observée", "Valeur": week_min_val},
                {"Paramètre": "SE max observée", "Valeur": week_max_val},
                {"Paramètre": "Année min observée", "Valeur": year_min_val},
                {"Paramètre": "Année max observée", "Valeur": year_max_val},
                {
                    "Paramètre": "Provinces distinctes",
                    "Valeur": int(_surveillance_clean_text_series(df_scope[COL_PROV]).nunique()) if COL_PROV in df_scope.columns else 0,
                },
                {
                    "Paramètre": "Zones de santé distinctes",
                    "Valeur": int(_surveillance_clean_text_series(df_scope[COL_ZS]).nunique()) if COL_ZS in df_scope.columns else 0,
                },
            ]
            return pd.DataFrame(rows)

        export_traceability = _build_export_traceability_table(df_f)
        export_quality_summary = standard_data_quality_summary(df_f)
        export_qc_flags = qc_flags(df_f)
        export_qc_resume = (
            export_qc_flags["flag"].value_counts().rename_axis("Flag").reset_index(name="Occurrences")
            if not export_qc_flags.empty else pd.DataFrame(columns=["Flag", "Occurrences"])
        )
        export_duplicates = duplicate_candidates_table(df_f)

        export_completeness = pd.DataFrame()
        export_completeness_by = None
        export_required_fields = [
            COL_PROV, COL_ZS, COL_AS, "YW", COL_WNUM, COL_SEX, COL_AGE,
            COL_UNIT, DATE_ONSET, COL_PREL, COL_TDR, COL_TDRR, COL_HOSP,
            COL_ISSUE, COL_CLASS,
        ]
        for group_col in [COL_PROV, COL_ZS, "YW", COL_WNUM]:
            if group_col in df_f.columns and df_f[group_col].notna().any():
                export_completeness = completeness_table(df_f, export_required_fields, by=group_col)
                if not export_completeness.empty:
                    export_completeness_by = group_col
                    break

        export_base_name = f"{str(disease_key).strip().lower()}_filtre"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cas exportés", format_metric_value(len(df_f)))
        m2.metric("Colonnes", format_metric_value(len(df_f.columns)))
        m3.metric(
            "Provinces distinctes",
            format_metric_value(int(_surveillance_clean_text_series(df_f[COL_PROV]).nunique()) if COL_PROV in df_f.columns else 0),
        )
        m4.metric(
            "ZS distinctes",
            format_metric_value(int(_surveillance_clean_text_series(df_f[COL_ZS]).nunique()) if COL_ZS in df_f.columns else 0),
        )

        info1, info2 = st.columns([0.9, 1.1])
        with info1:
            st.markdown("**Traçabilité de l’export**")
            st_dataframe_safe(export_traceability, height=360)
        with info2:
            st.markdown("**Résumé qualité inclus dans le pack d’export**")
            if not export_quality_summary.empty:
                st_dataframe_safe(export_quality_summary, height=360)
            else:
                st.info("Aucun résumé qualité n’est disponible pour le périmètre filtré.")

        with st.expander("Prévisualiser les contenus additionnels du pack d’export", expanded=False):
            if not export_qc_resume.empty:
                st.markdown("**Résumé des incohérences détectées**")
                st_dataframe_safe(export_qc_resume, height=260)
            else:
                st.caption("Aucune incohérence détectée par `qc_flags` sur le périmètre filtré.")

            if not export_duplicates.empty:
                st.markdown("**Doublons potentiels**")
                st_dataframe_safe(export_duplicates.head(100), height=260)
            else:
                st.caption("Aucun doublon potentiel n’a été identifié.")

            if export_completeness_by and not export_completeness.empty:
                st.markdown(f"**Complétude par {export_completeness_by}**")
                st_dataframe_safe(export_completeness.head(100), height=260)
            else:
                st.caption("Aucun tableau de complétude additionnel n’a pu être préparé pour l’export.")

        st.markdown("**Aperçu de la line list filtrée**")
        st_dataframe_safe(df_f, height=420)

        export_mode = st.radio(
            "Type d’export",
            ["Pack qualité + line list", "Line list uniquement"],
            index=0,
            horizontal=True,
            key="qualite_export_mode",
        )

        dl1, dl2, dl3 = st.columns([1, 1, 1])
        with dl1:
            st.download_button(
                "Télécharger CSV line list",
                data=df_to_csv_bytes(df_f),
                file_name=f"{export_base_name}.csv",
                mime="text/csv",
                key="dl_quality_export_csv_ll",
            )
        with dl2:
            if not export_qc_flags.empty:
                st.download_button(
                    "Télécharger QC flags (CSV)",
                    data=export_qc_flags.to_csv(index=False).encode("utf-8"),
                    file_name=f"{export_base_name}_qc_flags.csv",
                    mime="text/csv",
                    key="dl_quality_export_csv_qc",
                )
        with dl3:
            if not export_duplicates.empty:
                st.download_button(
                    "Télécharger doublons (CSV)",
                    data=export_duplicates.to_csv(index=False).encode("utf-8"),
                    file_name=f"{export_base_name}_doublons.csv",
                    mime="text/csv",
                    key="dl_quality_export_csv_dup",
                )

        try:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_f.to_excel(writer, sheet_name="LL_filtre", index=False)

                if export_mode == "Pack qualité + line list":
                    export_traceability.to_excel(writer, sheet_name="Traceabilite", index=False)
                    export_quality_summary.to_excel(writer, sheet_name="Resume_qualite", index=False)
                    if not export_qc_resume.empty:
                        export_qc_resume.to_excel(writer, sheet_name="QC_resume", index=False)
                    if not export_qc_flags.empty:
                        export_qc_flags.to_excel(writer, sheet_name="QC_detail", index=False)
                    if not export_duplicates.empty:
                        export_duplicates.to_excel(writer, sheet_name="Doublons", index=False)
                    if not export_completeness.empty:
                        export_completeness.to_excel(writer, sheet_name="Completude", index=False)

            excel_name = (
                f"{export_base_name}_pack_qualite.xlsx"
                if export_mode == "Pack qualité + line list"
                else f"{export_base_name}.xlsx"
            )
            st.download_button(
                "Télécharger Excel",
                data=buffer.getvalue(),
                file_name=excel_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_quality_export_xlsx",
            )
        except Exception:
            st.info("Exportation Excel indisponible (openpyxl manquant ?).")
        
    # =========================
    # TAB 7 — Labo / qualité / signaux
    # =========================
with tab_qualite:
    st.divider()
    render_section_title(6, "Qualité des données et alertes de gestion")
    if IDSR_MODE:
        st.info("Mode **IDSR agrégé hebdomadaire** : les analyses de liste linéaire sont désactivées dans cet espace. Veuillez utiliser l’onglet **IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Détecter incohérences, problèmes de complétude, goulots labo, et signaux d’alerte.
        
            **📖 Sections**
            - **Indicateurs rapides** : 3–5 KPI qualité/action
            - **QC Flags** : incohérences (dates, TDR, âge…)
            - **Complétude champs clés** : % remplissage par site
            - **Cascade labo** : cas → prélèvement → TDR → résultat valide → positif
            - **Alertes tendance** : hausse inhabituelle vs baseline simple
        
            **⚠️ Points d’attention**
            - Un signal ≠ confirmation d’épidémie : déclenche une investigation terrain.
            - Les % de cascade sont calculés sur une logique *entonnoir* (séquentielle).
            """,
            expanded=False
        )
        
        st.subheader("Contrôle qualité des données et alertes opérationnelles de surveillance")
        
        # -------- Helpers (robustes) --------
        def _get_pct_from_cascade(casc: pd.DataFrame, key: str) -> float:
            """Récupère le % de la première ligne dont Étape contient key (robuste aux libellés)."""
            if casc is None or casc.empty or "Étape" not in casc.columns or "%" not in casc.columns:
                return np.nan
            m = casc.loc[casc["Étape"].astype(str).str.contains(key, regex=False, na=False), "%"]
            return float(m.iloc[0]) if len(m) else np.nan
        
        def _safe_num(x):
            try:
                return float(x)
            except Exception:
                return np.nan
        
        # ==========================================================
        # 0) Indicateurs rapides (KPI)
        # ==========================================================
        n_total = len(df_f)
        
        kpi = compute_indicators(df_f)
        casc_global = cascade_metrics(df_f) if n_total else pd.DataFrame()
        has_tdr_chain = COL_TDR in df_f.columns and df_f[COL_TDR].notna().any()
        
        # KPI “qualité TDR” (sur cascade)
        kpi_incoh_res_wo_tdr = _get_pct_from_cascade(casc_global, "Résultat renseigné mais TDR_realise != Oui")
        kpi_status_in_result = _get_pct_from_cascade(casc_global, "Statut saisi dans TDR_Resultat")
        
        # ✅ 7 colonnes (ajout hospitalisation)
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        
        c1.metric(
            "Cas (n)",
            f"{kpi['n_cases']:,}".replace(",", " "),
            help="Nombre total de cas après application des filtres (Province/ZS/SE, etc.)."
        )
        
        c2.metric(
            "% prélèvement",
            "-" if np.isnan(kpi["prelev_pct"]) else f"{kpi['prelev_pct']:.1f}",
            help=f"Prélèvement=Oui / Tous les cas filtrés. n={kpi.get('prelev_num', 0)}/{kpi.get('prelev_den', kpi.get('n_cases', 0))}"
        )
        
        c3.metric(
            "Couverture TDR (%)" if has_tdr_chain else "Couverture test (%)",
            "-" if np.isnan(kpi["tdr_pct"]) else f"{kpi['tdr_pct']:.1f}",
            help=(
                f"TDR_realise=Oui / Tous les cas filtrés. n={kpi.get('tdr_num', 0)}/{kpi.get('tdr_den', kpi.get('n_cases', 0))}"
                if has_tdr_chain
                else f"Tests documentés / Tous les cas filtrés. n={kpi.get('tdr_num', 0)}/{kpi.get('tdr_den', kpi.get('n_cases', 0))}"
            )
        )
        
        # ✅ Positivité
        pos_label = "-"
        if not np.isnan(kpi["pos_pct"]):
            pos_label = f"{kpi['pos_pct']:.1f}"
        c4.metric(
            "Positivité TDR" if has_tdr_chain else "Positivité test",
            pos_label,
            help=(
                (
                    "Positifs / (Positifs + Négatifs) parmi les TDR interprétables "
                    "(TDR_realise=Oui ET résultat valide Pos/Nég). "
                )
                if has_tdr_chain
                else "Positifs / (Positifs + Négatifs) parmi les résultats labo interprétables. "
            ) + f"n={kpi.get('pos_num', 0)}/{kpi.get('pos_den', 0)}"
        )
        
        # 🆕 Taux hospitalisation
        c5.metric(
            "Hospitalisation (%)",
            "-" if np.isnan(kpi["hosp_pct"]) else f"{kpi['hosp_pct']:.1f}",
            help=f"Hospitalisation=Oui / Tous les cas filtrés. n={kpi.get('hosp_num', 0)}/{kpi.get('hosp_den', kpi.get('n_cases', 0))}"
        )
        
        c6.metric(
            "CFR (%)",
            "-" if np.isnan(kpi["cfr_pct"]) else f"{kpi['cfr_pct']:.2f}",
            help=f"Décès / Tous les cas filtrés. n={kpi.get('n_deaths', 0)}/{kpi.get('n_cases', 0)}"
        )
        
        # % invalides
        inv_label = "-"
        if "invalid_pct" in kpi and not np.isnan(kpi["invalid_pct"]):
            inv_label = f"{kpi['invalid_pct']:.1f}"
        c7.metric(
            "% TDR invalides" if has_tdr_chain else "% tests invalides",
            inv_label,
            help=(
                (
                    "Invalides (ex: INBA/bande absente) / TDR réalisés (TDR_realise=Oui). "
                    if has_tdr_chain
                    else "Invalides / tests documentés. "
                )
                + f"n={kpi.get('invalid_num', 0)}/{kpi.get('invalid_den', 0)}"
            )
        )
        
        # Alertes qualité tests (si dispo)
        if not np.isnan(kpi_incoh_res_wo_tdr) or not np.isnan(kpi_status_in_result):
            with st.expander("📌 Signaux qualité tests (données)", expanded=False):
                if not np.isnan(kpi_incoh_res_wo_tdr):
                    st.write(f"- **% Résultat renseigné mais TDR_realise ≠ Oui / statut test absent**: **{kpi_incoh_res_wo_tdr:.1f}%**")
                if not np.isnan(kpi_status_in_result):
                    st.write(f"- **% Statut saisi dans la colonne de résultat** (ex: non réalisé/non prélevé): **{kpi_status_in_result:.1f}%**")
        
        with st.expander("🔎 Détail cascade labo (entonnoir) + incohérences", expanded=False):
            st_dataframe_safe(casc_global)

        # ==========================================================
        # 0b) Résumé standard qualité / délais / disponibilité des champs
        # ==========================================================
        with st.expander("🔎 Résumé standard qualité / délais / disponibilité des champs", expanded=False):
            qsum = standard_data_quality_summary(df_f)
            if not qsum.empty:
                st_dataframe_safe(qsum)
            dsum = build_standard_delay_summary(df_f)
            if not dsum.empty:
                st.markdown("**Résumé standard des délais**")
                st_dataframe_safe(dsum)
            dup_tbl = duplicate_candidates_table(df_f)
            if not dup_tbl.empty:
                st.markdown("**Doublons potentiels à vérifier**")
                st_dataframe_safe(dup_tbl.head(100), height=320)
            fields_matrix = build_recommended_fields_matrix(df_f)
            if not fields_matrix.empty:
                st.markdown("**Disponibilité des champs standards recommandés**")
                bloc_sel = st.selectbox("Filtrer la matrice par bloc", ["Tous"] + sorted(fields_matrix["Bloc"].dropna().unique().tolist()), index=0, key="fields_matrix_bloc")
                if bloc_sel != "Tous":
                    fields_matrix = fields_matrix[fields_matrix["Bloc"] == bloc_sel]
                st_dataframe_safe(fields_matrix, height=360)
        
        # ==========================================================
        # 1) QC Flags (incohérences)
        # ==========================================================
        with st.expander("🔎 Incohérences (QC Flags)", expanded=False):
        
        
            flags = qc_flags(df_f)
            if flags.empty:
                st.success("Aucune incohérence n’a été détectée selon les règles de contrôle actuellement appliquées.")
            else:
                # Résumé
                resume = flags["flag"].value_counts().reset_index()
                resume.columns = ["Flag", "Occurrences"]
                st_dataframe_safe(resume)
        
                # Filtre par flag
                flag_list = sorted(flags["flag"].dropna().unique().tolist())
                flag_sel = st.selectbox("Filtrer le détail par type d’incohérence", ["Tous"] + flag_list, index=0)
        
                # Détail (merge + colonnes utiles)
                cols_show = [c for c in [
                    "Nom_complet", COL_PROV, COL_ZS, COL_AS, COL_SEX, COL_AGE, COL_UNIT,
                    "YW", COL_WNUM, DATE_ONSET, DATE_ADM, DATE_PREL,
                    COL_PREL, COL_TDR, COL_TDRR, COL_HOSP, COL_ISSUE, COL_CLASS
                ] if c in df_f.columns]
        
                detail = flags.merge(df_f.reset_index().rename(columns={"index": "row_id"}), on="row_id", how="left")
        
                if flag_sel != "Tous":
                    detail = detail[detail["flag"] == flag_sel]
        
                st.caption("Détail des lignes concernées (filtré, maximum 500 lignes)")
                st.dataframe(detail[["flag"] + cols_show].head(500), width="stretch", height=420)
        
        # ==========================================================
        # 2) Complétude des champs clés
        # ==========================================================
        with st.expander("🔎 Complétude des champs clés", expanded=False):
        
            champs_cles = [
                COL_PROV, COL_ZS, COL_AS, "YW", COL_WNUM,
                COL_SEX, COL_AGE, COL_UNIT, DATE_ONSET,
                COL_PREL, COL_TDR, COL_TDRR, COL_HOSP,
                COL_ISSUE, COL_CLASS
            ]
        
            group_choices = [c for c in [COL_PROV, COL_ZS, "YW", COL_WNUM] if c in df_f.columns]
            group_for_comp = st.selectbox("Analyser la complétude par", group_choices, index=0 if group_choices else 0)
        
            comp = completeness_table(df_f, champs_cles, by=group_for_comp) if group_choices else pd.DataFrame()
        
            if comp.empty:
                st.info("Impossible de calculer la complétude : variable de regroupement ou champs requis absents.")
            else:
                st_dataframe_safe(comp, height=520)
        
                # Bar chart plus lisible: top N pires scores
                topn = st.slider("Nombre de groupes les moins complets à afficher", min_value=10, max_value=80, value=25, step=5)
                comp_plot = comp.sort_values("score_completude_%").head(topn)
        
                figc = px.bar(
                    comp_plot,
                    x=group_for_comp,
                    y="score_completude_%",
                    title=f"Score complétude (%) – {topn} groupes les moins complets (par {group_for_comp})"
                )
                figc.update_layout(xaxis_tickangle=-45, yaxis=dict(range=[0, 100]))
                figc = apply_plotly_value_annotations(figc, annot_vals)
                st.plotly_chart(figc, width="stretch")
        
        
        # ==========================================================
        # 3) Cascade prélèvement → TDR → résultat → positif
        # ==========================================================
        with st.expander("🔎 Cascade prélèvement → TDR → résultat → positif", expanded=False):
        
            cascad = cascade_metrics(df_f) if n_total else pd.DataFrame()
            if cascad.empty:
                st.info("La cascade est indisponible : aucune donnée n’est disponible après application des filtres.")
            else:
                st_dataframe_safe(cascad)
        
            # Cascade par province (résumé robuste)
            if COL_PROV in df_f.columns and n_total:
                st.caption("Cascade par province (résumé)")
        
                rows = []
                for prov, sub in df_f.groupby(COL_PROV, dropna=False):
                    c = cascade_metrics(sub)
                    rows.append([
                        prov,
                        len(sub),
                        _get_pct_from_cascade(c, "Prélèvement=Oui"),
                        _get_pct_from_cascade(c, "TDR réalisé=Oui"),
                        _get_pct_from_cascade(c, "Résultat TDR valide"),
                        _get_pct_from_cascade(c, "TDR positif"),
                        _get_pct_from_cascade(c, "Résultat renseigné mais TDR_realise != Oui"),
                    ])
        
                df_cas = pd.DataFrame(
                    rows,
                    columns=[COL_PROV, "n", "% prélèvement", "% TDR", "% résultat valide", "% positif", "% incoh TDR"]
                )
        
                sort_col = st.selectbox(
                    "Trier par",
                    ["n", "% prélèvement", "% TDR", "% résultat valide", "% positif", "% incoh TDR"],
                    index=0
                )
                df_cas_sorted = df_cas.sort_values(sort_col, ascending=False if sort_col == "n" else True)
        
                st_dataframe_safe(df_cas_sorted, height=420)
        
        
        # ==========================================================
        # 4) Alertes tendance (hausse vs baseline simple)
        # ==========================================================
        with st.expander("🔎 Alertes tendance (hausse vs baseline simple)", expanded=False):
            alert_group_choices = [c for c in [COL_PROV, COL_ZS] if c in df_f.columns]
            alert_group = st.selectbox("Regrouper les alertes par", alert_group_choices, index=0 if alert_group_choices else 0)
        
            alerts = alerts_weekly_simple(df_f, alert_group) if alert_group_choices else pd.DataFrame()
        
            if alerts.empty:
                st.info("Les alertes sont indisponibles : variable temporelle absente, groupe indisponible ou historique insuffisant.")
            else:
                # Dernière semaine observée
                last_yw = alerts["YW"].dropna().max()
                st.caption(f"Dernière semaine observée: {last_yw}")
        
                last = alerts[alerts["YW"] == last_yw].copy()
        
                # sécurité var_% (éviter inf)
                if "Cas_prev" in last.columns and "Cas" in last.columns:
                    last["Cas_prev"] = last["Cas_prev"].fillna(0)
                    last["var_%"] = np.where(
                        last["Cas_prev"] > 0,
                        (last["Cas"] - last["Cas_prev"]) / last["Cas_prev"] * 100.0,
                        np.nan
                    )
        
                # classement: signal d’abord, puis plus gros volumes
                last["signal"] = last["signal"].fillna(False)
                last = last.sort_values(["signal", "Cas"], ascending=[False, False])
        
                cols_out = [c for c in [alert_group, "YW", "Cas", "Cas_prev", "var_%", "baseline_3w", "signal"] if c in last.columns]
                st_dataframe_safe(last[cols_out], height=520)
        
                # Top signaux
                sig = last[last["signal"] == True].head(30)
                if len(sig):
                    figa = px.bar(sig, x=alert_group, y="Cas", title=f"Signaux (semaine {last_yw}) – top 30")
                    figa.update_layout(xaxis_tickangle=-45)
                    figa = apply_plotly_value_annotations(figa, annot_vals)
                    st.plotly_chart(figa, width="stretch")
                else:
                    st.success("Aucun signal n’a été détecté avec les seuils actuellement définis (baseline × 1,5 et cas ≥ 10).")
with tab_sitrep:
    if IDSR_MODE:
        st.info("Mode **IDSR agrégé hebdomadaire** : les analyses de liste linéaire sont désactivées dans cet espace. Veuillez utiliser l’onglet **IDSR**.")
    else:
        render_section_title(1, "Synthèse automatique de la situation épidémiologique (SITREP)")

        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif**
            - Produire une lecture décisionnelle courte de la **semaine épidémiologique sélectionnée** à partir des filtres actifs.

            **📖 Logique de lecture**
            - Le SITREP met l’accent sur la **semaine ciblée**, sa comparaison à la semaine précédente disponible et le **cumul annuel**.
            - Les foyers géographiques, la létalité critique et les signaux utiles à l’action sont synthétisés en priorité.

            **🚫 Ce qui n’est pas répété ici**
            - Les analyses détaillées de délais restent dans **Surveillance**.
            - Les profils démographiques détaillés restent dans **Profil**.
            - Les tableaux exhaustifs de qualité et de complétude restent dans **Données, complétude & qualité**.

            **📤 Exportation**
            - Le PDF reprend la synthèse affichée dans cet onglet pour le périmètre actif.
            """,
            expanded=False
        )

        st.caption(
            "Le SITREP reprend uniquement les informations nécessaires à la décision rapide, sans dupliquer les analyses détaillées déjà portées par les autres onglets."
        )

        # =========================================================
        # 1) UI: SE / Année / Date de publication dépendants de df_f
        # =========================================================
        if (COL_WNUM in df_f.columns) and df_f[COL_WNUM].notna().any():
            w_series = pd.to_numeric(df_f[COL_WNUM], errors="coerce").dropna()
            w_min, w_max = int(w_series.min()), int(w_series.max())
        else:
            w_min, w_max = 1, 53

        if (COL_YEAR in df_f.columns) and df_f[COL_YEAR].notna().any():
            y_series = pd.to_numeric(df_f[COL_YEAR], errors="coerce").dropna()
            y_min, y_max = int(y_series.min()), int(y_series.max())
        else:
            y_min, y_max = 2020, date.today().year

        auto_last = st.checkbox(
            "Auto: utiliser la dernière SE/Année du filtrage",
            value=True,
            key="sitrep_auto_last"
        )

        colA, colB, colC = st.columns(3)
        with colA:
            semaine = st.number_input(
                "Semaine épidémiologique (SE)",
                min_value=int(w_min),
                max_value=int(w_max),
                value=int(w_max),
                step=1,
                key="sitrep_se",
            )
        with colB:
            annee = st.number_input(
                "Année",
                min_value=int(y_min),
                max_value=int(y_max),
                value=int(y_max),
                step=1,
                key="sitrep_year",
            )
        with colC:
            date_pub = st.date_input(
                "Date de publication",
                value=date.today(),
                key="sitrep_pubdate",
            )

        if auto_last:
            semaine = int(w_max)
            annee = int(y_max)

        st.caption(
            f"Périmètre SITREP : données filtrées (`df_f`). SE disponibles : {w_min}–{w_max}. "
            f"Années disponibles : {y_min}–{y_max}. Semaine ciblée : {_build_sitrep_week_label(semaine, annee)}."
        )

        # =========================================================
        # 2) Helpers supplémentaires (spatial, demo, délais, graphiques)
        # =========================================================
        def _safe_pct(num, den):
            return (num / den * 100.0) if den and den > 0 else np.nan

        def _plotly_to_png_bytes(fig, scale: int = 1):
            """
            Conversion Plotly -> PNG bytes robuste pour Streamlit Cloud.
            - Ne fait jamais planter l'application
            - Réduit la charge mémoire avec scale=1 par défaut
            """
            if fig is None:
                return None
            try:
                return fig.to_image(format="png", scale=scale)
            except Exception as e:
                logger.warning(f"[SITREP] Export PNG Plotly ignoré : {e}")
                return None

        def build_weekly_summary(df_scope):
            """Table hebdo Cas/Décès/CFR (sur df_scope filtré)."""
            if COL_YEAR not in df_scope.columns or COL_WNUM not in df_scope.columns:
                return pd.DataFrame()

            tmp = df_scope.copy()
            tmp["_cas_"] = 1
            tmp["_deces_"] = tmp["is_death"].astype(int) if "is_death" in tmp.columns else 0

            wk = (tmp.groupby([COL_YEAR, COL_WNUM], as_index=False)
                    .agg(Cas=("_cas_", "sum"), Décès=("_deces_", "sum")))

            wk["CFR_%"] = np.where(wk["Cas"] > 0, wk["Décès"] / wk["Cas"] * 100.0, np.nan)
            wk["YW"] = wk[COL_YEAR].astype(int).astype(str) + "W" + wk[COL_WNUM].astype(int).astype(str).str.zfill(2)
            wk = wk.sort_values([COL_YEAR, COL_WNUM])

            wk["Cas_prev"] = wk["Cas"].shift(1)
            wk["var_%"] = np.where(
                wk["Cas_prev"].fillna(0) > 0,
                (wk["Cas"] - wk["Cas_prev"]) / wk["Cas_prev"] * 100.0,
                np.nan
            )
            return wk

        def build_geo_tables(d_se, min_cas_zs=30, min_cas_prov=50):
            """Tables province/ZS pour la semaine (d_se)."""
            out = {}
            if "is_death" not in d_se.columns:
                d_se = d_se.copy()
                d_se["is_death"] = 0

            tmp = d_se.copy()
            tmp["_cas_"] = 1
            tmp["_deces_"] = tmp["is_death"].astype(int)

            if COL_PROV in tmp.columns:
                prov = (tmp.groupby(COL_PROV, as_index=False)
                          .agg(Cas=("_cas_", "sum"), Décès=("_deces_", "sum")))
                prov["CFR_%"] = np.where(prov["Cas"] > 0, prov["Décès"] / prov["Cas"] * 100.0, np.nan)
                out["prov_table"] = prov.sort_values("Cas", ascending=False)
                out["prov_cfr_crit"] = prov.query("Cas >= @min_cas_prov").sort_values("CFR_%", ascending=False)

            if COL_ZS in tmp.columns:
                group_cols = [c for c in [COL_PROV, COL_ZS] if c in tmp.columns]
                zs = (tmp.groupby(group_cols, as_index=False)
                        .agg(Cas=("_cas_", "sum"), Décès=("_deces_", "sum")))
                zs["CFR_%"] = np.where(zs["Cas"] > 0, zs["Décès"] / zs["Cas"] * 100.0, np.nan)
                out["zs_table"] = zs.sort_values("Cas", ascending=False)
                out["zs_cfr_crit"] = zs.query("Cas >= @min_cas_zs").sort_values("CFR_%", ascending=False)

            return out

        def build_demo_tables(d_se):
            """Sexe / tranches âge (si disponibles)."""
            out = {}
            if COL_SEX in d_se.columns:
                sex = (d_se.groupby(COL_SEX, as_index=False)
                         .size().rename(columns={"size": "Cas"})
                         .sort_values("Cas", ascending=False))
                out["sex_table"] = sex

            # Priorité aux tranches déjà calculées dans tes données
            age_group_col = None
            if COL_AGEG in d_se.columns:
                age_group_col = COL_AGEG
            elif COL_AGEG2 in d_se.columns:
                age_group_col = COL_AGEG2

            if age_group_col:
                age = (d_se.groupby(age_group_col, as_index=False)
                         .size().rename(columns={"size": "Cas"}))
                age = age.rename(columns={age_group_col: "Tranche_age"})
                out["age_table"] = age
            return out

        def build_delay_summary(d_se):
            """Résumé timeliness (début maladie → admission) si dates présentes."""
            if (DATE_ONSET not in d_se.columns) or (DATE_ADM not in d_se.columns):
                return pd.DataFrame()

            tmp = d_se[[DATE_ONSET, DATE_ADM]].copy()
            tmp[DATE_ONSET] = pd.to_datetime(tmp[DATE_ONSET], errors="coerce")
            tmp[DATE_ADM] = pd.to_datetime(tmp[DATE_ADM], errors="coerce")
            tmp["delai_onset_adm"] = (tmp[DATE_ADM] - tmp[DATE_ONSET]).dt.days

            # bornes raisonnables (0..30j)
            tmp = tmp[(tmp["delai_onset_adm"].notna()) & (tmp["delai_onset_adm"] >= 0) & (tmp["delai_onset_adm"] <= 30)]
            if tmp.empty:
                return pd.DataFrame()

            s = tmp["delai_onset_adm"]
            return pd.DataFrame([{
                "n": int(s.notna().sum()),
                "médiane": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "%≤1j": _safe_pct((s <= 1).sum(), s.notna().sum()),
                "%≤2j": _safe_pct((s <= 2).sum(), s.notna().sum()),
                "max": float(s.max()),
            }])

        # =========================================================
        # 3) Build payload (Tab8 autonome) — VERSION ENRICHIE
        # =========================================================
        def _build_sitrep_payload_from_df(
            df_scope,
            se,
            annee,
            date_pub,
            min_cas_zs=30,
            min_cas_prov=50,
            include_images=False,
        ):
            """
            Build un payload SITREP épidémiologique à partir de df_scope (ici df_f filtré).

            IMPORTANT:
            - include_images=False par défaut pour éviter de lancer Kaleido/Chromium
              à chaque rerun Streamlit.
            - Les images PNG pour le PDF ne sont générées qu'à la demande.
            """
            d = df_scope.copy()

            # Fix colonnes dupliquées
            if d.columns.duplicated().any():
                d = d.loc[:, ~d.columns.duplicated()].copy()

            # Filtre SE/Année
            d_se = d.copy()
            if COL_WNUM in d_se.columns:
                d_se = d_se[pd.to_numeric(d_se[COL_WNUM], errors="coerce") == int(se)]
            if COL_YEAR in d_se.columns:
                d_se = d_se[pd.to_numeric(d_se[COL_YEAR], errors="coerce") == int(annee)]

            # Cumul année <= SE
            d_cum = d.copy()
            if COL_YEAR in d_cum.columns:
                d_cum = d_cum[pd.to_numeric(d_cum[COL_YEAR], errors="coerce") == int(annee)]
            if COL_WNUM in d_cum.columns:
                d_cum = d_cum[pd.to_numeric(d_cum[COL_WNUM], errors="coerce") <= int(se)]

            def _kpi(df_):
                cases = int(len(df_))
                deaths = int(df_["is_death"].sum()) if "is_death" in df_.columns else 0
                cfr = (deaths / cases * 100.0) if cases > 0 else 0.0
                return cases, deaths, cfr

            cas_se, dec_se, cfr_se = _kpi(d_se)
            cas_cum, dec_cum, cfr_cum = _kpi(d_cum)
            weekly_summary = build_weekly_summary(d)
            selected_week_label = _build_sitrep_week_label(se, annee)
            previous_week_df = pd.DataFrame()
            previous_week_label = None

            if not weekly_summary.empty and {COL_YEAR, COL_WNUM}.issubset(weekly_summary.columns):
                weekly_reference = weekly_summary.reset_index(drop=True).copy()
                match_mask = (
                    pd.to_numeric(weekly_reference[COL_YEAR], errors="coerce") == int(annee)
                ) & (
                    pd.to_numeric(weekly_reference[COL_WNUM], errors="coerce") == int(se)
                )
                if bool(match_mask.any()):
                    selected_idx = int(np.flatnonzero(match_mask.to_numpy())[0])
                    if selected_idx > 0:
                        previous_row = weekly_reference.iloc[selected_idx - 1]
                        prev_year = pd.to_numeric(pd.Series([previous_row[COL_YEAR]]), errors="coerce").iloc[0]
                        prev_week = pd.to_numeric(pd.Series([previous_row[COL_WNUM]]), errors="coerce").iloc[0]
                        if pd.notna(prev_year) and pd.notna(prev_week):
                            previous_week_label = (
                                _build_sitrep_week_label(prev_week, prev_year)
                            )
                            prev_mask = pd.Series(True, index=d.index)
                            if COL_YEAR in d.columns:
                                prev_mask &= pd.to_numeric(d[COL_YEAR], errors="coerce") == int(prev_year)
                            if COL_WNUM in d.columns:
                                prev_mask &= pd.to_numeric(d[COL_WNUM], errors="coerce") == int(prev_week)
                            previous_week_df = d.loc[prev_mask].copy()

            prev_cases, prev_deaths, prev_cfr = _kpi(previous_week_df)
            provinces_reporting = (
                int(_surveillance_clean_text_series(d_se[COL_PROV]).nunique())
                if COL_PROV in d_se.columns else 0
            )
            zs_reporting = (
                int(_surveillance_clean_text_series(d_se[COL_ZS]).nunique())
                if COL_ZS in d_se.columns else 0
            )
            top_prov_focus = _build_surveillance_top_table(d_se, [COL_PROV], top_n=5)
            top_zs_focus = _build_surveillance_top_table(
                d_se,
                [c for c in [COL_PROV, COL_ZS] if c in d_se.columns],
                top_n=5,
            )

            # Table épidémiologique par ZS (SE sélectionnée)
            table_epi = pd.DataFrame()
            if (COL_ZS in d_se.columns) and len(d_se):
                tmp = d_se.copy()
                tmp["_cas_"] = 1
                tmp["_deces_"] = tmp["is_death"].astype(int) if "is_death" in tmp.columns else 0

                group_cols = [c for c in [COL_PROV, COL_ZS] if c in tmp.columns]
                table_epi = (
                    tmp.groupby(group_cols, as_index=False)
                       .agg(cas=("_cas_", "sum"), deces=("_deces_", "sum"))
                       .sort_values("cas", ascending=False)
                )
                if COL_PROV in table_epi.columns:
                    table_epi = table_epi.rename(columns={COL_PROV: "Province de notification"})
                if COL_ZS in table_epi.columns:
                    table_epi = table_epi.rename(columns={COL_ZS: "Zone de santé"})

            payload = {
                "meta": {"semaine": int(se), "annee": int(annee), "date_publication": date_pub},
                "kpi": {
                    "cas_semaine": cas_se,
                    "deces_semaine": dec_se,
                    "cfr_semaine": cfr_se,
                    "cas_semaine_prev": prev_cases,
                    "deces_semaine_prev": prev_deaths,
                    "cfr_semaine_prev": prev_cfr,
                    "cas_cumul": cas_cum,
                    "deces_cumul": dec_cum,
                    "cfr_cumul": cfr_cum,
                    "provinces_reporting": provinces_reporting,
                    "zs_reporting": zs_reporting,
                },
                "table_epi": table_epi,
                "selected_df": d_se,
                "cumulative_df": d_cum,
                "previous_df": previous_week_df,
                "selected_week_label": selected_week_label,
                "previous_week_label": previous_week_label,
                "top_prov_focus": top_prov_focus,
                "top_zs_focus": top_zs_focus,
            }

            # Cascade labo (si fonction dispo)
            payload["cascade"] = call_optional_function("cascade_metrics", d_se, default=pd.DataFrame())

            # Alertes sur la dernière semaine disponible — sur df_scope filtré
            payload["alertes_last"] = call_optional_function("build_alerts_last_week", d, default=pd.DataFrame())

            # Série hebdo filtrée pour visualisation / PDF
            payload["weekly"] = weekly_summary

            # Analyse spatiale et gravité
            payload.update(build_geo_tables(d_se, min_cas_zs=min_cas_zs, min_cas_prov=min_cas_prov))

            # Interprétation automatisée
            interpret = []
            provcrit = payload.get("prov_cfr_crit")
            if isinstance(provcrit, pd.DataFrame) and not provcrit.empty:
                top3 = provcrit.head(3)
                parts = [f"{r[COL_PROV]} (CFR {r['CFR_%']:.1f}%)" for _, r in top3.iterrows() if COL_PROV in top3.columns]
                if parts:
                    interpret.append("Provinces à létalité élevée (seuil) : " + ", ".join(parts))

            zscrit = payload.get("zs_cfr_crit")
            if isinstance(zscrit, pd.DataFrame) and not zscrit.empty:
                parts = []
                for _, r in zscrit.head(5).iterrows():
                    if COL_PROV in zscrit.columns:
                        parts.append(f"{r[COL_PROV]} / {r[COL_ZS]} (CFR {r['CFR_%']:.1f}%)")
                    else:
                        parts.append(f"{r[COL_ZS]} (CFR {r['CFR_%']:.1f}%)")
                if parts:
                    interpret.append("ZS à létalité élevée (seuil) : " + ", ".join(parts))

            payload["interpretation"] = interpret
            payload["summary_lines"] = _build_sitrep_summary_lines(payload)
            payload["decision_focus"] = _build_sitrep_action_lines(payload)
            payload["points_saillants"] = payload["summary_lines"][:5]
            payload["defis_besoins"] = payload["decision_focus"][:4]
            payload["perspectives"] = [
                "Les analyses détaillées de délais sont consultables dans l’onglet Surveillance.",
                "Les profils âge/sexe détaillés sont consultables dans l’onglet Profil.",
                "Les vérifications exhaustives de complétude et de qualité restent dans l’onglet Données, complétude & qualité.",
            ]
            payload["images"] = []

            if include_images:
                try:
                    wk = payload.get("weekly")
                    if isinstance(wk, pd.DataFrame) and not wk.empty and "YW" in wk.columns:
                        fig1 = build_weekly_cases_deaths_combo(
                            weekly_df=wk,
                            x_col="YW",
                            cases_col="Cas",
                            deaths_col="Décès",
                            titre=" ",
                            x_titre="Semaine (YW)",
                            y_titre_cas="Nombre de cas",
                            y_titre_deces="Nombre de décès",
                            rotation=0,
                        )
                        png1 = _plotly_to_png_bytes(fig1, scale=1)
                        if png1:
                            payload["images"].append(("Évolution hebdomadaire", png1))

                    provt = payload.get("prov_table")
                    if isinstance(provt, pd.DataFrame) and not provt.empty and COL_PROV in provt.columns:
                        fig2 = px.bar(provt.head(10), x=COL_PROV, y="Cas", title="Top 10 Provinces – Cas (SE)")
                        fig2.update_layout(xaxis_tickangle=-45)
                        png2 = _plotly_to_png_bytes(fig2, scale=1)
                        if png2:
                            payload["images"].append(("Top provinces (cas)", png2))

                    zst = payload.get("zs_table")
                    if isinstance(zst, pd.DataFrame) and not zst.empty:
                        zst2 = zst.copy()
                        if (COL_PROV in zst2.columns) and (COL_ZS in zst2.columns):
                            zst2["Prov/ZS"] = zst2[COL_PROV].astype(str) + " / " + zst2[COL_ZS].astype(str)
                            xcol = "Prov/ZS"
                        elif COL_ZS in zst2.columns:
                            xcol = COL_ZS
                        else:
                            xcol = None

                        if xcol is not None:
                            fig3 = px.bar(zst2.head(10), x=xcol, y="Cas", title="Top 10 ZS – Cas (SE)")
                            fig3.update_layout(xaxis_tickangle=-45)
                            png3 = _plotly_to_png_bytes(fig3, scale=1)
                            if png3:
                                payload["images"].append(("Top ZS (cas)", png3))
                except Exception as e:
                    logger.warning(f"[SITREP] Génération des images PDF ignorée : {e}")

            return payload

        # Paramètres de seuils (gravité)
        st.markdown("### Paramètres d’analyse et seuils d’alerte")
        cS1, cS2 = st.columns(2)
        with cS1:
            min_cas_zs = st.number_input("Seuil min cas ZS (pour CFR critique)", min_value=10, max_value=200, value=30, step=5)
        with cS2:
            min_cas_prov = st.number_input("Seuil min cas Province (pour CFR critique)", min_value=10, max_value=500, value=50, step=10)

        sitrep_payload = _build_sitrep_payload_from_df(
            df_f,
            semaine,
            annee,
            date_pub,
            min_cas_zs=min_cas_zs,
            min_cas_prov=min_cas_prov,
            include_images=False,
        )

        selected_df = sitrep_payload.get("selected_df", pd.DataFrame())
        selected_week_label = sitrep_payload.get("selected_week_label", _build_sitrep_week_label(semaine, annee))
        previous_week_label = sitrep_payload.get("previous_week_label")
        k = sitrep_payload["kpi"]

        st.divider()
        render_section_title(2, "Lecture rapide et message clé")
        st.caption(
            "Cette lecture reprend la semaine ciblée, sa comparaison avec la semaine précédente disponible et le cumul annuel, sans reprendre les analyses détaillées d’un autre onglet."
        )

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(
            "Cas (SE)",
            format_metric_value(k["cas_semaine"]),
            delta=_format_sitrep_metric_delta(k["cas_semaine"], k.get("cas_semaine_prev")),
        )
        k2.metric(
            "Décès (SE)",
            format_metric_value(k["deces_semaine"]),
            delta=_format_sitrep_metric_delta(k["deces_semaine"], k.get("deces_semaine_prev")),
        )
        k3.metric(
            "Létalité (%)",
            format_metric_value(k["cfr_semaine"], decimals=2),
            delta=_format_sitrep_metric_delta(k["cfr_semaine"], k.get("cfr_semaine_prev")),
        )
        k4.metric("Provinces actives", format_metric_value(k["provinces_reporting"]))
        k5.metric("ZS actives", format_metric_value(k["zs_reporting"]))

        if previous_week_label:
            st.caption(f"Comparaison de {selected_week_label} avec {previous_week_label}.")
        else:
            st.caption(f"Aucune semaine de référence antérieure n’est disponible pour comparer {selected_week_label}.")

        st.caption(
            (
                f"Cumul annuel jusqu’à {selected_week_label} : {format_metric_value(k['cas_cumul'])} cas, "
                f"{format_metric_value(k['deces_cumul'])} décès (létalité {format_metric_value(k['cfr_cumul'], decimals=2)}%)."
            )
        )

        if isinstance(selected_df, pd.DataFrame) and selected_df.empty:
            st.warning("La semaine sélectionnée ne contient aucun cas dans le périmètre filtré actuel.")

        summary_lines = sitrep_payload.get("summary_lines", [])
        if summary_lines:
            st.markdown("**Résumé automatique**")
            st.markdown("\n".join([f"- {line}" for line in summary_lines]))

        decision_focus = sitrep_payload.get("decision_focus", [])
        if decision_focus:
            st.markdown("**Priorités opérationnelles immédiates**")
            st.markdown("\n".join([f"- {line}" for line in decision_focus]))

        st.divider()
        render_section_title(3, "Foyers géographiques et dynamique")
        st.caption(
            "Le SITREP montre ici les foyers principaux et la dynamique utile à la décision. Les tableaux exhaustifs restent repliés pour éviter de surcharger la lecture."
        )

        geo1, geo2 = st.columns(2)
        with geo1:
            st.markdown("**Top 5 provinces de la semaine ciblée**")
            top_prov = sitrep_payload.get("top_prov_focus")
            if isinstance(top_prov, pd.DataFrame) and not top_prov.empty:
                st.caption(
                    _describe_surveillance_top_table(
                        top_prov,
                        [COL_PROV],
                        int(len(selected_df)) if isinstance(selected_df, pd.DataFrame) else 0,
                        "Aucune province exploitable pour cette semaine.",
                    )
                )
                st.dataframe(top_prov, width="stretch", hide_index=True)
            else:
                st.info("Aucune province exploitable n’est disponible pour la semaine sélectionnée.")

        with geo2:
            st.markdown("**Top 5 zones de santé de la semaine ciblée**")
            top_zs = sitrep_payload.get("top_zs_focus")
            zs_group_cols = [c for c in [COL_PROV, COL_ZS] if c in selected_df.columns] if isinstance(selected_df, pd.DataFrame) else []
            if isinstance(top_zs, pd.DataFrame) and not top_zs.empty:
                st.caption(
                    _describe_surveillance_top_table(
                        top_zs,
                        zs_group_cols,
                        int(len(selected_df)) if isinstance(selected_df, pd.DataFrame) else 0,
                        "Aucune zone de santé exploitable pour cette semaine.",
                    )
                )
                st.dataframe(top_zs, width="stretch", hide_index=True)
            else:
                st.info("Aucune zone de santé exploitable n’est disponible pour la semaine sélectionnée.")

        with st.expander("Afficher le tableau détaillé par zone de santé", expanded=False):
            table_epi = sitrep_payload.get("table_epi")
            if table_epi is not None and isinstance(table_epi, pd.DataFrame) and not table_epi.empty:
                st_dataframe_safe(table_epi, height=520)
            else:
                st.caption(
                    "Le tableau détaillé des zones de santé est indisponible : absence de données sur la période sélectionnée ou variable ZS manquante."
                )

        st.divider()
        render_section_title(4, "Signaux utiles à la décision")
        st.caption(
            "Cette section ne reprend que les signaux directement utiles à l’action immédiate. Les analyses détaillées restent accessibles dans leurs onglets spécialisés."
        )

        provcrit = sitrep_payload.get("prov_cfr_crit")
        zscrit = sitrep_payload.get("zs_cfr_crit")
        n_investigated = (
            int(pd.to_datetime(selected_df[DATE_INV], errors="coerce").notna().sum())
            if isinstance(selected_df, pd.DataFrame) and DATE_INV in selected_df.columns else 0
        )
        n_tdr = (
            int(_is_yes_series(selected_df[COL_TDR]).sum())
            if isinstance(selected_df, pd.DataFrame) and COL_TDR in selected_df.columns else 0
        )

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Cas investigués", format_metric_value(n_investigated))
        s2.metric("TDR documentés", format_metric_value(n_tdr))
        s3.metric(
            "Prov. CFR critique",
            format_metric_value(len(provcrit)) if isinstance(provcrit, pd.DataFrame) else "0",
        )
        s4.metric(
            "ZS CFR critique",
            format_metric_value(len(zscrit)) if isinstance(zscrit, pd.DataFrame) else "0",
        )

        signal_lines = []
        critical_summary = _build_sitrep_critical_cfr_summary(sitrep_payload)
        if critical_summary:
            signal_lines.append(critical_summary)
        alert_summary = _build_sitrep_alert_summary(sitrep_payload.get("alertes_last"))
        if alert_summary:
            signal_lines.append(alert_summary)

        if signal_lines:
            st.markdown("**Points de vigilance**")
            st.markdown("\n".join([f"- {line}" for line in signal_lines]))

        with st.expander("Cascade biologique et investigation", expanded=False):
            cascad = sitrep_payload.get("cascade")
            if cascad is not None and isinstance(cascad, pd.DataFrame) and not cascad.empty:
                st.markdown("**Cascade prélèvement → TDR → résultat**")
                st_dataframe_safe(cascad, height=320)
            else:
                st.caption("La cascade est indisponible : fonction absente, variables manquantes ou absence de données sur la semaine sélectionnée.")

        with st.expander("Alertes statistiques", expanded=False):
            al = sitrep_payload.get("alertes_last")
            if al is not None and isinstance(al, pd.DataFrame) and not al.empty:
                cols = [c for c in ["YW", "Cas", "Cas_prev", "var_%", "baseline_3w", "signal"] if c in al.columns]
                st_dataframe_safe(al[cols] if cols else al, height=420)
            else:
                st.caption("Les alertes sont indisponibles : fonction absente ou historique insuffisant.")

        with st.expander("Létalité critique : détail provinces et zones de santé", expanded=False):
            provt = sitrep_payload.get("prov_table")
            if provt is not None and isinstance(provt, pd.DataFrame) and not provt.empty:
                st.markdown("**Provinces — cas, décès et létalité (semaine sélectionnée)**")
                st_dataframe_safe(provt, height=300)

            if provcrit is not None and isinstance(provcrit, pd.DataFrame) and not provcrit.empty:
                st.markdown(f"**Provinces à CFR critique (Cas ≥ {int(min_cas_prov)})**")
                st_dataframe_safe(provcrit, height=260)
            else:
                st.caption("Aucune province ne dépasse le seuil critique défini.")

            if zscrit is not None and isinstance(zscrit, pd.DataFrame) and not zscrit.empty:
                st.markdown(f"**ZS à CFR critique (Cas ≥ {int(min_cas_zs)})**")
                st_dataframe_safe(zscrit.head(30), height=420)
            else:
                st.caption("Aucune ZS ne dépasse le seuil critique défini.")

        st.divider()
        render_section_title(5, "Articulation avec les autres onglets")
        st.caption("Chaque onglet garde une fonction distincte pour éviter les doublons dans le tableau de bord.")
        st.markdown(
            "\n".join(
                [
                    "- `SITREP` : synthèse courte, foyers prioritaires, signaux et export PDF.",
                    "- `Surveillance` : dynamique par fenêtres temporelles, létalité et analyses détaillées de promptitude.",
                    "- `Profil` : structure démographique, tableaux descriptifs détaillés et stratifications.",
                    "- `Données, complétude & qualité` : cohérence, doublons, complétude et tableaux de contrôle détaillés.",
                ]
            )
        )

        # =========================================================
        # 5) Exportation PDF
        # =========================================================
        st.divider()
        render_section_title(6, "Exportation")

        if "export_sitrep_pdf" in globals() and callable(export_sitrep_pdf):
            cexp1, cexp2 = st.columns([1, 1])

            with cexp1:
                prepare_pdf = st.button(
                    "Préparer le SITREP PDF",
                    type="primary",
                    key="prepare_sitrep_pdf_btn",
                )

            with cexp2:
                include_pdf_images = st.checkbox(
                    "Inclure les graphiques dans le PDF",
                    value=False,
                    key="include_pdf_images_chk",
                    help="Option plus lourde sur Streamlit Cloud. À activer seulement si nécessaire.",
                )

            if prepare_pdf:
                with st.spinner("Préparation du PDF en cours..."):
                    try:
                        pdf_payload = _build_sitrep_payload_from_df(
                            df_f,
                            semaine,
                            annee,
                            date_pub,
                            min_cas_zs=min_cas_zs,
                            min_cas_prov=min_cas_prov,
                            include_images=include_pdf_images,
                        )

                        pdf_bytes = export_sitrep_pdf(pdf_payload)

                        st.download_button(
                            "⬇️ Télécharger le SITREP épidémiologique (PDF)",
                            data=pdf_bytes,
                            file_name=f"SITREP_epidemiologique_CHOLERA_SE{int(semaine):02d}_{int(annee)}.pdf",
                            mime="application/pdf",
                            key="sitrep_dl_pdf",
                        )

                        if include_pdf_images:
                            st.caption("PDF généré avec tentative d’inclusion des graphiques.")
                        else:
                            st.caption("PDF généré sans graphiques intégrés pour maximiser la stabilité.")
                    except Exception as e:
                        st.error(f"Erreur lors de l’exportation PDF : {e}")
        else:
            st.error("La fonction export_sitrep_pdf(payload) n’est pas définie dans ce script.")

# =========================
# TAB 9 — IDSR : Helpers robuste
# =========================
with tab_idsr:
    st.markdown("## IDSR — Surveillance agrégée hebdomadaire")

    tab_help(
        "Comment lire cet onglet",
        """
    **🎯 Objectif** : analyser les tendances IDSR (cas/décès/CFR) par maladie et par niveau géographique (province/ZS),
    à partir d’un fichier agrégé par semaine.

    **✅ Inclus**
    - Évolution des cas/décès par semaine
    - CFR recalculé et comparaison avec Taux_letalite (si disponible)
    - Top provinces / ZS
    - Contrôles de cohérence (totaux vs tranches d’âge)
    - Mode secours si Année-Semaine (YW) non exploitable : filtre sur Numéro de semaine uniquement
    """,
        expanded=False
    )

    # -------------------------------------------------------------------------
    # 1) Chargement fichier IDSR
    # -------------------------------------------------------------------------
    st.caption("Téléverser un fichier IDSR agrégé (.xlsx) depuis cet onglet.")
    up = st.file_uploader("Fichier IDSR agrégé", type=["xlsx"], key="idsr_upl")

    default_path = "rdc_compilation_IDS_RDC_SE01_SE03_25_01_2026_00_07_33.xlsx"

    if up is not None:
        # priorité à la feuille IDS_RDC; sinon première feuille
        try:
            df_idsr = load_excel_cached(up, sheet_name="IDS_RDC")
        except Exception:
            df_idsr = load_excel_cached(up)
        src = "upload"
    else:
        try:
            df_idsr = load_excel_cached(default_path, sheet_name="IDS_RDC")
            src = default_path
        except Exception:
            try:
                df_idsr = load_excel_cached(default_path)
                src = default_path
            except Exception:
                df_idsr = pd.DataFrame()
                src = None

    if df_idsr.empty:
        st.info("Veuillez charger un fichier IDSR agrégé (.xlsx) pour afficher les analyses.")
    else:
        st.success(f"Fichier chargé : {src} | Lignes: {len(df_idsr):,}")

        # ---------------------------------------------------------------------
        # 2) Harmonisation colonnes (BRUT vs COMPILÉ)
        # ---------------------------------------------------------------------
        df_idsr = df_idsr.copy()

        rename_map = {
            # Identifiants
            "NUM": "Num",
            "PAYS": "Pays",
            "PROV": "Province_notification",
            "Province": "Province_notification",
            "ZS": "Zone_de_sante_notification",
            "Zone_de_sante": "Zone_de_sante_notification",
            "POP": "Population",

            # GIS (si disponible)
            "prov_GIS": "Province_GIS",
            "Prov_GIS": "Province_GIS",
            "Province_GIS": "Province_GIS",
            "zs_GIS": "ZS_GIS",
            "ZS_GIS": "ZS_GIS",
            "ZoneSante_GIS": "ZS_GIS",


            # Temps
            "NUMSEM": "Num_semaine_epid",
            "Semaine": "Num_semaine_epid",
            # DEBUTSEM reste inchangé

            # Maladie
            "MALADIE": "Maladie",
            "disease": "Maladie",

            # Tranches âge (cas)
            "C328TNN": "Cas_tnn",
            "C011MOIS": "Cas_0_11mois",
            "C1259MOIS": "Cas_12_59mois",
            "C515ANS": "Cas_5_14ans",
            "CP15ANS": "Cas_15plus",

            # Tranches âge (décès)
            "DTNN": "Deces_tnn",
            "D011MOIS": "Deces_0_11mois",
            "D1259MOIS": "Deces_12_59mois",
            "D515ANS": "Deces_5_14ans",
            "DP15ANS": "Deces_15plus",

            # Totaux & indicateurs
            "TOTALCAS": "Total_cas",
            "TOTALDECES": "Total_deces",
            "LETAL": "Taux_letalite",
            "ATTAQ": "Taux_attaque",

            # Statut & clé
            "RecStatus": "Recstatus",
            "UniqueKey": "Cle_unique",

            # Année / semaine compilées
            "Year": "Annee_epid",
            "year": "Annee_epid",
            "Annee": "Annee_epid",
        }

        df_idsr = df_idsr.rename(columns={k: v for k, v in rename_map.items() if k in df_idsr.columns})
        # ---------------------------------------------------------
        # ✅ Détecteur automatique BRUT vs COMPILÉ
        # ---------------------------------------------------------
        # BRUT: contient DEBUTSEM + NUMSEM (après rename NUMSEM -> Num_semaine_epid)
        is_brut = ("DEBUTSEM" in df_idsr.columns) and ("Num_semaine_epid" in df_idsr.columns)

        # COMPILÉ: a déjà Date_debut_semaine et/ou Annee_epid / Semaine_epid
        is_compiled = (
            ("Date_debut_semaine" in df_idsr.columns)
            or ("Annee_epid" in df_idsr.columns)
            or ("Semaine_epid" in df_idsr.columns)
        )

        # Petit diagnostic (optionnel, utile)
        with st.expander("🧩 Diagnostic colonnes (dérouler)", expanded=False):
            st.write({
                "version_detectee": "BRUTE (DEBUTSEM/NUMSEM)" if is_brut else "COMPILÉE",
                "colonnes_temps": [
                    c for c in ["DEBUTSEM", "Date_debut_semaine", "Annee_epid", "Num_semaine_epid", "Semaine_epid", "YW"]
                    if c in df_idsr.columns
                ]
            })


        # Colonnes standard
        COL_MAL = "Maladie"
        COL_PROV_ID = "Province_notification"
        COL_ZS_ID = "Zone_de_sante_notification"

        
        # ---------------------------------------------------------------------
        # 2.b) Normalisation texte (Province/ZS/Maladie) pour éviter les doublons
        # ---------------------------------------------------------------------
        for _c in ["Maladie", "Province_notification", "Zone_de_sante_notification", "Province_GIS", "ZS_GIS"]:
            if _c in df_idsr.columns:
                df_idsr[_c] = norm_text(df_idsr[_c])

        # ---------------------------------------------------------------------
        # 3) Standardisation TEMPS (robuste sur semaine)
        # ---------------------------------------------------------------------
        # 3.1 Semaine
        if "Num_semaine_epid" in df_idsr.columns:
            df_idsr["Num_semaine_epid"] = clean_week(df_idsr["Num_semaine_epid"])
        else:
            df_idsr["Num_semaine_epid"] = pd.NA

        # 3.2 Année
        if "Annee_epid" in df_idsr.columns:
            df_idsr["Annee_epid"] = clean_year(df_idsr["Annee_epid"])
        else:
            df_idsr["Annee_epid"] = pd.NA

        # si Annee_epid vide -> essayer depuis Semaine_epid
        if df_idsr["Annee_epid"].isna().all() and "Semaine_epid" in df_idsr.columns:
            df_idsr["Annee_epid"] = clean_year(df_idsr["Semaine_epid"])

        # si semaine vide -> essayer depuis Semaine_epid (dernier nombre)
        if df_idsr["Num_semaine_epid"].isna().all() and "Semaine_epid" in df_idsr.columns:
            wk = df_idsr["Semaine_epid"].astype("string").str.extract(r"(\d{1,2})\s*$", expand=False)
            df_idsr["Num_semaine_epid"] = clean_week(wk)

        # dernier recours: année depuis nom du fichier
        if df_idsr["Annee_epid"].isna().all():
            y_guess = parse_year_from_filename(src)
            if y_guess is not None:
                df_idsr["Annee_epid"] = pd.Series([y_guess] * len(df_idsr), dtype="Int64")

        
        # -----------------------------------------------------------------
        # 3.3 Si fichier COMPILÉ et dates disponibles : dériver Année/Semaine
        # -----------------------------------------------------------------
        # Si l'utilisateur a un fichier compilé avec Date_debut_semaine mais sans Annee/Num_semaine,
        # on reconstruit Annee_epid et Num_semaine_epid depuis la date (ISO year/week).
        if (("Date_debut_semaine" in df_idsr.columns) or ("DEBUTSEM" in df_idsr.columns)) and (
            df_idsr["Annee_epid"].isna().all() or df_idsr["Num_semaine_epid"].isna().all()
        ):
            _dt_src = None
            if "Date_debut_semaine" in df_idsr.columns:
                _dt_src = pd.to_datetime(df_idsr["Date_debut_semaine"], errors="coerce")
            elif "DEBUTSEM" in df_idsr.columns:
                _dt_src = pd.to_datetime(df_idsr["DEBUTSEM"], errors="coerce")

            if _dt_src is not None and _dt_src.notna().any():
                _iso = _dt_src.dt.isocalendar()
                if df_idsr["Annee_epid"].isna().all():
                    df_idsr["Annee_epid"] = pd.to_numeric(_iso["year"], errors="coerce").astype("Int64")
                if df_idsr["Num_semaine_epid"].isna().all():
                    df_idsr["Num_semaine_epid"] = pd.to_numeric(_iso["week"], errors="coerce").astype("Int64")

        # YW & YW_KEY (si année + semaine)
        df_idsr["YW"] = (
            df_idsr["Annee_epid"].astype("string")
            + "-W"
            + df_idsr["Num_semaine_epid"].astype("string").str.zfill(2)
        )
        df_idsr["YW_KEY"] = (
            df_idsr["Annee_epid"].astype("Int64") * 100
            + df_idsr["Num_semaine_epid"].astype("Int64")
        )

        # Date ISO reconstruite pour affichage (basée sur Année+Semaine)
        df_idsr["Date_debut_semaine_iso"] = [
            iso_monday_from_year_week(y, w)
            for y, w in zip(df_idsr["Annee_epid"].tolist(), df_idsr["Num_semaine_epid"].tolist())
        ]

        # ---------------------------------------------------------------------
        # 4) QC date vs semaine (si date source disponible)
        # IMPORTANT : comparaison faite en numpy float64 (évite pd.NA bool ambigu)
        # ---------------------------------------------------------------------
        if "Date_debut_semaine" in df_idsr.columns:
            src_dt = pd.to_datetime(df_idsr["Date_debut_semaine"], errors="coerce")
        elif "DEBUTSEM" in df_idsr.columns:
            src_dt = pd.to_datetime(df_idsr["DEBUTSEM"], errors="coerce")
            df_idsr["Date_debut_semaine"] = df_idsr["DEBUTSEM"]  # copie visible
        else:
            src_dt = pd.Series(pd.NaT, index=df_idsr.index)

        has_date = src_dt.notna()

        if has_date.any():
            iso = src_dt.dt.isocalendar()

            iso_year = pd.to_numeric(iso["year"], errors="coerce").to_numpy(dtype="float64")
            iso_week = pd.to_numeric(iso["week"], errors="coerce").to_numpy(dtype="float64")

            y = pd.to_numeric(df_idsr["Annee_epid"], errors="coerce").to_numpy(dtype="float64")
            w = pd.to_numeric(df_idsr["Num_semaine_epid"], errors="coerce").to_numpy(dtype="float64")

            ok_mask = has_date.to_numpy() & (iso_year == y) & (iso_week == w)

            df_idsr["QC_Date_vs_Semaine"] = np.where(
                ~has_date.to_numpy(), "NA",
                np.where(ok_mask, "✅ OK", "❌ KO")
            )
        else:
            df_idsr["QC_Date_vs_Semaine"] = "NA"

        # ---------------------------------------------------------------------
        # 5) Axe temps UNIQUE pour tri/plots (gère mode secours)
        # ---------------------------------------------------------------------
        # TIME_KEY : tri stable (priorité YW_KEY sinon Num_semaine)
        yw_key_num = pd.to_numeric(df_idsr.get("YW_KEY"), errors="coerce")
        wnum_num = pd.to_numeric(df_idsr.get("Num_semaine_epid"), errors="coerce")

        df_idsr["TIME_KEY"] = np.where(yw_key_num.notna(), yw_key_num, wnum_num)

        # TIME_LAB : affichage (priorité YW sinon W##)
        df_idsr["TIME_LAB"] = np.where(
            df_idsr.get("YW", pd.Series([""] * len(df_idsr), index=df_idsr.index)).astype("string").str.contains(r"-W", na=False),
            df_idsr["YW"].astype("string"),
            "W" + wnum_num.astype("Int64").astype("string").str.zfill(2)
        )

        # ⚠️ Confort utilisateur (sans changer la logique) :
        # - Si base BRUTE (ou année indisponible), afficher W## plutôt que YYYY-W##
        _wlab = "W" + wnum_num.astype("Int64").astype("string").str.zfill(2)
        if "is_brut" in locals() and is_brut:
            df_idsr["TIME_LAB"] = _wlab
        else:
            if "Annee_epid" in df_idsr.columns and df_idsr["Annee_epid"].isna().all():
                df_idsr["TIME_LAB"] = _wlab

        # ---------------------------------------------------------------------
        # 6) Conversions numériques (variables d'analyse)

        # ---------------------------------------------------------------------
        
        # ---------------------------------------------------------------------
        # 6.a) Somme des tranches d’âge (cas/décès) + reconstruction prudente des totaux
        # ---------------------------------------------------------------------
        # On calcule toujours la somme des tranches (utile pour QC/écarts),
        # puis on ne reconstruit Total_cas/Total_deces QUE s'ils sont absents
        # ou très majoritairement manquants.
        cas_parts = [c for c in ["Cas_tnn", "Cas_0_11mois", "Cas_12_59mois", "Cas_5_14ans", "Cas_15plus"] if c in df_idsr.columns]
        dec_parts = [c for c in ["Deces_tnn", "Deces_0_11mois", "Deces_12_59mois", "Deces_5_14ans", "Deces_15plus"] if c in df_idsr.columns]

        if cas_parts:
            df_idsr["Total_cas_age"] = df_idsr[cas_parts].sum(axis=1, min_count=1)
        else:
            df_idsr["Total_cas_age"] = pd.NA

        if dec_parts:
            df_idsr["Total_deces_age"] = df_idsr[dec_parts].sum(axis=1, min_count=1)
        else:
            df_idsr["Total_deces_age"] = pd.NA

        # Reconstruction / complétion prudente des totaux (ne pas écraser les totaux valides)
        if "Total_cas" not in df_idsr.columns:
            df_idsr["Total_cas"] = df_idsr["Total_cas_age"]
        else:
            if df_idsr["Total_cas"].isna().mean() > 0.5:
                df_idsr["Total_cas"] = df_idsr["Total_cas"].fillna(df_idsr["Total_cas_age"])

        if "Total_deces" not in df_idsr.columns:
            df_idsr["Total_deces"] = df_idsr["Total_deces_age"]
        else:
            if df_idsr["Total_deces"].isna().mean() > 0.5:
                df_idsr["Total_deces"] = df_idsr["Total_deces"].fillna(df_idsr["Total_deces_age"])

        # ---------------------------------------------------------------------
        df_idsr = to_numeric_cols(df_idsr, [
            "Population",
            "Total_cas", "Total_deces", "Taux_letalite", "Taux_attaque",
            "Cas_tnn", "Cas_0_11mois", "Cas_12_59mois", "Cas_5_14ans", "Cas_15plus",
            "Deces_tnn", "Deces_0_11mois", "Deces_12_59mois", "Deces_5_14ans", "Deces_15plus"
        ])

        # Diagnostic rapide
        with st.expander("🧩 Diagnostic (temps & QC) – déplier", expanded=False):
            st.write({
                "colonnes_temps": [c for c in [
                    "Annee_epid", "Num_semaine_epid", "YW", "YW_KEY",
                    "TIME_LAB", "TIME_KEY", "Date_debut_semaine_iso", "QC_Date_vs_Semaine"
                ] if c in df_idsr.columns],
                "qc_date_vs_semaine": df_idsr["QC_Date_vs_Semaine"].value_counts(dropna=False).to_dict()
            })

        # ---------------------------------------------------------------------
        # 6.b) QC actionnable : afficher & exporter les lignes KO (si existent)
        # ---------------------------------------------------------------------
        if "QC_Date_vs_Semaine" in df_idsr.columns:
            df_qc_ko = df_idsr[df_idsr["QC_Date_vs_Semaine"] == "❌ KO"].copy()
            if not df_qc_ko.empty:
                with st.expander("🚩 Top lignes QC KO (Date vs Année-Semaine) – déplier", expanded=False):
                    show_cols = [c for c in [
                        "Maladie", "Province_notification", "Zone_de_sante_notification",
                        "DEBUTSEM", "Date_debut_semaine",
                        "Annee_epid", "Num_semaine_epid", "Semaine_epid", "YW",
                        "QC_Date_vs_Semaine"
                    ] if c in df_qc_ko.columns]
                    st.dataframe(df_qc_ko[show_cols].head(20), width="stretch")

                    csv_ko = df_qc_ko[show_cols].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Télécharger QC_KO.csv",
                        data=csv_ko,
                        file_name="QC_KO.csv",
                        mime="text/csv",
                        key="tab9_dl_qc_ko"
                    )


        # ---------------------------------------------------------------------
        # 7) Filtres : maladie, province, ZS, semaines (mode normal ou secours)
        # ---------------------------------------------------------------------
        # ---- Filtres (sur une seule ligne) : Maladie / Province / ZS / Année(DEBUTSEM) / Temps
        cA, cB, cC, fD, cD = st.columns(5)

        with cA:
            maladies = sorted([
                x for x in df_idsr.get(COL_MAL, pd.Series(dtype="object")).dropna().unique().tolist()
                if str(x).strip() != ""
            ])
            mal_sel = st.multiselect(
                "Maladie",
                options=maladies,
                default=[],
                help="Laisse vide pour toutes les maladies",
                key="tab9_mal_sel"
            )

        with cB:
            provs = sorted([
                x for x in df_idsr.get(COL_PROV_ID, pd.Series(dtype="object")).dropna().unique().tolist()
                if str(x).strip() != ""
            ])
            prov_sel = st.multiselect(
                "Province",
                options=provs,
                default=[],
                help="Laisse vide pour toutes les provinces",
                key="tab9_prov_sel"
            )

        with cC:
            if COL_ZS_ID in df_idsr.columns:
                if prov_sel and (COL_PROV_ID in df_idsr.columns):
                    zs_pool = df_idsr[df_idsr[COL_PROV_ID].isin(prov_sel)]
                else:
                    zs_pool = df_idsr

                zss = sorted([
                    x for x in zs_pool.get(COL_ZS_ID, pd.Series(dtype="object")).dropna().unique().tolist()
                    if str(x).strip() != ""
                ])

                zs_sel = st.multiselect(
                    "Zone de santé",
                    options=zss,
                    default=[],
                    help="Vide = toutes les ZS (filtrées par province si province sélectionnée)",
                    key="tab9_zs_sel"
                )
            else:
                zs_sel = []
                st.info("Colonne Zone_de_sante_notification absente (filtre ZS indisponible).")

        # Filtre Année (DEBUTSEM) — choix multiple
        years_selected = None  # utilisé plus loin pour messages/contrôles
        with fD:
            if "DEBUTSEM" in df_idsr.columns:
                _year_pool = df_idsr.copy()

                if mal_sel and (COL_MAL in _year_pool.columns):
                    _year_pool = _year_pool[_year_pool[COL_MAL].isin(mal_sel)]

                if prov_sel and (COL_PROV_ID in _year_pool.columns):
                    _year_pool = _year_pool[_year_pool[COL_PROV_ID].isin(prov_sel)]

                if zs_sel and (COL_ZS_ID in _year_pool.columns):
                    _year_pool = _year_pool[_year_pool[COL_ZS_ID].isin(zs_sel)]

                _debutsem = _year_pool["DEBUTSEM"]
                if pd.api.types.is_numeric_dtype(_debutsem):
                    _debutsem_dt = pd.to_datetime(_debutsem, unit="D", origin="1899-12-30", errors="coerce")
                else:
                    _debutsem_dt = pd.to_datetime(_debutsem, errors="coerce")

                years_available = sorted(_debutsem_dt.dt.year.dropna().astype(int).unique().tolist())

                if years_available:
                    years_selected = st.multiselect(
                        "Année (DEBUTSEM)",
                        options=years_available,
                        default=years_available,
                        key="tab9_years_debutsem",
                        help="Très utile en mode BRUT (WNUM) pour éviter de mélanger plusieurs années."
                    )
                else:
                    years_selected = []
                    st.info("Aucune année exploitable trouvée dans DEBUTSEM.")
            else:
                years_selected = []
                st.info("Colonne DEBUTSEM absente (filtre Année indisponible).")


        # Filtre semaines : logique robuste BRUT vs COMPILÉ
        with cD:
            # Badge BRUT / COMPILÉ (aide visuelle)
            _tag = "BRUT" if is_brut else "COMPILÉ"
            _bg = "#ffecb5" if is_brut else "#d1e7dd"
            _border = "#d39e00" if is_brut else "#0f5132"
            _txt = "#111" if is_brut else "#0f5132"
            st.markdown(
                f"""<div style='display:inline-block;padding:2px 10px;border-radius:999px;
                background:{_bg};border:1px solid {_border};color:{_txt};font-weight:700;font-size:12px'>
                IDS {_tag}
                </div>""",
                unsafe_allow_html=True
            )
            # st.caption("BRUT : filtre par numéro de semaine. COMPILÉ : filtre Année–Semaine (YW) si disponible.")

            # ---- utilitaire local : liste de semaines exploitables
            def _get_weeks_list(_df: pd.DataFrame) -> list:
                w = pd.to_numeric(_df.get("Num_semaine_epid"), errors="coerce")
                weeks = (
                    w.dropna()
                    .astype(int)
                    .sort_values()
                    .unique()
                    .tolist()
                )
                # fallback : tenter depuis Semaine_epid si Num_semaine_epid vide
                if (not weeks) and ("Semaine_epid" in _df.columns):
                    wk = _df["Semaine_epid"].astype("string").str.extract(r"(\d{1,2})\s*$", expand=False)
                    weeks = (
                        pd.to_numeric(wk, errors="coerce")
                        .dropna()
                        .astype(int)
                        .sort_values()
                        .unique()
                        .tolist()
                    )
                return weeks

            # ---- Détection capacité YW
            yw_key_series = pd.to_numeric(df_idsr.get("YW_KEY"), errors="coerce")
            has_yw = ("YW_KEY" in df_idsr.columns) and yw_key_series.notna().any()

            # ---- Cas BRUT : on force le filtre Numéro de semaine (plus sûr en opérationnel)
            if is_brut:
                # st.info("Base IDS BRUTE détectée : filtre temporel par **Numéro de semaine**.")
                week_filter_mode = "WNUM"

                weeks = _get_weeks_list(df_idsr)

                if weeks:
                    col_min, col_max = st.columns(2)
                    with col_min:
                        w_min = st.selectbox(
                            "Semaine min (Numéro semaine)",
                            options=weeks,
                            index=0,
                            key="tab9_w_min",
                        )
                    with col_max:
                        w_max = st.selectbox(
                            "Semaine max (Numéro semaine)",
                            options=weeks,
                            index=len(weeks) - 1,
                            key="tab9_w_max",
                        )

                    if weeks.index(w_min) > weeks.index(w_max):
                        w_min, w_max = w_max, w_min
                else:
                    st.warning("Aucune semaine exploitable (Num_semaine_epid / Semaine_epid).")
                    week_filter_mode = None

            # ---- Cas COMPILÉ : proposer Année-Semaine (YW) si dispo, sinon Numéro de semaine
            else:
                week_filter_mode = None

                if has_yw:
                    # Mode normal : Année+Semaine
                    yw_table = df_idsr[["YW", "YW_KEY"]].copy()
                    yw_table["YW_KEY"] = pd.to_numeric(yw_table["YW_KEY"], errors="coerce")
                    yw_table = yw_table.dropna().drop_duplicates().sort_values("YW_KEY")

                    yws = yw_table["YW"].astype(str).tolist()
                    if yws:
                        col_min, col_max = st.columns(2)
                        with col_min:
                            yw_min = st.selectbox(
                                "Semaine min (Année-Semaine)",
                                options=yws,
                                index=0,
                                key="tab9_yw_min",
                            )
                        with col_max:
                            yw_max = st.selectbox(
                                "Semaine max (Année-Semaine)",
                                options=yws,
                                index=len(yws) - 1,
                                key="tab9_yw_max",
                            )

                        if yws.index(yw_min) > yws.index(yw_max):
                            yw_min, yw_max = yw_max, yw_min

                        min_key = float(yw_table.loc[yw_table["YW"] == yw_min, "YW_KEY"].iloc[0])
                        max_key = float(yw_table.loc[yw_table["YW"] == yw_max, "YW_KEY"].iloc[0])
                        week_filter_mode = "YW"

                # Fallback / option : Numéro de semaine (toujours utile)
                weeks = _get_weeks_list(df_idsr)
                if weeks:
                    col_min, col_max = st.columns(2)
                    with col_min:
                        w_min = st.selectbox(
                            "Semaine min (Numéro semaine)",
                            options=weeks,
                            index=0,
                            key="tab9_w_min",
                        )
                    with col_max:
                        w_max = st.selectbox(
                            "Semaine max (Numéro semaine)",
                            options=weeks,
                            index=len(weeks) - 1,
                            key="tab9_w_max",
                        )

                    if weeks.index(w_min) > weeks.index(w_max):
                        w_min, w_max = w_max, w_min

                    if week_filter_mode is None:
                        week_filter_mode = "WNUM"
                else:
                    if week_filter_mode is None:
                        st.warning("Aucune semaine exploitable (YW_KEY / Num_semaine_epid).")
                        week_filter_mode = None

        # 8) Appliquer filtres
        # ---------------------------------------------------------------------
        df9 = df_idsr.copy()

        if mal_sel and COL_MAL in df9.columns:
            df9 = df9[df9[COL_MAL].isin(mal_sel)]

        if prov_sel and COL_PROV_ID in df9.columns:
            df9 = df9[df9[COL_PROV_ID].isin(prov_sel)]

        if zs_sel and COL_ZS_ID in df9.columns:
            df9 = df9[df9[COL_ZS_ID].isin(zs_sel)]

        # Filtre Année (DEBUTSEM) si sélection disponible
        if years_selected and ("DEBUTSEM" in df9.columns):
            _debutsem = df9["DEBUTSEM"]
            if pd.api.types.is_numeric_dtype(_debutsem):
                _debutsem_dt = pd.to_datetime(_debutsem, unit="D", origin="1899-12-30", errors="coerce")
            else:
                _debutsem_dt = pd.to_datetime(_debutsem, errors="coerce")
            _yrs = _debutsem_dt.dt.year
            df9 = df9[_yrs.isin([int(y) for y in years_selected])]
        elif years_selected and ("Annee_epid" in df9.columns):
            # fallback si DEBUTSEM absent
            df9 = df9[pd.to_numeric(df9["Annee_epid"], errors="coerce").isin([int(y) for y in years_selected])]


        # Filtre semaines selon mode
        # Copie avant filtre semaines: utile pour 'Situation épidémiologique — dernière semaine disponible' (focus sur semaine max)
        df9_base = df9.copy()

        # Filtre semaines selon mode
        if week_filter_mode == "YW":
            df9["YW_KEY"] = pd.to_numeric(df9["YW_KEY"], errors="coerce")
            df9 = df9[df9["YW_KEY"].between(min_key, max_key, inclusive="both")]

        elif week_filter_mode == "WNUM":
            df9["Num_semaine_epid"] = pd.to_numeric(df9["Num_semaine_epid"], errors="coerce")
            df9 = df9[df9["Num_semaine_epid"].between(w_min, w_max, inclusive="both")]

        st.caption(f"📌 Périmètre analytique filtré : {len(df9):,} lignes")
        # -------------------------------------------------------------
        # Plusieurs années en mode WNUM → pas de deltas interprétables
        # -------------------------------------------------------------
        disable_deltas = False
        if week_filter_mode == "WNUM" and "Annee_epid" in df9.columns:
            _yrs_scope = pd.to_numeric(df9["Annee_epid"], errors="coerce").dropna().unique().tolist()
            if len(_yrs_scope) > 1:
                disable_deltas = True
                st.info(
                    "ℹ️ Plusieurs années détectées (mode BRUT / WNUM). "
                    "Les variations vs semaine-1 sont désactivées."
                )



        # ---------------------------------------------------------------------
        # 8.b) Résumé de la période filtrée (confort utilisateur)
        # ---------------------------------------------------------------------
        if not df9.empty:
            _tot_cas = pd.to_numeric(df9.get("Total_cas"), errors="coerce").sum(skipna=True) if "Total_cas" in df9.columns else np.nan
            _tot_dec = pd.to_numeric(df9.get("Total_deces"), errors="coerce").sum(skipna=True) if "Total_deces" in df9.columns else np.nan
            _cfr = (float(_tot_dec) / float(_tot_cas) * 100.0) if (pd.notna(_tot_cas) and _tot_cas > 0 and pd.notna(_tot_dec)) else np.nan

            _n_mal = df9[COL_MAL].nunique(dropna=True) if COL_MAL in df9.columns else 0
            _n_prov = df9[COL_PROV_ID].nunique(dropna=True) if COL_PROV_ID in df9.columns else 0
            _n_zs = df9[COL_ZS_ID].nunique(dropna=True) if COL_ZS_ID in df9.columns else 0

            _period_start = pd.to_datetime(df9.get("Date_debut_semaine_iso"), errors="coerce") if "Date_debut_semaine_iso" in df9.columns else pd.Series(dtype="datetime64[ns]")
            if not _period_start.empty and _period_start.notna().any():
                _period_min = _period_start.min()
                _period_max = _period_start.max() + pd.Timedelta(days=6)
                _period_label = f"{_period_min:%d/%m/%Y} -> {_period_max:%d/%m/%Y}"
            else:
                _period_label = "Période indisponible"

            _time_values = df9["TIME_LAB"].dropna().astype(str).tolist() if "TIME_LAB" in df9.columns else []
            _time_span = f"{min(_time_values)} -> {max(_time_values)}" if _time_values else "Fenêtre hebdo indisponible"

            st.markdown("### Résumé de la période filtrée")
            r1, r2, r3, r4, r5, r6 = st.columns(6)
            r1.metric("Cas (total)", f"{int(_tot_cas):,}" if pd.notna(_tot_cas) else "NA")
            r2.metric("Décès (total)", f"{int(_tot_dec):,}" if pd.notna(_tot_dec) else "NA")
            r3.metric("CFR (recalculé)", f"{_cfr:.2f}%" if pd.notna(_cfr) else "NA")
            r4.metric("Maladies", f"{_n_mal:,}")
            r5.metric("Provinces", f"{_n_prov:,}")
            r6.metric("Zones de santé", f"{_n_zs:,}")
            st.caption(f"Période couverte : **{_period_label}** | Fenêtre hebdo : **{_time_span}**")


        if df9.empty:
            st.info("Aucune donnée n’est disponible après application des filtres analytiques.")
        else:
            st.divider()

            with st.expander("🧭 Intensité géographique par maladie", expanded=False):
                if (COL_MAL in df9.columns) and (COL_PROV_ID in df9.columns) and ("Total_cas" in df9.columns):
                    heat_src = (
                        df9.groupby([COL_MAL, COL_PROV_ID], as_index=False)
                        .agg(Cas=("Total_cas", "sum"))
                    )
                    heat_src["Cas"] = pd.to_numeric(heat_src["Cas"], errors="coerce").fillna(0)
                    heat_src = heat_src[heat_src["Cas"] > 0]

                    if not heat_src.empty:
                        h1, h2 = st.columns(2)
                        with h1:
                            top_prov_n = st.slider(
                                "Top provinces à afficher",
                                min_value=5,
                                max_value=25,
                                value=12,
                                step=1,
                                key="tab9_heatmap_top_prov",
                            )
                        with h2:
                            top_mal_n = st.slider(
                                "Top maladies à afficher",
                                min_value=1,
                                max_value=20,
                                value=min(10, int(heat_src[COL_MAL].nunique())),
                                step=1,
                                key="tab9_heatmap_top_mal",
                            )

                        top_provs = (
                            heat_src.groupby(COL_PROV_ID)["Cas"]
                            .sum()
                            .sort_values(ascending=False)
                            .head(int(top_prov_n))
                            .index.tolist()
                        )
                        top_mals = (
                            heat_src.groupby(COL_MAL)["Cas"]
                            .sum()
                            .sort_values(ascending=False)
                            .head(int(top_mal_n))
                            .index.tolist()
                        )
                        heat_view = heat_src[
                            heat_src[COL_PROV_ID].isin(top_provs) & heat_src[COL_MAL].isin(top_mals)
                        ].copy()

                        heat_tbl = heat_view.pivot_table(
                            index=COL_MAL,
                            columns=COL_PROV_ID,
                            values="Cas",
                            aggfunc="sum",
                            fill_value=0,
                            observed=False,
                        )

                        if not heat_tbl.empty:
                            fig_heat = px.imshow(
                                heat_tbl,
                                aspect="auto",
                                color_continuous_scale=["#eef4fb", "#2369be", "#103d82"],
                                title="Cas agrégés par maladie et province",
                                labels={"x": "Province", "y": "Maladie", "color": "Cas"},
                            )
                            fig_heat.update_layout(height=460)
                            st.plotly_chart(fig_heat, width="stretch", key="idsr_heatmap_mal_prov")
                            with st.expander("Tableau maladie × province", expanded=False):
                                st.dataframe(heat_tbl.reset_index(), width="stretch", height=320, hide_index=True)
                        else:
                            st.info("La heatmap maladie × province est vide après application des filtres de volume.")
                    else:
                        st.info("Aucun volume exploitable n'est disponible pour construire la heatmap maladie × province.")
                else:
                    st.info("La heatmap maladie × province est indisponible : colonnes Maladie / Province / Total_cas absentes.")

            st.divider()
            render_idsr_maps_section(
                df_f=df9,
                province_col=COL_PROV_ID,
                zs_col=COL_ZS_ID if COL_ZS_ID in df9.columns else None,
                cases_col="Total_cas",
            )

            st.divider()

            # -----------------------------------------------------------------
            # 9) Série temporelle : cas/décès/CFR (robuste sur TIME_KEY/LAB)
            # -----------------------------------------------------------------
            required_cols = ["Total_cas", "Total_deces"]
            missing = [c for c in required_cols if c not in df9.columns]
            if missing:
                st.error(f"Variables manquantes pour l’analyse temporelle : {', '.join(missing)}")
            else:
                # Agrégation hebdo
                weekly = df9.groupby(["TIME_LAB", "TIME_KEY"], as_index=False).agg(
                    Cas=("Total_cas", "sum"),
                    Deces=("Total_deces", "sum"),
                    Taux_letalite_moy=("Taux_letalite", "mean") if "Taux_letalite" in df9.columns else ("Total_cas", "size"),
                    Taux_attaque_moy=("Taux_attaque", "mean") if "Taux_attaque" in df9.columns else ("Total_cas", "size"),
                )

                # CFR recalculé (en %) : plus fiable que moyenne LETAL
                weekly["CFR_recalc_pct"] = np.where(
                    weekly["Cas"] > 0,
                    (weekly["Deces"] / weekly["Cas"]) * 100.0,
                    np.nan
                )

                # si LETAL existe, garder une version en % (supposée déjà en %)
                if "Taux_letalite_moy" in weekly.columns:
                    weekly["LETAL_moy_pct"] = weekly["Taux_letalite_moy"]
                else:
                    weekly["LETAL_moy_pct"] = np.nan

                # si taux_ n'existent pas, on met NaN
                if "Taux_letalite" not in df9.columns:
                    weekly["Taux_letalite_moy"] = np.nan
                if "Taux_attaque" not in df9.columns:
                    weekly["Taux_attaque_moy"] = np.nan

                weekly["CFR_calc_%"] = np.where(
                    weekly["Cas"] > 0, (weekly["Deces"] / weekly["Cas"]) * 100, np.nan
                )

                weekly_sorted = weekly.sort_values("TIME_KEY").reset_index(drop=True)

                # -------------------------------------------------------------
                # 9.b) Comparaison "Tranches d’âge" vs "Totaux" (visualisation)
                # -------------------------------------------------------------
                # Objectif : afficher 2 lignes de KPI (Cas/Décès/CFR) :
                # - Ligne 1 : somme des tranches d’âge (Cas_* / Deces_*) => détecte incohérences
                # - Ligne 2 : Totaux (Total_cas / Total_deces) => référence opérationnelle

                age_case_cols = [c for c in ["Cas_tnn", "Cas_0_11mois", "Cas_12_59mois", "Cas_5_14ans", "Cas_15plus"] if c in df9.columns]
                age_death_cols = [c for c in ["Deces_tnn", "Deces_0_11mois", "Deces_12_59mois", "Deces_5_14ans", "Deces_15plus"] if c in df9.columns]

                weekly_age_sorted = None
                if age_case_cols or age_death_cols:
                    _tmp = df9.copy()

                    if age_case_cols:
                        _tmp["Cas_age_sum"] = _tmp[age_case_cols].sum(axis=1, min_count=1)
                    else:
                        _tmp["Cas_age_sum"] = np.nan

                    if age_death_cols:
                        _tmp["Deces_age_sum"] = _tmp[age_death_cols].sum(axis=1, min_count=1)
                    else:
                        _tmp["Deces_age_sum"] = np.nan

                    weekly_age = _tmp.groupby(["TIME_LAB", "TIME_KEY"], as_index=False).agg(
                        Cas=("Cas_age_sum", "sum"),
                        Deces=("Deces_age_sum", "sum"),
                    )
                    weekly_age["CFR_calc_%"] = np.where(
                        weekly_age["Cas"] > 0, (weekly_age["Deces"] / weekly_age["Cas"]) * 100, np.nan
                    )
                    weekly_age_sorted = weekly_age.sort_values("TIME_KEY").reset_index(drop=True)
                # KPI dernière semaine + variation vs semaine-1

                last = None
                prev = None

                # On calcule sur df9_base (filtres maladie/province/ZS), sans dépendre de week_min.
                if "df9_base" in locals() and not df9_base.empty:
                    if week_filter_mode == "YW" and "YW_KEY" in df9_base.columns:
                        _b = df9_base.copy()
                        _b["YW_KEY"] = pd.to_numeric(_b["YW_KEY"], errors="coerce")
                        last_key = max_key if max_key is not None else _b["YW_KEY"].dropna().max()

                        df_last_kpi = _b[_b["YW_KEY"] == last_key]
                        keys = _b["YW_KEY"].dropna().drop_duplicates().sort_values().tolist()
                        prev_key = keys[-2] if len(keys) >= 2 else None
                        df_prev_kpi = _b[_b["YW_KEY"] == prev_key] if prev_key is not None else pd.DataFrame()

                        cas_last = pd.to_numeric(df_last_kpi.get("Total_cas"), errors="coerce").sum(skipna=True)
                        dec_last = pd.to_numeric(df_last_kpi.get("Total_deces"), errors="coerce").sum(skipna=True)
                        cfr_last = (float(dec_last) / float(cas_last) * 100.0) if (pd.notna(cas_last) and cas_last > 0 and pd.notna(dec_last)) else np.nan

                        cas_prev = pd.to_numeric(df_prev_kpi.get("Total_cas"), errors="coerce").sum(skipna=True) if not df_prev_kpi.empty else np.nan
                        dec_prev = pd.to_numeric(df_prev_kpi.get("Total_deces"), errors="coerce").sum(skipna=True) if not df_prev_kpi.empty else np.nan
                        cfr_prev = (float(dec_prev) / float(cas_prev) * 100.0) if (pd.notna(cas_prev) and cas_prev > 0 and pd.notna(dec_prev)) else np.nan

                        lab_last = df_last_kpi["TIME_LAB"].iloc[0] if ("TIME_LAB" in df_last_kpi.columns and not df_last_kpi.empty) else str(int(last_key) if pd.notna(last_key) else "NA")

                        last = {"TIME_LAB": lab_last, "Cas": cas_last, "Deces": dec_last, "CFR_calc_%": cfr_last}
                        prev = {"Cas": cas_prev, "Deces": dec_prev, "CFR_calc_%": cfr_prev} if not df_prev_kpi.empty else None

                    elif week_filter_mode == "WNUM" and "Num_semaine_epid" in df9_base.columns and "Annee_epid" in df9_base.columns:
                        _b = df9_base.copy()
                        _b["Num_semaine_epid"] = pd.to_numeric(_b["Num_semaine_epid"], errors="coerce")
                        _b["Annee_epid"] = pd.to_numeric(_b["Annee_epid"], errors="coerce")

                        year_candidates = _b.loc[_b["Num_semaine_epid"] == w_max, "Annee_epid"].dropna()
                        last_year = int(year_candidates.max()) if not year_candidates.empty else None

                        df_last_kpi = _b[(_b["Annee_epid"] == last_year) & (_b["Num_semaine_epid"] == w_max)] if last_year is not None else pd.DataFrame()
                        if last_year is not None and int(w_max) > 1:
                            df_prev_kpi = _b[(_b["Annee_epid"] == last_year) & (_b["Num_semaine_epid"] == (int(w_max) - 1))]
                        elif last_year is not None:
                            df_prev_kpi = _b[(_b["Annee_epid"] == (last_year - 1)) & (_b["Num_semaine_epid"].isin([52, 53]))]
                            if not df_prev_kpi.empty:
                                prev_week_num = int(df_prev_kpi["Num_semaine_epid"].max())
                                df_prev_kpi = df_prev_kpi[df_prev_kpi["Num_semaine_epid"] == prev_week_num]
                        else:
                            df_prev_kpi = pd.DataFrame()

                        if not df_last_kpi.empty:
                            cas_last = pd.to_numeric(df_last_kpi.get("Total_cas"), errors="coerce").sum(skipna=True)
                            dec_last = pd.to_numeric(df_last_kpi.get("Total_deces"), errors="coerce").sum(skipna=True)
                            cfr_last = (float(dec_last) / float(cas_last) * 100.0) if (pd.notna(cas_last) and cas_last > 0 and pd.notna(dec_last)) else np.nan

                            cas_prev = pd.to_numeric(df_prev_kpi.get("Total_cas"), errors="coerce").sum(skipna=True) if not df_prev_kpi.empty else np.nan
                            dec_prev = pd.to_numeric(df_prev_kpi.get("Total_deces"), errors="coerce").sum(skipna=True) if not df_prev_kpi.empty else np.nan
                            cfr_prev = (float(dec_prev) / float(cas_prev) * 100.0) if (pd.notna(cas_prev) and cas_prev > 0 and pd.notna(dec_prev)) else np.nan

                            last = {"TIME_LAB": f"W{int(w_max):02d}", "Cas": cas_last, "Deces": dec_last, "CFR_calc_%": cfr_last}
                            prev = {"Cas": cas_prev, "Deces": dec_prev, "CFR_calc_%": cfr_prev} if not df_prev_kpi.empty else None

                # Fallback: si on n'a rien trouvé, on garde l'ancien comportement
                if last is None and len(weekly_sorted) >= 1:
                    last = weekly_sorted.iloc[-1]
                    prev = weekly_sorted.iloc[-2] if len(weekly_sorted) >= 2 else None

                d_cas = None if ("disable_deltas" in locals() and disable_deltas) else (pct_change_metric_safe(last["Cas"], prev["Cas"]) if (last is not None and prev is not None) else None)
                d_dec = None if ("disable_deltas" in locals() and disable_deltas) else (pct_change_metric_safe(last["Deces"], prev["Deces"]) if (last is not None and prev is not None) else None)
                d_cfr = None if ("disable_deltas" in locals() and disable_deltas) else (pct_change_metric_safe(last["CFR_calc_%"], prev["CFR_calc_%"]) if (last is not None and prev is not None) else None)

                st.markdown("### Situation épidémiologique — dernière semaine disponible")

                # Préparer la série "tranches d’âge" (Cas_* / Deces_*) pour comparer avec les totaux
                age_case_cols = [c for c in ["Cas_tnn", "Cas_0_11mois", "Cas_12_59mois", "Cas_5_14ans", "Cas_15plus"] if c in df9.columns]
                age_death_cols = [c for c in ["Deces_tnn", "Deces_0_11mois", "Deces_12_59mois", "Deces_5_14ans", "Deces_15plus"] if c in df9.columns]

                weekly_age_sorted = None
                if age_case_cols and age_death_cols:
                    _tmp = df9.copy()
                    _tmp["Cas_age_sum"] = _tmp[age_case_cols].sum(axis=1, skipna=True)
                    _tmp["Deces_age_sum"] = _tmp[age_death_cols].sum(axis=1, skipna=True)

                    weekly_age = _tmp.groupby(["TIME_LAB", "TIME_KEY"], as_index=False).agg(
                        Cas_age=("Cas_age_sum", "sum"),
                        Deces_age=("Deces_age_sum", "sum"),
                    )
                    weekly_age["CFR_age_%"] = np.where(
                        weekly_age["Cas_age"] > 0,
                        (weekly_age["Deces_age"] / weekly_age["Cas_age"]) * 100,
                        np.nan
                    )
                    weekly_age_sorted = weekly_age.sort_values("TIME_KEY").reset_index(drop=True)

                # Ligne 1 — Somme tranches d’âge (Cas_* / Deces_*) — focus sur la semaine max
                df_last_week = pd.DataFrame()
                df_prev_week = pd.DataFrame()
                last_lab_focus = None

                if "df9_base" in locals() and not df9_base.empty:
                    # 1) Déterminer la "dernière semaine" = borne haute du filtre (w_max ou max_key)
                    if week_filter_mode == "YW" and "YW_KEY" in df9_base.columns:
                        _base = df9_base.copy()
                        _base["YW_KEY"] = pd.to_numeric(_base["YW_KEY"], errors="coerce")

                        last_key = max_key if max_key is not None else _base["YW_KEY"].dropna().max()

                        df_last_week = _base[_base["YW_KEY"] == last_key]  # focus semaine max (YW)
                        uniq_keys = (
                            _base["YW_KEY"].dropna().drop_duplicates().sort_values().tolist()
                        )
                        prev_key = uniq_keys[-2] if len(uniq_keys) >= 2 else None
                        df_prev_week = _base[_base["YW_KEY"] == prev_key] if prev_key is not None else pd.DataFrame()

                        if not df_last_week.empty:
                            last_lab_focus = (
                                df_last_week["TIME_LAB"].iloc[0]
                                if "TIME_LAB" in df_last_week.columns
                                else str(int(max_key) if pd.notna(max_key) else "NA")
                            )

                    elif week_filter_mode == "WNUM" and "Num_semaine_epid" in df9_base.columns:
                        _base = df9_base.copy()
                        _base["Num_semaine_epid"] = pd.to_numeric(_base["Num_semaine_epid"], errors="coerce")
                        if "Annee_epid" in _base.columns:
                            _base["Annee_epid"] = pd.to_numeric(_base["Annee_epid"], errors="coerce")

                            # choisir l'année la plus récente qui contient la semaine w_max
                            year_candidates = _base.loc[_base["Num_semaine_epid"] == w_max, "Annee_epid"].dropna()
                            last_year = int(year_candidates.max()) if not year_candidates.empty else None

                            if last_year is not None:
                                df_last_week = _base[(_base["Annee_epid"] == last_year) & (_base["Num_semaine_epid"] == w_max)]

                                # semaine précédente (dans la même année si possible)
                                if int(w_max) > 1:
                                    df_prev_week = _base[(_base["Annee_epid"] == last_year) & (_base["Num_semaine_epid"] == (int(w_max) - 1))]
                                else:
                                    # Si w_max == 1 : chercher semaine 52/53 de l'année précédente
                                    df_prev_week = _base[(_base["Annee_epid"] == (last_year - 1)) & (_base["Num_semaine_epid"].isin([52, 53]))]
                                    if not df_prev_week.empty:
                                        prev_week_num = int(df_prev_week["Num_semaine_epid"].max())
                                        df_prev_week = df_prev_week[df_prev_week["Num_semaine_epid"] == prev_week_num]

                                last_lab_focus = f"W{int(w_max):02d}"

                    # Fallback si on n'a pas réussi à isoler la semaine max
                    if df_last_week.empty:
                        df_last_week = df9.copy()
                        df_prev_week = pd.DataFrame()
                        last_lab_focus = df_last_week["TIME_LAB"].iloc[0] if ("TIME_LAB" in df_last_week.columns and not df_last_week.empty) else "NA"

                # Note opérationnelle : si plusieurs années sont incluses en mode WNUM,
                # les deltas vs semaine-1 sont désactivés (comparaison non interprétable).
                if week_filter_mode == "WNUM" and "Annee_epid" in df_last_week.columns:
                    _yrs = pd.to_numeric(df_last_week["Annee_epid"], errors="coerce").dropna().unique().tolist()
                    if len(_yrs) > 1:
                        st.info("Plusieurs années ont été détectées pour cette semaine (mode brut / WNUM) : les variations par rapport à la semaine précédente sont désactivées.")
                        disable_deltas = True

                # 2) Affichage métriques "tranches d'âge" pour la semaine max
                if not df_last_week.empty and age_case_cols and age_death_cols:
                    cas_age_last = df_last_week[age_case_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0, skipna=True).sum()
                    dec_age_last = df_last_week[age_death_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0, skipna=True).sum()
                    cfr_age_last = (float(dec_age_last) / float(cas_age_last) * 100.0) if (pd.notna(cas_age_last) and cas_age_last > 0 and pd.notna(dec_age_last)) else np.nan

                    cas_age_prev = df_prev_week[age_case_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0, skipna=True).sum() if (not df_prev_week.empty) else np.nan
                    dec_age_prev = df_prev_week[age_death_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0, skipna=True).sum() if (not df_prev_week.empty) else np.nan
                    cfr_age_prev = (float(dec_age_prev) / float(cas_age_prev) * 100.0) if (pd.notna(cas_age_prev) and cas_age_prev > 0 and pd.notna(dec_age_prev)) else np.nan

                    d_cas_a = None if ("disable_deltas" in locals() and disable_deltas) else pct_change_metric_safe(cas_age_last, cas_age_prev)
                    d_dec_a = None if ("disable_deltas" in locals() and disable_deltas) else pct_change_metric_safe(dec_age_last, dec_age_prev)
                    d_cfr_a = None if ("disable_deltas" in locals() and disable_deltas) else pct_change_metric_safe(cfr_age_last, cfr_age_prev)

                    st.caption("Ligne 1 : somme des tranches d’âge (Cas_* / Deces_*)")
                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("Semaine", str(last_lab_focus))
                    a2.metric("Cas (tranches)", f"{int(cas_age_last):,}" if pd.notna(cas_age_last) else "NA", delta=None if d_cas_a is None else f"{d_cas_a:.1f}% vs semaine-1")
                    a3.metric("Décès (tranches)", f"{int(dec_age_last):,}" if pd.notna(dec_age_last) else "NA", delta=None if d_dec_a is None else f"{d_dec_a:.1f}% vs semaine-1")
                    a4.metric("CFR (tranches)", f"{cfr_age_last:.2f}%" if pd.notna(cfr_age_last) else "NA", delta=None if d_cfr_a is None else f"{d_cfr_a:.1f}% vs semaine-1")
                else:
                    st.caption("Ligne 1 : somme des tranches d’âge (Cas_* / Deces_*) — indisponible (colonnes manquantes ou aucune donnée)")

                
                # -----------------------------------------------------------------
                # Ligne 2 — Totaux (TOTALCAS / TOTALDECES) — focus sur la semaine max
                # Objectif: comparer directement avec la Ligne 1 (sommes des tranches d’âge)
                # -----------------------------------------------------------------
                tot_cas_lastwk = pd.to_numeric(df_last_week.get("Total_cas"), errors="coerce").sum(skipna=True) if (("Total_cas" in df9.columns) and (not df_last_week.empty)) else np.nan
                tot_dec_lastwk = pd.to_numeric(df_last_week.get("Total_deces"), errors="coerce").sum(skipna=True) if (("Total_deces" in df9.columns) and (not df_last_week.empty)) else np.nan
                cfr_tot_lastwk = (float(tot_dec_lastwk) / float(tot_cas_lastwk) * 100.0) if (pd.notna(tot_cas_lastwk) and tot_cas_lastwk > 0 and pd.notna(tot_dec_lastwk)) else np.nan

                tot_cas_prevwk = pd.to_numeric(df_prev_week.get("Total_cas"), errors="coerce").sum(skipna=True) if (("Total_cas" in df9.columns) and (not df_prev_week.empty)) else np.nan
                tot_dec_prevwk = pd.to_numeric(df_prev_week.get("Total_deces"), errors="coerce").sum(skipna=True) if (("Total_deces" in df9.columns) and (not df_prev_week.empty)) else np.nan
                cfr_tot_prevwk = (float(tot_dec_prevwk) / float(tot_cas_prevwk) * 100.0) if (pd.notna(tot_cas_prevwk) and tot_cas_prevwk > 0 and pd.notna(tot_dec_prevwk)) else np.nan

                d_cas_t = None if ("disable_deltas" in locals() and disable_deltas) else pct_change_metric_safe(tot_cas_lastwk, tot_cas_prevwk)
                d_dec_t = None if ("disable_deltas" in locals() and disable_deltas) else pct_change_metric_safe(tot_dec_lastwk, tot_dec_prevwk)
                d_cfr_t = None if ("disable_deltas" in locals() and disable_deltas) else pct_change_metric_safe(cfr_tot_lastwk, cfr_tot_prevwk)

                st.caption("Ligne 2 : totaux notifiés (TOTALCAS / TOTALDECES)")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Semaine", str(last_lab_focus) if last_lab_focus is not None else "NA")
                k2.metric("Cas (totaux)", f"{int(tot_cas_lastwk):,}" if pd.notna(tot_cas_lastwk) else "NA", delta=None if d_cas_t is None else f"{d_cas_t:.1f}% vs semaine-1")
                k3.metric("Décès (totaux)", f"{int(tot_dec_lastwk):,}" if pd.notna(tot_dec_lastwk) else "NA", delta=None if d_dec_t is None else f"{d_dec_t:.1f}% vs semaine-1")
                k4.metric("CFR (totaux)", f"{cfr_tot_lastwk:.2f}%" if pd.notna(cfr_tot_lastwk) else "NA", delta=None if d_cfr_t is None else f"{d_cfr_t:.1f}% vs semaine-1")

                # -----------------------------------------------------------------
                # Écarts Totaux vs Tranches (semaine max)
                # -----------------------------------------------------------------
                diff_cas = (tot_cas_lastwk - cas_age_last) if ("cas_age_last" in locals() and pd.notna(tot_cas_lastwk) and pd.notna(cas_age_last)) else np.nan
                diff_dec = (tot_dec_lastwk - dec_age_last) if ("dec_age_last" in locals() and pd.notna(tot_dec_lastwk) and pd.notna(dec_age_last)) else np.nan

                if pd.notna(diff_cas) and pd.notna(diff_dec):
                    if (diff_cas == 0) and (diff_dec == 0):
                        st.success("Aucun écart détecté : TOTALCAS/TOTALDECES correspond à la somme des tranches d’âge sur la semaine maximale.")
                    else:
                        pct_cas = (diff_cas / cas_age_last * 100.0) if ("cas_age_last" in locals() and pd.notna(cas_age_last) and cas_age_last != 0) else np.nan
                        pct_dec = (diff_dec / dec_age_last * 100.0) if ("dec_age_last" in locals() and pd.notna(dec_age_last) and dec_age_last != 0) else np.nan
                        st.error(
                            "❌ Écart détecté (Totaux − Tranches) – semaine max : "
                            f"Cas={diff_cas:+,} ({pct_cas:.1f}%) | Décès={diff_dec:+,} ({pct_dec:.1f}%)"
                        )
                else:
                    st.info("Écart non calculable : variables manquantes ou données insuffisantes.")

                # Note: cette section est volontairement centrée sur la semaine max,
                # même si l'utilisateur change semaine min.
                st.divider()
                with st.expander("### Qualité des dates (date vs semaine)", expanded=False):
                    if "QC_Date_vs_Semaine" in df9.columns:
                        st.write(df9["QC_Date_vs_Semaine"].value_counts(dropna=False))
                    else:
                        st.info("Le contrôle qualité temporel est indisponible : dates sources absentes.")

                # -----------------------------------------------------------------
                # 10) Signaux – Top en hausse (dernière semaine vs précédente)
                # -----------------------------------------------------------------
                with st.expander("### Signaux – Top en hausse (dernière semaine vs semaine précédente)", expanded=False):

                    if (COL_PROV_ID in df9.columns) and (len(weekly_sorted) >= 2):
                        last_t = weekly_sorted.iloc[-1]["TIME_LAB"]
                        prev_t = weekly_sorted.iloc[-2]["TIME_LAB"]

                        df_last = df9[df9["TIME_LAB"] == last_t]
                        df_prev = df9[df9["TIME_LAB"] == prev_t]

                        prov_last = df_last.groupby(COL_PROV_ID, as_index=False).agg(Cas=("Total_cas", "sum"))
                        prov_prev = df_prev.groupby(COL_PROV_ID, as_index=False).agg(Cas_prev=("Total_cas", "sum"))

                        prov_delta = prov_last.merge(prov_prev, on=COL_PROV_ID, how="outer").fillna(0)
                        prov_delta["Delta_cas"] = prov_delta["Cas"] - prov_delta["Cas_prev"]
                        prov_delta["Delta_%"] = np.where(
                            prov_delta["Cas_prev"] > 0,
                            (prov_delta["Delta_cas"] / prov_delta["Cas_prev"]) * 100,
                            np.nan
                        )

                        min_cases = st.slider(
                            "Seuil cas (dernière semaine) pour afficher",
                            0, 1000, 5, step=5, key="tab9_min_cases_up"
                        )
                        prov_delta = prov_delta[prov_delta["Cas"] >= min_cases].sort_values("Delta_cas", ascending=False)

                        with st.expander("📈 Top provinces en hausse (dérouler)", expanded=False):
                            n_up = st.slider("Nombre d’unités à afficher", 5, 50, 15, step=5, key="tab9_n_up_prov")
                            st.dataframe(prov_delta.head(n_up), width="stretch", height=420, hide_index=True)
                    else:
                        st.info("Le classement des provinces en hausse est indisponible : variable Province absente ou historique insuffisant.")

                    # -----------------------------------------------------------------
                    # 11) Top provinces / ZS sur la période
                    # -----------------------------------------------------------------
                    c3, c4 = st.columns(2)

                    with c3:
                        if COL_PROV_ID in df9.columns and "Total_cas" in df9.columns and "Total_deces" in df9.columns:
                            top_prov = df9.groupby(COL_PROV_ID, as_index=False).agg(
                                Cas=("Total_cas", "sum"),
                                Deces=("Total_deces", "sum")
                            )
                            top_prov["CFR_%"] = np.where(top_prov["Cas"] > 0, (top_prov["Deces"] / top_prov["Cas"]) * 100, np.nan)
                            top_prov = top_prov.sort_values("Cas", ascending=False)

                            with st.expander("🏥 Top provinces (dérouler)", expanded=False):
                                n_prov = st.slider("Nombre de provinces à afficher", 10, 200, 20, step=10, key="tab9_n_top_prov")
                                st.dataframe(top_prov.head(n_prov), width="stretch", height=420, hide_index=True)
                        else:
                            top_prov = None
                            st.info("Le classement des provinces est indisponible : variables requises manquantes.")

                    with c4:
                        if (COL_PROV_ID in df9.columns) and (COL_ZS_ID in df9.columns) and ("Total_cas" in df9.columns) and ("Total_deces" in df9.columns):
                            top_zs = df9.groupby([COL_PROV_ID, COL_ZS_ID], as_index=False).agg(
                                Cas=("Total_cas", "sum"),
                                Deces=("Total_deces", "sum")
                            )
                            top_zs["CFR_%"] = np.where(top_zs["Cas"] > 0, (top_zs["Deces"] / top_zs["Cas"]) * 100, np.nan)
                            top_zs = top_zs.sort_values("Cas", ascending=False)

                            with st.expander("🗺️ Top zones de santé (dérouler)", expanded=False):
                                n_zs = st.slider("Nombre de ZS à afficher", 10, 300, 20, step=10, key="tab9_n_top_zs")
                                st.dataframe(top_zs.head(n_zs), width="stretch", height=420, hide_index=True)
                        else:
                            top_zs = None
                            st.info("Le classement des zones de santé est indisponible : variables requises manquantes.")

            # -----------------------------------------------------------------
            # 12) Contrôles cohérence totaux vs tranches d’âge
            # -----------------------------------------------------------------
            with st.expander("### Contrôle cohérence des totaux (tranches d’âge vs Total)", expanded=False):
                show_qc_tables = st.checkbox(
                    "Afficher les tableaux détaillés QC (peut être lourd)",
                    value=False,
                    key="tab9_show_qc_tables"
                )


                age_case_cols = [c for c in ["Cas_tnn", "Cas_0_11mois", "Cas_12_59mois", "Cas_5_14ans", "Cas_15plus"] if c in df9.columns]
                age_death_cols = [c for c in ["Deces_tnn", "Deces_0_11mois", "Deces_12_59mois", "Deces_5_14ans", "Deces_15plus"] if c in df9.columns]

                qc = df9.copy()

                if age_case_cols and "Total_cas" in qc.columns:
                    qc["sum_cas_age"] = qc[age_case_cols].sum(axis=1, skipna=True)
                    qc["diff_cas"] = qc["Total_cas"] - qc["sum_cas_age"]

                if age_death_cols and "Total_deces" in qc.columns:
                    qc["sum_deces_age"] = qc[age_death_cols].sum(axis=1, skipna=True)
                    qc["diff_deces"] = qc["Total_deces"] - qc["sum_deces_age"]

                qc_view = qc.copy()
                qc_view["QC_Cas"] = np.where(qc_view.get("diff_cas", 0).fillna(0) == 0, "✅ OK", "❌ KO") if "diff_cas" in qc_view.columns else "NA"
                qc_view["QC_Deces"] = np.where(qc_view.get("diff_deces", 0).fillna(0) == 0, "✅ OK", "❌ KO") if "diff_deces" in qc_view.columns else "NA"

                if ("diff_cas" in qc_view.columns) and ("diff_deces" in qc_view.columns):
                    qc_view["QC_Global"] = np.where(
                        (qc_view["diff_cas"].fillna(0) == 0) & (qc_view["diff_deces"].fillna(0) == 0),
                        "✅ OK", "❌ KO"
                    )
                elif "diff_cas" in qc_view.columns:
                    qc_view["QC_Global"] = np.where(qc_view["diff_cas"].fillna(0) == 0, "✅ OK", "❌ KO")
                elif "diff_deces" in qc_view.columns:
                    qc_view["QC_Global"] = np.where(qc_view["diff_deces"].fillna(0) == 0, "✅ OK", "❌ KO")
                else:
                    qc_view["QC_Global"] = "NA"

                # Colonnes QC à afficher
                cols_show = [c for c in [
                    "TIME_LAB", "TIME_KEY", "Date_debut_semaine_iso",
                    COL_MAL, COL_PROV_ID, COL_ZS_ID,
                    "Total_cas", "sum_cas_age", "diff_cas",
                    "Total_deces", "sum_deces_age", "diff_deces"
                ] if c in qc_view.columns]

                def style_qc(row):
                    """Style cellule: surligner seulement les écarts et QC_Global."""
                    styles = [""] * len(row)
                    cols = list(row.index)

                    def set_cell(col, bg=None, fg=None, weight=None):
                        if col in cols:
                            i = cols.index(col)
                            css = []
                            if bg is not None:
                                css.append(f"background-color: {bg}")
                            if fg is not None:
                                css.append(f"color: {fg}")
                            if weight is not None:
                                css.append(f"font-weight: {weight}")
                            styles[i] = "; ".join(css)

                    if row.get("diff_cas", 0) != 0:
                        set_cell("diff_cas", bg="#fff3cd", fg="#111", weight="700")
                    if row.get("diff_deces", 0) != 0:
                        set_cell("diff_deces", bg="#ffe5e5", fg="#111", weight="700")

                    if row.get("QC_Global") == "❌ KO":
                        set_cell("QC_Global", bg="#f2f2f2", fg="#111", weight="700")
                    else:
                        set_cell("QC_Global", fg="#111", weight="700")

                    if "QC_Cas" in cols:
                        set_cell("QC_Cas", fg="#111", weight="700")
                    if "QC_Deces" in cols:
                        set_cell("QC_Deces", fg="#111", weight="700")

                    return styles

                # Filtres QC
                st.markdown("#### Filtres de contrôle qualité")
                f1, f2, f3, f4 = st.columns(4)

                with f1:
                    qc_global_sel = st.selectbox("Contrôle qualité global", options=["Tous", "✅ OK", "❌ KO"], index=0, key="tab9_qc_global_sel")
                with f2:
                    qc_cas_sel = st.selectbox("Contrôle qualité des cas", options=["Tous", "✅ OK", "❌ KO"], index=0, key="tab9_qc_cas_sel")
                with f3:
                    qc_deces_sel = st.selectbox("Contrôle qualité des décès", options=["Tous", "✅ OK", "❌ KO"], index=0, key="tab9_qc_deces_sel")
                with f4:
                    
                    abs_diff_min = st.number_input(
                        "|diff| minimum",
                        min_value=0,
                        value=0,
                        step=1,
                        help="Filtre sur l'écart absolu (cas ou décès). Mets 1 pour exclure les diff = 0.",
                        key="tab9_qc_abs_diff_min"
                    )

                show_all = st.checkbox(
                    "Afficher toutes les lignes (sinon seulement incohérences)",
                    value=False,
                    key="tab9_qc_show_all"
                )

                # Base: toutes lignes vs seulement incohérences
                if show_all:
                    base_tbl = qc_view.copy()
                else:
                    base_tbl = qc_view.copy()
                    if "diff_cas" in base_tbl.columns:
                        base_tbl = base_tbl[base_tbl["diff_cas"].fillna(0) != 0]
                    if "diff_deces" in base_tbl.columns:
                        base_tbl = base_tbl[base_tbl["diff_deces"].fillna(0) != 0]

                # Appliquer filtres
                table_to_show = base_tbl.copy()

                if qc_global_sel != "Tous" and "QC_Global" in table_to_show.columns:
                    table_to_show = table_to_show[table_to_show["QC_Global"] == qc_global_sel]

                if qc_cas_sel != "Tous" and "QC_Cas" in table_to_show.columns:
                    table_to_show = table_to_show[table_to_show["QC_Cas"] == qc_cas_sel]

                if qc_deces_sel != "Tous" and "QC_Deces" in table_to_show.columns:
                    table_to_show = table_to_show[table_to_show["QC_Deces"] == qc_deces_sel]

                # Seuil sur diff
                if abs_diff_min > 0:
                    cond = False
                    if "diff_cas" in table_to_show.columns:
                        cond = cond | (table_to_show["diff_cas"].fillna(0).abs() >= abs_diff_min)
                    if "diff_deces" in table_to_show.columns:
                        cond = cond | (table_to_show["diff_deces"].fillna(0).abs() >= abs_diff_min)
                    table_to_show = table_to_show[cond]

                st.caption(f"📌 Lignes après application des filtres de contrôle qualité : {len(table_to_show):,}")

                # Colonnes QC à afficher
                qc_cols = ["QC_Global", "QC_Cas", "QC_Deces"]
                qc_cols = [c for c in qc_cols if c in table_to_show.columns]
                final_cols = qc_cols + cols_show

                if show_qc_tables:
                    with st.expander("🧾 Tableau QC (OK/KO) – cas & décès (dérouler)", expanded=False):
                        st.dataframe(
                        table_to_show[final_cols].style.apply(style_qc, axis=1),
                        width="stretch",
                        height=520,
                        hide_index=True
                    )
                
            # ---------------------------------------------------------------------
            # 14) IDSR – Spécifications des sorties
            # ---------------------------------------------------------------------
            # 14.1) Histogramme des cas + courbe de létalité (CFR%)
            with st.expander("📈 Histogramme des cas avec courbe de létalité (par semaine)", expanded=True):
                if 'weekly_sorted' in locals() and isinstance(weekly_sorted, pd.DataFrame) and not weekly_sorted.empty:
                    _wk = weekly_sorted.copy()
                    # Libellé unique Année-Semaine (évite doublons W01/W02 quand plusieurs années)
                    if "YW" in _wk.columns:
                        _wk["_X_LAB"] = _wk["YW"].astype(str)
                    elif "TIME_KEY" in _wk.columns:
                        _wk["_X_LAB"] = _wk["TIME_KEY"].astype(str)
                    else:
                        _wk["_X_LAB"] = _wk.get("TIME_LAB", pd.Series(dtype="object")).astype(str)
                    # fmt_yw_label est centralisée dans dashboard_app.core
                    _wk["_X_LAB"] = _wk["_X_LAB"].map(fmt_yw_label)

                    # Sécurité sur colonnes
                    if ("_X_LAB" in _wk.columns) and ("Cas" in _wk.columns):
                        _wk["CFR_calc_%"] = pd.to_numeric(_wk.get("CFR_calc_%"), errors="coerce").astype(float)
                        # Plotly n'accepte pas pd.NA (NAType) -> forcer np.nan
                        _wk = _wk.replace({pd.NA: np.nan})

                        # Texte CFR (évite "NA %" et évite pd.NA)
                        _wk["_cfr_text"] = _wk["CFR_calc_%"].map(lambda x: "" if pd.isna(x) else f"{x:.2f} %")

                        fig_cas_cfr = go.Figure()
                        fig_cas_cfr.add_trace(go.Bar(
                            x=_wk["_X_LAB"],
                            y=pd.to_numeric(_wk["Cas"], errors="coerce").fillna(0).astype(float),
                            name="Cas",
                            yaxis="y1",
                        ))
                        fig_cas_cfr.add_trace(go.Scatter(
                            x=_wk["_X_LAB"],
                            y=_wk["CFR_calc_%"].astype(float),
                            name="Létalité (CFR%)",
                            mode="lines+markers+text",
                            yaxis="y2",
                            text=_wk["_cfr_text"],
                            textposition="top center",
                        ))
                        fig_cas_cfr.update_layout(
                            template="plotly_white",
                            xaxis_title="Semaine épidémiologique",
                            yaxis=dict(title="Nombre de cas"),
                            yaxis2=dict(title="Létalité (%)", overlaying="y", side="right", rangemode="tozero"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                            bargap=0.04,
                            bargroupgap=0.02,
                            margin=dict(t=70, b=60, l=60, r=60),
                            height=420,
                        )
                        fig_cas_cfr = apply_plotly_value_annotations(fig_cas_cfr, annot_vals)
                        st.plotly_chart(fig_cas_cfr, width="stretch", key="idsr_hist_cas_cfr")
                    else:
                        st.info("Variables insuffisantes pour tracer l’évolution hebdomadaire (TIME_LAB/Cas).")
                else:
                    st.info("Aucune donnée hebdomadaire agrégée n’est disponible après filtrage.")

            # 14.2) Camembert par tranche d'âge + tableau associé
            with st.expander("📈 Camembert – répartition des cas par tranche d’âge", expanded=True):
                if not df9.empty:
                    # Colonnes attendues (agrégé IDSR) : Cas_* et Deces_* par tranche
                    age_cases_map = {
                        "Cas_tnn": "<1 mois",
                        "Cas_0_11mois": "0–11 mois",
                        "Cas_12_59mois": "12–59 mois",
                        "Cas_5_14ans": "5–14 ans",
                        "Cas_15plus": "≥15 ans",
                    }
                    age_deaths_map = {
                        "Deces_tnn": "<1 mois",
                        "Deces_0_11mois": "0–11 mois",
                        "Deces_12_59mois": "12–59 mois",
                        "Deces_5_14ans": "5–14 ans",
                        "Deces_15plus": "≥15 ans",
                    }
                    rows_age = []
                    for c_col, label in age_cases_map.items():
                        if c_col in df9.columns:
                            cas = pd.to_numeric(df9[c_col], errors="coerce").sum(skipna=True)
                            d_col = [k for k, v in age_deaths_map.items() if v == label and k in df9.columns]
                            dec = pd.to_numeric(df9[d_col[0]], errors="coerce").sum(skipna=True) if d_col else np.nan
                            rows_age.append({"Tranche d'âge": label, "Cas": cas, "Décès": dec})
                    df_age = pd.DataFrame(rows_age)
                    if not df_age.empty:
                        df_age["Cas"] = pd.to_numeric(df_age["Cas"], errors="coerce").fillna(0).astype(int)
                        df_age["Décès"] = pd.to_numeric(df_age["Décès"], errors="coerce")
                        df_age["Décès"] = df_age["Décès"].fillna(0).astype(int)
                        total_cas_age = int(df_age["Cas"].sum())
                        df_age["Létalité (%)"] = np.where(df_age["Cas"] > 0, (df_age["Décès"] / df_age["Cas"]) * 100.0, np.nan)
                        df_age["Proportion des cas (%)"] = np.where(total_cas_age > 0, (df_age["Cas"] / total_cas_age) * 100.0, np.nan)
                
                        # Ordre logique
                        ordre_age = ["<1 mois", "0–11 mois", "12–59 mois", "5–14 ans", "≥15 ans"]
                        df_age["Tranche d'âge"] = pd.Categorical(df_age["Tranche d'âge"], categories=ordre_age, ordered=True)
                        df_age = df_age.sort_values("Tranche d'âge")
                
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            fig_pie_age = go.Figure(data=[go.Pie(
                                labels=df_age["Tranche d'âge"].astype(str),
                                values=df_age["Cas"],
                                hole=0.45,
                                textinfo="percent+label",
                                hovertemplate="%{label}<br>Cas=%{value}<br>%{percent}<extra></extra>",
                            )])
                            fig_pie_age.update_layout(template="plotly_white", height=420, margin=dict(t=30, b=10, l=10, r=10))
                            st.plotly_chart(fig_pie_age, width="stretch", key="idsr_pie_age")
                
                        with c2:
                            st.dataframe(
                                df_age[["Tranche d'âge", "Cas", "Décès", "Létalité (%)", "Proportion des cas (%)"]]
                                .assign(**{
                                    "Létalité (%)": df_age["Létalité (%)"].round(2),
                                    "Proportion des cas (%)": df_age["Proportion des cas (%)"].round(2),
                                }),
                                width="stretch",
                                height=420,
                                hide_index=True
                            )
                    else:
                        st.info("Aucune variable 'Cas_*' par tranche d’âge n’a été trouvée dans les données IDSR.")
                else:
                    st.info("Aucune donnée n’est disponible après filtrage pour produire la répartition par âge.")


            # 14.2.b) Analyses descriptives IDSR inspirées des listes linéaires
            with st.expander("👥 Profil descriptif IDSR", expanded=False):
                st.caption(
                    "Ces analyses reprennent la logique descriptive des listes linéaires, mais en l’adaptant au format agrégé IDSR. "
                    "Elles portent surtout sur la maladie, l’âge, le lieu et la distribution hebdomadaire."
                )

                # -------------------------------------------------------------
                # A) Profil par maladie
                # -------------------------------------------------------------
                st.markdown("#### A. Répartition des cas par maladie")
                if (COL_MAL in df9.columns) and ("Total_cas" in df9.columns):
                    df_mal_profile = (
                        df9.groupby(COL_MAL, as_index=False)
                        .agg(Cas=("Total_cas", "sum"), Deces=("Total_deces", "sum") if "Total_deces" in df9.columns else ("Total_cas", "size"))
                    )
                    if "Total_deces" not in df9.columns:
                        df_mal_profile["Deces"] = 0
                    df_mal_profile["CFR_%"] = np.where(
                        df_mal_profile["Cas"] > 0,
                        (df_mal_profile["Deces"] / df_mal_profile["Cas"]) * 100.0,
                        np.nan,
                    )
                    df_mal_profile = df_mal_profile.sort_values("Cas", ascending=False)

                    c_m1, c_m2 = st.columns([1.2, 1])
                    with c_m1:
                        fig_mal = px.bar(
                            df_mal_profile,
                            x=COL_MAL,
                            y="Cas",
                            title="Cas cumulés par maladie",
                            text="Cas",
                        )
                        fig_mal.update_layout(template="plotly_white", xaxis_tickangle=-35, height=420)
                        fig_mal = apply_plotly_value_annotations(fig_mal, annot_vals)
                        st.plotly_chart(fig_mal, width="stretch", key="idsr_profile_maladie")
                    with c_m2:
                        st.dataframe(
                            df_mal_profile.assign(**{"CFR_%": df_mal_profile["CFR_%"].round(2)}),
                            width="stretch",
                            height=420,
                            hide_index=True,
                        )
                else:
                    st.info("Le profil par maladie est indisponible : colonnes Maladie ou Total_cas absentes.")

                st.divider()

                # -------------------------------------------------------------
                # B) Structure d'âge par maladie (équivalent profil population)
                # -------------------------------------------------------------
                st.markdown("#### B. Structure des cas par tranche d’âge selon la maladie")
                age_case_cols = [c for c in ["Cas_tnn", "Cas_0_11mois", "Cas_12_59mois", "Cas_5_14ans", "Cas_15plus"] if c in df9.columns]
                age_label_map = {
                    "Cas_tnn": "<1 mois",
                    "Cas_0_11mois": "0–11 mois",
                    "Cas_12_59mois": "12–59 mois",
                    "Cas_5_14ans": "5–14 ans",
                    "Cas_15plus": "≥15 ans",
                }
                if (COL_MAL in df9.columns) and age_case_cols:
                    age_mal = (
                        df9.groupby(COL_MAL, as_index=False)[age_case_cols]
                        .sum(min_count=1)
                    )
                    age_mal_long = age_mal.melt(
                        id_vars=[COL_MAL],
                        value_vars=age_case_cols,
                        var_name="Tranche_source",
                        value_name="Cas",
                    )
                    age_mal_long["Tranche_age"] = age_mal_long["Tranche_source"].map(age_label_map)
                    age_mal_long["Cas"] = pd.to_numeric(age_mal_long["Cas"], errors="coerce").fillna(0)
                    age_mal_long = age_mal_long[age_mal_long["Cas"] > 0]

                    if not age_mal_long.empty:
                        ordre_age = ["<1 mois", "0–11 mois", "12–59 mois", "5–14 ans", "≥15 ans"]
                        age_mal_long["Tranche_age"] = pd.Categorical(age_mal_long["Tranche_age"], categories=ordre_age, ordered=True)

                        c_a1, c_a2 = st.columns([1.3, 1])
                        with c_a1:
                            fig_age_mal = px.bar(
                                age_mal_long.sort_values([COL_MAL, "Tranche_age"]),
                                x=COL_MAL,
                                y="Cas",
                                color="Tranche_age",
                                barmode="stack",
                                title="Cas par maladie et tranche d’âge",
                            )
                            fig_age_mal.update_layout(template="plotly_white", xaxis_tickangle=-35, height=460)
                            st.plotly_chart(fig_age_mal, width="stretch", key="idsr_profile_age_maladie")
                        with c_a2:
                            age_mal_tab = (
                                age_mal_long.pivot_table(
                                    index=COL_MAL,
                                    columns="Tranche_age",
                                    values="Cas",
                                    aggfunc="sum",
                                    fill_value=0,
                                    observed=False,
                                )
                                .reset_index()
                            )
                            st.dataframe(age_mal_tab, width="stretch", height=460, hide_index=True)
                    else:
                        st.info("Les colonnes d’âge existent mais ne contiennent pas de volume exploitable après filtrage.")
                else:
                    st.info("La structure par tranche d’âge selon la maladie est indisponible : colonnes Cas_* ou Maladie absentes.")

                st.divider()

                # -------------------------------------------------------------
                # C) Pyramide d’âge IDSR (cas vs décès)
                # -------------------------------------------------------------
                st.markdown("#### C. Pyramide d’âge IDSR")
                st.caption(
                    "Dans l’IDSR agrégé, la pyramide classique par sexe n’est généralement pas disponible. "
                    "La représentation ci-dessous compare donc les cas (à gauche) et les décès (à droite) "
                    "par tranche d’âge, à partir des colonnes agrégées du fichier IDSR."
                )

                age_pairs_pyr = [
                    ("Cas_tnn", "Deces_tnn", "<1 mois"),
                    ("Cas_0_11mois", "Deces_0_11mois", "0–11 mois"),
                    ("Cas_12_59mois", "Deces_12_59mois", "12–59 mois"),
                    ("Cas_5_14ans", "Deces_5_14ans", "5–14 ans"),
                    ("Cas_15plus", "Deces_15plus", "≥15 ans"),
                ]

                available_pairs_pyr = [
                    (c_col, d_col, label)
                    for c_col, d_col, label in age_pairs_pyr
                    if (c_col in df9.columns) or (d_col in df9.columns)
                ]

                if available_pairs_pyr:
                    rows_pyr = []
                    for c_col, d_col, label in available_pairs_pyr:
                        case_val = pd.to_numeric(df9[c_col], errors="coerce").fillna(0).sum() if c_col in df9.columns else 0
                        death_val = pd.to_numeric(df9[d_col], errors="coerce").fillna(0).sum() if d_col in df9.columns else 0
                        rows_pyr.append({
                            "Tranche_age": label,
                            "Cas": float(case_val),
                            "Décès": float(death_val),
                        })

                    ordre_age_pyr = ["<1 mois", "0–11 mois", "12–59 mois", "5–14 ans", "≥15 ans"]
                    pyr_display = pd.DataFrame(rows_pyr)

                    if not pyr_display.empty:
                        pyr_display["Tranche_age"] = pd.Categorical(
                            pyr_display["Tranche_age"],
                            categories=ordre_age_pyr,
                            ordered=True,
                        )
                        pyr_display = (
                            pyr_display.sort_values("Tranche_age")
                            .drop_duplicates(subset=["Tranche_age"], keep="first")
                            .reset_index(drop=True)
                        )

                    if (not pyr_display.empty) and (pyr_display[["Cas", "Décès"]].sum(axis=1) > 0).any():
                        total_cases_age = float(pyr_display["Cas"].sum())
                        total_deaths_age = float(pyr_display["Décès"].sum())
                        total_cases_global = float(pd.to_numeric(df9.get("Total_cas"), errors="coerce").fillna(0).sum()) if "Total_cas" in df9.columns else np.nan
                        total_deaths_global = float(pd.to_numeric(df9.get("Total_deces"), errors="coerce").fillna(0).sum()) if "Total_deces" in df9.columns else np.nan

                        plot_df = pyr_display.copy()
                        plot_df["Cas_plot"] = -plot_df["Cas"]
                        plot_df["Décès_plot"] = plot_df["Décès"]

                        fig_pyr_idsr = go.Figure()
                        fig_pyr_idsr.add_trace(go.Bar(
                            y=plot_df["Tranche_age"],
                            x=plot_df["Cas_plot"],
                            name="Cas",
                            orientation="h",
                            marker=dict(color="#E70B0B"),
                            text=plot_df["Cas"].map(lambda v: f"{int(v):,}".replace(",", " ")),
                            textposition="inside",
                            insidetextanchor="middle",
                            cliponaxis=False,
                            hovertemplate="Tranche d'âge: %{y}<br>Cas: %{text}<extra></extra>",
                        ))
                        fig_pyr_idsr.add_trace(go.Bar(
                            y=plot_df["Tranche_age"],
                            x=plot_df["Décès_plot"],
                            name="Décès",
                            orientation="h",
                            marker=dict(color="#4682B4"),
                            text=plot_df["Décès"].map(lambda v: f"{int(v):,}".replace(",", " ")),
                            textposition="inside",
                            insidetextanchor="middle",
                            cliponaxis=False,
                            hovertemplate="Tranche d'âge: %{y}<br>Décès: %{text}<extra></extra>",
                        ))

                        max_cases = float(plot_df["Cas"].max()) if not plot_df["Cas"].empty else 0.0
                        max_deaths = float(plot_df["Décès"].max()) if not plot_df["Décès"].empty else 0.0
                        x_abs_max = max(max_cases, max_deaths)
                        if x_abs_max <= 0:
                            x_abs_max = 1.0

                        fig_pyr_idsr.update_layout(
                            barmode="relative",
                            template="plotly_white",
                            title="Pyramide d’âge IDSR (Cas vs Décès)",
                            width=1100,
                            height=480,
                            margin=dict(t=70, b=50, l=80, r=40),
                            legend=dict(orientation="h", y=1.08, x=0),
                            xaxis=dict(
                                title="Nombre",
                                range=[-x_abs_max * 1.15, x_abs_max * 1.15],
                                tickformat=",",
                                tickvals=[-x_abs_max, -x_abs_max / 2, 0, x_abs_max / 2, x_abs_max],
                                ticktext=[
                                    f"{int(x_abs_max):,}".replace(",", " "),
                                    f"{int(x_abs_max / 2):,}".replace(",", " "),
                                    "0",
                                    f"{int(x_abs_max / 2):,}".replace(",", " "),
                                    f"{int(x_abs_max):,}".replace(",", " "),
                                ],
                                zeroline=True,
                                zerolinewidth=2,
                                zerolinecolor="LightGrey",
                            ),
                            yaxis=dict(
                                title="Tranche d’âge",
                                categoryorder="array",
                                categoryarray=ordre_age_pyr[::-1],
                            ),
                        )

                        c_p1, c_p2 = st.columns([1.15, 1])
                        with c_p1:
                            st.plotly_chart(fig_pyr_idsr, width="stretch", key="idsr_age_pyramid")

                        with c_p2:
                            pyr_display["Part_cas_%"] = np.where(
                                total_cases_age > 0,
                                (pyr_display["Cas"] / total_cases_age) * 100.0,
                                np.nan,
                            )
                            pyr_display["Part_décès_%"] = np.where(
                                total_deaths_age > 0,
                                (pyr_display["Décès"] / total_deaths_age) * 100.0,
                                np.nan,
                            )
                            st.dataframe(
                                pyr_display.assign(**{
                                    "Cas": pyr_display["Cas"].astype(int),
                                    "Décès": pyr_display["Décès"].astype(int),
                                    "Part_cas_%": pyr_display["Part_cas_%"].round(1),
                                    "Part_décès_%": pyr_display["Part_décès_%"].round(1),
                                }),
                                width="stretch",
                                height=380,
                            )

                            if np.isfinite(total_cases_global) and int(total_cases_age) != int(total_cases_global):
                                st.warning(
                                    f"Somme des cas par âge = {int(total_cases_age):,}".replace(",", " ")
                                    + f", alors que Total_cas = {int(total_cases_global):,}".replace(",", " ")
                                    + ". Cela indique un écart dans les données agrégées source."
                                )
                            if np.isfinite(total_deaths_global) and int(total_deaths_age) != int(total_deaths_global):
                                st.warning(
                                    f"Somme des décès par âge = {int(total_deaths_age):,}".replace(",", " ")
                                    + f", alors que Total_deces = {int(total_deaths_global):,}".replace(",", " ")
                                    + ". Cela indique un écart dans les données agrégées source."
                                )
                    else:
                        st.info("Les colonnes d’âge existent mais ne contiennent pas de volume exploitable après filtrage.")
                else:
                    st.info("La pyramide d’âge IDSR est indisponible : colonnes Cas_* / Deces_* absentes.")
                st.divider()

                # -------------------------------------------------------------
                # C) Répartition spatiale par maladie (équivalent personne/lieu)
                # -------------------------------------------------------------
                st.markdown("#### C. Répartition géographique des cas par maladie")
                if (COL_MAL in df9.columns) and (COL_PROV_ID in df9.columns) and ("Total_cas" in df9.columns):
                    geo_mal = (
                        df9.groupby([COL_MAL, COL_PROV_ID], as_index=False)
                        .agg(Cas=("Total_cas", "sum"), Deces=("Total_deces", "sum") if "Total_deces" in df9.columns else ("Total_cas", "size"))
                    )
                    if "Total_deces" not in df9.columns:
                        geo_mal["Deces"] = 0
                    geo_mal["CFR_%"] = np.where(geo_mal["Cas"] > 0, (geo_mal["Deces"] / geo_mal["Cas"]) * 100.0, np.nan)

                    maladies_geo = geo_mal[COL_MAL].dropna().astype(str).unique().tolist()
                    mal_geo_focus = st.selectbox(
                        "Maladie à profiler géographiquement",
                        options=maladies_geo,
                        key="idsr_geo_focus_mal",
                    )
                    geo_focus = geo_mal[geo_mal[COL_MAL] == mal_geo_focus].sort_values("Cas", ascending=False)

                    c_g1, c_g2 = st.columns([1.25, 1])
                    with c_g1:
                        fig_geo_focus = px.bar(
                            geo_focus.head(15),
                            x=COL_PROV_ID,
                            y="Cas",
                            title=f"Top provinces – {mal_geo_focus}",
                            text="Cas",
                        )
                        fig_geo_focus.update_layout(template="plotly_white", xaxis_tickangle=-40, height=430)
                        fig_geo_focus = apply_plotly_value_annotations(fig_geo_focus, annot_vals)
                        st.plotly_chart(fig_geo_focus, width="stretch", key="idsr_geo_focus_bar")
                    with c_g2:
                        st.dataframe(
                            geo_focus.assign(**{"CFR_%": geo_focus["CFR_%"].round(2)}),
                            width="stretch",
                            height=430,
                            hide_index=True,
                        )
                else:
                    st.info("La répartition géographique par maladie est indisponible : colonnes Province/Maladie/Total_cas absentes.")

                st.divider()

                # -------------------------------------------------------------
                # D) Profil hebdomadaire par maladie (équivalent dynamique par groupe)
                # -------------------------------------------------------------
                st.markdown("#### D. Dynamique hebdomadaire par maladie")
                if (COL_MAL in df9.columns) and ("Total_cas" in df9.columns) and ("TIME_LAB" in df9.columns) and ("TIME_KEY" in df9.columns):
                    wk_mal = (
                        df9.groupby(["TIME_LAB", "TIME_KEY", COL_MAL], as_index=False)
                        .agg(Cas=("Total_cas", "sum"))
                        .sort_values(["TIME_KEY", COL_MAL])
                    )
                    if not wk_mal.empty:
                        fig_wk_mal = px.line(
                            wk_mal,
                            x="TIME_LAB",
                            y="Cas",
                            color=COL_MAL,
                            markers=True,
                            title="Évolution hebdomadaire des cas par maladie",
                        )
                        fig_wk_mal.update_layout(template="plotly_white", xaxis_tickangle=-45, height=460)
                        st.plotly_chart(fig_wk_mal, width="stretch", key="idsr_weekly_by_disease")

                        with st.expander("Tableau hebdomadaire par maladie", expanded=False):
                            wk_mal_wide = wk_mal.pivot_table(
                                index="TIME_LAB",
                                columns=COL_MAL,
                                values="Cas",
                                aggfunc="sum",
                                fill_value=0,
                                observed=False,
                            ).reset_index()
                            st.dataframe(wk_mal_wide, width="stretch", height=420, hide_index=True)
                    else:
                        st.info("Aucune série hebdomadaire exploitable n’est disponible pour les maladies après filtrage.")
                else:
                    st.info("La dynamique hebdomadaire par maladie est indisponible : colonnes TIME_LAB/TIME_KEY/Maladie/Total_cas absentes.")

            # 14.3) Tableau d’évolution par province et semaine épidémiologique
            with st.expander("Tableau croisé – évolution par province et semaine", expanded=False):

                # Objectif : Provinces en lignes, Année-Semaine en colonnes, sous-colonnes Cas/Décès/Létalité (%)
                if (not df9.empty) and (COL_PROV_ID in df9.columns):

                    # Préparer les colonnes numériques (robuste)
                    tmp_pw = prepare_idsr_numeric(df9, col_cases="Total_cas", col_deaths="Total_deces")

                    # Choix du niveau d’affichage
                    level_pw = st.radio(
                        "Niveau d’affichage",
                        ["Province de notification", "Province + Zone de notification"],
                        horizontal=True,
                        key="idsr_pw_level",
                    )

                    # Colonnes de lignes (index) selon le niveau
                    zs_col = None
                    if ("COL_ZS_ID" in globals()) and (globals()["COL_ZS_ID"] in tmp_pw.columns):
                        zs_col = globals()["COL_ZS_ID"]
                    elif ("COL_ZS" in globals()) and (globals()["COL_ZS"] in tmp_pw.columns):
                        zs_col = globals()["COL_ZS"]

                    idx_cols = [COL_PROV_ID]
                    if (level_pw == "Province + Zone de notification") and (zs_col is not None):
                        idx_cols = [COL_PROV_ID, zs_col]

                    # Colonne semaine (unique) : privilégier Année-Semaine si dispo, sinon TIME_KEY, sinon TIME_LAB
                    week_series, _order_key_col = choose_week_column(tmp_pw)
                    if week_series.empty:
                        st.info("Variables manquantes pour produire le tableau province × semaine (YW/TIME_KEY/TIME_LAB).")
                    else:
                        # Construire pivot Cas/Décès/Létalité (%)
                        pivot = build_cases_deaths_cfr_pivot(
                            tmp_pw,
                            idx_cols=idx_cols,
                            week_series=week_series,
                            col_cases="Total_cas",
                            col_deaths="Total_deces",
                            week_name="_YW_COL",
                            cfr_label="Létalité (%)",
                        )

                        # Ordonner les semaines chronologiquement si possible
                        if "weekly_sorted" in locals() and isinstance(weekly_sorted, pd.DataFrame) and (not weekly_sorted.empty):
                            ordre_w = ordered_weeks_from_weekly_sorted(weekly_sorted, fmt=fmt_yw_label)
                            pivot = reorder_pivot_weeks(pivot, ordre_w, fill_value=0)
                        else:
                            # Fallback : ordre lexical sur YYYYWww (chronologique)
                            ordre_w = sorted(list(pivot.columns.levels[1]))
                            pivot = reorder_pivot_weeks(pivot, ordre_w, fill_value=0)

                        # Rendu standard : CFR arrondi + reset_index + affichage safe
                        render_pivot_with_cfr(pivot, cfr_label="Létalité (%)", cfr_decimals=2, height=520)

                else:
                    st.info("Aucune donnée n’est disponible après filtrage pour produire le tableau province × semaine.")

            # 14.4) Tableau croisé – totaux mensuels (Province / ZS)
            with st.expander("Tableau croisé – totaux mensuels (Province / ZS)", expanded=False):

                if df9.empty:
                    st.info("Aucune donnée n’est disponible après application des filtres analytiques.")
                else:
                    # ---------------------------------------------------------
                    # 1) Construire une date source robuste
                    # ---------------------------------------------------------
                    def _get_date_series(_df: pd.DataFrame) -> pd.Series:
                        # Priorité : Date_debut_semaine_iso (déjà calculée)
                        if "Date_debut_semaine_iso" in _df.columns:
                            s = pd.to_datetime(_df["Date_debut_semaine_iso"], errors="coerce")
                            if s.notna().any():
                                return s

                        # Sinon Date_debut_semaine si dispo
                        if "Date_debut_semaine" in _df.columns:
                            s = pd.to_datetime(_df["Date_debut_semaine"], errors="coerce")
                            if s.notna().any():
                                return s

                        # Sinon DEBUTSEM (Excel serial ou date)
                        if "DEBUTSEM" in _df.columns:
                            _debutsem = _df["DEBUTSEM"]
                            if pd.api.types.is_numeric_dtype(_debutsem):
                                # Excel serial -> date
                                s = pd.to_datetime(_debutsem, unit="D", origin="1899-12-30", errors="coerce")
                            else:
                                s = pd.to_datetime(_debutsem, errors="coerce")
                            if s.notna().any():
                                return s

                        return pd.Series(pd.NaT, index=_df.index)

                    tmp_m = df9.copy()
                    tmp_m["_dt"] = _get_date_series(tmp_m)

                    if tmp_m["_dt"].isna().all():
                        st.warning("Impossible de construire les mois : aucune date exploitable n’a été détectée (Date_debut_semaine_iso / Date_debut_semaine / DEBUTSEM).")
                    else:
                        # ---------------------------------------------------------
                        # 1bis) (Optionnel) filtrer dates absurdes pour éviter 1965/2037
                        # ---------------------------------------------------------
                        dt_min = pd.Timestamp("2000-01-01")
                        dt_max = pd.Timestamp.today() + pd.Timedelta(days=366)
                        tmp_m = tmp_m[tmp_m["_dt"].between(dt_min, dt_max)]

                        if tmp_m.empty:
                            st.warning("Toutes les dates disponibles sont hors de la plage attendue (2000 → année courante + 1). Veuillez vérifier DEBUTSEM/Date_debut_semaine.")
                        else:
                            # Mois (timestamp)
                            tmp_m["_month"] = tmp_m["_dt"].dt.to_period("M").dt.to_timestamp()

                            # Libellé mois en FR: "janv.-2024" (IMPORTANT: %Y pour éviter collisions 1924 vs 2024)
                            mois_fr = {
                                1: "janv.", 2: "févr.", 3: "mars", 4: "avr.", 5: "mai", 6: "juin",
                                7: "juil.", 8: "août", 9: "sept.", 10: "oct.", 11: "nov.", 12: "déc."
                            }
                            tmp_m["_month_lab"] = tmp_m["_dt"].dt.month.map(mois_fr) + "-" + tmp_m["_dt"].dt.strftime("%Y")

                            # ---------------------------------------------------------
                            # 2) Choix niveau: Province / Province+ZS
                            # ---------------------------------------------------------
                            level_m = st.radio(
                                "Niveau d’affichage",
                                ["Provincial", "Zonal (Province + ZS)"],
                                horizontal=True,
                                key="idsr_month_level",
                            )

                            # Colonnes id
                            col_mal = "Maladie" if "Maladie" in tmp_m.columns else COL_MAL
                            col_prov = COL_PROV_ID
                            col_zs = COL_ZS_ID if (COL_ZS_ID in tmp_m.columns) else None

                            idx_cols = [col_mal, col_prov]
                            if (level_m.startswith("Zonal")) and (col_zs is not None):
                                idx_cols = [col_mal, col_prov, col_zs]

                            # ---------------------------------------------------------
                            # 3) Indicateurs à produire
                            # ---------------------------------------------------------
                            metrics = [
                                ("Population", "Population exposée", "max"),
                                ("Cas_0_11mois", "Cas suspects 0 à 11mois", "sum"),
                                ("Cas_12_59mois", "Cas suspects 12mois à 5ans", "sum"),
                                ("Cas_5_14ans", "Cas suspects 5 à 14ans", "sum"),
                                ("Cas_15plus", "Cas suspects Adultes", "sum"),
                                ("Total_deces", "Nombre de décès", "sum"),
                            ]

                            # Garder uniquement les métriques existantes
                            metrics_ok = [(c, lab, agg) for (c, lab, agg) in metrics if c in tmp_m.columns]
                            if not metrics_ok:
                                st.info("Aucune colonne indicateur trouvée (Population / Cas_* / Total_deces).")
                            else:
                                # Préparer valeurs numériques
                                for c, _, _ in metrics_ok:
                                    tmp_m[c] = pd.to_numeric(tmp_m[c], errors="coerce")

                                # -----------------------------------------------------
                                # 4) Construire une table longue puis pivot mensuel
                                # -----------------------------------------------------
                                pieces = []
                                group_base = idx_cols + ["_month", "_month_lab"]

                                for c, lab, agg in metrics_ok:
                                    g = tmp_m[group_base + [c]].copy()

                                    if agg == "max":
                                        out = g.groupby(group_base, as_index=False)[c].max()
                                    else:
                                        out = g.groupby(group_base, as_index=False)[c].sum(min_count=1)

                                    out = out.rename(columns={c: "Valeur"})
                                    out["Données"] = lab
                                    pieces.append(out)

                                long_df = pd.concat(pieces, ignore_index=True)

                                pivot = (
                                    long_df.pivot_table(
                                        index=idx_cols + ["Données"],
                                        columns="_month",
                                        values="Valeur",
                                        aggfunc="sum",
                                        fill_value=0,
                                        observed=False,
                                    )
                                    .reset_index()
                                )

                                # Mapping mois timestamp -> label "janv.-2024"
                                month_map = (
                                    long_df.dropna(subset=["_month"])
                                    .drop_duplicates(subset=["_month"])[["_month", "_month_lab"]]
                                    .sort_values("_month")
                                    .set_index("_month")["_month_lab"]
                                    .to_dict()
                                )

                                new_cols = []
                                for col in pivot.columns:
                                    if isinstance(col, (pd.Timestamp, datetime)):
                                        new_cols.append(month_map.get(pd.Timestamp(col), pd.Timestamp(col).strftime("%b-%Y")))
                                    else:
                                        new_cols.append(col)
                                pivot.columns = new_cols

                                # Tri logique des lignes "Données"
                                order_data = [
                                    "Population exposée",
                                    "Cas suspects 0 à 11mois",
                                    "Cas suspects 12mois à 5ans",
                                    "Cas suspects 5 à 14ans",
                                    "Cas suspects Adultes",
                                    "Nombre de décès",
                                ]
                                pivot["Données"] = pd.Categorical(pivot["Données"], categories=order_data, ordered=True)
                                pivot = pivot.sort_values(idx_cols + ["Données"]).reset_index(drop=True)

                                # -----------------------------------------------------
                                # 5) IMPORTANT: rendre les colonnes uniques (Streamlit/PyArrow)
                                # -----------------------------------------------------
                                def _make_unique(cols):
                                    seen = {}
                                    out = []
                                    for x in cols:
                                        x = str(x)
                                        if x not in seen:
                                            seen[x] = 0
                                            out.append(x)
                                        else:
                                            seen[x] += 1
                                            out.append(f"{x}__{seen[x]}")
                                    return out

                                pivot.columns = _make_unique(pivot.columns)

                                st.dataframe(pivot, width="stretch", height=520, hide_index=True)

                                # -----------------------------------------------------
                                # 6) Exportation CSV / XLSX (colonnes déjà uniques)
                                # -----------------------------------------------------
                                csv_m = pivot.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "⬇️ Télécharger (mensuel) – CSV",
                                    data=csv_m,
                                    file_name="idsr_tableau_mensuel.csv",
                                    mime="text/csv",
                                    key="tab9_dl_monthly_pivot",
                                )

                                xlsx_buffer = BytesIO()
                                with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
                                    pivot.to_excel(writer, sheet_name="Tableau_mensuel", index=False)
                                xlsx_buffer.seek(0)

                                st.download_button(
                                    "⬇️ Télécharger (mensuel) – XLSX",
                                    data=xlsx_buffer,
                                    file_name="idsr_tableau_mensuel.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="tab9_dl_monthly_pivot_xlsx",
                                )

with tab_irep:
    st.subheader("Indice provincial composite de risque épidémique (IREP)")
    tab_help(
        "Lecture et interprétation",
        """
        **🎯 Objectif** : classer les provinces selon un risque combiné (0–100) qui intègre:
        - **Tendance** (hausse récente)
        - **Incidence** (si population disponible)
        - **Létalité**
        - **Promptitude** (retard de notification)
        - **Complétude** (qualité de saisie)

        **🧠 Interprétation** : plus l’**IREP** est élevé, plus la situation mérite attention (investigation / renfort / supervision).
        """,
        expanded=False,
    )

    if df is None or df.empty:
        st.info("Aucune donnée n’est disponible pour calculer l’IREP.")
    else:
        # -----------------------------
        # 1) Choisir colonne semaine
        # -----------------------------
        if "Semaine_epid" in df.columns:
            col_week_irep = "Semaine_epid"
        else:
            # fallback: YW / TIME_KEY / TIME_LAB
            _wk, _ = choose_week_column(df)
            if _wk is not None and _wk.notna().any():
                df["_WEEK_TMP_"] = _wk.astype(str)
                col_week_irep = "_WEEK_TMP_"
            else:
                st.error("Aucune variable semaine n’a été détectée (Semaine_epid / YW / TIME_KEY / TIME_LAB).")
                st.stop()

        # Liste des semaines (tri robuste)
        week_vals = sorted(df[col_week_irep].dropna().astype(str).unique().tolist())
        if not week_vals:
            st.info("Aucune semaine valide n’est disponible pour calculer l’IREP.")
            st.stop()

        # -----------------------------
        # 2) Population (optionnel)
        # -----------------------------
        st.markdown("### Population provinciale (optionnelle, pour le calcul de l’incidence)")
        pop_upl = st.file_uploader(
            "Téléverser un fichier population (csv/xlsx) avec colonnes: Province, Population",
            type=["csv", "xlsx", "xls"],
            key="pop_upload_irep"
        )

        population_map = {}
        if pop_upl is not None:
            try:
                if pop_upl.name.lower().endswith(".csv"):
                    pop_df = pd.read_csv(pop_upl)
                else:
                    pop_df = pd.read_excel(pop_upl)

                # normaliser noms colonnes
                pop_df.columns = [str(c).strip() for c in pop_df.columns]
                # heuristiques colonnes
                prov_col = None
                for c in ["Province_notification", "Province", "province", "PROVINCE"]:
                    if c in pop_df.columns:
                        prov_col = c
                        break
                pop_col = None
                for c in ["Population", "POPULATION", "pop", "POP", "population"]:
                    if c in pop_df.columns:
                        pop_col = c
                        break

                if prov_col is None or pop_col is None:
                    st.warning("Fichier de population non reconnu. Colonnes attendues : 'Province' et 'Population'.")
                else:
                    pop_df = pop_df[[prov_col, pop_col]].dropna()
                    pop_df[prov_col] = pop_df[prov_col].astype(str).str.strip()
                    pop_df[pop_col] = pd.to_numeric(pop_df[pop_col], errors="coerce")
                    pop_df = pop_df.dropna(subset=[pop_col])

                    population_map = dict(zip(pop_df[prov_col], pop_df[pop_col].astype(int)))
                    st.success(f"Population chargée pour {len(population_map)} provinces.")
                    with st.expander("Aperçu population"):
                        st.dataframe(pop_df.head(30), width="stretch")
            except Exception as e:
                st.warning(f"Impossible de lire le fichier population : {e}")

        # -----------------------------
        # 3) Paramètres score (poids & fenêtres)
        # -----------------------------
        st.markdown("### Paramètres de calcul")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            w_trend = st.slider("Poids Tendance", 0.0, 1.0, 0.30, 0.05)
        with c2:
            w_inc = st.slider("Poids Incidence", 0.0, 1.0, 0.25, 0.05)
        with c3:
            w_cfr = st.slider("Poids Létalité", 0.0, 1.0, 0.20, 0.05)
        with c4:
            w_time = st.slider("Poids Promptitude", 0.0, 1.0, 0.15, 0.05)
        with c5:
            w_comp = st.slider("Poids Complétude", 0.0, 1.0, 0.10, 0.05)

        # Normaliser pour éviter somme=0
        w_user = {"trend": w_trend, "incidence": w_inc, "cfr": w_cfr, "timeliness": w_time, "completeness": w_comp}
        if sum(w_user.values()) == 0:
            st.warning("Tous les poids sont à 0. Réinitialisation aux valeurs par défaut.")
            w_user = {"trend": 0.30, "incidence": 0.25, "cfr": 0.20, "timeliness": 0.15, "completeness": 0.10}

        current_week = st.selectbox(
            "Semaine courante",
            options=week_vals,
            index=len(week_vals) - 1
        )

        # Seuil de promptitude (réutilise celui de la sidebar si présent)
        try:
            threshold_days = get_session_int("seuil_jours", 2)
        except Exception:
            threshold_days = 2

        # -----------------------------
        # 4) Préparation minimale des colonnes cas/décès si besoin (line list)
        # -----------------------------
        df_irep = df.copy()

        if "Total_cas" not in df_irep.columns:
            df_irep["Total_cas"] = 1

        if "Total_deces" not in df_irep.columns:
            if COL_ISSUE in df_irep.columns:
                df_irep["Total_deces"] = df_irep[COL_ISSUE].apply(lambda x: 1 if is_death(x) else 0)
            else:
                df_irep["Total_deces"] = 0

        # -----------------------------
        # 5) Calcul IREP
        # -----------------------------
        irep = compute_irep_province(
            df_irep,
            col_prov=COL_PROV if COL_PROV in df_irep.columns else "Province",
            col_week=col_week_irep,
            col_cases="Total_cas",
            col_deaths="Total_deces",
            current_week=str(current_week),
            population_map=population_map,
            date_onset=DATE_ONSET,
            date_notif="Date_notification",
            w=w_user,
            threshold_days=threshold_days,
        )

        if irep is None or irep.empty:
            st.info("IREP: aucun résultat (vérifie les colonnes Province / Semaine / Cas).")
        else:
            # KPIs synthèse
            st.markdown("### Synthèse")
            kA, kB, kC, kD = st.columns(4)
            kA.metric("Provinces (IREP calculé)", str(irep[COL_PROV].nunique() if COL_PROV in irep.columns else len(irep)))
            kB.metric("IREP moyen", f"{irep['IREP'].mean():.1f}" if 'IREP' in irep.columns else "-")
            kC.metric("IREP max", f"{irep['IREP'].max():.1f}" if 'IREP' in irep.columns else "-")
            kD.metric("Semaine", str(current_week))

            # Top 5
            st.markdown("### Top provinces à risque")
            st.dataframe(irep.head(10), width="stretch", height=320)

            # Graphique
            try:
                plot_df = irep.copy()
                prov_col = COL_PROV if COL_PROV in plot_df.columns else plot_df.columns[0]
                fig = px.bar(
                    plot_df,
                    x=prov_col,
                    y="IREP",
                    color="Risque_cat" if "Risque_cat" in plot_df.columns else None,
                    title="IREP par province (plus haut = plus à risque)",
                )
                fig.update_layout(xaxis_tickangle=-45)
                fig = apply_plotly_value_annotations(fig, annot_vals)
                st.plotly_chart(fig, width="stretch")
            except Exception:
                pass

            # Download
            st.download_button(
                "⬇️ Télécharger IREP (CSV)",
                data=df_to_csv_bytes(irep),
                file_name=f"IREP_provinces_{current_week}.csv",
                mime="text/csv"
            )

with tab_maps:
    render_detailed_maps_tab(
        df_f=df_f,
        show_maps=show_maps,
        idsr_mode=IDSR_MODE,
    )

render_footer()
