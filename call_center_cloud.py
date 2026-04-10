import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

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

st.set_page_config(layout="wide")

GEOJSON_PROVINCES = "data/geometry_rdc_provinces.geojson"
GEOJSON_PROVINCES_FALLBACK = "geometry_rdc_provinces.geojson"

COL_DATE = "Date"
COL_HEURE = "Heure"
COL_PROVINCE = "Province_notification"
COL_TERRITOIRE = "Territoire_notification"
COL_ZONE_SANTE = "Zone_de_sante_notification"
COL_PATHOLOGIE = "Type_pathologie"
COL_CATEGORIE = "Categorie"
COL_STATUT = "Statut_appel"
COL_ITEM = "Item"
COL_SEXE = "Sexe"
COL_QUALIFICATION = "Nom_qualification"
COL_RESOLUTION = "Resolution"
COL_PROVINCE_NORM = "_province_norm"
COL_HEURE_NUM = "_heure_num"

REQUIRED_COLUMNS = [
    COL_DATE,
    COL_PROVINCE,
    COL_TERRITOIRE,
    COL_PATHOLOGIE,
    COL_CATEGORIE,
    COL_STATUT,
    COL_ITEM,
]

PROVINCE_PATTERNS = [
    (r"^\s*bas[\s_-]*uele\s*$", "Bas Uele"),
    (r"^\s*equateur\s*$", "Equateur"),
    (r"^\s*haut[\s_-]*katanga\s*$", "Haut Katanga"),
    (r"^\s*haut[\s_-]*lomami\s*$", "Haut Lomami"),
    (r"^\s*haut[\s_-]*uele\s*$", "Haut Uele"),
    (r"^\s*ituri\s*$", "Ituri"),
    (r"^\s*kasai[\s_-]*central\s*$", "Kasai Central"),
    (r"^\s*kasai\s*$", "Kasai"),
    (r"^\s*kinshasa\s*$", "Kinshasa"),
    (r"^\s*kongo[\s_-]*central\s*$", "Kongo Central"),
    (r"^\s*kasai[\s_-]*oriental\s*$", "Kasai Oriental"),
    (r"^\s*kwango\s*$", "Kwango"),
    (r"^\s*kwilu\s*$", "Kwilu"),
    (r"^\s*lomami\s*$", "Lomami"),
    (r"^\s*lualaba\s*$", "Lualaba"),
    (r"^\s*mai[\s_-]*ndombe\s*$", "Maindombe"),
    (r"^\s*maindombe\s*$", "Maindombe"),
    (r"^\s*maniema\s*$", "Maniema"),
    (r"^\s*mongala\s*$", "Mongala"),
    (r"^\s*nord[\s_-]*kivu\s*$", "Nord Kivu"),
    (r"^\s*nord[\s_-]*ubangi\s*$", "Nord Ubangi"),
    (r"^\s*sankuru\s*$", "Sankuru"),
    (r"^\s*sud[\s_-]*kivu\s*$", "Sud Kivu"),
    (r"^\s*sud[\s_-]*ubangi\s*$", "Sud Ubangi"),
    (r"^\s*tanganyika\s*$", "Tanganyika"),
    (r"^\s*tshuapa\s*$", "Tshuapa"),
    (r"^\s*tshopo\s*$", "Tshopo"),
]


def normalize_text(value):
    return str(value).lower().strip().replace("-", " ")


def strip_accents(value):
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(character for character in normalized if not unicodedata.combining(character))


def normalize_column_key(value):
    normalized = strip_accents(value).strip().lower()
    normalized = normalized.replace("n°", "n").replace("nº", "n").replace("№", "n")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


COLUMN_NAME_ALIASES = {
    "n": "N",
    "no": "N",
    "numero": "N",
    "date": COL_DATE,
    "heure": COL_HEURE,
    "nom_de_la_qualification": COL_QUALIFICATION,
    "nom_qualification": COL_QUALIFICATION,
    "qualification": COL_QUALIFICATION,
    "numero_appelant": "Numero_appelant",
    "numero_de_l_appelant": "Numero_appelant",
    "num_appelant": "Numero_appelant",
    "autre_numero": "Autre_numero",
    "autre_numero_telephone": "Autre_numero",
    "nom": "Nom_complet",
    "nom_complet": "Nom_complet",
    "prenom": "Prenom",
    "province": COL_PROVINCE,
    "province_notification": COL_PROVINCE,
    "territoire": COL_TERRITOIRE,
    "territoire_notification": COL_TERRITOIRE,
    "zone_de_sante": COL_ZONE_SANTE,
    "zone_de_sante_notification": COL_ZONE_SANTE,
    "genre": COL_SEXE,
    "sexe": COL_SEXE,
    "categorie": COL_CATEGORIE,
    "categorie_appel": COL_CATEGORIE,
    "type": COL_PATHOLOGIE,
    "type_pathologie": COL_PATHOLOGIE,
    "type_de_pathologie": COL_PATHOLOGIE,
    "item": COL_ITEM,
    "details_de_l_appel": "Details_appel",
    "details_appel": "Details_appel",
    "detail_appel": "Details_appel",
    "resolution": COL_RESOLUTION,
    "statutappel": COL_STATUT,
    "statut_appel": COL_STATUT,
}


def normalize_province_name(value):
    cleaned_value = strip_accents(value).strip()
    cleaned_value = re.sub(r"\s+", " ", cleaned_value)
    for pattern, province_name in PROVINCE_PATTERNS:
        if re.match(pattern, cleaned_value, flags=re.IGNORECASE):
            return province_name
    return normalize_text(strip_accents(cleaned_value)).title()


def normalize_status(value):
    return strip_accents(str(value).casefold().strip())


def format_number(value):
    return f"{int(value):,}".replace(",", " ")


@st.cache_data(show_spinner=False)
def load_provinces_geojson(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def resolve_provinces_geojson_path():
    for path in [GEOJSON_PROVINCES, GEOJSON_PROVINCES_FALLBACK]:
        if Path(path).exists():
            return path
    return None


def reset_file_pointer(file_obj):
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)


def get_excel_sheet_names(uploaded_file):
    try:
        reset_file_pointer(uploaded_file)
        sheet_names = pd.ExcelFile(uploaded_file).sheet_names
    except Exception as exc:
        st.error(f"Impossible de lire le fichier Excel : {exc}")
        st.stop()
    finally:
        reset_file_pointer(uploaded_file)
    return sheet_names


def read_excel_file(uploaded_file, sheet_name=None):
    sheet_names = get_excel_sheet_names(uploaded_file)
    selected_sheet = str(sheet_name).strip() if sheet_name else ""
    selected_sheet = selected_sheet or sheet_names[0]

    if selected_sheet not in sheet_names:
        st.error(f"La feuille Excel '{selected_sheet}' est introuvable.")
        st.write("Feuilles disponibles :", ", ".join(sheet_names))
        st.stop()

    try:
        reset_file_pointer(uploaded_file)
        df_loaded = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    except Exception as exc:
        st.error(f"Impossible de lire la feuille Excel '{selected_sheet}' : {exc}")
        st.stop()
    finally:
        reset_file_pointer(uploaded_file)

    df_loaded.columns = df_loaded.columns.str.strip()
    return df_loaded


def validate_table_identifier(identifier):
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*", str(identifier).strip()))


def build_postgresql_query(query_mode, table_name, sql_query):
    if query_mode == "Table":
        clean_table_name = str(table_name).strip()
        if not clean_table_name:
            st.error("Renseigne le nom de la table PostgreSQL.")
            st.stop()
        if not validate_table_identifier(clean_table_name):
            st.error("Le nom de table contient des caracteres non autorises.")
            st.stop()
        return f"SELECT * FROM {clean_table_name}"

    clean_query = str(sql_query).strip()
    if not clean_query:
        st.error("Renseigne une requete SQL PostgreSQL.")
        st.stop()
    return clean_query


def read_postgresql_file(host, port, database, user, password, query):
    if not SQLALCHEMY_AVAILABLE and not PSYCOPG2_AVAILABLE:
        st.error("Le connecteur PostgreSQL n'est pas installe. Ajoute `sqlalchemy` et `psycopg2-binary`.")
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
        st.error(f"Impossible de charger les donnees PostgreSQL : {exc}")
        st.stop()

    df_loaded.columns = df_loaded.columns.str.strip()
    return df_loaded


def is_blank_series(series):
    return series.isna() | series.astype(str).str.strip().eq("")


def merge_duplicate_columns(df_loaded):
    merged = pd.DataFrame(index=df_loaded.index)
    for column in list(dict.fromkeys(df_loaded.columns)):
        same_columns = df_loaded.loc[:, df_loaded.columns == column]
        combined = same_columns.iloc[:, 0]
        for idx in range(1, same_columns.shape[1]):
            next_series = same_columns.iloc[:, idx]
            combined = combined.where(~is_blank_series(combined), next_series)
        merged[column] = combined
    return merged


def standardize_source_columns(df_loaded):
    df_loaded = df_loaded.copy()
    df_loaded.columns = [str(column).strip() for column in df_loaded.columns]

    renamed_columns = {}
    for column in df_loaded.columns:
        normalized_key = normalize_column_key(column)
        renamed_columns[column] = COLUMN_NAME_ALIASES.get(normalized_key, column)

    df_loaded = df_loaded.rename(columns=renamed_columns)
    df_loaded = merge_duplicate_columns(df_loaded)

    if "Prenom" in df_loaded.columns:
        nom_series = (
            df_loaded["Nom_complet"].fillna("").astype(str).str.strip()
            if "Nom_complet" in df_loaded.columns
            else pd.Series("", index=df_loaded.index, dtype="object")
        )
        prenom_series = df_loaded["Prenom"].fillna("").astype(str).str.strip()
        full_name = (nom_series + " " + prenom_series).str.strip()
        if "Nom_complet" in df_loaded.columns:
            df_loaded["Nom_complet"] = full_name.where(full_name.ne(""), df_loaded["Nom_complet"])
        else:
            df_loaded["Nom_complet"] = full_name
        df_loaded = df_loaded.drop(columns=["Prenom"])

    return df_loaded


def require_columns(df_loaded):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df_loaded.columns]
    if missing_columns:
        st.error("Les donnees ne contiennent pas toutes les colonnes attendues, meme apres harmonisation.")
        st.write("Colonnes manquantes :", ", ".join(missing_columns))
        st.write("Colonnes detectees :", ", ".join(df_loaded.columns))
        st.stop()


def sorted_options(series):
    clean_series = series.dropna().astype(str).str.strip().replace("", pd.NA).dropna()
    return sorted(clean_series.unique())


def selected_values(label, options, key):
    all_options = ["Toutes"] + options
    default_value = st.session_state.get(key, ["Toutes"])
    default_value = [value for value in default_value if value in all_options] or ["Toutes"]
    selected = st.sidebar.multiselect(label, options=all_options, default=default_value, key=key)
    if len(selected) > 1 and "Toutes" in selected:
        return [value for value in selected if value != "Toutes"]
    return selected


def apply_multiselect_filter(df_loaded, column, selected):
    if selected and "Toutes" not in selected:
        return df_loaded[df_loaded[column].isin(selected)]
    return df_loaded


def apply_optional_multiselect(df_loaded, column, selected):
    if column in df_loaded.columns:
        return apply_multiselect_filter(df_loaded, column, selected)
    return df_loaded


def optional_selected_values(label, df_loaded, column, key):
    if column not in df_loaded.columns:
        return ["Toutes"]
    return selected_values(label, sorted_options(df_loaded[column]), key)


def parse_hour_value(value):
    if pd.isna(value):
        return pd.NA
    if hasattr(value, "hour"):
        return int(value.hour)
    if isinstance(value, pd.Timedelta):
        return int(value.total_seconds() // 3600) % 24
    text_value = str(value).strip()
    if not text_value:
        return pd.NA
    parsed = pd.to_datetime(text_value, errors="coerce")
    if pd.isna(parsed):
        return pd.NA
    return int(parsed.hour)


def prepare_data(df_loaded):
    df_loaded = standardize_source_columns(df_loaded)
    require_columns(df_loaded)

    df_loaded[COL_DATE] = pd.to_datetime(df_loaded[COL_DATE], errors="coerce")
    text_columns = [
        COL_PROVINCE,
        COL_TERRITOIRE,
        COL_PATHOLOGIE,
        COL_CATEGORIE,
        COL_STATUT,
        COL_ITEM,
    ]
    for optional_text_column in [COL_ZONE_SANTE, COL_SEXE, COL_QUALIFICATION, COL_RESOLUTION]:
        if optional_text_column in df_loaded.columns:
            text_columns.append(optional_text_column)

    for text_column in text_columns:
        df_loaded[text_column] = df_loaded[text_column].fillna("").astype(str).str.strip()

    df_loaded[COL_PROVINCE_NORM] = df_loaded[COL_PROVINCE].apply(normalize_province_name)
    if COL_HEURE in df_loaded.columns:
        df_loaded[COL_HEURE_NUM] = df_loaded[COL_HEURE].apply(parse_hour_value).astype("Int64")

    if df_loaded[COL_DATE].dropna().empty:
        st.error("La colonne Date ne contient aucune date valide.")
        st.stop()

    return df_loaded


def render_css():
    st.markdown(
        """
<style>
.stApp {background: linear-gradient(135deg, #e8f1ee 0%, #e4edf5 52%, #eef3ea 100%);}
section[data-testid="stSidebar"] {background: #dfeaf2;}
section[data-testid="stSidebar"] > div {background: #dfeaf2;}
section[data-testid="stSidebar"] label {color: #001b47; font-weight: 700;}
.main .block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 1500px;}
.header {
    background: linear-gradient(90deg, #001b47 0%, #062a67 62%, #0a4f92 100%);
    padding: 18px 24px;
    border-radius: 8px;
    color: white;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 10px 24px rgba(0, 31, 84, 0.16);
}
.header h2 {margin: 0; letter-spacing: 1.5px; font-size: 28px; font-weight: 800;}
.header p {margin: 4px 0 0; font-size: 17px; opacity: 0.92; font-weight: 600;}
.kpi {
    min-height: 118px;
    padding: 18px 18px 14px;
    border-radius: 8px;
    color: #001b47;
    font-weight: 700;
    background: #ffffff;
    border: 1px solid #d8e5f2;
    box-shadow: 0 5px 15px rgba(0, 31, 84, 0.10);
    position: relative;
    overflow: hidden;
}
.kpi::before {
    content: "";
    width: 58px;
    height: 58px;
    border-radius: 50%;
    position: absolute;
    left: 18px;
    top: 22px;
    background: var(--accent, #0757b8);
}
.kpi-title {font-size: 13px; letter-spacing: 1px; text-transform: uppercase; margin-left: 78px;}
.kpi-value {font-size: 31px; margin: 8px 0 0 78px; line-height: 1; color: var(--accent, #0757b8);}
.kpi-note {font-size: 13px; margin: 11px 0 0 78px; color: #23395d; font-weight: 500;}
.kpi-blue {--accent: #0757b8;}
.kpi-green {--accent: #15803d; background: #f4fbf5;}
.kpi-orange {--accent: #d97706; background: #fff8ee;}
.kpi-dark {--accent: #002b66;}
.block {
    background: #fbfdfd;
    padding: 16px 18px;
    border-radius: 8px;
    border: 1px solid #dce7f3;
    box-shadow: 0 4px 12px rgba(0, 31, 84, 0.08);
    min-height: 100%;
}
.map-card {
    background: #fbfaf7;
    padding: 16px 18px;
    border-radius: 8px;
    border: 1px solid #d8d8d2;
    box-shadow: 0 4px 12px rgba(0, 31, 84, 0.08);
    min-height: 100%;
}
.block-title {
    color: #001b47;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.hint {
    background: #e9f2ff;
    color: #002b66;
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 14px;
    font-weight: 700;
    font-size: 12px;
}
.rank-row {
    display: grid;
    grid-template-columns: minmax(90px, 1fr) 64px minmax(85px, 1fr) 48px;
    gap: 10px;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid #e7eef6;
    color: #17233c;
    font-size: 13px;
    font-weight: 700;
}
.rank-head {color: #001b47; font-weight: 800;}
.rank-bar {height: 18px; background: #e7eef6; border-radius: 3px; overflow: hidden;}
.rank-fill {display: block; height: 100%; background: #003979; border-radius: 3px;}
.footer {
    background: #001b47;
    color: white;
    padding: 18px 28px;
    border-radius: 8px;
    border: 1px solid #3d75b8;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 24px;
    font-weight: 700;
    margin-top: 12px;
}
.footer small {display: block; margin-top: 4px; opacity: 0.78; font-weight: 500;}
@media (max-width: 900px) {
    .footer {grid-template-columns: 1fr;}
    .header h2 {font-size: 22px;}
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
<div class="header">
<h2>DASHBOARD DE CALL CENTER SANTE PUBLIQUE</h2>
<p>Suivi des appels et surveillance epidemiologique</p>
</div>
""",
        unsafe_allow_html=True,
    )


def filter_data(df_loaded):
    date_range = st.sidebar.date_input(
        "Date",
        [df_loaded[COL_DATE].min().date(), df_loaded[COL_DATE].max().date()],
    )

    if len(date_range) != 2:
        st.info("Selectionne une date de debut et une date de fin.")
        st.stop()

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    df_by_date = df_loaded[(df_loaded[COL_DATE] >= start_date) & (df_loaded[COL_DATE] <= end_date)]

    province_options = sorted_options(df_by_date[COL_PROVINCE_NORM])
    clicked_province = st.session_state.pop("map_clicked_province", None)
    if clicked_province in province_options:
        st.session_state["province_sel"] = [clicked_province]

    province_sel = selected_values("Province", province_options, "province_sel")

    df_for_territoire = apply_multiselect_filter(df_by_date, COL_PROVINCE_NORM, province_sel)
    territoire_sel = selected_values("Territoire", sorted_options(df_for_territoire[COL_TERRITOIRE]), "territoire_sel")
    df_for_zone_sante = apply_multiselect_filter(df_for_territoire, COL_TERRITOIRE, territoire_sel)

    zone_sante_sel = optional_selected_values("Zone de santé", df_for_zone_sante, COL_ZONE_SANTE, "zone_sante_sel")
    df_for_pathologie = apply_optional_multiselect(df_for_zone_sante, COL_ZONE_SANTE, zone_sante_sel)

    pathologie_sel = selected_values("Pathologie", sorted_options(df_for_pathologie[COL_PATHOLOGIE]), "pathologie_sel")
    df_for_categorie = apply_multiselect_filter(df_for_pathologie, COL_PATHOLOGIE, pathologie_sel)

    categorie_sel = selected_values("Categorie", sorted_options(df_for_categorie[COL_CATEGORIE]), "categorie_sel")
    df_for_statut = apply_multiselect_filter(df_for_categorie, COL_CATEGORIE, categorie_sel)

    statut_sel = selected_values("Statut appel", sorted_options(df_for_statut[COL_STATUT]), "statut_sel")
    df_for_sexe = apply_multiselect_filter(df_for_statut, COL_STATUT, statut_sel)

    sexe_sel = optional_selected_values("Genre / sexe", df_for_sexe, COL_SEXE, "sexe_sel")
    df_for_qualification = apply_optional_multiselect(df_for_sexe, COL_SEXE, sexe_sel)

    qualification_sel = optional_selected_values(
        "Nom de la qualification",
        df_for_qualification,
        COL_QUALIFICATION,
        "qualification_sel",
    )
    df_for_resolution = apply_optional_multiselect(df_for_qualification, COL_QUALIFICATION, qualification_sel)

    hour_range = None
    if COL_HEURE_NUM in df_for_resolution.columns and df_for_resolution[COL_HEURE_NUM].notna().any():
        min_hour = int(df_for_resolution[COL_HEURE_NUM].dropna().min())
        max_hour = int(df_for_resolution[COL_HEURE_NUM].dropna().max())
        if min_hour < max_hour:
            hour_range = st.sidebar.slider(
                "Plage horaire",
                min_value=0,
                max_value=23,
                value=(min_hour, max_hour),
                step=1,
                format="%dh",
            )

    st.sidebar.divider()
    top_n = st.sidebar.slider("Nombre d'elements dans les tops", min_value=3, max_value=15, value=5, step=1)
    time_grain = st.sidebar.selectbox("Granularite de la courbe", ["Jour", "Semaine", "Mois"], index=0)
    show_curve_labels = st.sidebar.checkbox("Afficher les valeurs sur la courbe", value=True)

    df_filtered = df_by_date.copy()
    df_filtered = apply_multiselect_filter(df_filtered, COL_PROVINCE_NORM, province_sel)
    df_filtered = apply_multiselect_filter(df_filtered, COL_TERRITOIRE, territoire_sel)
    df_filtered = apply_optional_multiselect(df_filtered, COL_ZONE_SANTE, zone_sante_sel)
    df_filtered = apply_multiselect_filter(df_filtered, COL_PATHOLOGIE, pathologie_sel)
    df_filtered = apply_multiselect_filter(df_filtered, COL_CATEGORIE, categorie_sel)
    df_filtered = apply_multiselect_filter(df_filtered, COL_STATUT, statut_sel)
    df_filtered = apply_optional_multiselect(df_filtered, COL_SEXE, sexe_sel)
    df_filtered = apply_optional_multiselect(df_filtered, COL_QUALIFICATION, qualification_sel)
    if hour_range and COL_HEURE_NUM in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered[COL_HEURE_NUM] >= hour_range[0])
            & (df_filtered[COL_HEURE_NUM] <= hour_range[1])
        ]

    return df_filtered, {
        "top_n": top_n,
        "time_grain": time_grain,
        "show_curve_labels": show_curve_labels,
    }


def render_kpis(df_filtered):
    total = len(df_filtered)
    clotures = df_filtered[df_filtered[COL_STATUT].apply(normalize_status) == "cloture"].shape[0]
    non_clotures = total - clotures
    taux = (clotures / total * 100) if total > 0 else 0
    moyenne = int(total / max(1, df_filtered[COL_DATE].nunique()))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(
        f"""
<div class='kpi kpi-blue'>
  <div class='kpi-title'>Total appels recus</div>
  <div class='kpi-value'>{format_number(total)}</div>
  <div class='kpi-note'>Appels enregistres</div>
</div>
""",
        unsafe_allow_html=True,
    )
    k2.markdown(
        f"""
<div class='kpi kpi-green'>
  <div class='kpi-title'>Appels clotures</div>
  <div class='kpi-value'>{format_number(clotures)}</div>
  <div class='kpi-note'>Statut = Cloture</div>
</div>
""",
        unsafe_allow_html=True,
    )
    k3.markdown(
        f"""
<div class='kpi kpi-orange'>
  <div class='kpi-title'>Appels non clotures</div>
  <div class='kpi-value'>{format_number(non_clotures)}</div>
  <div class='kpi-note'>A traiter ou suivre</div>
</div>
""",
        unsafe_allow_html=True,
    )
    k4.markdown(
        f"""
<div class='kpi kpi-dark'>
  <div class='kpi-title'>Taux de cloture</div>
  <div class='kpi-value'>{taux:.1f}%</div>
  <div class='kpi-note'>Clotures / total</div>
</div>
""",
        unsafe_allow_html=True,
    )
    k5.markdown(
        f"""
<div class='kpi kpi-blue'>
  <div class='kpi-title'>Moyenne appels / jour</div>
  <div class='kpi-value'>{format_number(moyenne)}</div>
  <div class='kpi-note'>Moyenne sur la periode</div>
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def build_map_dataset(df_filtered, geojson_path):
    geojson = load_provinces_geojson(geojson_path)
    features = geojson.get("features", [])
    if not features:
        raise ValueError("Le GeoJSON ne contient aucune feature.")

    candidate_keys = ["province", "name", "name_1", "nom", "adm1_name", "shapeName"]
    feature_key = None
    for key in candidate_keys:
        if all(key in feature.get("properties", {}) for feature in features[: min(5, len(features))]):
            feature_key = key
            break

    if feature_key is None:
        sample_keys = list(features[0].get("properties", {}).keys())
        for key in sample_keys:
            values = [str(feature.get("properties", {}).get(key, "")).strip() for feature in features]
            if sum(bool(v) for v in values) >= max(5, len(values) // 2):
                feature_key = key
                break

    if feature_key is None:
        raise ValueError("Impossible d'identifier le champ province dans le GeoJSON.")

    rows = []
    for idx, feature in enumerate(features):
        province_raw = feature.get("properties", {}).get(feature_key, "")
        province_norm = normalize_province_name(province_raw)
        feature["properties"]["feature_id"] = str(idx)
        feature["properties"]["province_norm"] = province_norm
        rows.append({
            "feature_id": str(idx),
            "Province_geojson": str(province_raw),
            "prov_norm": province_norm,
        })

    map_df = pd.DataFrame(rows)
    counts = (
        df_filtered[[COL_PROVINCE_NORM]]
        .rename(columns={COL_PROVINCE_NORM: "prov_norm"})
        .groupby("prov_norm", as_index=False)
        .size()
        .rename(columns={"size": "nb"})
    )
    map_df = map_df.merge(counts, on="prov_norm", how="left")
    map_df["nb"] = map_df["nb"].fillna(0).astype(int)

    return geojson, map_df


def get_selected_map_point(selection_state):
    if not selection_state:
        return None
    selection = selection_state.get("selection", {}) if hasattr(selection_state, "get") else {}
    points = selection.get("points", []) if isinstance(selection, dict) else []
    return points[0] if points else None


def render_map(df_filtered):
    st.markdown("<div class='map-card'><div class='block-title'>Repartition des appels par province</div>", unsafe_allow_html=True)

    geojson_path = resolve_provinces_geojson_path()
    if geojson_path is None:
        st.info("GeoJSON introuvable. Ajoute 'data/geometry_rdc_provinces.geojson' ou 'geometry_rdc_provinces.geojson'.")
        province_counts = df_filtered[COL_PROVINCE_NORM].value_counts().reset_index()
        province_counts.columns = ["Province", "Nombre"]
        if not province_counts.empty:
            fig_fallback = px.bar(
                province_counts.sort_values("Nombre", ascending=True),
                x="Nombre",
                y="Province",
                orientation="h",
                color_discrete_sequence=["#9b001f"],
            )
            fig_fallback.update_layout(height=500, margin=dict(l=8, r=8, t=8, b=8), showlegend=False)
            st.plotly_chart(fig_fallback, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
        return df_filtered

    try:
        geojson, map_df = build_map_dataset(df_filtered, geojson_path)
    except Exception as exc:
        st.warning(f"Carte indisponible : {exc}")
        st.markdown("</div>", unsafe_allow_html=True)
        return df_filtered

    fig_map = px.choropleth(
        map_df,
        geojson=geojson,
        locations="feature_id",
        featureidkey="properties.feature_id",
        color="nb",
        custom_data=["Province_geojson", "prov_norm"],
        color_continuous_scale=[
            [0.0, "#fff7f3"],
            [0.15, "#fde7df"],
            [0.45, "#fca487"],
            [0.75, "#fb5a49"],
            [1.0, "#9b001f"],
        ],
        projection="mercator",
    )
    fig_map.update_traces(
        marker_line_color="#d2d2cc",
        marker_line_width=1.0,
        hovertemplate="<b>%{customdata[0]}</b><br>Appels: %{z}<extra></extra>",
    )
    fig_map.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="#fbfaf7",
        domain=dict(x=[0.02, 0.98], y=[0.02, 0.98]),
        showland=True,
        lakecolor="#fbfaf7",
        landcolor="#fbfaf7",
        showocean=True,
        oceancolor="#fbfaf7",
        showlakes=True,
        showcountries=False,
        showcoastlines=False,
        showframe=False,
    )
    fig_map.update_layout(
        height=500,
        margin=dict(l=8, r=8, t=4, b=4),
        paper_bgcolor="#fbfaf7",
        plot_bgcolor="#fbfaf7",
        coloraxis_showscale=False,
        dragmode="pan",
    )

    selection_state = st.plotly_chart(
        fig_map,
        width="stretch",
        height=520,
        key="province_map",
        on_select="rerun",
        selection_mode="points",
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["zoomInGeo", "zoomOutGeo", "resetGeo"],
        },
    )

    clicked_point = get_selected_map_point(selection_state)
    if clicked_point:
        customdata = clicked_point.get("customdata") or []
        province = customdata[0] if len(customdata) > 0 else None
        province_norm = customdata[1] if len(customdata) > 1 else None
        current_selection = st.session_state.get("province_sel", [])
        if province_norm and current_selection != [province_norm]:
            st.session_state["map_clicked_province"] = province_norm
            st.markdown(
                "<div class='hint'>Filtre actif : {}</div>".format(html.escape(str(province))),
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.rerun()

    st.markdown(
        "<div class='hint'>Clique sur une province pour filtrer. Utilise la molette ou la barre d'outils pour zoomer/dezoomer.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return df_filtered


def style_figure(fig, height=235):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#17233c", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf2f7", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def render_ranking(title, df_ranked, label_column):
    total = df_ranked["Nombre"].sum()
    max_value = max(1, df_ranked["Nombre"].max()) if not df_ranked.empty else 1

    rows = [
        "<div class='rank-row rank-head'><span>{}</span><span>Appels</span><span></span><span>%</span></div>".format(
            label_column
        )
    ]

    for _, row in df_ranked.iterrows():
        percent = (row["Nombre"] / total * 100) if total else 0
        width = row["Nombre"] / max_value * 100
        label = html.escape(str(row[label_column]))
        count = format_number(row["Nombre"])
        rows.append(
            f"""
<div class='rank-row'>
  <span>{label}</span>
  <span>{count}</span>
  <span class='rank-bar'><span class='rank-fill' style='width:{width:.1f}%'></span></span>
  <span>{percent:.1f}%</span>
</div>
"""
        )

    st.markdown(
        f"<div class='block'><div class='block-title'>{title}</div>{''.join(rows)}</div>",
        unsafe_allow_html=True,
    )


def render_sexe_chart(df_filtered):
    st.markdown("<div class='block'><div class='block-title'>Repartition par sexe</div>", unsafe_allow_html=True)
    if COL_SEXE not in df_filtered.columns:
        st.info("La colonne Sexe n'est pas presente dans le fichier Excel.")
    else:
        df_sexe = df_filtered[df_filtered[COL_SEXE] != ""]
        if df_sexe.empty:
            st.info("Aucune valeur exploitable dans la colonne Sexe.")
        else:
            fig_sexe = px.pie(
                df_sexe,
                names=COL_SEXE,
                hole=0.58,
                color_discrete_sequence=["#1f7ae0", "#e3342f", "#16a34a", "#d97706"],
            )
            fig_sexe.update_traces(textinfo="percent+label")
            st.plotly_chart(style_figure(fig_sexe, height=220), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def build_time_series(df_filtered, time_grain):
    df_time = df_filtered[[COL_DATE, COL_STATUT]].dropna(subset=[COL_DATE]).copy()
    df_time["cloture"] = df_time[COL_STATUT].apply(normalize_status) == "cloture"

    if time_grain == "Semaine":
        df_time["time_order"] = df_time[COL_DATE].dt.to_period("W").apply(lambda value: value.start_time)
        df_time["Periode"] = "Sem. " + df_time["time_order"].dt.strftime("%d/%m/%Y")
    elif time_grain == "Mois":
        df_time["time_order"] = df_time[COL_DATE].dt.to_period("M").apply(lambda value: value.start_time)
        df_time["Periode"] = df_time["time_order"].dt.strftime("%m/%Y")
    else:
        df_time["time_order"] = df_time[COL_DATE].dt.normalize()
        df_time["Periode"] = df_time["time_order"].dt.strftime("%d/%m/%Y")

    grouped = (
        df_time.groupby(["time_order", "Periode"], as_index=False)
        .agg(**{"Appels recus": (COL_DATE, "size"), "Appels clotures": ("cloture", "sum")})
        .sort_values("time_order")
    )

    return grouped.melt(
        id_vars=["time_order", "Periode"],
        value_vars=["Appels recus", "Appels clotures"],
        var_name="Indicateur",
        value_name="Nombre",
    )


def render_time_trend_chart(df_filtered, chart_options):
    st.markdown("<div class='block'><div class='block-title'>Evolution des appels dans le temps</div>", unsafe_allow_html=True)
    df_time = build_time_series(df_filtered, chart_options["time_grain"])
    fig_time = px.line(
        df_time,
        x="Periode",
        y="Nombre",
        color="Indicateur",
        markers=True,
        text="Nombre" if chart_options["show_curve_labels"] else None,
        color_discrete_map={"Appels recus": "#1f7ae0", "Appels clotures": "#15803d"},
    )
    if chart_options["show_curve_labels"]:
        fig_time.update_traces(textposition="top center")
    st.plotly_chart(style_figure(fig_time, height=255), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def render_hour_chart(df_filtered):
    st.markdown("<div class='block'><div class='block-title'>Appels par heure</div>", unsafe_allow_html=True)
    if COL_HEURE_NUM not in df_filtered.columns or df_filtered[COL_HEURE_NUM].dropna().empty:
        st.info("La colonne Heure n'est pas exploitable pour cette analyse.")
    else:
        hour_counts = (
            df_filtered[COL_HEURE_NUM]
            .dropna()
            .astype(int)
            .value_counts()
            .reindex(range(24), fill_value=0)
            .rename_axis("Heure_num")
            .reset_index(name="Nombre")
        )
        hour_counts["Heure"] = hour_counts["Heure_num"].apply(lambda value: f"{value:02d}h")
        fig_hour = px.bar(
            hour_counts,
            x="Heure",
            y="Nombre",
            color_discrete_sequence=["#0f62fe"],
        )
        fig_hour.update_layout(showlegend=False)
        st.plotly_chart(style_figure(fig_hour, height=250), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def render_weekday_chart(df_filtered):
    st.markdown("<div class='block'><div class='block-title'>Appels par jour de semaine</div>", unsafe_allow_html=True)
    if df_filtered[COL_DATE].dropna().empty:
        st.info("La colonne Date n'est pas exploitable pour cette analyse.")
    else:
        weekday_labels = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}
        weekday_counts = (
            df_filtered[COL_DATE]
            .dropna()
            .dt.weekday
            .value_counts()
            .reindex(range(7), fill_value=0)
            .rename_axis("Jour_num")
            .reset_index(name="Nombre")
        )
        weekday_counts["Jour"] = weekday_counts["Jour_num"].map(weekday_labels)
        fig_weekday = px.bar(
            weekday_counts,
            x="Jour",
            y="Nombre",
            color_discrete_sequence=["#1a936f"],
        )
        fig_weekday.update_layout(showlegend=False)
        st.plotly_chart(style_figure(fig_weekday, height=250), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def render_closure_rate_by_province(df_filtered, top_n):
    st.markdown("<div class='block'><div class='block-title'>Taux de cloture par province</div>", unsafe_allow_html=True)
    df_rate = df_filtered[[COL_PROVINCE_NORM, COL_STATUT]].copy()
    if df_rate.empty:
        st.info("Pas assez de donnees pour calculer le taux de cloture.")
    else:
        df_rate["Cloture"] = df_rate[COL_STATUT].apply(normalize_status) == "cloture"
        province_rate = (
            df_rate.groupby(COL_PROVINCE_NORM, as_index=False)
            .agg(Total=(COL_STATUT, "size"), Clotures=("Cloture", "sum"))
            .sort_values(["Total", COL_PROVINCE_NORM], ascending=[False, True])
            .head(top_n)
        )
        province_rate["Taux_cloture"] = province_rate["Clotures"] / province_rate["Total"] * 100
        province_rate = province_rate.sort_values("Taux_cloture", ascending=True)
        province_rate["Texte"] = province_rate["Taux_cloture"].round(1).astype(str) + "%"

        fig_rate = px.bar(
            province_rate,
            x="Taux_cloture",
            y=COL_PROVINCE_NORM,
            orientation="h",
            text="Texte",
            color="Taux_cloture",
            color_continuous_scale=["#fee2e2", "#fb7185", "#9f1239"],
        )
        fig_rate.update_traces(
            textposition="outside",
            customdata=province_rate[["Total", "Clotures"]],
            hovertemplate="<b>%{y}</b><br>Taux: %{x:.1f}%<br>Total: %{customdata[0]}<br>Clotures: %{customdata[1]}<extra></extra>",
        )
        fig_rate.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_figure(fig_rate, height=250), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def render_resolution_chart(df_filtered, top_n):
    st.markdown("<div class='block'><div class='block-title'>Analyse par resolution</div>", unsafe_allow_html=True)
    if COL_RESOLUTION not in df_filtered.columns:
        st.info("La colonne Resolution n'est pas disponible.")
    else:
        df_resolution = df_filtered[df_filtered[COL_RESOLUTION].fillna("").astype(str).str.strip() != ""]
        if df_resolution.empty:
            st.info("Aucune valeur exploitable dans la colonne Resolution.")
        else:
            resolution_counts = df_resolution[COL_RESOLUTION].value_counts().head(top_n).reset_index()
            resolution_counts.columns = ["Resolution", "Nombre"]
            fig_resolution = px.bar(
                resolution_counts.sort_values("Nombre", ascending=True),
                x="Nombre",
                y="Resolution",
                orientation="h",
                color_discrete_sequence=["#7c3aed"],
            )
            fig_resolution.update_layout(showlegend=False)
            st.plotly_chart(style_figure(fig_resolution, height=250), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def render_pathology_province_heatmap(df_filtered, top_n):
    st.markdown("<div class='block'><div class='block-title'>Pathologie x province</div>", unsafe_allow_html=True)
    if df_filtered.empty:
        st.info("Pas assez de donnees pour ce croisement.")
    else:
        top_pathologies = df_filtered[COL_PATHOLOGIE].value_counts().head(min(top_n, 6)).index
        top_provinces = df_filtered[COL_PROVINCE_NORM].value_counts().head(min(top_n, 6)).index
        df_matrix = df_filtered[
            df_filtered[COL_PATHOLOGIE].isin(top_pathologies)
            & df_filtered[COL_PROVINCE_NORM].isin(top_provinces)
        ]
        if df_matrix.empty:
            st.info("Pas assez de donnees pour construire la matrice.")
        else:
            matrix = pd.crosstab(df_matrix[COL_PATHOLOGIE], df_matrix[COL_PROVINCE_NORM])
            matrix = matrix.reindex(index=top_pathologies, columns=top_provinces, fill_value=0)
            fig_heatmap = px.imshow(
                matrix,
                text_auto=True,
                color_continuous_scale=["#eff6ff", "#60a5fa", "#1d4ed8"],
                aspect="auto",
                labels=dict(x="Province", y="Pathologie", color="Appels"),
            )
            fig_heatmap.update_xaxes(side="top")
            fig_heatmap.update_layout(
                height=320,
                margin=dict(l=8, r=8, t=8, b=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#17233c", size=12),
                coloraxis_colorbar=dict(title="Appels"),
            )
            st.plotly_chart(fig_heatmap, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def render_charts(df_filtered, chart_options):
    top_n = chart_options["top_n"]
    c1, c2, c3 = st.columns([1.18, 1.62, 1.0])

    with c1:
        df_filtered = render_map(df_filtered)

    with c2:
        mid_top_left, mid_top_right = st.columns(2)
        with mid_top_left:
            st.markdown(f"<div class='block'><div class='block-title'>Appels par pathologie (Top {top_n})</div>", unsafe_allow_html=True)
            df_patho = df_filtered[COL_PATHOLOGIE].value_counts().head(top_n).reset_index()
            df_patho.columns = ["Pathologie", "Nombre"]
            fig_patho = px.bar(df_patho, x="Nombre", y="Pathologie", orientation="h", color_discrete_sequence=["#1f7ae0"])
            st.plotly_chart(style_figure(fig_patho, height=220), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        with mid_top_right:
            render_sexe_chart(df_filtered)

        mid_bottom_left, mid_bottom_right = st.columns(2)
        with mid_bottom_left:
            st.markdown(f"<div class='block'><div class='block-title'>Repartition des appels par item (Top {top_n})</div>", unsafe_allow_html=True)
            df_item = df_filtered[COL_ITEM].value_counts().head(top_n).reset_index()
            df_item.columns = ["Item", "Nombre"]
            fig_item = px.bar(df_item, x="Nombre", y="Item", orientation="h", color_discrete_sequence=["#208a3c"])
            st.plotly_chart(style_figure(fig_item, height=220), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        with mid_bottom_right:
            st.markdown("<div class='block'><div class='block-title'>Appels par categorie</div>", unsafe_allow_html=True)
            fig_categorie = px.pie(
                df_filtered,
                names=COL_CATEGORIE,
                hole=0.58,
                color_discrete_sequence=["#e3342f", "#1f7ae0", "#8f99a3", "#16a34a", "#d97706"],
            )
            fig_categorie.update_traces(textinfo="none")
            st.plotly_chart(style_figure(fig_categorie, height=220), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        df_province = df_filtered[COL_PROVINCE_NORM].value_counts().head(top_n).reset_index()
        df_province.columns = ["Province", "Nombre"]
        render_ranking(f"Top {top_n} provinces", df_province, "Province")
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        df_patho_rank = df_filtered[COL_PATHOLOGIE].value_counts().head(top_n).reset_index()
        df_patho_rank.columns = ["Pathologie", "Nombre"]
        render_ranking(f"Top {top_n} pathologies", df_patho_rank, "Pathologie")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    tab_tendance, tab_charge, tab_performance, tab_croisement = st.tabs(
        ["Tendance", "Charge horaire", "Performance", "Croisements"]
    )

    with tab_tendance:
        render_time_trend_chart(df_filtered, chart_options)

    with tab_charge:
        extra_row_1_col_1, extra_row_1_col_2 = st.columns(2)
        with extra_row_1_col_1:
            render_hour_chart(df_filtered)
        with extra_row_1_col_2:
            render_weekday_chart(df_filtered)

    with tab_performance:
        extra_row_2_col_1, extra_row_2_col_2 = st.columns(2)
        with extra_row_2_col_1:
            render_closure_rate_by_province(df_filtered, top_n)
        with extra_row_2_col_2:
            render_resolution_chart(df_filtered, top_n)

    with tab_croisement:
        render_pathology_province_heatmap(df_filtered, top_n)


def render_footer():
    st.markdown(
        """
<div class="footer">
  <div>Pour toute urgence sanitaire, appelez le<br><small>+243 800 000 115</small></div>
  <div>Donnees fiables pour des decisions rapides<br><small>Protegeons nos communautes</small></div>
  <div>Source : Base de donnees Call Center<br><small>COUSP - RDC</small></div>
</div>
""",
        unsafe_allow_html=True,
    )


def main():
    render_css()
    render_header()

    data_source = st.sidebar.selectbox("Source de donnees", ["Excel", "PostgreSQL"], index=0)

    if data_source == "Excel":
        file = st.sidebar.file_uploader("Fichier Excel", type=["xlsx"])
        if file is None:
            st.stop()

        sheet_names = get_excel_sheet_names(file)
        default_sheet = sheet_names[0]
        sheet_upl = st.sidebar.text_input("Nom feuille (si Excel upload)", value=default_sheet)
        selected_sheet = sheet_upl.strip() or default_sheet
        df_source = read_excel_file(file, selected_sheet)
    else:
        st.sidebar.caption("Connexion a une base PostgreSQL")
        host = st.sidebar.text_input("Hote PostgreSQL", value="localhost")
        port = st.sidebar.number_input("Port PostgreSQL", min_value=1, max_value=65535, value=5432, step=1)
        database = st.sidebar.text_input("Base de donnees", value="")
        user = st.sidebar.text_input("Utilisateur", value="")
        password = st.sidebar.text_input("Mot de passe", value="", type="password")
        query_mode = st.sidebar.radio("Mode de lecture", ["Table", "Requete SQL"], index=0)
        table_name = ""
        sql_query = ""

        if query_mode == "Table":
            table_name = st.sidebar.text_input("Nom de la table", value="")
        else:
            sql_query = st.sidebar.text_area("Requete SQL", value="SELECT * FROM call_center", height=120)

        if not all([str(host).strip(), str(database).strip(), str(user).strip()]):
            st.info("Renseigne les parametres PostgreSQL pour charger les donnees.")
            st.stop()

        query = build_postgresql_query(query_mode, table_name, sql_query)
        df_source = read_postgresql_file(host, int(port), database.strip(), user.strip(), password, query)

    df_loaded = prepare_data(df_source)
    df_filtered, chart_options = filter_data(df_loaded)

    render_kpis(df_filtered)
    st.markdown("<br>", unsafe_allow_html=True)
    render_charts(df_filtered, chart_options)
    render_footer()


if __name__ == "__main__":
    main()
