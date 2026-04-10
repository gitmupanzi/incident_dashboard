import html
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from shapely.geometry import MultiPolygon
from shapely.geometry.polygon import orient

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
    return gpd.read_file(path)


def resolve_provinces_geojson_path():
    for path in [GEOJSON_PROVINCES, GEOJSON_PROVINCES_FALLBACK]:
        if Path(path).exists():
            return path
    return GEOJSON_PROVINCES


def orient_for_plotly(geometry):
    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.geom_type == "Polygon":
        return orient(geometry, sign=1.0)
    if geometry.geom_type == "MultiPolygon":
        return MultiPolygon([orient(polygon, sign=1.0) for polygon in geometry.geoms])
    return geometry


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


def require_columns(df_loaded):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df_loaded.columns]
    if missing_columns:
        st.error("Le fichier Excel ne contient pas toutes les colonnes attendues.")
        st.write("Colonnes manquantes :", ", ".join(missing_columns))
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

    df_for_territoire = df_by_date
    df_for_territoire = apply_multiselect_filter(df_for_territoire, COL_PROVINCE_NORM, province_sel)

    territoire_sel = selected_values("Territoire", sorted_options(df_for_territoire[COL_TERRITOIRE]), "territoire_sel")
    df_for_zone_sante = apply_multiselect_filter(df_for_territoire, COL_TERRITOIRE, territoire_sel)

    zone_sante_sel = optional_selected_values(
        "Zone de santé",
        df_for_zone_sante,
        COL_ZONE_SANTE,
        "zone_sante_sel",
    )
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
    static_map = st.sidebar.checkbox("Carte statique si la carte interactive bloque", value=True)

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
        "static_map": static_map,
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


def build_map_gdf(df_filtered):
    try:
        gdf = load_provinces_geojson(resolve_provinces_geojson_path()).copy()
    except Exception as exc:
        st.error(f"Impossible de charger la carte RDC : {exc}")
        st.stop()

    geo_columns = [column for column in gdf.columns if column.lower() in ["name", "province", "name_1"]]
    if not geo_columns:
        st.error("Impossible d'identifier la colonne province dans le GeoJSON.")
        st.stop()

    geo_col = geo_columns[0]
    gdf["map_id"] = gdf.index.astype(str)
    gdf["prov_norm"] = gdf[geo_col].apply(normalize_province_name)

    data = df_filtered[[COL_PROVINCE_NORM]].copy()
    data = data.rename(columns={COL_PROVINCE_NORM: "prov_norm"})
    data = data.groupby("prov_norm", as_index=False).size()
    data.columns = ["prov_norm", "nb"]

    gdf = gdf.merge(data, on="prov_norm", how="left")
    gdf["nb"] = gdf["nb"].fillna(0)
    gdf["geometry"] = gdf.geometry.apply(orient_for_plotly)

    return gdf, geo_col


def carte_rdc(df_filtered):
    gdf, geo_col = build_map_gdf(df_filtered)

    fig = px.choropleth(
        gdf,
        geojson=gdf.__geo_interface__,
        locations="map_id",
        featureidkey="properties.map_id",
        color="nb",
        color_continuous_scale=[
            [0.0, "#fff7f3"],
            [0.15, "#fde7df"],
            [0.45, "#fca487"],
            [0.75, "#fb5a49"],
            [1.0, "#9b001f"],
        ],
        projection="mercator",
    )

    fig.update_traces(
        customdata=gdf[[geo_col, "prov_norm"]],
        marker_line_color="#d2d2cc",
        marker_line_width=1.0,
        hovertemplate="<b>%{customdata[0]}</b><br>Appels: %{z}<extra></extra>",
    )
    fig.update_geos(
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
    fig.update_layout(
        height=560,
        margin=dict(l=8, r=8, t=4, b=4),
        paper_bgcolor="#fbfaf7",
        plot_bgcolor="#fbfaf7",
        coloraxis_showscale=False,
        dragmode="pan",
    )

    return fig, gdf, geo_col


def carte_rdc_statique(df_filtered):
    gdf, geo_col = build_map_gdf(df_filtered)
    gdf_plot = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(8.5, 6.2), facecolor="#fbfaf7")
    ax.set_facecolor("#fbfaf7")
    gdf_plot.plot(
        column="nb",
        cmap="Reds",
        ax=ax,
        legend=True,
        edgecolor="#d2d2cc",
        linewidth=0.8,
        missing_kwds={"color": "#fff7f3"},
    )

    for _, row in gdf_plot.iterrows():
        if row["nb"] <= 0:
            continue
        point = row.geometry.representative_point()
        ax.text(
            point.x,
            point.y,
            f"{row[geo_col]}\n{int(row['nb'])}",
            ha="center",
            va="center",
            fontsize=6,
            color="#1f2a44",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff", edgecolor="none", alpha=0.75),
        )

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    scale_km = 100
    x_start = x_min + (x_max - x_min) * 0.08
    y_start = y_min + (y_max - y_min) * 0.08
    ax.plot([x_start, x_start + scale_km * 1000], [y_start, y_start], color="#1f2a44", linewidth=1.5)
    ax.text(x_start + scale_km * 500, y_start + (y_max - y_min) * 0.02, f"{scale_km} km", ha="center", fontsize=7)

    ax.set_title("RDC - Appels par province", fontsize=11, fontweight="bold", color="#001b47")
    ax.axis("off")
    fig.tight_layout()
    return fig


def get_selected_map_point(selection_state):
    if not selection_state:
        return None

    selection = selection_state.get("selection", {}) if hasattr(selection_state, "get") else {}
    points = selection.get("points", []) if isinstance(selection, dict) else []
    return points[0] if points else None


def get_clicked_province(point, gdf_map, geo_col):
    customdata = point.get("customdata") or []
    province = customdata[0] if len(customdata) > 0 else None
    province_norm = customdata[1] if len(customdata) > 1 else None

    if province_norm:
        return province, province_norm

    location = point.get("location")
    if location is not None:
        match = gdf_map[gdf_map["map_id"] == str(location)]
        if not match.empty:
            row = match.iloc[0]
            return row[geo_col], row["prov_norm"]

    point_index = point.get("pointIndex", point.get("point_number"))
    if point_index is not None and 0 <= point_index < len(gdf_map):
        row = gdf_map.iloc[point_index]
        return row[geo_col], row["prov_norm"]

    return None, None


def render_map(df_filtered, static_map=False):
    st.markdown("<div class='map-card'><div class='block-title'>Repartition des appels par province</div>", unsafe_allow_html=True)

    if static_map:
        fig_static = carte_rdc_statique(df_filtered)
        st.pyplot(fig_static, width="stretch")
        plt.close(fig_static)
        st.markdown("<div class='hint'>Carte statique activee. Decoche l'option sidebar pour reactiver le clic.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return df_filtered

    fig_map, gdf_map, geo_col = carte_rdc(df_filtered)

    selection_state = st.plotly_chart(
        fig_map,
        width="stretch",
        height=580,
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
        province, province_norm = get_clicked_province(clicked_point, gdf_map, geo_col)
        current_selection = st.session_state.get("province_sel", [])
        if province_norm and current_selection != [province_norm]:
            st.session_state["map_clicked_province"] = province_norm
            st.markdown("<div class='hint'>Filtre actif : {}</div>".format(html.escape(str(province))), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.rerun()

    st.markdown("<div class='hint'>Clique sur une province pour filtrer. Utilise la molette ou la barre d'outils pour zoomer/dezoomer.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    return df_filtered


def style_figure(fig, height=260):
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
        fig_sexe = px.pie(
            df_filtered[df_filtered[COL_SEXE] != ""],
            names=COL_SEXE,
            hole=0.58,
            color_discrete_sequence=["#1f7ae0", "#e3342f", "#16a34a", "#d97706"],
        )
        fig_sexe.update_traces(textinfo="percent+label")
        st.plotly_chart(style_figure(fig_sexe), width="stretch")
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


def render_charts(df_filtered, chart_options):
    top_n = chart_options["top_n"]
    c1, c2, c3 = st.columns([1.25, 1.7, 1.1])

    with c1:
        df_filtered = render_map(df_filtered, static_map=chart_options["static_map"])

    with c2:
        mid_top_left, mid_top_right = st.columns(2)
        with mid_top_left:
            st.markdown(f"<div class='block'><div class='block-title'>Appels par pathologie (Top {top_n})</div>", unsafe_allow_html=True)
            df_patho = df_filtered[COL_PATHOLOGIE].value_counts().head(top_n).reset_index()
            df_patho.columns = ["Pathologie", "Nombre"]
            fig_patho = px.bar(df_patho, x="Nombre", y="Pathologie", orientation="h", color_discrete_sequence=["#1f7ae0"])
            st.plotly_chart(style_figure(fig_patho), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        with mid_top_right:
            render_sexe_chart(df_filtered)

        mid_bottom_left, mid_bottom_right = st.columns(2)
        with mid_bottom_left:
            st.markdown(f"<div class='block'><div class='block-title'>Repartition des appels par item (Top {top_n})</div>", unsafe_allow_html=True)
            df_item = df_filtered[COL_ITEM].value_counts().head(top_n).reset_index()
            df_item.columns = ["Item", "Nombre"]
            fig_item = px.bar(df_item, x="Nombre", y="Item", orientation="h", color_discrete_sequence=["#208a3c"])
            st.plotly_chart(style_figure(fig_item), width="stretch")
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
            st.plotly_chart(style_figure(fig_categorie), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
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
        st.plotly_chart(style_figure(fig_time, height=260), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        df_province = df_filtered[COL_PROVINCE_NORM].value_counts().head(top_n).reset_index()
        df_province.columns = ["Province", "Nombre"]
        render_ranking(f"Top {top_n} provinces", df_province, "Province")
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        df_patho_rank = df_filtered[COL_PATHOLOGIE].value_counts().head(top_n).reset_index()
        df_patho_rank.columns = ["Pathologie", "Nombre"]
        render_ranking(f"Top {top_n} pathologies", df_patho_rank, "Pathologie")


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
