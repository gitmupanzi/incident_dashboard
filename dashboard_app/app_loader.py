"""Utilitaires de selection des sources line list, PostgreSQL et DHIS2 Tracker."""

from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import URL

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    create_engine = None
    text = None
    URL = None

try:
    import psycopg2

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None

from dashboard_app.column_mapping import (
    auto_map_columns,
    build_auto_applied_mapping,
    normalize_column_name,
    rename_dataframe_to_standard,
)
from dashboard_app.colonne_nettoyage import standardiser_nom, standardiser_noms_colonnes
from dashboard_app.domain import DISEASE_SPECS, standardize_ll_by_disease
from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINE_LIST_DIR = PROJECT_ROOT / "line_list"
LINE_LIST_BUNDLE_LABEL = "line_list/"
DEFAULT_RENAME_COLUMNS_FILE = PROJECT_ROOT / "data" / "Rename_columns.xlsx"
DHIS2_DEFAULT_TIMEOUT = 300
DHIS2_DEFAULT_CONNECT_TIMEOUT = 30
DHIS2_DEFAULT_MAX_RETRIES = 2


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


def _normalize_file_hint(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm_key(value))


def guess_preferred_included_file(
    available_files: list[Path],
    disease_key: str,
    default_sheet: str,
) -> Optional[Path]:
    if not available_files:
        return None

    default_sheet_norm = _normalize_file_hint(default_sheet or "")
    disease_label = DISEASE_SPECS.get(disease_key, {}).get("label", disease_key)
    disease_norms = {
        _normalize_file_hint(disease_key),
        _normalize_file_hint(disease_label),
    }

    scored_files: list[tuple[int, Path]] = []
    for path in available_files:
        score = 0
        file_name_norm = _normalize_file_hint(path.stem)
        if default_sheet_norm and default_sheet_norm in file_name_norm:
            score += 4
        if any(d_norm and d_norm in file_name_norm for d_norm in disease_norms):
            score += 2
        if path.suffix.lower() in {".xlsx", ".xls"} and default_sheet:
            try:
                sheets = get_excel_sheet_names_from_path(path)
            except Exception:
                sheets = []
            if default_sheet in sheets:
                score += 6
        scored_files.append((score, path))

    scored_files.sort(key=lambda item: (-item[0], item[1].name.lower()))
    return scored_files[0][1] if scored_files else available_files[0]


@st.cache_data(show_spinner=False)
def _get_excel_sheet_names_from_path_cached(path_str: str, mtime_ns: int) -> list[str]:
    del mtime_ns
    return pd.ExcelFile(path_str).sheet_names


def get_excel_sheet_names_from_path(path: Path) -> list[str]:
    try:
        resolved_path = path.resolve()
        return _get_excel_sheet_names_from_path_cached(
            str(resolved_path),
            resolved_path.stat().st_mtime_ns,
        )
    except Exception as exc:
        st.error(f"Impossible de lire le fichier Excel local : {exc}")
        st.stop()


def validate_table_identifier(identifier: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\\.]*", str(identifier).strip()))


def validate_read_only_sql_query(sql_query: str) -> bool:
    query = str(sql_query or "").strip()
    if not query:
        return False

    query = re.sub(r";+\s*$", "", query)
    if ";" in query:
        return False

    if not re.match(r"^(select|with)\b", query, flags=re.IGNORECASE):
        return False

    forbidden = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "grant",
        "revoke",
        "comment",
        "call",
        "do",
        "copy",
        "vacuum",
        "analyze",
        "refresh",
    ]
    return re.search(r"\b(" + "|".join(forbidden) + r")\b", query, flags=re.IGNORECASE) is None


def build_postgresql_query(query_mode: str, table_name: str, sql_query: str) -> str:
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
    if not validate_read_only_sql_query(clean_query):
        st.error("La requete SQL doit etre une requete de lecture unique (`SELECT` ou `WITH ... SELECT`).")
        st.stop()
    return re.sub(r";+\s*$", "", clean_query)


def read_postgresql_file(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    query: str,
) -> pd.DataFrame:
    if not SQLALCHEMY_AVAILABLE and not PSYCOPG2_AVAILABLE:
        st.error("Le connecteur PostgreSQL n'est pas installe. Ajoute `sqlalchemy` et `psycopg2-binary`.")
        st.stop()

    try:
        if SQLALCHEMY_AVAILABLE:
            connection_url = URL.create(
                "postgresql+psycopg2",
                username=user,
                password=password,
                host=host,
                port=int(port),
                database=database,
            )
            engine = create_engine(connection_url, pool_pre_ping=True)
            try:
                with engine.connect() as connection:
                    df_loaded = pd.read_sql_query(text(query), connection)
            finally:
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

    df_loaded.columns = df_loaded.columns.astype(str).str.strip()
    return df_loaded


def _detect_dhis2_format_from_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("L'URL DHIS2 ne peut pas etre vide.")

    url_lower = url.lower()
    if ".json" in url_lower:
        return "json"
    if ".csv" in url_lower:
        return "csv"
    raise ValueError("Impossible de detecter le format DHIS2 depuis l'URL. Utilise une URL .json ou .csv.")


def _normalize_dhis2_format(format_sortie: Optional[str]) -> Optional[str]:
    if format_sortie is None:
        return None

    normalized = str(format_sortie).strip().lower()
    if normalized not in {"json", "csv"}:
        raise ValueError("Le format DHIS2 doit etre `json` ou `csv`.")
    return normalized


def _resolve_dhis2_format(url: str, format_sortie: Optional[str] = None) -> str:
    format_from_url = _detect_dhis2_format_from_url(url)
    requested = _normalize_dhis2_format(format_sortie)
    if requested and requested != format_from_url:
        raise ValueError(
            f"Format incoherent : l'URL semble etre en `{format_from_url}` mais `format_sortie={requested}` a ete demande."
        )
    return requested or format_from_url


def _normalize_dhis2_timeout(connect_timeout: int, read_timeout: int) -> tuple[float, float]:
    if int(connect_timeout) <= 0 or int(read_timeout) <= 0:
        raise ValueError("Les timeouts DHIS2 doivent etre strictement positifs.")
    return (float(connect_timeout), float(read_timeout))


def _create_dhis2_session(max_retries: int = DHIS2_DEFAULT_MAX_RETRIES) -> requests.Session:
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _validate_dhis2_response(response: requests.Response, format_attendu: str) -> None:
    content_type = (response.headers.get("content-type") or "").lower()
    content = response.text.lstrip()
    preview = content[:500].lower()

    if "text/html" in content_type or preview.startswith("<!doctype html") or preview.startswith("<html"):
        raise ValueError(
            "DHIS2 a renvoye une page HTML au lieu des donnees API. Verifie l'URL, le compte ou la session."
        )

    if format_attendu == "json" and not (
        "json" in content_type or content.startswith("{") or content.startswith("[")
    ):
        raise ValueError("La reponse DHIS2 ne semble pas etre du JSON.")

    if format_attendu == "csv" and (
        "json" in content_type or content.startswith("{") or content.startswith("[")
    ):
        raise ValueError("La reponse DHIS2 semble etre du JSON alors que le format CSV a ete demande.")


def _fetch_dhis2_response(
    url: str,
    username: str,
    password: str,
    connect_timeout: int,
    read_timeout: int,
    max_retries: int,
) -> requests.Response:
    timeout = _normalize_dhis2_timeout(connect_timeout, read_timeout)
    auth = (username, password) if username and password else None
    session = _create_dhis2_session(max_retries=max(0, int(max_retries)))
    try:
        response = session.get(url, timeout=timeout, auth=auth)
        response.raise_for_status()
        return response
    except requests.Timeout as exc:
        raise TimeoutError(
            "Le serveur DHIS2 a depasse le delai de lecture. Essaie un timeout plus grand ou une requete plus legere."
        ) from exc
    finally:
        session.close()


def _normalize_dhis2_json_enrollments(data: dict) -> pd.DataFrame:
    rows = data.get("rows", [])
    headers = data.get("headers", [])
    if not rows:
        return pd.DataFrame()
    if not isinstance(headers, list) or not isinstance(rows, list):
        raise ValueError("La structure JSON DHIS2 est invalide : `headers` et `rows` sont attendus.")

    column_names: list[str] = []
    for header in headers:
        if isinstance(header, dict):
            column_names.append(header.get("column") or header.get("name") or header.get("label"))
        else:
            column_names.append(str(header))

    if not column_names or any(name is None for name in column_names):
        raise ValueError("Impossible de determiner les noms de colonnes DHIS2.")

    return pd.DataFrame(rows, columns=column_names)


def _clean_dhis2_org_unit(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.replace("SantÃƒÂ©", "Sante").replace("SantÃƒâ€°", "Sante")
    text = re.sub(r"^[A-Za-z]{2,4}\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+Province$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+Zone\s+de\s+Sant\S*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+Aire\s+de\s+Sant\S*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _extract_dhis2_code_and_name(value) -> tuple[Optional[str], Optional[str]]:
    if value is None or pd.isna(value):
        return (None, None)

    text = str(value).strip()
    if not text:
        return (None, None)

    match = re.match(r"^(?P<code>[A-Za-z]{2,4})\s+(?P<name>.+)$", text)
    if match:
        return (match.group("code").strip(), _clean_dhis2_org_unit(match.group("name")))
    return (None, _clean_dhis2_org_unit(text))


def _find_dhis2_hierarchy_column(
    df: pd.DataFrame,
    colonne_hierarchy: Optional[str] = None,
) -> Optional[str]:
    if colonne_hierarchy:
        return colonne_hierarchy if colonne_hierarchy in df.columns else None

    for candidate in (
        "Organisation unit name hierarchy",
        "Organisation_unit_name_hierarchy",
        "ounamehierarchy",
    ):
        if candidate in df.columns:
            return candidate
    return None


def _add_dhis2_notification_columns(
    df: pd.DataFrame,
    colonne_hierarchy: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    hierarchy_col = _find_dhis2_hierarchy_column(out, colonne_hierarchy=colonne_hierarchy)
    if hierarchy_col is None:
        return out

    def _extract_row(hierarchy_value) -> pd.Series:
        if hierarchy_value is None or pd.isna(hierarchy_value):
            return pd.Series([None, None, None, None])

        parts = [part.strip() for part in str(hierarchy_value).split("/") if str(part).strip()]
        code_province, province = _extract_dhis2_code_and_name(parts[1]) if len(parts) > 1 else (None, None)
        zone = _clean_dhis2_org_unit(parts[2]) if len(parts) > 2 else None
        aire = _clean_dhis2_org_unit(parts[3]) if len(parts) > 3 else None
        return pd.Series([code_province, province, zone, aire])

    out[
        [
            "Code_province",
            "Province_notification",
            "Zone_de_sante_notification",
            "Aire_de_sante_notification",
        ]
    ] = out[hierarchy_col].apply(_extract_row)
    return out


@st.cache_data(show_spinner=False)
def _load_reference_rename_mapping(path_str: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    return pd.read_excel(path_str)


def _build_reference_rename_map(
    mapping_file: Path,
    disease_key: Optional[str] = None,
) -> dict[str, str]:
    resolved = mapping_file.resolve()
    if not resolved.exists():
        return {}

    try:
        mapping_df = _load_reference_rename_mapping(str(resolved), resolved.stat().st_mtime_ns)
    except Exception:
        return {}

    required_columns = {"Original", "Renamed"}
    if not required_columns.issubset(set(mapping_df.columns)):
        return {}

    work = mapping_df.copy()
    work = work.dropna(subset=["Original", "Renamed"])
    work["Original"] = work["Original"].astype(str).str.strip()
    work["Renamed"] = work["Renamed"].astype(str).str.strip()
    work = work[(work["Original"] != "") & (work["Renamed"] != "")]

    rename_map: dict[str, str] = {}
    for _, row in work.iterrows():
        rename_map[str(row["Original"]).strip()] = str(row["Renamed"]).strip()
    return rename_map


def _rename_columns_from_reference(
    df: pd.DataFrame,
    mapping_file: Path = DEFAULT_RENAME_COLUMNS_FILE,
    disease_key: Optional[str] = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    exact_map = _build_reference_rename_map(mapping_file, disease_key=disease_key)
    if not exact_map:
        return df.copy()

    normalized_map = {}
    for source_name, target_name in exact_map.items():
        normalized_source = standardiser_nom(source_name)
        if normalized_source:
            normalized_map[normalized_source] = target_name

    rename_map: dict[str, str] = {}
    for col in df.columns:
        col_text = str(col).strip()
        if col_text in exact_map:
            rename_map[col] = exact_map[col_text]
            continue

        normalized_col = standardiser_nom(col_text)
        if normalized_col in normalized_map:
            rename_map[col] = normalized_map[normalized_col]

    return df.rename(columns=rename_map) if rename_map else df.copy()


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if not df.columns.duplicated().any():
        return df.copy()

    unique_columns = list(dict.fromkeys(df.columns.tolist()))
    merged_series: list[pd.Series] = []

    for col in unique_columns:
        selection = df.loc[:, df.columns == col]
        if isinstance(selection, pd.Series) or selection.shape[1] == 1:
            serie = selection if isinstance(selection, pd.Series) else selection.iloc[:, 0].copy()
        else:
            serie = selection.iloc[:, 0].copy()
            for idx in range(1, selection.shape[1]):
                serie = serie.combine_first(selection.iloc[:, idx])

        serie.name = col
        merged_series.append(serie)

    return pd.concat(merged_series, axis=1)


def _build_similar_merge_key(column_name: object) -> str:
    normalized = standardiser_nom(column_name)
    if not normalized:
        return ""

    # Aligne les variantes generees par Excel/pandas : `.1`, `_1`, `_2`, etc.
    normalized = re.sub(r"(?:_\d+)+$", "", normalized)
    return normalized.strip("_")


def _coalesce_similar_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    groups: dict[str, list[object]] = {}
    for col in df.columns.tolist():
        key = _build_similar_merge_key(col)
        if not key:
            key = str(col)
        groups.setdefault(key, []).append(col)

    if not any(len(cols) > 1 for cols in groups.values()):
        return df.copy()

    out = pd.DataFrame(index=df.index)
    consumed: set[object] = set()

    for col in df.columns.tolist():
        if col in consumed:
            continue

        key = _build_similar_merge_key(col) or str(col)
        similar_cols = groups.get(key, [col])
        consumed.update(similar_cols)

        selection = df.loc[:, similar_cols]
        if isinstance(selection, pd.Series) or selection.shape[1] == 1:
            serie = selection if isinstance(selection, pd.Series) else selection.iloc[:, 0].copy()
        else:
            serie = selection.iloc[:, 0].copy()
            for idx in range(1, selection.shape[1]):
                serie = serie.combine_first(selection.iloc[:, idx])

        preferred_name = next(
            (
                str(name)
                for name in similar_cols
                if _build_similar_merge_key(name) == str(name)
            ),
            str(similar_cols[0]),
        )
        serie.name = preferred_name
        out[preferred_name] = serie

    return out


def _standardize_dhis2_columns(
    df: pd.DataFrame,
    disease_key: Optional[str] = None,
    mapping_file: Path = DEFAULT_RENAME_COLUMNS_FILE,
) -> pd.DataFrame:
    out = standardiser_noms_colonnes(df.copy(), mapping_file=None)
    out = _rename_columns_from_reference(out, mapping_file=mapping_file, disease_key=disease_key)
    out = _coalesce_duplicate_columns(out)
    out = _coalesce_similar_normalized_columns(out)
    if out.empty:
        return out

    auto_mapping, auto_metadata = auto_map_columns(out.columns)
    reliable_mapping = build_auto_applied_mapping(auto_metadata, include_derived=False)
    reliable_mapping.update({standard_name: source for standard_name, source in auto_mapping.items() if source})
    out = rename_dataframe_to_standard(out, reliable_mapping, keep_unmapped_columns=True)

    if disease_key:
        out = standardize_ll_by_disease(out, str(disease_key).strip())
        out = _coalesce_duplicate_columns(out)
        out = _coalesce_similar_normalized_columns(out)

    return out


def read_dhis2_tracker_file(
    url: str,
    username: str,
    password: str,
    format_sortie: Optional[str] = "json",
    connect_timeout: int = DHIS2_DEFAULT_CONNECT_TIMEOUT,
    read_timeout: int = DHIS2_DEFAULT_TIMEOUT,
    max_retries: int = DHIS2_DEFAULT_MAX_RETRIES,
    ajouter_localisation_notification: bool = True,
    renommer_variable: bool = True,
    variables_brute: bool = False,
    disease_key: Optional[str] = None,
    mapping_file: Path = DEFAULT_RENAME_COLUMNS_FILE,
) -> pd.DataFrame:
    try:
        format_final = _resolve_dhis2_format(url=url, format_sortie=format_sortie)
        response = _fetch_dhis2_response(
            url=url,
            username=username,
            password=password,
            connect_timeout=int(connect_timeout),
            read_timeout=int(read_timeout),
            max_retries=int(max_retries),
        )
        _validate_dhis2_response(response, format_final)

        if format_final == "csv":
            df_loaded = pd.read_csv(StringIO(response.text))
        else:
            df_loaded = _normalize_dhis2_json_enrollments(response.json())

        if ajouter_localisation_notification:
            df_loaded = _add_dhis2_notification_columns(df_loaded)

        if not variables_brute and renommer_variable:
            df_loaded = _standardize_dhis2_columns(
                df_loaded,
                disease_key=disease_key,
                mapping_file=mapping_file,
            )

        timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
        df_loaded["Provenance"] = "DHIS2"
        df_loaded["Date_telechargement"] = timestamp
    except Exception as exc:
        st.error(f"Impossible de charger les donnees DHIS2 Tracker : {exc}")
        st.stop()

    if not isinstance(df_loaded, pd.DataFrame):
        st.error("Le chargement DHIS2 Tracker n'a pas retourne de DataFrame pandas.")
        st.stop()

    df_loaded.columns = df_loaded.columns.astype(str).str.strip()
    return df_loaded
