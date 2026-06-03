"""Utilitaires de sélection des sources line list et de chargement PostgreSQL."""

from pathlib import Path
import re
from typing import Optional

import pandas as pd
import streamlit as st

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

from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINE_LIST_DIR = PROJECT_ROOT / "line_list"
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
            st.error("Le nom de table contient des caractères non autorisés.")
            st.stop()
        return f"SELECT * FROM {clean_table_name}"

    clean_query = str(sql_query).strip()
    if not clean_query:
        st.error("Renseigne une requête SQL PostgreSQL.")
        st.stop()
    if not validate_read_only_sql_query(clean_query):
        st.error("La requÃªte SQL doit Ãªtre une requÃªte de lecture unique (`SELECT` ou `WITH ... SELECT`).")
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
        st.error("Le connecteur PostgreSQL n'est pas installé. Ajoute `sqlalchemy` et `psycopg2-binary`.")
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
            engine = create_engine(
                connection_url,
                pool_pre_ping=True,
            )
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
        st.error(f"Impossible de charger les données PostgreSQL : {exc}")
        st.stop()

    df_loaded.columns = df_loaded.columns.astype(str).str.strip()
    return df_loaded
