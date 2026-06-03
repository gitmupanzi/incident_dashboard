"""Utilitaires de mapping des colonnes hétérogènes vers les standards du dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from difflib import SequenceMatcher
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from dashboard_app.core import _normalize_name, _strip_accents
from dashboard_app.domain import _to_dt

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - repli conservé pour les environnements sans rapidfuzz
    fuzz = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_DIR = PROJECT_ROOT / "data" / "mappings"
DEFAULT_CONFIDENCE_THRESHOLD = 85
AUTO_APPLY_CONFIDENCE_THRESHOLD = 90

SOURCE_COLUMNS: dict[str, dict[str, Any]] = {
    "Date_notification": {"role": "Repere temporel principal de notification", "required": False, "level": "important"},
    "Date_debut_maladie": {"role": "Date de debut des symptomes ou de la maladie", "required": False, "level": "important"},
    "Province_notification": {"role": "Province de notification", "required": True, "level": "critical"},
    "Zone_de_sante_notification": {"role": "Zone de sante de notification", "required": True, "level": "critical"},
    "Aire_de_sante_notification": {"role": "Aire de sante de notification", "required": False, "level": "optional"},
    "Sexe": {"role": "Sexe du cas", "required": False, "level": "optional"},
    "Age": {"role": "Age brut du cas", "required": False, "level": "important"},
    "Unite_age": {"role": "Unite utilisee pour l'age", "required": False, "level": "important"},
    "Issue": {"role": "Issue du cas ou outcome", "required": False, "level": "optional"},
    "Classification_investigation": {"role": "Classification d'investigation du cas", "required": False, "level": "optional"},
    "Classification_finale": {"role": "Classification finale du cas", "required": False, "level": "optional"},
    "Investigation": {"role": "Statut d'investigation du cas", "required": False, "level": "optional"},
    "Prelevement": {"role": "Prelevement ou echantillon realise", "required": False, "level": "optional"},
    "TDR_realise": {"role": "Statut de realisation du TDR", "required": False, "level": "optional"},
    "TDR_Resultat": {"role": "Resultat du TDR", "required": False, "level": "optional"},
    "Resultat_labo": {"role": "Resultat de laboratoire", "required": False, "level": "optional"},
    "Hospitalisation": {"role": "Statut d'hospitalisation", "required": False, "level": "optional"},
    "Date_prelevement": {"role": "Date de prelevement", "required": False, "level": "optional"},
    "Date_resultat": {"role": "Date de resultat", "required": False, "level": "optional"},
    "Date_reception_labo": {"role": "Date de reception au laboratoire", "required": False, "level": "optional"},
    "Date_issue": {"role": "Date d'issue, de deces ou de sortie", "required": False, "level": "optional"},
    "Nom_complet": {"role": "Nom complet ou identifiant du cas", "required": False, "level": "optional"},
    "N_labo": {"role": "Numero ou identifiant laboratoire", "required": False, "level": "optional"},
    "Nom_laboratoire": {"role": "Nom du laboratoire", "required": False, "level": "optional"},
    "Type_de_prelevement": {"role": "Type de prelevement ou specimen", "required": False, "level": "optional"},
    "Nombre_dose_recues": {"role": "Nombre de doses vaccinales recues", "required": False, "level": "optional"},
    "Date_derniere_vaccination": {"role": "Date de la derniere vaccination connue", "required": False, "level": "optional"},
}

DERIVED_COLUMNS: dict[str, dict[str, Any]] = {
    "Semaine_epid": {
        "role": "Libelle standard de semaine epidemiologique",
        "type": "derived_temporal",
        "depends_on_any": ["Date_notification", "Date_debut_maladie"],
        "fallback": ["Semaine_epid", ["Annee_epid", "Num_semaine_epid"]],
    },
    "Num_semaine_epid": {
        "role": "Numero de semaine epidemiologique",
        "type": "derived_temporal",
        "depends_on_any": ["Date_notification", "Date_debut_maladie"],
        "fallback": [["Annee_epid", "Num_semaine_epid"]],
    },
    "Annee_epid": {
        "role": "Annee epidemiologique",
        "type": "derived_temporal",
        "depends_on_any": ["Date_notification", "Date_debut_maladie"],
        "fallback": [["Annee_epid", "Num_semaine_epid"]],
    },
    "Age_en_ans": {
        "role": "Age converti en annees",
        "type": "derived_age",
        "depends_on_all": ["Age", "Unite_age"],
        "fallback": ["Age_en_ans"],
    },
    "Tranche_age": {
        "role": "Tranche d'age calculee",
        "type": "derived_age",
        "depends_on_all": ["Age", "Unite_age"],
        "fallback": ["Tranche_age", "Age_en_ans"],
    },
}

CRITICAL_COLUMNS = [
    name for name, meta in SOURCE_COLUMNS.items()
    if meta.get("level") == "critical"
]

IMPORTANT_COLUMNS = [
    name for name, meta in SOURCE_COLUMNS.items()
    if meta.get("level") == "important"
]

OPTIONAL_COLUMNS = [
    name for name, meta in SOURCE_COLUMNS.items()
    if meta.get("level") == "optional"
]

STANDARD_COLUMNS: dict[str, dict[str, Any]] = {
    **{name: {**meta, "derived": False} for name, meta in SOURCE_COLUMNS.items()},
    **{name: {**meta, "derived": True, "required": False} for name, meta in DERIVED_COLUMNS.items()},
}

COLUMN_VARIANTS: dict[str, list[str]] = {
    "Date_notification": [
        "date_notif",
        "date notification",
        "date_consultation",
        "date_rep",
        "daterep",
        "date_rapportage",
        "date_de_notification",
        "date_notification_cas",
    ],
    "Date_debut_maladie": [
        "date_debut",
        "date_onset",
        "date_debut_symptomes",
        "date_apparition_signes",
        "date_symptomes",
    ],
    "Semaine_epid": [
        "semaine_epid",
        "semaine epi",
        "epi_week_label",
        "year_week",
        "yw",
    ],
    "Num_semaine_epid": [
        "num_semaine_epid",
        "numsem",
        "num_sem",
        "week",
        "epi_week",
        "numero semaine",
    ],
    "Annee_epid": [
        "annee_epid",
        "annee",
        "year",
        "epi_year",
    ],
    "Province_notification": [
        "province",
        "prov",
        "div_prov",
        "province_notif",
        "province notification",
    ],
    "Zone_de_sante_notification": [
        "zs",
        "zone_sante",
        "zone de sante",
        "zone de santé",
        "district",
        "zone_sante_notification",
        "zone notification",
    ],
    "Aire_de_sante_notification": [
        "as",
        "aire_sante",
        "aire de sante",
        "aire de santé",
        "aire_sante_notification",
        "health_area",
    ],
    "Sexe": [
        "sexe_cas",
        "sex",
        "gender",
        "genre",
    ],
    "Age": [
        "age_cas",
        "age_years",
        "age_annee",
        "age_annees",
        "age_value",
    ],
    "Age_en_ans": [
        "age_en_ans",
        "age_years_clean",
        "age_annees_calcule",
    ],
    "Unite_age": [
        "age_unite",
        "age_unit",
        "unite_age",
        "unite",
        "uom_age",
    ],
    "Issue": [
        "evolution",
        "outcome",
        "issue",
        "statut_sortie",
        "etat_sortie_malade",
        "statut_a_l_arrivee",
    ],
    "Classification_investigation": [
        "classification_investigation",
        "classification investigation",
        "classification_investig",
    ],
    "Classification_finale": [
        "status_cas",
        "statut_cas",
        "classification",
        "classif",
        "classification_finale_du_cas",
    ],
    "Investigation": [
        "investigation",
        "investigated",
        "cas_investigue",
        "cas_investiguee",
        "statut_investigation",
    ],
    "Prelevement": [
        "echantillon_preleve",
        "prelevement_realise",
        "sample_collected",
        "prelevement_investigation",
        "prelevement_realise_au_moment_de_investigation",
        "prelevement_sang",
        "prelevement_urine",
        "prelevement_respiratoire",
        "autre_prelevement",
    ],
    "TDR_realise": [
        "tdr",
        "tdr_realise",
        "test_rapide_realise",
        "rdt_done",
    ],
    "TDR_Resultat": [
        "tdr_resultat",
        "resultat_tdr",
        "tdr_result",
    ],
    "Resultat_labo": [
        "resultats_labo",
        "resultat_labo",
        "Resultat_final_labo",
        "resultat_final_labo",
        "lab_result",
        "pcr_result",
        "quel_est_le_resultats",
        "resultat_final_opx",
        "resultat_igm",
        "resultat_igm_rubeole",
        "resultat_pcr_labo_national",
        "resultat_machd_labo_national",
    ],
    "Hospitalisation": [
        "hospitalisation",
        "hospit",
        "hospitalized",
        "datehospit",
    ],
    "Date_prelevement": [
        "date_prelevement",
        "date_prelevement_clean",
        "date_prelev",
        "date_prelevement_echantillon",
    ],
    "Date_resultat": [
        "date_resultat",
        "date_reception_resultat",
        "date_analyse",
        "date_result",
        "date_envoi_resultat",
        "date_partage_resultat_pcr",
        "date_partage_resultat_tdr_surveillance_epi",
        "date_partage_resultat_machd_surveillance_epi",
        "date_resultat_igm_labo_national",
    ],
    "Date_reception_labo": [
        "date_reception_labo",
        "date_de_reception",
        "date_reception_echantillon",
        "date_reception",
        "date_reception_echantillon_labo",
    ],
    "Date_issue": [
        "date_issue",
        "date_deces",
        "date_sortie_au_ct",
        "date_de_guerie",
        "date_sortie",
    ],
    "Nom_complet": [
        "nom_complet",
        "nom_complet_cas_suspect",
        "nom_cas",
        "patient_name",
        "nom complet",
    ],
    "N_labo": [
        "n_labo",
        "numero_labo",
        "num_labo",
        "lab_id",
        "identifiant_labo",
    ],
    "Nom_laboratoire": [
        "nom_laboratoire",
        "laboratoire",
        "lab_name",
        "nom labo",
    ],
    "Type_de_prelevement": [
        "type_de_prelevement",
        "type_prelevement",
        "specimen_type",
        "sample_type",
        "nature_prelevement",
    ],
    "Nombre_dose_recues": [
        "nombre_dose_recues",
        "nombre_doses_vaccin",
        "nombre_dose",
        "doses_recues",
        "nb_doses",
    ],
    "Date_derniere_vaccination": [
        "date_derniere_vaccination",
        "date_derniere_dose",
        "last_vaccination_date",
        "date_vaccination",
    ],
    "Tranche_age": [
        "tranche_age",
        "age_group",
        "agegroup",
        "age_group2",
    ],
}


def normalize_column_name(name: Any) -> str:
    """Retourne un nom de colonne normalisé pour un appariement robuste."""
    if name is None:
        return ""
    return _normalize_name(str(name))


def get_column_level(column_name: str) -> str:
    if column_name in SOURCE_COLUMNS:
        level = SOURCE_COLUMNS[column_name].get("level")
        if level == "critical":
            return "Critique"
        if level == "important":
            return "Importante"
        if level == "optional":
            return "Optionnelle"
    if column_name in DERIVED_COLUMNS:
        return "Derivee"
    return "Autre"


def _similarity(left: str, right: str) -> int:
    if not left or not right:
        return 0
    if fuzz is not None:
        return int(max(fuzz.ratio(left, right), fuzz.token_sort_ratio(left, right)))
    return int(SequenceMatcher(None, left, right).ratio() * 100)


def _mapping_has(mapping: dict[str, Optional[str]], column_name: str) -> bool:
    return bool(mapping.get(column_name))


def _spec_is_satisfied(mapping: dict[str, Optional[str]], spec: Any) -> bool:
    if isinstance(spec, str):
        return _mapping_has(mapping, spec)
    if isinstance(spec, (list, tuple, set)):
        return all(_mapping_has(mapping, col) for col in spec)
    return False


def _build_empty_metadata(method: str = "not_mapped") -> dict[str, Any]:
    return {
        "source_column": None,
        "confidence": 0,
        "method": method,
        "accepted": False,
    }


def describe_mapping_candidate(
    standard_name: str,
    source_column: Any,
    threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Décrit une colonne source candidate pour une variable standard."""
    if source_column is None:
        return _build_empty_metadata()

    source_text = str(source_column).strip()
    if not source_text:
        return _build_empty_metadata()

    aliases = COLUMN_VARIANTS.get(standard_name, [])
    normalized_source = normalize_column_name(source_text)
    normalized_standard = normalize_column_name(standard_name)
    normalized_aliases = {alias: normalize_column_name(alias) for alias in aliases}

    if source_text == standard_name:
        return {
            "source_column": source_column,
            "confidence": 100,
            "method": "exact_match",
            "accepted": True,
        }

    if source_text in aliases or source_text.casefold() in {str(alias).casefold() for alias in aliases}:
        return {
            "source_column": source_column,
            "confidence": 100,
            "method": "variant_match",
            "accepted": True,
        }

    if normalized_source == normalized_standard:
        return {
            "source_column": source_column,
            "confidence": 98,
            "method": "normalized_match",
            "accepted": True,
        }

    if normalized_source in normalized_aliases.values():
        return {
            "source_column": source_column,
            "confidence": 95,
            "method": "variant_match",
            "accepted": True,
        }

    best_score = max(
        [_similarity(normalized_source, normalized_standard)]
        + [_similarity(normalized_source, alias_norm) for alias_norm in normalized_aliases.values()],
        default=0,
    )
    return {
        "source_column": source_column,
        "confidence": int(best_score),
        "method": "fuzzy_match" if best_score > 0 else "not_mapped",
        "accepted": int(best_score) >= int(threshold),
    }


def resolve_mapping_selection_metadata(
    standard_name: str,
    selected_source: Any,
    auto_metadata: dict[str, dict[str, Any]],
    threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Décrit la source actuellement sélectionnée pour une variable standard."""
    if selected_source is None:
        return _build_empty_metadata()

    suggested = auto_metadata.get(standard_name, _build_empty_metadata())
    if suggested.get("source_column") == selected_source and suggested.get("accepted"):
        return dict(suggested)

    manual_meta = describe_mapping_candidate(standard_name, selected_source, threshold=threshold)
    manual_meta["method"] = "manual"
    manual_meta["accepted"] = True
    return manual_meta


def auto_map_columns(
    raw_columns: Iterable[Any],
    threshold: int = AUTO_APPLY_CONFIDENCE_THRESHOLD,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Propose des appariements source -> standard via matching exact, normalisé, par variantes et fuzzy."""
    raw_list = list(raw_columns)
    mapping: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    used_sources: set[Any] = set()

    for standard_name in STANDARD_COLUMNS:
        best_meta = _build_empty_metadata()
        best_source = None

        for raw_col in raw_list:
            if raw_col in used_sources:
                continue
            candidate_meta = describe_mapping_candidate(standard_name, raw_col, threshold=threshold)
            if (
                candidate_meta["confidence"] > best_meta["confidence"]
                or (
                    candidate_meta["confidence"] == best_meta["confidence"]
                    and bool(candidate_meta["accepted"])
                    and not bool(best_meta["accepted"])
                )
            ):
                best_meta = candidate_meta
                best_source = raw_col

        if best_source is not None and best_meta["confidence"] > 0:
            best_meta = dict(best_meta)
            best_meta["source_column"] = best_source
            metadata[standard_name] = best_meta
            if best_meta.get("accepted"):
                mapping[standard_name] = best_source
                used_sources.add(best_source)
        else:
            metadata[standard_name] = _build_empty_metadata()

    return mapping, metadata


def build_auto_applied_mapping(
    auto_metadata: dict[str, dict[str, Any]],
    threshold: int = AUTO_APPLY_CONFIDENCE_THRESHOLD,
    include_derived: bool = False,
) -> dict[str, Any]:
    """Retourne uniquement les mappings qui doivent préremplir automatiquement l'interface."""
    mapping: dict[str, Any] = {}
    for standard_name, meta in auto_metadata.items():
        if standard_name in DERIVED_COLUMNS and not include_derived:
            continue
        source_column = meta.get("source_column")
        confidence = int(meta.get("confidence", 0) or 0)
        if source_column is not None and confidence >= int(threshold):
            mapping[standard_name] = source_column
    return mapping


def apply_auto_prefill_to_selection_state(
    current_state: dict[str, Any],
    auto_prefill_mapping: dict[str, Any],
    placeholder: str,
) -> dict[str, Any]:
    """Remplit seulement les sélections manquantes ou temporaires à partir de suggestions très fiables."""
    updated_state = dict(current_state)
    for standard_name, source_column in auto_prefill_mapping.items():
        current_value = updated_state.get(standard_name, placeholder)
        if current_value in {None, "", placeholder}:
            updated_state[standard_name] = source_column
    return updated_state


def build_mapping_warnings(mapping: dict[str, Optional[str]]) -> list[str]:
    warnings: list[str] = []
    for column_name in IMPORTANT_COLUMNS:
        if column_name in DERIVED_COLUMNS:
            continue
        if not _mapping_has(mapping, column_name):
            warnings.append(f"Colonne importante non associee : {column_name}")
    return warnings


def validate_mapping(
    mapping: dict[str, Optional[str]],
    required_columns: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Valide les sources requises et les groupes alternatifs pour le temps et l'âge."""
    errors: list[str] = []

    assigned_sources = [source for source in mapping.values() if source]
    duplicate_sources = sorted({source for source in assigned_sources if assigned_sources.count(source) > 1})
    if duplicate_sources:
        errors.append(
            "Une meme colonne source est assignee plusieurs fois : "
            + ", ".join(str(source) for source in duplicate_sources)
        )

    for standard_name in CRITICAL_COLUMNS:
        if not _mapping_has(mapping, standard_name):
            errors.append(f"Colonne critique non associee : {standard_name}")

    for column_name in required_columns or []:
        if not _mapping_has(mapping, column_name):
            errors.append(f"Colonne requise non associee : {column_name}")

    time_group_valid = any(
        [
            _mapping_has(mapping, "Date_notification"),
            _mapping_has(mapping, "Date_debut_maladie"),
            _mapping_has(mapping, "Semaine_epid"),
            _mapping_has(mapping, "Annee_epid") and _mapping_has(mapping, "Num_semaine_epid"),
        ]
    )
    if not time_group_valid:
        errors.append(
            "Renseigne au moins une information temporelle exploitable : Date_notification, "
            "Date_debut_maladie, Semaine_epid, ou le couple Annee_epid + Num_semaine_epid."
        )

    age_group_valid = any(
        [
            _mapping_has(mapping, "Age") and _mapping_has(mapping, "Unite_age"),
            _mapping_has(mapping, "Age_en_ans"),
            _mapping_has(mapping, "Tranche_age"),
        ]
    )
    if not age_group_valid:
        errors.append(
            "Renseigne au moins une information d'age exploitable : Age + Unite_age, Age_en_ans, ou Tranche_age."
        )

    for derived_name, meta in DERIVED_COLUMNS.items():
        if _mapping_has(mapping, derived_name):
            continue

        derivable = False
        if meta.get("depends_on_any") and any(_mapping_has(mapping, col) for col in meta["depends_on_any"]):
            derivable = True
        if meta.get("depends_on_all") and all(_mapping_has(mapping, col) for col in meta["depends_on_all"]):
            derivable = True
        if any(_spec_is_satisfied(mapping, fallback) for fallback in meta.get("fallback", [])):
            derivable = True

        if derived_name in {"Semaine_epid", "Num_semaine_epid", "Annee_epid"} and not time_group_valid:
            errors.append(f"Impossible d'obtenir la colonne derivee : {derived_name}")
        if derived_name in {"Age_en_ans", "Tranche_age"} and not age_group_valid:
            errors.append(f"Impossible d'obtenir la colonne derivee : {derived_name}")
        if not derivable and derived_name not in {"Semaine_epid", "Num_semaine_epid", "Annee_epid", "Age_en_ans", "Tranche_age"}:
            errors.append(f"Impossible d'obtenir la colonne derivee : {derived_name}")

    return (len(errors) == 0), errors


def rename_dataframe_to_standard(
    df: pd.DataFrame,
    mapping: dict[str, Optional[str]],
    keep_unmapped_columns: bool = True,
) -> pd.DataFrame:
    """Renomme les colonnes source vers les noms standards sans écraser les colonnes standards existantes."""
    out = df.copy()
    rename_map: dict[str, str] = {}
    drop_columns: list[str] = []
    mapped_sources = {source for source in mapping.values() if source in out.columns}

    for standard_name, source_name in mapping.items():
        if not source_name or source_name not in out.columns:
            continue
        if source_name == standard_name:
            continue

        if standard_name in out.columns:
            out[standard_name] = out[standard_name].combine_first(out[source_name])
            drop_columns.append(source_name)
            continue

        rename_map[source_name] = standard_name

    if rename_map:
        out = out.rename(columns=rename_map)
    if drop_columns:
        out = out.drop(columns=drop_columns, errors="ignore")

    if not keep_unmapped_columns:
        kept_columns = set(STANDARD_COLUMNS).union({col for col in out.columns if col not in mapped_sources})
        out = out[[col for col in out.columns if col in kept_columns]]

    return out


def _parse_semaine_epid_series(week_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    series = week_series.astype("string").str.strip().str.upper()
    series = series.str.replace("SE", "S", regex=False)
    series = series.str.replace(r"\s+", "", regex=True)

    year_out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    week_out = pd.Series(pd.NA, index=series.index, dtype="Int64")

    patterns = [
        r"^(?P<year>\d{4})[-_/]?[SW](?P<week>\d{1,2})$",
        r"^[SW](?P<week>\d{1,2})[-_/]?(?P<year>\d{4})$",
        r"^(?P<year>\d{4})[-_/](?P<week>\d{1,2})$",
    ]

    for pattern in patterns:
        extracted = series.str.extract(pattern)
        if extracted.empty:
            continue
        year_vals = pd.to_numeric(extracted.get("year"), errors="coerce").astype("Int64")
        week_vals = pd.to_numeric(extracted.get("week"), errors="coerce").astype("Int64")
        year_out = year_out.fillna(year_vals)
        week_out = week_out.fillna(week_vals)

    valid_week = week_out.between(1, 53, inclusive="both")
    week_out = week_out.where(valid_week, pd.NA)
    year_out = year_out.where(week_out.notna(), pd.NA)
    return year_out, week_out


def _normalize_age_units(unit_series: pd.Series) -> pd.Series:
    out = unit_series.astype("string").str.strip().str.lower()
    out = out.apply(lambda value: _strip_accents(value) if pd.notna(value) else value)
    return out.replace(
        {
            "a": "ans",
            "an": "ans",
            "ans": "ans",
            "annee": "ans",
            "annees": "ans",
            "annees(s)": "ans",
            "year": "ans",
            "years": "ans",
            "yr": "ans",
            "yrs": "ans",
            "y": "ans",
            "m": "mois",
            "mo": "mois",
            "mos": "mois",
            "mois": "mois",
            "month": "mois",
            "months": "mois",
            "s": "semaines",
            "sem": "semaines",
            "semaine": "semaines",
            "semaines": "semaines",
            "week": "semaines",
            "weeks": "semaines",
            "w": "semaines",
            "j": "jours",
            "jr": "jours",
            "jour": "jours",
            "jours": "jours",
            "day": "jours",
            "days": "jours",
            "d": "jours",
        }
    )


def _compute_age_years(age_series: pd.Series, unit_series: pd.Series) -> pd.Series:
    age = pd.to_numeric(age_series, errors="coerce")
    unit = _normalize_age_units(unit_series)
    age_years = np.where(
        unit.eq("ans"),
        age,
        np.where(
            unit.eq("mois"),
            age / 12.0,
            np.where(
                unit.eq("semaines"),
                age / 52.0,
                np.where(unit.eq("jours"), age / 365.25, np.nan),
            ),
        ),
    )
    out = pd.Series(age_years, index=age.index, dtype="float64")
    return out.where((out >= 0) & (out <= 120), np.nan)


def _compute_age_group(age_years: pd.Series) -> pd.Series:
    a = pd.to_numeric(age_years, errors="coerce")
    conds = [
        a.notna() & (a < (1 / 12)),
        a.notna() & (a >= (1 / 12)) & (a < 5),
        a.notna() & (a >= 5) & (a < 15),
        a.notna() & (a >= 15),
    ]
    labels = ["<1 mois", "1-59 mois", "5-14 ans", ">=15 ans"]
    return pd.Series(np.select(conds, labels, default=pd.NA), index=a.index, dtype="object")


def add_derived_columns_after_mapping(
    df: pd.DataFrame,
    replace_existing: bool = False,
    return_info: bool = False,
) -> Any:
    """Alimente les colonnes dérivées épidémiologiques et d'âge après renommage des sources."""
    out = df.copy()
    derived_info: dict[str, Any] = {
        "weeks_calculated": 0,
        "age_years_calculated": 0,
        "age_groups_calculated": 0,
    }

    before_week_non_null = int(out["Semaine_epid"].notna().sum()) if "Semaine_epid" in out.columns else 0
    before_age_years_non_null = int(pd.to_numeric(out["Age_en_ans"], errors="coerce").notna().sum()) if "Age_en_ans" in out.columns else 0
    before_age_group_non_null = int(out["Tranche_age"].notna().sum()) if "Tranche_age" in out.columns else 0

    for date_col in ["Date_notification", "Date_debut_maladie", "Date_prelevement", "Date_resultat", "Date_reception_labo", "Date_issue"]:
        if date_col in out.columns:
            out[date_col] = _to_dt(out[date_col])

    date_reference = None
    if "Date_notification" in out.columns:
        date_reference = out["Date_notification"]
    if "Date_debut_maladie" in out.columns:
        onset_series = out["Date_debut_maladie"]
        date_reference = onset_series if date_reference is None else date_reference.combine_first(onset_series)

    parsed_year = pd.Series(pd.NA, index=out.index, dtype="Int64")
    parsed_week = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if "Semaine_epid" in out.columns and out["Semaine_epid"].notna().any():
        parsed_year, parsed_week = _parse_semaine_epid_series(out["Semaine_epid"])

    current_year = (
        pd.to_numeric(out["Annee_epid"], errors="coerce").astype("Int64")
        if "Annee_epid" in out.columns
        else pd.Series(pd.NA, index=out.index, dtype="Int64")
    )
    current_week = (
        pd.to_numeric(out["Num_semaine_epid"], errors="coerce").astype("Int64")
        if "Num_semaine_epid" in out.columns
        else pd.Series(pd.NA, index=out.index, dtype="Int64")
    )

    if replace_existing:
        current_year = parsed_year
        current_week = parsed_week
    else:
        current_year = current_year.fillna(parsed_year)
        current_week = current_week.fillna(parsed_week)

    if date_reference is not None and pd.Series(date_reference).notna().any():
        iso = date_reference.dt.isocalendar()
        derived_year = iso["year"].astype("Int64")
        derived_week = iso["week"].astype("Int64")
        current_year = derived_year if replace_existing else current_year.fillna(derived_year)
        current_week = derived_week if replace_existing else current_week.fillna(derived_week)

    if replace_existing or "Annee_epid" not in out.columns:
        out["Annee_epid"] = current_year
    else:
        existing_year = pd.to_numeric(out["Annee_epid"], errors="coerce").astype("Int64")
        out["Annee_epid"] = existing_year.where(existing_year.notna(), current_year)

    if replace_existing or "Num_semaine_epid" not in out.columns:
        out["Num_semaine_epid"] = current_week
    else:
        existing_week = pd.to_numeric(out["Num_semaine_epid"], errors="coerce").astype("Int64")
        out["Num_semaine_epid"] = existing_week.where(existing_week.notna(), current_week)

    week_label = (
        current_year.astype("string") + "-W" + current_week.astype("string").str.zfill(2)
    ).mask(current_year.isna() | current_week.isna(), pd.NA)
    if replace_existing or "Semaine_epid" not in out.columns:
        out["Semaine_epid"] = week_label
    else:
        existing_week_label = out["Semaine_epid"].astype("string")
        out["Semaine_epid"] = existing_week_label.where(existing_week_label.notna(), week_label)

    if "Age" in out.columns and "Unite_age" in out.columns:
        computed_age_years = _compute_age_years(out["Age"], out["Unite_age"])
        if replace_existing or "Age_en_ans" not in out.columns:
            out["Age_en_ans"] = computed_age_years
        else:
            out["Age_en_ans"] = pd.to_numeric(out["Age_en_ans"], errors="coerce").fillna(computed_age_years)
    elif "Age_en_ans" in out.columns:
        out["Age_en_ans"] = pd.to_numeric(out["Age_en_ans"], errors="coerce")

    if "Age_en_ans" in out.columns:
        age_group = _compute_age_group(out["Age_en_ans"])
        if replace_existing or "Tranche_age" not in out.columns:
            out["Tranche_age"] = age_group
        else:
            out["Tranche_age"] = out["Tranche_age"].astype("string").fillna(age_group)

    after_week_non_null = int(out["Semaine_epid"].notna().sum()) if "Semaine_epid" in out.columns else 0
    after_age_years_non_null = int(pd.to_numeric(out["Age_en_ans"], errors="coerce").notna().sum()) if "Age_en_ans" in out.columns else 0
    after_age_group_non_null = int(out["Tranche_age"].notna().sum()) if "Tranche_age" in out.columns else 0

    derived_info["weeks_calculated"] = max(after_week_non_null - before_week_non_null, 0)
    derived_info["age_years_calculated"] = max(after_age_years_non_null - before_age_years_non_null, 0)
    derived_info["age_groups_calculated"] = max(after_age_group_non_null - before_age_group_non_null, 0)

    if return_info:
        return out, derived_info
    return out


def build_mapping_preview_table(
    mapping: dict[str, Optional[str]],
    auto_metadata: dict[str, dict[str, Any]],
    threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for standard_name, meta in STANDARD_COLUMNS.items():
        is_derived = bool(meta.get("derived"))
        selected_source = mapping.get(standard_name)
        auto_entry = auto_metadata.get(standard_name, _build_empty_metadata())

        derivable = False
        if is_derived:
            derivable = any(
                [
                    bool(meta.get("depends_on_any")) and any(_mapping_has(mapping, col) for col in meta.get("depends_on_any", [])),
                    bool(meta.get("depends_on_all")) and all(_mapping_has(mapping, col) for col in meta.get("depends_on_all", [])),
                    any(_spec_is_satisfied(mapping, fallback) for fallback in meta.get("fallback", [])),
                ]
            )

        if selected_source:
            selection_meta = resolve_mapping_selection_metadata(
                standard_name,
                selected_source,
                auto_metadata,
                threshold=threshold,
            )
            method = selection_meta["method"]
            confidence = selection_meta["confidence"]
            source_display = selection_meta["source_column"]
            if method == "manual":
                status = "À vérifier"
            elif confidence >= threshold:
                status = "OK"
            else:
                status = "À vérifier"
        elif is_derived and derivable:
            method = "derived"
            confidence = 100
            source_display = "Calcul automatique"
            status = "Calculable automatiquement"
        else:
            method = auto_entry.get("method", "not_mapped")
            confidence = auto_entry.get("confidence", 0)
            source_display = auto_entry.get("source_column")
            if auto_entry.get("accepted") and auto_entry.get("source_column"):
                status = "OK"
            elif auto_entry.get("source_column") and confidence > 0:
                status = "À vérifier"
            elif standard_name in CRITICAL_COLUMNS:
                status = "Manuel requis"
            else:
                status = "Absent"

        rows.append(
            {
                "Variable standard": standard_name,
                "Type de variable": "Dérivée" if is_derived else "Source",
                "Colonne source proposée": source_display,
                "Méthode de détection": method,
                "Score de confiance": confidence,
                "Statut": status,
                "_level": get_column_level(standard_name),
            }
        )

    preview_df = pd.DataFrame(rows)
    return preview_df


def extract_profile_mapping(mapping: dict[str, Optional[str]]) -> dict[str, Optional[str]]:
    """Conserve les mappings source et ne persiste que les colonnes dérivées réellement utilisées comme repli."""
    profile_mapping = {
        column_name: source_name
        for column_name, source_name in mapping.items()
        if column_name in SOURCE_COLUMNS and source_name
    }

    temporal_source_available = any(
        [
            _mapping_has(mapping, "Date_notification"),
            _mapping_has(mapping, "Date_debut_maladie"),
        ]
    )
    age_source_available = _mapping_has(mapping, "Age") and _mapping_has(mapping, "Unite_age")

    for column_name in ["Semaine_epid", "Num_semaine_epid", "Annee_epid"]:
        if _mapping_has(mapping, column_name) and not temporal_source_available:
            profile_mapping[column_name] = mapping[column_name]

    for column_name in ["Age_en_ans", "Tranche_age"]:
        if _mapping_has(mapping, column_name) and not age_source_available:
            profile_mapping[column_name] = mapping[column_name]

    return profile_mapping


def save_mapping_profile(
    mapping: dict,
    profile_name: str,
    metadata: dict | None = None,
    mapping_dir: Path | None = None,
) -> Path:
    """Enregistre un profil de mapping validé au format JSON."""
    profile_txt = str(profile_name).strip()
    if not profile_txt:
        raise ValueError("Le nom du profil de mapping est obligatoire.")

    target_dir = Path(mapping_dir or DEFAULT_MAPPING_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{normalize_column_name(profile_txt) or 'mapping_profile'}.json"
    target_path = target_dir / file_name

    payload = {
        "profile_name": profile_txt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mapping": extract_profile_mapping(mapping),
        "metadata": metadata or {},
    }
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target_path


def load_mapping_profile(
    profile_name: str,
    mapping_dir: Path | None = None,
) -> dict:
    """Charge le contenu JSON d'un profil de mapping enregistré."""
    target_dir = Path(mapping_dir or DEFAULT_MAPPING_DIR)
    candidate_names = [
        target_dir / f"{profile_name}.json",
        target_dir / f"{normalize_column_name(profile_name)}.json",
    ]
    for path in candidate_names:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Profil de mapping introuvable : {profile_name}")


def list_mapping_profiles(mapping_dir: Path | None = None) -> list[str]:
    """Liste les noms des profils de mapping disponibles."""
    target_dir = Path(mapping_dir or DEFAULT_MAPPING_DIR)
    if not target_dir.exists():
        return []
    profiles = []
    for path in sorted(target_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            profiles.append(str(payload.get("profile_name") or path.stem))
        except Exception:
            profiles.append(path.stem)
    return profiles


def build_mapping_quality_report(
    df: pd.DataFrame,
    mapping: dict,
    derived_info: dict | None = None,
) -> dict:
    """Construit un résumé compact de qualité après mapping et dérivation."""
    derived_info = derived_info or {}
    n_rows = int(len(df))
    original_columns = list(derived_info.get("original_columns", list(df.columns)))
    mapped_sources = [source for source in mapping.values() if source]
    mapped_source_columns = [
        source for column_name, source in mapping.items()
        if source and column_name in SOURCE_COLUMNS
    ]
    unmapped_source_columns = [col for col in original_columns if col not in mapped_sources]

    recognized_standard_columns = [
        col for col in STANDARD_COLUMNS
        if col in df.columns and df[col].notna().any()
    ]

    important_non_null_pct: dict[str, float] = {}
    for col in IMPORTANT_COLUMNS:
        if col in df.columns and n_rows > 0:
            important_non_null_pct[col] = round(float(df[col].notna().mean() * 100.0), 1)

    date_valid_series = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if "Date_notification" in df.columns:
        date_valid_series = _to_dt(df["Date_notification"])
    if "Date_debut_maladie" in df.columns:
        date_valid_series = date_valid_series.combine_first(_to_dt(df["Date_debut_maladie"]))

    age_valid_series = pd.Series(False, index=df.index, dtype="bool")
    if "Age_en_ans" in df.columns:
        age_num = pd.to_numeric(df["Age_en_ans"], errors="coerce")
        age_valid_series = age_num.between(0, 120, inclusive="both").fillna(False)
    elif "Tranche_age" in df.columns:
        age_valid_series = df["Tranche_age"].notna()

    def _missing_columns(columns: list[str]) -> list[str]:
        missing = []
        for col in columns:
            if col not in df.columns or not df[col].notna().any():
                missing.append(col)
        return missing

    return {
        "Nombre de lignes": n_rows,
        "Nombre de colonnes sources": len(set(mapped_source_columns)),
        "Nombre de colonnes standards reconnues": len(recognized_standard_columns),
        "Nombre de colonnes non reconnues": len(unmapped_source_columns),
        "Colonnes non reconnues": unmapped_source_columns,
        "Pourcentage de valeurs non nulles": important_non_null_pct,
        "Dates valides": {
            "valid": int(date_valid_series.notna().sum()),
            "total": n_rows,
        },
        "Âges valides": {
            "valid": int(age_valid_series.sum()),
            "total": n_rows,
        },
        "Semaines épidémiologiques calculées": int(derived_info.get("weeks_calculated", int(df["Semaine_epid"].notna().sum()) if "Semaine_epid" in df.columns else 0)),
        "Tranches d’âge calculées": int(derived_info.get("age_groups_calculated", int(df["Tranche_age"].notna().sum()) if "Tranche_age" in df.columns else 0)),
        "Colonnes critiques absentes": _missing_columns(CRITICAL_COLUMNS),
        "Colonnes importantes absentes": _missing_columns(IMPORTANT_COLUMNS),
        "Colonnes optionnelles absentes": _missing_columns(OPTIONAL_COLUMNS),
    }


def dataframe_to_standardized_excel_bytes(
    df: pd.DataFrame,
    sheet_name: str = "LineList_standardisee",
) -> bytes:
    """Sérialise une line list standardisée vers un classeur Excel en mémoire."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    buffer.seek(0)
    return buffer.getvalue()
