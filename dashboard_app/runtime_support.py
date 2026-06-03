"""Support d'exécution partagé pour les modules extraits du dashboard."""

import glob
import hashlib
import html
import json
import logging
import os
import re
import tempfile
import unicodedata
from datetime import date, datetime
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import dashboard_app.advanced as advanced_api
from dashboard_app.core import _normalize_metric_alias_columns
from dashboard_app.domain import (
    _is_yes_series,
    _norm_key,
    _resolve_map_filter_value,
    _tdr_result_norm,
)
from dashboard_app.overview import format_range_label_for_display
from dashboard_app.standard_transverse import (
    build_standard_analysis_capability_matrix,
    build_standard_capability_note,
    build_standard_care_issue_audit,
    build_standard_classification_audit,
    build_standard_disease_profile,
    build_standard_file_structure_audit,
    build_standard_semantic_status_summary,
    build_standard_symptom_audit,
)

_STATIC_RUNTIME_CONTEXT = {
    name: getattr(advanced_api, name)
    for name in dir(advanced_api)
    if not name.startswith("__")
}
_STATIC_RUNTIME_CONTEXT.update(
    {
        "glob": glob,
        "hashlib": hashlib,
        "html": html,
        "json": json,
        "logging": logging,
        "os": os,
        "re": re,
        "tempfile": tempfile,
        "unicodedata": unicodedata,
        "date": date,
        "datetime": datetime,
        "SequenceMatcher": SequenceMatcher,
        "BytesIO": BytesIO,
        "Path": Path,
        "Any": Any,
        "Dict": Dict,
        "Iterable": Iterable,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "Union": Union,
        "_normalize_metric_alias_columns": _normalize_metric_alias_columns,
        "_is_yes_series": _is_yes_series,
        "_norm_key": _norm_key,
        "_resolve_map_filter_value": _resolve_map_filter_value,
        "_tdr_result_norm": _tdr_result_norm,
        "build_standard_capability_note": build_standard_capability_note,
        "build_standard_classification_audit": build_standard_classification_audit,
        "build_standard_analysis_capability_matrix": build_standard_analysis_capability_matrix,
        "build_standard_care_issue_audit": build_standard_care_issue_audit,
        "build_standard_disease_profile": build_standard_disease_profile,
        "build_standard_file_structure_audit": build_standard_file_structure_audit,
        "build_standard_semantic_status_summary": build_standard_semantic_status_summary,
        "build_standard_symptom_audit": build_standard_symptom_audit,
        "format_range_label_for_display": format_range_label_for_display,
    }
)

_NARRATIVE_CONTEXT_CACHE = None


def _get_narrative_runtime_context() -> dict:
    """Charge paresseusement les utilitaires narratifs pour éviter les imports circulaires au chargement du module."""
    global _NARRATIVE_CONTEXT_CACHE
    if _NARRATIVE_CONTEXT_CACHE is None:
        from dashboard_app import narratives as narratives_api

        _NARRATIVE_CONTEXT_CACHE = {
            name: getattr(narratives_api, name)
            for name in dir(narratives_api)
            if not name.startswith("__")
        }
    return dict(_NARRATIVE_CONTEXT_CACHE)


def inject_runtime_support(target_globals: dict) -> None:
    """Injecte dans les globals du module l'espace de noms runtime historique du dashboard."""
    target_globals.update(_STATIC_RUNTIME_CONTEXT)



def build_runtime_context(**runtime_values):
    """Construit le contexte d'exécution transmis aux modules d'onglets extraits."""
    ctx = dict(_STATIC_RUNTIME_CONTEXT)
    ctx.update(_get_narrative_runtime_context())
    ctx.update(runtime_values)
    return ctx
