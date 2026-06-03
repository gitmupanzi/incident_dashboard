"""Shared runtime support for extracted dashboard modules."""

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
    build_standard_classification_audit,
    build_standard_analysis_capability_matrix,
    build_standard_care_issue_audit,
    build_standard_file_structure_audit,
    build_standard_symptom_audit,
)
from dashboard_app.overview import format_range_label_for_display

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
        "build_standard_classification_audit": build_standard_classification_audit,
        "build_standard_analysis_capability_matrix": build_standard_analysis_capability_matrix,
        "build_standard_care_issue_audit": build_standard_care_issue_audit,
        "build_standard_file_structure_audit": build_standard_file_structure_audit,
        "build_standard_symptom_audit": build_standard_symptom_audit,
        "format_range_label_for_display": format_range_label_for_display,
    }
)

_NARRATIVE_CONTEXT_CACHE = None


def _get_narrative_runtime_context() -> dict:
    """Lazy-load narrative helpers to avoid circular imports at module import time."""
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
    """Populate module globals with the legacy dashboard runtime namespace."""
    target_globals.update(_STATIC_RUNTIME_CONTEXT)



def build_runtime_context(**runtime_values):
    """Build the execution context passed to extracted tab modules."""
    ctx = dict(_STATIC_RUNTIME_CONTEXT)
    ctx.update(_get_narrative_runtime_context())
    ctx.update(runtime_values)
    return ctx
