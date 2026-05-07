"""Render the decision-oriented IREP tab."""

from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())

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


def _irep_normalize_key(value: object) -> str:
    """Normalise une clé géographique pour les jointures de référence."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        cleaned = clean_str(pd.Series([text], dtype="string")).iloc[0]
        cleaned = unicodedata.normalize("NFKD", str(cleaned))
        cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
        cleaned = re.sub(r"\s+", " ", cleaned).strip().upper()
        return cleaned
    except Exception:
        fallback = unicodedata.normalize("NFKD", text)
        fallback = "".join(ch for ch in fallback if not unicodedata.combining(ch))
        return re.sub(r"\s+", " ", fallback).strip().upper()


def _irep_reference_population_total(
    pop_ref: pd.DataFrame,
    table: pd.DataFrame,
    geography_level: str,
    denominator_mode: str,
) -> float:
    """Calcule une population couverte de repli depuis le référentiel OCHA."""
    if pop_ref is None or pop_ref.empty or table is None or table.empty:
        return np.nan

    ref = pop_ref.copy()
    ref["Population_reference"] = pd.to_numeric(ref["Population_reference"], errors="coerce")
    ref = ref.dropna(subset=["Population_reference"])
    ref = ref[ref["Population_reference"] > 0]
    if ref.empty:
        return np.nan

    if COL_PROV in table.columns:
        scope_provinces = {
            _irep_normalize_key(v)
            for v in table[COL_PROV].dropna().astype(str).tolist()
            if _irep_normalize_key(v)
        }
        if scope_provinces:
            ref = ref[ref["_prov_norm"].isin(scope_provinces)].copy()
    if ref.empty:
        return np.nan

    if geography_level == "province":
        if denominator_mode == "group_max":
            total = (
                ref.dropna(subset=["_prov_norm"])
                .groupby("_prov_norm", dropna=False)["Population_reference"]
                .max()
                .sum()
            )
        else:
            if ref["_zone_norm"].notna().any():
                total = (
                    ref.dropna(subset=["_prov_norm", "_zone_norm"])
                    .groupby(["_prov_norm", "_zone_norm"], dropna=False)["Population_reference"]
                    .max()
                    .groupby(level=0)
                    .sum()
                    .sum()
                )
            else:
                total = (
                    ref.dropna(subset=["_prov_norm"])
                    .groupby("_prov_norm", dropna=False)["Population_reference"]
                    .max()
                    .sum()
                )
        return float(total) if pd.notna(total) else np.nan

    if COL_ZS in table.columns:
        pair_rows = []
        for _, row in table[[COL_PROV, COL_ZS]].dropna().drop_duplicates().iterrows():
            pair_rows.append((_irep_normalize_key(row[COL_PROV]), _irep_normalize_key(row[COL_ZS])))
        pair_keys = {pair for pair in pair_rows if all(pair)}
        if pair_keys:
            matched = ref[
                ref.apply(
                    lambda row: (str(row["_prov_norm"]), str(row["_zone_norm"])) in pair_keys,
                    axis=1,
                )
            ].copy()
            if not matched.empty:
                total = (
                    matched.dropna(subset=["_prov_norm", "_zone_norm"])
                    .groupby(["_prov_norm", "_zone_norm"], dropna=False)["Population_reference"]
                    .max()
                    .sum()
                )
                return float(total) if pd.notna(total) else np.nan

        zone_keys = {
            _irep_normalize_key(v)
            for v in table[COL_ZS].dropna().astype(str).tolist()
            if _irep_normalize_key(v)
        }
        if zone_keys:
            matched = ref[ref["_zone_norm"].isin(zone_keys)].copy()
            if not matched.empty:
                total = (
                    matched.dropna(subset=["_zone_norm"])
                    .groupby("_zone_norm", dropna=False)["Population_reference"]
                    .max()
                    .sum()
                )
                return float(total) if pd.notna(total) else np.nan

    total = (
        ref.dropna(subset=["_prov_norm", "_zone_norm"])
        .groupby(["_prov_norm", "_zone_norm"], dropna=False)["Population_reference"]
        .max()
        .sum()
    ) if ref["_zone_norm"].notna().any() else (
        ref.dropna(subset=["_prov_norm"])
        .groupby("_prov_norm", dropna=False)["Population_reference"]
        .max()
        .sum()
    )
    return float(total) if pd.notna(total) else np.nan


def _irep_clean_label_series(series: pd.Series) -> pd.Series:
    """Nettoie un libellé géographique tout en conservant sa casse d'affichage."""
    out = series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.replace({"": pd.NA, "<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out


def _irep_clean_province_value(value: object) -> object:
    """Ramène les libellés de province vers une forme canonique."""
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for pattern, replacement in PROVINCE_PATTERNS:
        if re.match(pattern, normalized, flags=re.IGNORECASE):
            return replacement
    return _irep_clean_label_series(pd.Series([text], dtype="string")).iloc[0]


def _irep_clean_province_series(series: pd.Series) -> pd.Series:
    """Applique le nettoyage canonique des provinces à une série."""
    out = series.apply(_irep_clean_province_value)
    return out.astype("string")


def _irep_prepare_population_reference_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Prépare une table de population exploitable pour provinces et zones."""
    empty = pd.DataFrame(
        columns=[
            "Province_reference",
            "Zone_de_sante_reference",
            "ZSCode_reference",
            "Population_reference",
            "_prov_norm",
            "_zone_norm",
            "_code_norm",
        ]
    )
    if raw_df is None or raw_df.empty:
        return empty

    work = raw_df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    province_col = next(
        (c for c in ["PROVINCE", "Province", "province", "Province_notification"] if c in work.columns),
        None,
    )
    zone_col = next(
        (
            c
            for c in [
                "Nom",
                "Zone_de_sante_notification",
                "Zone_de_sante",
                "Zone de santé",
                "Zone de sante",
                "ZS",
            ]
            if c in work.columns
        ),
        None,
    )
    code_col = next(
        (
            c
            for c in [
                "ZSCode",
                "Code_ZS",
                "Code zone de santé",
                "Code zone de sante",
                "Zone_de_sante_code",
                "zs_code",
            ]
            if c in work.columns
        ),
        None,
    )
    pop_col = next(
        (c for c in ["Population", "POPULATION", "POP", "population", "pop"] if c in work.columns),
        None,
    )

    if pop_col is None or (province_col is None and zone_col is None and code_col is None):
        return empty

    out = pd.DataFrame(index=work.index)
    out["Province_reference"] = (
        _irep_clean_province_series(work[province_col]) if province_col else pd.Series(pd.NA, index=work.index, dtype="string")
    )
    out["Zone_de_sante_reference"] = (
        _irep_clean_label_series(work[zone_col]) if zone_col else pd.Series(pd.NA, index=work.index, dtype="string")
    )
    out["ZSCode_reference"] = (
        _irep_clean_label_series(work[code_col]) if code_col else pd.Series(pd.NA, index=work.index, dtype="string")
    )
    out["Population_reference"] = pd.to_numeric(work[pop_col], errors="coerce")
    out = out.dropna(subset=["Population_reference"])
    out = out[out["Population_reference"] > 0].copy()
    if out.empty:
        return empty

    out["_prov_norm"] = out["Province_reference"].map(_irep_normalize_key)
    out["_zone_norm"] = out["Zone_de_sante_reference"].map(_irep_normalize_key)
    out["_code_norm"] = out["ZSCode_reference"].map(_irep_normalize_key)
    out.loc[out["_prov_norm"] == "", "_prov_norm"] = pd.NA
    out.loc[out["_zone_norm"] == "", "_zone_norm"] = pd.NA
    out.loc[out["_code_norm"] == "", "_code_norm"] = pd.NA

    dedupe_keys = []
    if out["_code_norm"].notna().any():
        dedupe_keys = ["_code_norm"]
    elif out["_prov_norm"].notna().any() and out["_zone_norm"].notna().any():
        dedupe_keys = ["_prov_norm", "_zone_norm"]
    elif out["_zone_norm"].notna().any():
        dedupe_keys = ["_zone_norm"]
    elif out["_prov_norm"].notna().any():
        dedupe_keys = ["_prov_norm"]
    else:
        return empty

    agg_spec = {
        "Population_reference": "max",
        "Province_reference": "first",
        "Zone_de_sante_reference": "first",
        "ZSCode_reference": "first",
    }
    cleaned = (
        out.sort_values("Population_reference", ascending=False)
        .groupby(dedupe_keys, dropna=False, as_index=False)
        .agg(agg_spec)
    )
    cleaned["_prov_norm"] = cleaned["Province_reference"].map(_irep_normalize_key)
    cleaned["_zone_norm"] = cleaned["Zone_de_sante_reference"].map(_irep_normalize_key)
    cleaned["_code_norm"] = cleaned["ZSCode_reference"].map(_irep_normalize_key)
    cleaned.loc[cleaned["_prov_norm"] == "", "_prov_norm"] = pd.NA
    cleaned.loc[cleaned["_zone_norm"] == "", "_zone_norm"] = pd.NA
    cleaned.loc[cleaned["_code_norm"] == "", "_code_norm"] = pd.NA
    return cleaned


def _irep_default_population_path() -> Path:
    """Retourne le chemin du fichier de population par défaut."""
    filename = "RDC_Zone_de_sante_OCHA.xlsx"
    candidates = [
        Path(r"C:\Users\Benjamin MUPANZI\Documents\incident_dashboard\data") / filename,
        Path.cwd() / "data" / filename,
        Path(__file__).resolve().parents[2] / "data" / filename,
        Path(__file__).resolve().parents[1] / "data" / filename,
        Path(__file__).resolve().parents[3] / "incident_dashboard" / "data" / filename,
    ]

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    return candidates[0]


def _irep_display_population_path(path: Path) -> str:
    """Retourne un chemin d'affichage portable pour l'interface."""
    try:
        parts = list(path.parts)
        if "data" in parts:
            idx = parts.index("data")
            return "/".join(parts[idx:])
    except Exception:
        pass
    return path.name


def _irep_load_population_reference(uploaded_file: object | None) -> tuple[pd.DataFrame, str]:
    """Charge la population de référence depuis le fichier par défaut ou un upload."""
    source_label = "Aucun fichier de population chargé"
    try:
        if uploaded_file is not None:
            upload_name = str(getattr(uploaded_file, "name", "population"))
            if upload_name.lower().endswith(".csv"):
                pop_raw = pd.read_csv(uploaded_file)
            else:
                pop_raw = load_excel_cached(uploaded_file)
            source_label = f"Fichier chargé : {upload_name}"
        else:
            default_path = _irep_default_population_path()
            if not default_path.exists():
                return pd.DataFrame(), f"Fichier par défaut introuvable : {_irep_display_population_path(default_path)}"
            pop_raw = load_excel_cached(default_path)
            source_label = f"Fichier par défaut : {_irep_display_population_path(default_path)}"
        return _irep_prepare_population_reference_frame(pop_raw), source_label
    except Exception as exc:
        return pd.DataFrame(), f"Lecture impossible du fichier population : {exc}"


def _irep_build_population_lookups(pop_ref: pd.DataFrame) -> dict:
    """Construit les tables de correspondance population province / zone."""
    lookups = {
        "province_sum": {},
        "province_max": {},
        "province": {},
        "zone_pair": {},
        "zone_name": {},
        "zscode": {},
    }
    if pop_ref is None or pop_ref.empty:
        return lookups

    work = pop_ref.copy()
    work["Population_reference"] = pd.to_numeric(work["Population_reference"], errors="coerce")
    work = work.dropna(subset=["Population_reference"])
    if work.empty:
        return lookups

    if {"_prov_norm", "_zone_norm"}.issubset(work.columns):
        zone_work = work.dropna(subset=["_prov_norm", "_zone_norm"]).copy()
        if not zone_work.empty:
            pair_tbl = (
                zone_work.groupby(["_prov_norm", "_zone_norm"], dropna=False, as_index=False)
                .agg(Population_reference=("Population_reference", "max"))
            )
            lookups["zone_pair"] = {
                (str(row["_prov_norm"]), str(row["_zone_norm"])): float(row["Population_reference"])
                for _, row in pair_tbl.iterrows()
            }
            prov_tbl = (
                pair_tbl.groupby("_prov_norm", dropna=False, as_index=False)
                .agg(Population_reference=("Population_reference", "sum"))
            )
            lookups["province_sum"] = {
                str(row["_prov_norm"]): float(row["Population_reference"])
                for _, row in prov_tbl.iterrows()
            }
            zone_name_tbl = (
                zone_work.groupby("_zone_norm", dropna=False, as_index=False)
                .agg(Population_reference=("Population_reference", "max"))
            )
            lookups["zone_name"] = {
                str(row["_zone_norm"]): float(row["Population_reference"])
                for _, row in zone_name_tbl.iterrows()
            }

    if "_code_norm" in work.columns and work["_code_norm"].notna().any():
        code_tbl = (
            work.dropna(subset=["_code_norm"])
            .groupby("_code_norm", dropna=False, as_index=False)
            .agg(Population_reference=("Population_reference", "max"))
        )
        lookups["zscode"] = {
            str(row["_code_norm"]): float(row["Population_reference"])
            for _, row in code_tbl.iterrows()
        }

    if not lookups["province"] and "_prov_norm" in work.columns and work["_prov_norm"].notna().any():
        prov_tbl = (
            work.dropna(subset=["_prov_norm"])
            .groupby("_prov_norm", dropna=False, as_index=False)
            .agg(Population_reference=("Population_reference", "max"))
        )
        lookups["province_max"] = {
            str(row["_prov_norm"]): float(row["Population_reference"])
            for _, row in prov_tbl.iterrows()
        }

    if "_prov_norm" in work.columns and work["_prov_norm"].notna().any() and not lookups["province_max"]:
        prov_max_tbl = (
            work.dropna(subset=["_prov_norm"])
            .groupby("_prov_norm", dropna=False, as_index=False)
            .agg(Population_reference=("Population_reference", "max"))
        )
        lookups["province_max"] = {
            str(row["_prov_norm"]): float(row["Population_reference"])
            for _, row in prov_max_tbl.iterrows()
        }

    lookups["province"] = lookups["province_sum"] or lookups["province_max"]

    return lookups


def _irep_prepare_analysis_scope(df_source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prépare le périmètre temporel et les variables minimales pour l'IREP."""
    if df_source is None or df_source.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["order", "label"])

    from dashboard_app.narratives import _prepare_surveillance_period_scope

    scoped, reference = _prepare_surveillance_period_scope(df_source)
    if scoped.empty or reference.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["order", "label"])

    work = scoped.copy()
    if COL_PROV in work.columns:
        work[COL_PROV] = _irep_clean_province_series(work[COL_PROV])
    if COL_ZS in work.columns:
        work[COL_ZS] = _irep_clean_label_series(work[COL_ZS])

    if "Total_cas" in work.columns:
        work["_irep_cases"] = pd.to_numeric(work["Total_cas"], errors="coerce")
        if work["_irep_cases"].notna().mean() < 0.20:
            work["_irep_cases"] = 1.0
        work["_irep_cases"] = work["_irep_cases"].fillna(0.0)
    else:
        work["_irep_cases"] = 1.0

    if "Total_deces" in work.columns:
        work["_irep_deaths"] = pd.to_numeric(work["Total_deces"], errors="coerce").fillna(0.0)
    elif "is_death" in work.columns:
        work["_irep_deaths"] = pd.to_numeric(work["is_death"], errors="coerce").fillna(0.0)
    elif COL_ISSUE in work.columns:
        work["_irep_deaths"] = work[COL_ISSUE].apply(lambda x: 1.0 if is_death(x) else 0.0)
    else:
        work["_irep_deaths"] = 0.0

    if "delai_onset_to_notif" in work.columns:
        work["_irep_delay_days"] = pd.to_numeric(work["delai_onset_to_notif"], errors="coerce")
    elif DATE_ONSET in work.columns and DATE_NOTIF in work.columns:
        onset = pd.to_datetime(work[DATE_ONSET], errors="coerce")
        notif = pd.to_datetime(work[DATE_NOTIF], errors="coerce")
        work["_irep_delay_days"] = (notif - onset).dt.days
    else:
        work["_irep_delay_days"] = np.nan
    work.loc[work["_irep_delay_days"] < 0, "_irep_delay_days"] = np.nan
    return work, reference


def _irep_non_missing_ratio(series: pd.Series) -> float:
    """Mesure la proportion de valeurs renseignées en traitant les blancs comme manquants."""
    if series is None or len(series) == 0:
        return np.nan
    if pd.api.types.is_numeric_dtype(series):
        return float(series.notna().mean())
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({"": pd.NA, "<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})
    return float(cleaned.notna().mean())


def _irep_group_completeness(df_group: pd.DataFrame, required_cols: list[str]) -> float:
    """Calcule une complétude groupe-friendly."""
    if "score_completude_core_%" in df_group.columns:
        score = pd.to_numeric(df_group["score_completude_core_%"], errors="coerce").dropna()
        if not score.empty:
            return float(score.mean())

    cols = [c for c in required_cols if c in df_group.columns]
    if not cols:
        return np.nan
    per_col = [_irep_non_missing_ratio(df_group[c]) for c in cols]
    per_col = [x for x in per_col if pd.notna(x)]
    if not per_col:
        return np.nan
    return float(np.mean(per_col) * 100.0)


def _irep_group_timeliness(df_group: pd.DataFrame, threshold_days: float) -> tuple[float, float]:
    """Retourne la promptitude sous seuil et le délai médian."""
    delay = pd.to_numeric(df_group["_irep_delay_days"], errors="coerce")
    delay = delay[delay.notna() & (delay >= 0)]
    if delay.empty:
        return np.nan, np.nan
    promptitude = float((delay <= float(threshold_days)).mean() * 100.0)
    median_delay = float(delay.median())
    return promptitude, median_delay


def _irep_score_metric(series: pd.Series) -> pd.Series:
    """Scoring 0-100 robuste même pour de petits effectifs."""
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    if valid.nunique() == 1:
        return pd.Series(np.where(values.notna(), 50.0, np.nan), index=values.index, dtype="float64")
    if valid.size < 3:
        ranks = valid.rank(method="average", pct=True)
        return values.map(ranks.mul(100.0)).astype("float64")
    return _score_quantile_0_100(values)


def _irep_resolve_population(
    row: pd.Series,
    geography_level: str,
    lookups: dict,
    group_cols: list[str],
    denominator_mode: str = "zs_sum",
) -> float:
    """Retrouve la population d'une province ou d'une zone de santé."""
    if geography_level == "province":
        prov_val = row.get(COL_PROV if COL_PROV in group_cols else group_cols[0])
        prov_key = _irep_normalize_key(prov_val)
        province_map = lookups["province_max"] if denominator_mode == "group_max" else (lookups["province_sum"] or lookups["province_max"])
        return province_map.get(prov_key, np.nan)

    zscode_candidates = [
        "ZSCode",
        "Code_ZS",
        "Zone_de_sante_code",
        "zs_code",
    ]
    for col in zscode_candidates:
        if col in row.index:
            code_key = _irep_normalize_key(row.get(col))
            if code_key and code_key in lookups["zscode"]:
                return lookups["zscode"][code_key]

    prov_val = row.get(COL_PROV, row.get(group_cols[0] if group_cols else COL_PROV))
    zone_val = row.get(COL_ZS, row.get(group_cols[-1] if group_cols else COL_ZS))
    pair_key = (_irep_normalize_key(prov_val), _irep_normalize_key(zone_val))
    if all(pair_key) and pair_key in lookups["zone_pair"]:
        return lookups["zone_pair"][pair_key]

    zone_key = _irep_normalize_key(zone_val)
    return lookups["zone_name"].get(zone_key, np.nan)


def _irep_factor_summary(row: pd.Series) -> str:
    """Identifie les facteurs qui tirent le score de risque vers le haut."""
    score_map = {
        "Tendance": row.get("TrendScore"),
        "Incidence": row.get("IncidenceScore"),
        "Létalité": row.get("CFRScore"),
        "Promptitude": row.get("PromptitudeScore"),
        "Complétude": row.get("CompletenessScore"),
    }
    ordered = [
        label
        for label, score in sorted(
            score_map.items(),
            key=lambda item: float(item[1]) if pd.notna(item[1]) else -1.0,
            reverse=True,
        )
        if pd.notna(score)
    ]
    if not ordered:
        return "-"
    return ", ".join(ordered[:2])


def _irep_build_window_risk_table(
    window_df: pd.DataFrame,
    base_df: pd.DataFrame,
    *,
    group_cols: list[str],
    geography_level: str,
    pop_ref: pd.DataFrame,
    trend_current_orders: list[float],
    trend_previous_orders: list[float],
    completeness_required: list[str],
    threshold_days: float,
    weights: dict,
    denominator_mode: str = "zs_sum",
    incidence_multiplier: int = 100000,
) -> tuple[pd.DataFrame, dict]:
    """Construit la table IREP pour une fenêtre et un niveau géographique."""
    meta = {
        "latest_label": None,
        "latest_order": None,
        "window_weeks": 0,
        "total_population": np.nan,
        "population_coverage": 0,
        "total_cases": np.nan,
        "global_attack": np.nan,
        "global_incidence": np.nan,
        "incidence_col": f"Incidence_pour_{int(incidence_multiplier)}",
        "top_attack_label": None,
        "top_attack_value": np.nan,
        "top_incidence_label": None,
        "top_incidence_value": np.nan,
        "denominator_mode": denominator_mode,
    }
    if window_df is None or window_df.empty:
        return pd.DataFrame(), meta

    valid_group_cols = [c for c in group_cols if c in window_df.columns]
    if not valid_group_cols:
        return pd.DataFrame(), meta

    work = window_df.copy()
    for col in valid_group_cols:
        if col == COL_PROV:
            work[col] = _irep_clean_province_series(work[col])
        else:
            work[col] = _irep_clean_label_series(work[col])
    work = work.dropna(subset=valid_group_cols)
    if work.empty:
        return pd.DataFrame(), meta

    latest_order = float(work["_surv_order"].max())
    latest_label_series = work.loc[work["_surv_order"] == latest_order, "_surv_label"].dropna().astype(str)
    latest_label = latest_label_series.iloc[0] if not latest_label_series.empty else None
    meta["latest_order"] = latest_order
    meta["latest_label"] = latest_label
    meta["window_weeks"] = int(work["_surv_order"].dropna().nunique())

    work["_irep_cases_latest"] = np.where(work["_surv_order"] == latest_order, work["_irep_cases"], 0.0)
    main = (
        work.groupby(valid_group_cols, dropna=False, as_index=False)
        .agg(
            Cas=("_irep_cases", "sum"),
            Décès=("_irep_deaths", "sum"),
            Lignes=("_irep_cases", "size"),
            Semaines_actives=("_surv_order", "nunique"),
            Cas_derniere_semaine=("_irep_cases_latest", "sum"),
        )
    )

    quality_rows = []
    for keys, df_group in work.groupby(valid_group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        promptitude, median_delay = _irep_group_timeliness(df_group, threshold_days)
        quality_row = {col: key for col, key in zip(valid_group_cols, keys)}
        quality_row["Complétude_%"] = _irep_group_completeness(df_group, completeness_required)
        quality_row["Promptitude_%"] = promptitude
        quality_row["Délai_notification_médian_j"] = median_delay
        quality_rows.append(quality_row)
    quality = pd.DataFrame(quality_rows)

    out = main.merge(quality, on=valid_group_cols, how="left")
    out["Semaines_silencieuses"] = np.maximum(int(meta["window_weeks"]) - out["Semaines_actives"], 0)
    out["Couverture_hebdo_%"] = np.where(
        meta["window_weeks"] > 0,
        (out["Semaines_actives"] / float(meta["window_weeks"])) * 100.0,
        np.nan,
    )

    trend_base = base_df.copy()
    for col in valid_group_cols:
        if col in trend_base.columns:
            if col == COL_PROV:
                trend_base[col] = _irep_clean_province_series(trend_base[col])
            else:
                trend_base[col] = _irep_clean_label_series(trend_base[col])
    trend_base = trend_base.dropna(subset=valid_group_cols)

    if trend_current_orders:
        trend_current = (
            trend_base[trend_base["_surv_order"].isin(trend_current_orders)]
            .groupby(valid_group_cols, dropna=False, as_index=False)
            .agg(Cas_tendance_courant=("_irep_cases", "sum"))
        )
        out = out.merge(trend_current, on=valid_group_cols, how="left")
    else:
        out["Cas_tendance_courant"] = np.nan

    if trend_previous_orders:
        trend_previous = (
            trend_base[trend_base["_surv_order"].isin(trend_previous_orders)]
            .groupby(valid_group_cols, dropna=False, as_index=False)
            .agg(Cas_tendance_précédent=("_irep_cases", "sum"))
        )
        out = out.merge(trend_previous, on=valid_group_cols, how="left")
        out["Ratio_tendance"] = out["Cas_tendance_courant"] / (out["Cas_tendance_précédent"].fillna(0.0) + 1.0)
    else:
        out["Cas_tendance_précédent"] = np.nan
        out["Ratio_tendance"] = np.nan

    lookups = _irep_build_population_lookups(pop_ref)
    out["Population_exposée"] = out.apply(
        lambda row: _irep_resolve_population(
            row,
            geography_level,
            lookups,
            valid_group_cols,
            denominator_mode=denominator_mode,
        ),
        axis=1,
    )
    out["Taux_attaque_%"] = np.where(
        out["Population_exposée"] > 0,
        (out["Cas"] / out["Population_exposée"]) * 100.0,
        np.nan,
    )
    incidence_col = meta["incidence_col"]
    out[incidence_col] = np.where(
        out["Population_exposée"] > 0,
        (out["Cas"] / out["Population_exposée"]) * float(incidence_multiplier),
        np.nan,
    )
    out["Létalité_%"] = np.where(out["Cas"] > 0, (out["Décès"] / out["Cas"]) * 100.0, np.nan)
    out["Risque_promptitude"] = 100.0 - out["Promptitude_%"]
    out["Risque_complétude"] = 100.0 - out["Complétude_%"]

    out["TrendScore"] = _irep_score_metric(out["Ratio_tendance"])
    out["IncidenceScore"] = _irep_score_metric(out[incidence_col])
    out["CFRScore"] = _irep_score_metric(out["Létalité_%"])
    out["PromptitudeScore"] = _irep_score_metric(out["Risque_promptitude"])
    out["CompletenessScore"] = _irep_score_metric(out["Risque_complétude"])

    score_cols = {
        "trend": "TrendScore",
        "incidence": "IncidenceScore",
        "cfr": "CFRScore",
        "timeliness": "PromptitudeScore",
        "completeness": "CompletenessScore",
    }

    def _row_irep(row: pd.Series) -> float:
        available = {k: weights.get(k, 0.0) for k, col in score_cols.items() if pd.notna(row.get(col))}
        if not available or sum(available.values()) <= 0:
            return np.nan
        weight_sum = sum(available.values())
        return float(
            sum((available[k] / weight_sum) * float(row[score_cols[k]]) for k in available)
        )

    out["IREP"] = out.apply(_row_irep, axis=1)
    out["Risque_cat"] = pd.cut(
        out["IREP"],
        bins=[-np.inf, 30, 60, 80, np.inf],
        labels=["Faible", "Modéré", "Élevé", "Très élevé"],
    )
    out["Facteurs_prioritaires"] = out.apply(_irep_factor_summary, axis=1)
    out["_irep_label"] = (
        out[valid_group_cols]
        .astype("string")
        .fillna("Non renseigné")
        .astype(str)
        .agg(" / ".join, axis=1)
    )

    meta["population_coverage"] = int(out["Population_exposée"].notna().sum())
    if out["Population_exposée"].notna().any():
        meta["total_population"] = float(out["Population_exposée"].dropna().sum())
    elif pop_ref is not None and not pop_ref.empty:
        meta["total_population"] = _irep_reference_population_total(
            pop_ref,
            out,
            geography_level,
            denominator_mode,
        )
    total_cases = pd.to_numeric(out["Cas"], errors="coerce").sum(skipna=True)
    meta["total_cases"] = float(total_cases) if pd.notna(total_cases) else np.nan
    if pd.notna(meta["total_population"]) and float(meta["total_population"]) > 0 and pd.notna(meta["total_cases"]):
        meta["global_attack"] = (float(meta["total_cases"]) / float(meta["total_population"])) * 100.0
        meta["global_incidence"] = (float(meta["total_cases"]) / float(meta["total_population"])) * float(incidence_multiplier)

    valid_rates = out.dropna(subset=["Population_exposée", "Taux_attaque_%", incidence_col]).copy()
    if not valid_rates.empty:
        top_attack = valid_rates.sort_values("Taux_attaque_%", ascending=False).iloc[0]
        top_incidence = valid_rates.sort_values(incidence_col, ascending=False).iloc[0]
        meta["top_attack_label"] = str(top_attack.get("_irep_label", ""))
        meta["top_attack_value"] = float(top_attack["Taux_attaque_%"])
        meta["top_incidence_label"] = str(top_incidence.get("_irep_label", ""))
        meta["top_incidence_value"] = float(top_incidence[incidence_col])

    ordered_cols = [
        *valid_group_cols,
        "Cas",
        "Décès",
        "Population_exposée",
        "Taux_attaque_%",
        incidence_col,
        "Létalité_%",
        "Promptitude_%",
        "Délai_notification_médian_j",
        "Complétude_%",
        "Semaines_actives",
        "Semaines_silencieuses",
        "Couverture_hebdo_%",
        "Cas_derniere_semaine",
        "Cas_tendance_courant",
        "Cas_tendance_précédent",
        "Ratio_tendance",
        "TrendScore",
        "IncidenceScore",
        "CFRScore",
        "PromptitudeScore",
        "CompletenessScore",
        "IREP",
        "Risque_cat",
        "Facteurs_prioritaires",
        "_irep_label",
    ]
    ordered_cols = [col for col in ordered_cols if col in out.columns]
    out = out[ordered_cols].sort_values(
        ["IREP", incidence_col, "Cas"],
        ascending=[False, False, False],
        na_position="last",
    )
    return out.reset_index(drop=True), meta


def _irep_reference_units_from_population(
    pop_ref: pd.DataFrame,
    geography_level: str,
    scope_provinces: set[str],
) -> pd.DataFrame:
    """Construit l'univers attendu à partir de la population de référence."""
    if pop_ref is None or pop_ref.empty:
        return pd.DataFrame()

    work = pop_ref.copy()
    if scope_provinces:
        work = work[work["_prov_norm"].isin(scope_provinces)].copy()
    if work.empty:
        return pd.DataFrame()

    if geography_level == "province":
        cols = ["Province_reference", "_prov_norm", "Population_reference"]
        work = work.dropna(subset=["_prov_norm"])
        if work.empty:
            return pd.DataFrame()
        if work["_zone_norm"].notna().any():
            work = (
                work.groupby(["Province_reference", "_prov_norm"], dropna=False, as_index=False)
                .agg(Population_reference=("Population_reference", "sum"))
            )
        else:
            work = (
                work.groupby(["Province_reference", "_prov_norm"], dropna=False, as_index=False)
                .agg(Population_reference=("Population_reference", "max"))
            )
        return work[cols]

    cols = [
        "Province_reference",
        "Zone_de_sante_reference",
        "_prov_norm",
        "_zone_norm",
        "Population_reference",
    ]
    work = work.dropna(subset=["_prov_norm", "_zone_norm"])
    if work.empty:
        return pd.DataFrame()
    return (
        work.groupby(
            ["Province_reference", "Zone_de_sante_reference", "_prov_norm", "_zone_norm"],
            dropna=False,
            as_index=False,
        )
        .agg(Population_reference=("Population_reference", "max"))
    )[cols]


def _irep_build_silence_table(
    window_df: pd.DataFrame,
    base_df: pd.DataFrame,
    *,
    group_cols: list[str],
    geography_level: str,
    pop_ref: pd.DataFrame,
) -> pd.DataFrame:
    """Repère les unités silencieuses sur la fenêtre et sur la dernière semaine."""
    valid_group_cols = [c for c in group_cols if c in base_df.columns]
    if base_df is None or base_df.empty or not valid_group_cols:
        return pd.DataFrame()

    def _unit_frame(source: pd.DataFrame) -> pd.DataFrame:
        frame = source.copy()
        for col in valid_group_cols:
            if col == COL_PROV:
                frame[col] = _irep_clean_province_series(frame[col])
            else:
                frame[col] = _irep_clean_label_series(frame[col])
        frame = frame.dropna(subset=valid_group_cols)
        if frame.empty:
            return pd.DataFrame()
        frame["_prov_norm"] = frame[COL_PROV].map(_irep_normalize_key) if COL_PROV in frame.columns else pd.NA
        if geography_level == "zone" and COL_ZS in frame.columns:
            frame["_zone_norm"] = frame[COL_ZS].map(_irep_normalize_key)
        else:
            frame["_zone_norm"] = pd.NA
        keep = [*valid_group_cols, "_prov_norm", "_zone_norm"]
        return frame[keep].drop_duplicates()

    base_units = _unit_frame(base_df)
    if base_units.empty:
        return pd.DataFrame()

    scope_provinces = set(base_units["_prov_norm"].dropna().astype(str).tolist())
    expected_units = _irep_reference_units_from_population(pop_ref, geography_level, scope_provinces)
    if not expected_units.empty:
        if geography_level == "province":
            expected_units = expected_units.rename(columns={"Province_reference": COL_PROV})
            expected_units[COL_PROV] = _irep_clean_province_series(expected_units[COL_PROV])
            expected_units["_zone_norm"] = pd.NA
        else:
            expected_units = expected_units.rename(
                columns={
                    "Province_reference": COL_PROV,
                    "Zone_de_sante_reference": COL_ZS,
                }
            )
            expected_units[COL_PROV] = _irep_clean_province_series(expected_units[COL_PROV])
            expected_units[COL_ZS] = _irep_clean_label_series(expected_units[COL_ZS])
        expected_cols = [*valid_group_cols, "_prov_norm", "_zone_norm", "Population_reference"]
        expected_units = expected_units[expected_cols].drop_duplicates()
        universe = pd.concat([base_units, expected_units], ignore_index=True, sort=False)
    else:
        universe = base_units.copy()

    latest_order = window_df["_surv_order"].max() if window_df is not None and not window_df.empty else np.nan
    window_units = _unit_frame(window_df) if window_df is not None and not window_df.empty else pd.DataFrame()
    latest_units = (
        _unit_frame(window_df[window_df["_surv_order"] == latest_order])
        if window_df is not None and not window_df.empty and pd.notna(latest_order)
        else pd.DataFrame()
    )

    merge_keys = ["_prov_norm"] if geography_level == "province" else ["_prov_norm", "_zone_norm"]
    if geography_level == "province":
        display_cols = [COL_PROV]
    else:
        display_cols = [c for c in [COL_PROV, COL_ZS] if c in valid_group_cols]

    if "Population_reference" in universe.columns:
        universe["_pop_priority"] = universe["Population_reference"].notna().astype(int)
        universe = universe.sort_values(["_pop_priority", *display_cols], ascending=[False, *([True] * len(display_cols))])
    else:
        universe = universe.sort_values(display_cols)
    universe = universe.drop_duplicates(subset=merge_keys, keep="first")
    if "_pop_priority" in universe.columns:
        universe = universe.drop(columns=["_pop_priority"])
    window_flag = window_units[merge_keys].drop_duplicates().assign(_seen_window=True) if not window_units.empty else pd.DataFrame(columns=[*merge_keys, "_seen_window"])
    latest_flag = latest_units[merge_keys].drop_duplicates().assign(_seen_latest=True) if not latest_units.empty else pd.DataFrame(columns=[*merge_keys, "_seen_latest"])

    out = universe.merge(window_flag, on=merge_keys, how="left").merge(latest_flag, on=merge_keys, how="left")
    out["Silence_fenêtre"] = ~out["_seen_window"].eq(True)
    out["Silence_dernière_semaine"] = ~out["_seen_latest"].eq(True)
    out = out[out["Silence_fenêtre"] | out["Silence_dernière_semaine"]].copy()
    if out.empty:
        return out

    keep = [*display_cols]
    if "Population_reference" in out.columns:
        keep.append("Population_reference")
    keep.extend(["Silence_fenêtre", "Silence_dernière_semaine"])
    sort_cols = ["Silence_fenêtre", "Silence_dernière_semaine"]
    ascending = [False, False]
    if "Population_reference" in keep:
        sort_cols.append("Population_reference")
        ascending.append(False)
    out = out[keep].sort_values(sort_cols, ascending=ascending, na_position="last")
    return out.reset_index(drop=True)


def _irep_display_columns(
    table: pd.DataFrame,
    geography_level: str,
    incidence_col: str,
) -> list[str]:
    """Détermine les colonnes à afficher dans la table de décision."""
    geo_cols = [COL_PROV] if geography_level == "province" else [c for c in [COL_PROV, COL_ZS] if c in table.columns]
    cols = [
        *geo_cols,
        "Cas",
        "Population_exposée",
        "Taux_attaque_%",
        incidence_col,
        "Létalité_%",
        "Promptitude_%",
        "Délai_notification_médian_j",
        "Complétude_%",
        "Semaines_actives",
        "Semaines_silencieuses",
        "Cas_derniere_semaine",
        "Ratio_tendance",
        "IREP",
        "Risque_cat",
        "Facteurs_prioritaires",
    ]
    return [col for col in cols if col in table.columns]


def _irep_format_display_table(table: pd.DataFrame, geography_level: str) -> pd.DataFrame:
    """Prépare la vue tabulaire affichée à l'utilisateur."""
    if table is None or table.empty:
        return pd.DataFrame()
    incidence_candidates = [c for c in table.columns if str(c).startswith("Incidence_pour_")]
    incidence_col = incidence_candidates[0] if incidence_candidates else "Incidence_pour_100000"
    display = table[_irep_display_columns(table, geography_level, incidence_col)].copy()
    incidence_suffix = incidence_col.replace("Incidence_pour_", "") if incidence_col.startswith("Incidence_pour_") else "100000"
    rename_map = {
        COL_PROV: "Province",
        COL_ZS: "Zone de santé",
        "Population_exposée": "Population exposée",
        "Taux_attaque_%": "Taux d'attaque (%)",
        incidence_col: f"Incidence / {incidence_suffix}",
        "Létalité_%": "Létalité (%)",
        "Promptitude_%": "Promptitude (%)",
        "Délai_notification_médian_j": "Délai médian notif. (j)",
        "Complétude_%": "Complétude (%)",
        "Semaines_actives": "Semaines actives",
        "Semaines_silencieuses": "Semaines silencieuses",
        "Cas_derniere_semaine": "Cas dernière semaine",
        "Ratio_tendance": "Ratio tendance",
        "Risque_cat": "Catégorie de risque",
        "Facteurs_prioritaires": "Facteurs prioritaires",
    }
    display = display.rename(columns=rename_map)
    numeric_cols = [
        "Population exposée",
        "Taux d'attaque (%)",
        f"Incidence / {incidence_suffix}",
        "Létalité (%)",
        "Promptitude (%)",
        "Délai médian notif. (j)",
        "Complétude (%)",
        "Ratio tendance",
        "IREP",
    ]
    for col in numeric_cols:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(1)
    if "Population exposée" in display.columns:
        display["Population exposée"] = pd.to_numeric(display["Population exposée"], errors="coerce").round(0)
    return display


def _irep_render_geo_window(
    *,
    title_prefix: str,
    geography_label: str,
    geography_level: str,
    table: pd.DataFrame,
    silence_table: pd.DataFrame,
    meta: dict,
    top_n: int,
    threshold_days: float,
    quality_threshold: int,
    download_suffix: str,
    incidence_multiplier: int,
) -> None:
    """Affiche un bloc décisionnel IREP pour un niveau géographique."""
    if table is None or table.empty:
        st.info(f"Aucune donnée exploitable pour l'analyse IREP par {geography_label.lower()}.")
        return

    if "_irep_label" not in table.columns:
        label_cols = [COL_PROV] if geography_level == "province" else [c for c in [COL_PROV, COL_ZS] if c in table.columns]
        table = table.copy()
        if label_cols:
            table["_irep_label"] = (
                table[label_cols]
                .astype("string")
                .fillna("Non renseigné")
                .astype(str)
                .agg(" / ".join, axis=1)
            )
        else:
            table["_irep_label"] = "Non renseigné"

    st.caption(
        f"{title_prefix} | Lecture {geography_label.lower()} : un faible nombre de cas peut rester prioritaire "
        "si l'incidence, la promptitude ou la complétude augmentent le risque."
    )

    total_population = meta.get("total_population", np.nan)
    incidence_col = str(meta.get("incidence_col", f"Incidence_pour_{int(incidence_multiplier)}"))
    incidence_label = f"Incidence / {int(incidence_multiplier):,}"
    incidence_max = pd.to_numeric(table[incidence_col], errors="coerce").max() if incidence_col in table.columns else np.nan
    completeness_median = pd.to_numeric(table["Complétude_%"], errors="coerce").median() if "Complétude_%" in table.columns else np.nan
    silent_latest_count = int(silence_table["Silence_dernière_semaine"].fillna(False).sum()) if not silence_table.empty and "Silence_dernière_semaine" in silence_table.columns else 0
    global_attack = meta.get("global_attack", np.nan)
    global_incidence = meta.get("global_incidence", np.nan)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Unités classées", format_metric_value(len(table)))
    k2.metric("Cas", format_metric_value(pd.to_numeric(table["Cas"], errors="coerce").sum()))
    k3.metric("Population couverte", "-" if pd.isna(total_population) else f"{int(round(float(total_population))):,}")
    k4.metric(incidence_label, "-" if pd.isna(global_incidence) else f"{float(global_incidence):.2f}")
    k5.metric("Complétude médiane (%)", "-" if pd.isna(completeness_median) else f"{float(completeness_median):.1f}")
    k6.metric("Silencieuses dernière semaine", format_metric_value(silent_latest_count))

    denominator_label = (
        "Population maximale du groupe"
        if str(meta.get("denominator_mode")) == "group_max"
        else "Somme des populations uniques par ZS"
    )
    if pd.notna(global_attack) or pd.notna(global_incidence):
        attack_txt = "-" if pd.isna(global_attack) else f"{float(global_attack):.3f}%"
        incidence_txt = "-" if pd.isna(global_incidence) else f"{float(global_incidence):.2f} pour {int(incidence_multiplier):,}"
        st.info(
            f"Lecture automatique : sur cette fenêtre, le taux d'attaque global est **{attack_txt}** "
            f"et l'incidence globale est **{incidence_txt} habitants**. "
            f"Dénominateur utilisé : **{denominator_label}**."
        )

    top_attack_label = meta.get("top_attack_label")
    top_incidence_label = meta.get("top_incidence_label")
    top_attack_value = meta.get("top_attack_value", np.nan)
    top_incidence_value = meta.get("top_incidence_value", np.nan)
    if top_attack_label or top_incidence_label:
        fragments = []
        if top_attack_label and pd.notna(top_attack_value):
            fragments.append(
                f"Le taux d'attaque le plus élevé est observé pour **{top_attack_label}** ({float(top_attack_value):.3f}%)."
            )
        if top_incidence_label and pd.notna(top_incidence_value):
            fragments.append(
                f"L'incidence la plus élevée est observée pour **{top_incidence_label}** "
                f"({float(top_incidence_value):.2f} pour {int(incidence_multiplier):,} habitants)."
            )
        if fragments:
            st.caption(" ".join(fragments))

    chart_cols = st.columns([1.25, 1.0])
    with chart_cols[0]:
        bar_df = table.head(int(top_n)).sort_values("IREP", ascending=True).copy()
        fig = px.bar(
            bar_df,
            x="IREP",
            y="_irep_label",
            orientation="h",
            color="Risque_cat" if "Risque_cat" in bar_df.columns else None,
            title=f"Top {min(int(top_n), len(bar_df))} {geography_label.lower()} selon l'IREP",
            hover_data={
                "Cas": True,
                incidence_col: ":.2f" if incidence_col in bar_df.columns else False,
                "Promptitude_%": ":.1f" if "Promptitude_%" in bar_df.columns else False,
                "Complétude_%": ":.1f" if "Complétude_%" in bar_df.columns else False,
                "_irep_label": False,
            },
        )
        fig.update_layout(xaxis_title="IREP", yaxis_title=geography_label)
        try:
            st_plot(fig, key=f"irep_bar_{download_suffix}")
        except Exception:
            st.plotly_chart(fig, width="stretch", key=f"irep_bar_{download_suffix}")

    with chart_cols[1]:
        focus_col = incidence_col if incidence_col in table.columns and table[incidence_col].notna().any() else "Complétude_%"
        focus_title = incidence_label if focus_col == incidence_col else "Complétude (%)"
        focus_df = (
            table.dropna(subset=[focus_col])
            .sort_values(focus_col, ascending=False if focus_col == incidence_col else True)
            .head(int(top_n))
            .sort_values(focus_col, ascending=True)
            .copy()
        )
        if focus_df.empty:
            st.info("Aucun second axe de lecture n'est disponible pour cette fenêtre.")
        else:
            fig_focus = px.bar(
                focus_df,
                x=focus_col,
                y="_irep_label",
                orientation="h",
                color="IREP",
                title=f"Lecture complémentaire : {focus_title}",
                color_continuous_scale=["#dbe8f9", "#b91c1c"],
            )
            fig_focus.update_layout(coloraxis_colorbar_title="IREP", xaxis_title=focus_title, yaxis_title=geography_label)
            try:
                st_plot(fig_focus, key=f"irep_focus_{download_suffix}")
            except Exception:
                st.plotly_chart(fig_focus, width="stretch", key=f"irep_focus_{download_suffix}")

    display_table = _irep_format_display_table(table, geography_level)
    st.dataframe(display_table, width="stretch", height=420, hide_index=True)

    quality_flags = table[
        (pd.to_numeric(table["Complétude_%"], errors="coerce") < float(quality_threshold))
        | (pd.to_numeric(table["Promptitude_%"], errors="coerce") < float(quality_threshold))
    ].copy()
    quality_display = _irep_format_display_table(
        quality_flags.sort_values(["IREP", "Complétude_%"], ascending=[False, True], na_position="last").head(max(10, int(top_n))),
        geography_level,
    )

    detail_cols = st.columns([1.1, 1.0])
    with detail_cols[0]:
        st.markdown("**Unités silencieuses à vérifier**")
        if silence_table.empty:
            st.success("Aucune unité silencieuse détectée sur cette fenêtre.")
        else:
            silence_display = silence_table.rename(
                columns={
                    COL_PROV: "Province",
                    COL_ZS: "Zone de santé",
                    "Population_reference": "Population exposée",
                    "Silence_fenêtre": "Silence fenêtre",
                    "Silence_dernière_semaine": "Silence dernière semaine",
                }
            ).copy()
            if "Population exposée" in silence_display.columns:
                silence_display["Population exposée"] = pd.to_numeric(silence_display["Population exposée"], errors="coerce").round(0)
            st.dataframe(silence_display.head(60), width="stretch", height=300, hide_index=True)

    with detail_cols[1]:
        st.markdown(f"**Qualité ou promptitude < {quality_threshold}%**")
        if quality_display.empty:
            st.success("Aucune unité active ne cumule de faibles performances de qualité sur la fenêtre.")
        else:
            st.dataframe(quality_display, width="stretch", height=300, hide_index=True)

    st.caption(
        f"Promptitude (%) = part des notifications arrivées en {threshold_days:g} jour(s) ou moins. "
        "Les unités silencieuses ou avec faible complétude méritent une relecture avant toute conclusion opérationnelle."
    )

    st.download_button(
        f"Télécharger IREP {geography_label.lower()} (CSV)",
        data=df_to_csv_bytes(table),
        file_name=f"irep_{download_suffix}.csv",
        mime="text/csv",
    )


def _irep_render_window(
    *,
    title: str,
    intro: str,
    window_df: pd.DataFrame,
    base_df: pd.DataFrame,
    pop_ref: pd.DataFrame,
    trend_current_orders: list[float],
    trend_previous_orders: list[float],
    completeness_required: list[str],
    threshold_days: float,
    weights: dict,
    denominator_mode: str = "zs_sum",
    incidence_multiplier: int = 100000,
    quality_threshold: int = 80,
    top_province_n: int = 10,
    top_zs_n: int = 15,
    download_suffix: str = "irep",
) -> None:
    """Affiche l'IREP sur une fenêtre temporelle donnée."""
    st.markdown(f"### {title}")
    st.caption(intro)

    if window_df is None or window_df.empty:
        st.info("Aucune donnée n'est disponible pour cette fenêtre.")
        return

    geo_tabs = st.tabs(["Province", "Zone de santé"])
    with geo_tabs[0]:
        province_table, province_meta = _irep_build_window_risk_table(
            window_df,
            base_df,
            group_cols=[COL_PROV],
            geography_level="province",
            pop_ref=pop_ref,
            trend_current_orders=trend_current_orders,
            trend_previous_orders=trend_previous_orders,
            completeness_required=completeness_required,
            threshold_days=threshold_days,
            weights=weights,
            denominator_mode=denominator_mode,
            incidence_multiplier=incidence_multiplier,
        )
        province_silence = _irep_build_silence_table(
            window_df,
            base_df,
            group_cols=[COL_PROV],
            geography_level="province",
            pop_ref=pop_ref,
        )
        _irep_render_geo_window(
            title_prefix=intro,
            geography_label="Province",
            geography_level="province",
            table=province_table,
            silence_table=province_silence,
            meta=province_meta,
            top_n=top_province_n,
            threshold_days=threshold_days,
            quality_threshold=quality_threshold,
            download_suffix=f"{download_suffix}_province",
            incidence_multiplier=incidence_multiplier,
        )

    with geo_tabs[1]:
        zone_table, zone_meta = _irep_build_window_risk_table(
            window_df,
            base_df,
            group_cols=[COL_PROV, COL_ZS],
            geography_level="zone",
            pop_ref=pop_ref,
            trend_current_orders=trend_current_orders,
            trend_previous_orders=trend_previous_orders,
            completeness_required=completeness_required,
            threshold_days=threshold_days,
            weights=weights,
            denominator_mode=denominator_mode,
            incidence_multiplier=incidence_multiplier,
        )
        zone_silence = _irep_build_silence_table(
            window_df,
            base_df,
            group_cols=[COL_PROV, COL_ZS],
            geography_level="zone",
            pop_ref=pop_ref,
        )
        _irep_render_geo_window(
            title_prefix=intro,
            geography_label="Zone de santé",
            geography_level="zone",
            table=zone_table,
            silence_table=zone_silence,
            meta=zone_meta,
            top_n=top_zs_n,
            threshold_days=threshold_days,
            quality_threshold=quality_threshold,
            download_suffix=f"{download_suffix}_zone_sante",
            incidence_multiplier=incidence_multiplier,
        )


def render_irep_tab(ctx: dict) -> None:
    """Render the decision-oriented IREP tab."""
    globals().update(ctx)
    render_section_title(13, "Indice composite de risque épidémique (IREP)")
    render_tab_narrative("irep")
    tab_help(
        "Lecture et interprétation",
        """
        **Objectif** : classer les provinces et zones de santé en combinant le volume de cas, la population exposée,
        la létalité, la promptitude de notification et la complétude des données.

        **Logique de lecture**
        - **Situation hebdomadaire** : lecture de la dernière semaine visible.
        - **Situation des 4 dernières semaines** : lecture de la tendance récente.
        - **Situation cumulée** : consolidation de toute la fenêtre active.

        **Approche analytique attendue**
        - Ne pas s'arrêter aux cas bruts.
        - Vérifier l'incidence, la qualité des données, les zones silencieuses et les délais de notification.
        - Une province avec moins de cas peut rester plus prioritaire si le risque relatif est plus élevé.
        """,
        expanded=False,
    )

    source_df = df_f if "df_f" in globals() and isinstance(df_f, pd.DataFrame) else df
    if source_df is None or source_df.empty:
        render_absence_narrative("risk")
        return

    df_irep_scope, reference = _irep_prepare_analysis_scope(source_df)
    if df_irep_scope.empty or reference.empty:
        st.info("Aucune semaine épidémiologique valide n'est disponible pour calculer l'IREP.")
        return

    st.caption(
        "Le classement ci-dessous aide à passer d'une lecture descriptive des cas à une lecture décisionnelle du risque."
    )
    with st.expander("Comprendre les calculs de taux d’attaque et d’incidence", expanded=False):
        st.markdown(
            """
**Taux d’attaque**  
Proportion de nouveaux cas parmi la population exposée sur la période analysée.

```text
Taux d’attaque (%) = (Nombre de nouveaux cas / Population exposée ou à risque) × 100
```

**Incidence**  
Fréquence des nouveaux cas dans une population pendant une période donnée. Pour comparer des territoires, on l’exprime souvent pour 1 000, 10 000 ou 100 000 habitants.

```text
Incidence = (Nombre de nouveaux cas / Population à risque) × multiplicateur
```

**Point de vigilance**  
Le résultat dépend fortement du dénominateur population. Si la population de référence est portée au niveau zone de santé, il faut éviter de la sommer plusieurs fois.
"""
        )

    pop_upload = st.file_uploader(
        "Remplacer le fichier population par défaut (optionnel)",
        type=["xlsx", "xls", "csv"],
        key="irep_population_upload",
        help="Par défaut, l'onglet utilise `data/RDC_Zone_de_sante_OCHA.xlsx` pour agréger la population par zone et par province.",
    )
    pop_ref, pop_source_label = _irep_load_population_reference(pop_upload)
    pop_lookup = _irep_build_population_lookups(pop_ref)

    st.markdown("### Paramètres de calcul")
    cfg1, cfg2, cfg3, cfg4, cfg5 = st.columns([1.2, 1.0, 1.0, 1.0, 0.9])
    with cfg1:
        st.caption(pop_source_label)
        if pop_ref.empty:
            st.warning("Population non disponible : l'incidence et le taux d'attaque seront partiellement indisponibles.")
        else:
            st.success(
                f"Population prête : {len(pop_lookup['province'])} provinces et {len(pop_lookup['zone_pair']) or len(pop_lookup['zone_name'])} zones référencées."
            )
    with cfg2:
        denominator_label = st.selectbox(
            "Dénominateur population",
            options=["Somme des populations uniques par ZS", "Population maximale du groupe"],
            index=0,
            key="irep_denominator_mode",
            help=(
                "Somme des populations uniques par ZS : adaptée quand la population de référence est au niveau zone de santé. "
                "Population maximale du groupe : utile si le fichier chargé contient déjà une population agrégée répétée."
            ),
        )
    with cfg3:
        incidence_multiplier = int(
            st.selectbox(
                "Incidence pour",
                options=[1000, 10000, 100000],
                index=2,
                key="irep_incidence_multiplier",
            )
        )
    with cfg4:
        threshold_days = st.number_input(
            "Seuil promptitude (jours)",
            min_value=0.0,
            max_value=30.0,
            value=float(get_session_int("seuil_jours", 2)),
            step=1.0,
            key="irep_threshold_days",
        )
    with cfg5:
        quality_threshold = st.slider(
            "Seuil d'alerte qualité (%)",
            min_value=0,
            max_value=100,
            value=80,
            step=5,
            key="irep_quality_threshold",
        )

    w1, w2, w3, w4, w5 = st.columns(5)
    with w1:
        w_trend = st.slider("Poids Tendance", 0.0, 1.0, 0.30, 0.05, key="irep_w_trend")
    with w2:
        w_inc = st.slider("Poids Incidence", 0.0, 1.0, 0.25, 0.05, key="irep_w_inc")
    with w3:
        w_cfr = st.slider("Poids Létalité", 0.0, 1.0, 0.20, 0.05, key="irep_w_cfr")
    with w4:
        w_time = st.slider("Poids Promptitude", 0.0, 1.0, 0.15, 0.05, key="irep_w_time")
    with w5:
        w_comp = st.slider("Poids Complétude", 0.0, 1.0, 0.10, 0.05, key="irep_w_comp")

    completeness_candidates = [
        c
        for c in [
            COL_PROV,
            COL_ZS,
            COL_SEX,
            COL_AGE,
            DATE_ONSET,
            DATE_NOTIF,
            COL_ISSUE,
            COL_CLASS,
            DATE_ADM,
            DATE_PREL,
        ]
        if c in df_irep_scope.columns
    ]
    default_completeness = [c for c in [COL_PROV, COL_ZS] if c in completeness_candidates]

    p1, p2, p3 = st.columns([1.4, 1.0, 1.0])
    with p1:
        completeness_required = st.multiselect(
            "Champs critiques pour la complétude",
            options=completeness_candidates,
            default=default_completeness,
            key="irep_completeness_fields",
        )
    with p2:
        top_province_n = st.number_input(
            "Top provinces affichées",
            min_value=3,
            max_value=30,
            value=10,
            step=1,
            key="irep_top_province_n",
        )
    with p3:
        top_zs_n = st.number_input(
            "Top zones affichées",
            min_value=5,
            max_value=50,
            value=15,
            step=1,
            key="irep_top_zs_n",
        )

    weights = {
        "trend": float(w_trend),
        "incidence": float(w_inc),
        "cfr": float(w_cfr),
        "timeliness": float(w_time),
        "completeness": float(w_comp),
    }
    if sum(weights.values()) <= 0:
        st.warning("Les poids ne peuvent pas tous être nuls. Les poids par défaut sont réappliqués.")
        weights = {
            "trend": 0.30,
            "incidence": 0.25,
            "cfr": 0.20,
            "timeliness": 0.15,
            "completeness": 0.10,
        }
    denominator_mode = "group_max" if denominator_label == "Population maximale du groupe" else "zs_sum"

    latest_order = reference["order"].iloc[-1]
    latest_label = str(reference["label"].iloc[-1])
    last4_reference = reference.tail(4).copy()
    last4_orders = last4_reference["order"].tolist()
    last4_labels = last4_reference["label"].astype(str).tolist()
    first_label = str(reference["label"].iloc[0])

    prev_week_orders = reference.tail(2).head(1)["order"].tolist() if len(reference) >= 2 else []
    prev4_reference = reference.iloc[max(len(reference) - 8, 0): max(len(reference) - 4, 0)].copy() if len(reference) > 4 else pd.DataFrame()
    prev4_orders = prev4_reference["order"].tolist() if not prev4_reference.empty else []

    df_latest_week = df_irep_scope[df_irep_scope["_surv_order"] == latest_order].copy()
    df_last4_weeks = df_irep_scope[df_irep_scope["_surv_order"].isin(last4_orders)].copy()

    _irep_render_window(
        title="1. Situation hebdomadaire",
        intro=f"Semaine la plus récente visible : {latest_label}. Le score compare le signal actuel à la semaine précédente quand elle existe.",
        window_df=df_latest_week,
        base_df=df_irep_scope,
        pop_ref=pop_ref,
        trend_current_orders=[latest_order],
        trend_previous_orders=prev_week_orders,
        completeness_required=completeness_required,
        threshold_days=float(threshold_days),
        weights=weights,
        denominator_mode=denominator_mode,
        incidence_multiplier=incidence_multiplier,
        quality_threshold=int(quality_threshold),
        top_province_n=int(top_province_n),
        top_zs_n=int(top_zs_n),
        download_suffix="hebdomadaire",
    )

    st.divider()

    _irep_render_window(
        title="2. Situation des 4 dernières semaines",
        intro=f"Lecture glissante sur {len(last4_labels)} semaine(s) : {', '.join(last4_labels)}. La tendance compare cette fenêtre aux 4 semaines précédentes quand elles existent.",
        window_df=df_last4_weeks,
        base_df=df_irep_scope,
        pop_ref=pop_ref,
        trend_current_orders=last4_orders,
        trend_previous_orders=prev4_orders,
        completeness_required=completeness_required,
        threshold_days=float(threshold_days),
        weights=weights,
        denominator_mode=denominator_mode,
        incidence_multiplier=incidence_multiplier,
        quality_threshold=int(quality_threshold),
        top_province_n=int(top_province_n),
        top_zs_n=int(top_zs_n),
        download_suffix="quatre_semaines",
    )

    st.divider()

    recent4_orders = last4_orders
    _irep_render_window(
        title="3. Situation cumulée",
        intro=f"Cumul de toute la fenêtre active : {first_label} à {latest_label}. Le score garde un regard sur l'accélération récente via les 4 dernières semaines.",
        window_df=df_irep_scope,
        base_df=df_irep_scope,
        pop_ref=pop_ref,
        trend_current_orders=recent4_orders,
        trend_previous_orders=prev4_orders,
        completeness_required=completeness_required,
        threshold_days=float(threshold_days),
        weights=weights,
        denominator_mode=denominator_mode,
        incidence_multiplier=incidence_multiplier,
        quality_threshold=int(quality_threshold),
        top_province_n=int(top_province_n),
        top_zs_n=int(top_zs_n),
        download_suffix="cumule",
    )
