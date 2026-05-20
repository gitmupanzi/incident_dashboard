import html

from dashboard_app.domain import (
    AGE_UNIT_DAY_PATTERN,
    AGE_UNIT_MONTH_PATTERN,
    AGE_UNIT_WEEK_PATTERN,
    AGE_UNIT_YEAR_PATTERN,
    APP_BUILD_TAG,
    Any,
    BytesIO,
    COLOR_CASES,
    COLOR_CFR,
    COLOR_DEATHS,
    COLOR_FEMININ,
    COLOR_INCONNU,
    COLOR_MASCULIN,
    COL_AGE,
    COL_AGEG,
    COL_AGEG2,
    COL_AS,
    COL_CLASS,
    COL_DEHY,
    COL_HOSP,
    COL_ISSUE,
    COL_PREL,
    COL_PROV,
    COL_SEX,
    COL_TDR,
    COL_TDRR,
    COL_UNIT,
    COL_WEEK,
    COL_WNUM,
    COL_YEAR,
    COL_ZS,
    DATE_ADM,
    DATE_CONS,
    DATE_INV,
    DATE_ISSUE,
    DATE_NOTIF,
    DATE_ONSET,
    DATE_PREL,
    DATE_RECEP,
    DATE_RES,
    DISEASE_SPECS,
    Dict,
    EPIDEMIE,
    HAS_CUSTOM_VIZ,
    HAS_RAPIDFUZZ,
    Iterable,
    List,
    MAP_ANNOTATION_MODE_OPTIONS,
    METRIC_COLUMN_ALIASES,
    MISSING_LABEL,
    MISSING_LABEL_VERBOSE,
    MultiPolygon,
    Optional,
    PROVINCES_END,
    PROVINCES_EPID,
    PROVINCE_PATTERNS,
    Path,
    SEX_COLOR_MAP,
    STANDARD_DELAY_LABELS,
    SequenceMatcher,
    TDR_NEG_SET,
    TDR_POS_SET,
    Tuple,
    Union,
    YES_SET,
    alerts_weekly_simple,
    apply_plotly_value_annotations,
    as_list,
    build_cases_deaths_cfr_pivot,
    build_delay_group_summary,
    build_interactive_geo_map,
    build_operational_risk_score,
    build_recommended_fields_matrix,
    build_standard_action_tracker_template,
    build_standard_signal_table,
    build_spatiotemporal_cluster_table,
    build_standard_delay_summary,
    build_weekly_alerts,
    build_weekly_cases_cfr_combo,
    build_weekly_cases_deaths_combo,
    build_weekly_multiline_by_group,
    call_optional_function,
    carte_statique_matplotlib,
    cascade_metrics,
    choose_week_column,
    clean_str,
    compile_from_folder,
    completeness_table,
    compter_par_categorie,
    compute_group_indicators,
    compute_indicators,
    ctx,
    date,
    datetime,
    delay_days,
    df_to_csv_bytes,
    duplicate_candidates_table,
    enrich_fuzzy_geo_map_labels,
    ensure_lower,
    export_sitrep_pdf,
    extraire_numero,
    extraire_ordre_tranche,
    flatten_columns,
    fmt_yw_label,
    gdf_to_plotly_geojson,
    get_clicked_map_label,
    get_provinces_epid,
    get_selected_map_point,
    get_session_int,
    get_toggle_flag,
    glob,
    go,
    gpd,
    graphique_barres_facette,
    graphique_pyramide_age,
    group_cfr,
    group_rate,
    hashlib,
    inject_professional_dashboard_css,
    is_death,
    is_disease_enabled,
    is_numeric_dtype,
    is_positive,
    joindre_donnees_fuzzy_geo,
    json,
    list_available_standard_delays,
    load_data_from_excel,
    logger,
    logging,
    make_unique,
    make_yw,
    merge_standard_action_tracker_template,
    norm_yesno,
    np,
    ordered_weeks_from_weekly_sorted,
    orient,
    os,
    pct_change_metric_safe,
    pct_change_safe,
    pct_under_threshold,
    pd,
    pick_age_col,
    plot_barres_pct_sous_seuil,
    plot_boxplot_delais_plotly,
    plot_camembert_interactif,
    plot_courbe_par_categories_plotly,
    plot_courbe_plotly,
    plot_evolution_multi_auto,
    plot_histogramme_groupe_interactif_empile,
    plot_pyramide_symetrique,
    plt,
    prepare_idsr_numeric,
    px,
    qc_flags,
    re,
    render_footer,
    render_pivot_with_cfr,
    render_professional_header,
    render_section_title,
    render_standards_note,
    reorder_pivot_weeks,
    safe_pct,
    safe_to_datetime,
    st,
    st_dataframe_safe,
    st_plot,
    standard_data_quality_summary,
    standardize_df,
    standardize_ll_by_disease,
    standardize_ll_core,
    taux_binaire,
    tempfile,
    unicodedata,
    verifier_presence_colonnes,
)
from dashboard_app.domain import (
    _is_yes_series,
    _norm_key,
    _normalize_province_name_for_matching,
    _resolve_map_filter_value,
    _tdr_result_norm,
)
from dashboard_app.core import _normalize_metric_alias_columns

def tab_help(title: str, md: str, expanded: bool = False):
    with st.expander(f"ℹ {title}", expanded=expanded):
        st.markdown(md)


def _normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def infer_age_years_generic(df_: pd.DataFrame) -> pd.Series:
    age_raw = df_[COL_AGE] if (COL_AGE in df_.columns) else pd.Series([pd.NA] * len(df_), index=df_.index)
    age_num = pd.to_numeric(age_raw, errors="coerce")

    if COL_UNIT in df_.columns:
        unit_raw = df_[COL_UNIT].astype("string").str.lower().str.strip()
    else:
        unit_raw = pd.Series([pd.NA] * len(df_), index=df_.index, dtype="string")

    age_txt = age_raw.astype("string").str.lower()
    extracted_num = age_txt.str.extract(r"(?P<num>\d+(?:[\.,]\d+)?)")["num"].str.replace(",", ".", regex=False)
    extracted_num = pd.to_numeric(extracted_num, errors="coerce")
    age_val = age_num.where(age_num.notna(), extracted_num)

    unit_from_age = pd.Series([pd.NA] * len(df_), index=df_.index, dtype="string")
    unit_from_age = unit_from_age.mask(age_txt.str.contains(AGE_UNIT_MONTH_PATTERN, na=False), "mois")
    unit_from_age = unit_from_age.mask(age_txt.str.contains(AGE_UNIT_WEEK_PATTERN, na=False), "semaine")
    unit_from_age = unit_from_age.mask(age_txt.str.contains(AGE_UNIT_DAY_PATTERN, na=False), "jour")
    unit_from_age = unit_from_age.mask(age_txt.str.contains(AGE_UNIT_YEAR_PATTERN, na=False), "an")

    unit = unit_raw.where(unit_raw.notna(), unit_from_age)

    years = pd.Series(np.nan, index=df_.index, dtype="float")
    years = years.mask(unit.str.contains(AGE_UNIT_MONTH_PATTERN, na=False), age_val / 12.0)
    years = years.mask(unit.str.contains(AGE_UNIT_WEEK_PATTERN, na=False), age_val / 52.0)
    years = years.mask(unit.str.contains(AGE_UNIT_DAY_PATTERN, na=False), age_val / 365.25)
    years = years.mask(unit.str.contains(AGE_UNIT_YEAR_PATTERN, na=False), age_val)
    years = years.mask(years.isna() & age_val.notna(), age_val)
    return pd.to_numeric(years, errors="coerce")


def derive_age_4cat_generic(df_: pd.DataFrame) -> pd.Series:
    years = infer_age_years_generic(df_)
    out = pd.Series(pd.NA, index=df_.index, dtype="string")
    out = out.mask((years >= 0) & (years < 1), "0-11 mois")
    out = out.mask((years >= 1) & (years < 5), "12-59 mois")
    out = out.mask((years >= 5) & (years <= 15), "5-15 ans")
    out = out.mask(years > 15, ">15 ans")
    return out


def derive_age_5yr_generic(df_: pd.DataFrame) -> pd.Series:
    years = infer_age_years_generic(df_)
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, np.inf]
    labels = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60+"]
    return pd.cut(years, bins=bins, labels=labels, right=False).astype("string")


@st.cache_data(show_spinner=False)
def build_global_summary_table(df_: pd.DataFrame) -> pd.DataFrame:
    n_cases = int(len(df_))
    n_deaths = int(df_["is_death"].sum()) if "is_death" in df_.columns else 0
    cfr = (n_deaths / n_cases * 100.0) if n_cases else np.nan
    rows = [("Nombre total de cas", n_cases), ("Nombre total de décès", n_deaths), ("Létalité (%)", None if pd.isna(cfr) else round(cfr, 2))]
    for col, label in [(COL_PROV, "Nombre de provinces touchées"), (COL_ZS, "Nombre de zones de santé touchées"), (COL_AS, "Nombre d'aires de santé touchées")]:
        if col in df_.columns:
            rows.append((label, int(df_[col].dropna().nunique())))
    analysis_period = compute_analysis_period_value(df_)
    if analysis_period != "-":
        analysis_period = format_range_label_for_display(analysis_period)
    if analysis_period != "-":
        rows.append(("Période analysée", analysis_period))

    week_coverage = _build_week_coverage_text(df_)
    if week_coverage:
        week_coverage = format_range_label_for_display(week_coverage)
    if week_coverage:
        rows.append(("Semaines épidémiologiques couvertes", week_coverage))
    return pd.DataFrame(rows, columns=["Indicateur", "Valeur"])


@st.cache_data(show_spinner=False)
def build_frequency_table(df_: pd.DataFrame, col: str, top_n: int | None = None) -> pd.DataFrame:
    if col not in df_.columns:
        return pd.DataFrame(columns=[col, 'n', '%'])
    freq = df_[col].fillna('Inconnu').astype(str).str.strip().replace('', 'Inconnu').value_counts(dropna=False).reset_index()
    freq.columns = [col, 'n']
    freq['%'] = (freq['n'] / max(len(df_), 1) * 100).round(1)
    if top_n is not None:
        freq = freq.head(int(top_n))
    return freq


@st.cache_data(show_spinner=False)
def build_simple_lab_table(df_: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_cases = int(len(df_))
    if COL_PREL in df_.columns:
        n_prel = int(_is_yes_series(df_[COL_PREL]).sum())
        rows.append(("Prélèvement réalisé", n_prel, round(n_prel / n_cases * 100, 1) if n_cases else np.nan))
    if COL_TDR in df_.columns and df_[COL_TDR].notna().any():
        n_tdr = int(_is_yes_series(df_[COL_TDR]).sum())
        rows.append(("TDR réalisé", n_tdr, round(n_tdr / n_cases * 100, 1) if n_cases else np.nan))
    result_col = None
    if COL_TDRR in df_.columns and df_[COL_TDRR].notna().any():
        result_col = COL_TDRR
    elif "Resultat_labo" in df_.columns and df_["Resultat_labo"].notna().any():
        result_col = "Resultat_labo"
        rows.append(("R\u00e9sultat labo renseign\u00e9", int(df_["Resultat_labo"].notna().sum()), round(int(df_["Resultat_labo"].notna().sum()) / n_cases * 100, 1) if n_cases else np.nan))
    if result_col is not None:
        res_n = _tdr_result_norm(df_[result_col])
        n_pos = int(res_n.isin(TDR_POS_SET).sum())
        n_neg = int(res_n.isin(TDR_NEG_SET).sum())
        n_valid = n_pos + n_neg
        rows.append(("Résultat valide (Pos/Nég)", n_valid, round(n_valid / n_cases * 100, 1) if n_cases else np.nan))
        rows.append(("TDR positif", n_pos, round(n_pos / n_valid * 100, 1) if n_valid else np.nan))
        rows.append(("TDR négatif", n_neg, round(n_neg / n_valid * 100, 1) if n_valid else np.nan))
    df_out = pd.DataFrame(rows, columns=["Indicateur labo", "n", "%"])
    if result_col == "Resultat_labo":
        df_out["Indicateur labo"] = df_out["Indicateur labo"].replace({
            "TDR positif": "Tests positifs",
            "TDR négatif": "Tests négatifs",
        })
    return df_out


@st.cache_data(show_spinner=False)
def build_weekly_lab_summary(df_: pd.DataFrame) -> pd.DataFrame:
    """Construit un suivi hebdomadaire des tests valides et de la positivite."""
    week_col = resolve_week_column(df_)
    result_col = None
    if COL_TDRR in df_.columns and df_[COL_TDRR].notna().any():
        result_col = COL_TDRR
    elif "Resultat_labo" in df_.columns and df_["Resultat_labo"].notna().any():
        result_col = "Resultat_labo"
    if week_col is None or result_col is None:
        return pd.DataFrame(columns=["Semaine", "Tests valides", "Tests positifs", "Positivité (%)"])

    tmp = df_.copy()
    if week_col == "YW":
        tmp = tmp[tmp["YW"].notna()].copy()
        tmp["Semaine"] = tmp["YW"].astype(str)
        tmp["_order"] = tmp["Semaine"]
    elif week_col == COL_WNUM:
        tmp["_week_num"] = pd.to_numeric(tmp[COL_WNUM], errors="coerce")
        tmp = tmp[tmp["_week_num"].notna()].copy()
        tmp["_order"] = tmp["_week_num"].astype(int)
        tmp["Semaine"] = tmp["_order"].apply(lambda x: f"SE{x:02d}")
    else:
        tmp = tmp[tmp[COL_WEEK].notna()].copy()
        tmp["Semaine"] = tmp[COL_WEEK].astype(str)
        tmp["_order"] = tmp["Semaine"]

    if tmp.empty:
        return pd.DataFrame(columns=["Semaine", "Tests valides", "Tests positifs", "Positivité (%)"])

    res_n = _tdr_result_norm(tmp[result_col])
    tmp["test_valide"] = res_n.isin(TDR_POS_SET.union(TDR_NEG_SET)).astype(int)
    tmp["test_positif"] = res_n.isin(TDR_POS_SET).astype(int)

    summary = (
        tmp.groupby(["_order", "Semaine"], as_index=False)
        .agg(
            **{
                "Tests valides": ("test_valide", "sum"),
                "Tests positifs": ("test_positif", "sum"),
            }
        )
        .sort_values("_order")
    )
    summary["Positivité (%)"] = np.where(
        summary["Tests valides"] > 0,
        summary["Tests positifs"] / summary["Tests valides"] * 100.0,
        np.nan,
    )
    return summary.drop(columns=["_order"])


def build_delay_summary_table(
    df_: pd.DataFrame,
    delay_cols: list[str],
    seuil_jours: int | float | None = None,
) -> pd.DataFrame:
    seuil = get_session_int("seuil_jours", 2) if seuil_jours is None else float(seuil_jours)
    rows = []
    for col in delay_cols:
        s = pd.to_numeric(df_.get(col), errors='coerce')
        s = s[s >= 0]
        if len(s) == 0:
            continue
        rows.append({
            'Délai': col,
            'n': int(s.notna().sum()),
            'Moyenne': round(float(s.mean()), 1),
            'Médiane': round(float(s.median()), 1),
            'Min': round(float(s.min()), 1),
            'Max': round(float(s.max()), 1),
            f'% ≤ {seuil:g} jours': round(float((s <= seuil).mean() * 100), 1),
        })
    return pd.DataFrame(rows)


def _safe_top_label(df_: pd.DataFrame, col: str) -> str:
    if col not in df_.columns:
        return "non disponible"
    s = df_[col].fillna("Inconnu").astype(str).str.strip().replace("", "Inconnu")
    if s.empty:
        return "non disponible"
    vc = s.value_counts(dropna=False)
    return str(vc.index[0]) if len(vc) else "non disponible"


def _build_week_coverage_text(df_: pd.DataFrame) -> str:
    week_pairs = pd.DataFrame(columns=["year", "week"])

    if "YW" in df_.columns and df_["YW"].notna().any():
        extracted = (
            df_["YW"]
            .astype("string")
            .dropna()
            .str.extract(r"(?P<year>\d{4}).*?W(?P<week>\d{1,2})")
        )
        extracted["year"] = pd.to_numeric(extracted["year"], errors="coerce")
        extracted["week"] = pd.to_numeric(extracted["week"], errors="coerce")
        week_pairs = extracted.dropna(subset=["year", "week"]).copy()
    elif COL_YEAR in df_.columns and COL_WNUM in df_.columns:
        week_pairs = pd.DataFrame(
            {
                "year": pd.to_numeric(df_[COL_YEAR], errors="coerce"),
                "week": pd.to_numeric(df_[COL_WNUM], errors="coerce"),
            }
        ).dropna(subset=["year", "week"]).copy()
    elif COL_WNUM in df_.columns and pd.to_numeric(df_[COL_WNUM], errors="coerce").notna().any():
        week_values = pd.to_numeric(df_[COL_WNUM], errors="coerce").dropna().astype(int)
        return f"SE{int(week_values.min()):02d} à SE{int(week_values.max()):02d}"

    if week_pairs.empty:
        return ""

    week_pairs["year"] = week_pairs["year"].astype(int)
    week_pairs["week"] = week_pairs["week"].astype(int)
    week_pairs = week_pairs[(week_pairs["week"] >= 1) & (week_pairs["week"] <= 53)].drop_duplicates()
    if week_pairs.empty:
        return ""

    first = week_pairs.sort_values(["year", "week"]).iloc[0]
    last = week_pairs.sort_values(["year", "week"]).iloc[-1]
    return f"SE{int(first['week']):02d}-{int(first['year'])} à SE{int(last['week']):02d}-{int(last['year'])}"


def _build_narrative_period_text(df_: pd.DataFrame) -> tuple[str, str]:
    week_start, week_end = _extract_iso_week_bounds(df_)
    week_txt = _build_week_coverage_text(df_)
    if week_start is not None and week_end is not None:
        period_txt = f"sur la période allant du {week_start:%d/%m/%Y} au {week_end:%d/%m/%Y}"
        return period_txt, week_txt

    for col in [DATE_NOTIF, DATE_ONSET]:
        if col in df_.columns:
            s = pd.to_datetime(df_[col], errors="coerce")
            s = s[s.notna()]
            if s.empty:
                continue
            period_txt = f"sur la période documentée du {s.min():%d/%m/%Y} au {s.max():%d/%m/%Y}"
            return period_txt, week_txt

    return "sur une période non documentée", week_txt


@st.cache_data(show_spinner=False)
def build_who_narrative_summary(df_: pd.DataFrame) -> str:
    """
    Résumé automatisé rédigé dans un langage de surveillance compatible
    avec l'usage OMS/IDSR : période, charge de morbidité, létalité observée,
    profil des cas, distribution géographique et informations de laboratoire.
    """
    n_cases = int(len(df_))
    n_deaths = int(df_["is_death"].sum()) if "is_death" in df_.columns else 0
    cfr = safe_pct(n_deaths, n_cases)

    period_txt, week_txt = _build_narrative_period_text(df_)

    sex_top = _safe_top_label(df_, COL_SEX)
    age_col = None
    if COL_AGEG2 in df_.columns and df_[COL_AGEG2].notna().any():
        age_col = COL_AGEG2
    elif COL_AGEG in df_.columns and df_[COL_AGEG].notna().any():
        age_col = COL_AGEG
    age_top = _safe_top_label(df_, age_col) if age_col else "non documenté"
    prov_top = _safe_top_label(df_, COL_PROV)
    zs_top = _safe_top_label(df_, COL_ZS)

    lab_tbl = build_simple_lab_table(df_)
    lab_txt = "Aucun résultat de laboratoire interprétable n’est disponible dans le périmètre analysé."
    positive_label = "TDR positif" if "TDR positif" in lab_tbl["Indicateur labo"].values else ("Tests positifs" if "Tests positifs" in lab_tbl["Indicateur labo"].values else None)
    if not lab_tbl.empty and positive_label is not None:
        row = lab_tbl.loc[lab_tbl["Indicateur labo"] == positive_label].iloc[0]
        pct_val = row["%"] if pd.notna(row["%"]) else "-"
        lab_txt = (
            f"Au plan du laboratoire, {int(row['n'])} résultat(s) positif(s) ont été documentés, "
            f"pour une positivité observée de {pct_val} % parmi les résultats interprétables."
        )

    cfr_txt = "non calculable" if pd.isna(cfr) else format_metric_value(cfr, decimals=2)
    geo_txt = (
        f"La province la plus représentée est « {prov_top} »"
        if prov_top != "non disponible"
        else "La distribution provinciale n’est pas suffisamment documentée"
    )
    if zs_top != "non disponible":
        geo_txt += f", avec une concentration notable des notifications dans la zone de santé « {zs_top} »"

    morbi_txt = (
        f"Au total, {format_metric_value(n_cases)} cas, dont {format_metric_value(n_deaths)} décès, "
        f"ont été enregistrés {period_txt}"
    )
    if week_txt:
        morbi_txt += f". La couverture hebdomadaire observée s’étend de {week_txt}."
    else:
        morbi_txt += "."

    return (
        f"{morbi_txt} "
        f"La létalité observée (CFR) est estimée à {cfr_txt} %. "
        f"Le profil dominant des cas met en évidence le sexe « {sex_top} » et le groupe d’âge « {age_top} ». "
        f"{geo_txt}. "
        f"{lab_txt} "
        f"Cette synthèse descriptive doit être interprétée à la lumière de la complétude, de la promptitude et de la qualité globale des données disponibles."
    )


def format_metric_value(value: Any, decimals: int = 0, suffix: str = "") -> str:
    """Formate proprement les valeurs numeriques affichees dans les KPI."""
    if value is None or pd.isna(value):
        return "-"
    try:
        if decimals <= 0:
            return f"{int(round(float(value))):,}".replace(",", " ") + suffix
        return f"{float(value):,.{int(decimals)}f}".replace(",", " ") + suffix
    except Exception:
        return f"{value}{suffix}"


def format_pct_delta(current_value: Any, previous_value: Any) -> str:
    """Construit un delta lisible pour les cartes KPI."""
    delta = pct_change_safe(current_value, previous_value)
    if pd.isna(delta):
        return "Référence indisponible"
    return f"{delta:+.1f}% vs semaine précédente"


def resolve_week_column(df_: pd.DataFrame) -> Optional[str]:
    """Identifie la meilleure colonne semaine disponible pour les vues synthèse."""
    if "YW" in df_.columns and df_["YW"].notna().any():
        return "YW"
    if COL_WNUM in df_.columns and pd.to_numeric(df_[COL_WNUM], errors="coerce").notna().any():
        return COL_WNUM
    if COL_WEEK in df_.columns and df_[COL_WEEK].notna().any():
        return COL_WEEK
    return None


def _extract_iso_week_bounds(df_: pd.DataFrame) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """DÃ©duit la borne min/max Ã  partir des semaines ISO disponibles dans les donnÃ©es filtrÃ©es."""
    week_pairs = pd.DataFrame(columns=["year", "week"])

    if "YW" in df_.columns and df_["YW"].notna().any():
        extracted = (
            df_["YW"]
            .astype("string")
            .dropna()
            .str.extract(r"(?P<year>\d{4}).*?W(?P<week>\d{1,2})")
        )
        extracted["year"] = pd.to_numeric(extracted["year"], errors="coerce")
        extracted["week"] = pd.to_numeric(extracted["week"], errors="coerce")
        week_pairs = extracted.dropna(subset=["year", "week"]).copy()
    elif COL_YEAR in df_.columns and COL_WNUM in df_.columns:
        week_pairs = pd.DataFrame(
            {
                "year": pd.to_numeric(df_[COL_YEAR], errors="coerce"),
                "week": pd.to_numeric(df_[COL_WNUM], errors="coerce"),
            }
        ).dropna(subset=["year", "week"]).copy()

    if week_pairs.empty:
        return None, None

    week_pairs["year"] = week_pairs["year"].astype(int)
    week_pairs["week"] = week_pairs["week"].astype(int)
    week_pairs = week_pairs[(week_pairs["week"] >= 1) & (week_pairs["week"] <= 53)].drop_duplicates()
    if week_pairs.empty:
        return None, None

    bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for row in week_pairs.itertuples(index=False):
        try:
            start = pd.Timestamp.fromisocalendar(int(row.year), int(row.week), 1)
            end = pd.Timestamp.fromisocalendar(int(row.year), int(row.week), 7)
            bounds.append((start, end))
        except ValueError:
            continue

    if not bounds:
        return None, None

    return min(v[0] for v in bounds), max(v[1] for v in bounds)


def compute_analysis_period_value(df_: pd.DataFrame) -> str:
    """Construit une pÃ©riode cohÃ©rente avec la fenÃªtre analytique affichÃ©e."""
    week_start, week_end = _extract_iso_week_bounds(df_)
    if week_start is not None and week_end is not None:
        return f"{week_start:%d/%m/%Y} -> {week_end:%d/%m/%Y}"

    date_candidates = []
    for col in [DATE_ONSET, DATE_NOTIF]:
        if col in df_.columns:
            s = pd.to_datetime(df_[col], errors="coerce")
            if s.notna().any():
                date_candidates.append((col, s.min(), s.max(), int(s.notna().sum())))

    if date_candidates:
        _, dmin, dmax, _ = max(date_candidates, key=lambda item: item[3])
        return f"{dmin:%d/%m/%Y} -> {dmax:%d/%m/%Y}"

    return "-"


def format_range_label_for_display(value: Any) -> str:
    """Uniformise les libellés compacts de période et de fenêtre à l'affichage."""
    if value is None or pd.isna(value):
        return "-"

    text = str(value).strip()
    if not text:
        return "-"

    text = re.sub(r"\s*->\s*", " → ", text)
    endpoint_pattern = r"(?:\d{2}/\d{2}/\d{4}|SE\d{2}(?:-\d{4})?|W\d{2}|(?:\d{4}-W\d{2}))"
    compact_match = re.match(fr"^({endpoint_pattern})\s+à\s+({endpoint_pattern})$", text)
    if compact_match:
        return f"{compact_match.group(1)} → {compact_match.group(2)}"

    return text


def build_weekly_overview_table(df_: pd.DataFrame) -> pd.DataFrame:
    """Construit la série hebdomadaire standard utilisée dans la page d'accueil."""
    week_col = resolve_week_column(df_)
    if week_col is None or df_.empty:
        return pd.DataFrame(columns=["order_key", "label", "Cas", "Décès", "Létalité (%)"])

    weekly = df_.copy()
    weekly["_death_flag"] = (
        pd.to_numeric(weekly["is_death"], errors="coerce").fillna(0).astype(int)
        if "is_death" in weekly.columns
        else 0
    )
    if week_col == "YW":
        weekly = weekly[weekly["YW"].notna()].copy()
        weekly["order_key"] = weekly["YW"].astype(str)
        weekly["label"] = weekly["YW"].astype(str)
    elif week_col == COL_WNUM:
        weekly["_week_num"] = pd.to_numeric(weekly[COL_WNUM], errors="coerce")
        weekly = weekly[weekly["_week_num"].notna()].copy()
        weekly["order_key"] = weekly["_week_num"].astype(int)
        weekly["label"] = weekly["order_key"].apply(lambda x: f"SE{x:02d}")
    else:
        weekly = weekly[weekly[COL_WEEK].notna()].copy()
        weekly["order_key"] = weekly[COL_WEEK].astype(str)
        weekly["label"] = weekly[COL_WEEK].astype(str)

    if weekly.empty:
        return pd.DataFrame(columns=["order_key", "label", "Cas", "Décès", "Létalité (%)"])

    grouped = (
        weekly.groupby(["order_key", "label"], as_index=False)
        .agg(Cas=("label", "size"), Décès=("_death_flag", "sum"))
        .sort_values("order_key")
    )
    grouped["Létalité (%)"] = np.where(grouped["Cas"] > 0, grouped["Décès"] / grouped["Cas"] * 100.0, np.nan)
    return grouped


@st.cache_data(show_spinner=False)
def build_dashboard_kpi_payload(df_: pd.DataFrame) -> Dict[str, Any]:
    """Calcule les KPI principaux de la page d'accueil."""
    kpi = compute_indicators(df_)
    weekly = build_weekly_overview_table(df_)
    weekly = _normalize_metric_alias_columns(weekly)
    classification_counts = _build_classification_counts(df_)
    issue_counts = _build_issue_counts(df_)
    lab_counts = _build_lab_counts(df_, kpi)
    quality_focus = _build_quality_focus_metrics(df_)
    delay_focus = _build_delay_focus_metrics(df_)
    hotspots = _build_hotspots_table(df_)
    priority_actions = _build_priority_actions(df_)

    surveillance_chain = [
        {
            "label": "Notifications",
            "value": int(kpi["n_cases"]),
            "subtitle": "Cas filtrés",
            "theme": "blue",
            "color": "#2b74ca",
        },
        {
            "label": "Suspects",
            "value": int(classification_counts.get("Suspect", 0)),
            "subtitle": "Classification standardisée",
            "theme": "orange",
            "color": "#f29b38",
        },
        {
            "label": "Probables",
            "value": int(classification_counts.get("Probable", 0)),
            "subtitle": "Classification standardisée",
            "theme": "amber",
            "color": "#f2a53a",
        },
        {
            "label": "Confirmés",
            "value": int(classification_counts.get("Confirmé", 0)),
            "subtitle": "Cas confirmés",
            "theme": "green",
            "color": "#27a063",
        },
        {
            "label": "Prélevés",
            "value": int(lab_counts.get("preleves", 0)),
            "subtitle": "Prélèvement documenté",
            "theme": "purple",
            "color": "#7b4dff",
        },
        {
            "label": "Positifs labo",
            "value": int(lab_counts.get("positifs", 0)),
            "subtitle": "Résultats positifs",
            "theme": "red",
            "color": "#e84b4b",
        },
        {
            "label": "Décès",
            "value": int(issue_counts.get("Décédé", 0)),
            "subtitle": "Issue documentée",
            "theme": "slate",
            "color": "#5d6d86",
        },
        {
            "label": "Guéris",
            "value": int(issue_counts.get("Guéri", 0)),
            "subtitle": "Issue documentée",
            "theme": "green",
            "color": "#1f8d57",
        },
    ]

    provinces_epid = get_provinces_epid()
    total_provinces = len(EPIDEMIE)
    total_provinces_epid = len(provinces_epid)
    reported_provinces = int(df_[COL_PROV].dropna().nunique()) if COL_PROV in df_.columns else 0
    reported_epid_provinces = (
        int(df_.loc[df_[COL_PROV].isin(provinces_epid), COL_PROV].dropna().nunique())
        if COL_PROV in df_.columns
        else 0
    )
    reported_zs = int(df_[COL_ZS].dropna().nunique()) if COL_ZS in df_.columns else 0

    week_min = "-"
    week_max = "-"
    if not weekly.empty:
        week_min = str(weekly["label"].iloc[0])
        week_max = str(weekly["label"].iloc[-1])
    elif COL_WNUM in df_.columns and pd.to_numeric(df_[COL_WNUM], errors="coerce").notna().any():
        week_values = pd.to_numeric(df_[COL_WNUM], errors="coerce").dropna().astype(int)
        week_min = f"SE{int(week_values.min()):02d}"
        week_max = f"SE{int(week_values.max()):02d}"

    latest = weekly.iloc[-1].to_dict() if not weekly.empty else {}
    previous = weekly.iloc[-2].to_dict() if len(weekly) > 1 else {}
    promptitude_pct, promptitude_n = pct_under_threshold(df_.get("delai_onset_to_adm"), get_session_int("seuil_jours", 2))
    analysis_period = compute_analysis_period_value(df_)

    return {
        "cases": int(kpi["n_cases"]),
        "deaths": int(kpi["n_deaths"]),
        "cfr": float(kpi["cfr_pct"]) if not pd.isna(kpi["cfr_pct"]) else np.nan,
        "week_span": f"{week_min} -> {week_max}" if week_min != "-" and week_max != "-" else "-",
        "week_min": week_min,
        "week_max": week_max,
        "reported_epid_provinces": reported_epid_provinces,
        "total_provinces_epid": total_provinces_epid,
        "reported_provinces": reported_provinces,
        "total_provinces": total_provinces,
        "reported_zs": reported_zs,
        "coverage_epid_pct": safe_pct(reported_epid_provinces, total_provinces_epid),
        "coverage_nat_pct": safe_pct(reported_provinces, total_provinces),
        "weekly": weekly,
        "latest": latest,
        "previous": previous,
        "promptitude_pct": promptitude_pct,
        "promptitude_n": promptitude_n,
        "analysis_period": analysis_period,
        "top_province": _safe_top_label(df_, COL_PROV),
        "top_zs": _safe_top_label(df_, COL_ZS),
        "classification_counts": classification_counts,
        "issue_counts": issue_counts,
        "lab_counts": lab_counts,
        "quality_focus": quality_focus,
        "delay_focus": delay_focus,
        "hotspots": hotspots,
        "priority_actions": priority_actions,
        "surveillance_chain": surveillance_chain,
    }


def render_context_row(files_used: list[str], disease_key: str, df_: pd.DataFrame, payload: Dict[str, Any]) -> None:
    """Affiche quelques repères analytiques juste sous le bandeau principal."""
    disease_label = DISEASE_SPECS.get(disease_key, {}).get("label", str(disease_key))
    source_value = "Aucun fichier" if not files_used else str(files_used[0]).replace("upload:", "")
    if len(source_value) > 42:
        source_value = source_value[:39] + "..."

    period_value = payload.get("analysis_period", "-")
    if period_value == "-":
        period_value = compute_analysis_period_value(df_)
    if period_value == "-":
        period_value = payload.get("week_span", "-")
    period_value = format_range_label_for_display(period_value)

    chips = [
        ("Source", source_value),
        ("Périmètre", disease_label),
        ("Période", period_value),
        ("Couverture", f"{payload.get('reported_provinces', 0)} provinces | {payload.get('reported_zs', 0)} ZS"),
    ]
    html_blocks = []
    for label, value in chips:
        html_blocks.append(
            f"""
<div class="cousp-context-chip">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
</div>
"""
        )
    st.markdown(f"<div class='cousp-context-row'>{''.join(html_blocks)}</div>", unsafe_allow_html=True)


def build_dashboard_kpi_card_html(title: str, value: str, subtitle: str, theme: str, span: int = 1) -> str:
    """Construit une carte KPI HTML pour la bande de synthèse principale."""
    span_class = " span-2" if int(span) >= 2 else ""
    title_html = html.escape(str(title))
    value_html = html.escape(str(value))
    subtitle_html = html.escape(str(subtitle))
    return f"""
<div class="cousp-kpi-card {theme}{span_class}">
  <div class="cousp-kpi-title">{title_html}</div>
  <div class="cousp-kpi-value">{value_html}</div>
  <div class="cousp-kpi-subtitle">{subtitle_html}</div>
</div>
"""


def _get_surveillance_value(payload: Dict[str, Any], label_key: str, default: Optional[int] = None) -> Optional[int]:
    """Retourne la valeur d'un indicateur de chaîne de surveillance si présent."""
    label_norm = _norm_key(str(label_key))
    for item in payload.get("surveillance_chain", []):
        item_label = _norm_key(str(item.get("label", "")))
        if label_norm == item_label:
            raw_value = item.get("value")
            if raw_value is None:
                return default
            try:
                return int(raw_value)
            except Exception:
                return default
    return default


def _estimate_alive_issue_count(payload: Dict[str, Any]) -> int:
    """Estime les vivants à partir des issues documentées si disponibles."""
    issue_counts = payload.get("issue_counts", {}) or {}
    if not issue_counts:
        return max(int(payload.get("cases", 0) or 0) - int(payload.get("deaths", 0) or 0), 0)

    alive_tokens = ("guer", "vivan", "trait", "sort", "en cours", "alive", "surviv")
    unknown_tokens = ("non document", "inconnu", "unknown", "missing")
    death_tokens = ("deced", "deces", "decede", "decede", "mort", "death", "dead")

    alive_total = 0
    documented_non_death = 0
    for key, value in issue_counts.items():
        key_norm = _norm_key(str(key))
        value_int = int(value or 0)
        if any(token in key_norm for token in death_tokens):
            continue
        if any(token in key_norm for token in unknown_tokens):
            continue
        documented_non_death += value_int
        if any(token in key_norm for token in alive_tokens):
            alive_total += value_int

    if alive_total > 0:
        return alive_total
    if documented_non_death > 0:
        return documented_non_death
    return max(int(payload.get("cases", 0) or 0) - int(payload.get("deaths", 0) or 0), 0)


def render_dashboard_kpis(payload: Dict[str, Any]) -> None:
    """Affiche la ligne horizontale des KPI principaux."""
    payload = {**payload, "week_span": format_range_label_for_display(payload.get("week_span", "-"))}
    cards = [
        ("Total cas", format_metric_value(payload.get("cases", 0)), "Périmètre filtré", "blue compact", 1),
        ("Total décès", format_metric_value(payload.get("deaths", 0)), "Périmètre filtré", "navy compact", 1),
        ("CFR (%)", format_metric_value(payload.get("cfr"), decimals=2), "Létalité observée", "orange compact", 1),
        ("Période", payload.get("week_span", "-"), "Fenêtre analytique", "blue compact", 2),
        (
            "Couverture nationale",
            f"{payload.get('reported_provinces', 0)} / {payload.get('total_provinces', 0)}",
            "-" if pd.isna(payload.get("coverage_nat_pct")) else f"{payload.get('coverage_nat_pct', 0):.1f}% de couverture",
            "green compact",
            1,
        ),
        ("ZS touchées", format_metric_value(payload.get("reported_zs", 0)), "Notifications consolidées", "green compact", 1),
        ("Probables", format_metric_value(_get_surveillance_value(payload, "Cas probables", 0)), "Classification disponible", "orange compact", 1),
        ("Suspects", format_metric_value(_get_surveillance_value(payload, "Cas suspects", 0)), "Classification disponible", "amber compact", 1),
        ("Positifs", format_metric_value(_get_surveillance_value(payload, "Cas positifs", 0)), "Résultats labo", "red compact", 1),
        ("Négatifs", format_metric_value(_get_surveillance_value(payload, "Cas negatifs", 0)), "Résultats labo", "green compact", 1),
        ("Invalides", format_metric_value(_get_surveillance_value(payload, "Resultats invalides", 0)), "Analyses non concluantes", "slate compact", 1),
        ("Guéris", format_metric_value(_estimate_alive_issue_count(payload)), "Vivants documentés", "green compact", 1),
    ]
    cards_html = "".join(build_dashboard_kpi_card_html(*card) for card in cards)
    st.markdown(f"<div class='cousp-kpi-grid cousp-kpi-grid-compact'>{cards_html}</div>", unsafe_allow_html=True)


def _clean_count_series(series: pd.Series, fallback_label: str = "Non documenté") -> pd.Series:
    """Normalise une série textuelle pour des comptages robustes."""
    return (
        series.astype("string")
        .fillna(fallback_label)
        .str.strip()
        .replace({"": fallback_label})
    )


def _extract_lab_result_series(df_: pd.DataFrame) -> pd.Series:
    """Retourne la meilleure série de résultats labo normalisés disponible."""
    if COL_TDRR in df_.columns and df_[COL_TDRR].notna().any():
        return _tdr_result_norm(df_[COL_TDRR])
    if "Resultat_labo" in df_.columns and df_["Resultat_labo"].notna().any():
        return _tdr_result_norm(df_["Resultat_labo"])
    return pd.Series(pd.NA, index=df_.index, dtype="string")


def _build_classification_counts(df_: pd.DataFrame) -> dict[str, int]:
    """Compte les grandes classes épidémiologiques standardisées."""
    if "Classification_finale_std" in df_.columns and df_["Classification_finale_std"].notna().any():
        series = _clean_count_series(df_["Classification_finale_std"])
    elif COL_CLASS in df_.columns:
        series = _clean_count_series(df_[COL_CLASS])
    else:
        return {}
    counts = series.value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _build_issue_counts(df_: pd.DataFrame) -> dict[str, int]:
    """Compte les issues standardisées disponibles."""
    if "Issue_std" in df_.columns and df_["Issue_std"].notna().any():
        series = _clean_count_series(df_["Issue_std"])
    elif COL_ISSUE in df_.columns:
        series = _clean_count_series(df_[COL_ISSUE])
    else:
        return {}
    counts = series.value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _build_lab_counts(df_: pd.DataFrame, kpi: Dict[str, Any]) -> dict[str, int]:
    """Construit les principaux volumes de la chaîne laboratoire."""
    result_series = _extract_lab_result_series(df_)
    result_norm = _clean_count_series(result_series, fallback_label="Non documenté")
    valid_mask = result_norm.isin(TDR_POS_SET.union(TDR_NEG_SET))
    invalid_mask = result_norm.isin({"indetermine", "invalide", "invalid", "inba", "bande absente"})
    waiting_mask = result_norm.isin({"en attente", "non teste"})
    positive_mask = result_norm.isin(TDR_POS_SET)
    negative_mask = result_norm.isin(TDR_NEG_SET)
    received_count = int(pd.to_datetime(df_[DATE_RECEP], errors="coerce").notna().sum()) if DATE_RECEP in df_.columns else 0

    return {
        "preleves": int(kpi.get("prelev_num", 0) or 0),
        "recus": received_count,
        "tests_documentes": int(kpi.get("tdr_num", 0) or 0),
        "resultats_valides": int(valid_mask.sum()),
        "analyses": int((valid_mask | invalid_mask).sum()),
        "positifs": int(positive_mask.sum()),
        "negatifs": int(negative_mask.sum()),
        "invalides": int(invalid_mask.sum()),
        "en_attente": int(waiting_mask.sum()),
    }


def _build_alert_proxy_counts(df_: pd.DataFrame, classification_counts: dict[str, int]) -> dict[str, Any]:
    """Construit des proxies simples de chaîne d'alerte pour les line lists."""
    notified_alerts = int(len(df_))
    if "Classification_finale_std" in df_.columns and df_["Classification_finale_std"].notna().any():
        classification_series = _clean_count_series(df_["Classification_finale_std"])
    elif COL_CLASS in df_.columns and df_[COL_CLASS].notna().any():
        classification_series = _clean_count_series(df_[COL_CLASS])
    else:
        classification_series = None

    verified_alerts = None
    if classification_series is not None:
        verified_alerts = int((~classification_series.isin(["Non cas", "Non documenté"])).sum())

    return {
        "notified_alerts": notified_alerts,
        "verified_alerts": verified_alerts,
        "has_classification": classification_series is not None,
    }


def _build_quality_focus_metrics(df_: pd.DataFrame) -> list[dict[str, Any]]:
    """Prépare quelques indicateurs simples de qualité pour l'accueil."""
    n_total = max(int(len(df_)), 1)

    age_years = infer_age_years_generic(df_)
    age_missing_pct = float(age_years.isna().mean() * 100.0) if len(age_years) else 0.0

    if COL_SEX in df_.columns:
        sex_missing = df_[COL_SEX].isna() | df_[COL_SEX].astype("string").str.strip().eq("")
        sex_missing_pct = float(sex_missing.mean() * 100.0)
    else:
        sex_missing_pct = 100.0

    if COL_ZS in df_.columns:
        geo_missing = df_[COL_ZS].isna() | df_[COL_ZS].astype("string").str.strip().eq("")
        geo_missing_pct = float(geo_missing.mean() * 100.0)
    else:
        geo_missing_pct = 100.0

    suspect_mask = (
        df_["Classification_finale_std"].astype("string").eq("Suspect")
        if "Classification_finale_std" in df_.columns
        else pd.Series(False, index=df_.index)
    )
    if COL_PREL in df_.columns and suspect_mask.any():
        prelev_yes = _is_yes_series(df_[COL_PREL])
        suspects_without_sample_pct = float(((suspect_mask & ~prelev_yes).sum() / max(int(suspect_mask.sum()), 1)) * 100.0)
    else:
        suspects_without_sample_pct = 0.0

    result_norm = _extract_lab_result_series(df_)
    prelev_yes_global = _is_yes_series(df_[COL_PREL]) if COL_PREL in df_.columns else pd.Series(False, index=df_.index)
    if len(result_norm):
        result_available = result_norm.isin(TDR_POS_SET.union(TDR_NEG_SET).union({"indetermine"}))
        prelev_without_result_pct = float(((prelev_yes_global & ~result_available).sum() / max(int(prelev_yes_global.sum()), 1)) * 100.0) if prelev_yes_global.any() else 0.0
    else:
        prelev_without_result_pct = 0.0

    duplicate_pct = float(df_["duplicate_potential"].fillna(False).mean() * 100.0) if "duplicate_potential" in df_.columns else 0.0
    chrono_pct = float(df_["chronologie_invalide"].fillna(False).mean() * 100.0) if "chronologie_invalide" in df_.columns else 0.0

    metrics = [
        {"label": "Cas sans âge", "value": round(age_missing_pct, 1), "theme": "blue"},
        {"label": "Cas sans sexe", "value": round(sex_missing_pct, 1), "theme": "blue"},
        {"label": "Cas sans zone de santé", "value": round(geo_missing_pct, 1), "theme": "blue"},
        {"label": "Suspects sans prélèvement", "value": round(suspects_without_sample_pct, 1), "theme": "orange"},
        {"label": "Prélèvements sans résultat", "value": round(prelev_without_result_pct, 1), "theme": "orange"},
        {"label": "Doublons potentiels", "value": round(duplicate_pct, 1), "theme": "red"},
        {"label": "Chronologie invalide", "value": round(chrono_pct, 1), "theme": "red"},
    ]
    return metrics


def _build_delay_focus_metrics(df_: pd.DataFrame) -> list[dict[str, Any]]:
    """Sélectionne quelques délais clés pour le résumé opérationnel."""
    labels = {
        "Début → notification": "Notification",
        "Notification → investigation": "Investigation",
        "Début → prélèvement": "Prélèvement",
        "Prélèvement → résultat": "Résultat",
    }
    delay_summary = build_standard_delay_summary(df_)
    if delay_summary.empty:
        return []

    selected = delay_summary[delay_summary["Type_delai"].isin(labels.keys())].copy()
    if selected.empty:
        selected = delay_summary.head(4).copy()

    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        rows.append(
            {
                "label": labels.get(str(row["Type_delai"]), str(row["Type_delai"])),
                "full_label": str(row["Type_delai"]),
                "median_days": float(row["Médiane_j"]),
                "n": int(row["n"]),
            }
        )
    return rows


def _build_hotspot_level(cases: int, deaths: int, confirmed: int, max_cases: int) -> tuple[str, str]:
    """Déduit un niveau simple pour la table des zones actives."""
    if confirmed > 0 or deaths > 0:
        return "Urgence", "danger"
    if cases >= max(10, int(max_cases * 0.55)):
        return "Sous surveillance", "warning"
    if cases >= max(5, int(max_cases * 0.25)):
        return "Active", "blue"
    return "Stable", "green"


def _build_hotspots_table(df_: pd.DataFrame, top_n: int = 5) -> tuple[pd.DataFrame, dict[str, str]]:
    """Construit un top des aires / zones / provinces les plus actives."""
    if COL_AS in df_.columns and df_[COL_AS].notna().any():
        level_col = COL_AS
        title = "Top 5 aires de santé actives"
        subtitle = "Classement dynamique selon le volume de notifications et de confirmations."
    elif COL_ZS in df_.columns and df_[COL_ZS].notna().any():
        level_col = COL_ZS
        title = "Top 5 zones de santé actives"
        subtitle = "Classement dynamique selon le volume de notifications et de confirmations."
    else:
        level_col = COL_PROV
        title = "Top 5 provinces actives"
        subtitle = "Classement dynamique selon le volume de notifications et de confirmations."

    if level_col not in df_.columns:
        return pd.DataFrame(columns=["Lieu", "Province", "Zone de santé", "Cas", "Décès", "Confirmés", "Niveau", "Niveau_theme"]), {
            "title": "Top territoires actifs",
            "subtitle": "Ventilation géographique non disponible.",
        }

    work = df_.copy()
    work["_cases"] = 1
    work["_deaths"] = work["is_death"].fillna(False).astype(int) if "is_death" in work.columns else 0
    if "Classification_finale_std" in work.columns:
        work["_confirmed"] = (
            work["Classification_finale_std"]
            .astype("string")
            .eq("Confirmé")
            .fillna(False)
            .astype(int)
        )
    else:
        work["_confirmed"] = 0

    aggregations: dict[str, Any] = {
        "Cas": ("_cases", "sum"),
        "Décès": ("_deaths", "sum"),
        "Confirmés": ("_confirmed", "sum"),
    }
    if COL_PROV in work.columns:
        aggregations["Province"] = (COL_PROV, lambda s: next((str(v).strip() for v in s if pd.notna(v) and str(v).strip()), "-"))
    if level_col == COL_AS and COL_ZS in work.columns:
        aggregations["Zone de santé"] = (COL_ZS, lambda s: next((str(v).strip() for v in s if pd.notna(v) and str(v).strip()), "-"))

    table = (
        work.groupby(level_col, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .rename(columns={level_col: "Lieu"})
    )
    if table.empty:
        return pd.DataFrame(columns=["Lieu", "Province", "Zone de santé", "Cas", "Décès", "Confirmés", "Niveau", "Niveau_theme"]), {
            "title": title,
            "subtitle": subtitle,
        }

    table["Lieu"] = table["Lieu"].fillna("Non documenté").astype(str).str.strip().replace({"": "Non documenté"})
    if "Province" not in table.columns:
        table["Province"] = "-"
    if "Zone de santé" not in table.columns:
        table["Zone de santé"] = "-"

    table = table.sort_values(["Cas", "Confirmés", "Décès"], ascending=[False, False, False]).head(max(int(top_n), 1)).reset_index(drop=True)
    max_cases = int(table["Cas"].max()) if not table.empty else 0
    levels = table.apply(lambda row: _build_hotspot_level(int(row["Cas"]), int(row["Décès"]), int(row["Confirmés"]), max_cases), axis=1)
    table["Niveau"] = [level[0] for level in levels]
    table["Niveau_theme"] = [level[1] for level in levels]
    return table[["Lieu", "Province", "Zone de santé", "Cas", "Décès", "Confirmés", "Niveau", "Niveau_theme"]], {
        "title": title,
        "subtitle": subtitle,
    }


def _build_priority_actions(df_: pd.DataFrame) -> list[dict[str, str]]:
    """Construit une courte liste d'actions prioritaires à afficher à l'accueil."""
    try:
        standard_signals = build_standard_signal_table(
            df_,
            week_col=resolve_week_column(df_) or "YW",
            timeliness_threshold_days=float(get_session_int("seuil_jours", 2)),
        )
    except Exception:
        standard_signals = pd.DataFrame()

    if standard_signals.empty:
        return []

    active = standard_signals[standard_signals["À surveiller"].astype(str).str.strip().eq("Oui")].copy()
    if active.empty:
        return []

    actions: list[dict[str, str]] = []
    for _, row in active.head(4).iterrows():
        status = str(row.get("Statut", "À suivre")).strip() or "À suivre"
        priority = "Haute" if status == "Alerte" else "Moyenne" if status == "À suivre" else "Routine"
        actions.append(
            {
                "priority": priority,
                "theme": "danger" if priority == "Haute" else "warning" if priority == "Moyenne" else "blue",
                "label": str(row.get("Indicateur", "Point à suivre")).strip() or "Point à suivre",
                "action": str(row.get("Action proposée", "")).strip() or str(row.get("Ce qu'on observe", "")).strip(),
            }
        )
    return actions


def build_surveillance_cascade_figure(payload: Dict[str, Any]) -> Optional[go.Figure]:
    """Construit un graphique horizontal compact pour la cascade de suivi."""
    chain = payload.get("surveillance_chain", [])
    if not chain:
        return None

    labels = [item["label"] for item in chain]
    values = [int(item.get("value", 0)) for item in chain]
    colors = [item.get("color", "#2b74ca") for item in chain]

    fig = go.Figure(
        go.Bar(
            x=values[::-1],
            y=labels[::-1],
            orientation="h",
            marker=dict(color=colors[::-1], line=dict(color="rgba(255,255,255,0.65)", width=1.2)),
            text=[format_metric_value(value) for value in values[::-1]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(t=10, r=28, b=10, l=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(title=None, ticks="", showgrid=False, automargin=True),
    )
    return fig


def render_surveillance_chain_section(payload: Dict[str, Any]) -> None:
    """Affiche la chaîne de surveillance sous forme de cartes horizontales."""
    chain = payload.get("surveillance_chain", [])
    if not chain:
        return

    cards_html = []
    steps_html = []
    for item in chain:
        theme = str(item.get("theme", "blue"))
        label = html.escape(str(item.get("label", "")))
        value = html.escape(format_metric_value(item.get("value", 0)))
        subtitle = html.escape(str(item.get("subtitle", "")))
        cards_html.append(
            f"""
<div class="cousp-chain-card {theme}">
  <div class="cousp-chain-label">{label}</div>
  <div class="cousp-chain-value">{value}</div>
  <div class="cousp-chain-subtitle">{subtitle}</div>
</div>
"""
        )
        steps_html.append(
            f"""
<div class="cousp-chain-step">
  <span class="dot {theme}"></span>
  <span>{label}</span>
</div>
"""
        )

    st.markdown(
        f"""
<div class="cousp-chain-panel">
  <div class="cousp-panel-title">Chaîne de surveillance</div>
  <div class="cousp-chain-grid">{''.join(cards_html)}</div>
  <div class="cousp-chain-stepper">{''.join(steps_html)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_hotspots_panel(payload: Dict[str, Any]) -> None:
    """Affiche le top des zones ou provinces les plus actives."""
    table = payload.get("hotspots")
    st.markdown("<div class='cousp-panel-title'>Top territoires actifs</div>", unsafe_allow_html=True)
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        st.info("Aucune ventilation géographique exploitable n'est disponible pour le top des territoires.")
        return

    rows = []
    for _, row in table.iterrows():
        place = html.escape(str(row.get("Lieu", "-")))
        province = html.escape(str(row.get("Province", "-")))
        cases = format_metric_value(row.get("Cas", 0))
        confirmed = format_metric_value(row.get("Confirmés", 0))
        theme = html.escape(str(row.get("Niveau_theme", "blue")))
        level = html.escape(str(row.get("Niveau", "Active")))
        rows.append(
            f"""
<tr>
  <td><strong>{place}</strong><span>{province}</span></td>
  <td>{cases}</td>
  <td>{confirmed}</td>
  <td><span class="cousp-badge {theme}">{level}</span></td>
</tr>
"""
        )

    st.markdown(
        f"""
<div class="cousp-summary-box">
  <table class="cousp-mini-table">
    <thead>
      <tr>
        <th>Lieu</th>
        <th>Cas</th>
        <th>Confirmés</th>
        <th>Niveau</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )


def render_quality_snapshot_panel(payload: Dict[str, Any]) -> None:
    """Affiche des barres de progression pour quelques indicateurs qualité."""
    metrics = payload.get("quality_focus", [])
    st.markdown("<div class='cousp-panel-title'>Qualité des données</div>", unsafe_allow_html=True)
    if not metrics:
        st.info("Aucun indicateur qualité n'est disponible sur le périmètre filtré.")
        return

    rows = []
    for item in metrics[:6]:
        label = html.escape(str(item.get("label", "")))
        value = float(item.get("value", 0.0))
        theme = html.escape(str(item.get("theme", "blue")))
        rows.append(
            f"""
<div class="cousp-progress-row">
  <div class="cousp-progress-top">
    <span>{label}</span>
    <strong>{value:.1f}%</strong>
  </div>
  <div class="cousp-progress-track">
    <span class="cousp-progress-fill {theme}" style="width: {min(max(value, 0.0), 100.0):.1f}%;"></span>
  </div>
</div>
"""
        )

    st.markdown(
        f"""
<div class="cousp-summary-box">
  <div class="cousp-progress-list">{''.join(rows)}</div>
  <div class="cousp-summary-footnote">Objectif recommandé : maintenir ces indicateurs au niveau le plus bas possible.</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_delay_snapshot_panel(payload: Dict[str, Any]) -> None:
    """Affiche quelques délais opérationnels clés."""
    delays = payload.get("delay_focus", [])
    st.markdown("<div class='cousp-panel-title'>Délais opérationnels</div>", unsafe_allow_html=True)
    if not delays:
        st.info("Aucun délai standard n'est disponible dans les données filtrées.")
        return

    rows = []
    for item in delays:
        label = html.escape(str(item.get("label", "")))
        full_label = html.escape(str(item.get("full_label", "")))
        median_days = float(item.get("median_days", 0.0))
        n_obs = int(item.get("n", 0))
        rows.append(
            f"""
<div class="cousp-kv-item">
  <div class="cousp-kv-label">{label}</div>
  <div class="cousp-kv-value">{median_days:.1f} j</div>
  <div class="cousp-kv-sub">{full_label} · n={n_obs}</div>
</div>
"""
        )

    st.markdown(f"<div class='cousp-kv-grid'>{''.join(rows)}</div>", unsafe_allow_html=True)


def render_priority_actions_panel(payload: Dict[str, Any]) -> None:
    """Affiche les principaux points à traiter à court terme."""
    actions = payload.get("priority_actions", [])
    st.markdown("<div class='cousp-panel-title'>Actions prioritaires</div>", unsafe_allow_html=True)
    if not actions:
        st.info("Aucune action prioritaire automatique n'est remontée avec les seuils actifs.")
        return

    rows = []
    for item in actions:
        label = html.escape(str(item.get("label", "")))
        action = html.escape(str(item.get("action", "")))
        priority = html.escape(str(item.get("priority", "Moyenne")))
        theme = html.escape(str(item.get("theme", "blue")))
        rows.append(
            f"""
<div class="cousp-action-row">
  <span class="cousp-badge {theme}">{priority}</span>
  <div class="cousp-action-copy">
    <strong>{label}</strong>
    <span>{action}</span>
  </div>
</div>
"""
        )

    st.markdown(f"<div class='cousp-summary-box'>{''.join(rows)}</div>", unsafe_allow_html=True)


def render_briefing_panel(payload: Dict[str, Any]) -> None:
    """Affiche une lecture opérationnelle rapide de la période active."""
    st.markdown("<div class='cousp-panel-title'>Briefing opérationnel</div>", unsafe_allow_html=True)

    latest = payload.get("latest", {})
    previous = payload.get("previous", {})
    promptitude = format_metric_value(payload.get("promptitude_pct"), decimals=1, suffix="%")
    cards = [
        ("Cas semaine", format_metric_value(latest.get("Cas", np.nan)), format_pct_delta(latest.get("Cas", np.nan), previous.get("Cas", np.nan))),
        ("Décès semaine", format_metric_value(latest.get("Décès", np.nan)), format_pct_delta(latest.get("Décès", np.nan), previous.get("Décès", np.nan))),
        ("CFR semaine", format_metric_value(latest.get("Létalité (%)", np.nan), decimals=2, suffix="%"), format_pct_delta(latest.get("Létalité (%)", np.nan), previous.get("Létalité (%)", np.nan))),
        (f"Admission <= {get_session_int('seuil_jours', 2)} j", promptitude, f"n={int(payload.get('promptitude_n', 0) or 0)}"),
    ]

    metrics_html = []
    for label, value, subtitle in cards:
        metrics_html.append(
            f"""
<div class="cousp-briefing-metric">
  <span>{html.escape(label)}</span>
  <strong>{html.escape(value)}</strong>
  <small>{html.escape(subtitle)}</small>
</div>
"""
        )

    top_province = html.escape(str(payload.get("top_province", "non disponible")))
    top_zs = html.escape(str(payload.get("top_zs", "non disponible")))
    week_span = html.escape(format_range_label_for_display(payload.get("week_span", "-")))
    case_total = html.escape(format_metric_value(payload.get("cases", 0)))

    st.markdown(
        f"""
<div class="cousp-summary-box">
  <div class="cousp-briefing-grid">{''.join(metrics_html)}</div>
  <div class="cousp-summary-footnote">
    Province la plus notifiée : <strong>{top_province}</strong><br/>
    Zone de santé la plus notifiée : <strong>{top_zs}</strong><br/>
    Fenêtre couverte : <strong>{week_span}</strong> avec <strong>{case_total}</strong> cas analysés.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def build_geo_pair_label(province_value: Any, zone_value: Any) -> str:
    """Construit un libellé lisible Province / Zone pour les cartes de ZS."""
    province_txt = "" if province_value is None or pd.isna(province_value) else str(province_value).strip()
    zone_txt = "" if zone_value is None or pd.isna(zone_value) else str(zone_value).strip()
    if province_txt and zone_txt:
        return f"{province_txt} / {zone_txt}"
    return zone_txt or province_txt


def build_geo_pair_key(province_value: Any, zone_value: Any) -> str:
    """Construit une clé robuste Province + Zone pour éviter les collisions de noms."""
    zone_txt = "" if zone_value is None or pd.isna(zone_value) else str(zone_value).strip()
    if not zone_txt:
        return ""

    province_txt = "" if province_value is None or pd.isna(province_value) else str(province_value).strip()
    if province_txt:
        province_txt = _normalize_province_name_for_matching(province_txt)
        return f"{_norm_key(province_txt)}||{_norm_key(zone_txt)}"
    return _norm_key(zone_txt)


def split_geo_pair_label(label_value: Any) -> tuple[Optional[str], Optional[str]]:
    """Extrait Province et Zone depuis un libellé composite de carte."""
    if label_value is None or pd.isna(label_value):
        return None, None

    text = str(label_value).strip()
    if not text:
        return None, None
    if " / " not in text:
        return None, text

    province_txt, zone_txt = [part.strip() for part in text.split(" / ", 1)]
    return (province_txt or None), (zone_txt or None)


@st.cache_data(show_spinner=False)
def _read_geo_file_cached(path_str: str, mtime_ns: int):
    del mtime_ns
    return gpd.read_file(path_str)


def _read_geo_file(path: Union[str, Path]):
    path_obj = Path(path).resolve()
    return _read_geo_file_cached(str(path_obj), path_obj.stat().st_mtime_ns)


def _attach_zone_map_labels(gdf_geo: pd.DataFrame) -> pd.DataFrame:
    """Ajoute province + zone au GeoDataFrame des zones de santé."""
    gdf_zone = gdf_geo.copy()
    if "_map_label" in gdf_zone.columns and "_map_join_key" in gdf_zone.columns:
        return gdf_zone

    zone_names = (
        gdf_zone["name"].astype(str).str.strip()
        if "name" in gdf_zone.columns
        else pd.Series("", index=gdf_zone.index, dtype="string")
    )
    province_names = pd.Series([pd.NA] * len(gdf_zone), index=gdf_zone.index, dtype="object")

    province_geo_path = Path("data/geometry_rdc_provinces.geojson")
    if province_geo_path.exists() and gpd is not None and "geometry" in gdf_zone.columns:
        try:
            gdf_prov = _read_geo_file(province_geo_path)[["name", "geometry"]].copy()
            try:
                if gdf_zone.crs is not None and gdf_prov.crs is not None and str(gdf_zone.crs) != str(gdf_prov.crs):
                    gdf_prov = gdf_prov.to_crs(gdf_zone.crs)
            except Exception:
                pass

            province_shapes = [
                (str(row["name"]).strip(), row.geometry)
                for _, row in gdf_prov.iterrows()
                if row.geometry is not None and not row.geometry.is_empty
            ]
            zone_points = gdf_zone.geometry.representative_point()
            province_values: list[Optional[str]] = []
            for point in zone_points:
                province_match = None
                if point is not None and not point.is_empty:
                    for province_name, province_geom in province_shapes:
                        try:
                            if province_geom.contains(point) or point.within(province_geom) or province_geom.intersects(point):
                                province_match = province_name
                                break
                        except Exception:
                            continue
                province_values.append(province_match)
            province_names = pd.Series(province_values, index=gdf_zone.index, dtype="object")
        except Exception:
            province_names = pd.Series([pd.NA] * len(gdf_zone), index=gdf_zone.index, dtype="object")

    gdf_zone["_map_province"] = province_names
    gdf_zone["_map_label"] = [
        build_geo_pair_label(province_value, zone_value)
        for province_value, zone_value in zip(gdf_zone["_map_province"], zone_names)
    ]
    gdf_zone["_map_join_key"] = [
        build_geo_pair_key(province_value, zone_value)
        for province_value, zone_value in zip(gdf_zone["_map_province"], zone_names)
    ]
    return gdf_zone


@st.cache_data(show_spinner=False)
def _prepare_geo_matching_payload(
    df_source: pd.DataFrame,
    geo_path: str,
    group_col: str,
    value_col: str,
    match_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame, str, str]:
    """Prépare la jointure carte/données avec une clé Province + Zone pour les ZS."""
    gdf_geo = _read_geo_file(geo_path)
    source_label_col = group_col
    geo_label_col = "name"
    geo_key_col = "name"
    data_key_col = group_col
    df_carte = df_source.copy()

    if group_col == COL_ZS and COL_PROV in df_source.columns and df_source[COL_PROV].notna().any():
        gdf_geo = _attach_zone_map_labels(gdf_geo)
        geo_label_col = "_map_label"
        geo_key_col = "_map_join_key"
        source_label_col = "_map_label"

        df_carte = df_source[[c for c in [COL_PROV, COL_ZS, value_col] if c in df_source.columns]].copy()
        df_carte = df_carte[df_carte[COL_ZS].notna()].copy()
        df_carte["_map_label"] = [
            build_geo_pair_label(province_value, zone_value)
            for province_value, zone_value in zip(df_carte.get(COL_PROV), df_carte.get(COL_ZS))
        ]
        df_carte["_map_join_key"] = [
            build_geo_pair_key(province_value, zone_value)
            for province_value, zone_value in zip(df_carte.get(COL_PROV), df_carte.get(COL_ZS))
        ]
        data_key_col = "_map_join_key"

    gdf_join, df_map, match_rate = joindre_donnees_fuzzy_geo(
        carte_gdf=gdf_geo,
        df_donnees=df_carte,
        colonne_cle_geo=geo_key_col,
        colonne_cle_data=data_key_col,
        colonne_valeurs=value_col,
        seuil=match_threshold,
    )
    label_source = df_carte[[source_label_col]].dropna().copy() if source_label_col in df_carte.columns else pd.DataFrame()
    return gdf_join, df_map, match_rate, label_source, source_label_col, geo_label_col


@st.cache_data(show_spinner=False)
def prepare_overview_map_data(
    df_: pd.DataFrame,
    level: str,
    match_threshold: float = 0.90,
):
    """Prépare les données géographiques de synthèse pour les cartes province / zone."""
    if gpd is None:
        return None, None, "geopandas n'est pas disponible.", None, None, None

    if level == "province":
        geo_path = "data/geometry_rdc_provinces.geojson"
        group_col = COL_PROV
        value_col = "nb_cas_prov"
        title = "RDC - cas notifiés par province"
    else:
        geo_path = "data/geometry_rdc_zones_sante.geojson"
        group_col = COL_ZS
        value_col = "nb_cas_zs"
        title = "RDC - cas notifiés par zone de santé"

    if not Path(geo_path).exists():
        return None, None, "GeoJSON non disponible dans le dépôt.", value_col, group_col, title
    if group_col not in df_.columns or df_[group_col].dropna().empty:
        return None, None, "Variable géographique indisponible.", value_col, group_col, title

    try:
        df_counts = df_[[group_col]].dropna().copy()
        if level == "zone" and COL_PROV in df_.columns:
            df_counts[COL_PROV] = df_.loc[df_counts.index, COL_PROV]
        df_counts[value_col] = 1
        gdf_join, df_match, match_rate, _, _, _ = _prepare_geo_matching_payload(
            df_source=df_counts,
            geo_path=geo_path,
            group_col=group_col,
            value_col=value_col,
            match_threshold=match_threshold,
        )
        note = f"Taux de correspondance carte/données : {match_rate:.1%}"
        return gdf_join, df_match, note, value_col, group_col, title
    except Exception as exc:
        return None, None, f"Carte indisponible : {exc}", value_col, group_col, title


def build_static_map_overview(
    df_: pd.DataFrame,
    level: str,
    annotation_mode: str = "aucun",
    annotation_threshold: float = 1,
) -> tuple[Optional[plt.Figure], str]:
    """Construit une carte statique par province ou zone de sante avec les GeoJSON du depot."""
    gdf_join, _, note, value_col, _, title = prepare_overview_map_data(df_, level=level, match_threshold=0.90)
    if gdf_join is None or not value_col or not title:
        return None, note

    try:
        label_col = "_map_label" if "_map_label" in gdf_join.columns else "name"
        fig = carte_statique_matplotlib(
            gdf=gdf_join,
            colonne_valeurs=value_col,
            titre=title,
            annoter=annotation_mode != "aucun",
            mode_annotation=annotation_mode,
            nom_zone=label_col,
            fmt_valeurs="{:.0f}",
            seuil_affichage=annotation_threshold,
            cmap="YlOrRd",
            afficher_fond_carte=False,
            longueur_barre_km=50,
            figsize=(8.4, 6.6) if level == "province" else (8.4, 6.9),
        )
        return fig, note
    except Exception as exc:
        return None, f"Carte indisponible : {exc}"


def render_static_map_overview(title: str, fig: Optional[plt.Figure], note: str) -> None:
    """Affiche une carte dans la synthèse ou un message de fallback propre."""
    st.markdown(f"<div class='cousp-panel-title'>{title}</div>", unsafe_allow_html=True)
    if fig is None:
        st.info(note)
        return
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(note)


def render_interactive_map_overview(
    title: str,
    gdf_join,
    df_map,
    note: str,
    value_col: Optional[str],
    source_df: pd.DataFrame,
    source_label_col: Optional[str],
    chart_key: str,
    clicked_state_key: str,
    filter_state_key: str,
    height: int = 540,
) -> None:
    """Affiche une carte interactive de synthèse et synchronise les filtres latéraux."""
    st.markdown(f"<div class='cousp-panel-title'>{title}</div>", unsafe_allow_html=True)
    if gdf_join is None or df_map is None or not value_col or not source_label_col:
        st.info(note)
        return

    gdf_map_ready = enrich_fuzzy_geo_map_labels(
        gdf_join=gdf_join,
        df_map=df_map,
        df_source=source_df[[source_label_col]].dropna().copy(),
        source_label_col=source_label_col,
    )
    fig_map, gdf_map = build_interactive_geo_map(
        gdf=gdf_map_ready,
        value_col=value_col,
        label_col="_map_label",
        hover_metric_label="Cas",
        height=height,
    )
    if fig_map is None:
        st.info(note)
        return

    selection_state = st.plotly_chart(
        fig_map,
        width="stretch",
        height=height,
        key=chart_key,
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
    selected_label = get_clicked_map_label(clicked_point, gdf_map, label_col="_map_label")
    selected_value = _resolve_map_filter_value(
        selected_label,
        source_df[source_label_col].dropna().unique().tolist(),
    )
    current_filter = st.session_state.get(filter_state_key, ["Toutes"])
    if selected_value and current_filter != [selected_value]:
        st.session_state[clicked_state_key] = selected_value
        st.rerun()

    st.caption(note)
    st.caption("Clique sur un point de la carte pour filtrer le tableau de bord.")


def _upload_geojson_to_temp(upl_obj, suffix: str = ".geojson"):
    """Persist a Streamlit upload to a temporary file so geopandas can read it."""
    if upl_obj is None:
        return None
    data = upl_obj.getvalue()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def _render_detailed_geo_level_map(
    df_f: pd.DataFrame,
    geo_path: Optional[str],
    level_label: str,
    group_col: str,
    value_col: str,
    chart_key: str,
    clicked_state_key: str,
    filter_state_key: str,
    match_threshold: float,
    map_display_mode: str,
    annoter_map: bool,
    annoter_map_mode: str,
    seuil_aff: float,
    afficher_fond: bool,
    longueur_km: float,
    height: int,
    static_title: str,
) -> None:
    """Render one detailed map level with shared controls and filter synchronization."""
    st.subheader(f"Carte des cas par {level_label}")
    if not (geo_path and Path(geo_path).exists() and group_col in df_f.columns):
        st.info(f"Carte {level_label}: charge un GeoJSON {level_label} et assure-toi que la colonne requise est présente.")
        return

    df_carte = df_f[[group_col]].dropna().copy()
    if group_col == COL_ZS and COL_PROV in df_f.columns:
        df_carte[COL_PROV] = df_f.loc[df_carte.index, COL_PROV]
    df_carte[value_col] = 1
    gdf_join, df_map, match_rate, label_source_df, label_source_col, geo_label_col = _prepare_geo_matching_payload(
        df_source=df_carte,
        geo_path=geo_path,
        group_col=group_col,
        value_col=value_col,
        match_threshold=match_threshold,
    )

    st.caption(f"Taux de correspondance (données→carte) : {match_rate:.1%}")
    with st.expander(f"Diagnostic matching {level_label} (pire en haut)"):
        st.dataframe(df_map.head(50), width="stretch")

    if map_display_mode == "Interactive":
        gdf_map_ready = enrich_fuzzy_geo_map_labels(
            gdf_join=gdf_join,
            df_map=df_map,
            df_source=label_source_df,
            source_label_col=label_source_col,
            geo_label_col=geo_label_col,
        )
        fig_map, gdf_map = build_interactive_geo_map(
            gdf=gdf_map_ready,
            value_col=value_col,
            label_col="_map_label",
            hover_metric_label="Cas",
            height=height,
        )
        if fig_map:
            selection_state = st.plotly_chart(
                fig_map,
                width="stretch",
                height=height,
                key=chart_key,
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
            selected_label = get_clicked_map_label(clicked_point, gdf_map, label_col="_map_label")
            current_filter = st.session_state.get(filter_state_key, ["Toutes"])
            if group_col == COL_ZS and clicked_state_key == "map_clicked_zone":
                province_label, zone_label = split_geo_pair_label(selected_label)
                if zone_label:
                    composite_label = build_geo_pair_label(province_label, zone_label)
                    if st.session_state.get(clicked_state_key) != composite_label:
                        st.session_state[clicked_state_key] = composite_label
                        st.rerun()
            else:
                selected_value = _resolve_map_filter_value(
                    selected_label,
                    df_f[group_col].dropna().unique().tolist(),
                )
                if selected_value and current_filter != [selected_value]:
                    st.session_state[clicked_state_key] = selected_value
                    st.rerun()
            st.caption("Clique sur un point de la carte pour mettre à jour le filtre latéral.")
        else:
            st.error(f"Impossible de générer la carte {level_label}.")
    else:
        label_col = geo_label_col if geo_label_col in gdf_join.columns else "name"
        fig = carte_statique_matplotlib(
            gdf=gdf_join,
            colonne_valeurs=value_col,
            titre=static_title,
            annoter=annoter_map,
            mode_annotation=annoter_map_mode,
            nom_zone=label_col,
            fmt_valeurs="{:.0f}",
            seuil_affichage=float(seuil_aff),
            cmap="Reds",
            afficher_fond_carte=afficher_fond,
            longueur_barre_km=float(longueur_km),
        )

        if fig:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.error(f"Impossible de générer la carte {level_label}.")


def render_detailed_maps_tab(
    df_f: pd.DataFrame,
    show_maps: bool,
    idsr_mode: bool,
) -> None:
    """Render the detailed cartography tab for line-list analysis."""
    render_section_title(14, "Cartographie détaillée de la distribution géographique")
    tab_help(
        "Comment lire cet onglet",
        """
        **🎯 Objectif** : explorer la distribution géographique des notifications à un niveau plus détaillé.

        **✅ Inclus**
        - Cartes province et zone de santé
        - Mode statique ou interactif
        - Synchronisation carte → filtres géographiques
        - Diagnostic de matching entre les données et les GeoJSON
        """,
        expanded=False,
    )

    if idsr_mode:
        st.info("La cartographie détaillée de cet onglet est disponible pour les analyses de listes linéaires. En mode IDSR agrégé, utilisez l’onglet IDSR pour les analyses géographiques disponibles.")
        return

    if not show_maps:
        st.markdown(
            """
            <div class="cousp-detail-empty">
                <strong>Cartographie prête à être activée</strong>
                Activez l’option correspondante dans la barre latérale pour afficher les cartes détaillées, charger des GeoJSON personnalisés et utiliser la synchronisation avec les filtres géographiques.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if gpd is None:
        st.warning("geopandas n'est pas installé. Ajoute 'geopandas' dans requirements.txt si tu veux les cartes.")
        return

    st.caption("Cartes provinces / zones avec mode statique ou interactif. Jointure fuzzy tolérante sur 'name'.")

    geo_prov_default = "data/geometry_rdc_provinces.geojson"
    geo_zs_default = "data/geometry_rdc_zones_sante.geojson"
    with st.expander("GeoJSON personnalisés (optionnel)", expanded=False):
        st.caption("Utilise les fichiers du dépôt par défaut, ou remplace-les ici si tu veux des fonds de carte spécifiques.")

        up_col1, up_col2 = st.columns(2)
        with up_col1:
            geo_prov_upl = st.file_uploader(
                " Provinces",
                type=["geojson", "json"],
                key="geojson_prov",
            )
            if Path(geo_prov_default).exists():
                st.caption("Par défaut : GeoJSON provinces du dépôt.")
            else:
                st.caption("Aucun GeoJSON provinces par défaut détecté.")
        with up_col2:
            geo_zs_upl = st.file_uploader(
                " Zones de santé",
                type=["geojson", "json"],
                key="geojson_zs",
            )
            if Path(geo_zs_default).exists():
                st.caption("Par défaut : GeoJSON zones de santé du dépôt.")
            else:
                st.caption("Aucun GeoJSON zones de santé par défaut détecté.")

        col_reset1, col_reset2 = st.columns([1, 3])
        with col_reset1:
            if st.button("↩ Réinitialiser", key="reset_geojson_uploads"):
                st.session_state["geojson_prov"] = None
                st.session_state["geojson_zs"] = None
                st.rerun()
        with col_reset2:
            st.caption("Réinitialise les uploads et revient aux GeoJSON par défaut du dépôt (si présents).")

    geo_prov = _upload_geojson_to_temp(geo_prov_upl) or (geo_prov_default if Path(geo_prov_default).exists() else None)
    geo_zs = _upload_geojson_to_temp(geo_zs_upl) or (geo_zs_default if Path(geo_zs_default).exists() else None)

    with st.expander("Paramètres avancés des cartes", expanded=False):
        seuil_match = st.slider("Seuil de matching (fuzzy)", 0.70, 1.00, 0.90, 0.01)
        map_display_mode = st.radio(
            "Mode de carte",
            ["Statique", "Interactive"],
            index=0,
            horizontal=True,
            key="detail_map_display_mode",
        )
        annoter_map_label = st.selectbox(
            "Contenu des annotations",
            options=list(MAP_ANNOTATION_MODE_OPTIONS.keys()),
            index=3,
            key="detail_map_annotation_mode",
        )
        annoter_map_mode = MAP_ANNOTATION_MODE_OPTIONS[annoter_map_label]
        annoter_map = annoter_map_mode != "aucun"
        seuil_aff = st.number_input("Seuil affichage annotation (valeur >)", min_value=0, max_value=100000, value=1, step=1)
        afficher_fond = False
        longueur_km = 50
        if map_display_mode == "Statique":
            afficher_fond = st.checkbox("Afficher fond de carte (contextily)", value=False)
            longueur_km = st.number_input("Longueur barre échelle (km)", min_value=5, max_value=300, value=50, step=5)
        else:
            st.info("Mode interactif activé : clique sur une zone de la carte pour synchroniser les filtres géographiques.")

    _render_detailed_geo_level_map(
        df_f=df_f,
        geo_path=geo_prov,
        level_label="province",
        group_col=COL_PROV,
        value_col="nb_cas_prov",
        chart_key="detail_province_map",
        clicked_state_key="map_clicked_province",
        filter_state_key="prov_sel",
        match_threshold=seuil_match,
        map_display_mode=map_display_mode,
        annoter_map=annoter_map,
        annoter_map_mode=annoter_map_mode,
        seuil_aff=seuil_aff,
        afficher_fond=afficher_fond,
        longueur_km=longueur_km,
        height=560,
        static_title="RDC - Cas notifiés par province",
    )

    st.divider()

    _render_detailed_geo_level_map(
        df_f=df_f,
        geo_path=geo_zs,
        level_label="zone de santé",
        group_col=COL_ZS,
        value_col="nb_cas_zs",
        chart_key="detail_zone_map",
        clicked_state_key="map_clicked_zone",
        filter_state_key="zs_sel",
        match_threshold=seuil_match,
        map_display_mode=map_display_mode,
        annoter_map=annoter_map,
        annoter_map_mode=annoter_map_mode,
        seuil_aff=seuil_aff,
        afficher_fond=afficher_fond,
        longueur_km=longueur_km,
        height=620,
        static_title="RDC - Cas notifiés par zone de santé",
    )


def _render_idsr_geo_level_map(
    df_f: pd.DataFrame,
    geo_path: Optional[str],
    level_label: str,
    group_col: str,
    cases_col: str,
    chart_key: str,
    filter_state_key: str,
    clear_state_keys: list[str],
    match_threshold: float,
    map_display_mode: str,
    annoter_map: bool,
    annoter_map_mode: str,
    seuil_aff: float,
    afficher_fond: bool,
    longueur_km: float,
    height: int,
    static_title: str,
) -> None:
    """Render a detailed IDSR map aggregated on Total_cas and sync the IDSR filters."""
    st.subheader(f"Carte des cas agrégés par {level_label}")
    if not (geo_path and Path(geo_path).exists() and group_col in df_f.columns and cases_col in df_f.columns):
        st.info(f"Carte {level_label}: charge un GeoJSON {level_label} et assure-toi que les colonnes requises sont présentes.")
        return

    df_carte = df_f[[group_col, cases_col]].copy()
    df_carte[cases_col] = pd.to_numeric(df_carte[cases_col], errors="coerce").fillna(0)
    df_carte = df_carte[df_carte[group_col].notna()].copy()
    if df_carte.empty:
        st.info(f"Aucune donnée exploitable n'est disponible pour la carte {level_label}.")
        return

    if group_col == COL_ZS and COL_PROV in df_f.columns:
        df_carte[COL_PROV] = df_f.loc[df_carte.index, COL_PROV]

    gdf_join, df_map, match_rate, label_source_df, label_source_col, geo_label_col = _prepare_geo_matching_payload(
        df_source=df_carte,
        geo_path=geo_path,
        group_col=group_col,
        value_col=cases_col,
        match_threshold=match_threshold,
    )

    st.caption(f"Taux de correspondance (données→carte) : {match_rate:.1%}")
    with st.expander(f"Diagnostic matching {level_label} (pire en haut)"):
        st.dataframe(df_map.head(50), width="stretch")

    if map_display_mode == "Interactive":
        gdf_map_ready = enrich_fuzzy_geo_map_labels(
            gdf_join=gdf_join,
            df_map=df_map,
            df_source=label_source_df,
            source_label_col=label_source_col,
            geo_label_col=geo_label_col,
        )
        fig_map, gdf_map = build_interactive_geo_map(
            gdf=gdf_map_ready,
            value_col=cases_col,
            label_col="_map_label",
            hover_metric_label="Cas",
            height=height,
        )
        if fig_map is not None:
            selection_state = st.plotly_chart(
                fig_map,
                width="stretch",
                height=height,
                key=chart_key,
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
            selected_label = get_clicked_map_label(clicked_point, gdf_map, label_col="_map_label")
            current_filter = st.session_state.get(filter_state_key, [])
            if group_col == COL_ZS and COL_PROV in df_f.columns:
                province_label, zone_label = split_geo_pair_label(selected_label)
                selected_zone = _resolve_map_filter_value(
                    zone_label,
                    df_f[group_col].dropna().unique().tolist(),
                ) if zone_label else None
                selected_province = _resolve_map_filter_value(
                    province_label,
                    df_f[COL_PROV].dropna().unique().tolist(),
                ) if province_label else None
                if selected_zone and (current_filter != [selected_zone] or st.session_state.get("tab9_prov_sel", []) != ([selected_province] if selected_province else [])):
                    st.session_state[filter_state_key] = [selected_zone]
                    st.session_state["tab9_prov_sel"] = [selected_province] if selected_province else []
                    st.rerun()
            else:
                selected_value = _resolve_map_filter_value(
                    selected_label,
                    df_f[group_col].dropna().unique().tolist(),
                )
                if selected_value and current_filter != [selected_value]:
                    st.session_state[filter_state_key] = [selected_value]
                    for clear_key in clear_state_keys:
                        st.session_state[clear_key] = []
                    st.rerun()
            st.caption("Clique sur une zone pour mettre à jour les filtres IDSR.")
        else:
            st.error(f"Impossible de générer la carte {level_label}.")
    else:
        label_col = geo_label_col if geo_label_col in gdf_join.columns else "name"
        fig = carte_statique_matplotlib(
            gdf=gdf_join,
            colonne_valeurs=cases_col,
            titre=static_title,
            annoter=annoter_map,
            mode_annotation=annoter_map_mode,
            nom_zone=label_col,
            fmt_valeurs="{:.0f}",
            seuil_affichage=float(seuil_aff),
            cmap="Reds",
            afficher_fond_carte=afficher_fond,
            longueur_barre_km=float(longueur_km),
        )
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.error(f"Impossible de générer la carte {level_label}.")


def render_idsr_maps_section(
    df_f: pd.DataFrame,
    province_col: str,
    zs_col: Optional[str],
    cases_col: str = "Total_cas",
) -> None:
    """Render detailed IDSR maps with static/interactive modes."""
    st.markdown("### Cartographie IDSR détaillée")
    st.caption("Cartes provinces / zones de santé sur les cas agrégés filtrés, avec modes statique et interactif.")
    if "idsr_show_detailed_maps" not in st.session_state:
        st.session_state["idsr_show_detailed_maps"] = False

    cases_total = (
        pd.to_numeric(df_f[cases_col], errors="coerce").fillna(0).sum()
        if cases_col in df_f.columns
        else 0
    )
    provinces_total = (
        int(df_f[province_col].dropna().nunique())
        if province_col in df_f.columns
        else 0
    )
    zs_total = (
        int(df_f[zs_col].dropna().nunique())
        if zs_col and zs_col in df_f.columns
        else 0
    )

    with st.expander(
        "Cartographie IDSR détaillée",
        expanded=bool(st.session_state.get("idsr_show_detailed_maps", False)),
    ):
        r1, r2, r3 = st.columns(3)
        r1.metric("Cas agrégés", f"{int(cases_total):,}".replace(",", " "))
        r2.metric("Provinces", f"{provinces_total:,}".replace(",", " "))
        r3.metric("Zones de santé", f"{zs_total:,}".replace(",", " "))

        if st.session_state.get("idsr_show_detailed_maps", False):
            st.caption("La cartographie détaillée est active. Tu peux la refermer si tu veux alléger la page.")
            if st.button("Masquer la cartographie", key="idsr_hide_detailed_maps"):
                st.session_state["idsr_show_detailed_maps"] = False
                st.rerun()
        else:
            st.caption("Charge les cartes uniquement quand tu en as besoin, avec GeoJSON personnalisés et interactions carte → filtres.")
            if st.button("Charger la cartographie", key="idsr_open_detailed_maps"):
                st.session_state["idsr_show_detailed_maps"] = True
                st.rerun()

    if not st.session_state.get("idsr_show_detailed_maps", False):
        return

    if gpd is None:
        st.warning("geopandas n'est pas installé. Ajoute 'geopandas' dans requirements.txt si tu veux les cartes IDSR.")
        return

    geo_prov_default = "data/geometry_rdc_provinces.geojson"
    geo_zs_default = "data/geometry_rdc_zones_sante.geojson"
    with st.expander("GeoJSON personnalisés (optionnel)", expanded=False):
        st.caption("Utilise les fichiers du dépôt par défaut, ou remplace-les ici si tu veux des fonds de carte spécifiques.")

        up_col1, up_col2 = st.columns(2)
        with up_col1:
            geo_prov_upl = st.file_uploader(
                " Provinces IDSR",
                type=["geojson", "json"],
                key="idsr_geojson_prov",
            )
            if Path(geo_prov_default).exists():
                st.caption("Par défaut : GeoJSON provinces du dépôt.")
            else:
                st.caption("Aucun GeoJSON provinces par défaut détecté.")
        with up_col2:
            geo_zs_upl = st.file_uploader(
                " Zones de santé IDSR",
                type=["geojson", "json"],
                key="idsr_geojson_zs",
            )
            if Path(geo_zs_default).exists():
                st.caption("Par défaut : GeoJSON zones de santé du dépôt.")
            else:
                st.caption("Aucun GeoJSON zones de santé par défaut détecté.")

        col_reset1, col_reset2 = st.columns([1, 3])
        with col_reset1:
            if st.button("↩ Réinitialiser", key="idsr_reset_geojson_uploads"):
                st.session_state["idsr_geojson_prov"] = None
                st.session_state["idsr_geojson_zs"] = None
                st.rerun()
        with col_reset2:
            st.caption("Réinitialise les GeoJSON IDSR téléversés et revient aux fichiers par défaut du dépôt.")

    geo_prov = _upload_geojson_to_temp(geo_prov_upl) or (geo_prov_default if Path(geo_prov_default).exists() else None)
    geo_zs = _upload_geojson_to_temp(geo_zs_upl) or (geo_zs_default if Path(geo_zs_default).exists() else None)

    with st.expander("Paramètres avancés des cartes IDSR", expanded=False):
        seuil_match = st.slider("Seuil de matching (fuzzy)", 0.70, 1.00, 0.90, 0.01, key="idsr_map_match_threshold")
        map_display_mode = st.radio(
            "Mode de carte",
            ["Statique", "Interactive"],
            index=0,
            horizontal=True,
            key="idsr_map_display_mode",
        )
        annoter_map_label = st.selectbox(
            "Contenu des annotations",
            options=list(MAP_ANNOTATION_MODE_OPTIONS.keys()),
            index=3,
            key="idsr_map_annotation_mode",
        )
        annoter_map_mode = MAP_ANNOTATION_MODE_OPTIONS[annoter_map_label]
        annoter_map = annoter_map_mode != "aucun"
        seuil_aff = st.number_input(
            "Seuil affichage annotation (valeur >)",
            min_value=0,
            max_value=100000,
            value=1,
            step=1,
            key="idsr_map_annotation_threshold",
        )
        afficher_fond = False
        longueur_km = 50
        if map_display_mode == "Statique":
            afficher_fond = st.checkbox("Afficher fond de carte (contextily)", value=False, key="idsr_map_show_background")
            longueur_km = st.number_input(
                "Longueur barre échelle (km)",
                min_value=5,
                max_value=300,
                value=50,
                step=5,
                key="idsr_map_scale_km",
            )
        else:
            st.info("Mode interactif activé : clique sur une zone pour mettre à jour les filtres IDSR.")

    if zs_col:
        col_prov, col_zs = st.columns(2)
        with col_prov:
            _render_idsr_geo_level_map(
                df_f=df_f,
                geo_path=geo_prov,
                level_label="province",
                group_col=province_col,
                cases_col=cases_col,
                chart_key="idsr_detail_province_map",
                filter_state_key="tab9_prov_sel",
                clear_state_keys=["tab9_zs_sel"],
                match_threshold=seuil_match,
                map_display_mode=map_display_mode,
                annoter_map=annoter_map,
                annoter_map_mode=annoter_map_mode,
                seuil_aff=seuil_aff,
                afficher_fond=afficher_fond,
                longueur_km=longueur_km,
                height=560,
                static_title="RDC - Cas agrégés IDSR par province",
            )
        with col_zs:
            _render_idsr_geo_level_map(
                df_f=df_f,
                geo_path=geo_zs,
                level_label="zone de santé",
                group_col=zs_col,
                cases_col=cases_col,
                chart_key="idsr_detail_zs_map",
                filter_state_key="tab9_zs_sel",
                clear_state_keys=["tab9_prov_sel"],
                match_threshold=seuil_match,
                map_display_mode=map_display_mode,
                annoter_map=annoter_map,
                annoter_map_mode=annoter_map_mode,
                seuil_aff=seuil_aff,
                afficher_fond=afficher_fond,
                longueur_km=longueur_km,
                height=620,
                static_title="RDC - Cas agrégés IDSR par zone de santé",
            )
    else:
        _render_idsr_geo_level_map(
            df_f=df_f,
            geo_path=geo_prov,
            level_label="province",
            group_col=province_col,
            cases_col=cases_col,
            chart_key="idsr_detail_province_map",
            filter_state_key="tab9_prov_sel",
            clear_state_keys=["tab9_zs_sel"],
            match_threshold=seuil_match,
            map_display_mode=map_display_mode,
            annoter_map=annoter_map,
            annoter_map_mode=annoter_map_mode,
            seuil_aff=seuil_aff,
            afficher_fond=afficher_fond,
            longueur_km=longueur_km,
            height=560,
            static_title="RDC - Cas agrégés IDSR par province",
        )

# DECISION_ORIENTED_OVERRIDES

def build_dashboard_kpi_payload(df_: pd.DataFrame) -> Dict[str, Any]:
    """Version orientee decision de la synthese d'accueil."""
    kpi = compute_indicators(df_)
    weekly = build_weekly_overview_table(df_)
    weekly = _normalize_metric_alias_columns(weekly)
    classification_counts = _build_classification_counts(df_)
    issue_counts = _build_issue_counts(df_)
    lab_counts = _build_lab_counts(df_, kpi)
    alert_proxy = _build_alert_proxy_counts(df_, classification_counts)
    quality_focus = _build_quality_focus_metrics(df_)
    delay_focus = _build_delay_focus_metrics(df_)
    hotspots, hotspots_meta = _build_hotspots_table(df_)
    priority_actions = _build_priority_actions(df_)

    analyses_available = lab_counts.get("analyses", 0) > 0
    has_classification = bool(alert_proxy.get("has_classification"))
    confirmed_count = next(
        (
            int(value)
            for key, value in classification_counts.items()
            if "confirm" in str(key).strip().lower()
        ),
        0,
    )
    if confirmed_count == 0 and analyses_available:
        confirmed_count = int(lab_counts.get("positifs", 0))

    surveillance_chain = [
        {"label": "Alertes notifiees", "value": int(alert_proxy.get("notified_alerts", 0)), "subtitle": "Notifications issues de la line list", "theme": "blue", "color": "#2b74ca", "available": True},
        {"label": "Alertes verifiees", "value": alert_proxy.get("verified_alerts"), "subtitle": "Proxy base sur la classification finale", "theme": "green", "color": "#27a063", "available": alert_proxy.get("verified_alerts") is not None},
        {"label": "Cas probables", "value": int(classification_counts.get("Probable", 0)) if has_classification else None, "subtitle": "Classification standardisee", "theme": "orange", "color": "#f29b38", "available": has_classification},
        {"label": "Cas suspects", "value": int(classification_counts.get("Suspect", 0)) if has_classification else None, "subtitle": "Classification standardisee", "theme": "amber", "color": "#f2a53a", "available": has_classification},
        {"label": "Echantillons recus", "value": int(lab_counts.get("recus", 0)) if lab_counts.get("recus", 0) > 0 else int(lab_counts.get("preleves", 0)) if lab_counts.get("preleves", 0) > 0 else None, "subtitle": "Reception labo ou proxy prelevements documentes", "theme": "purple", "color": "#7b4dff", "available": bool(lab_counts.get("recus", 0) > 0 or lab_counts.get("preleves", 0) > 0)},
        {"label": "Echantillons analyses", "value": int(lab_counts.get("analyses", 0)) if analyses_available else None, "subtitle": "Resultats valides ou invalides disponibles", "theme": "navy", "color": "#425a7d", "available": analyses_available},
        {"label": "Cas positifs", "value": int(lab_counts.get("positifs", 0)) if analyses_available else None, "subtitle": "Resultats positifs", "theme": "red", "color": "#e84b4b", "available": analyses_available},
        {"label": "Cas negatifs", "value": int(lab_counts.get("negatifs", 0)) if analyses_available else None, "subtitle": "Resultats negatifs", "theme": "slate", "color": "#5d6d86", "available": analyses_available},
        {"label": "Resultats invalides", "value": int(lab_counts.get("invalides", 0)) if analyses_available else None, "subtitle": "Analyses non concluantes", "theme": "orange", "color": "#c68423", "available": analyses_available},
        {"label": "Cas confirmes", "value": confirmed_count if has_classification else int(lab_counts.get("positifs", 0)) if analyses_available else None, "subtitle": "Classification finale ou positivite labo", "theme": "green", "color": "#1f8d57", "available": bool(has_classification or analyses_available)},
    ]

    provinces_epid = get_provinces_epid()
    total_provinces = len(EPIDEMIE)
    total_provinces_epid = len(provinces_epid)
    reported_provinces = int(df_[COL_PROV].dropna().nunique()) if COL_PROV in df_.columns else 0
    reported_epid_provinces = int(df_.loc[df_[COL_PROV].isin(provinces_epid), COL_PROV].dropna().nunique()) if COL_PROV in df_.columns else 0
    reported_zs = int(df_[COL_ZS].dropna().nunique()) if COL_ZS in df_.columns else 0

    week_min = '-'
    week_max = '-'
    if not weekly.empty:
        week_min = str(weekly['label'].iloc[0])
        week_max = str(weekly['label'].iloc[-1])
    elif COL_WNUM in df_.columns and pd.to_numeric(df_[COL_WNUM], errors='coerce').notna().any():
        week_values = pd.to_numeric(df_[COL_WNUM], errors='coerce').dropna().astype(int)
        week_min = f"SE{int(week_values.min()):02d}"
        week_max = f"SE{int(week_values.max()):02d}"

    latest = weekly.iloc[-1].to_dict() if not weekly.empty else {}
    previous = weekly.iloc[-2].to_dict() if len(weekly) > 1 else {}
    promptitude_pct, promptitude_n = pct_under_threshold(df_.get('delai_onset_to_adm'), get_session_int('seuil_jours', 2))
    analysis_period = compute_analysis_period_value(df_)

    return {
        'cases': int(kpi['n_cases']), 'deaths': int(kpi['n_deaths']), 'cfr': float(kpi['cfr_pct']) if not pd.isna(kpi['cfr_pct']) else np.nan,
        'week_span': f'{week_min} -> {week_max}' if week_min != '-' and week_max != '-' else '-', 'week_min': week_min, 'week_max': week_max,
        'reported_epid_provinces': reported_epid_provinces, 'total_provinces_epid': total_provinces_epid, 'reported_provinces': reported_provinces, 'total_provinces': total_provinces,
        'reported_zs': reported_zs, 'coverage_epid_pct': safe_pct(reported_epid_provinces, total_provinces_epid), 'coverage_nat_pct': safe_pct(reported_provinces, total_provinces),
        'weekly': weekly, 'latest': latest, 'previous': previous, 'promptitude_pct': promptitude_pct, 'promptitude_n': promptitude_n, 'analysis_period': analysis_period,
        'top_province': _safe_top_label(df_, COL_PROV), 'top_zs': _safe_top_label(df_, COL_ZS), 'classification_counts': classification_counts, 'issue_counts': issue_counts,
        'lab_counts': lab_counts, 'quality_focus': quality_focus, 'delay_focus': delay_focus, 'hotspots': hotspots, 'hotspots_meta': hotspots_meta,
        'priority_actions': priority_actions, 'surveillance_chain': surveillance_chain, 'alert_proxy': alert_proxy,
    }


def build_surveillance_cascade_figure(payload: Dict[str, Any]) -> Optional[go.Figure]:
    chain = [item for item in payload.get('surveillance_chain', []) if item.get('available', True) and item.get('value') is not None]
    if not chain:
        return None
    labels = [str(item['label']) for item in chain]
    values = [int(item['value']) for item in chain]
    colors = [item.get('color', '#2b74ca') for item in chain]
    fig = go.Figure(go.Bar(x=values[::-1], y=labels[::-1], orientation='h', marker=dict(color=colors[::-1], line=dict(color='rgba(255,255,255,0.65)', width=1.2)), text=[format_metric_value(value) for value in values[::-1]], textposition='outside', cliponaxis=False, hovertemplate='%{y}: %{x}<extra></extra>'))
    fig.update_layout(height=400, margin=dict(t=10, r=28, b=10, l=12), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False, showgrid=False, zeroline=False), yaxis=dict(title=None, ticks='', showgrid=False, automargin=True))
    return fig


def render_surveillance_chain_section(payload: Dict[str, Any]) -> None:
    chain = payload.get('surveillance_chain', [])
    if not chain:
        return
    cards_html = []
    steps_html = []
    for item in chain:
        theme = str(item.get('theme', 'blue'))
        label = html.escape(str(item.get('label', '')))
        raw_value = item.get('value')
        value = html.escape(format_metric_value(raw_value) if raw_value is not None else '-')
        subtitle = html.escape(str(item.get('subtitle', '')))
        cards_html.append(f"""
<div class=\"cousp-chain-card {theme}\">
  <div class=\"cousp-chain-label\">{label}</div>
  <div class=\"cousp-chain-value\">{value}</div>
  <div class=\"cousp-chain-subtitle\">{subtitle}</div>
</div>
""")
        if item.get('available', True) and raw_value is not None:
            steps_html.append(f"""
<div class=\"cousp-chain-step\">
  <span class=\"dot {theme}\"></span>
  <span>{label}</span>
</div>
""")
    st.markdown(f"""
<div class=\"cousp-chain-panel\">
  <div class=\"cousp-panel-title\">Chaine de surveillance</div>
  <div class=\"cousp-chain-grid\">{''.join(cards_html)}</div>
  <div class=\"cousp-chain-stepper\">{''.join(steps_html)}</div>
</div>
""", unsafe_allow_html=True)


def render_hotspots_panel(payload: Dict[str, Any]) -> None:
    table = payload.get('hotspots')
    meta = payload.get('hotspots_meta', {}) or {}
    st.markdown(f"<div class='cousp-panel-title'>{html.escape(str(meta.get('title', 'Top territoires actifs')))}</div>", unsafe_allow_html=True)
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        st.info("Aucune ventilation geographique exploitable n'est disponible pour le top des territoires.")
        return
    rows = []
    for _, row in table.iterrows():
        place = html.escape(str(row.get('Lieu', '-')))
        province = html.escape(str(row.get('Province', '-')))
        zone = html.escape(str(row.get('Zone de santé', '-')))
        context = zone if zone != '-' else province
        cases = format_metric_value(row.get('Cas', 0))
        confirmed = format_metric_value(row.get('Confirmés', 0))
        theme = html.escape(str(row.get('Niveau_theme', 'blue')))
        level = html.escape(str(row.get('Niveau', 'Active')))
        rows.append(f"""
<tr>
  <td><strong>{place}</strong><span>{context}</span></td>
  <td>{cases}</td>
  <td>{confirmed}</td>
  <td><span class=\"cousp-badge {theme}\">{level}</span></td>
</tr>
""")
    subtitle = html.escape(str(meta.get('subtitle', '')))
    subtitle_html = f"<div class='cousp-panel-subtitle'>{subtitle}</div>" if subtitle else ''
    st.markdown(f"""
{subtitle_html}
<div class=\"cousp-summary-box\">
  <table class=\"cousp-mini-table\">
    <thead>
      <tr>
        <th>Lieu</th>
        <th>Cas</th>
        <th>Confirmes</th>
        <th>Niveau</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

