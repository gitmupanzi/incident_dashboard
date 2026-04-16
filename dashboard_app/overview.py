from dashboard_app.domain import *
from dashboard_app.domain import (
    _is_yes_series,
    _resolve_map_filter_value,
    _tdr_result_norm,
)
from dashboard_app.core import _normalize_metric_alias_columns

def tab_help(title: str, md: str, expanded: bool = False):
    with st.expander(f"ℹ️ {title}", expanded=expanded):
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


def build_global_summary_table(df_: pd.DataFrame) -> pd.DataFrame:
    n_cases = int(len(df_))
    n_deaths = int(df_["is_death"].sum()) if "is_death" in df_.columns else 0
    cfr = (n_deaths / n_cases * 100.0) if n_cases else np.nan
    rows = [("Nombre total de cas", n_cases), ("Nombre total de décès", n_deaths), ("Létalité (%)", None if pd.isna(cfr) else round(cfr, 2))]
    for col, label in [(COL_PROV, "Nombre de provinces touchées"), (COL_ZS, "Nombre de zones de santé touchées"), (COL_AS, "Nombre d'aires de santé touchées")]:
        if col in df_.columns:
            rows.append((label, int(df_[col].dropna().nunique())))
    for col, label in [(DATE_NOTIF, "Période de notification"), (DATE_ONSET, "Période début maladie")]:
        if col in df_.columns and df_[col].notna().any():
            dmin = pd.to_datetime(df_[col], errors="coerce").min()
            dmax = pd.to_datetime(df_[col], errors="coerce").max()
            if pd.notna(dmin) and pd.notna(dmax):
                rows.append((label, f"{dmin:%Y-%m-%d} à {dmax:%Y-%m-%d}"))
                break
    if "YW" in df_.columns and df_["YW"].notna().any():
        rows.append(("Semaines épidémiologiques couvertes", f"{df_['YW'].dropna().astype(str).min()} à {df_['YW'].dropna().astype(str).max()}"))
    elif COL_WNUM in df_.columns and df_[COL_WNUM].notna().any():
        rows.append(("Semaines épidémiologiques couvertes", f"SE{int(df_[COL_WNUM].min()):02d} à SE{int(df_[COL_WNUM].max()):02d}"))
    return pd.DataFrame(rows, columns=["Indicateur", "Valeur"])


def build_frequency_table(df_: pd.DataFrame, col: str, top_n: int | None = None) -> pd.DataFrame:
    if col not in df_.columns:
        return pd.DataFrame(columns=[col, 'n', '%'])
    freq = df_[col].fillna('Inconnu').astype(str).str.strip().replace('', 'Inconnu').value_counts(dropna=False).reset_index()
    freq.columns = [col, 'n']
    freq['%'] = (freq['n'] / max(len(df_), 1) * 100).round(1)
    if top_n is not None:
        freq = freq.head(int(top_n))
    return freq


def build_simple_lab_table(df_: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_cases = int(len(df_))
    if COL_PREL in df_.columns:
        n_prel = int(_is_yes_series(df_[COL_PREL]).sum())
        rows.append(("Prélèvement réalisé", n_prel, round(n_prel / n_cases * 100, 1) if n_cases else np.nan))
    if COL_TDR in df_.columns:
        n_tdr = int(_is_yes_series(df_[COL_TDR]).sum())
        rows.append(("TDR réalisé", n_tdr, round(n_tdr / n_cases * 100, 1) if n_cases else np.nan))
    if COL_TDRR in df_.columns:
        res_n = _tdr_result_norm(df_[COL_TDRR])
        n_pos = int(res_n.isin(TDR_POS_SET).sum())
        n_neg = int(res_n.isin(TDR_NEG_SET).sum())
        n_valid = n_pos + n_neg
        rows.append(("Résultat valide (Pos/Nég)", n_valid, round(n_valid / n_cases * 100, 1) if n_cases else np.nan))
        rows.append(("TDR positif", n_pos, round(n_pos / n_valid * 100, 1) if n_valid else np.nan))
        rows.append(("TDR négatif", n_neg, round(n_neg / n_valid * 100, 1) if n_valid else np.nan))
    return pd.DataFrame(rows, columns=["Indicateur labo", "n", "%"])


def build_weekly_lab_summary(df_: pd.DataFrame) -> pd.DataFrame:
    """Construit un suivi hebdomadaire des tests valides et de la positivite."""
    week_col = resolve_week_column(df_)
    if week_col is None or COL_TDRR not in df_.columns:
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

    res_n = _tdr_result_norm(tmp[COL_TDRR])
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


def build_delay_summary_table(df_: pd.DataFrame, delay_cols: list[str]) -> pd.DataFrame:
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
            f'% ≤ {seuil_jours} jours': round(float((s <= seuil_jours).mean() * 100), 1),
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


def build_who_narrative_summary(df_: pd.DataFrame) -> str:
    """
    Résumé automatisé rédigé dans un langage de surveillance compatible
    avec l'usage OMS/IDSR : période, charge de morbidité, létalité observée,
    profil des cas, distribution géographique et informations de laboratoire.
    """
    n_cases = int(len(df_))
    n_deaths = int(df_["is_death"].sum()) if "is_death" in df_.columns else 0
    cfr = safe_pct(n_deaths, n_cases)

    period_txt = "sur une période non documentée"
    for col in [DATE_NOTIF, DATE_ONSET]:
        if col in df_.columns and pd.to_datetime(df_[col], errors="coerce").notna().any():
            s = pd.to_datetime(df_[col], errors="coerce")
            period_txt = f"sur la période du {s.min():%d/%m/%Y} au {s.max():%d/%m/%Y}"
            break

    week_txt = ""
    if "YW" in df_.columns and df_["YW"].notna().any():
        w = df_["YW"].dropna().astype(str)
        week_txt = f" ; couverture hebdomadaire : {w.min()} à {w.max()}"
    elif COL_WNUM in df_.columns and pd.to_numeric(df_[COL_WNUM], errors='coerce').notna().any():
        w = pd.to_numeric(df_[COL_WNUM], errors='coerce').dropna().astype(int)
        week_txt = f" ; couverture hebdomadaire : SE{w.min():02d} à SE{w.max():02d}"

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
    if not lab_tbl.empty and "TDR positif" in lab_tbl["Indicateur labo"].values:
        row = lab_tbl.loc[lab_tbl["Indicateur labo"] == "TDR positif"].iloc[0]
        pct_val = row["%"] if pd.notna(row["%"]) else "-"
        lab_txt = (
            f"Au plan du laboratoire, {int(row['n'])} résultat(s) positif(s) ont été documentés, "
            f"pour une positivité observée de {pct_val} % parmi les résultats interprétables."
        )

    cfr_txt = "non calculable" if pd.isna(cfr) else f"{cfr:.2f}"
    geo_txt = (
        f"La province la plus représentée est « {prov_top} »"
        if prov_top != "non disponible"
        else "La distribution provinciale n’est pas suffisamment documentée"
    )
    if zs_top != "non disponible":
        geo_txt += f", avec une concentration notable des notifications dans la zone de santé « {zs_top} »"

    return (
        f"Au total, {n_cases} cas et {n_deaths} décès ont été enregistrés {period_txt}{week_txt}. "
        f"La létalité observée (CFR) est estimée à {cfr_txt} %. "
        f"Le profil des cas met principalement en évidence le sexe « {sex_top} » et le groupe d’âge « {age_top} ». "
        f"{geo_txt}. {lab_txt} "
        f"Cette synthèse descriptive doit être interprétée en tenant compte de la complétude, de la promptitude et de la qualité des données disponibles."
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


def build_weekly_overview_table(df_: pd.DataFrame) -> pd.DataFrame:
    """Construit la série hebdomadaire standard utilisée dans la page d'accueil."""
    week_col = resolve_week_column(df_)
    if week_col is None or df_.empty:
        return pd.DataFrame(columns=["order_key", "label", "Cas", "Décès", "Létalité (%)"])

    weekly = df_.copy()
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
        .agg(Cas=("label", "size"), Décès=("is_death", "sum"))
        .sort_values("order_key")
    )
    grouped["Létalité (%)"] = np.where(grouped["Cas"] > 0, grouped["Décès"] / grouped["Cas"] * 100.0, np.nan)
    return grouped


def build_dashboard_kpi_payload(df_: pd.DataFrame) -> Dict[str, Any]:
    """Calcule les KPI principaux de la page d'accueil."""
    kpi = compute_indicators(df_)
    weekly = build_weekly_overview_table(df_)
    weekly = _normalize_metric_alias_columns(weekly)

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
        "top_province": _safe_top_label(df_, COL_PROV),
        "top_zs": _safe_top_label(df_, COL_ZS),
    }


def render_context_row(files_used: list[str], disease_key: str, df_: pd.DataFrame, payload: Dict[str, Any]) -> None:
    """Affiche quelques repères analytiques juste sous le bandeau principal."""
    disease_label = DISEASE_SPECS.get(disease_key, {}).get("label", str(disease_key))
    source_value = "Aucun fichier" if not files_used else str(files_used[0]).replace("upload:", "")
    if len(source_value) > 42:
        source_value = source_value[:39] + "..."

    if DATE_NOTIF in df_.columns and pd.to_datetime(df_[DATE_NOTIF], errors="coerce").notna().any():
        notif_dates = pd.to_datetime(df_[DATE_NOTIF], errors="coerce")
        period_value = f"{notif_dates.min():%d/%m/%Y} -> {notif_dates.max():%d/%m/%Y}"
    else:
        period_value = payload.get("week_span", "-")

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
    return f"""
<div class="cousp-kpi-card {theme}{span_class}">
  <div class="cousp-kpi-title">{title}</div>
  <div class="cousp-kpi-value">{value}</div>
  <div class="cousp-kpi-subtitle">{subtitle}</div>
</div>
"""


def render_dashboard_kpis(payload: Dict[str, Any]) -> None:
    """Affiche la ligne horizontale des KPI principaux."""
    cards = [
        ("Total cas", format_metric_value(payload.get("cases", 0)), "Périmètre filtré", "blue", 1),
        ("Total décès", format_metric_value(payload.get("deaths", 0)), "Périmètre filtré", "navy", 1),
        ("CFR (%)", format_metric_value(payload.get("cfr"), decimals=2), "Létalité observée", "orange", 1),
        ("Semaine min -> max", payload.get("week_span", "-"), "Fenêtre analytique", "blue", 2),
        (
            "Provinces épidémiques",
            f"{payload.get('reported_epid_provinces', 0)} / {payload.get('total_provinces_epid', 0)}",
            "-" if pd.isna(payload.get("coverage_epid_pct")) else f"{payload.get('coverage_epid_pct', 0):.1f}% de couverture",
            "green",
            1,
        ),
        (
            "Couverture nationale",
            f"{payload.get('reported_provinces', 0)} / {payload.get('total_provinces', 0)}",
            "-" if pd.isna(payload.get("coverage_nat_pct")) else f"{payload.get('coverage_nat_pct', 0):.1f}% de couverture",
            "green",
            1,
        ),
        ("Zones de santé touchées", format_metric_value(payload.get("reported_zs", 0)), "Notifications consolidées", "green", 1),
    ]
    cards_html = "".join(build_dashboard_kpi_card_html(*card) for card in cards)
    st.markdown(f"<div class='cousp-kpi-grid'>{cards_html}</div>", unsafe_allow_html=True)


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
        gdf_geo = gpd.read_file(geo_path)
        df_counts = df_[[group_col]].dropna().copy()
        df_counts[value_col] = 1
        df_counts = df_counts.groupby(group_col, as_index=False)[value_col].sum()
        gdf_join, df_match, match_rate = joindre_donnees_fuzzy_geo(
            carte_gdf=gdf_geo,
            df_donnees=df_counts,
            colonne_cle_geo="name",
            colonne_cle_data=group_col,
            colonne_valeurs=value_col,
            seuil=match_threshold,
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
        fig = carte_statique_matplotlib(
            gdf=gdf_join,
            colonne_valeurs=value_col,
            titre=title,
            annoter=annotation_mode != "aucun",
            mode_annotation=annotation_mode,
            nom_zone="name",
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


