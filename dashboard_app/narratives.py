"""Narrative helpers and surveillance/SITREP text builders."""

from dashboard_app.overview import compute_analysis_period_value
from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())

def _prepare_surveillance_period_scope(df_scope: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit un repère temporel robuste pour les fenêtres de surveillance."""
    if df_scope is None or df_scope.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["order", "label"])

    scoped = df_scope.copy()
    scoped["_surv_order"] = np.nan
    scoped["_surv_label"] = pd.NA

    if COL_WNUM in scoped.columns and pd.to_numeric(scoped[COL_WNUM], errors="coerce").notna().any():
        week_num = pd.to_numeric(scoped[COL_WNUM], errors="coerce")
        if COL_YEAR in scoped.columns and pd.to_numeric(scoped[COL_YEAR], errors="coerce").notna().any():
            year_num = pd.to_numeric(scoped[COL_YEAR], errors="coerce")
            mask = week_num.notna() & year_num.notna()
            scoped.loc[mask, "_surv_order"] = (year_num[mask] * 100) + week_num[mask]
            scoped.loc[mask, "_surv_label"] = [
                f"SE{int(w):02d}-{int(y)}"
                for y, w in zip(year_num[mask], week_num[mask])
            ]
        else:
            mask = week_num.notna()
            scoped.loc[mask, "_surv_order"] = week_num[mask]
            scoped.loc[mask, "_surv_label"] = [f"SE{int(w):02d}" for w in week_num[mask]]
    elif "YW" in scoped.columns and scoped["YW"].notna().any():
        yw_text = scoped["YW"].astype("string").fillna("").str.strip()
        parsed = yw_text.str.extract(r"(?P<year>\d{4}).*?(?P<week>\d{1,2})")
        year_num = pd.to_numeric(parsed["year"], errors="coerce")
        week_num = pd.to_numeric(parsed["week"], errors="coerce")
        mask = year_num.notna() & week_num.notna()
        if mask.any():
            scoped.loc[mask, "_surv_order"] = (year_num[mask] * 100) + week_num[mask]
            scoped.loc[mask, "_surv_label"] = [
                f"SE{int(w):02d}-{int(y)}"
                for y, w in zip(year_num[mask], week_num[mask])
            ]
        else:
            valid_mask = yw_text != ""
            scoped.loc[valid_mask, "_surv_label"] = yw_text[valid_mask]
            categories = pd.Categorical(scoped.loc[valid_mask, "_surv_label"])
            scoped.loc[valid_mask, "_surv_order"] = categories.codes

    reference = (
        scoped.loc[scoped["_surv_order"].notna() & scoped["_surv_label"].notna(), ["_surv_order", "_surv_label"]]
        .drop_duplicates()
        .rename(columns={"_surv_order": "order", "_surv_label": "label"})
        .sort_values("order")
        .reset_index(drop=True)
    )
    return scoped, reference


def _build_surveillance_top_table(
    df_scope: pd.DataFrame,
    group_cols: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """Construit un top géographique avec cas, décès et létalité."""
    valid_group_cols = [c for c in group_cols if c in df_scope.columns]
    if df_scope is None or df_scope.empty or not valid_group_cols:
        return pd.DataFrame()

    tmp = df_scope.copy()
    tmp["_cas_"] = 1
    tmp["_deces_"] = (
        pd.to_numeric(tmp["is_death"], errors="coerce").fillna(0).astype(int)
        if "is_death" in tmp.columns
        else 0
    )

    for col in valid_group_cols:
        tmp[col] = (
            tmp[col]
            .fillna("Inconnu")
            .astype(str)
            .str.strip()
            .replace("", "Inconnu")
        )

    grouped = (
        tmp.groupby(valid_group_cols, as_index=False)
        .agg(Cas=("_cas_", "sum"), Décès=("_deces_", "sum"))
        .sort_values(["Cas", "Décès"], ascending=[False, False])
        .head(int(top_n))
        .copy()
    )
    grouped["Létalité (%)"] = np.where(
        grouped["Cas"] > 0,
        grouped["Décès"] / grouped["Cas"] * 100.0,
        np.nan,
    ).round(2)
    grouped.insert(0, "Rang", range(1, len(grouped) + 1))
    return grouped


def _describe_surveillance_top_table(
    table: pd.DataFrame,
    label_cols: list[str],
    total_cases: int,
    empty_label: str,
) -> str:
    """Génère une lecture rapide du top 5 géographique."""
    if table.empty or total_cases <= 0:
        return empty_label

    leader = table.iloc[0]
    leader_parts = [str(leader[col]) for col in label_cols if col in table.columns and pd.notna(leader[col])]
    leader_label = " / ".join([part for part in leader_parts if part and part != "nan"]) or "Non renseigné"
    leader_share = float(leader["Cas"]) / float(total_cases) * 100.0
    top5_share = float(table["Cas"].sum()) / float(total_cases) * 100.0
    return (
        f"{leader_label} concentre {leader_share:.1f}% des cas de cette période ; "
        f"le top 5 en concentre {top5_share:.1f}%."
    )


def _surveillance_scope_kpis(df_scope: pd.DataFrame) -> tuple[int, int, float]:
    """Retourne cas, décès et létalité pour une fenêtre donnée."""
    if df_scope is None or df_scope.empty:
        return 0, 0, np.nan
    total_cases = int(len(df_scope))
    total_deaths = int(
        pd.to_numeric(df_scope["is_death"], errors="coerce").fillna(0).sum()
    ) if "is_death" in df_scope.columns else 0
    cfr = safe_pct(total_deaths, total_cases)
    return total_cases, total_deaths, cfr


def _surveillance_clean_text_series(series: pd.Series) -> pd.Series:
    """Nettoie une série texte pour les comptages métier."""
    if series is None:
        return pd.Series(dtype="object")
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned[cleaned != ""]


def _surveillance_period_bounds(df_scope: pd.DataFrame) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[str]]:
    """Déduit la meilleure période lisible à partir de la fenêtre analytique active."""
    period_value = compute_analysis_period_value(df_scope)
    if period_value and period_value != "-" and "->" in period_value:
        start_txt, end_txt = [part.strip() for part in period_value.split("->", 1)]
        start_dt = pd.to_datetime(start_txt, errors="coerce", dayfirst=True)
        end_dt = pd.to_datetime(end_txt, errors="coerce", dayfirst=True)
        if pd.notna(start_dt) and pd.notna(end_dt):
            return pd.Timestamp(start_dt), pd.Timestamp(end_dt), "analysis_period"

    for col in [DATE_NOTIF, DATE_ONSET, DATE_ADM, DATE_INV]:
        if col in df_scope.columns:
            dt = pd.to_datetime(df_scope[col], errors="coerce")
            if dt.notna().any():
                return dt.min(), dt.max(), col
    return None, None, None


def _format_surveillance_date(dt_value: Optional[pd.Timestamp]) -> str:
    """Formate une date de résumé si disponible."""
    if dt_value is None or pd.isna(dt_value):
        return "-"
    try:
        return pd.Timestamp(dt_value).strftime("%d/%m/%Y")
    except Exception:
        return str(dt_value)


def _normalize_severity_label(value: Any) -> str:
    """Harmonise les modalités de sévérité/déshydratation."""
    if value is None or pd.isna(value):
        return MISSING_LABEL
    raw = str(value).strip()
    if not raw:
        return MISSING_LABEL
    norm = "".join(
        c for c in unicodedata.normalize("NFD", raw.lower())
        if unicodedata.category(c) != "Mn"
    )
    if "sever" in norm:
        return "Sévère"
    if "moder" in norm:
        return "Modérée"
    if "leger" in norm or "legere" in norm or "mild" in norm:
        return "Légère"
    if "inconnu" in norm or "non renseigne" in norm or norm == "na":
        return MISSING_LABEL
    return raw


def _build_severity_summary_text(df_scope: pd.DataFrame) -> Optional[str]:
    """Construit une phrase courte sur la sévérité si la variable est disponible."""
    if COL_DEHY not in df_scope.columns:
        return None
    severity = _surveillance_clean_text_series(df_scope[COL_DEHY])
    if severity.empty:
        return None
    counts = (
        severity.map(_normalize_severity_label)
        .value_counts()
        .rename_axis("Modalité")
        .reset_index(name="n")
    )
    preferred_order = {"Légère": 0, "Modérée": 1, "Sévère": 2, MISSING_LABEL: 3}
    counts["_order"] = counts["Modalité"].map(lambda x: preferred_order.get(x, 99))
    counts = counts.sort_values(["_order", "n"], ascending=[True, False])
    parts = [f"{row['Modalité']} : {int(row['n'])}" for _, row in counts.iterrows()]
    if not parts:
        return None
    return "Degré de sévérité selon la variable disponible : " + " | ".join(parts)


def _build_investigation_summary_text(df_scope: pd.DataFrame) -> Optional[str]:
    """Construit un résumé standard des cas investigués."""
    if DATE_INV not in df_scope.columns:
        return None
    total_cases = int(len(df_scope))
    if total_cases <= 0:
        return None
    investigated = int(pd.to_datetime(df_scope[DATE_INV], errors="coerce").notna().sum())
    return (
        f"Cas investigués : {format_metric_value(investigated)} "
        f"({format_metric_value(safe_pct(investigated, total_cases), decimals=1)}%)."
    )


def _build_tdr_summary_text(df_scope: pd.DataFrame) -> Optional[str]:
    """Construit un résumé standard des TDR réalisés et de la positivité."""
    result_col = None
    if COL_TDRR in df_scope.columns and df_scope[COL_TDRR].notna().any():
        result_col = COL_TDRR
    elif "Resultat_labo" in df_scope.columns and df_scope["Resultat_labo"].notna().any():
        result_col = "Resultat_labo"

    if COL_TDR in df_scope.columns:
        tdr_yes = _is_yes_series(df_scope[COL_TDR])
        n_tdr = int(tdr_yes.sum())
        if n_tdr <= 0:
            return "TDR réalisés : 0."
        if result_col is None:
            return f"TDR réalisés : {format_metric_value(n_tdr)}."
        label_prefix = "TDR réalisés"
    else:
        if result_col is None:
            return None
        tdr_yes = pd.Series(True, index=df_scope.index)
        n_tdr = int(df_scope[result_col].notna().sum())
        label_prefix = "Résultats labo renseignés"

    res_n = _tdr_result_norm(df_scope[result_col])
    valid_mask = tdr_yes & res_n.isin(TDR_POS_SET.union(TDR_NEG_SET))
    n_valid = int(valid_mask.sum())
    n_pos = int((tdr_yes & res_n.isin(TDR_POS_SET)).sum())
    if n_valid > 0:
        positivity = safe_pct(n_pos, n_valid)
        return (
            f"{label_prefix} : {format_metric_value(n_tdr)} "
            f"(positivité : {format_metric_value(positivity, decimals=1)}%; "
            f"{format_metric_value(n_pos)} positifs sur {format_metric_value(n_valid)} résultats interprétables)."
        )
    return f"{label_prefix} : {format_metric_value(n_tdr)} (positivité non calculable : résultats interprétables absents)."


def _build_comparison_sentence(
    current_df: pd.DataFrame,
    previous_df: Optional[pd.DataFrame],
    current_label: Optional[str] = None,
    previous_label: Optional[str] = None,
) -> Optional[str]:
    """Produit une phrase standard de tendance par rapport à une période de référence."""
    if previous_df is None or previous_df.empty:
        return None

    cur_cases, cur_deaths, cur_cfr = _surveillance_scope_kpis(current_df)
    prev_cases, prev_deaths, prev_cfr = _surveillance_scope_kpis(previous_df)
    if cur_cases <= 0:
        return None

    cur_label_txt = current_label or "la période courante"
    prev_label_txt = previous_label or "la période précédente"

    if prev_cases <= 0:
        return (
            f"{cur_label_txt} : {format_metric_value(cur_cases)} cas et {format_metric_value(cur_deaths)} décès "
            f"(létalité {format_metric_value(cur_cfr, decimals=1)}%). "
            f"Aucun cas n’avait été rapporté durant {prev_label_txt}."
        )

    delta = pct_change_safe(cur_cases, prev_cases)
    if pd.isna(delta):
        trend_text = "La variation par rapport à la période précédente n’est pas calculable."
    elif delta > 0:
        trend_text = f"Une hausse de {abs(delta):.1f}% est observée"
    elif delta < 0:
        trend_text = f"Une baisse de {abs(delta):.1f}% est observée"
    else:
        trend_text = "La situation est stable"

    return (
        f"{cur_label_txt} : {format_metric_value(cur_cases)} cas et {format_metric_value(cur_deaths)} décès "
        f"(létalité {format_metric_value(cur_cfr, decimals=1)}%). "
        f"{trend_text} par rapport à {prev_label_txt} "
        f"({format_metric_value(prev_cases)} cas, {format_metric_value(prev_deaths)} décès, "
        f"létalité {format_metric_value(prev_cfr, decimals=1)}%)."
    )


def _build_top_province_summary_text(df_scope: pd.DataFrame) -> Optional[str]:
    """Construit une phrase standard sur la concentration géographique provinciale."""
    total_cases, _, _ = _surveillance_scope_kpis(df_scope)
    top_prov = _build_surveillance_top_table(df_scope, [COL_PROV], top_n=5)
    if top_prov.empty or total_cases <= 0:
        return None

    top5_share = safe_pct(top_prov["Cas"].sum(), total_cases)
    province_bits = []
    for _, row in top_prov.iterrows():
        province_bits.append(
            f"{row[COL_PROV]} ({safe_pct(row['Cas'], total_cases):.1f}%)"
        )
    return (
        f"La majorité des cas ({top5_share:.1f}%) est concentrée dans les provinces suivantes : "
        + "; ".join(province_bits)
        + "."
    )


def _build_scope_overview_text(
    df_scope: pd.DataFrame,
    scope_kind: str,
    latest_week_df: Optional[pd.DataFrame] = None,
    latest_label: Optional[str] = None,
) -> Optional[str]:
    """Construit le paragraphe d'ouverture adapté à la fenêtre courante."""
    total_cases, total_deaths, cfr = _surveillance_scope_kpis(df_scope)
    if total_cases <= 0:
        return None

    n_prov = (
        _surveillance_clean_text_series(df_scope[COL_PROV]).nunique()
        if COL_PROV in df_scope.columns else 0
    )
    n_zs = (
        _surveillance_clean_text_series(df_scope[COL_ZS]).nunique()
        if COL_ZS in df_scope.columns else 0
    )
    dmin, dmax, _ = _surveillance_period_bounds(df_scope)
    date_span = None
    if dmin is not None and dmax is not None:
        date_span = f"du {_format_surveillance_date(dmin)} au {_format_surveillance_date(dmax)}"

    if scope_kind == "weekly":
        prefix = f"Durant {latest_label or 'la semaine la plus récente'}"
    elif scope_kind == "recent4":
        prefix = "Au cours des 4 dernières semaines"
    else:
        prefix = f"Sur le cumul {date_span}" if date_span else "Sur l’ensemble de la fenêtre sélectionnée"

    geo_text = ""
    if n_zs > 0 and n_prov > 0:
        geo_text = f"{format_metric_value(n_zs)} ZS dans {format_metric_value(n_prov)} provinces"
    elif n_zs > 0:
        geo_text = f"{format_metric_value(n_zs)} ZS"
    elif n_prov > 0:
        geo_text = f"{format_metric_value(n_prov)} provinces"

    base = (
        f"{prefix}, {format_metric_value(total_cases)} cas suspects et {format_metric_value(total_deaths)} décès "
        f"ont été enregistrés (létalité : {format_metric_value(cfr, decimals=1)}%)."
    )

    if geo_text:
        base += f" Les notifications proviennent de {geo_text}."

    if scope_kind == "cumulative" and latest_week_df is not None and not latest_week_df.empty:
        wk_cases, wk_deaths, wk_cfr = _surveillance_scope_kpis(latest_week_df)
        if latest_label:
            base += (
                f" La semaine la plus récente ({latest_label}) totalise "
                f"{format_metric_value(wk_cases)} cas, {format_metric_value(wk_deaths)} décès "
                f"et une létalité de {format_metric_value(wk_cfr, decimals=1)}%."
            )
    return base


def _build_province_interpretations(
    df_scope: pd.DataFrame,
    comparison_df: Optional[pd.DataFrame] = None,
    comparison_label: Optional[str] = None,
    top_n: int = 5,
) -> list[str]:
    """Construit un résumé narratif standard par province pour le top 5."""
    if df_scope is None or df_scope.empty or COL_PROV not in df_scope.columns:
        return []

    top_prov = _build_surveillance_top_table(df_scope, [COL_PROV], top_n=top_n)
    if top_prov.empty:
        return []

    summaries: list[str] = []
    for _, row in top_prov.iterrows():
        province_name = row.get(COL_PROV, MISSING_LABEL)
        prov_mask = df_scope[COL_PROV].fillna("").astype(str).str.strip() == str(province_name).strip()
        prov_df = df_scope.loc[prov_mask].copy()
        if prov_df.empty:
            continue

        prov_cases, prov_deaths, prov_cfr = _surveillance_scope_kpis(prov_df)
        sentence = (
            f"{province_name} : {format_metric_value(prov_cases)} cas suspects, "
            f"dont {format_metric_value(prov_deaths)} décès "
            f"(létalité : {format_metric_value(prov_cfr, decimals=1)}%)."
        )

        if comparison_df is not None and not comparison_df.empty and COL_PROV in comparison_df.columns:
            prev_mask = comparison_df[COL_PROV].fillna("").astype(str).str.strip() == str(province_name).strip()
            prev_df = comparison_df.loc[prev_mask].copy()
            if not prev_df.empty:
                prev_cases, prev_deaths, prev_cfr = _surveillance_scope_kpis(prev_df)
                if prev_cases > 0:
                    delta = pct_change_safe(prov_cases, prev_cases)
                    if not pd.isna(delta):
                        if delta > 0:
                            trend = f"Une hausse de {abs(delta):.1f}% est observée"
                        elif delta < 0:
                            trend = f"Une baisse de {abs(delta):.1f}% est observée"
                        else:
                            trend = "La situation est stable"
                        sentence += (
                            f" {trend} par rapport à {comparison_label or 'la période précédente'} "
                            f"({format_metric_value(prev_cases)} cas, {format_metric_value(prev_deaths)} décès, "
                            f"létalité {format_metric_value(prev_cfr, decimals=1)}%)."
                        )
                else:
                    sentence += f" Aucun cas n’avait été rapporté durant {comparison_label or 'la période précédente'}."

        if COL_ZS in prov_df.columns:
            zs_series = _surveillance_clean_text_series(prov_df[COL_ZS])
            n_zs_reporting = int(zs_series.nunique())
            if n_zs_reporting > 0:
                mean_cases = prov_cases / n_zs_reporting
                sentence += (
                    f" Une moyenne de {mean_cases:.1f} cas par ZS est observée parmi "
                    f"{format_metric_value(n_zs_reporting)} ZS ayant notifié des cas."
                )

                top_zs = _build_surveillance_top_table(prov_df, [COL_ZS], top_n=3)
                if not top_zs.empty:
                    top_zs_bits = [
                        f"{zs_row[COL_ZS]} ({format_metric_value(zs_row['Cas'])} cas)"
                        for _, zs_row in top_zs.iterrows()
                    ]
                    sentence += " Les ZS les plus touchées sont : " + ", ".join(top_zs_bits) + "."

        severity_text = _build_severity_summary_text(prov_df)
        if severity_text is not None:
            sentence += f" {severity_text}."

        summaries.append(sentence)

    return summaries


def _build_surveillance_summary_lines(
    df_scope: pd.DataFrame,
    scope_kind: str,
    current_label: Optional[str] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    comparison_label: Optional[str] = None,
    latest_week_df: Optional[pd.DataFrame] = None,
    latest_label: Optional[str] = None,
) -> list[str]:
    """Assemble le résumé automatique standard à afficher dans chaque situation."""
    lines: list[str] = []

    overview_text = _build_scope_overview_text(
        df_scope,
        scope_kind=scope_kind,
        latest_week_df=latest_week_df,
        latest_label=latest_label,
    )
    if overview_text:
        lines.append(overview_text)

    comparison_text = _build_comparison_sentence(
        current_df=df_scope,
        previous_df=comparison_df,
        current_label=current_label,
        previous_label=comparison_label,
    )
    if comparison_text:
        lines.append(comparison_text)

    top_prov_text = _build_top_province_summary_text(df_scope)
    if top_prov_text:
        lines.append(top_prov_text)

    severity_text = _build_severity_summary_text(df_scope)
    if severity_text:
        lines.append(severity_text + ".")

    investigation_text = _build_investigation_summary_text(df_scope)
    if investigation_text:
        lines.append(investigation_text)

    tdr_text = _build_tdr_summary_text(df_scope)
    if tdr_text:
        lines.append(tdr_text)

    return lines


def _render_surveillance_window(
    title: str,
    df_scope: pd.DataFrame,
    description: str,
    empty_message: str,
    narrative_context: Optional[dict[str, Any]] = None,
) -> None:
    """Affiche une fenêtre standardisée de surveillance."""
    st.markdown(f"### {title}")
    st.caption(description)

    if df_scope.empty:
        st.info(empty_message)
        render_reader_narrative(
            "Comment lire cette absence",
            "Aucune ligne ne répond aux critères actuels. Cela peut venir d'une absence réelle de notification, "
            "d'un filtre trop restrictif ou d'un retard de saisie. La conclusion doit rester prudente tant que la complétude n'est pas vérifiée.",
            tone="missing",
        )
        return

    total_cases = int(len(df_scope))
    total_deaths = int(
        pd.to_numeric(df_scope["is_death"], errors="coerce").fillna(0).sum()
    ) if "is_death" in df_scope.columns else 0
    cfr = (total_deaths / total_cases * 100.0) if total_cases > 0 else np.nan

    k1, k2, k3 = st.columns(3)
    k1.metric("Cas", format_metric_value(total_cases))
    k2.metric("Décès", format_metric_value(total_deaths))
    k3.metric("Létalité (%)", format_metric_value(cfr, decimals=2))

    narrative_context = narrative_context or {}
    summary_lines = _build_surveillance_summary_lines(
        df_scope=df_scope,
        scope_kind=narrative_context.get("scope_kind", "cumulative"),
        current_label=narrative_context.get("current_label"),
        comparison_df=narrative_context.get("comparison_df"),
        comparison_label=narrative_context.get("comparison_label"),
        latest_week_df=narrative_context.get("latest_week_df"),
        latest_label=narrative_context.get("latest_label"),
    )

    if summary_lines:
        st.markdown("**Résumé automatique**")
        st.markdown("\n".join([f"- {line}" for line in summary_lines]))

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Analyse des 5 provinces les plus touchées**")
        top_prov = _build_surveillance_top_table(df_scope, [COL_PROV], top_n=5)
        if top_prov.empty:
            st.info("L’analyse provinciale est indisponible : variable Province absente ou non renseignée.")
        else:
            st.caption(
                _describe_surveillance_top_table(
                    top_prov,
                    [COL_PROV],
                    total_cases,
                    "Aucune province exploitable dans cette fenêtre.",
                )
            )
            st.dataframe(top_prov, width="stretch", hide_index=True)

    with g2:
        st.markdown("**Analyse des 5 zones de santé les plus touchées**")
        zs_group_cols = [c for c in [COL_PROV, COL_ZS] if c in df_scope.columns]
        top_zs = _build_surveillance_top_table(df_scope, zs_group_cols, top_n=5)
        if top_zs.empty:
            st.info("L’analyse des zones de santé est indisponible : variable ZS absente ou non renseignée.")
        else:
            st.caption(
                _describe_surveillance_top_table(
                    top_zs,
                    zs_group_cols,
                    total_cases,
                    "Aucune zone de santé exploitable dans cette fenêtre.",
                )
            )
            st.dataframe(top_zs, width="stretch", hide_index=True)

    province_interpretations = _build_province_interpretations(
        df_scope,
        comparison_df=narrative_context.get("comparison_df"),
        comparison_label=narrative_context.get("comparison_label"),
        top_n=5,
    )
    if province_interpretations:
        with st.expander("Interprétation automatique par province (top 5)", expanded=False):
            for province_text in province_interpretations:
                st.markdown(f"- {province_text}")


def _build_sitrep_week_label(se: Any, annee: Any) -> str:
    """Formate un libellé standard de semaine pour le SITREP."""
    try:
        return f"SE{int(float(se)):02d}/{int(float(annee))}"
    except Exception:
        return f"SE{se}/{annee}"


def _format_sitrep_metric_delta(current: Any, previous: Any) -> Optional[str]:
    """Retourne un delta court et lisible pour les KPI du SITREP."""
    if previous is None or pd.isna(previous):
        return None
    try:
        cur_val = float(current)
        prev_val = float(previous)
    except Exception:
        return None

    if prev_val > 0:
        delta = pct_change_safe(cur_val, prev_val)
        if pd.isna(delta):
            return None
        return f"{delta:+.1f}%"
    if cur_val > 0:
        return f"+{int(cur_val)} vs 0"
    return "stable"


def _build_sitrep_alert_summary(alert_df: pd.DataFrame) -> Optional[str]:
    """Construit une phrase courte sur les signaux statistiques disponibles."""
    if alert_df is None or not isinstance(alert_df, pd.DataFrame) or alert_df.empty:
        return None

    signal_col = next(
        (col for col in ["signal", "Signal", "alerte", "Alerte"] if col in alert_df.columns),
        None,
    )
    if signal_col is not None:
        signals = _surveillance_clean_text_series(alert_df[signal_col])
        if not signals.empty:
            counts = signals.value_counts().head(3)
            details = [f"{label} ({int(count)})" for label, count in counts.items()]
            return "Signaux statistiques à vérifier sur la dernière semaine disponible : " + ", ".join(details) + "."

    return (
        f"{format_metric_value(len(alert_df))} signaux statistiques sont remontés sur la dernière "
        "semaine disponible et nécessitent une vérification."
    )


def _build_sitrep_critical_cfr_summary(payload: dict[str, Any]) -> Optional[str]:
    """Résume les unités géographiques à létalité élevée pour le SITREP."""
    provcrit = payload.get("prov_cfr_crit")
    zscrit = payload.get("zs_cfr_crit")
    parts: list[str] = []

    if isinstance(provcrit, pd.DataFrame) and not provcrit.empty and COL_PROV in provcrit.columns:
        focus = [
            f"{row[COL_PROV]} ({format_metric_value(row['CFR_%'], decimals=1)}%)"
            for _, row in provcrit.head(3).iterrows()
        ]
        if focus:
            parts.append("provinces " + ", ".join(focus))

    if isinstance(zscrit, pd.DataFrame) and not zscrit.empty and COL_ZS in zscrit.columns:
        focus = []
        for _, row in zscrit.head(5).iterrows():
            if COL_PROV in zscrit.columns:
                label = f"{row[COL_PROV]} / {row[COL_ZS]}"
            else:
                label = str(row[COL_ZS])
            focus.append(f"{label} ({format_metric_value(row['CFR_%'], decimals=1)}%)")
        if focus:
            parts.append("zones de santé " + ", ".join(focus))

    if not parts:
        return None

    return "La létalité doit être vérifiée en priorité dans les " + " ; ".join(parts) + "."


def _build_sitrep_summary_lines(payload: dict[str, Any]) -> list[str]:
    """Assemble les messages-clés du SITREP à partir du périmètre sélectionné."""
    lines: list[str] = []
    current_df = payload.get("selected_df", pd.DataFrame())
    cumulative_df = payload.get("cumulative_df", pd.DataFrame())
    previous_df = payload.get("previous_df", pd.DataFrame())
    current_label = payload.get("selected_week_label")
    previous_label = payload.get("previous_week_label")

    if isinstance(current_df, pd.DataFrame) and not current_df.empty:
        overview_text = _build_scope_overview_text(
            current_df,
            scope_kind="weekly",
            latest_week_df=current_df,
            latest_label=current_label,
        )
        if overview_text:
            lines.append(overview_text)

        comparison_text = _build_comparison_sentence(
            current_df=current_df,
            previous_df=previous_df if isinstance(previous_df, pd.DataFrame) and not previous_df.empty else None,
            current_label=current_label,
            previous_label=previous_label,
        )
        if comparison_text:
            lines.append(comparison_text)

        top_prov_text = _build_top_province_summary_text(current_df)
        if top_prov_text:
            lines.append(top_prov_text)

        investigation_text = _build_investigation_summary_text(current_df)
        if investigation_text:
            lines.append(investigation_text)

        tdr_text = _build_tdr_summary_text(current_df)
        if tdr_text:
            lines.append(tdr_text)
    else:
        lines.append(
            f"Aucun cas n’est rapporté pour {current_label or 'la semaine sélectionnée'} dans le périmètre filtré."
        )

    if isinstance(cumulative_df, pd.DataFrame) and not cumulative_df.empty:
        cumulative_text = _build_scope_overview_text(
            cumulative_df,
            scope_kind="cumulative",
            latest_week_df=current_df if isinstance(current_df, pd.DataFrame) and not current_df.empty else None,
            latest_label=current_label,
        )
        if cumulative_text:
            lines.append(cumulative_text)

    critical_text = _build_sitrep_critical_cfr_summary(payload)
    if critical_text:
        lines.append(critical_text)

    alert_text = _build_sitrep_alert_summary(payload.get("alertes_last"))
    if alert_text:
        lines.append(alert_text)

    return lines


def _build_sitrep_action_lines(payload: dict[str, Any]) -> list[str]:
    """Construit des priorités opérationnelles courtes pour le SITREP."""
    current_df = payload.get("selected_df", pd.DataFrame())
    current_label = payload.get("selected_week_label", "la semaine sélectionnée")
    actions: list[str] = []

    if not isinstance(current_df, pd.DataFrame) or current_df.empty:
        return [
            f"Vérifier si l’absence de cas en {current_label} reflète réellement la situation ou un retard de notification."
        ]

    total_cases = int(len(current_df))
    top_prov = payload.get("top_prov_focus")
    if isinstance(top_prov, pd.DataFrame) and not top_prov.empty and total_cases > 0:
        leader = top_prov.iloc[0]
        leader_share = safe_pct(leader["Cas"], total_cases)
        actions.append(
            f"Cibler en priorité {leader[COL_PROV]}, qui concentre {leader_share:.1f}% des cas de {current_label}."
        )

    critical_text = _build_sitrep_critical_cfr_summary(payload)
    if critical_text:
        actions.append(critical_text)

    alert_text = _build_sitrep_alert_summary(payload.get("alertes_last"))
    if alert_text:
        actions.append(alert_text)

    if COL_TDR in current_df.columns:
        n_tdr = int(_is_yes_series(current_df[COL_TDR]).sum())
        if n_tdr <= 0:
            actions.append(
                "Aucun TDR n’est documenté sur la semaine sélectionnée : renforcer la confirmation biologique si elle est attendue."
            )

    if DATE_INV in current_df.columns:
        investigated = int(pd.to_datetime(current_df[DATE_INV], errors="coerce").notna().sum())
        if investigated < total_cases:
            actions.append(
                f"Compléter l’investigation de {format_metric_value(total_cases - investigated)} cas pour consolider la lecture opérationnelle."
            )

    return actions[:4]


# =========================
# NAVIGATION COMPACTE
# - Surveillance & promptitude = anciens onglets 1 + 2 + 3
# - Profil descriptif = anciens onglets 4 + 4b
# - Données, complétude & qualité = anciens onglets 5 + 6 + 7
# - Cartographie, SITREP, IDSR et IREP restent dédiés
# =========================

def render_reader_narrative(title: str, text: Any, *, tone: str = "standard") -> None:
    """Affiche un narratif court ou structuré lisible par un public mixte."""
    accent = {
        "standard": "#1f6feb",
        "missing": "#b45309",
        "decision": "#047857",
    }.get(tone, "#1f6feb")

    if isinstance(text, dict):
        section_blocks: list[str] = []
        for label, key in [
            ("Constat", "constat"),
            ("Interprétation", "interpretation"),
            ("Action recommandée", "action"),
        ]:
            content = str(text.get(key, "")).strip()
            if not content:
                continue
            safe_content = html.escape(content, quote=False).replace("\n", "<br>")
            section_blocks.append(
                f'<div style="margin-top: 0.35rem;"><strong>{label} :</strong> {safe_content}</div>'
            )

        if not section_blocks:
            return

        safe_title = html.escape(str(title), quote=False)
        st.markdown(
            f"""
            <div class="cousp-detail-empty" style="border-left: 4px solid {accent};">
                <strong>{safe_title}</strong>
                {''.join(section_blocks)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="cousp-detail-empty" style="border-left: 4px solid {accent};">
            <strong>{title}</strong><br>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


TAB_NARRATIVES = {
    "surveillance": (
        "Lecture de l'onglet",
        "Cet onglet suit l'évolution des notifications dans le temps. "
        "Pour un lecteur non spécialiste, une hausse signale surtout un changement à vérifier. "
        "Pour l'équipe épidémiologique, elle doit être interprétée avec la complétude, les retards de notification "
        "et la distribution géographique avant de conclure à une flambée."
    ),
    "promptitude": (
        "Lecture des délais",
        "Les délais décrivent la rapidité du parcours entre début des signes, notification, prise en charge "
        "et confirmation. Un délai long peut traduire un accès tardif aux soins, un retard de saisie ou une donnée incomplète."
    ),
    "profil": (
        "Lecture du profil des cas",
        "Les tableaux décrivent qui est touché, où les cas sont rapportés et quelles informations de laboratoire sont disponibles. "
        "Ces résultats décrivent les notifications reçues ; ils ne remplacent pas une enquête de terrain ni une estimation de risque populationnel."
    ),
    "qualite": (
        "Lecture qualité et décision",
        "Cet onglet sert à distinguer les signaux de santé publique des problèmes de données. "
        "Une alerte ou un score élevé indique une priorité de vérification, pas une confirmation automatique d'épidémie."
    ),
    "export": (
        "Lecture et partage",
        "Les exports reprennent le périmètre filtré à l'écran. Ils sont adaptés à la revue technique, au partage opérationnel "
        "et à la traçabilité des analyses."
    ),
    "sitrep": (
        "Lecture du SITREP",
        "Le SITREP résume la situation pour une décision rapide. Il met en avant les messages clés, les zones à suivre "
        "et les limites de lecture, sans remplacer les onglets détaillés."
    ),
    "idsr": (
        "Lecture IDSR",
        "L'IDSR agrégé donne une lecture hebdomadaire par maladie et par zone. "
        "Les volumes doivent être interprétés avec les règles de rapportage, la complétude et les éventuelles corrections tardives."
    ),
    "irep": (
        "Lecture de l'indice",
        "L'IREP classe les provinces selon plusieurs dimensions de risque. "
        "Il aide à prioriser l'attention mais ne doit pas être lu comme un diagnostic isolé."
    ),
    "cartographie": (
        "Lecture cartographique",
        "La carte situe les notifications dans l'espace. Elle dépend de la qualité des noms géographiques et des fichiers GeoJSON utilisés."
    ),
}


ABSENCE_NARRATIVES = {
    "idsr_line_list": (
        "Analyse non applicable dans ce mode",
        "Vous êtes en mode IDSR agrégé. Les analyses de liste linéaire reposent sur des cas individuels ; "
        "utilisez l'onglet IDSR pour lire les données hebdomadaires agrégées."
    ),
    "week": (
        "Lecture temporelle indisponible",
        "Aucune variable de semaine exploitable n'a été trouvée dans le périmètre filtré. "
        "Vérifiez les champs de date, d'année ou de semaine épidémiologique avant d'interpréter l'absence de courbe."
    ),
    "delays": (
        "Délais non calculables",
        "Les dates nécessaires ne sont pas disponibles ou ne sont pas suffisamment exploitables. "
        "L'absence d'indicateur ne signifie pas que les délais sont bons ou mauvais ; elle signale d'abord une limite de documentation."
    ),
    "geo": (
        "Lecture géographique limitée",
        "Aucune variable géographique exploitable n'est disponible dans le périmètre filtré. "
        "Vérifiez les champs Province, Zone de santé ou Aire de santé, ainsi que les filtres actifs."
    ),
    "alerts": (
        "Alertes non calculables",
        "L'historique disponible est insuffisant ou les variables nécessaires sont absentes. "
        "Une absence d'alerte dans ce contexte ne doit pas être interprétée comme une absence de risque."
    ),
    "risk": (
        "Priorisation non calculable",
        "Le score de risque a besoin d'au moins une dimension géographique et de quelques indicateurs exploitables. "
        "Lorsque ces éléments manquent, il faut revenir aux tableaux descriptifs et à la qualité des données."
    ),
    "profile": (
        "Profil incomplet",
        "La variable attendue est absente ou vide. Cette limite doit être mentionnée dans toute restitution, "
        "car elle peut modifier la lecture des groupes les plus représentés."
    ),
    "quality": (
        "Contrôle limité",
        "Les champs nécessaires au contrôle ne sont pas disponibles dans le périmètre actuel. "
        "Cela oriente d'abord vers une revue de structure du fichier plutôt que vers une conclusion sanitaire."
    ),
}


def render_tab_narrative(key: str) -> None:
    title, text = TAB_NARRATIVES[key]
    render_reader_narrative(title, text)


def render_absence_narrative(key: str) -> None:
    title, text = ABSENCE_NARRATIVES[key]
    render_reader_narrative(title, text, tone="missing")
