"""Render the surveillance analytics tab."""

from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


def _surv_clean_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Nettoie les colonnes numériques avant calculs/graphes Plotly."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[~np.isfinite(out[col]), col] = np.nan
    return out


def _surv_prepare_delay_scope(df: pd.DataFrame, delay_cols: list[str]) -> pd.DataFrame:
    """Prépare les délais pour les indicateurs et graphiques de promptitude."""
    out = _surv_clean_numeric_columns(df, delay_cols)
    for col in delay_cols:
        if col in out.columns:
            # Les délais négatifs sont considérés comme incohérents et exclus des calculs.
            out.loc[out[col] < 0, col] = np.nan
    return out


def _surv_plotly_frame(df: pd.DataFrame, numeric_cols: list[str] | None = None) -> pd.DataFrame:
    """Retourne un DataFrame compatible Plotly JSON (pd.NA/NaT -> None)."""
    out = df.copy()
    for col in numeric_cols or []:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[~np.isfinite(out[col]), col] = np.nan
    return out.astype(object).where(pd.notna(out), None)


def _surv_metric_pct(value: Any) -> str:
    """Formate un pourcentage de manière sûre pour st.metric."""
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.1f}"
    except Exception:
        return "-"


def _surv_threshold_metric(series: Any, threshold: float) -> tuple[float, int]:
    """Calcule un indicateur <= seuil sans échouer si la série est absente."""
    if series is None:
        return np.nan, 0
    try:
        return pct_under_threshold(series, threshold)
    except Exception:
        return np.nan, 0


def _surv_join_list(values: list[str], max_items: int = 10) -> str:
    """Assemble une liste courte et lisible pour les tableaux linéaires."""
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    if not cleaned:
        return "-"
    if len(cleaned) <= int(max_items):
        return ", ".join(cleaned)
    return ", ".join(cleaned[: int(max_items)]) + f", +{len(cleaned) - int(max_items)} autre(s)"


def _surv_alpha_sort_key(value: object) -> str:
    """Retourne une clé de tri alphabétique robuste pour les libellés géographiques."""
    return str(value).strip().casefold()


def _build_surveillance_completeness_matrices(
    df: pd.DataFrame,
    province_col: str,
    zs_col: str,
    *,
    expected_mode: str = "union",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Construit les matrices de complétude surveillance province × semaine."""
    if df is None or df.empty or province_col not in df.columns or zs_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype="float64"), pd.DataFrame()

    work, _reference = _prepare_surveillance_period_scope(df)
    if work.empty or "_surv_label" not in work.columns or "_surv_order" not in work.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype="float64"), pd.DataFrame()

    work = work.copy()
    work["_surv_province"] = work[province_col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    work["_surv_zs"] = work[zs_col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    work = work.dropna(subset=["_surv_province", "_surv_zs", "_surv_label", "_surv_order"])
    work = work[
        (work["_surv_province"] != "")
        & (work["_surv_zs"] != "")
        & (work["_surv_label"].astype(str).str.strip() != "")
    ]
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype="float64"), pd.DataFrame()

    unique_reporting = work[["_surv_province", "_surv_zs", "_surv_label", "_surv_order"]].drop_duplicates()
    week_ref = (
        unique_reporting[["_surv_label", "_surv_order"]]
        .drop_duplicates()
        .sort_values("_surv_order")
    )
    week_labels = week_ref["_surv_label"].astype(str).tolist()

    counts = (
        unique_reporting
        .groupby(["_surv_province", "_surv_label"], as_index=False)
        .agg(Nombre_ZS=("_surv_zs", "nunique"))
    )
    count_pivot = (
        counts
        .pivot_table(
            index="_surv_province",
            columns="_surv_label",
            values="Nombre_ZS",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .astype(int)
    )
    province_order = sorted(count_pivot.index.astype(str).tolist(), key=_surv_alpha_sort_key)
    count_pivot = count_pivot.reindex(index=province_order, columns=week_labels, fill_value=0)

    if expected_mode == "max_week":
        expected = count_pivot.max(axis=1).replace(0, np.nan)
    else:
        expected = (
            unique_reporting
            .groupby("_surv_province")["_surv_zs"]
            .nunique()
            .reindex(count_pivot.index)
            .replace(0, np.nan)
        )

    rel_pivot = count_pivot.div(expected, axis=0).clip(lower=0, upper=1).fillna(0)
    summary = pd.DataFrame({
        "Province": count_pivot.index.astype(str),
        "ZS attendues": expected.fillna(0).astype(int).values,
        "Semaines analysées": int(len(count_pivot.columns)),
        "Moyenne ZS rapportantes": count_pivot.mean(axis=1).round(1).values,
        "Complétude moyenne (%)": (rel_pivot.mean(axis=1) * 100).round(1).values,
        "Minimum ZS rapportantes": count_pivot.min(axis=1).astype(int).values,
        "Semaine la plus faible": count_pivot.idxmin(axis=1).astype(str).values,
        "Dernière semaine": str(count_pivot.columns[-1]) if len(count_pivot.columns) else "",
        "ZS dernière semaine": count_pivot.iloc[:, -1].astype(int).values if len(count_pivot.columns) else 0,
        "Complétude dernière semaine (%)": (rel_pivot.iloc[:, -1] * 100).round(1).values if len(rel_pivot.columns) else 0,
    })
    return count_pivot, rel_pivot, expected, summary


def _make_surveillance_completeness_heatmap(count_pivot: pd.DataFrame, rel_pivot: pd.DataFrame) -> object:
    """Construit la figure Plotly du tableau de complétude surveillance."""
    if count_pivot is None or count_pivot.empty or rel_pivot is None or rel_pivot.empty:
        return None

    y_labels = [str(x).upper() for x in count_pivot.index.tolist()]
    x_labels = [str(x) for x in count_pivot.columns.tolist()]
    z_values = rel_pivot.to_numpy(dtype=float)
    text_values = count_pivot.astype(int).astype(str).to_numpy()

    hover = []
    for prov_idx, prov in enumerate(y_labels):
        row = []
        for week_idx, week in enumerate(x_labels):
            row.append(
                f"Province: {prov}<br>"
                f"Semaine: {week}<br>"
                f"ZS rapportantes: {int(count_pivot.iloc[prov_idx, week_idx])}<br>"
                f"Complétude relative: {float(rel_pivot.iloc[prov_idx, week_idx]) * 100:.1f}%"
            )
        hover.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "black"},
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            zmin=0,
            zmax=1,
            colorscale=[
                [0.00, "#ff0000"],
                [0.25, "#ff9900"],
                [0.50, "#ffff00"],
                [0.75, "#99ff00"],
                [1.00, "#00ee00"],
            ],
            colorbar={"title": "Complétude<br>(relative)", "tickformat": ".0%"},
        )
    )
    height = min(max(480, 26 * len(y_labels) + 180), 1050)
    fig.update_layout(
        title="Tableau de complétude de la surveillance (nombre de ZS par province)",
        xaxis_title="Semaines épidémiologiques de notification",
        yaxis_title="Divisions provinciales de la santé (DPS)",
        height=height,
        margin=dict(l=120, r=80, t=80, b=70),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(type="category", tickangle=0, showgrid=True, gridcolor="rgba(0,0,0,0.10)")
    fig.update_yaxes(
        type="category",
        autorange="reversed",
        categoryorder="array",
        categoryarray=y_labels,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)",
    )
    return fig


def _render_surveillance_completeness_section(
    df_scope: pd.DataFrame,
    province_col: str,
    zs_col: str,
) -> None:
    """Affiche un tableau de complétude surveillance sous forme de listes linéaires."""
    with st.expander("Tableau de complétude de la surveillance", expanded=False):
        st.markdown("### Tableau de complétude de la surveillance")
        st.caption(
            "Lecture : les listes ci-dessous résument, par province et par semaine, le nombre de zones de santé "
            "ayant rapporté au moins un cas dans la fenêtre de surveillance active."
        )

        if df_scope is None or df_scope.empty:
            st.info("Aucune donnée disponible pour calculer la complétude de la surveillance.")
            return

        missing = [c for c in [province_col, zs_col] if c not in df_scope.columns]
        if missing:
            st.info("Complétude surveillance indisponible : colonne(s) manquante(s) : " + ", ".join(missing) + ".")
            return

        c1, c2, c3 = st.columns([1.25, 1, 1])
        with c1:
            expected_mode_label = st.selectbox(
                "Référence de complétude",
                options=["Total ZS observées sur la période", "Maximum hebdomadaire observé"],
                index=0,
                key="surv_completeness_expected_mode",
                help=(
                    "Total ZS observées : dénominateur stable par province sur toute la période filtrée. "
                    "Maximum hebdomadaire : dénominateur basé sur la meilleure semaine observée."
                ),
            )
        with c2:
            completeness_threshold = st.slider(
                "Seuil d'alerte (%)",
                min_value=0,
                max_value=100,
                value=80,
                step=5,
                key="surv_completeness_threshold",
            )
        with c3:
            show_completion_table = st.checkbox(
                "Afficher le tableau source",
                value=False,
                key="surv_completeness_show_table",
            )

        expected_mode = "max_week" if expected_mode_label == "Maximum hebdomadaire observé" else "union"
        count_pivot, rel_pivot, expected, summary = _build_surveillance_completeness_matrices(
            df_scope,
            province_col,
            zs_col,
            expected_mode=expected_mode,
        )
        if count_pivot.empty:
            st.info("Aucune combinaison Province / Zone de santé / Semaine exploitable pour produire la complétude.")
            return

        threshold_ratio = float(completeness_threshold) / 100.0
        latest_col = count_pivot.columns[-1]
        total_expected = float(expected.fillna(0).sum())
        latest_count = float(count_pivot[latest_col].sum())
        latest_pct = (latest_count / total_expected * 100.0) if total_expected > 0 else np.nan
        mean_pct = float(rel_pivot.mean().mean() * 100.0) if not rel_pivot.empty else np.nan
        alert_cells = int((rel_pivot < threshold_ratio).sum().sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Provinces", f"{count_pivot.shape[0]:,}")
        k2.metric("Semaines", f"{count_pivot.shape[1]:,}")
        k3.metric("Complétude moyenne", "NA" if pd.isna(mean_pct) else f"{mean_pct:.1f}%")
        k4.metric(f"Dernière semaine ({latest_col})", "NA" if pd.isna(latest_pct) else f"{latest_pct:.1f}%")

        if alert_cells > 0:
            st.caption(
                f"Cellules sous le seuil de {int(completeness_threshold)}% : {alert_cells:,}. "
                "Ces provinces et semaines méritent une vérification avec les équipes concernées."
            )

        fig = _make_surveillance_completeness_heatmap(count_pivot, rel_pivot)
        if fig is not None:
            try:
                st_plot(fig, key="surv_completeness_heatmap")
            except Exception:
                st.plotly_chart(fig, width="stretch", key="surv_completeness_heatmap")

        province_summary = summary.sort_values(
            ["Province"],
            ascending=[True],
            key=lambda col: col.map(_surv_alpha_sort_key) if col.name == "Province" else col,
        ).copy()

        week_rows = []
        zero_by_week_rows = []
        for week in count_pivot.columns.tolist():
            week_total = float(count_pivot[week].sum())
            week_pct = (week_total / total_expected * 100.0) if total_expected > 0 else np.nan
            week_rel = rel_pivot[week]
            provinces_below = [str(p) for p in week_rel.index[week_rel < threshold_ratio].tolist()]
            provinces_zero = [str(p) for p in count_pivot.index[count_pivot[week].eq(0)].tolist()]
            week_rows.append({
                "Semaine": str(week),
                "ZS rapportantes": int(week_total),
                "Complétude globale (%)": round(week_pct, 1) if not pd.isna(week_pct) else np.nan,
                f"Provinces sous {int(completeness_threshold)}%": int(len(provinces_below)),
                "Provinces sans partage": int(len(provinces_zero)),
                "Liste provinces sous seuil": _surv_join_list(provinces_below, max_items=6),
            })
            if provinces_zero:
                zero_by_week_rows.append({
                    "Semaine": str(week),
                    "Nombre de provinces sans partage": int(len(provinces_zero)),
                    "Provinces sans partage": _surv_join_list(provinces_zero, max_items=10),
                })

        threshold_by_province_rows = []
        zero_by_province_rows = []
        zero_mask = count_pivot.eq(0)
        for province in count_pivot.index.tolist():
            weeks_below = [str(w) for w in rel_pivot.columns[rel_pivot.loc[province] < threshold_ratio].tolist()]
            weeks_zero = [str(w) for w in count_pivot.columns[zero_mask.loc[province]].tolist()]
            if weeks_below:
                threshold_by_province_rows.append({
                    "Province": str(province),
                    "Nombre de semaines sous seuil": int(len(weeks_below)),
                    "Semaines sous seuil": _surv_join_list(weeks_below, max_items=12),
                })
            if weeks_zero:
                zero_by_province_rows.append({
                    "Province": str(province),
                    "Nombre de semaines sans partage": int(len(weeks_zero)),
                    "Semaines sans partage": _surv_join_list(weeks_zero, max_items=12),
                })

        week_summary = pd.DataFrame(week_rows).sort_values(
            ["Complétude globale (%)", "Semaine"],
            ascending=[True, True],
            na_position="last",
        )
        threshold_by_province = pd.DataFrame(threshold_by_province_rows).sort_values(
            ["Nombre de semaines sous seuil", "Province"],
            ascending=[False, True],
            key=lambda col: col.map(_surv_alpha_sort_key) if col.name == "Province" else col,
        ) if threshold_by_province_rows else pd.DataFrame()
        zero_by_province = pd.DataFrame(zero_by_province_rows).sort_values(
            ["Nombre de semaines sans partage", "Province"],
            ascending=[False, True],
            key=lambda col: col.map(_surv_alpha_sort_key) if col.name == "Province" else col,
        ) if zero_by_province_rows else pd.DataFrame()
        zero_by_week = pd.DataFrame(zero_by_week_rows).sort_values(
            ["Nombre de provinces sans partage", "Semaine"],
            ascending=[False, True],
        ) if zero_by_week_rows else pd.DataFrame()

        p1, p2 = st.columns([1.15, 1.0])
        with p1:
            st.markdown("**Liste linéaire par province**")
            st.dataframe(province_summary, width="stretch", height=320, hide_index=True)
        with p2:
            st.markdown("**Liste linéaire par semaine**")
            st.dataframe(week_summary, width="stretch", height=320, hide_index=True)

        p3, p4 = st.columns([1.0, 1.0])
        with p3:
            st.markdown(f"**Provinces avec semaines sous {int(completeness_threshold)}%**")
            if threshold_by_province.empty:
                st.success("Aucune province n'est passée sous le seuil choisi dans la fenêtre analysée.")
            else:
                st.dataframe(threshold_by_province, width="stretch", height=260, hide_index=True)
        with p4:
            st.markdown("**Semaines avec provinces sans partage**")
            if zero_by_week.empty:
                st.success("Aucune semaine sans partage total par province n'a été détectée.")
            else:
                st.dataframe(zero_by_week, width="stretch", height=260, hide_index=True)

        if not zero_by_province.empty:
            st.caption(
                "Note : les semaines sans partage correspondent aux provinces présentes dans le fichier mais sans aucune "
                "zone de santé rapportante sur la semaine concernée."
            )

        if show_completion_table:
            st.markdown("**Tableau source : nombre de ZS rapportantes par province et semaine**")
            count_view = count_pivot.reset_index().rename(columns={"_surv_province": "Province"})
            st.dataframe(count_view, width="stretch", height=360, hide_index=True)


def render_surveillance_tab(ctx: dict) -> None:
    """Render the surveillance analytics tab."""
    globals().update(ctx)
    if IDSR_MODE:
        render_absence_narrative("idsr_line_list")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : suivre la situation selon trois niveaux de lecture complémentaires pour distinguer l’hebdomadaire, la tendance récente et le cumul.

            **📖 Logique de lecture**
            - La **situation hebdomadaire** reflète la semaine la plus récente visible dans la plage filtrée.
            - La **situation des dernières semaines** permet de lire rapidement la tendance récente sur une fenêtre glissante paramétrable.
            - La **situation cumulée** consolide l’ensemble de la fenêtre active pour orienter la réponse.

            **⚠️ Point d’attention**
            - Les indicateurs dépendent directement des filtres actifs de semaines, de géographie et de classification.
            """,
            expanded=False
        )

        render_section_title(1, "Surveillance épidémiologique")
        render_tab_narrative("surveillance")
        st.caption(
            "Cette organisation permet une lecture progressive de la situation à partir de la plage de semaines active dans la barre latérale."
        )
        cfg1, cfg2, cfg3 = st.columns([0.9, 0.9, 1.2])
        with cfg1:
            top_province_n = st.number_input(
                "Nombre de provinces à afficher",
                min_value=1,
                max_value=30,
                value=5,
                step=1,
                key="surveillance_top_province_n",
            )
        with cfg2:
            top_zs_n = st.number_input(
                "Nombre de zones de santé à afficher",
                min_value=1,
                max_value=30,
                value=5,
                step=1,
                key="surveillance_top_zs_n",
            )
        with cfg3:
            recent_window_weeks = st.number_input(
                "Nombre de dernières semaines à analyser",
                min_value=2,
                max_value=26,
                value=4,
                step=1,
                key="surveillance_recent_window_weeks",
                help="La situation récente reste réglée par défaut sur 4 semaines, mais vous pouvez choisir une autre fenêtre glissante.",
            )

        df_surv_scope, surv_reference = _prepare_surveillance_period_scope(df_f)

        if not surv_reference.empty:
            if COL_PROV in df_surv_scope.columns and COL_ZS in df_surv_scope.columns:
                _render_surveillance_completeness_section(df_surv_scope, COL_PROV, COL_ZS)
                st.divider()

            latest_order = surv_reference["order"].iloc[-1]
            latest_label = str(surv_reference["label"].iloc[-1])
            recent_window_weeks = int(min(max(int(recent_window_weeks), 2), len(surv_reference)))
            recent_reference = surv_reference.tail(int(recent_window_weeks)).copy()
            recent_orders = recent_reference["order"].tolist()
            recent_labels = recent_reference["label"].astype(str).tolist()
            first_label = str(surv_reference["label"].iloc[0])
            previous_week_reference = surv_reference.tail(2).head(1).copy() if len(surv_reference) >= 2 else pd.DataFrame()
            previous_week_df = pd.DataFrame()
            previous_week_label = None
            if not previous_week_reference.empty:
                previous_week_label = str(previous_week_reference["label"].iloc[0])
                previous_week_df = df_surv_scope[
                    df_surv_scope["_surv_order"] == previous_week_reference["order"].iloc[0]
                ].copy()

            prev_recent_reference = (
                surv_reference.iloc[
                    max(len(surv_reference) - (2 * int(recent_window_weeks)), 0): max(len(surv_reference) - int(recent_window_weeks), 0)
                ].copy()
                if len(surv_reference) > int(recent_window_weeks) else pd.DataFrame()
            )
            prev_recent_df = pd.DataFrame()
            prev_recent_label = None
            if not prev_recent_reference.empty:
                prev_recent_orders = prev_recent_reference["order"].tolist()
                prev_recent_df = df_surv_scope[df_surv_scope["_surv_order"].isin(prev_recent_orders)].copy()
                prev_recent_labels = prev_recent_reference["label"].astype(str).tolist()
                if prev_recent_labels:
                    prev_recent_label = ", ".join(prev_recent_labels)

            df_latest_week = df_surv_scope[df_surv_scope["_surv_order"] == latest_order].copy()
            df_recent_weeks = df_surv_scope[df_surv_scope["_surv_order"].isin(recent_orders)].copy()

            _render_surveillance_window(
                "1. Situation hebdomadaire",
                df_latest_week,
                f"Semaine la plus récente dans la fenêtre filtrée : {latest_label}.",
                "Aucune donnée n’est disponible pour la semaine hebdomadaire sélectionnée.",
                narrative_context={
                    "scope_kind": "weekly",
                    "current_label": latest_label,
                    "comparison_df": previous_week_df if not previous_week_df.empty else None,
                    "comparison_label": previous_week_label,
                    "latest_week_df": df_latest_week,
                    "latest_label": latest_label,
                },
                top_province_n=int(top_province_n),
                top_zs_n=int(top_zs_n),
            )

            st.divider()

            _render_surveillance_window(
                f"2. Situation des {int(recent_window_weeks)} dernières semaines",
                df_recent_weeks,
                f"Lecture glissante sur les {len(recent_labels)} semaines les plus récentes de la sélection : {', '.join(recent_labels)}.",
                f"Aucune donnée n’est disponible pour construire la tendance des {int(recent_window_weeks)} dernières semaines.",
                narrative_context={
                    "scope_kind": "recent_window",
                    "current_label": f"Les {int(recent_window_weeks)} dernières semaines",
                    "recent_window_weeks": int(recent_window_weeks),
                    "comparison_df": prev_recent_df if not prev_recent_df.empty else None,
                    "comparison_label": f"les {int(recent_window_weeks)} semaines précédentes" if prev_recent_label else None,
                    "latest_week_df": df_latest_week,
                    "latest_label": latest_label,
                },
                top_province_n=int(top_province_n),
                top_zs_n=int(top_zs_n),
            )

            st.divider()

            _render_surveillance_window(
                "3. Situation cumulée",
                df_surv_scope,
                f"Cumul de toute la fenêtre active : {first_label} à {latest_label}.",
                "Aucune donnée n’est disponible pour la situation cumulée.",
                narrative_context={
                    "scope_kind": "cumulative",
                    "current_label": f"{first_label} à {latest_label}",
                    "comparison_df": None,
                    "comparison_label": None,
                    "latest_week_df": df_latest_week,
                    "latest_label": latest_label,
                },
                top_province_n=int(top_province_n),
                top_zs_n=int(top_zs_n),
            )


            if (
                COL_PROV in df_surv_scope.columns
                and df_surv_scope[COL_PROV].notna().any()
                and "_surv_label" in df_surv_scope.columns
                and df_surv_scope["_surv_label"].notna().any()
            ):
                st.divider()
                st.markdown("### Courbe épidémiologique des cas par province")
                prov_totals = df_surv_scope[[COL_PROV]].dropna().copy()
                prov_totals["_prov"] = prov_totals[COL_PROV].astype(str).str.strip()
                prov_totals = prov_totals[prov_totals["_prov"] != ""]
                prov_options = prov_totals["_prov"].value_counts().index.tolist()
                default_provs = prov_options if len(prov_options) <= 10 else prov_options[:10]
                selected_curve_provs = st.multiselect(
                    "Provinces à afficher",
                    options=prov_options,
                    default=default_provs,
                    key="surveillance_multi_curve_provinces",
                    help="Tu peux aussi cliquer sur la légende du graphique pour masquer ou afficher une province.",
                )
                if selected_curve_provs:
                    fig_multi_prov = build_weekly_multiline_by_group(
                        df=df_surv_scope,
                        week_col="_surv_label",
                        group_col=COL_PROV,
                        selected_groups=selected_curve_provs,
                        titre=" ",
                        x_titre="Semaine épidémiologique",
                        y_titre="Nombre de cas",
                        rotation=45,
                        pas_x=int(pas_x),
                        annot=annot_vals,
                        taille_fig=(1500, 700),
                    )
                    if fig_multi_prov is not None:
                        fig_multi_prov.update_layout(
                            legend=dict(
                                title=dict(text=COL_PROV),
                                orientation="v",
                                yanchor="top",
                                y=1,
                                xanchor="left",
                                x=1.02,
                                itemclick="toggle",
                                itemdoubleclick="toggleothers",
                            ),
                            margin=dict(r=220),
                        )
                        fig_multi_prov = sanitize_plotly_figure_for_streamlit(fig_multi_prov)
                        st.plotly_chart(
                            fig_multi_prov,
                            width="stretch",
                            key="surveillance_multi_curve_province",
                        )
                        st.caption(
                            "Astuce : clique sur une province dans la légende pour masquer ou afficher sa courbe. "
                            "Double-clique pour isoler une province."
                        )
                    else:
                        st.info("Aucune courbe exploitable n'a pu être construite pour les provinces sélectionnées.")
                else:
                    st.info("Sélectionne au moins une province pour afficher la courbe épidémiologique multi-provinces.")

            if "_surv_label" in df_surv_scope.columns and df_surv_scope["_surv_label"].notna().any():
                st.divider()
                st.markdown("### Alertes automatiques et clusters récents")
                alert_group_options = [
                    c for c in [COL_PROV, COL_ZS, COL_AS]
                    if c in df_surv_scope.columns and df_surv_scope[c].notna().any()
                ]
                if alert_group_options:
                    a1, a2, a3, a4 = st.columns([1.1, 0.9, 0.9, 0.9])
                    with a1:
                        alert_group_col = st.selectbox(
                            "Niveau d'alerte",
                            options=alert_group_options,
                            key="surv_alert_group_col",
                        )
                    with a2:
                        alert_min_cases = st.number_input(
                            "Cas min",
                            min_value=1,
                            max_value=500,
                            value=10,
                            step=1,
                            key="surv_alert_min_cases",
                        )
                    with a3:
                        alert_ratio = st.number_input(
                            "Ratio alerte",
                            min_value=1.0,
                            max_value=10.0,
                            value=1.5,
                            step=0.1,
                            key="surv_alert_ratio",
                        )
                    with a4:
                        alert_baseline = st.number_input(
                            "Semaines ref.",
                            min_value=2,
                            max_value=8,
                            value=3,
                            step=1,
                            key="surv_alert_baseline_weeks",
                        )

                    alert_group_cols = [alert_group_col]
                    if alert_group_col == COL_ZS and COL_PROV in df_surv_scope.columns:
                        alert_group_cols = [COL_PROV, COL_ZS]
                    elif alert_group_col == COL_AS:
                        alert_group_cols = [
                            c for c in [COL_PROV, COL_ZS, COL_AS]
                            if c in df_surv_scope.columns
                        ]
                    alert_group_key = "__alert_geo_key"
                    alert_label_sep = " / "
                    alert_scope = df_surv_scope.copy()
                    alert_scope[alert_group_cols] = (
                        alert_scope[alert_group_cols]
                        .astype("string")
                        .fillna("Non renseigné")
                    )
                    alert_scope[alert_group_key] = (
                        alert_scope[alert_group_cols]
                        .astype(str)
                        .agg(alert_label_sep.join, axis=1)
                    )

                    alert_tbl = build_weekly_alerts(
                        alert_scope,
                        alert_group_key,
                        week_col="_surv_label",
                        baseline_weeks=int(alert_baseline),
                        min_baseline_periods=2,
                        min_cases=int(alert_min_cases),
                        alert_ratio=float(alert_ratio),
                    )
                    if alert_tbl.empty:
                        render_absence_narrative("alerts")
                    else:
                        latest_alert_week = alert_tbl["_surv_label"].dropna().max()
                        latest_alerts = alert_tbl[alert_tbl["_surv_label"] == latest_alert_week].copy()
                        if alert_group_key in latest_alerts.columns:
                            split_cols = latest_alerts[alert_group_key].astype(str).str.split(
                                alert_label_sep,
                                expand=True,
                            )
                            for idx, geo_col in enumerate(alert_group_cols):
                                if idx < split_cols.shape[1]:
                                    latest_alerts[geo_col] = split_cols[idx]
                            latest_alerts = latest_alerts.drop(columns=[alert_group_key])
                        alert_display_cols = [
                            c for c in alert_group_cols
                            if c in latest_alerts.columns
                        ] + [
                            c for c in ["_surv_label", "Cas", "Cas_prev", "var_%", "baseline", "ratio_baseline", "signal_level", "signal"]
                            if c in latest_alerts.columns
                        ]
                        latest_alerts = latest_alerts.sort_values(
                            ["signal", "Cas", "ratio_baseline"],
                            ascending=[False, False, False],
                            na_position="last",
                        )
                        sig_count = int(latest_alerts["signal"].fillna(False).sum())
                        st.caption(
                            f"Dernière semaine analysée : {latest_alert_week} | Alertes détectées : {sig_count}. "
                            "Un signal invite à vérifier la situation locale, il ne confirme pas à lui seul une flambée."
                        )
                        st.dataframe(latest_alerts[alert_display_cols].head(100), width="stretch", height=360, hide_index=True)

                        sig_plot = latest_alerts[latest_alerts["signal"] == True].head(30)
                        if not sig_plot.empty:
                            sig_plot["_geo_label"] = (
                                sig_plot[alert_group_cols]
                                .astype("string")
                                .fillna("Non renseigné")
                                .astype(str)
                                .agg(alert_label_sep.join, axis=1)
                            )
                            fig_alert = px.bar(
                                sig_plot.sort_values("Cas", ascending=True),
                                x="Cas",
                                y="_geo_label",
                                orientation="h",
                                color="ratio_baseline",
                                title="Groupes en alerte sur la dernière semaine",
                                color_continuous_scale=["#fde68a", "#b91c1c"],
                            )
                            fig_alert.update_layout(coloraxis_colorbar_title="Ratio")
                            fig_alert = apply_plotly_value_annotations(fig_alert, annot_vals)
                            st_plot(fig_alert, key="surv_alert_latest_chart")

                    cluster_cols = alert_group_cols
                    cluster_tbl = build_spatiotemporal_cluster_table(
                        df_surv_scope,
                        group_cols=cluster_cols if cluster_cols else [alert_group_col],
                        week_col="_surv_label",
                        recent_weeks=2,
                        previous_weeks=4,
                        min_recent_cases=max(5, int(alert_min_cases // 2)),
                        growth_ratio=float(alert_ratio),
                    )
                    if not cluster_tbl.empty:
                        with st.expander("Clusters spatio-temporels récents", expanded=False):
                            st.caption(
                                "Ce tableau compare les deux dernières semaines à la fenêtre précédente. "
                                "Il aide à repérer des foyers récents à vérifier avec les équipes locales."
                            )
                            st.dataframe(cluster_tbl.head(100), width="stretch", height=420, hide_index=True)
                    else:
                        render_absence_narrative("alerts")

                    weekly_lab_tbl = build_weekly_lab_summary(df_surv_scope)
                    if not weekly_lab_tbl.empty:
                        with st.expander("Suivi hebdomadaire des indicateurs laboratoire", expanded=False):
                            st.caption(
                                "Ce suivi temporel standard résume, par semaine, le volume de tests interprétables, "
                                "les tests positifs et la positivité observée dans la fenêtre active."
                            )
                            l1, l2, l3 = st.columns(3)
                            l1.metric(
                                "Semaines avec suivi labo",
                                format_metric_value(len(weekly_lab_tbl)),
                            )
                            l2.metric(
                                "Tests valides cumulés",
                                format_metric_value(pd.to_numeric(weekly_lab_tbl["Tests valides"], errors="coerce").sum()),
                            )
                            latest_pos = pd.to_numeric(
                                weekly_lab_tbl["Positivité (%)"],
                                errors="coerce",
                            ).dropna()
                            l3.metric(
                                "Positivité récente (%)",
                                "-" if latest_pos.empty else f"{float(latest_pos.iloc[-1]):.1f}",
                            )

                            fig_lab_trend = go.Figure()
                            fig_lab_trend.add_trace(
                                go.Bar(
                                    x=weekly_lab_tbl["Semaine"],
                                    y=weekly_lab_tbl["Tests valides"],
                                    name="Tests valides",
                                    marker_color="#4f81bd",
                                )
                            )
                            fig_lab_trend.add_trace(
                                go.Bar(
                                    x=weekly_lab_tbl["Semaine"],
                                    y=weekly_lab_tbl["Tests positifs"],
                                    name="Tests positifs",
                                    marker_color="#d97b16",
                                )
                            )
                            fig_lab_trend.add_trace(
                                go.Scatter(
                                    x=weekly_lab_tbl["Semaine"],
                                    y=weekly_lab_tbl["Positivité (%)"],
                                    name="Positivité (%)",
                                    mode="lines+markers",
                                    line=dict(color="#b9353f", width=3),
                                    marker=dict(size=8),
                                    yaxis="y2",
                                )
                            )
                            fig_lab_trend.update_layout(
                                title=" ",
                                barmode="group",
                                xaxis_title="Semaine épidémiologique",
                                yaxis_title="Nombre de tests",
                                yaxis2=dict(
                                    title="Positivité (%)",
                                    overlaying="y",
                                    side="right",
                                    rangemode="tozero",
                                ),
                            )
                            st_plot(fig_lab_trend, key="surv_weekly_lab_combo", annotate_values=False)
                            st_dataframe_safe(weekly_lab_tbl, height=320)
                else:
                    render_absence_narrative("geo")
        else:
            render_absence_narrative("week")
    # Section suivante : promptitude. Les indicateurs de performance et de létalité déjà présentés plus haut ne sont pas répétés ici afin d’éviter les redondances.

    st.divider()
    render_section_title(2, "Promptitude de notification, investigation et prise en charge")
    if IDSR_MODE:
        render_absence_narrative("idsr_line_list")
    else:
        render_tab_narrative("promptitude")
        tab_help(
            "Comment lire cet onglet",
            f"""
            **🎯 Objectif** : Mesurer la rapidité de détection et d’accès aux soins.
        
            **📖 Indicateurs**
            - Délai **début maladie → admission**
            - Délai **début maladie → prélèvement**
            - **% ≤ {seuil_jours} jours** : proportion de cas pris en charge rapidement.
        
            **⚠️ Points d’attention**
            - Des délais longs augmentent le risque de transmission communautaire.
            - Des délais négatifs ou extrêmes = erreurs de saisie ou dates incorrectes.
            """,
            expanded=False
        )
        
        st.subheader("Analyse de la promptitude des principales étapes du parcours du cas et de la notification")
        
        delais_cols = [
            c for c in ["delai_onset_to_adm", "delai_onset_to_prel", "delai_prel_to_result"]
            if c in df_f.columns
        ]
        
        if not delais_cols:
            render_absence_narrative("delays")
        else:
            df_del = _surv_prepare_delay_scope(df_f, delais_cols)

            delay_summary_std = build_standard_delay_summary(df_del)
            available_delay_pairs = list_available_standard_delays(df_del)
            seuil_val = float(seuil_jours)
            seuil_lab = int(seuil_val) if seuil_val.is_integer() else round(seuil_val, 1)

            st.markdown("**Lecture opérationnelle prioritaire**")
            st.caption(
                "Les indicateurs et classements ci-dessous sont plus utiles pour l’action immédiate que la distribution graphique brute."
            )

            p1, n1 = _surv_threshold_metric(df_del["delai_onset_to_adm"] if "delai_onset_to_adm" in df_del.columns else None, seuil_jours)
            p2, n2 = _surv_threshold_metric(df_del["delai_onset_to_prel"] if "delai_onset_to_prel" in df_del.columns else None, seuil_jours)
            p3, n3 = _surv_threshold_metric(df_del["delai_prel_to_result"] if "delai_prel_to_result" in df_del.columns else None, seuil_jours)

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("Admission ≤ seuil (%)", _surv_metric_pct(p1), help=f"n = {n1}")
            with c2:
                st.metric("Prélèvement ≤ seuil (%)", _surv_metric_pct(p2), help=f"n = {n2}")
            with c3:
                st.metric("Délais admission documentés", format_metric_value(n1))
            with c4:
                st.metric("Délais prélèvement documentés", format_metric_value(n2))
            with c5:
                st.metric("Résultat ≤ seuil (%)", _surv_metric_pct(p3), help=f"n = {n3}")
            with c6:
                st.metric("Délais résultat documentés", format_metric_value(n3))

            if not delay_summary_std.empty:
                st.markdown("**Résumé standard des délais disponibles**")
                st_dataframe_safe(delay_summary_std, height=320)

            if COL_PROV in df_del.columns:
                st.markdown("**Provinces à surveiller en priorité**")
                ranking_specs = []
                if "delai_onset_to_adm" in delais_cols:
                    ranking_specs.append(("delai_onset_to_adm", "Début maladie → admission"))
                if "delai_onset_to_prel" in delais_cols:
                    ranking_specs.append(("delai_onset_to_prel", "Début maladie → prélèvement"))
                if "delai_prel_to_result" in delais_cols:
                    ranking_specs.append(("delai_prel_to_result", "Prélèvement → résultat"))

                if ranking_specs:
                    rank_cols = st.columns(len(ranking_specs))
                    for col_ui, (delay_col, delay_label) in zip(rank_cols, ranking_specs):
                        delay_group_tbl = build_delay_group_summary(
                            df_del,
                            delay_col=delay_col,
                            group_col=COL_PROV,
                            threshold=seuil_jours,
                        )
                        pct_col = f"% <= {seuil_lab} j"
                        if not delay_group_tbl.empty and pct_col in delay_group_tbl.columns:
                            delay_group_tbl = delay_group_tbl[
                                pd.to_numeric(delay_group_tbl["n"], errors="coerce").fillna(0) > 0
                            ].copy()
                            delay_group_tbl = (
                                delay_group_tbl
                                .sort_values([pct_col, "Mediane_j", "n"], ascending=[True, False, False], na_position="last")
                                .head(10)
                                .copy()
                            )
                        with col_ui:
                            st.markdown(f"**{delay_label}**")
                            if delay_group_tbl.empty:
                                render_absence_narrative("delays")
                            else:
                                st.dataframe(delay_group_tbl, width="stretch", height=360, hide_index=True)

            with st.expander("Distribution détaillée des délais observés (graphique)", expanded=False):
                if use_custom_viz and HAS_CUSTOM_VIZ:
                    plot_delay_cols = list(delais_cols)
                    if COL_PROV in df_del.columns:
                        plot_delay_cols.append(COL_PROV)
                    df_delay_plot = _surv_plotly_frame(df_del[plot_delay_cols], numeric_cols=delais_cols)
                    fig = plot_boxplot_delais_plotly(
                        df=df_delay_plot,
                        colonnes_delais=delais_cols,
                        col_groupe=COL_PROV if COL_PROV in df_delay_plot.columns else None,
                        titre=" ",
                        taille_fig=(1500, 600),
                        rotation=45
                    )
                    st_plot(fig, key="boxplot_delais_custom")
                else:
                    long = (
                        df_del.melt(value_vars=delais_cols, var_name="Type_delai", value_name="Jours")
                        .copy()
                    )
                    long["Jours"] = pd.to_numeric(long["Jours"], errors="coerce")
                    long.loc[~np.isfinite(long["Jours"]), "Jours"] = np.nan
                    long = long.dropna(subset=["Type_delai", "Jours"])
                    if long.empty:
                        render_absence_narrative("delays")
                    else:
                        long = _surv_plotly_frame(long, numeric_cols=["Jours"])
                        fig = px.box(long, x="Type_delai", y="Jours", points="outliers", title="Boxplot des délais (global)")
                        fig = apply_plotly_value_annotations(fig, annot_vals)
                        st_plot(fig, key="boxplot_delais_standard")

            if available_delay_pairs:
                st.divider()
                st.markdown("**Analyse détaillée d'un délai standard**")
                delay_label_to_col = {label: col for col, label in available_delay_pairs}
                delay_focus_label = st.selectbox(
                    "Délai standard à profiler",
                    options=list(delay_label_to_col.keys()),
                    key="timeliness_delay_focus",
                )

                group_candidates = []
                for c in [COL_PROV, COL_ZS, pick_age_col(df_del), COL_SEX, COL_CLASS, "Type_de_prelevement", "Nom_laboratoire"]:
                    if c and c in df_del.columns and df_del[c].notna().any() and c not in group_candidates:
                        group_candidates.append(c)

                if group_candidates:
                    g1, g2, g3 = st.columns([1.15, 1.15, 0.9])
                    with g1:
                        delay_group_focus = st.selectbox(
                            "Variable de regroupement",
                            options=group_candidates,
                            key="timeliness_group_focus",
                        )
                    with g2:
                        delay_metric_focus = st.selectbox(
                            "Indicateur a classer",
                            options=["Mediane (jours)", f"% <= {seuil_jours} jours"],
                            key="timeliness_metric_focus",
                        )
                    with g3:
                        delay_topn = st.slider(
                            "Top groupes",
                            min_value=5,
                            max_value=30,
                            value=15,
                            step=1,
                            key="timeliness_group_topn",
                        )

                    delay_focus_col = delay_label_to_col[delay_focus_label]
                    delay_group_tbl = build_delay_group_summary(
                        df_del,
                        delay_col=delay_focus_col,
                        group_col=delay_group_focus,
                        threshold=seuil_jours,
                    )

                    if not delay_group_tbl.empty:
                        pct_col = f"% <= {seuil_lab} j"
                        sort_col = "Mediane_j" if delay_metric_focus.startswith("Mediane") else pct_col
                        ascending = bool(delay_metric_focus.startswith("Mediane"))
                        delay_group_view = (
                            delay_group_tbl.sort_values(sort_col, ascending=ascending, na_position="last")
                            .head(int(delay_topn))
                            .copy()
                        )

                        t1, t2 = st.columns([1.05, 1.35])
                        with t1:
                            st.dataframe(delay_group_view, width="stretch", height=420, hide_index=True)
                        with t2:
                            plot_df = delay_group_view.sort_values(sort_col, ascending=True, na_position="last").copy()
                            plot_df = _surv_plotly_frame(plot_df, numeric_cols=[sort_col])
                            fig_delay_focus = px.bar(
                                plot_df,
                                x=sort_col,
                                y=delay_group_focus,
                                orientation="h",
                                text=sort_col,
                                title=f"{delay_focus_label} par {delay_group_focus}",
                                color=sort_col,
                                color_continuous_scale=["#dbe8f9", "#2b74ca"],
                            )
                            fig_delay_focus.update_layout(
                                coloraxis_showscale=False,
                                xaxis_title=sort_col,
                                yaxis_title=delay_group_focus,
                            )
                            fig_delay_focus = apply_plotly_value_annotations(fig_delay_focus, annot_vals)
                            st_plot(fig_delay_focus, key="timeliness_delay_focus_chart")
                    else:
                        render_absence_narrative("delays")
                else:
                    render_absence_narrative("profile")

    # =========================
    # TAB 4: Démographie
    # =========================

