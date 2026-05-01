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
            - La **situation des 4 dernières semaines** permet de lire rapidement la tendance récente.
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

        df_surv_scope, surv_reference = _prepare_surveillance_period_scope(df_f)

        if not surv_reference.empty:
            latest_order = surv_reference["order"].iloc[-1]
            latest_label = str(surv_reference["label"].iloc[-1])
            last4_reference = surv_reference.tail(4).copy()
            last4_orders = last4_reference["order"].tolist()
            last4_labels = last4_reference["label"].astype(str).tolist()
            first_label = str(surv_reference["label"].iloc[0])
            previous_week_reference = surv_reference.tail(2).head(1).copy() if len(surv_reference) >= 2 else pd.DataFrame()
            previous_week_df = pd.DataFrame()
            previous_week_label = None
            if not previous_week_reference.empty:
                previous_week_label = str(previous_week_reference["label"].iloc[0])
                previous_week_df = df_surv_scope[
                    df_surv_scope["_surv_order"] == previous_week_reference["order"].iloc[0]
                ].copy()

            prev4_reference = (
                surv_reference.iloc[max(len(surv_reference) - 8, 0): max(len(surv_reference) - 4, 0)].copy()
                if len(surv_reference) > 4 else pd.DataFrame()
            )
            prev4_df = pd.DataFrame()
            prev4_label = None
            if not prev4_reference.empty:
                prev4_orders = prev4_reference["order"].tolist()
                prev4_df = df_surv_scope[df_surv_scope["_surv_order"].isin(prev4_orders)].copy()
                prev4_labels = prev4_reference["label"].astype(str).tolist()
                if prev4_labels:
                    prev4_label = ", ".join(prev4_labels)

            df_latest_week = df_surv_scope[df_surv_scope["_surv_order"] == latest_order].copy()
            df_last4_weeks = df_surv_scope[df_surv_scope["_surv_order"].isin(last4_orders)].copy()

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
            )

            st.divider()

            _render_surveillance_window(
                "2. Situation des 4 dernières semaines",
                df_last4_weeks,
                f"Lecture glissante sur les {len(last4_labels)} semaines les plus récentes de la sélection : {', '.join(last4_labels)}.",
                "Aucune donnée n’est disponible pour construire la tendance des 4 dernières semaines.",
                narrative_context={
                    "scope_kind": "recent4",
                    "current_label": "Les 4 dernières semaines",
                    "comparison_df": prev4_df if not prev4_df.empty else None,
                    "comparison_label": "les 4 semaines précédentes" if prev4_label else None,
                    "latest_week_df": df_latest_week,
                    "latest_label": latest_label,
                },
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
                        st_plot(
                            fig_multi_prov,
                            key="surveillance_multi_curve_province",
                            height=700,
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
        
        delais_cols = [c for c in ["delai_onset_to_adm", "delai_onset_to_prel"] if c in df_f.columns]
        
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

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                p1, n1 = pct_under_threshold(df_del.get("delai_onset_to_adm"), seuil_jours)
                st.metric("Admission ≤ seuil (%)", _surv_metric_pct(p1), help=f"n = {n1}")
            with c2:
                p2, n2 = pct_under_threshold(df_del.get("delai_onset_to_prel"), seuil_jours)
                st.metric("Prélèvement ≤ seuil (%)", _surv_metric_pct(p2), help=f"n = {n2}")
            with c3:
                st.metric("Délais admission documentés", format_metric_value(n1))
            with c4:
                st.metric("Délais prélèvement documentés", format_metric_value(n2))

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
                for c in [COL_PROV, COL_ZS, pick_age_col(df_del), COL_SEX, COL_CLASS]:
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

