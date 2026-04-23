"""Render the data quality and export tab."""

from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


def render_quality_tab(ctx: dict) -> None:
    """Render the data quality and export tab."""
    globals().update(ctx)
    render_section_title(4, "Complétude des données et couverture des rapports")
    if IDSR_MODE:
        render_absence_narrative("idsr_line_list")
    else:
        render_tab_narrative("qualite")
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Vérifier si les provinces attendues notifient (complétude géographique).
        
            **📖 Interprétation**
            - **Manquantes** : silence épidémiologique ou problème de remontée/rapportage.
            - Le tableau croisé aide à repérer les zones/provinces dominantes ou sous-notifiantes.
        
            **⚠️ Points d’attention**
            - Une province silencieuse pendant une épidémie = signal d’alerte système à investiguer.
            """,
            expanded=False
        )     
        
        with st.expander("Définir les provinces en épidémie (attendues dans la line list)", expanded=False):

            # ---- Init states ----
            if "epidemie_state_tab5" not in st.session_state:
                st.session_state.epidemie_state_tab5 = EPIDEMIE.copy()

            if "epid_version_tab5" not in st.session_state:
                st.session_state.epid_version_tab5 = 0

            # ---- Callbacks (ne modifient PAS les keys existantes) ----
            def _apply_bulk_tab5(value: bool):
                # Met à jour le dict + change de version -> recrée les checkboxes
                for p in EPIDEMIE.keys():
                    st.session_state.epidemie_state_tab5[p] = value
                st.session_state.epid_version_tab5 += 1

            def _reset_defaults_tab5():
                st.session_state.epidemie_state_tab5 = EPIDEMIE.copy()
                st.session_state.epid_version_tab5 += 1

            def _sync_one_tab5(prov: str, widget_key: str):
                # Synchronise le dict à partir de l'état réel du widget
                st.session_state.epidemie_state_tab5[prov] = bool(st.session_state.get(widget_key, False))

            st.markdown("✅ **Coche** = province considérée **en épidémie** (attendue dans la line list)")

            provs = sorted(list(EPIDEMIE.keys()))
            cols = st.columns(3)

            # ---- Checkboxes (keys versionnées) ----
            v = st.session_state.epid_version_tab5
            for i, prov in enumerate(provs):
                with cols[i % 3]:
                    wkey = f"chk_epid_{prov}_v{v}"  # <-- clé versionnée
                    st.checkbox(
                        prov,
                        value=st.session_state.epidemie_state_tab5.get(prov, False),
                        key=wkey,
                        on_change=_sync_one_tab5,
                        args=(prov, wkey),
                    )

            # ---- Boutons (on_click = safe) ----
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.button("Sélectionner toutes les provinces", key="tab5_all", on_click=_apply_bulk_tab5, args=(True,))
            with c2:
                st.button("Désélectionner toutes les provinces", key="tab5_none", on_click=_apply_bulk_tab5, args=(False,))
            with c3:
                st.button("Réinitialiser selon les paramètres par défaut du script", key="tab5_reset", on_click=_reset_defaults_tab5)

            # ✅ Provinces attendues (UI Tab5)
            PROVINCES_EPID = [p for p, ok in st.session_state.epidemie_state_tab5.items() if ok]
       
        st.subheader("Suivi de la complétude de notification : provinces attendues versus provinces effectivement rapportées")
        
        if COL_PROV not in df_f.columns:
            st.info("La variable Province_notification est absente du fichier analysé.")
        else:
            if COL_WNUM in df_f.columns and df_f[COL_WNUM].notna().any():
                last_w = int(df_f[COL_WNUM].max())
                present = sorted(df_f.loc[df_f[COL_WNUM] == last_w, COL_PROV].dropna().unique().tolist())
                st.caption(f"Calcul sur la semaine max filtrée: SE{last_w:02d}")
            else:
                present = sorted(df_f[COL_PROV].dropna().unique().tolist())
                st.caption("Calcul sur l’ensemble filtré (pas de Num_semaine_epid exploitable).")
        
            missing = [p for p in PROVINCES_EPID if p not in present]
            nb_att = len(PROVINCES_EPID)
            nb_rec = len([p for p in PROVINCES_EPID if p in present])
            compl = (nb_rec / nb_att * 100) if nb_att > 0 else np.nan
        
            c1, c2, c3 = st.columns(3)
            c1.metric("Provinces attendues", str(nb_att))
            c2.metric("Provinces trouvées", str(nb_rec))
            c3.metric("Complétude (%)", f"{compl:.1f}")
            if missing:
                st.warning("Provinces attendues non reçues : " + ", ".join(missing))
        
            with st.expander("Tableau provinces attendues vs reçues"):
                df_comp = pd.DataFrame({
                    "Province attendue": PROVINCES_EPID,
                    "Présente": [p in present for p in PROVINCES_EPID],
                    "Manquante": [p if p in missing else "" for p in PROVINCES_EPID],
                })
                st_dataframe_safe(df_comp)
        
            with st.expander("Cas par province (complétude / volume)", expanded=True):
                prov_counts = df_f[COL_PROV].fillna("Inconnu").value_counts().reset_index()
                prov_counts.columns = [COL_PROV, "Cas"]
                figp = px.bar(prov_counts, x=COL_PROV, y="Cas", title=" ")
                figp.update_layout(xaxis_tickangle=-45)
                figp = apply_plotly_value_annotations(figp, annot_vals)
                st.plotly_chart(figp, width="stretch")
        
            # TCD
            with st.expander("Tableau croisé dynamique – occurrences", expanded=False):
                # --- Scope: même logique que ton calcul "semaine max filtrée"
                scope_last_week = st.checkbox(
                    "Calculer uniquement sur la semaine max filtrée (même scope que la complétude)",
                    value=True,
                    key="ct_scope_last_week"
                )
                df_scope = df_f.copy()
                if scope_last_week and (COL_WNUM in df_scope.columns) and df_scope[COL_WNUM].notna().any():
                    last_w = int(df_scope[COL_WNUM].max())
                    df_scope = df_scope.loc[df_scope[COL_WNUM] == last_w].copy()
                    st.caption(f"Scope: SE{last_w:02d}")
                else:
                    st.caption("Scope: ensemble filtré")
        
                # --- Outils UX (global)
                cUX1, cUX2, cUX3, cUX4 = st.columns([1.1, 1.1, 1.1, 0.9])
                with cUX1:
                    show_pct = st.checkbox("Afficher les pourcentages", value=False, key="ct_show_pct")
                with cUX2:
                    show_bar = st.checkbox("Afficher les barres dans le tableau", value=True, key="ct_show_bar")
                with cUX3:
                    tbl_height = st.number_input("Hauteur du tableau", min_value=250, max_value=1200, value=520, step=50, key="ct_tbl_height")
                with cUX4:
                    do_download = st.checkbox("Activer l’export", value=True, key="ct_export_on")
        
                # --- Choix du niveau d’agrégation (on maintient les 3 options)
                level = st.radio(
                    "Niveau d’agrégation",
                    ["Province (occurrences)", "Province + Zone de santé", "Tableau croisé Province × Zone"],
                    index=0,
                    horizontal=True,
                    key="ct_level"
                )
        
                # Helper: affiche tableau + option export
                def _show_table(df_to_show: pd.DataFrame, name: str):
                    st.dataframe(
                        df_to_show, width='stretch', height=int(tbl_height),
                        hide_index=False,
                        column_config=None
                    )
                    if do_download:
                        csv = df_to_show.to_csv(index=True).encode("utf-8")
                        st.download_button(
                            f"Télécharger {name} (CSV)",
                            data=csv,
                            file_name=f"{name}.csv".replace(" ", "_").lower(),
                            mime="text/csv",
                            key=f"dl_{name}"
                        )
        
                # 1) Province (occurrences)
                if level == "Province (occurrences)":
                    if COL_PROV not in df_scope.columns:
                        st.info("La variable Province_notification est absente du fichier analysé.")
                    else:
                        piv = (
                            df_scope.assign(_prov=df_scope[COL_PROV].fillna("Inconnu"))
                            .groupby("_prov", dropna=False)
                            .size()
                            .reset_index(name="Occurrences")
                            .sort_values("Occurrences", ascending=False)
                            .rename(columns={"_prov": COL_PROV})
                        )
        
                        if show_pct:
                            total = int(piv["Occurrences"].sum()) if len(piv) else 0
                            piv["%"] = (piv["Occurrences"] / total * 100).round(1) if total > 0 else 0.0
        
                        if show_bar:
                            st.dataframe(
                                piv, width='stretch', height=int(tbl_height),
                                column_config={
                                    "Occurrences": st.column_config.ProgressColumn(
                                        "Occurrences",
                                        help="Occurrences (barres)",
                                        format="%d",
                                        min_value=0,
                                        max_value=int(piv["Occurrences"].max()) if len(piv) else 1,
                                    )
                                },
                            )
                            if do_download:
                                csv = df_to_csv_bytes(piv)
                                st.download_button(
                                    "Télécharger province_occurrences (CSV)",
                                    data=csv,
                                    file_name="province_occurrences.csv",
                                    mime="text/csv",
                                    key="dl_prov_occ"
                                )
                        else:
                            _show_table(piv, "province_occurrences")
        
                    with st.expander("Graphique (top provinces)"):
                        topk = st.number_input("Nombre de provinces à afficher", min_value=5, max_value=30, value=15, step=1, key="ct_topk_prov")
                        figp = px.bar(piv.head(int(topk)), x=COL_PROV, y="Occurrences", title="Top provinces – occurrences")
                        figp.update_layout(xaxis_tickangle=-45)
                        figp = apply_plotly_value_annotations(figp, annot_vals)
                        st.plotly_chart(figp, width="stretch")
        
                # 2) Province + Zone de santé
                elif level == "Province + Zone de santé":
                    if (COL_PROV not in df_scope.columns) or (COL_ZS not in df_scope.columns):
                        st.info("Les variables Province_notification et/ou Zone_de_sante_notification sont absentes.")
                    else:
                        colA, colB, colC = st.columns([1.2, 1.2, 1.6])
                        with colA:
                            view_mode = st.radio(
                                "Vue",
                                ["Top N (table longue)", "Déroulable Province → Zone"],
                                index=1,
                                horizontal=True,
                                key="ct_view_mode_pz"
                            )
                        with colB:
                            limit_zones = st.checkbox("Limiter le nombre de zones de santé (performance)", value=True, key="ct_limit_zones_pz")
                        with colC:
                            top_z = st.number_input("Nombre maximum de zones de santé", min_value=10, max_value=2000, value=250, step=25, key="ct_top_z_pz")
        
                        df_scope2 = df_scope.copy()
                        if limit_zones:
                            zones_top = (
                                df_scope2[COL_ZS].fillna("Inconnu")
                                .value_counts()
                                .head(int(top_z))
                                .index.tolist()
                            )
                            df_scope2 = df_scope2[df_scope2[COL_ZS].fillna("Inconnu").isin(zones_top)].copy()
        
                        piv = (
                            df_scope2.assign(
                                _prov=df_scope2[COL_PROV].fillna("Inconnu"),
                                _zs=df_scope2[COL_ZS].fillna("Inconnu"),
                            )
                            .groupby(["_prov", "_zs"], dropna=False)
                            .size()
                            .reset_index(name="Occurrences")
                            .sort_values("Occurrences", ascending=False)
                            .rename(columns={"_prov": COL_PROV, "_zs": COL_ZS})
                        )
        
                        if show_pct:
                            tot_prov = piv.groupby(COL_PROV, as_index=False)["Occurrences"].sum().rename(columns={"Occurrences": "Total_province"})
                            piv = piv.merge(tot_prov, on=COL_PROV, how="left")
                            piv["%_dans_province"] = (piv["Occurrences"] / piv["Total_province"] * 100).round(1)
                            piv = piv.drop(columns=["Total_province"])
        
                        tot_prov = (
                            piv.groupby(COL_PROV, as_index=False)["Occurrences"].sum()
                            .sort_values("Occurrences", ascending=False)
                        )
        
                        if view_mode == "Top N (table longue)":
                            top_n = st.number_input("Nombre maximum de lignes à afficher", min_value=10, max_value=20000, value=500, step=50, key="ct_topn_long")
                            df_show = piv.head(int(top_n)).copy()
        
                            if show_bar:
                                st.dataframe(
                                    df_show, width='stretch', height=int(tbl_height),
                                    column_config={
                                        "Occurrences": st.column_config.ProgressColumn(
                                            "Occurrences",
                                            format="%d",
                                            min_value=0,
                                            max_value=int(piv["Occurrences"].max()) if len(piv) else 1,
                                        )
                                    },
                                )
                            else:
                                _show_table(df_show, "province_zone_topN")
        
                        else:
                            tcd = (
                                piv.set_index([COL_PROV, COL_ZS])[["Occurrences"]]
                                .sort_values("Occurrences", ascending=False)
                            )
                            tcd = tcd.reindex(tot_prov[COL_PROV].tolist(), level=0)
        
                            st.caption("Clique sur les triangles à gauche pour dérouler/replier Province → Zone.")
                            st.dataframe(tcd, width='stretch', height=int(tbl_height))
        
                            if do_download:
                                csv = tcd.reset_index().to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Télécharger province_zone_deroulable (CSV)",
                                    data=csv,
                                    file_name="province_zone_deroulable.csv",
                                    mime="text/csv",
                                    key="dl_pz_deroulable"
                                )
        
                        with st.expander("Totaux par province (somme des zones)"):
                            if show_bar:
                                st.dataframe(
                                    tot_prov, width='stretch', height=450,
                                    column_config={
                                        "Occurrences": st.column_config.ProgressColumn(
                                            "Occurrences",
                                            format="%d",
                                            min_value=0,
                                            max_value=int(tot_prov["Occurrences"].max()) if len(tot_prov) else 1,
                                        )
                                    },
                                )
                            else:
                                st_dataframe_safe(tot_prov)
        
                        with st.expander("Graphique (top provinces)"):
                            topk = st.number_input("Nombre de provinces à afficher", min_value=5, max_value=30, value=15, step=1, key="ct_topk_pz")
                            figp = px.bar(tot_prov.head(int(topk)), x=COL_PROV, y="Occurrences", title="Top provinces – occurrences (scope)")
                            figp.update_layout(xaxis_tickangle=-45)
                            figp = apply_plotly_value_annotations(figp, annot_vals)
                            st.plotly_chart(figp, width="stretch")
        
                # 3) Tableau croisé Province × Zone
                else:
                    if (COL_PROV not in df_scope.columns) or (COL_ZS not in df_scope.columns):
                        st.info("Les variables Province_notification et/ou Zone_de_sante_notification sont absentes.")
                    else:
                        cA, cB, cC = st.columns([1.1, 1.3, 1.6])
                        with cA:
                            limit_zones = st.checkbox("Limiter aux zones les plus fréquentes", value=True, key="ct_limit_zones_wide")
                        with cB:
                            top_z = st.number_input("Top zones", min_value=10, max_value=1500, value=120, step=10, key="ct_topz_wide")
                        with cC:
                            show_heatmap = st.checkbox("Afficher en heatmap", value=False, key="ct_show_heatmap")
        
                        if limit_zones:
                            zones_top = (
                                df_scope[COL_ZS].fillna("Inconnu")
                                .value_counts()
                                .head(int(top_z))
                                .index.tolist()
                            )
                            df_ct = df_scope[df_scope[COL_ZS].fillna("Inconnu").isin(zones_top)].copy()
                        else:
                            df_ct = df_scope.copy()
        
                        ct = pd.crosstab(
                            index=df_ct[COL_PROV].fillna("Inconnu"),
                            columns=df_ct[COL_ZS].fillna("Inconnu"),
                            margins=True,
                            margins_name="Total",
                            dropna=False
                        )
        
                        sort_totals = st.checkbox("Trier par total décroissant", value=True, key="ct_sort_totals")
                        if sort_totals and "Total" in ct.columns and "Total" in ct.index:
                            rows = ct.drop(index="Total", errors="ignore").sort_values("Total", ascending=False)
                            cols_tot = ct.drop(columns="Total", errors="ignore").loc["Total"].sort_values(ascending=False).index.tolist() \
                                if "Total" in ct.index else ct.drop(columns="Total", errors="ignore").columns.tolist()
                            ct = rows[cols_tot]
                            ct.loc["Total"] = ct.sum(axis=0)
                            ct["Total"] = ct.sum(axis=1)
                            ct = ct.fillna(0).astype(int)
        
                        st.dataframe(ct, width='stretch', height=int(tbl_height))
        
                        if do_download:
                            csv = ct.to_csv(index=True).encode("utf-8")
                            st.download_button(
                                "Télécharger province_x_zone (CSV)",
                                data=csv,
                                file_name="province_x_zone.csv",
                                mime="text/csv",
                                key="dl_ct_wide"
                            )
        
                        if show_heatmap:
                            ct_heat = ct.drop(index="Total", errors="ignore").drop(columns="Total", errors="ignore")
                            fig_hm = px.imshow(
                                ct_heat,
                                aspect="auto",
                                labels=dict(x="Zone de santé", y="Province", color="Occurrences"),
                                title="Heatmap – Occurrences Province × Zone"
                            )
                            fig_hm.update_layout(height=700)
                            st.plotly_chart(fig_hm, width="stretch")
        
    # =========================
    # TAB 6: DATA & EXPORT
    # =========================
    st.divider()
    render_section_title(5, "Extraction, revue et export des données")
    if IDSR_MODE:
        render_absence_narrative("idsr_line_list")
    else:
        render_tab_narrative("export")
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Consulter et exporter les données filtrées pour analyses/partage.
        
            **📖 Utilisation**
            - Exportation **CSV/Excel** pour analyses complémentaires (R/Python/DHIS2).
            - Vérifier les filtres actifs avant export.
        
            **⚠️ Points d’attention**
            - Les exports reflètent exactement le périmètre filtré (province/ZS/AS/semaine/classification).
            """,
            expanded=False
        )
        
        st.subheader("Extraction des données filtrées, traçabilité et options d’export")

        def _build_export_traceability_table(df_scope: pd.DataFrame) -> pd.DataFrame:
            """Construit une table simple de traçabilité pour les exports."""
            week_min_val = "-"
            week_max_val = "-"
            year_min_val = "-"
            year_max_val = "-"

            if COL_WNUM in df_scope.columns and df_scope[COL_WNUM].notna().any():
                week_num = pd.to_numeric(df_scope[COL_WNUM], errors="coerce").dropna()
                if not week_num.empty:
                    week_min_val = int(week_num.min())
                    week_max_val = int(week_num.max())

            if COL_YEAR in df_scope.columns and df_scope[COL_YEAR].notna().any():
                year_num = pd.to_numeric(df_scope[COL_YEAR], errors="coerce").dropna()
                if not year_num.empty:
                    year_min_val = int(year_num.min())
                    year_max_val = int(year_num.max())

            rows = [
                {"Paramètre": "Date export", "Valeur": str(date.today())},
                {"Paramètre": "Maladie / line list", "Valeur": DISEASE_SPECS.get(disease_key, {}).get("label", disease_key)},
                {"Paramètre": "Sources chargées", "Valeur": ", ".join(files_used) if isinstance(files_used, list) and files_used else "Non documenté"},
                {"Paramètre": "Cas exportés", "Valeur": int(len(df_scope))},
                {"Paramètre": "Colonnes exportées", "Valeur": int(len(df_scope.columns))},
                {"Paramètre": "SE min observée", "Valeur": week_min_val},
                {"Paramètre": "SE max observée", "Valeur": week_max_val},
                {"Paramètre": "Année min observée", "Valeur": year_min_val},
                {"Paramètre": "Année max observée", "Valeur": year_max_val},
                {
                    "Paramètre": "Provinces distinctes",
                    "Valeur": int(_surveillance_clean_text_series(df_scope[COL_PROV]).nunique()) if COL_PROV in df_scope.columns else 0,
                },
                {
                    "Paramètre": "Zones de santé distinctes",
                    "Valeur": int(_surveillance_clean_text_series(df_scope[COL_ZS]).nunique()) if COL_ZS in df_scope.columns else 0,
                },
            ]
            return pd.DataFrame(rows)

        export_traceability = _build_export_traceability_table(df_f)
        export_quality_summary = standard_data_quality_summary(df_f)
        export_qc_flags = qc_flags(df_f)
        export_qc_resume = (
            export_qc_flags["flag"].value_counts().rename_axis("Flag").reset_index(name="Occurrences")
            if not export_qc_flags.empty else pd.DataFrame(columns=["Flag", "Occurrences"])
        )
        export_duplicates = duplicate_candidates_table(df_f)
        export_risk_group = COL_ZS if COL_ZS in df_f.columns and df_f[COL_ZS].notna().any() else (
            COL_PROV if COL_PROV in df_f.columns and df_f[COL_PROV].notna().any() else None
        )
        export_risk_score = (
            build_operational_risk_score(
                df_f,
                group_col=export_risk_group,
                week_col="YW" if "YW" in df_f.columns else COL_WEEK,
                recent_weeks=4,
                threshold_days=int(seuil_jours),
            )
            if export_risk_group else pd.DataFrame()
        )
        export_alerts = (
            build_weekly_alerts(
                df_f,
                export_risk_group,
                week_col="YW" if "YW" in df_f.columns else COL_WEEK,
                baseline_weeks=3,
                min_baseline_periods=2,
                min_cases=10,
                alert_ratio=1.5,
            )
            if export_risk_group and (("YW" in df_f.columns) or (COL_WEEK in df_f.columns)) else pd.DataFrame()
        )
        export_clusters = (
            build_spatiotemporal_cluster_table(
                df_f,
                group_cols=[c for c in [COL_PROV, COL_ZS] if c in df_f.columns],
                week_col="YW" if "YW" in df_f.columns else COL_WEEK,
                recent_weeks=2,
                previous_weeks=4,
                min_recent_cases=5,
                growth_ratio=1.5,
            )
            if (("YW" in df_f.columns) or (COL_WEEK in df_f.columns)) else pd.DataFrame()
        )

        export_completeness = pd.DataFrame()
        export_completeness_by = None
        export_required_fields = [
            COL_PROV, COL_ZS, COL_AS, "YW", COL_WNUM, COL_SEX, COL_AGE,
            COL_UNIT, DATE_ONSET, COL_PREL, COL_TDR, COL_TDRR, COL_HOSP,
            COL_ISSUE, COL_CLASS,
        ]
        for group_col in [COL_PROV, COL_ZS, "YW", COL_WNUM]:
            if group_col in df_f.columns and df_f[group_col].notna().any():
                export_completeness = completeness_table(df_f, export_required_fields, by=group_col)
                if not export_completeness.empty:
                    export_completeness_by = group_col
                    break

        export_base_name = f"{str(disease_key).strip().lower()}_filtre"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cas exportés", format_metric_value(len(df_f)))
        m2.metric("Colonnes", format_metric_value(len(df_f.columns)))
        m3.metric(
            "Provinces distinctes",
            format_metric_value(int(_surveillance_clean_text_series(df_f[COL_PROV]).nunique()) if COL_PROV in df_f.columns else 0),
        )
        m4.metric(
            "ZS distinctes",
            format_metric_value(int(_surveillance_clean_text_series(df_f[COL_ZS]).nunique()) if COL_ZS in df_f.columns else 0),
        )

        info1, info2 = st.columns([0.9, 1.1])
        with info1:
            st.markdown("**Traçabilité de l’export**")
            st_dataframe_safe(export_traceability, height=360)
        with info2:
            st.markdown("**Résumé qualité inclus dans le pack d’export**")
            if not export_quality_summary.empty:
                st_dataframe_safe(export_quality_summary, height=360)
            else:
                st.info("Aucun résumé qualité n’est disponible pour le périmètre filtré.")

        with st.expander("Prévisualiser les contenus additionnels du pack d’export", expanded=False):
            if not export_qc_resume.empty:
                st.markdown("**Résumé des incohérences détectées**")
                st_dataframe_safe(export_qc_resume, height=260)
            else:
                st.caption("Aucune incohérence détectée par `qc_flags` sur le périmètre filtré.")

            if not export_duplicates.empty:
                st.markdown("**Doublons potentiels**")
                st_dataframe_safe(export_duplicates.head(100), height=260)
            else:
                st.caption("Aucun doublon potentiel n’a été identifié.")

            if export_completeness_by and not export_completeness.empty:
                st.markdown(f"**Complétude par {export_completeness_by}**")
                st_dataframe_safe(export_completeness.head(100), height=260)
            else:
                st.caption("Aucun tableau de complétude additionnel n’a pu être préparé pour l’export.")

            if not export_risk_score.empty:
                st.markdown(f"**Score de risque opérationnel ({export_risk_group})**")
                st_dataframe_safe(export_risk_score.head(100), height=300)
            else:
                st.caption("Aucun score de risque opérationnel n’a pu être préparé pour l’export.")

            if not export_alerts.empty:
                st.markdown("**Alertes hebdomadaires automatiques**")
                st_dataframe_safe(export_alerts.tail(100), height=300)

            if not export_clusters.empty:
                st.markdown("**Clusters spatio-temporels récents**")
                st_dataframe_safe(export_clusters.head(100), height=300)

        st.markdown("**Aperçu de la line list filtrée**")
        st_dataframe_safe(df_f, height=420)

        export_mode = st.radio(
            "Type d’export",
            ["Pack qualité + line list", "Line list uniquement"],
            index=0,
            horizontal=True,
            key="qualite_export_mode",
        )

        dl1, dl2, dl3 = st.columns([1, 1, 1])
        with dl1:
            st.download_button(
                "Télécharger CSV line list",
                data=df_to_csv_bytes(df_f),
                file_name=f"{export_base_name}.csv",
                mime="text/csv",
                key="dl_quality_export_csv_ll",
            )
        with dl2:
            if not export_qc_flags.empty:
                st.download_button(
                    "Télécharger QC flags (CSV)",
                    data=export_qc_flags.to_csv(index=False).encode("utf-8"),
                    file_name=f"{export_base_name}_qc_flags.csv",
                    mime="text/csv",
                    key="dl_quality_export_csv_qc",
                )
        with dl3:
            if not export_duplicates.empty:
                st.download_button(
                    "Télécharger doublons (CSV)",
                    data=export_duplicates.to_csv(index=False).encode("utf-8"),
                    file_name=f"{export_base_name}_doublons.csv",
                    mime="text/csv",
                    key="dl_quality_export_csv_dup",
                )

        try:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_f.to_excel(writer, sheet_name="LL_filtre", index=False)

                if export_mode == "Pack qualité + line list":
                    export_traceability.to_excel(writer, sheet_name="Traceabilite", index=False)
                    export_quality_summary.to_excel(writer, sheet_name="Resume_qualite", index=False)
                    if not export_qc_resume.empty:
                        export_qc_resume.to_excel(writer, sheet_name="QC_resume", index=False)
                    if not export_qc_flags.empty:
                        export_qc_flags.to_excel(writer, sheet_name="QC_detail", index=False)
                    if not export_duplicates.empty:
                        export_duplicates.to_excel(writer, sheet_name="Doublons", index=False)
                    if not export_completeness.empty:
                        export_completeness.to_excel(writer, sheet_name="Completude", index=False)
                    if not export_risk_score.empty:
                        export_risk_score.to_excel(writer, sheet_name="Score_risque", index=False)
                    if not export_alerts.empty:
                        export_alerts.to_excel(writer, sheet_name="Alertes", index=False)
                    if not export_clusters.empty:
                        export_clusters.to_excel(writer, sheet_name="Clusters", index=False)

            excel_name = (
                f"{export_base_name}_pack_qualite.xlsx"
                if export_mode == "Pack qualité + line list"
                else f"{export_base_name}.xlsx"
            )
            st.download_button(
                "Télécharger Excel",
                data=buffer.getvalue(),
                file_name=excel_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_quality_export_xlsx",
            )
        except Exception:
            st.info("Exportation Excel indisponible (openpyxl manquant ?).")
        
    # =========================
    # TAB 7 — Labo / qualité / signaux
    # =========================
    st.divider()
    render_section_title(6, "Qualité des données et alertes de gestion")
    if IDSR_MODE:
        render_absence_narrative("idsr_line_list")
    else:
        render_tab_narrative("qualite")
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Détecter incohérences, problèmes de complétude, goulots labo, et signaux d’alerte.
        
            **📖 Sections**
            - **Indicateurs rapides** : 3–5 KPI qualité/action
            - **QC Flags** : incohérences (dates, TDR, âge…)
            - **Complétude champs clés** : % remplissage par site
            - **Cascade labo** : cas → prélèvement → TDR → résultat valide → positif
            - **Alertes tendance** : hausse inhabituelle vs baseline simple
        
            **⚠️ Points d’attention**
            - Un signal ≠ confirmation d’épidémie : déclenche une investigation terrain.
            - Les % de cascade sont calculés sur une logique *entonnoir* (séquentielle).
            """,
            expanded=False
        )
        
        st.subheader("Contrôle qualité des données et alertes opérationnelles de surveillance")
        
        # -------- Helpers (robustes) --------
        def _get_pct_from_cascade(casc: pd.DataFrame, key: str) -> float:
            """Récupère le % de la première ligne dont Étape contient key (robuste aux libellés)."""
            if casc is None or casc.empty or "Étape" not in casc.columns or "%" not in casc.columns:
                return np.nan
            m = casc.loc[casc["Étape"].astype(str).str.contains(key, regex=False, na=False), "%"]
            return float(m.iloc[0]) if len(m) else np.nan
        
        def _safe_num(x):
            try:
                return float(x)
            except Exception:
                return np.nan
        
        # ==========================================================
        # 0) Indicateurs rapides (KPI)
        # ==========================================================
        n_total = len(df_f)
        
        kpi = compute_indicators(df_f)
        casc_global = cascade_metrics(df_f) if n_total else pd.DataFrame()
        has_tdr_chain = COL_TDR in df_f.columns and df_f[COL_TDR].notna().any()
        
        # KPI “qualité TDR” (sur cascade)
        kpi_incoh_res_wo_tdr = _get_pct_from_cascade(casc_global, "Résultat renseigné mais TDR_realise != Oui")
        kpi_status_in_result = _get_pct_from_cascade(casc_global, "Statut saisi dans TDR_Resultat")
        
        # ✅ 7 colonnes (ajout hospitalisation)
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        
        c1.metric(
            "Cas (n)",
            f"{kpi['n_cases']:,}".replace(",", " "),
            help="Nombre total de cas après application des filtres (Province/ZS/SE, etc.)."
        )
        
        c2.metric(
            "% prélèvement",
            "-" if np.isnan(kpi["prelev_pct"]) else f"{kpi['prelev_pct']:.1f}",
            help=f"Prélèvement=Oui / Tous les cas filtrés. n={kpi.get('prelev_num', 0)}/{kpi.get('prelev_den', kpi.get('n_cases', 0))}"
        )
        
        c3.metric(
            "Couverture TDR (%)" if has_tdr_chain else "Couverture test (%)",
            "-" if np.isnan(kpi["tdr_pct"]) else f"{kpi['tdr_pct']:.1f}",
            help=(
                f"TDR_realise=Oui / Tous les cas filtrés. n={kpi.get('tdr_num', 0)}/{kpi.get('tdr_den', kpi.get('n_cases', 0))}"
                if has_tdr_chain
                else f"Tests documentés / Tous les cas filtrés. n={kpi.get('tdr_num', 0)}/{kpi.get('tdr_den', kpi.get('n_cases', 0))}"
            )
        )
        
        # ✅ Positivité
        pos_label = "-"
        if not np.isnan(kpi["pos_pct"]):
            pos_label = f"{kpi['pos_pct']:.1f}"
        c4.metric(
            "Positivité TDR" if has_tdr_chain else "Positivité test",
            pos_label,
            help=(
                (
                    "Positifs / (Positifs + Négatifs) parmi les TDR interprétables "
                    "(TDR_realise=Oui ET résultat valide Pos/Nég). "
                )
                if has_tdr_chain
                else "Positifs / (Positifs + Négatifs) parmi les résultats labo interprétables. "
            ) + f"n={kpi.get('pos_num', 0)}/{kpi.get('pos_den', 0)}"
        )
        
        # 🆕 Taux hospitalisation
        c5.metric(
            "Hospitalisation (%)",
            "-" if np.isnan(kpi["hosp_pct"]) else f"{kpi['hosp_pct']:.1f}",
            help=f"Hospitalisation=Oui / Tous les cas filtrés. n={kpi.get('hosp_num', 0)}/{kpi.get('hosp_den', kpi.get('n_cases', 0))}"
        )
        
        c6.metric(
            "CFR (%)",
            "-" if np.isnan(kpi["cfr_pct"]) else f"{kpi['cfr_pct']:.2f}",
            help=f"Décès / Tous les cas filtrés. n={kpi.get('n_deaths', 0)}/{kpi.get('n_cases', 0)}"
        )
        
        # % invalides
        inv_label = "-"
        if "invalid_pct" in kpi and not np.isnan(kpi["invalid_pct"]):
            inv_label = f"{kpi['invalid_pct']:.1f}"
        c7.metric(
            "% TDR invalides" if has_tdr_chain else "% tests invalides",
            inv_label,
            help=(
                (
                    "Invalides (ex: INBA/bande absente) / TDR réalisés (TDR_realise=Oui). "
                    if has_tdr_chain
                    else "Invalides / tests documentés. "
                )
                + f"n={kpi.get('invalid_num', 0)}/{kpi.get('invalid_den', 0)}"
            )
        )
        
        # Alertes qualité tests (si dispo)
        if not np.isnan(kpi_incoh_res_wo_tdr) or not np.isnan(kpi_status_in_result):
            with st.expander("📌 Signaux qualité tests (données)", expanded=False):
                if not np.isnan(kpi_incoh_res_wo_tdr):
                    st.write(f"- **% Résultat renseigné mais TDR_realise ≠ Oui / statut test absent**: **{kpi_incoh_res_wo_tdr:.1f}%**")
                if not np.isnan(kpi_status_in_result):
                    st.write(f"- **% Statut saisi dans la colonne de résultat** (ex: non réalisé/non prélevé): **{kpi_status_in_result:.1f}%**")
        
        with st.expander("🔎 Détail cascade labo (entonnoir) + incohérences", expanded=False):
            st_dataframe_safe(casc_global)

        # ==========================================================
        # 0a) Score de risque operationnel par zone/province
        # ==========================================================
        risk_group_options = [
            c for c in [COL_ZS, COL_PROV, COL_AS]
            if c in df_f.columns and df_f[c].notna().any()
        ]
        if risk_group_options:
            with st.expander("Priorisation operationnelle par zone/province", expanded=True):
                r1, r2, r3 = st.columns([1.15, 0.95, 0.95])
                with r1:
                    risk_group_col = st.selectbox(
                        "Niveau de priorisation",
                        options=risk_group_options,
                        key="operational_risk_group_col",
                    )
                with r2:
                    risk_recent_weeks = st.number_input(
                        "Semaines recentes",
                        min_value=2,
                        max_value=8,
                        value=4,
                        step=1,
                        key="operational_risk_recent_weeks",
                    )
                with r3:
                    risk_topn = st.slider(
                        "Top priorites",
                        min_value=5,
                        max_value=50,
                        value=20,
                        step=5,
                        key="operational_risk_topn",
                    )

                risk_tbl = build_operational_risk_score(
                    df_f,
                    group_col=risk_group_col,
                    week_col="YW" if "YW" in df_f.columns else COL_WEEK,
                    recent_weeks=int(risk_recent_weeks),
                    threshold_days=int(seuil_jours),
                )
                if risk_tbl.empty:
                    render_absence_narrative("risk")
                else:
                    risk_view = risk_tbl.head(int(risk_topn)).copy()
                    k_r1, k_r2, k_r3 = st.columns(3)
                    k_r1.metric("Groupes classes", f"{len(risk_tbl):,}".replace(",", " "))
                    k_r2.metric("Priorite tres elevee", str(int((risk_tbl["Priorite"] == "Tres elevee").sum())))
                    k_r3.metric("Score max", f"{pd.to_numeric(risk_tbl['Score_risque'], errors='coerce').max():.1f}")

                    left_risk, right_risk = st.columns([1.15, 1.35])
                    with left_risk:
                        st.dataframe(risk_view, width="stretch", height=520, hide_index=True)
                    with right_risk:
                        plot_risk = risk_view.sort_values("Score_risque", ascending=True)
                        fig_risk = px.bar(
                            plot_risk,
                            x="Score_risque",
                            y=risk_group_col,
                            orientation="h",
                            color="Priorite",
                            title="Score de risque operationnel",
                        )
                        fig_risk.update_layout(xaxis_title="Score 0-100", yaxis_title=risk_group_col)
                        fig_risk = apply_plotly_value_annotations(fig_risk, annot_vals)
                        st.plotly_chart(fig_risk, width="stretch", key="operational_risk_chart")

                    csv_risk = risk_tbl.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Telecharger score de risque (CSV)",
                        data=csv_risk,
                        file_name="score_risque_operationnel.csv",
                        mime="text/csv",
                        key="download_operational_risk_score",
                    )

        # ==========================================================
        # 0b) Résumé standard qualité / délais / disponibilité des champs
        # ==========================================================
        with st.expander("🔎 Résumé standard qualité / délais / disponibilité des champs", expanded=False):
            qsum = standard_data_quality_summary(df_f)
            if not qsum.empty:
                st_dataframe_safe(qsum)
            dsum = build_standard_delay_summary(df_f)
            if not dsum.empty:
                st.markdown("**Résumé standard des délais**")
                st_dataframe_safe(dsum)
            dup_tbl = duplicate_candidates_table(df_f)
            if not dup_tbl.empty:
                st.markdown("**Doublons potentiels à vérifier**")
                st_dataframe_safe(dup_tbl.head(100), height=320)
            fields_matrix = build_recommended_fields_matrix(df_f)
            if not fields_matrix.empty:
                st.markdown("**Disponibilité des champs standards recommandés**")
                bloc_sel = st.selectbox("Filtrer la matrice par bloc", ["Tous"] + sorted(fields_matrix["Bloc"].dropna().unique().tolist()), index=0, key="fields_matrix_bloc")
                if bloc_sel != "Tous":
                    fields_matrix = fields_matrix[fields_matrix["Bloc"] == bloc_sel]
                st_dataframe_safe(fields_matrix, height=360)
        
        # ==========================================================
        # 1) QC Flags (incohérences)
        # ==========================================================
        with st.expander("🔎 Incohérences (QC Flags)", expanded=False):
        
        
            flags = qc_flags(df_f)
            if flags.empty:
                st.success("Aucune incohérence n’a été détectée selon les règles de contrôle actuellement appliquées.")
            else:
                # Résumé
                resume = flags["flag"].value_counts().reset_index()
                resume.columns = ["Flag", "Occurrences"]
                st_dataframe_safe(resume)
        
                # Filtre par flag
                flag_list = sorted(flags["flag"].dropna().unique().tolist())
                flag_sel = st.selectbox("Filtrer le détail par type d’incohérence", ["Tous"] + flag_list, index=0)
        
                # Détail (merge + colonnes utiles)
                cols_show = [c for c in [
                    "Nom_complet", COL_PROV, COL_ZS, COL_AS, COL_SEX, COL_AGE, COL_UNIT,
                    "YW", COL_WNUM, DATE_ONSET, DATE_ADM, DATE_PREL,
                    COL_PREL, COL_TDR, COL_TDRR, COL_HOSP, COL_ISSUE, COL_CLASS
                ] if c in df_f.columns]
        
                detail = flags.merge(df_f.reset_index().rename(columns={"index": "row_id"}), on="row_id", how="left")
        
                if flag_sel != "Tous":
                    detail = detail[detail["flag"] == flag_sel]
        
                st.caption("Détail des lignes concernées (filtré, maximum 500 lignes)")
                st.dataframe(detail[["flag"] + cols_show].head(500), width="stretch", height=420)
        
        # ==========================================================
        # 2) Complétude des champs clés
        # ==========================================================
        with st.expander("🔎 Complétude des champs clés", expanded=False):
        
            champs_cles = [
                COL_PROV, COL_ZS, COL_AS, "YW", COL_WNUM,
                COL_SEX, COL_AGE, COL_UNIT, DATE_ONSET,
                COL_PREL, COL_TDR, COL_TDRR, COL_HOSP,
                COL_ISSUE, COL_CLASS
            ]
        
            group_choices = [c for c in [COL_PROV, COL_ZS, "YW", COL_WNUM] if c in df_f.columns]
            group_for_comp = st.selectbox("Analyser la complétude par", group_choices, index=0 if group_choices else 0)
        
            comp = completeness_table(df_f, champs_cles, by=group_for_comp) if group_choices else pd.DataFrame()
        
            if comp.empty:
                render_absence_narrative("quality")
            else:
                st_dataframe_safe(comp, height=520)
        
                # Bar chart plus lisible: top N pires scores
                topn = st.slider("Nombre de groupes les moins complets à afficher", min_value=10, max_value=80, value=25, step=5)
                comp_plot = comp.sort_values("score_completude_%").head(topn)
        
                figc = px.bar(
                    comp_plot,
                    x=group_for_comp,
                    y="score_completude_%",
                    title=f"Score complétude (%) – {topn} groupes les moins complets (par {group_for_comp})"
                )
                figc.update_layout(xaxis_tickangle=-45, yaxis=dict(range=[0, 100]))
                figc = apply_plotly_value_annotations(figc, annot_vals)
                st.plotly_chart(figc, width="stretch")
        
        
        # ==========================================================
        # 3) Cascade prélèvement → TDR → résultat → positif
        # ==========================================================
        with st.expander("🔎 Cascade prélèvement → TDR → résultat → positif", expanded=False):
        
            cascad = cascade_metrics(df_f) if n_total else pd.DataFrame()
            if cascad.empty:
                render_absence_narrative("quality")
            else:
                st_dataframe_safe(cascad)
        
            # Cascade par province (résumé robuste)
            if COL_PROV in df_f.columns and n_total:
                st.caption("Cascade par province (résumé)")
        
                rows = []
                for prov, sub in df_f.groupby(COL_PROV, dropna=False):
                    c = cascade_metrics(sub)
                    rows.append([
                        prov,
                        len(sub),
                        _get_pct_from_cascade(c, "Prélèvement=Oui"),
                        _get_pct_from_cascade(c, "TDR réalisé=Oui"),
                        _get_pct_from_cascade(c, "Résultat valide"),
                        _get_pct_from_cascade(c, "Positifs"),
                        _get_pct_from_cascade(c, "Résultat renseigné mais TDR_realise != Oui"),
                    ])
        
                df_cas = pd.DataFrame(
                    rows,
                    columns=[COL_PROV, "n", "% prélèvement", "% TDR", "% résultat valide", "% positif", "% incoh TDR"]
                )
        
                sort_col = st.selectbox(
                    "Trier par",
                    ["n", "% prélèvement", "% TDR", "% résultat valide", "% positif", "% incoh TDR"],
                    index=0
                )
                df_cas_sorted = df_cas.sort_values(sort_col, ascending=False if sort_col == "n" else True)
        
                st_dataframe_safe(df_cas_sorted, height=420)
        
        
        # ==========================================================
        # 4) Alertes tendance (hausse vs baseline simple)
        # ==========================================================
        with st.expander("🔎 Alertes tendance (hausse vs baseline simple)", expanded=False):
            alert_group_choices = [c for c in [COL_PROV, COL_ZS] if c in df_f.columns]
            if not alert_group_choices:
                render_absence_narrative("geo")
                alert_group = None
                alerts = pd.DataFrame()
            else:
                alert_group = st.selectbox("Regrouper les alertes par", alert_group_choices, index=0)
                min_alert_cases = st.number_input(
                    "Cas minimum pour signal",
                    min_value=1,
                    max_value=500,
                    value=10,
                    step=1,
                    key="quality_alert_min_cases",
                )
                alert_ratio_quality = st.number_input(
                    "Ratio alerte vs baseline",
                    min_value=1.0,
                    max_value=10.0,
                    value=1.5,
                    step=0.1,
                    key="quality_alert_ratio",
                )
                alerts = build_weekly_alerts(
                    df_f,
                    alert_group,
                    week_col="YW",
                    baseline_weeks=3,
                    min_baseline_periods=2,
                    min_cases=int(min_alert_cases),
                    alert_ratio=float(alert_ratio_quality),
                ) if "YW" in df_f.columns else pd.DataFrame()
        
            if alerts.empty:
                render_absence_narrative("alerts")
            else:
                # Dernière semaine observée
                last_yw = alerts["YW"].dropna().max()
                st.caption(
                    f"Dernière semaine observée : {last_yw}. "
                    "Les signaux ci-dessous servent à prioriser la vérification, pas à confirmer seuls une épidémie."
                )
        
                last = alerts[alerts["YW"] == last_yw].copy()
        
                # sécurité var_% (éviter inf)
                if "Cas_prev" in last.columns and "Cas" in last.columns:
                    last["Cas_prev"] = last["Cas_prev"].fillna(0)
                    last["var_%"] = np.where(
                        last["Cas_prev"] > 0,
                        (last["Cas"] - last["Cas_prev"]) / last["Cas_prev"] * 100.0,
                        np.nan
                    )
        
                # classement: signal d’abord, puis plus gros volumes
                last["signal"] = last["signal"].fillna(False)
                last = last.sort_values(["signal", "Cas"], ascending=[False, False])
        
                cols_out = [c for c in [alert_group, "YW", "Cas", "Cas_prev", "var_%", "baseline", "ratio_baseline", "signal_level", "signal"] if c in last.columns]
                st_dataframe_safe(last[cols_out], height=520)
        
                # Top signaux
                sig = last[last["signal"] == True].head(30)
                if len(sig):
                    figa = px.bar(sig, x=alert_group, y="Cas", title=f"Signaux (semaine {last_yw}) – top 30")
                    figa.update_layout(xaxis_tickangle=-45)
                    figa = apply_plotly_value_annotations(figa, annot_vals)
                    st.plotly_chart(figa, width="stretch")
                else:
                    st.success("Aucun signal n’a été détecté avec les seuils actuellement définis (baseline × 1,5 et cas ≥ 10).")

                cluster_quality = build_spatiotemporal_cluster_table(
                    df_f,
                    group_cols=[c for c in [COL_PROV, COL_ZS] if c in df_f.columns],
                    week_col="YW",
                    recent_weeks=2,
                    previous_weeks=4,
                    min_recent_cases=max(5, int(min_alert_cases // 2)),
                    growth_ratio=float(alert_ratio_quality),
                ) if "YW" in df_f.columns else pd.DataFrame()
                if not cluster_quality.empty:
                    st.markdown("**Clusters recents a investiguer**")
                    st_dataframe_safe(cluster_quality.head(100), height=360)

