"""Render the epidemiological profile tab."""

from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


def render_profile_tab(ctx: dict) -> None:
    """Render the epidemiological profile tab."""
    globals().update(ctx)
    if IDSR_MODE:
        render_absence_narrative("idsr_line_list")
    else:
        render_tab_narrative("profil")
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Identifier les groupes les plus touchés.
        
            **📖 Interprétation**
            - Répartition **sexe** : différences d’exposition ou d’accès aux soins.
            - Répartition **âge** : identifie les groupes vulnérables/à risque.
            - **Pyramide âge/sexe** : profil de transmission (domicile, école, activités, etc.).
        
            **⚠️ Points d’attention**
            - Vérifier la complétude de l’âge et du sexe : beaucoup de “Inconnu” biaise la lecture.
            """,
            expanded=False
        )
        
      
        st.divider()

        st.subheader("Contrôle qualité des variables d’âge")

        # --- Indicateurs rapides ---
        n_total = len(df_f)

        # Manquants âge: on considère Age OU une tranche (Tranche_age/Tranche_age_en_ans)
        has_age_num = (COL_AGE in df_f.columns)
        has_tr4 = (COL_AGEG2 in df_f.columns)
        has_tr5 = (COL_AGEG in df_f.columns)

        age_num_na = df_f[COL_AGE].isna() if has_age_num else pd.Series([True]*n_total, index=df_f.index)
        tr4_na = df_f[COL_AGEG2].isna() if has_tr4 else pd.Series([True]*n_total, index=df_f.index)
        tr5_na = df_f[COL_AGEG].isna() if has_tr5 else pd.Series([True]*n_total, index=df_f.index)

        missing_age_mask = age_num_na & tr4_na & tr5_na
        pct_age_missing = float(missing_age_mask.mean() * 100.0) if n_total else 0.0

        # Unité incohérente
        incoh_mask = pd.Series([False]*n_total, index=df_f.index)
        if COL_UNIT in df_f.columns and df_f[COL_UNIT].notna().any():
            u = df_f[COL_UNIT].astype("string").str.lower().str.strip()
            ok = (
                u.str.contains(AGE_UNIT_YEAR_PATTERN, na=False)
                | u.str.contains(AGE_UNIT_MONTH_PATTERN, na=False)
                | u.str.contains(AGE_UNIT_WEEK_PATTERN, na=False)
                | u.str.contains(AGE_UNIT_DAY_PATTERN, na=False)
            )
            incoh_mask = u.notna() & (~ok)
        pct_unit_incoh = float(incoh_mask.mean() * 100.0) if n_total else 0.0

        # Âges extrêmes (convertis en années quand possible)
        years = infer_age_years_generic(df_f) if has_age_num else pd.Series([np.nan]*n_total, index=df_f.index)
        extreme_mask = years.notna() & ((years < 0) | (years > 110))
        pct_extreme = float(extreme_mask.mean() * 100.0) if n_total else 0.0

        # --- Affichage KPI ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Âge manquant", f"{pct_age_missing:.1f}%")
        m2.metric("Unité âge incohérente", f"{pct_unit_incoh:.1f}%")
        m3.metric("Âges extrêmes (<0 ou >110 ans)", f"{pct_extreme:.1f}%")
        m4.metric("N (après filtres)", f"{n_total:,}".replace(",", " "))

        with st.expander("Détails qualité (unités, âges extrêmes)"):
            if COL_UNIT in df_f.columns:
                unit_dist = (
                    df_f[COL_UNIT].astype("string").fillna("NA").str.lower().str.strip()
                    .value_counts().reset_index()
                )
                unit_dist.columns = ["Unite_age (valeur)", "N"]
                st.dataframe(unit_dist, width="stretch", height=260)
            else:
                render_absence_narrative("profile")

            if extreme_mask.any():
                show_cols = [c for c in [COL_PROV, COL_ZS, COL_AGE, COL_UNIT, DATE_ONSET, DATE_ADM, DATE_NOTIF] if c in df_f.columns]
                df_ext = df_f.loc[extreme_mask, show_cols].copy().head(50)
                df_ext.insert(0, "Age_en_ans_estime", years.loc[extreme_mask].head(50).round(2).values)
                st.warning("Exemples de valeurs extrêmes (maximum 50) à vérifier et corriger si nécessaire.")
                st.dataframe(df_ext, width="stretch", height=320)
            else:
                st.success("Aucune valeur d’âge extrême n’a été détectée selon les règles en vigueur.")

      
    # =========================
    # TAB 4B: Analyse descriptive standard
    # =========================
    st.divider()
    render_section_title(3, "Analyse descriptive selon le modèle Temps-Lieu-Personne")
    if IDSR_MODE:
        render_absence_narrative("idsr_line_list")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : présenter une analyse descriptive conforme à une logique standard.

            **📖 Structure **
            - Vue d'ensemble
            - Personne
            - Lieu
            - Laboratoire
            - Tableaux descriptifs complémentaires
            """,
            expanded=False
        )

        st.subheader("Résumé automatisé conforme au langage de surveillance")
        st.info(build_who_narrative_summary(df_f))

        st.subheader("1. Situation générale")
        st_dataframe_safe(build_global_summary_table(df_f))
                
        st.divider()
        st.subheader("2. Dimension personne — tableaux détaillés et structure avancée")
        st.caption("Les visuels rapides sexe, âge et pyramide de synthèse sont regroupés sur la page d’accueil. Ici, l’accent est mis sur les tableaux analytiques détaillés.")
        a1, a2 = st.columns(2)
        with a1:
            if COL_SEX in df_f.columns:
                sex_tbl = build_frequency_table(df_f, COL_SEX)
                st.markdown("**Table de fréquence par sexe**")
                st_dataframe_safe(sex_tbl)
            else:
                render_absence_narrative("profile")
        with a2:
            age_display_col = None
            if COL_AGEG2 in df_f.columns and df_f[COL_AGEG2].notna().any():
                age_display_col = COL_AGEG2
            elif COL_AGEG in df_f.columns and df_f[COL_AGEG].notna().any():
                age_display_col = COL_AGEG
            if age_display_col is not None:
                age_tbl = build_frequency_table(df_f, age_display_col)
                st.markdown(f"**Table de fréquence par {age_display_col}**")
                st_dataframe_safe(age_tbl)
            else:
                years = infer_age_years_generic(df_f)
                if years.notna().any():
                    age_num = pd.DataFrame({'Age_en_ans': years.dropna()})
                    st.markdown("**Résumé statistique de l’âge en années**")
                    st.dataframe(age_num.describe().T, width='stretch')
                else:
                    render_absence_narrative("profile")

        df_desc = df_f.copy()
        df_desc['Tranche_age_4cat_std'] = derive_age_4cat_generic(df_desc)
        df_desc['Tranche_age_5ans_std'] = derive_age_5yr_generic(df_desc)
        if use_custom_viz and HAS_CUSTOM_VIZ and age_col and COL_SEX in df_desc.columns and COL_PROV in df_desc.columns:
            st.markdown("**Structure âge-sexe détaillée par province**")
            fig = graphique_pyramide_age(
                df=df_desc,
                col_tranche=age_col,
                col_sexe=COL_SEX,
                col_valeur=COL_UNIT if COL_UNIT in df_desc.columns else COL_SEX,
                valeurs_neg=['Masculin', 'Homme', 'M'],
                titre='Pyramides âge-sexe par province',
                seuil_min=10,
                croissant=False,
                afficher_signe_negatif_dans_label=False,
                facette_col=COL_PROV,
                annot=annot_vals,
                taille_fig=(1500, 900),
                return_fig=True,
                couleur_contour_facette="#777772"
            )
            st_plot(fig, key='oms_pyr_faceted_prov')
        else:
            render_absence_narrative("profile")

        st.divider()
        st.subheader("3. Dimension lieu — répartition par province, zone de santé et aire de santé")
        geo_cols = [c for c in [COL_PROV, COL_ZS, COL_AS] if c in df_f.columns]
        if geo_cols:
            geo_choice = st.selectbox('Niveau géographique d’analyse', geo_cols, key='oms_geo_choice')
            top_n_geo = st.slider('Nombre de catégories à afficher', 5, 30, 15, key='oms_top_geo')
            geo_tbl = build_frequency_table(df_f, geo_choice, top_n=top_n_geo)
            fig = px.bar(geo_tbl, x=geo_choice, y='n', title=f'Répartition des cas par {geo_choice}')
            fig.update_layout(xaxis_tickangle=-45)
            fig = apply_plotly_value_annotations(fig, annot_vals)
            st.plotly_chart(fig, width='stretch', key='oms_geo_bar')
            st_dataframe_safe(geo_tbl)
        else:
            render_absence_narrative("geo")

        st.divider()
        st.subheader("4. Composante laboratoire — résumé opérationnel")
        lab_tbl = build_simple_lab_table(df_f)
        if not lab_tbl.empty:
            l1, l2 = st.columns([1, 1])
            with l1:
                st_dataframe_safe(lab_tbl)
            with l2:
                fig = px.bar(lab_tbl, x='Indicateur labo', y='n', title='Résumé des indicateurs de laboratoire')
                fig.update_layout(xaxis_tickangle=-45)
                fig = apply_plotly_value_annotations(fig, annot_vals)
                st.plotly_chart(fig, width='stretch', key='oms_lab_bar')

            weekly_lab = build_weekly_lab_summary(df_f)
            if not weekly_lab.empty:
                st.markdown("**Suivi hebdomadaire des tests valides, tests positifs et taux de positivité**")
                fig_lab_combo = go.Figure()
                fig_lab_combo.add_trace(
                    go.Bar(
                        x=weekly_lab["Semaine"],
                        y=weekly_lab["Tests valides"],
                        name="Tests valides",
                        marker_color="#4f81bd",
                    )
                )
                fig_lab_combo.add_trace(
                    go.Bar(
                        x=weekly_lab["Semaine"],
                        y=weekly_lab["Tests positifs"],
                        name="Tests positifs",
                        marker_color="#d97b16",
                    )
                )
                fig_lab_combo.add_trace(
                    go.Scatter(
                        x=weekly_lab["Semaine"],
                        y=weekly_lab["Positivité (%)"],
                        name="Positivité (%)",
                        mode="lines+markers",
                        line=dict(color="#b9353f", width=3),
                        marker=dict(size=8),
                        yaxis="y2",
                    )
                )
                fig_lab_combo.update_layout(
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
                st_plot(fig_lab_combo, key="lab_weekly_combo", annotate_values=False)
                with st.expander("Afficher la table hebdomadaire des indicateurs laboratoire", expanded=False):
                    st_dataframe_safe(weekly_lab, height=320)

            has_tdr_chain = COL_TDR in df_f.columns and df_f[COL_TDR].notna().any()
            coverage_label = "TDR réalisé (%)" if has_tdr_chain else "Tests documentés (%)"
            positivity_label = "Positivité TDR (%)" if has_tdr_chain else "Positivité labo (%)"

            if COL_PROV in df_f.columns:
                st.markdown("**Tableau provincial consolidé des indicateurs clés de surveillance**")
                prov_kpi = compute_group_indicators(df_f, COL_PROV).sort_values("Cas", ascending=False).head(15).copy()
                prov_kpi = _normalize_metric_alias_columns(prov_kpi)
                prov_kpi = prov_kpi.rename(
                    columns={
                        "Décès": "Décès",
                        "CFR_%": "CFR (%)",
                        "Prélèvement_%": "Prélèvement (%)",
                        "Hospitalisation_%": "Hospitalisation (%)",
                        "TDR_réalisé_%": coverage_label,
                        "Positivité_TDR_%": positivity_label,
                    }
                )
                st_dataframe_safe(prov_kpi, height=420)

            lab_detail_cols = [
                c for c in [
                    "TDR_Resultat",
                    "Resultat_labo",
                    "Type_de_prelevement",
                    "Nom_laboratoire",
                    "Etat_echantillon",
                    "Nombre_dose_recues",
                ]
                if c in df_f.columns and df_f[c].notna().any()
            ]
            lab_date_cols = [
                c for c in ["Date_prelevement", "Date_reception_labo", "Date_resultat", "Date_derniere_vaccination"]
                if c in df_f.columns
            ]

            if lab_detail_cols or lab_date_cols:
                st.markdown("**Profil détaillé des variables laboratoire et vaccination**")

                if lab_date_cols:
                    availability_rows = []
                    for col in lab_date_cols:
                        non_null = int(df_f[col].notna().sum())
                        availability_rows.append(
                            {
                                "Variable": col,
                                "Renseigné": non_null,
                                "%": round(non_null / max(len(df_f), 1) * 100.0, 1),
                            }
                        )
                    st_dataframe_safe(pd.DataFrame(availability_rows), height=220)

                if lab_detail_cols:
                    lab_profile_col = st.selectbox(
                        "Variable laboratoire / vaccination à décrire",
                        options=lab_detail_cols,
                        index=0,
                        key="profile_lab_detail_col",
                    )
                    lab_profile_tbl = build_frequency_table(df_f, lab_profile_col, top_n=20)
                    if not lab_profile_tbl.empty:
                        d1, d2 = st.columns([1, 1.25])
                        with d1:
                            st_dataframe_safe(lab_profile_tbl, height=320)
                        with d2:
                            fig_lab_detail = px.bar(
                                lab_profile_tbl.sort_values("n", ascending=False),
                                x=lab_profile_col,
                                y="n",
                                title=f"Répartition de {lab_profile_col}",
                            )
                            fig_lab_detail.update_layout(xaxis_tickangle=-45)
                            fig_lab_detail = apply_plotly_value_annotations(fig_lab_detail, annot_vals)
                            st.plotly_chart(fig_lab_detail, width="stretch", key="profile_lab_detail_chart")
        else:
            render_absence_narrative("profile")

        st.divider()
        st.subheader("5. Indicateurs standards stratifiés")
        st.caption(
            "Vue transversale standard des cas, décès, CFR et indicateurs de surveillance, "
            "applicable à toute line list standardisée."
        )

        strat_age_col = pick_age_col(df_f)
        strat_candidates = []
        for c in [COL_SEX, strat_age_col, COL_PROV, COL_ZS, COL_AS, COL_CLASS, "Type_de_prelevement", "Nom_laboratoire"]:
            if c and c in df_f.columns and df_f[c].notna().any() and c not in strat_candidates:
                strat_candidates.append(c)

        if strat_candidates:
            metric_map = {
                "Cas": "Cas",
                "Décès": "Décès",
                "CFR (%)": "CFR (%)",
                "Prélèvement (%)": "Prélèvement (%)",
                "Hospitalisation (%)": "Hospitalisation (%)",
                coverage_label: coverage_label,
                positivity_label: positivity_label,
            }

            s_cfg1, s_cfg2, s_cfg3 = st.columns([1.15, 1.15, 0.9])
            with s_cfg1:
                strat_choice = st.selectbox(
                    "Variable de stratification",
                    options=strat_candidates,
                    key="std_strat_choice",
                )
            with s_cfg2:
                strat_metric_label = st.selectbox(
                    "Indicateur à classer",
                    options=list(metric_map.keys()),
                    index=0,
                    key="std_strat_metric",
                )
            with s_cfg3:
                strat_topn = st.slider(
                    "Top modalités",
                    min_value=5,
                    max_value=30,
                    value=15,
                    step=1,
                    key="std_strat_topn",
                )

            strat_tbl = compute_group_indicators(df_f, strat_choice).copy()
            strat_tbl = _normalize_metric_alias_columns(strat_tbl)
            strat_tbl = strat_tbl.rename(
                columns={
                    "Décès": "Décès",
                    "CFR_%": "CFR (%)",
                    "Prélèvement_%": "Prélèvement (%)",
                    "Hospitalisation_%": "Hospitalisation (%)",
                    "TDR_réalisé_%": coverage_label,
                    "Positivité_TDR_%": positivity_label,
                }
            )

            if not strat_tbl.empty:
                total_cases_strat = pd.to_numeric(strat_tbl["Cas"], errors="coerce").sum()
                strat_tbl["Part des cas (%)"] = np.where(
                    total_cases_strat > 0,
                    (pd.to_numeric(strat_tbl["Cas"], errors="coerce") / total_cases_strat) * 100.0,
                    np.nan,
                ).round(1)

                sort_col = metric_map[strat_metric_label]
                strat_view = (
                    strat_tbl.sort_values(sort_col, ascending=False, na_position="last")
                    .head(int(strat_topn))
                    .copy()
                )

                s_tbl, s_fig = st.columns([1.05, 1.35])
                with s_tbl:
                    st.dataframe(strat_view, width="stretch", height=430, hide_index=True)
                with s_fig:
                    plot_df = strat_view.sort_values(sort_col, ascending=True, na_position="last")
                    fig_strat = px.bar(
                        plot_df,
                        x=sort_col,
                        y=strat_choice,
                        orientation="h",
                        text=sort_col,
                        title=f"{strat_metric_label} par {strat_choice}",
                        color=sort_col,
                        color_continuous_scale=["#e7f1df", "#2d7d46"],
                    )
                    fig_strat.update_layout(
                        coloraxis_showscale=False,
                        xaxis_title=strat_metric_label,
                        yaxis_title=strat_choice,
                    )
                    fig_strat = apply_plotly_value_annotations(fig_strat, annot_vals)
                    st.plotly_chart(fig_strat, width="stretch", key="std_strat_chart")
            else:
                render_absence_narrative("profile")
        else:
            render_absence_narrative("profile")

        st.divider()
        st.subheader("6. Tableaux descriptifs des variables catégorielles")
        st.caption("Les analyses de délais sont centralisées dans l’onglet Surveillance afin d’éviter leur répétition ici.")
        default_cat_candidates = [
            COL_SEX,
            COL_PROV,
            COL_ZS,
            COL_AS,
            COL_AGEG2,
            COL_AGEG,
            COL_ISSUE,
            COL_PREL,
            COL_TDR,
            COL_TDRR,
            COL_HOSP,
            COL_DEHY,
            COL_CLASS,
            "Resultat_labo",
            "Type_de_prelevement",
            "Nom_laboratoire",
            "Etat_echantillon",
            "Nombre_dose_recues",
        ]
        cat_candidates = [c for c in default_cat_candidates if c in df_f.columns]
        extra_candidates = [c for c in df_f.columns if (not is_numeric_dtype(df_f[c])) and c not in cat_candidates]
        cat_options = cat_candidates + extra_candidates[:20]
        if cat_options:
            cat_choice = st.multiselect('Variables catégorielles à décrire', cat_options, default=cat_candidates[:4], key='oms_cat_choice')
            for c in cat_choice:
                with st.expander(f'Fréquences — {c}', expanded=False):
                    if c == COL_ZS and COL_PROV in df_f.columns:
                        tbl = (
                            df_f.assign(
                                _province=df_f[COL_PROV].fillna('Inconnu').astype(str).str.strip().replace('', 'Inconnu'),
                                _modalite=df_f[COL_ZS].fillna('Inconnu').astype(str).str.strip().replace('', 'Inconnu'),
                            )
                            .groupby(['_province', '_modalite'], dropna=False)
                            .size()
                            .reset_index(name='n')
                            .rename(columns={
                                '_province': 'Province de notification',
                                '_modalite': 'Zone_de_sante_notification',
                            })
                        )
                        tbl['%'] = (tbl['n'] / max(len(df_f), 1) * 100).round(1)
                        tbl = tbl.sort_values(['n', 'Province de notification', 'Zone_de_sante_notification'], ascending=[False, True, True])
                        st_dataframe_safe(tbl)
                    elif c == COL_AS and COL_PROV in df_f.columns and COL_ZS in df_f.columns:
                        tbl = (
                            df_f.assign(
                                _province=df_f[COL_PROV].fillna('Inconnu').astype(str).str.strip().replace('', 'Inconnu'),
                                _zone=df_f[COL_ZS].fillna('Inconnu').astype(str).str.strip().replace('', 'Inconnu'),
                                _modalite=df_f[COL_AS].fillna('Inconnu').astype(str).str.strip().replace('', 'Inconnu'),
                            )
                            .groupby(['_province', '_zone', '_modalite'], dropna=False)
                            .size()
                            .reset_index(name='n')
                            .rename(columns={
                                '_province': 'Province de notification',
                                '_zone': 'Zone de notification',
                                '_modalite': 'Aire_de_sante_notification',
                            })
                        )
                        tbl['%'] = (tbl['n'] / max(len(df_f), 1) * 100).round(1)
                        tbl = tbl.sort_values(['n', 'Province de notification', 'Zone de notification', 'Aire_de_sante_notification'], ascending=[False, True, True, True])
                        st_dataframe_safe(tbl)
                    else:
                        st_dataframe_safe(build_frequency_table(df_f, c))
        else:
            st.info("Aucune variable catégorielle exploitable n’a été détectée.")

    # =========================
    # TAB 5: Complétude
    # =========================

