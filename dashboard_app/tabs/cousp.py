"""Render the COUSP standard analytics tab."""

from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


def _cousp_sheet_overview_table(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sheet_name, sheet_df in sheets.items():
        if not isinstance(sheet_df, pd.DataFrame):
            continue
        rows.append(
            {
                "Feuille": sheet_name,
                "Lignes": int(sheet_df.shape[0]),
                "Colonnes": int(sheet_df.shape[1]),
            }
        )
    return pd.DataFrame(rows)


def render_cousp_tab(ctx: dict) -> None:
    """Render the COUSP standard analytics tab."""
    globals().update(ctx)

    render_section_title(9, "Pack d'analyse COUSP standard")
    if IDSR_MODE:
        render_absence_narrative("idsr_line_list")
        return

    tab_help(
        "Comment lire cet onglet",
        """
        **Objectif** : exposer directement le pack d'analyse COUSP standard a partir du perimetre filtre.

        **Lecture recommandee**
        - **Synthese operationnelle** : commence par les KPI COUSP et les delais prioritaires.
        - **Completude** : verifie les variables cles P1/P2 attendues par le plan standard.
        - **Anomalies de dates** : repere les chronologies incoherentes a corriger.
        - **Cas a relancer** : identifie les dossiers incomplets pour l'investigation ou le labo.
        """,
        expanded=False,
    )

    if df_f is None or not isinstance(df_f, pd.DataFrame) or df_f.empty:
        st.info("Aucune donnee filtree n'est disponible pour construire l'analyse COUSP.")
        return

    with st.container():
        st.markdown("**Parametres de completude COUSP**")
        st.caption(
            "Ces seuils pilotent directement la colonne `Decision / observation` "
            "dans la feuille `Completeness_variables_cles` et dans l'export Excel."
        )
        c1, c2, c3 = st.columns([1.1, 1, 1])
        with c1:
            anonymiser_cousp = st.checkbox(
                "Anonymiser la feuille Recherche_dataset",
                value=False,
                key="cousp_tab_anonymiser_recherche",
            )
        with c2:
            seuil_acceptable = st.number_input(
                "Seuil acceptable (% missing)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=1.0,
                key="cousp_tab_seuil_acceptable",
            )
        with c3:
            seuil_surveillance = st.number_input(
                "Seuil surveillance (% missing)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                key="cousp_tab_seuil_surveillance",
            )

    if float(seuil_acceptable) > float(seuil_surveillance):
        st.warning(
            "Le seuil acceptable ne peut pas depasser le seuil de surveillance. "
            "Les valeurs ont ete realignees automatiquement."
        )
        seuil_surveillance = seuil_acceptable

    info_c1, info_c2 = st.columns([1.35, 1.0])
    with info_c1:
        st.caption(
            "Le contenu de cet onglet est calcule a partir de la line list deja filtree dans le dashboard."
        )
    with info_c2:
        st.caption(
            f"Decision / observation : OK=0%, Acceptable<={float(seuil_acceptable):.1f}%, "
            f"A surveiller<={float(seuil_surveillance):.1f}%, sinon Prioritaire."
        )

    sheets, error = build_cousp_standard_export_package(
        df_f,
        anonymiser_recherche=anonymiser_cousp,
        seuil_acceptable=float(seuil_acceptable),
        seuil_surveillance=float(seuil_surveillance),
    )
    if error:
        st.warning(error)
        return
    if not sheets:
        st.info("Aucune feuille COUSP n'a ete generee.")
        return

    summary_df = _cousp_sheet_overview_table(sheets)
    synthese_df = sheets.get("Synthese_operationnelle", pd.DataFrame())
    completeness_df = sheets.get("Completeness_variables_cles", pd.DataFrame())
    anomalies_df = sheets.get("Anomalies_dates", pd.DataFrame())
    relances_df = sheets.get("Cas_a_relancer", pd.DataFrame())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Feuilles COUSP", len(sheets))
    m2.metric("Lignes filtrees", int(len(df_f)))
    m3.metric("Anomalies de dates", int(len(anomalies_df)))
    m4.metric("Cas a relancer", int(len(relances_df)))

    st.markdown("**Resume du pack genere**")
    st_dataframe_safe(summary_df, height=220)

    try:
        cousp_excel_bytes = workbook_bytes_from_sheet_dict(sheets)
    except Exception as exc:
        st.warning(f"Le telechargement du pack COUSP est indisponible : {exc}")
    else:
        export_base_name = f"{str(disease_key).strip().lower()}_filtre"
        st.download_button(
            "Telecharger le pack COUSP standard (Excel)",
            data=cousp_excel_bytes,
            file_name=f"{export_base_name}_pack_cousp_standard.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_cousp_tab_pack_xlsx",
            use_container_width=True,
        )

    (
        tab_synthese,
        tab_completude,
        tab_anomalies,
        tab_relances,
        tab_recherche,
    ) = st.tabs(
        [
            "Synthese operationnelle",
            "Completude",
            "Anomalies de dates",
            "Cas a relancer",
            "Recherche dataset",
        ]
    )

    with tab_synthese:
        if synthese_df.empty:
            st.info("Aucune synthese operationnelle disponible.")
        else:
            if "Section" in synthese_df.columns:
                kpi_df = synthese_df.loc[synthese_df["Section"] == "KPI"].copy()
                delay_df = synthese_df.loc[synthese_df["Section"] != "KPI"].copy()
            else:
                kpi_df = synthese_df.copy()
                delay_df = pd.DataFrame()

            if not kpi_df.empty:
                st.markdown("**KPI COUSP**")
                st_dataframe_safe(kpi_df, height=320)
            if not delay_df.empty:
                st.markdown("**Delais prioritaires**")
                st_dataframe_safe(delay_df, height=320)

    with tab_completude:
        if completeness_df.empty:
            st.info("Aucune analyse de completude disponible.")
        else:
            st.caption(
                f"Seuils actifs : acceptable <= {float(seuil_acceptable):.1f}% missing ; "
                f"a surveiller <= {float(seuil_surveillance):.1f}% missing."
            )
            k_m1, k_m2, k_m3, k_m4 = st.columns(4)
            k_m1.metric("Variables suivies", str(len(completeness_df)))
            k_m2.metric(
                "Variables prioritaires",
                str(int((completeness_df["Decision / observation"] == "Prioritaire").sum())),
            )
            k_m3.metric(
                "Variables sans missing",
                str(int((completeness_df["Decision / observation"] == "OK").sum())),
            )
            k_m4.metric(
                "Missing moyen (%)",
                f"{pd.to_numeric(completeness_df['% missing'], errors='coerce').mean():.1f}",
            )

            decision_options = sorted(
                completeness_df["Decision / observation"].dropna().astype(str).unique().tolist()
            )
            decision_sel = st.multiselect(
                "Niveau de priorite a afficher",
                options=decision_options,
                default=decision_options,
                key="cousp_missing_decision_filter",
            )
            if decision_sel:
                completeness_view = completeness_df[
                    completeness_df["Decision / observation"].isin(decision_sel)
                ].copy()
            else:
                completeness_view = completeness_df.copy()

            st_dataframe_safe(completeness_view, height=520)
            st.download_button(
                "Telecharger le tableau de completude (CSV)",
                data=completeness_view.to_csv(index=False).encode("utf-8"),
                file_name="cousp_completude_variables.csv",
                mime="text/csv",
                key="download_cousp_missing_csv",
            )

            topn = st.slider(
                "Nombre de variables a afficher",
                min_value=5,
                max_value=min(80, max(5, len(completeness_view))),
                value=min(20, len(completeness_view)),
                step=5,
                key="cousp_missing_topn",
            )
            comp_plot = completeness_view.sort_values(
                ["% missing", "Manquantes"],
                ascending=[False, False],
            ).head(topn)

            figc = px.bar(
                comp_plot,
                x="Variable cle",
                y="% missing",
                color="Decision / observation",
                title=f"Variables prioritaires selon le taux de missing ({topn})",
            )
            figc.update_layout(xaxis_tickangle=-45, yaxis=dict(range=[0, 100]))
            figc = apply_plotly_value_annotations(figc, annot_vals)
            st.plotly_chart(figc, width="stretch")

    with tab_anomalies:
        if anomalies_df.empty:
            st.success("Aucune anomalie de dates detectee dans le perimetre filtre.")
        else:
            st_dataframe_safe(anomalies_df, height=540)

    with tab_relances:
        if relances_df.empty:
            st.success("Aucun cas a relancer detecte dans le perimetre filtre.")
        else:
            st_dataframe_safe(relances_df, height=540)

    with tab_recherche:
        recherche_df = sheets.get("Recherche_dataset", pd.DataFrame())
        if recherche_df.empty:
            st.info("Aucun dataset de recherche disponible.")
        else:
            st.caption("Apercu du dataset standardise COUSP utilise pour la recherche et l'export.")
            st_dataframe_safe(recherche_df.head(200), height=540)
