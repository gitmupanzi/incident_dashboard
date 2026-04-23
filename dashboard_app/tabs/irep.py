"""Render the provincial IREP tab."""

from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


def render_irep_tab(ctx: dict) -> None:
    """Render the provincial IREP tab."""
    globals().update(ctx)
    st.subheader("Indice provincial composite de risque épidémique (IREP)")
    render_tab_narrative("irep")
    tab_help(
        "Lecture et interprétation",
        """
        **🎯 Objectif** : classer les provinces selon un risque combiné (0–100) qui intègre:
        - **Tendance** (hausse récente)
        - **Incidence** (si population disponible)
        - **Létalité**
        - **Promptitude** (retard de notification)
        - **Complétude** (qualité de saisie)

        **🧠 Interprétation** : plus l’**IREP** est élevé, plus la situation mérite attention (investigation / renfort / supervision).
        """,
        expanded=False,
    )

    if df is None or df.empty:
        render_absence_narrative("risk")
    else:
        # -----------------------------
        # 1) Choisir colonne semaine
        # -----------------------------
        if "Semaine_epid" in df.columns:
            col_week_irep = "Semaine_epid"
        else:
            # fallback: YW / TIME_KEY / TIME_LAB
            _wk, _ = choose_week_column(df)
            if _wk is not None and _wk.notna().any():
                df["_WEEK_TMP_"] = _wk.astype(str)
                col_week_irep = "_WEEK_TMP_"
            else:
                st.error("Aucune variable semaine n’a été détectée (Semaine_epid / YW / TIME_KEY / TIME_LAB).")
                st.stop()

        # Liste des semaines (tri robuste)
        week_vals = sorted(df[col_week_irep].dropna().astype(str).unique().tolist())
        if not week_vals:
            st.info("Aucune semaine valide n’est disponible pour calculer l’IREP.")
            st.stop()

        # -----------------------------
        # 2) Population (optionnel)
        # -----------------------------
        st.markdown("### Population provinciale (optionnelle, pour le calcul de l’incidence)")
        pop_upl = st.file_uploader(
            "Téléverser un fichier population (csv/xlsx) avec colonnes: Province, Population",
            type=["csv", "xlsx", "xls"],
            key="pop_upload_irep"
        )

        population_map = {}
        if pop_upl is not None:
            try:
                if pop_upl.name.lower().endswith(".csv"):
                    pop_df = pd.read_csv(pop_upl)
                else:
                    pop_df = pd.read_excel(pop_upl)

                # normaliser noms colonnes
                pop_df.columns = [str(c).strip() for c in pop_df.columns]
                # heuristiques colonnes
                prov_col = None
                for c in ["Province_notification", "Province", "province", "PROVINCE"]:
                    if c in pop_df.columns:
                        prov_col = c
                        break
                pop_col = None
                for c in ["Population", "POPULATION", "pop", "POP", "population"]:
                    if c in pop_df.columns:
                        pop_col = c
                        break

                if prov_col is None or pop_col is None:
                    st.warning("Fichier de population non reconnu. Colonnes attendues : 'Province' et 'Population'.")
                else:
                    pop_df = pop_df[[prov_col, pop_col]].dropna()
                    pop_df[prov_col] = pop_df[prov_col].astype(str).str.strip()
                    pop_df[pop_col] = pd.to_numeric(pop_df[pop_col], errors="coerce")
                    pop_df = pop_df.dropna(subset=[pop_col])

                    population_map = dict(zip(pop_df[prov_col], pop_df[pop_col].astype(int)))
                    st.success(f"Population chargée pour {len(population_map)} provinces.")
                    with st.expander("Aperçu population"):
                        st.dataframe(pop_df.head(30), width="stretch")
            except Exception as e:
                st.warning(f"Impossible de lire le fichier population : {e}")

        # -----------------------------
        # 3) Paramètres score (poids & fenêtres)
        # -----------------------------
        st.markdown("### Paramètres de calcul")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            w_trend = st.slider("Poids Tendance", 0.0, 1.0, 0.30, 0.05)
        with c2:
            w_inc = st.slider("Poids Incidence", 0.0, 1.0, 0.25, 0.05)
        with c3:
            w_cfr = st.slider("Poids Létalité", 0.0, 1.0, 0.20, 0.05)
        with c4:
            w_time = st.slider("Poids Promptitude", 0.0, 1.0, 0.15, 0.05)
        with c5:
            w_comp = st.slider("Poids Complétude", 0.0, 1.0, 0.10, 0.05)

        # Normaliser pour éviter somme=0
        w_user = {"trend": w_trend, "incidence": w_inc, "cfr": w_cfr, "timeliness": w_time, "completeness": w_comp}
        if sum(w_user.values()) == 0:
            st.warning("Tous les poids sont à 0. Réinitialisation aux valeurs par défaut.")
            w_user = {"trend": 0.30, "incidence": 0.25, "cfr": 0.20, "timeliness": 0.15, "completeness": 0.10}

        current_week = st.selectbox(
            "Semaine courante",
            options=week_vals,
            index=len(week_vals) - 1
        )

        # Seuil de promptitude (réutilise celui de la sidebar si présent)
        try:
            threshold_days = get_session_int("seuil_jours", 2)
        except Exception:
            threshold_days = 2

        # -----------------------------
        # 4) Préparation minimale des colonnes cas/décès si besoin (line list)
        # -----------------------------
        df_irep = df.copy()

        if "Total_cas" not in df_irep.columns:
            df_irep["Total_cas"] = 1

        if "Total_deces" not in df_irep.columns:
            if COL_ISSUE in df_irep.columns:
                df_irep["Total_deces"] = df_irep[COL_ISSUE].apply(lambda x: 1 if is_death(x) else 0)
            else:
                df_irep["Total_deces"] = 0

        # -----------------------------
        # 5) Calcul IREP
        # -----------------------------
        irep = compute_irep_province(
            df_irep,
            col_prov=COL_PROV if COL_PROV in df_irep.columns else "Province",
            col_week=col_week_irep,
            col_cases="Total_cas",
            col_deaths="Total_deces",
            current_week=str(current_week),
            population_map=population_map,
            date_onset=DATE_ONSET,
            date_notif="Date_notification",
            w=w_user,
            threshold_days=threshold_days,
        )

        if irep is None or irep.empty:
            render_absence_narrative("risk")
        else:
            # KPIs synthèse
            st.markdown("### Synthèse")
            kA, kB, kC, kD = st.columns(4)
            kA.metric("Provinces (IREP calculé)", str(irep[COL_PROV].nunique() if COL_PROV in irep.columns else len(irep)))
            kB.metric("IREP moyen", f"{irep['IREP'].mean():.1f}" if 'IREP' in irep.columns else "-")
            kC.metric("IREP max", f"{irep['IREP'].max():.1f}" if 'IREP' in irep.columns else "-")
            kD.metric("Semaine", str(current_week))

            # Top 5
            st.markdown("### Top provinces à risque")
            st.dataframe(irep.head(10), width="stretch", height=320)

            # Graphique
            try:
                plot_df = irep.copy()
                prov_col = COL_PROV if COL_PROV in plot_df.columns else plot_df.columns[0]
                fig = px.bar(
                    plot_df,
                    x=prov_col,
                    y="IREP",
                    color="Risque_cat" if "Risque_cat" in plot_df.columns else None,
                    title="IREP par province (plus haut = plus à risque)",
                )
                fig.update_layout(xaxis_tickangle=-45)
                fig = apply_plotly_value_annotations(fig, annot_vals)
                st.plotly_chart(fig, width="stretch")
            except Exception:
                pass

            # Download
            st.download_button(
                "⬇️ Télécharger IREP (CSV)",
                data=df_to_csv_bytes(irep),
                file_name=f"IREP_provinces_{current_week}.csv",
                mime="text/csv"
            )


