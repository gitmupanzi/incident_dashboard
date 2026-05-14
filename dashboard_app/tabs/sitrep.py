"""Render the SITREP generation tab."""

from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


def render_sitrep_tab(ctx: dict) -> None:
    """Render the SITREP generation tab."""
    globals().update(ctx)
    if IDSR_MODE:
        render_absence_narrative("idsr_line_list")
    else:
        render_section_title(7, "Synthèse automatique de la situation épidémiologique (SITREP)")
        render_tab_narrative("sitrep")

        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif**
            - Donner une vue courte et utile de la **semaine sélectionnée** à partir des filtres actifs.

            **📖 Comment lire**
            - Le SITREP met l’accent sur la **semaine ciblée**, la compare à la semaine précédente et rappelle aussi le **cumul annuel**.
            - Il met surtout en avant les zones les plus touchées, la gravité et les points qui demandent une action rapide.

            **🚫 Ce qui reste dans les autres onglets**
            - Les analyses détaillées de délais restent dans **Surveillance**.
            - Les profils détaillés restent dans **Profil**.
            - Les tableaux complets de qualité et de complétude restent dans **Données, complétude & qualité**.

            **📤 Exportation**
            - Le PDF reprend la synthèse affichée dans cet onglet.
            """,
            expanded=False
        )

        st.caption(
            "Le SITREP reprend seulement les informations les plus utiles pour une lecture rapide, sans répéter les analyses détaillées des autres onglets."
        )

        # =========================================================
        # 1) UI: SE / Année / Date de publication dépendants de df_f
        # =========================================================
        if (COL_WNUM in df_f.columns) and df_f[COL_WNUM].notna().any():
            w_series = pd.to_numeric(df_f[COL_WNUM], errors="coerce").dropna()
            w_min, w_max = int(w_series.min()), int(w_series.max())
        else:
            w_min, w_max = 1, 53

        if (COL_YEAR in df_f.columns) and df_f[COL_YEAR].notna().any():
            y_series = pd.to_numeric(df_f[COL_YEAR], errors="coerce").dropna()
            y_min, y_max = int(y_series.min()), int(y_series.max())
        else:
            y_min, y_max = 2020, date.today().year

        auto_last = st.checkbox(
            "Auto: utiliser la dernière SE/Année du filtrage",
            value=True,
            key="sitrep_auto_last"
        )

        colA, colB, colC = st.columns(3)
        with colA:
            semaine = st.number_input(
                "Semaine épidémiologique (SE)",
                min_value=int(w_min),
                max_value=int(w_max),
                value=int(w_max),
                step=1,
                key="sitrep_se",
            )
        with colB:
            annee = st.number_input(
                "Année",
                min_value=int(y_min),
                max_value=int(y_max),
                value=int(y_max),
                step=1,
                key="sitrep_year",
            )
        with colC:
            date_pub = st.date_input(
                "Date de publication",
                value=date.today(),
                key="sitrep_pubdate",
            )

        if auto_last:
            semaine = int(w_max)
            annee = int(y_max)

        st.caption(
            f"Données utilisées pour le SITREP : données filtrées (`df_f`). SE disponibles : {w_min}–{w_max}. "
            f"Années disponibles : {y_min}–{y_max}. Semaine ciblée : {_build_sitrep_week_label(semaine, annee)}."
        )

        # =========================================================
        # 2) Helpers supplémentaires (spatial, demo, délais, graphiques)
        # =========================================================
        def _safe_pct(num, den):
            return (num / den * 100.0) if den and den > 0 else np.nan

        def _plotly_to_png_bytes(fig, scale: int = 1):
            """
            Conversion Plotly -> PNG bytes robuste pour Streamlit Cloud.
            - Ne fait jamais planter l'application
            - Réduit la charge mémoire avec scale=1 par défaut
            """
            if fig is None:
                return None
            try:
                return fig.to_image(format="png", scale=scale)
            except Exception as e:
                logger.warning(f"[SITREP] Export PNG Plotly ignoré : {e}")
                return None

        @st.cache_data(show_spinner=False)
        def build_weekly_summary(df_scope):
            """Table hebdo Cas/Décès/CFR (sur df_scope filtré)."""
            if COL_YEAR not in df_scope.columns or COL_WNUM not in df_scope.columns:
                return pd.DataFrame()

            tmp = df_scope.copy()
            tmp["_cas_"] = 1
            tmp["_deces_"] = tmp["is_death"].astype(int) if "is_death" in tmp.columns else 0

            wk = (tmp.groupby([COL_YEAR, COL_WNUM], as_index=False)
                    .agg(Cas=("_cas_", "sum"), Décès=("_deces_", "sum")))

            wk["CFR_%"] = np.where(wk["Cas"] > 0, wk["Décès"] / wk["Cas"] * 100.0, np.nan)
            wk["YW"] = wk[COL_YEAR].astype(int).astype(str) + "W" + wk[COL_WNUM].astype(int).astype(str).str.zfill(2)
            wk = wk.sort_values([COL_YEAR, COL_WNUM])

            wk["Cas_prev"] = wk["Cas"].shift(1)
            wk["var_%"] = np.where(
                wk["Cas_prev"].fillna(0) > 0,
                (wk["Cas"] - wk["Cas_prev"]) / wk["Cas_prev"] * 100.0,
                np.nan
            )
            return wk

        @st.cache_data(show_spinner=False)
        def build_geo_tables(d_se, min_cas_zs=30, min_cas_prov=50):
            """Tables province/ZS pour la semaine (d_se)."""
            out = {}
            if "is_death" not in d_se.columns:
                d_se = d_se.copy()
                d_se["is_death"] = 0

            tmp = d_se.copy()
            tmp["_cas_"] = 1
            tmp["_deces_"] = tmp["is_death"].astype(int)

            if COL_PROV in tmp.columns:
                prov = (tmp.groupby(COL_PROV, as_index=False)
                          .agg(Cas=("_cas_", "sum"), Décès=("_deces_", "sum")))
                prov["CFR_%"] = np.where(prov["Cas"] > 0, prov["Décès"] / prov["Cas"] * 100.0, np.nan)
                out["prov_table"] = prov.sort_values("Cas", ascending=False)
                out["prov_cfr_crit"] = prov.query("Cas >= @min_cas_prov").sort_values("CFR_%", ascending=False)

            if COL_ZS in tmp.columns:
                group_cols = [c for c in [COL_PROV, COL_ZS] if c in tmp.columns]
                zs = (tmp.groupby(group_cols, as_index=False)
                        .agg(Cas=("_cas_", "sum"), Décès=("_deces_", "sum")))
                zs["CFR_%"] = np.where(zs["Cas"] > 0, zs["Décès"] / zs["Cas"] * 100.0, np.nan)
                out["zs_table"] = zs.sort_values("Cas", ascending=False)
                out["zs_cfr_crit"] = zs.query("Cas >= @min_cas_zs").sort_values("CFR_%", ascending=False)

            return out

        @st.cache_data(show_spinner=False)
        def build_demo_tables(d_se):
            """Sexe / tranches âge (si disponibles)."""
            out = {}
            if COL_SEX in d_se.columns:
                sex = (d_se.groupby(COL_SEX, as_index=False)
                         .size().rename(columns={"size": "Cas"})
                         .sort_values("Cas", ascending=False))
                out["sex_table"] = sex

            # Priorité aux tranches déjà calculées dans tes données
            age_group_col = None
            if COL_AGEG in d_se.columns:
                age_group_col = COL_AGEG
            elif COL_AGEG2 in d_se.columns:
                age_group_col = COL_AGEG2

            if age_group_col:
                age = (d_se.groupby(age_group_col, as_index=False)
                         .size().rename(columns={"size": "Cas"}))
                age = age.rename(columns={age_group_col: "Tranche_age"})
                out["age_table"] = age
            return out

        @st.cache_data(show_spinner=False)
        def build_delay_summary(d_se):
            """Résumé timeliness (début maladie → admission) si dates présentes."""
            if (DATE_ONSET not in d_se.columns) or (DATE_ADM not in d_se.columns):
                return pd.DataFrame()

            tmp = d_se[[DATE_ONSET, DATE_ADM]].copy()
            tmp[DATE_ONSET] = pd.to_datetime(tmp[DATE_ONSET], errors="coerce")
            tmp[DATE_ADM] = pd.to_datetime(tmp[DATE_ADM], errors="coerce")
            tmp["delai_onset_adm"] = (tmp[DATE_ADM] - tmp[DATE_ONSET]).dt.days

            # bornes raisonnables (0..30j)
            tmp = tmp[(tmp["delai_onset_adm"].notna()) & (tmp["delai_onset_adm"] >= 0) & (tmp["delai_onset_adm"] <= 30)]
            if tmp.empty:
                return pd.DataFrame()

            s = tmp["delai_onset_adm"]
            return pd.DataFrame([{
                "n": int(s.notna().sum()),
                "médiane": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "%≤1j": _safe_pct((s <= 1).sum(), s.notna().sum()),
                "%≤2j": _safe_pct((s <= 2).sum(), s.notna().sum()),
                "max": float(s.max()),
            }])

        # =========================================================
        # 3) Build payload (onglet SITREP extrait) — VERSION ENRICHIE
        # =========================================================
        @st.cache_data(show_spinner=False)
        def _build_sitrep_payload_from_df(
            df_scope,
            se,
            annee,
            date_pub,
            min_cas_zs=30,
            min_cas_prov=50,
            include_images=False,
        ):
            """
            Build un payload SITREP épidémiologique à partir de df_scope (ici df_f filtré).

            IMPORTANT:
            - include_images=False par défaut pour éviter de lancer Kaleido/Chromium
              à chaque rerun Streamlit.
            - Les images PNG pour le PDF ne sont générées qu'à la demande.
            """
            d = df_scope.copy()

            # Fix colonnes dupliquées
            if d.columns.duplicated().any():
                d = d.loc[:, ~d.columns.duplicated()].copy()

            # Filtre SE/Année
            d_se = d.copy()
            if COL_WNUM in d_se.columns:
                d_se = d_se[pd.to_numeric(d_se[COL_WNUM], errors="coerce") == int(se)]
            if COL_YEAR in d_se.columns:
                d_se = d_se[pd.to_numeric(d_se[COL_YEAR], errors="coerce") == int(annee)]

            # Cumul année <= SE
            d_cum = d.copy()
            if COL_YEAR in d_cum.columns:
                d_cum = d_cum[pd.to_numeric(d_cum[COL_YEAR], errors="coerce") == int(annee)]
            if COL_WNUM in d_cum.columns:
                d_cum = d_cum[pd.to_numeric(d_cum[COL_WNUM], errors="coerce") <= int(se)]

            def _kpi(df_):
                cases = int(len(df_))
                deaths = int(df_["is_death"].sum()) if "is_death" in df_.columns else 0
                cfr = (deaths / cases * 100.0) if cases > 0 else 0.0
                return cases, deaths, cfr

            cas_se, dec_se, cfr_se = _kpi(d_se)
            cas_cum, dec_cum, cfr_cum = _kpi(d_cum)
            weekly_summary = build_weekly_summary(d)
            selected_week_label = _build_sitrep_week_label(se, annee)
            previous_week_df = pd.DataFrame()
            previous_week_label = None

            if not weekly_summary.empty and {COL_YEAR, COL_WNUM}.issubset(weekly_summary.columns):
                weekly_reference = weekly_summary.reset_index(drop=True).copy()
                match_mask = (
                    pd.to_numeric(weekly_reference[COL_YEAR], errors="coerce") == int(annee)
                ) & (
                    pd.to_numeric(weekly_reference[COL_WNUM], errors="coerce") == int(se)
                )
                if bool(match_mask.any()):
                    selected_idx = int(np.flatnonzero(match_mask.to_numpy())[0])
                    if selected_idx > 0:
                        previous_row = weekly_reference.iloc[selected_idx - 1]
                        prev_year = pd.to_numeric(pd.Series([previous_row[COL_YEAR]]), errors="coerce").iloc[0]
                        prev_week = pd.to_numeric(pd.Series([previous_row[COL_WNUM]]), errors="coerce").iloc[0]
                        if pd.notna(prev_year) and pd.notna(prev_week):
                            previous_week_label = (
                                _build_sitrep_week_label(prev_week, prev_year)
                            )
                            prev_mask = pd.Series(True, index=d.index)
                            if COL_YEAR in d.columns:
                                prev_mask &= pd.to_numeric(d[COL_YEAR], errors="coerce") == int(prev_year)
                            if COL_WNUM in d.columns:
                                prev_mask &= pd.to_numeric(d[COL_WNUM], errors="coerce") == int(prev_week)
                            previous_week_df = d.loc[prev_mask].copy()

            prev_cases, prev_deaths, prev_cfr = _kpi(previous_week_df)
            provinces_reporting = (
                int(_surveillance_clean_text_series(d_se[COL_PROV]).nunique())
                if COL_PROV in d_se.columns else 0
            )
            zs_reporting = (
                int(_surveillance_clean_text_series(d_se[COL_ZS]).nunique())
                if COL_ZS in d_se.columns else 0
            )
            top_prov_focus = _build_surveillance_top_table(d_se, [COL_PROV], top_n=5)
            top_zs_focus = _build_surveillance_top_table(
                d_se,
                [c for c in [COL_PROV, COL_ZS] if c in d_se.columns],
                top_n=5,
            )

            # Table épidémiologique par ZS (SE sélectionnée)
            table_epi = pd.DataFrame()
            if (COL_ZS in d_se.columns) and len(d_se):
                tmp = d_se.copy()
                tmp["_cas_"] = 1
                tmp["_deces_"] = tmp["is_death"].astype(int) if "is_death" in tmp.columns else 0

                group_cols = [c for c in [COL_PROV, COL_ZS] if c in tmp.columns]
                table_epi = (
                    tmp.groupby(group_cols, as_index=False)
                       .agg(cas=("_cas_", "sum"), deces=("_deces_", "sum"))
                       .sort_values("cas", ascending=False)
                )
                if COL_PROV in table_epi.columns:
                    table_epi = table_epi.rename(columns={COL_PROV: "Province de notification"})
                if COL_ZS in table_epi.columns:
                    table_epi = table_epi.rename(columns={COL_ZS: "Zone de santé"})

            payload = {
                "meta": {"semaine": int(se), "annee": int(annee), "date_publication": date_pub},
                "kpi": {
                    "cas_semaine": cas_se,
                    "deces_semaine": dec_se,
                    "cfr_semaine": cfr_se,
                    "cas_semaine_prev": prev_cases,
                    "deces_semaine_prev": prev_deaths,
                    "cfr_semaine_prev": prev_cfr,
                    "cas_cumul": cas_cum,
                    "deces_cumul": dec_cum,
                    "cfr_cumul": cfr_cum,
                    "provinces_reporting": provinces_reporting,
                    "zs_reporting": zs_reporting,
                },
                "table_epi": table_epi,
                "selected_df": d_se,
                "cumulative_df": d_cum,
                "previous_df": previous_week_df,
                "selected_week_label": selected_week_label,
                "previous_week_label": previous_week_label,
                "top_prov_focus": top_prov_focus,
                "top_zs_focus": top_zs_focus,
            }

            # Cascade labo (si fonction dispo)
            payload["cascade"] = call_optional_function("cascade_metrics", d_se, default=pd.DataFrame())

            # Alertes sur la dernière semaine disponible — sur df_scope filtré
            payload["alertes_last"] = call_optional_function("build_alerts_last_week", d, default=pd.DataFrame())

            # Série hebdo filtrée pour visualisation / PDF
            payload["weekly"] = weekly_summary

            # Analyse spatiale et gravité
            payload.update(build_geo_tables(d_se, min_cas_zs=min_cas_zs, min_cas_prov=min_cas_prov))

            # Interprétation automatisée
            interpret = []
            provcrit = payload.get("prov_cfr_crit")
            if isinstance(provcrit, pd.DataFrame) and not provcrit.empty:
                top3 = provcrit.head(3)
                parts = [f"{r[COL_PROV]} (CFR {r['CFR_%']:.1f}%)" for _, r in top3.iterrows() if COL_PROV in top3.columns]
                if parts:
                    interpret.append("Provinces à létalité élevée (seuil) : " + ", ".join(parts))

            zscrit = payload.get("zs_cfr_crit")
            if isinstance(zscrit, pd.DataFrame) and not zscrit.empty:
                parts = []
                for _, r in zscrit.head(5).iterrows():
                    if COL_PROV in zscrit.columns:
                        parts.append(f"{r[COL_PROV]} / {r[COL_ZS]} (CFR {r['CFR_%']:.1f}%)")
                    else:
                        parts.append(f"{r[COL_ZS]} (CFR {r['CFR_%']:.1f}%)")
                if parts:
                    interpret.append("ZS à létalité élevée (seuil) : " + ", ".join(parts))

            payload["interpretation"] = interpret
            payload["summary_lines"] = _build_sitrep_summary_lines(payload)
            payload["decision_focus"] = _build_sitrep_action_lines(payload)
            payload["points_saillants"] = payload["summary_lines"][:5]
            payload["defis_besoins"] = payload["decision_focus"][:4]
            payload["perspectives"] = [
                "Les analyses détaillées de délais sont consultables dans l’onglet Surveillance.",
                "Les profils âge/sexe détaillés sont consultables dans l’onglet Profil.",
                "Les vérifications exhaustives de complétude et de qualité restent dans l’onglet Données, complétude & qualité.",
            ]
            payload["images"] = []

            if include_images:
                try:
                    wk = payload.get("weekly")
                    if isinstance(wk, pd.DataFrame) and not wk.empty and "YW" in wk.columns:
                        fig1 = build_weekly_cases_deaths_combo(
                            weekly_df=wk,
                            x_col="YW",
                            cases_col="Cas",
                            deaths_col="Décès",
                            titre=" ",
                            x_titre="Semaine (YW)",
                            y_titre_cas="Nombre de cas",
                            y_titre_deces="Nombre de décès",
                            rotation=0,
                        )
                        png1 = _plotly_to_png_bytes(fig1, scale=1)
                        if png1:
                            payload["images"].append(("Évolution hebdomadaire", png1))

                    provt = payload.get("prov_table")
                    if isinstance(provt, pd.DataFrame) and not provt.empty and COL_PROV in provt.columns:
                        fig2 = px.bar(provt.head(10), x=COL_PROV, y="Cas", title="Top 10 Provinces – Cas (SE)")
                        fig2.update_layout(xaxis_tickangle=-45)
                        png2 = _plotly_to_png_bytes(fig2, scale=1)
                        if png2:
                            payload["images"].append(("Top provinces (cas)", png2))

                    zst = payload.get("zs_table")
                    if isinstance(zst, pd.DataFrame) and not zst.empty:
                        zst2 = zst.copy()
                        if (COL_PROV in zst2.columns) and (COL_ZS in zst2.columns):
                            zst2["Prov/ZS"] = zst2[COL_PROV].astype(str) + " / " + zst2[COL_ZS].astype(str)
                            xcol = "Prov/ZS"
                        elif COL_ZS in zst2.columns:
                            xcol = COL_ZS
                        else:
                            xcol = None

                        if xcol is not None:
                            fig3 = px.bar(zst2.head(10), x=xcol, y="Cas", title="Top 10 ZS – Cas (SE)")
                            fig3.update_layout(xaxis_tickangle=-45)
                            png3 = _plotly_to_png_bytes(fig3, scale=1)
                            if png3:
                                payload["images"].append(("Top ZS (cas)", png3))
                except Exception as e:
                    logger.warning(f"[SITREP] Génération des images PDF ignorée : {e}")

            return payload

        # Paramètres de seuils (gravité)
        st.markdown("### Paramètres d’analyse et seuils d’alerte")
        cS1, cS2 = st.columns(2)
        with cS1:
            min_cas_zs = st.number_input("Seuil min cas ZS (pour CFR critique)", min_value=10, max_value=200, value=30, step=5)
        with cS2:
            min_cas_prov = st.number_input("Seuil min cas Province (pour CFR critique)", min_value=10, max_value=500, value=50, step=10)

        sitrep_payload = _build_sitrep_payload_from_df(
            df_f,
            semaine,
            annee,
            date_pub,
            min_cas_zs=min_cas_zs,
            min_cas_prov=min_cas_prov,
            include_images=False,
        )

        selected_df = sitrep_payload.get("selected_df", pd.DataFrame())
        selected_week_label = sitrep_payload.get("selected_week_label", _build_sitrep_week_label(semaine, annee))
        previous_week_label = sitrep_payload.get("previous_week_label")
        k = sitrep_payload["kpi"]

        st.divider()
        render_section_title(8, "Lecture rapide et message clé")
        st.caption(
            "Cette lecture reprend la semaine ciblée, sa comparaison avec la semaine précédente disponible et le cumul annuel, sans reprendre les analyses détaillées d’un autre onglet."
        )

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(
            "Cas (SE)",
            format_metric_value(k["cas_semaine"]),
            delta=_format_sitrep_metric_delta(k["cas_semaine"], k.get("cas_semaine_prev")),
        )
        k2.metric(
            "Décès (SE)",
            format_metric_value(k["deces_semaine"]),
            delta=_format_sitrep_metric_delta(k["deces_semaine"], k.get("deces_semaine_prev")),
        )
        k3.metric(
            "Létalité (%)",
            format_metric_value(k["cfr_semaine"], decimals=2),
            delta=_format_sitrep_metric_delta(k["cfr_semaine"], k.get("cfr_semaine_prev")),
        )
        k4.metric("Provinces actives", format_metric_value(k["provinces_reporting"]))
        k5.metric("ZS actives", format_metric_value(k["zs_reporting"]))

        if previous_week_label:
            st.caption(f"Comparaison de {selected_week_label} avec {previous_week_label}.")
        else:
            st.caption(f"Aucune semaine de référence antérieure n’est disponible pour comparer {selected_week_label}.")

        st.caption(
            (
                f"Cumul annuel jusqu’à {selected_week_label} : {format_metric_value(k['cas_cumul'])} cas, "
                f"{format_metric_value(k['deces_cumul'])} décès (létalité {format_metric_value(k['cfr_cumul'], decimals=2)}%)."
            )
        )

        if isinstance(selected_df, pd.DataFrame) and selected_df.empty:
            st.warning("La semaine sélectionnée ne contient aucun cas dans le périmètre filtré actuel.")
            render_reader_narrative(
                "Lecture de l'absence de cas",
                "Cette absence peut refléter une situation réellement calme, un retard de notification ou un filtre trop restrictif. "
                "Avant diffusion, il est utile de vérifier la complétude des rapports et les filtres appliqués.",
                tone="missing",
            )

        summary_lines = sitrep_payload.get("summary_lines", [])
        if summary_lines:
            st.markdown("**Résumé automatique**")
            st.markdown("\n".join([f"- {line}" for line in summary_lines]))

        decision_focus = sitrep_payload.get("decision_focus", [])
        if decision_focus:
            st.markdown("**Priorités opérationnelles immédiates**")
            st.markdown("\n".join([f"- {line}" for line in decision_focus]))

        st.divider()
        render_section_title(9, "Foyers géographiques et dynamique")
        st.caption(
            "Le SITREP montre ici les foyers principaux et la dynamique utile à la décision. Les tableaux exhaustifs restent repliés pour éviter de surcharger la lecture."
        )

        geo1, geo2 = st.columns(2)
        with geo1:
            st.markdown("**Top 5 provinces de la semaine ciblée**")
            top_prov = sitrep_payload.get("top_prov_focus")
            if isinstance(top_prov, pd.DataFrame) and not top_prov.empty:
                st.caption(
                    _describe_surveillance_top_table(
                        top_prov,
                        [COL_PROV],
                        int(len(selected_df)) if isinstance(selected_df, pd.DataFrame) else 0,
                        "Aucune province exploitable pour cette semaine.",
                    )
                )
                st.dataframe(top_prov, width="stretch", hide_index=True)
            else:
                st.info("Aucune province exploitable n’est disponible pour la semaine sélectionnée.")

        with geo2:
            st.markdown("**Top 5 zones de santé de la semaine ciblée**")
            top_zs = sitrep_payload.get("top_zs_focus")
            zs_group_cols = [c for c in [COL_PROV, COL_ZS] if c in selected_df.columns] if isinstance(selected_df, pd.DataFrame) else []
            if isinstance(top_zs, pd.DataFrame) and not top_zs.empty:
                st.caption(
                    _describe_surveillance_top_table(
                        top_zs,
                        zs_group_cols,
                        int(len(selected_df)) if isinstance(selected_df, pd.DataFrame) else 0,
                        "Aucune zone de santé exploitable pour cette semaine.",
                    )
                )
                st.dataframe(top_zs, width="stretch", hide_index=True)
            else:
                st.info("Aucune zone de santé exploitable n’est disponible pour la semaine sélectionnée.")

        with st.expander("Afficher le tableau détaillé par zone de santé", expanded=False):
            table_epi = sitrep_payload.get("table_epi")
            if table_epi is not None and isinstance(table_epi, pd.DataFrame) and not table_epi.empty:
                st_dataframe_safe(table_epi, height=520)
            else:
                st.caption(
                    "Le tableau détaillé des zones de santé est indisponible : absence de données sur la période sélectionnée ou variable ZS manquante."
                )

        st.divider()
        render_section_title(10, "Signaux utiles à la décision")
        st.caption(
            "Cette section ne reprend que les signaux directement utiles à l’action immédiate. Les analyses détaillées restent accessibles dans leurs onglets spécialisés."
        )

        provcrit = sitrep_payload.get("prov_cfr_crit")
        zscrit = sitrep_payload.get("zs_cfr_crit")
        n_investigated = (
            int(pd.to_datetime(selected_df[DATE_INV], errors="coerce").notna().sum())
            if isinstance(selected_df, pd.DataFrame) and DATE_INV in selected_df.columns else 0
        )
        n_tdr = (
            int(_is_yes_series(selected_df[COL_TDR]).sum())
            if isinstance(selected_df, pd.DataFrame) and COL_TDR in selected_df.columns else 0
        )

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Cas investigués", format_metric_value(n_investigated))
        s2.metric("TDR documentés", format_metric_value(n_tdr))
        s3.metric(
            "Prov. CFR critique",
            format_metric_value(len(provcrit)) if isinstance(provcrit, pd.DataFrame) else "0",
        )
        s4.metric(
            "ZS CFR critique",
            format_metric_value(len(zscrit)) if isinstance(zscrit, pd.DataFrame) else "0",
        )

        signal_lines = []
        critical_summary = _build_sitrep_critical_cfr_summary(sitrep_payload)
        if critical_summary:
            signal_lines.append(critical_summary)
        alert_summary = _build_sitrep_alert_summary(sitrep_payload.get("alertes_last"))
        if alert_summary:
            signal_lines.append(alert_summary)

        if signal_lines:
            st.markdown("**Points de vigilance**")
            st.markdown("\n".join([f"- {line}" for line in signal_lines]))

        with st.expander("Cascade biologique et investigation", expanded=False):
            cascad = sitrep_payload.get("cascade")
            if cascad is not None and isinstance(cascad, pd.DataFrame) and not cascad.empty:
                st.markdown("**Cascade prélèvement → TDR → résultat**")
                st_dataframe_safe(cascad, height=320)
            else:
                st.caption("La cascade est indisponible : fonction absente, variables manquantes ou absence de données sur la semaine sélectionnée.")

        with st.expander("Alertes statistiques", expanded=False):
            al = sitrep_payload.get("alertes_last")
            if al is not None and isinstance(al, pd.DataFrame) and not al.empty:
                cols = [c for c in ["YW", "Cas", "Cas_prev", "var_%", "baseline_3w", "signal"] if c in al.columns]
                st_dataframe_safe(al[cols] if cols else al, height=420)
            else:
                st.caption("Les alertes sont indisponibles : fonction absente ou historique insuffisant.")

        with st.expander("Létalité critique : détail provinces et zones de santé", expanded=False):
            provt = sitrep_payload.get("prov_table")
            if provt is not None and isinstance(provt, pd.DataFrame) and not provt.empty:
                st.markdown("**Provinces — cas, décès et létalité (semaine sélectionnée)**")
                st_dataframe_safe(provt, height=300)

            if provcrit is not None and isinstance(provcrit, pd.DataFrame) and not provcrit.empty:
                st.markdown(f"**Provinces à CFR critique (Cas ≥ {int(min_cas_prov)})**")
                st_dataframe_safe(provcrit, height=260)
            else:
                st.caption("Aucune province ne dépasse le seuil critique défini.")

            if zscrit is not None and isinstance(zscrit, pd.DataFrame) and not zscrit.empty:
                st.markdown(f"**ZS à CFR critique (Cas ≥ {int(min_cas_zs)})**")
                st_dataframe_safe(zscrit.head(30), height=420)
            else:
                st.caption("Aucune ZS ne dépasse le seuil critique défini.")

        st.divider()
        render_section_title(11, "Articulation avec les autres onglets")
        st.caption("Chaque onglet garde une fonction distincte pour éviter les doublons dans le tableau de bord.")
        st.markdown(
            "\n".join(
                [
                    "- `SITREP` : synthèse courte, foyers prioritaires, signaux et export PDF.",
                    "- `Surveillance` : dynamique par fenêtres temporelles, létalité et analyses détaillées de promptitude.",
                    "- `Profil` : structure démographique, tableaux descriptifs détaillés et stratifications.",
                    "- `Données, complétude & qualité` : cohérence, doublons, complétude et tableaux de contrôle détaillés.",
                ]
            )
        )

        # =========================================================
        # 5) Exportation PDF
        # =========================================================
        st.divider()
        render_section_title(12, "Exportation")

        if "export_sitrep_pdf" in globals() and callable(export_sitrep_pdf):
            cexp1, cexp2 = st.columns([1, 1])

            with cexp1:
                prepare_pdf = st.button(
                    "Préparer le SITREP PDF",
                    type="primary",
                    key="prepare_sitrep_pdf_btn",
                )

            with cexp2:
                include_pdf_images = st.checkbox(
                    "Inclure les graphiques dans le PDF",
                    value=False,
                    key="include_pdf_images_chk",
                    help="Option plus lourde sur Streamlit Cloud. À activer seulement si nécessaire.",
                )

            if prepare_pdf:
                with st.spinner("Préparation du PDF en cours..."):
                    try:
                        pdf_payload = _build_sitrep_payload_from_df(
                            df_f,
                            semaine,
                            annee,
                            date_pub,
                            min_cas_zs=min_cas_zs,
                            min_cas_prov=min_cas_prov,
                            include_images=include_pdf_images,
                        )

                        pdf_bytes = export_sitrep_pdf(pdf_payload)

                        st.download_button(
                            "⬇️ Télécharger le SITREP épidémiologique (PDF)",
                            data=pdf_bytes,
                            file_name=f"SITREP_epidemiologique_CHOLERA_SE{int(semaine):02d}_{int(annee)}.pdf",
                            mime="application/pdf",
                            key="sitrep_dl_pdf",
                        )

                        if include_pdf_images:
                            st.caption("PDF généré avec tentative d’inclusion des graphiques.")
                        else:
                            st.caption("PDF généré sans graphiques intégrés pour maximiser la stabilité.")
                    except Exception as e:
                        st.error(f"Erreur lors de l’exportation PDF : {e}")
        else:
            st.error("La fonction export_sitrep_pdf(payload) n’est pas définie dans ce script.")

# =========================
# TAB 9 — IDSR : Helpers robuste
# =========================

