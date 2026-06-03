"""Render the data quality and export tab."""

from dashboard_app.domain import (
    TDR_NEG_SET,
    TDR_POS_SET,
    _is_yes_series,
    _standard_surveillance_evidence_masks,
    _standard_test_documented_mask,
    _tdr_result_norm,
)
from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


def _quality_build_export_traceability_table(
    df_scope: pd.DataFrame,
    disease_key_value: str,
    files_used_tuple: tuple[str, ...],
) -> pd.DataFrame:
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
        {"Paramètre": "Maladie / line list", "Valeur": DISEASE_SPECS.get(disease_key_value, {}).get("label", disease_key_value)},
        {"Paramètre": "Sources chargées", "Valeur": ", ".join(files_used_tuple) if files_used_tuple else "Non documenté"},
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


_QUALITY_NA_VALUES = {
    "", " ", "-", "n/a", "na", "nan", "null", "none",
    "<na>", "<nat>", "<null>",
    "inconnu", "non renseigné", "non renseigne", "aucun", "aucune",
    "aucune information", "aucune donnée", "aucune donnee",
    "aucune donnée renseignée", "aucune donnee renseignee",
}


def _quality_is_na_like(val: object) -> bool:
    """Retourne True si val doit être considéré comme manquant."""
    if val is None or pd.isna(val):
        return True
    if isinstance(val, str):
        return val.strip().casefold() in _QUALITY_NA_VALUES
    return False


def _quality_decision_missing(
    pct_missing: float,
    *,
    colonne_absente: bool,
    seuil_acceptable: float,
    seuil_surveillance: float,
) -> str:
    if colonne_absente:
        return "Colonne absente"
    if pct_missing == 0:
        return "OK"
    if pct_missing <= seuil_acceptable:
        return "Acceptable"
    if pct_missing <= seuil_surveillance:
        return "A surveiller"
    return "Prioritaire"


@st.cache_data(show_spinner=False)
def analyser_missing_colonnes(
    df: pd.DataFrame,
    colonnes: Optional[Iterable[str]] = None,
    *,
    considerer_na_like: bool = True,
    seuil_acceptable: float = 5.0,
    seuil_surveillance: float = 20.0,
    observations: Optional[dict[str, str]] = None,
    arrondi: int = 2,
) -> pd.DataFrame:
    """
    Analyse les valeurs manquantes des colonnes choisies.

    Si `colonnes=None`, la fonction analyse directement toutes les colonnes
    présentes dans le DataFrame fourni.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un DataFrame pandas.")
    if seuil_acceptable < 0 or seuil_surveillance < 0:
        raise ValueError("Les seuils de missing doivent être positifs.")
    if seuil_acceptable > seuil_surveillance:
        raise ValueError("seuil_acceptable ne peut pas être supérieur à seuil_surveillance.")

    observations = observations or {}
    total_lignes = len(df)

    if colonnes is None:
        colonnes_a_analyser = df.columns.tolist()
    elif isinstance(colonnes, str):
        colonnes_a_analyser = [colonnes]
    else:
        colonnes_a_analyser = list(dict.fromkeys(colonnes))

    colonnes_presentes = {col for col in colonnes_a_analyser if col in df.columns}

    resultats = []
    for colonne in colonnes_a_analyser:
        colonne_absente = colonne not in colonnes_presentes

        if colonne_absente:
            nb_manquantes = total_lignes
            nb_renseignees = 0
        else:
            serie = df[colonne]
            masque_missing = serie.map(_quality_is_na_like) if considerer_na_like else serie.isna()
            nb_manquantes = int(masque_missing.sum())
            nb_renseignees = int(total_lignes - nb_manquantes)

        pct_missing = round((nb_manquantes / total_lignes) * 100, arrondi) if total_lignes > 0 else 0.0
        decision = _quality_decision_missing(
            pct_missing,
            colonne_absente=colonne_absente,
            seuil_acceptable=seuil_acceptable,
            seuil_surveillance=seuil_surveillance,
        )

        resultats.append(
            {
                "Variable": colonne,
                "Total lignes": total_lignes,
                "Renseignées": nb_renseignees,
                "Manquantes": nb_manquantes,
                "% missing": pct_missing,
                "Décision / observation": observations.get(colonne, decision),
            }
        )

    return pd.DataFrame(resultats)


def _quality_to_datetime_series(
    serie: pd.Series,
    *,
    min_date: str = "2000-01-01",
    future_tolerance_days: int = 14,
) -> pd.Series:
    """Convertit une série en dates plausibles pour l'analyse de promptitude."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        dt = pd.to_datetime(serie, errors="coerce")
    else:
        try:
            dt = pd.to_datetime(serie, errors="coerce", format="mixed")
        except TypeError:
            dt = pd.to_datetime(serie, errors="coerce")
    if dt.empty:
        return dt

    borne_min = pd.Timestamp(min_date)
    borne_max = pd.Timestamp(date.today()) + pd.Timedelta(days=future_tolerance_days)
    dt = dt.where(dt.between(borne_min, borne_max), pd.NaT)
    return dt


@st.cache_data(show_spinner=False)
def construire_table_promptitude(
    df: pd.DataFrame,
    date_pairs: Iterable[tuple[str, str, str]],
    *,
    seuil_jours: float = 7.0,
) -> pd.DataFrame:
    """
    Construit un tableau de promptitude entre paires de dates.

    Chaque tuple de `date_pairs` suit le format :
    (libellé, colonne_date_debut, colonne_date_fin)
    """
    lignes = []

    for libelle, col_debut, col_fin in date_pairs:
        if col_debut not in df.columns or col_fin not in df.columns:
            continue

        debut = _quality_to_datetime_series(df[col_debut])
        fin = _quality_to_datetime_series(df[col_fin])

        dispo_debut = debut.notna()
        dispo_fin = fin.notna()
        comparables = dispo_debut & dispo_fin

        n_source = int(dispo_debut.sum())
        n_cible = int(dispo_fin.sum())
        n_comparables = int(comparables.sum())
        n_cible_manquante = int((dispo_debut & ~dispo_fin).sum())

        if n_comparables > 0:
            delais = (fin[comparables] - debut[comparables]).dt.days.astype(float)
            delais_valides = delais.dropna()
            pct_seuil = round(float((delais_valides <= seuil_jours).mean() * 100), 2) if not delais_valides.empty else np.nan
            pct_negatif = round(float((delais_valides < 0).mean() * 100), 2) if not delais_valides.empty else np.nan
            mediane = round(float(delais_valides.median()), 1) if not delais_valides.empty else np.nan
            p90 = round(float(delais_valides.quantile(0.90)), 1) if not delais_valides.empty else np.nan
            delai_max = round(float(delais_valides.max()), 1) if not delais_valides.empty else np.nan
        else:
            pct_seuil = np.nan
            pct_negatif = np.nan
            mediane = np.nan
            p90 = np.nan
            delai_max = np.nan

        lignes.append(
            {
                "Étape": libelle,
                "Date début": col_debut,
                "Date fin": col_fin,
                "Lignes avec date début": n_source,
                "Lignes avec date fin": n_cible,
                "Lignes comparables": n_comparables,
                "Date fin manquante (%)": round((n_cible_manquante / n_source) * 100, 2) if n_source > 0 else np.nan,
                "Médiane délai (jours)": mediane,
                "P90 délai (jours)": p90,
                "Délai max (jours)": delai_max,
                f"% <= {int(seuil_jours)} jours": pct_seuil,
                "% délais négatifs": pct_negatif,
            }
        )

    return pd.DataFrame(lignes)


def construire_resume_coherence_par_groupe(
    df: pd.DataFrame,
    flags: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Résume les incohérences QC par groupe géographique ou temporel."""
    if (
        not isinstance(df, pd.DataFrame)
        or not isinstance(flags, pd.DataFrame)
        or group_col not in df.columns
        or flags.empty
        or "row_id" not in flags.columns
        or "flag" not in flags.columns
    ):
        return pd.DataFrame()

    base = df.reset_index().rename(columns={"index": "row_id"}).copy()
    base[group_col] = base[group_col].fillna("Inconnu").astype(str)

    total_cases = (
        base.groupby(group_col, dropna=False)
        .size()
        .rename("Cas")
        .reset_index()
    )

    flags_detail = flags.merge(base[["row_id", group_col]], on="row_id", how="left")
    flags_detail[group_col] = flags_detail[group_col].fillna("Inconnu").astype(str)

    rows_flagged = (
        flags_detail.groupby(group_col)["row_id"]
        .nunique()
        .rename("Lignes avec incohérences")
        .reset_index()
    )

    total_flags = (
        flags_detail.groupby(group_col)
        .size()
        .rename("Total incohérences")
        .reset_index()
    )

    dominant = (
        flags_detail.groupby([group_col, "flag"])
        .size()
        .rename("Occurrences")
        .reset_index()
        .sort_values([group_col, "Occurrences"], ascending=[True, False])
        .drop_duplicates(subset=[group_col])
        .rename(columns={"flag": "Incohérence dominante", "Occurrences": "Occurrences dominante"})
    )

    resume = total_cases.merge(rows_flagged, on=group_col, how="left")
    resume = resume.merge(total_flags, on=group_col, how="left")
    resume = resume.merge(dominant[[group_col, "Incohérence dominante", "Occurrences dominante"]], on=group_col, how="left")

    for col in ["Lignes avec incohérences", "Total incohérences", "Occurrences dominante"]:
        resume[col] = pd.to_numeric(resume[col], errors="coerce").fillna(0).astype(int)

    resume["% lignes touchées"] = ((resume["Lignes avec incohérences"] / resume["Cas"]) * 100).round(2)
    resume["Incohérences / 100 cas"] = ((resume["Total incohérences"] / resume["Cas"]) * 100).round(2)
    resume = resume.sort_values(
        ["% lignes touchées", "Total incohérences", "Cas"],
        ascending=[False, False, False],
    )
    return resume


@st.cache_data(show_spinner=False)
def _quality_build_cascade_group_summary(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Construit un résumé groupé de la chaîne laboratoire en un seul passage."""
    columns = [group_col, "n", "% prélèvement", "% test documenté", "% résultat valide", "% positif", "% incoh TDR"]
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    work[group_col] = work[group_col].fillna("Inconnu").astype(str).str.strip().replace("", "Inconnu")

    evidence = _standard_surveillance_evidence_masks(work)
    prelev_documented = evidence["prelev_documented"].fillna(False)
    test_documented = _standard_test_documented_mask(work).fillna(False)
    result_documented = evidence["result_documented"].fillna(False)

    result_col = None
    if COL_TDRR in work.columns and work[COL_TDRR].notna().any():
        result_col = COL_TDRR
    elif "Resultat_labo" in work.columns and work["Resultat_labo"].notna().any():
        result_col = "Resultat_labo"

    if result_col is not None:
        res_n = _tdr_result_norm(work[result_col])
    else:
        res_n = pd.Series(pd.NA, index=work.index, dtype="string")

    valid_result_mask = prelev_documented & test_documented & res_n.isin(TDR_POS_SET.union(TDR_NEG_SET))
    if "is_tdr_pos" in work.columns:
        is_pos = pd.to_numeric(work["is_tdr_pos"], errors="coerce").fillna(0).astype(int).eq(1)
        positive_mask = valid_result_mask & is_pos
    else:
        positive_mask = prelev_documented & test_documented & res_n.isin(TDR_POS_SET)

    if COL_TDR in work.columns:
        tdr_yes = _is_yes_series(work[COL_TDR]).fillna(False)
    else:
        tdr_yes = pd.Series(False, index=work.index)
    incoherent_result_mask = result_documented & ~tdr_yes

    grouped = (
        pd.DataFrame(
            {
                group_col: work[group_col],
                "_n": 1,
                "_prelev": prelev_documented.astype(int),
                "_test": (prelev_documented & test_documented).astype(int),
                "_valid": valid_result_mask.astype(int),
                "_pos": positive_mask.astype(int),
                "_incoh": incoherent_result_mask.astype(int),
            }
        )
        .groupby(group_col, as_index=False)
        .sum()
        .rename(
            columns={
                "_n": "n",
                "_prelev": "n_prelev",
                "_test": "n_test",
                "_valid": "n_valid",
                "_pos": "n_pos",
                "_incoh": "n_incoh",
            }
        )
    )
    grouped["% prélèvement"] = np.where(grouped["n"] > 0, grouped["n_prelev"] / grouped["n"] * 100.0, np.nan)
    grouped["% test documenté"] = np.where(grouped["n_prelev"] > 0, grouped["n_test"] / grouped["n_prelev"] * 100.0, np.nan)
    grouped["% résultat valide"] = np.where(grouped["n_test"] > 0, grouped["n_valid"] / grouped["n_test"] * 100.0, np.nan)
    grouped["% positif"] = np.where(grouped["n_valid"] > 0, grouped["n_pos"] / grouped["n_valid"] * 100.0, np.nan)
    grouped["% incoh TDR"] = np.where(grouped["n"] > 0, grouped["n_incoh"] / grouped["n"] * 100.0, np.nan)
    return grouped[columns].sort_values("n", ascending=False).reset_index(drop=True)


def masquer_identifiants_techniques(
    df: pd.DataFrame,
    colonnes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Retire des vues les identifiants techniques non utiles à la lecture."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    colonnes_a_masquer = list(colonnes) if colonnes is not None else [
        "Signal_ID",
        "ID signal",
        "row_id",
        "__source_file__",
        "source_file",
        "Source_fichier",
    ]
    colonnes_presentes = [col for col in colonnes_a_masquer if col in df.columns]
    if not colonnes_presentes:
        return df
    return df.drop(columns=colonnes_presentes)


def render_quality_tab(ctx: dict) -> None:
    """Render the data quality and export tab."""
    globals().update(ctx)
    render_section_title(5, "Qualité des données, promptitude et cohérence opérationnelle")
    if IDSR_MODE:
        render_absence_narrative("idsr_line_list")
    else:
        render_tab_narrative("qualite")
        tab_help(
            "Comment lire cet onglet",
            """
            **Objectif** : appuyer la revue opérationnelle de la qualité des line lists pour la coordination du COUSP/RDC.

            **Lecture recommandée**
            - **Couverture de notification** : identifie les provinces attendues mais non observées dans le rapport courant.
            - **Cohérence** : repère les anomalies de saisie et les enregistrements incompatibles avec les règles métier.
            - **Complétude** : mesure le niveau de renseignement de l’ensemble des colonnes utiles à l’analyse.
            - **Promptitude** : apprécie les délais entre détection, notification, investigation et étapes de laboratoire.

            **Point d’attention**
            - Un signal de qualité ne confirme pas à lui seul une défaillance opérationnelle ; il doit guider la vérification.
            """,
            expanded=False
        )
        if isinstance(df_f, pd.DataFrame):
            st.caption(build_standard_capability_note(df_f))
            with st.expander("Repere standard multi-maladies", expanded=False):
                st.dataframe(build_standard_disease_profile(disease_key, df_f), width="stretch", hide_index=True)
        
        with st.expander("Paramétrage des provinces attendues", expanded=False):

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

            st.markdown("Sélectionner les provinces considérées comme attendues dans la line list pour le suivi de couverture.")

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
       
        st.subheader("Couverture de notification des provinces attendues")
        
        if COL_PROV not in df_f.columns:
            st.info("La variable Province_notification est absente du fichier analysé.")
        else:
            if COL_WNUM in df_f.columns and df_f[COL_WNUM].notna().any():
                last_w = int(df_f[COL_WNUM].max())
                present = sorted(df_f.loc[df_f[COL_WNUM] == last_w, COL_PROV].dropna().unique().tolist())
                st.caption(f"Lecture effectuée sur la semaine épidémiologique la plus récente du périmètre filtré : SE{last_w:02d}")
            else:
                present = sorted(df_f[COL_PROV].dropna().unique().tolist())
                st.caption("Lecture effectuée sur l’ensemble filtré, faute de semaine épidémiologique exploitable.")
        
            missing = [p for p in PROVINCES_EPID if p not in present]
            nb_att = len(PROVINCES_EPID)
            nb_rec = len([p for p in PROVINCES_EPID if p in present])
            compl = (nb_rec / nb_att * 100) if nb_att > 0 else np.nan
        
            c1, c2, c3 = st.columns(3)
            c1.metric("Provinces attendues", str(nb_att))
            c2.metric("Provinces trouvées", str(nb_rec))
            c3.metric("Complétude (%)", f"{compl:.1f}")
            if missing:
                st.warning("Provinces attendues non observées dans les données : " + ", ".join(missing))
        
            with st.expander("Tableau de couverture des provinces attendues"):
                df_comp = pd.DataFrame({
                    "Province attendue": PROVINCES_EPID,
                    "Présente": [p in present for p in PROVINCES_EPID],
                    "Manquante": [p if p in missing else "" for p in PROVINCES_EPID],
                })
                st_dataframe_safe(df_comp)
        
            with st.expander("Volume de cas rapportés par province", expanded=True):
                prov_counts = df_f[COL_PROV].fillna("Inconnu").value_counts().reset_index()
                prov_counts.columns = [COL_PROV, "Cas"]
                figp = px.bar(prov_counts, x=COL_PROV, y="Cas", title=" ")
                figp.update_layout(xaxis_tickangle=-45)
                figp = apply_plotly_value_annotations(figp, annot_vals)
                st.plotly_chart(figp, width="stretch")
        
            # TCD
            with st.expander("Répartition détaillée des occurrences", expanded=False):
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
    st.markdown("### Extraction, revue et export des données")
    if IDSR_MODE:
        render_absence_narrative("idsr_line_list")
    else:
        render_tab_narrative("export")
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : consulter et exporter les données filtrées pour analyse et partage.
        
            **📖 Utilisation**
            - Exportation **CSV/Excel** pour analyses complémentaires (R/Python/DHIS2).
            - Vérifier les filtres actifs avant export.
        
            **Points d'attention**
            - Les exports reflètent exactement les filtres actifs (province/ZS/AS/semaine/classification).
            """,
            expanded=False
        )
        
        st.subheader("Extraction des données filtrées, traçabilité et options d’export")

        @st.cache_data(show_spinner=False)
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

        export_base_name = f"{str(disease_key).strip().lower()}_filtre"
        prepare_export_pack = st.checkbox(
            "Préparer le pack d'export qualité (Excel + tableaux avancés)",
            value=False,
            key="quality_prepare_export_pack",
            help="Laisse l'onglet chargé immédiatement et ne prépare les tableaux avancés et l'Excel complet que si nécessaire.",
        )
        export_traceability = _build_export_traceability_table(df_f)
        export_quality_summary = standard_data_quality_summary(df_f) if prepare_export_pack else pd.DataFrame()
        export_qc_flags = qc_flags(df_f) if prepare_export_pack else pd.DataFrame()
        export_qc_resume = (
            export_qc_flags["flag"].value_counts().rename_axis("Flag").reset_index(name="Occurrences")
            if not export_qc_flags.empty else pd.DataFrame(columns=["Flag", "Occurrences"])
        )
        export_duplicates = duplicate_candidates_table(df_f) if prepare_export_pack else pd.DataFrame()
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
            if prepare_export_pack and export_risk_group else pd.DataFrame()
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
            if prepare_export_pack and export_risk_group and (("YW" in df_f.columns) or (COL_WEEK in df_f.columns)) else pd.DataFrame()
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
            if prepare_export_pack and (("YW" in df_f.columns) or (COL_WEEK in df_f.columns)) else pd.DataFrame()
        )
        export_standard_signals = (
            build_standard_signal_table(
                df_f,
                week_col="YW" if "YW" in df_f.columns else COL_WEEK,
                completeness_threshold=80.0,
                timeliness_threshold_days=float(seuil_jours),
                timeliness_target_pct=80.0,
                investigation_target_pct=90.0,
                positivity_high_threshold=40.0,
                cfr_high_threshold=3.0,
                min_alert_cases=10,
                alert_ratio=1.5,
            )
            if prepare_export_pack else pd.DataFrame()
        )
        export_standard_tracker = (
            build_standard_action_tracker_template(
                export_standard_signals,
                disease_label=DISEASE_SPECS.get(disease_key, {}).get("label", disease_key),
                analysis_label=compute_analysis_period_value(df_f),
                generated_on=str(date.today()),
            )
            if prepare_export_pack else pd.DataFrame()
        )

        export_completeness = pd.DataFrame()
        export_completeness_by = None
        export_required_fields = [
            COL_PROV, COL_ZS, COL_AS, "YW", COL_WNUM, COL_SEX, COL_AGE,
            COL_UNIT, DATE_ONSET, COL_PREL, COL_TDR, COL_TDRR, COL_HOSP,
            COL_ISSUE, COL_CLASS,
        ]
        if prepare_export_pack:
            for group_col in [COL_PROV, COL_ZS, "YW", COL_WNUM]:
                if group_col in df_f.columns and df_f[group_col].notna().any():
                    export_completeness = completeness_table(df_f, export_required_fields, by=group_col)
                    if not export_completeness.empty:
                        export_completeness_by = group_col
                        break

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
        if not prepare_export_pack:
            st.caption(
                "Préparation avancée désactivée : les tableaux secondaires et l'Excel complet seront calculés uniquement si vous activez l'option ci-dessus."
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
                st_dataframe_safe(
                    masquer_identifiants_techniques(export_duplicates.head(100)),
                    height=260,
                )
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

            if not export_standard_signals.empty:
                st.markdown("**Points à suivre en priorité**")
                st_dataframe_safe(
                    masquer_identifiants_techniques(export_standard_signals),
                    height=300,
                )

        st.divider()
        st.markdown("**Suivi des actions**")
        st.caption(
            "Le tableau ci-dessous est prérempli à partir des points à suivre. "
            "Tu peux compléter les responsables, les échéances, l'avancement et les commentaires avant export."
        )
        tracker_signature = hashlib.md5(
            "|".join(export_standard_tracker.get("Signal_ID", pd.Series(dtype="string")).astype(str).tolist()).encode("utf-8")
        ).hexdigest()
        tracker_state_key = "quality_standard_action_tracker"
        tracker_signature_key = "quality_standard_action_tracker_signature"

        if st.session_state.get(tracker_signature_key) != tracker_signature:
            existing_tracker = st.session_state.get(tracker_state_key)
            st.session_state[tracker_state_key] = merge_standard_action_tracker_template(export_standard_tracker, existing_tracker)
            st.session_state[tracker_signature_key] = tracker_signature

        tracker_col1, tracker_col2 = st.columns([1, 1])
        with tracker_col1:
            if st.button("Réinitialiser le tracker", key="quality_standard_tracker_reset"):
                st.session_state[tracker_state_key] = export_standard_tracker.copy()
                st.session_state[tracker_signature_key] = tracker_signature
        with tracker_col2:
            if not export_standard_tracker.empty:
                st.caption(f"{len(export_standard_tracker)} action(s) proposée(s) pour le périmètre courant.")

        current_tracker = st.session_state.get(tracker_state_key, export_standard_tracker.copy())
        if current_tracker is None or not isinstance(current_tracker, pd.DataFrame):
            current_tracker = export_standard_tracker.copy()
        current_tracker = merge_standard_action_tracker_template(export_standard_tracker, current_tracker)

        if current_tracker.empty:
            st.success("Aucun point prioritaire n'alimente le suivi des actions pour le périmètre filtré.")
        else:
            edited_tracker = st.data_editor(
                current_tracker,
                width="stretch",
                height=360,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "Signal_ID": None,
                    "Maladie_source": st.column_config.TextColumn("Maladie", disabled=True),
                    "Perimetre_analyse": st.column_config.TextColumn("Périmètre", disabled=True),
                    "Bloc": st.column_config.TextColumn("Bloc", disabled=True),
                    "Indicateur": st.column_config.TextColumn("Indicateur", disabled=True),
                    "Statut_signal": st.column_config.TextColumn("Niveau d'alerte", disabled=True),
                    "Priorite_action": st.column_config.SelectboxColumn(
                        "Priorité",
                        options=["Urgent", "Cette semaine", "Routine"],
                    ),
                    "Niveau_reponse": st.column_config.SelectboxColumn(
                        "Niveau",
                        options=["Province", "Surveillance", "Technique / soins", "Coordination", "National", "Autre"],
                    ),
                    "Constat": st.column_config.TextColumn("Ce qu'on observe", disabled=True, width="large"),
                    "Action_a_suivre": st.column_config.TextColumn("Action proposée", width="large"),
                    "Responsable": st.column_config.TextColumn("Responsable"),
                    "Echeance": st.column_config.TextColumn("Échéance"),
                    "Statut_action": st.column_config.SelectboxColumn(
                        "Avancement",
                        options=["À démarrer", "À suivre", "En cours", "Terminé", "Bloqué", "Planifié"],
                    ),
                    "Commentaire": st.column_config.TextColumn("Commentaire", width="large"),
                    "Date_generation": st.column_config.TextColumn("Date de génération", disabled=True),
                },
                key="quality_standard_action_tracker_editor",
            )
            st.session_state[tracker_state_key] = edited_tracker.copy()
            current_tracker = edited_tracker.copy()

            tracker_open = int(current_tracker["Statut_action"].astype(str).isin(["À démarrer", "À suivre", "En cours"]).sum())
            tracker_done = int(current_tracker["Statut_action"].astype(str).eq("Terminé").sum())
            tracker_blocked = int(current_tracker["Statut_action"].astype(str).eq("Bloqué").sum())
            t1, t2, t3 = st.columns(3)
            t1.metric("Actions ouvertes", format_metric_value(tracker_open))
            t2.metric("Actions terminées", format_metric_value(tracker_done))
            t3.metric("Actions bloquées", format_metric_value(tracker_blocked))

            tracker_export = current_tracker.rename(
                columns={
                    "Signal_ID": "ID signal",
                    "Maladie_source": "Maladie",
                    "Perimetre_analyse": "Périmètre",
                    "Statut_signal": "Niveau d'alerte",
                    "Priorite_action": "Priorité",
                    "Niveau_reponse": "Niveau",
                    "Constat": "Ce qu'on observe",
                    "Action_a_suivre": "Action proposée",
                    "Echeance": "Échéance",
                    "Statut_action": "Avancement",
                    "Date_generation": "Date de génération",
                }
            )
            tracker_csv = tracker_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger le suivi des actions (CSV)",
                data=tracker_csv,
                file_name=f"{export_base_name}_tracker_actions.csv",
                mime="text/csv",
                key="dl_quality_export_csv_tracker",
            )

        st.markdown("**Aperçu de la line list filtrée**")
        st_dataframe_safe(
            masquer_identifiants_techniques(df_f),
            height=420,
        )

        export_mode = st.radio(
            "Type d’export",
            ["Pack qualité + line list", "Line list uniquement"],
            index=0,
            horizontal=True,
            key="qualite_export_mode",
        )

        excel_bytes = None
        excel_name = None
        excel_export_error = False
        if prepare_export_pack:
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
                        if not export_standard_signals.empty:
                            export_standard_signals.to_excel(writer, sheet_name="Points_prioritaires", index=False)
                        if current_tracker is not None and isinstance(current_tracker, pd.DataFrame) and not current_tracker.empty:
                            tracker_export.to_excel(writer, sheet_name="Suivi_actions", index=False)

                excel_name = (
                    f"{export_base_name}_pack_qualite.xlsx"
                    if export_mode == "Pack qualité + line list"
                    else f"{export_base_name}.xlsx"
                )
                excel_bytes = buffer.getvalue()
            except Exception:
                excel_export_error = True

        csv_col, excel_col = st.columns(2)
        with csv_col:
            st.download_button(
                "Télécharger CSV line list",
                data=df_to_csv_bytes(df_f),
                file_name=f"{export_base_name}.csv",
                mime="text/csv",
                key="dl_quality_export_csv_ll",
                use_container_width=True,
            )
        with excel_col:
            if excel_bytes is not None and excel_name is not None:
                st.download_button(
                    "Télécharger Excel",
                    data=excel_bytes,
                    file_name=excel_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_quality_export_xlsx",
                    use_container_width=True,
                )
            elif excel_export_error:
                st.info("Export Excel indisponible.")
            elif not prepare_export_pack:
                st.info("Activez la préparation du pack pour générer l'Excel.")

        extra_exports = []
        if not export_qc_flags.empty:
            extra_exports.append(
                (
                    "Télécharger QC flags (CSV)",
                    export_qc_flags.to_csv(index=False).encode("utf-8"),
                    f"{export_base_name}_qc_flags.csv",
                    "dl_quality_export_csv_qc",
                )
            )
        if not export_duplicates.empty:
            extra_exports.append(
                (
                    "Télécharger doublons (CSV)",
                    export_duplicates.to_csv(index=False).encode("utf-8"),
                    f"{export_base_name}_doublons.csv",
                    "dl_quality_export_csv_dup",
                )
            )

        if extra_exports:
            extra_cols = st.columns(len(extra_exports))
            for col, (label, data, file_name, key) in zip(extra_cols, extra_exports):
                with col:
                    st.download_button(
                        label,
                        data=data,
                        file_name=file_name,
                        mime="text/csv",
                        key=key,
                        use_container_width=True,
                    )
        
    # =========================
    # TAB 7 — Labo / qualité / signaux
    # =========================
    st.divider()
    st.markdown("### Qualité des données et alertes de gestion")
    if IDSR_MODE:
        render_absence_narrative("idsr_line_list")
    else:
        render_tab_narrative("qualite")
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : repérer les incohérences, les données manquantes, les difficultés de laboratoire et les alertes.
        
            **📖 Sections**
            - **Indicateurs rapides** : quelques chiffres clés
            - **QC Flags** : incohérences (dates, TDR, âge…)
            - **Complétude champs clés** : niveau de remplissage par site
            - **Cascade labo** : cas → prélèvement documenté → test documenté → résultat valide → positif
            - **Alertes tendance** : hausse inhabituelle par rapport aux semaines précédentes
        
            **Points d'attention**
            - Un signal ne confirme pas à lui seul une épidémie : il faut vérifier sur le terrain.
            - Les % de cascade suivent une logique en étapes successives.
            - Un prélèvement ou un test peut être reconnu par preuve documentaire (date, résultat, réception, numéro labo), même si une case Oui/Non est vide.
            """,
            expanded=False
        )
        
        st.subheader("Revue opérationnelle de la qualité des données")
        
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
        
        # KPI "qualité TDR" (sur cascade)
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
            help=f"Prélèvement documenté / Tous les cas filtrés. n={kpi.get('prelev_num', 0)}/{kpi.get('prelev_den', kpi.get('n_cases', 0))}"
        )
        
        c3.metric(
            "Couverture TDR (%)" if has_tdr_chain else "Couverture test (%)",
            "-" if np.isnan(kpi["tdr_pct"]) else f"{kpi['tdr_pct']:.1f}",
            help=(
                f"TDR/tests documentés / Tous les cas filtrés. n={kpi.get('tdr_num', 0)}/{kpi.get('tdr_den', kpi.get('n_cases', 0))}"
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
            help=f"Hospitalisation documentée / Tous les cas filtrés. n={kpi.get('hosp_num', 0)}/{kpi.get('hosp_den', kpi.get('n_cases', 0))}"
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
            with st.expander("Signaux de qualité des données de laboratoire", expanded=False):
                if not np.isnan(kpi_incoh_res_wo_tdr):
                    st.write(f"- **% Résultat documenté mais TDR_realise ≠ Oui / statut test absent**: **{kpi_incoh_res_wo_tdr:.1f}%**")
                if not np.isnan(kpi_status_in_result):
                    st.write(f"- **% Statut saisi dans la colonne de résultat** (ex: non réalisé/non prélevé): **{kpi_status_in_result:.1f}%**")
        
        with st.expander("Cascade de prélèvement et de confirmation", expanded=False):
            st.caption(
                "Cette cascade suit la même logique standard que la chaîne et les relances: un prélèvement ou un test peut être "
                "reconnu par preuve documentaire (date de prélèvement, réception labo, résultat, numéro labo) si le statut Oui/Non manque."
            )
            st_dataframe_safe(casc_global)

        # ==========================================================
        # 0a) Score de risque operationnel par zone/province
        # ==========================================================
        risk_group_options = [
            c for c in [COL_ZS, COL_PROV, COL_AS]
            if c in df_f.columns and df_f[c].notna().any()
        ]
        if risk_group_options:
            with st.expander("Priorisation opérationnelle des zones et provinces", expanded=True):
                r1, r2, r3 = st.columns([1.15, 0.95, 0.95])
                with r1:
                    risk_group_col = st.selectbox(
                        "Niveau géographique",
                        options=risk_group_options,
                        key="operational_risk_group_col",
                    )
                with r2:
                    risk_recent_weeks = st.number_input(
                        "Semaines récentes",
                        min_value=2,
                        max_value=8,
                        value=4,
                        step=1,
                        key="operational_risk_recent_weeks",
                    )
                with r3:
                    risk_topn = st.slider(
                        "Groupes à afficher",
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
                    k_r1.metric("Groupes évalués", f"{len(risk_tbl):,}".replace(",", " "))
                    k_r2.metric("Priorité très élevée", str(int((risk_tbl["Priorite"] == "Tres elevee").sum())))
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
                            title="Score de risque opérationnel",
                        )
                        fig_risk.update_layout(xaxis_title="Score 0-100", yaxis_title=risk_group_col)
                        fig_risk = apply_plotly_value_annotations(fig_risk, annot_vals)
                        st.plotly_chart(fig_risk, width="stretch", key="operational_risk_chart")

                    csv_risk = risk_tbl.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Télécharger le score de risque (CSV)",
                        data=csv_risk,
                        file_name="score_risque_operationnel.csv",
                        mime="text/csv",
                        key="download_operational_risk_score",
                    )

        # ==========================================================
        # 1) QC Flags (incohérences)
        # ==========================================================
        with st.expander("Contrôle de cohérence des enregistrements", expanded=False):
        
        
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
                flag_sel = st.selectbox("Type d’incohérence à afficher", ["Tous"] + flag_list, index=0)
        
                # Détail (merge + colonnes utiles)
                cols_show = [c for c in [
                    "Nom_complet", COL_PROV, COL_ZS, COL_AS, COL_SEX, COL_AGE, COL_UNIT,
                    "YW", COL_WNUM, DATE_ONSET, DATE_ADM, DATE_PREL,
                    DATE_RECEP, DATE_RES,
                    COL_PREL, COL_TDR, COL_TDRR, "Resultat_labo", "Type_de_prelevement", "Nom_laboratoire", "N_labo",
                    "Nombre_dose_recues", "Date_derniere_vaccination",
                    COL_HOSP, COL_ISSUE, COL_CLASS
                ] if c in df_f.columns]
        
                detail = flags.merge(df_f.reset_index().rename(columns={"index": "row_id"}), on="row_id", how="left")
        
                if flag_sel != "Tous":
                    detail = detail[detail["flag"] == flag_sel]
        
                st.caption("Détail des enregistrements concernés, limité à 500 lignes.")
                st.dataframe(detail[["flag"] + cols_show].head(500), width="stretch", height=420)

                coherence_group_choices = [c for c in [COL_PROV, COL_ZS, COL_AS, "YW", COL_WNUM] if c in df_f.columns]
                if coherence_group_choices:
                    st.markdown("**Synthèse territoriale des incohérences**")
                    coherence_group_col = st.selectbox(
                        "Niveau de synthèse",
                        coherence_group_choices,
                        index=0,
                        key="quality_coherence_group_col",
                    )
                    coherence_tbl = construire_resume_coherence_par_groupe(df_f, flags, coherence_group_col)
                    if not coherence_tbl.empty:
                        st_dataframe_safe(coherence_tbl, height=360)
                        st.download_button(
                            "Télécharger la synthèse de cohérence (CSV)",
                            data=coherence_tbl.to_csv(index=False).encode("utf-8"),
                            file_name="quality_coherence_summary.csv",
                            mime="text/csv",
                            key="download_quality_coherence_csv",
                        )

                        topn_coh = st.slider(
                            "Nombre de groupes prioritaires à afficher",
                            min_value=5,
                            max_value=min(50, max(5, len(coherence_tbl))),
                            value=min(20, len(coherence_tbl)),
                            step=5,
                            key="quality_coherence_topn",
                        )
                        coherence_plot = coherence_tbl.head(topn_coh).sort_values("% lignes touchées", ascending=True)
                        fig_coh = px.bar(
                            coherence_plot,
                            x="% lignes touchées",
                            y=coherence_group_col,
                            orientation="h",
                            color="Total incohérences",
                            title="Groupes prioritaires selon le taux d’incohérences"
                        )
                        fig_coh.update_layout(xaxis_title="% lignes touchées", yaxis_title=coherence_group_col)
                        fig_coh = apply_plotly_value_annotations(fig_coh, annot_vals)
                        st.plotly_chart(fig_coh, width="stretch")

        # ==========================================================
        # 1b) Cas à relancer dans la chaîne standard
        # ==========================================================
        with st.expander("Cas à relancer dans la chaîne standard", expanded=False):
            st.caption(
                "Ce tableau standard multi-maladies met en évidence les dossiers qui bloquent la chaîne alerte -> investigation -> "
                "prélèvement -> laboratoire -> issue. Il aide à prioriser les relances terrain et labo."
            )
            st.caption(
                "Les règles tiennent compte des preuves documentaires disponibles pour éviter de relancer à tort un dossier déjà prélevé "
                "ou déjà documenté côté laboratoire."
            )
            followup_summary, followup_detail = build_standard_followup_tables(df_f)
            if followup_summary.empty:
                render_absence_narrative("quality")
            else:
                followup_count_col = next(
                    (col for col in followup_summary.columns if "relancer" in str(col).casefold()),
                    followup_summary.columns[2],
                )
                followup_rule_col = next(
                    (col for col in followup_summary.columns if "gle" in str(col).casefold()),
                    followup_summary.columns[0],
                )
                followup_count_series = pd.to_numeric(followup_summary[followup_count_col], errors="coerce").fillna(0)
                k_fu1, k_fu2, k_fu3 = st.columns(3)
                k_fu1.metric("Règles suivies", str(len(followup_summary)))
                k_fu2.metric(
                    "Règles avec cas à relancer",
                    str(int((followup_count_series > 0).sum())),
                )
                k_fu3.metric(
                    "Cas à relancer cumulés",
                    format_metric_value(followup_count_series.sum()),
                )

                st_dataframe_safe(followup_summary, height=320)
                st.download_button(
                    "Télécharger le résumé des relances (CSV)",
                    data=followup_summary.to_csv(index=False).encode("utf-8"),
                    file_name="quality_cas_a_relancer_resume.csv",
                    mime="text/csv",
                    key="download_quality_followup_summary_csv",
                )

                if not followup_detail.empty:
                    followup_rules = followup_summary.loc[
                        followup_count_series > 0,
                        followup_rule_col,
                    ].astype(str).tolist()
                    detail_rule_col = next(
                        (col for col in followup_detail.columns if "gle" in str(col).casefold()),
                        followup_detail.columns[0],
                    )
                    selected_rule = st.selectbox(
                        "Règle à détailler",
                        options=followup_rules,
                        key="quality_followup_rule_select",
                    )
                    detail_view = followup_detail[followup_detail[detail_rule_col].astype(str) == str(selected_rule)].copy()
                    st.caption("Détail des dossiers à relancer pour la règle sélectionnée.")
                    st.dataframe(detail_view.head(500), width="stretch", height=420, hide_index=True)
                    st.download_button(
                        "Télécharger le détail des relances (CSV)",
                        data=detail_view.to_csv(index=False).encode("utf-8"),
                        file_name="quality_cas_a_relancer_detail.csv",
                        mime="text/csv",
                        key="download_quality_followup_detail_csv",
                    )

        # ==========================================================
        # 2) Complétude de l'ensemble des colonnes
        # ==========================================================
        with st.expander("Complétude de l'ensemble des variables", expanded=False):
            df_missing_scope = (
                df_f_source
                if "df_f_source" in globals() and isinstance(df_f_source, pd.DataFrame) and not df_f_source.empty
                else df_f
            )
            st.caption(
                "Le tableau ci-dessous présente, pour les colonnes du fichier chargé dans le périmètre filtré, "
                "la présence de la colonne, le volume renseigné, le volume manquant et le niveau de priorité du missing."
            )

            st.markdown("**Paramètres de classification du missing**")
            st.caption("Modifiez ici les seuils utilisés pour classer le niveau de complétude des colonnes.")
            seuil_col1, seuil_col2 = st.columns(2)
            with seuil_col1:
                seuil_missing_acceptable = st.number_input(
                    "Seuil acceptable (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=1.0,
                    key="quality_missing_threshold_acceptable",
                )
            with seuil_col2:
                seuil_missing_surveillance = st.number_input(
                    "Seuil de surveillance (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0,
                    step=1.0,
                    key="quality_missing_threshold_surveillance",
                )

            if seuil_missing_acceptable > seuil_missing_surveillance:
                st.warning(
                    "Le seuil acceptable ne peut pas dépasser le seuil de surveillance. "
                    "Les valeurs ont été réalignées automatiquement."
                )
                seuil_missing_surveillance = seuil_missing_acceptable

            missing_tbl = analyser_missing_colonnes(
                df=df_missing_scope,
                seuil_acceptable=float(seuil_missing_acceptable),
                seuil_surveillance=float(seuil_missing_surveillance),
            )

            if missing_tbl.empty:
                render_absence_narrative("quality")
            else:
                k_m1, k_m2, k_m3, k_m4 = st.columns(4)
                k_m1.metric("Variables suivies", str(len(missing_tbl)))
                k_m2.metric(
                    "Variables prioritaires",
                    str(int((missing_tbl["Décision / observation"] == "Prioritaire").sum()))
                )
                k_m3.metric(
                    "Variables sans missing",
                    str(int((missing_tbl["Décision / observation"] == "OK").sum()))
                )
                k_m4.metric(
                    "Missing moyen (%)",
                    f"{pd.to_numeric(missing_tbl['% missing'], errors='coerce').mean():.1f}"
                )

                decision_options = sorted(missing_tbl["Décision / observation"].dropna().unique().tolist())
                decision_sel = st.multiselect(
                    "Niveau de priorité à afficher",
                    options=decision_options,
                    default=decision_options,
                    key="quality_missing_decision_filter",
                )
                if decision_sel:
                    missing_view = missing_tbl[missing_tbl["Décision / observation"].isin(decision_sel)].copy()
                else:
                    missing_view = missing_tbl.copy()

                st_dataframe_safe(missing_view, height=520)
                st.download_button(
                    "Télécharger le tableau de complétude (CSV)",
                    data=missing_view.to_csv(index=False).encode("utf-8"),
                    file_name="quality_missing_colonnes.csv",
                    mime="text/csv",
                    key="download_quality_missing_csv",
                )

                topn = st.slider(
                    "Nombre de colonnes à afficher",
                    min_value=5,
                    max_value=min(80, max(5, len(missing_view))),
                    value=min(20, len(missing_view)),
                    step=5,
                    key="quality_missing_topn",
                )
                comp_plot = missing_view.sort_values(["% missing", "Manquantes"], ascending=[False, False]).head(topn)

                figc = px.bar(
                    comp_plot,
                    x="Variable",
                    y="% missing",
                    color="Décision / observation",
                    title=f"Variables prioritaires selon le taux de missing ({topn})"
                )
                figc.update_layout(xaxis_tickangle=-45, yaxis=dict(range=[0, 100]))
                figc = apply_plotly_value_annotations(figc, annot_vals)
                st.plotly_chart(figc, width="stretch")

        # ==========================================================
        # 2b) Promptitude des dates clés
        # ==========================================================
        with st.expander("Promptitude des étapes clés", expanded=False):
            promptitude_pairs = [
                ("Début maladie → notification", DATE_ONSET, "Date_notification"),
                ("Début maladie → investigation", DATE_ONSET, "Date_investigation"),
                ("Début maladie → admission", "Date_debut_maladie", "Date_admission_au_CT"),
                ("Début maladie → prélèvement", DATE_ONSET, DATE_PREL),
                ("Notification → investigation", DATE_NOTIF, DATE_INV),
                ("Notification → prélèvement", DATE_NOTIF, DATE_PREL),
                ("Notification → admission", DATE_NOTIF, DATE_ADM),
                ("Prélèvement → réception labo", DATE_PREL, DATE_RECEP),
                ("Réception labo → résultat", DATE_RECEP, DATE_RES),
                ("Admission → issue", DATE_ADM, DATE_ISSUE),
            ]

            seuil_prompt = st.number_input(
                "Seuil opérationnel (jours)",
                min_value=1.0,
                max_value=30.0,
                value=float(seuil_jours),
                step=1.0,
                key="quality_promptitude_threshold_days",
            )

            prompt_tbl = construire_table_promptitude(
                df_f,
                promptitude_pairs,
                seuil_jours=float(seuil_prompt),
            )

            if prompt_tbl.empty:
                render_absence_narrative("quality")
            else:
                st_dataframe_safe(prompt_tbl, height=360)

                valid_prompt = prompt_tbl.dropna(subset=["Médiane délai (jours)"]).copy()
                if not valid_prompt.empty:
                    figp_delay = px.bar(
                        valid_prompt,
                        x="Étape",
                        y="Médiane délai (jours)",
                        color=f"% <= {int(seuil_prompt)} jours",
                        title="Délais médians par étape"
                    )
                    figp_delay.update_layout(xaxis_tickangle=-35)
                    figp_delay = apply_plotly_value_annotations(figp_delay, annot_vals)
                    st.plotly_chart(figp_delay, width="stretch")
        
        
        # ==========================================================
        # 3) Cascade prélèvement → TDR → résultat → positif
        # ==========================================================
        with st.expander("Performance de la chaîne laboratoire", expanded=False):
        
            cascad = cascade_metrics(df_f) if n_total else pd.DataFrame()
            if cascad.empty:
                render_absence_narrative("quality")
            else:
                st_dataframe_safe(cascad)
        
            # Cascade par province (résumé robuste)
            if COL_PROV in df_f.columns and n_total:
                st.caption("Lecture provinciale synthétique de la chaîne laboratoire")
        
                df_cas = _quality_build_cascade_group_summary(df_f, COL_PROV).rename(
                    columns={"% test documenté": "% TDR"}
                )
        
                sort_col = st.selectbox(
                    "Indicateur de tri",
                    ["n", "% prélèvement", "% TDR", "% résultat valide", "% positif", "% incoh TDR"],
                    index=0
                )
                df_cas_sorted = df_cas.sort_values(sort_col, ascending=False if sort_col == "n" else True)
        
                st_dataframe_safe(df_cas_sorted, height=420)
        
        
        # ==========================================================
        # 4) Alertes tendance (hausse vs baseline simple)
        # ==========================================================
        with st.expander("Signaux hebdomadaires et alertes de tendance", expanded=False):
            alert_group_choices = [c for c in [COL_PROV, COL_ZS] if c in df_f.columns]
            if not alert_group_choices:
                render_absence_narrative("geo")
                alert_group = None
                alerts = pd.DataFrame()
            else:
                alert_group = st.selectbox("Niveau d'analyse", alert_group_choices, index=0)
                min_alert_cases = st.number_input(
                    "Volume minimal pour signal",
                    min_value=1,
                    max_value=500,
                    value=10,
                    step=1,
                    key="quality_alert_min_cases",
                )
                alert_ratio_quality = st.number_input(
                    "Seuil relatif vs baseline",
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

