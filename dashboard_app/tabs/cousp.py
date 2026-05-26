"""Render the COUSP standard analytics tab."""

from __future__ import annotations

from io import BytesIO
import re

from dashboard_app.runtime_support import inject_runtime_support

# Les deux fonctions de haut niveau existent dans dashboard_app.domain.
# dashboard_app.cousp_export contient surtout le générateur local
# generer_feuilles_sortie_cousp_local, utilisé comme fallback ci-dessous.
try:
    from dashboard_app.domain import (
        build_cousp_standard_export_package as _domain_build_cousp_standard_export_package,
        workbook_bytes_from_sheet_dict as _domain_workbook_bytes_from_sheet_dict,
    )
except Exception as _exc:  # fallback géré dans render_cousp_tab()
    _domain_build_cousp_standard_export_package = None
    _domain_workbook_bytes_from_sheet_dict = None
    _COUSP_DOMAIN_IMPORT_ERROR = _exc
else:
    _COUSP_DOMAIN_IMPORT_ERROR = None

try:
    from dashboard_app.cousp_export import SCHEMA_VARIABLES_COUSP as _SCHEMA_VARIABLES_COUSP
except Exception:
    _SCHEMA_VARIABLES_COUSP = []

inject_runtime_support(globals())


def _fallback_build_cousp_standard_export_package(
    df: pd.DataFrame,
    *,
    anonymiser_recherche: bool = False,
    seuil_acceptable: float = 5.0,
    seuil_surveillance: float = 20.0,
) -> tuple[dict[str, pd.DataFrame], str | None]:
    """Fallback local si les wrappers de dashboard_app.domain ne sont pas importables."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {}, "Aucune donnée filtrée n'est disponible pour construire le pack COUSP standard."

    work = df.copy()
    if work.columns.duplicated().any():
        work = work.loc[:, ~work.columns.duplicated()].copy()

    try:
        from dashboard_app.cousp_export import generer_feuilles_sortie_cousp_local
    except Exception as exc:
        return {}, f"Impossible de charger le module COUSP standard : {exc}"

    try:
        sheets = generer_feuilles_sortie_cousp_local(
            work,
            anonymiser_recherche=anonymiser_recherche,
            seuil_acceptable=seuil_acceptable,
            seuil_surveillance=seuil_surveillance,
        )
    except Exception as exc:
        return {}, f"Erreur pendant la génération du pack COUSP standard : {exc}"

    return sheets, None


def _fallback_workbook_bytes_from_sheet_dict(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Fallback local pour sérialiser un dictionnaire de DataFrame en Excel."""
    if not isinstance(sheets, dict) or not sheets:
        raise ValueError("Le dictionnaire de feuilles à exporter ne peut pas être vide.")

    def _sheet_safe(name: str) -> str:
        cleaned = re.sub(r"[\[\]:*?/\\]", "_", str(name))
        return cleaned[:31] or "Feuille1"

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            if not isinstance(sheet_df, pd.DataFrame):
                continue
            sheet_df.to_excel(writer, sheet_name=_sheet_safe(sheet_name), index=False)

    return buffer.getvalue()


def _resolve_cousp_helpers():
    """Résout les helpers COUSP sans dépendre uniquement de build_runtime_context()."""
    build_package = globals().get("build_cousp_standard_export_package")
    workbook_builder = globals().get("workbook_bytes_from_sheet_dict")

    if not callable(build_package) and callable(_domain_build_cousp_standard_export_package):
        build_package = _domain_build_cousp_standard_export_package

    if not callable(workbook_builder) and callable(_domain_workbook_bytes_from_sheet_dict):
        workbook_builder = _domain_workbook_bytes_from_sheet_dict

    if not callable(build_package):
        build_package = _fallback_build_cousp_standard_export_package

    if not callable(workbook_builder):
        workbook_builder = _fallback_workbook_bytes_from_sheet_dict

    return build_package, workbook_builder


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


def _cousp_added_variables_dictionary() -> pd.DataFrame:
    """Dictionnaire des variables ajoutées/derivées dans le dataset COUSP."""
    rows = [
        {
            "N°": 1,
            "Nom": "Annee_epid",
            "Definition": "Annee epidemiologique derivee a partir de la date de notification ou, a defaut, de la date de debut des symptomes.",
            "Commentaire": "Permet les analyses temporelles annuelles standardisees.",
        },
        {
            "N°": 2,
            "Nom": "Num_semaine_epid",
            "Definition": "Numero de semaine epidemiologique derive a partir de la date de notification ou de la date de debut des symptomes.",
            "Commentaire": "Utilise pour les analyses hebdomadaires et les comparaisons de tendance.",
        },
        {
            "N°": 3,
            "Nom": "Semaine_epid",
            "Definition": "Libelle standard de semaine epidemiologique, par exemple SE05.",
            "Commentaire": "Facilite la lecture des tableaux et l'export vers les rapports.",
        },
        {
            "N°": 4,
            "Nom": "Investigation",
            "Definition": "Variable renseignee ou derivee a Oui lorsqu'une date d'investigation est disponible.",
            "Commentaire": "Permet de suivre la conversion des alertes en investigations documentees.",
        },
        {
            "N°": 5,
            "Nom": "Prelevement",
            "Definition": "Variable renseignee ou derivee a Oui lorsqu'une date de prelevement est disponible.",
            "Commentaire": "Permet de suivre le passage des cas suspects vers le laboratoire.",
        },
        {
            "N°": 6,
            "Nom": "Date_confirmation",
            "Definition": "Date de confirmation derivee a partir de la date d'analyse lorsque le resultat labo est positif et que la date manque.",
            "Commentaire": "Aide a completer la chaine de confirmation biologique.",
        },
        {
            "N°": 7,
            "Nom": "Date_issue",
            "Definition": "Date d'issue harmonisee, completee a partir de la date de sortie du CT ou du rapportage du statut final selon le contexte.",
            "Commentaire": "Necessaire pour la letalite, la duree de sejour et les analyses de sortie.",
        },
        {
            "N°": 8,
            "Nom": "Age_en_ans",
            "Definition": "Age converti en annees a partir de Age, Age_annee, Age_mois et Unite_age.",
            "Commentaire": "Standardise les analyses par age, quel que soit le format source.",
        },
        {
            "N°": 9,
            "Nom": "Tranche_age_en_ans",
            "Definition": "Tranche d'age derivee a partir de l'age en annees (<1, 1-4, 5-14, etc.).",
            "Commentaire": "Permet les analyses de profil epidemiologique harmonisees.",
        },
        {
            "N°": 10,
            "Nom": "Tranche_age",
            "Definition": "Tranche d'age harmonisee, alimentee a partir de Tranche_age_en_ans lorsqu'elle manque.",
            "Commentaire": "Maintient un format de tranche exploitable dans les tableaux du dashboard.",
        },
        {
            "N°": 11,
            "Nom": "Duree_du_sejour_au_CT",
            "Definition": "Duree calculee entre la date d'admission au CT et la date d'issue.",
            "Commentaire": "Aide a suivre l'occupation et la dynamique de prise en charge.",
        },
        {
            "N°": 12,
            "Nom": "delai_symptomes_notification",
            "Definition": "Delai entre la date de debut des symptomes et la date de notification.",
            "Commentaire": "Mesure la rapidite de detection et de notification initiale.",
        },
        {
            "N°": 13,
            "Nom": "delai_notification_investigation",
            "Definition": "Delai entre la date de notification et la date d'investigation.",
            "Commentaire": "Mesure la rapidite de la reponse terrain apres notification.",
        },
        {
            "N°": 14,
            "Nom": "delai_notification_prelevement",
            "Definition": "Delai entre la date de notification et la date de prelevement.",
            "Commentaire": "Indique la vitesse de passage vers la confirmation biologique.",
        },
        {
            "N°": 15,
            "Nom": "delai_prelevement_reception_labo",
            "Definition": "Delai entre la date de prelevement et la date de reception de l'echantillon au laboratoire.",
            "Commentaire": "Suit la performance d'acheminement des echantillons.",
        },
        {
            "N°": 16,
            "Nom": "delai_reception_analyse_labo",
            "Definition": "Delai entre la date de reception de l'echantillon au laboratoire et la date d'analyse.",
            "Commentaire": "Mesure la promptitude analytique du laboratoire.",
        },
        {
            "N°": 17,
            "Nom": "delai_notification_admission_ct",
            "Definition": "Delai entre la date de notification et la date d'admission au centre de traitement.",
            "Commentaire": "Aide a suivre la rapidite de mise sous prise en charge.",
        },
        {
            "N°": 18,
            "Nom": "delai_admission_issue",
            "Definition": "Delai entre la date d'admission au centre de traitement et la date d'issue.",
            "Commentaire": "Informe sur la duree de prise en charge clinique.",
        },
        {
            "N°": 19,
            "Nom": "delai_symptomes_investigation",
            "Definition": "Delai entre la date de debut des symptomes et la date d'investigation.",
            "Commentaire": "Permet d'evaluer le temps ecoule avant investigation apres apparition des symptomes.",
        },
        {
            "N°": 20,
            "Nom": "delai_symptomes_prelevement",
            "Definition": "Delai entre la date de debut des symptomes et la date de prelevement.",
            "Commentaire": "Utile pour suivre la rapidite de la confirmation biologique apres debut clinique.",
        },
        {
            "N°": 21,
            "Nom": "delai_symptomes_reception_labo",
            "Definition": "Delai entre la date de debut des symptomes et la date de reception au laboratoire.",
            "Commentaire": "Mesure le temps cumule jusqu'a l'arrivee de l'echantillon au laboratoire.",
        },
        {
            "N°": 22,
            "Nom": "delai_symptomes_analyse_labo",
            "Definition": "Delai entre la date de debut des symptomes et la date d'analyse laboratoire.",
            "Commentaire": "Permet de suivre le temps total jusqu'a l'analyse biologique.",
        },
        {
            "N°": 23,
            "Nom": "delai_symptomes_admission_ct",
            "Definition": "Delai entre la date de debut des symptomes et la date d'admission au CT.",
            "Commentaire": "Suit la rapidite globale de la prise en charge clinique.",
        },
        {
            "N°": 24,
            "Nom": "delai_symptomes_issue",
            "Definition": "Delai entre la date de debut des symptomes et la date d'issue.",
            "Commentaire": "Donne une vue globale du parcours du cas jusqu'a sa sortie finale.",
        },
        {
            "N°": 25,
            "Nom": "delai_notification_reception_labo",
            "Definition": "Delai entre la date de notification et la date de reception au laboratoire.",
            "Commentaire": "Suit le temps ecoule entre alerte documentee et reception de l'echantillon.",
        },
        {
            "N°": 26,
            "Nom": "delai_notification_analyse_labo",
            "Definition": "Delai entre la date de notification et la date d'analyse laboratoire.",
            "Commentaire": "Mesure le temps total entre notification et resultat analytique.",
        },
        {
            "N°": 27,
            "Nom": "delai_notification_issue",
            "Definition": "Delai entre la date de notification et la date d'issue.",
            "Commentaire": "Resume la duree du parcours apres entree dans la surveillance.",
        },
        {
            "N°": 28,
            "Nom": "delai_investigation_prelevement",
            "Definition": "Delai entre la date d'investigation et la date de prelevement.",
            "Commentaire": "Evalue la continuité entre investigation terrain et acte de prelevement.",
        },
        {
            "N°": 29,
            "Nom": "delai_investigation_reception_labo",
            "Definition": "Delai entre la date d'investigation et la date de reception au laboratoire.",
            "Commentaire": "Suit la chaine investigation -> transport -> laboratoire.",
        },
        {
            "N°": 30,
            "Nom": "delai_investigation_analyse_labo",
            "Definition": "Delai entre la date d'investigation et la date d'analyse laboratoire.",
            "Commentaire": "Mesure le temps jusqu'a l'analyse apres investigation.",
        },
        {
            "N°": 31,
            "Nom": "delai_investigation_admission_ct",
            "Definition": "Delai entre la date d'investigation et la date d'admission au CT.",
            "Commentaire": "Permet d'evaluer la rapidite d'orientation clinique apres investigation.",
        },
        {
            "N°": 32,
            "Nom": "delai_prelevement_analyse_labo",
            "Definition": "Delai entre la date de prelevement et la date d'analyse laboratoire.",
            "Commentaire": "Mesure la performance combinee transport + analyse.",
        },
        {
            "N°": 33,
            "Nom": "delai_analyse_confirmation",
            "Definition": "Delai entre la date d'analyse et la date de confirmation.",
            "Commentaire": "Utile lorsqu'une validation supplementaire est necessaire apres l'analyse.",
        },
        {
            "N°": 34,
            "Nom": "delai_admission_sortie_ct",
            "Definition": "Delai entre la date d'admission au CT et la date de sortie du CT.",
            "Commentaire": "Permet de suivre la duree de sejour strictement liee au centre de traitement.",
        },
        {
            "N°": 35,
            "Nom": "delai_issue_rapportage_statut_final",
            "Definition": "Delai entre la date d'issue et la date de rapportage du statut final.",
            "Commentaire": "Mesure la rapidite de cloture administrative apres issue.",
        },
        {
            "N°": 36,
            "Nom": "delai_sortie_rapportage_statut_final",
            "Definition": "Delai entre la date de sortie du CT et la date de rapportage du statut final.",
            "Commentaire": "Permet de suivre la fermeture documentaire apres sortie.",
        },
    ]
    return pd.DataFrame(rows)


def _cousp_candidate_filter_columns(df: pd.DataFrame) -> list[str]:
    """Retourne les colonnes clés/catégorielles les plus utiles au filtrage COUSP."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    ordered: list[str] = []
    preferred = [
        "Province_notification",
        "Zone_de_sante_notification",
        "Aire_de_sante_notification",
        "Localite",
        "Sexe",
        "Investigation",
        "Prelevement",
        "Classification_investigation",
        "Classification_finale",
        "Resultat_final_labo",
        "Issue",
        "Semaine_epid",
        "Annee_epid",
        "Num_semaine_epid",
        "Source_alerte",
        "Provenance",
        "Nom_laboratoire",
        "Type_prelevement",
        "Patient_en_isolement",
    ]
    for col in preferred:
        if col in df.columns and col not in ordered:
            ordered.append(col)

    for item in _SCHEMA_VARIABLES_COUSP:
        col = str(item.get("Variable cle", "")).strip()
        type_variable = str(item.get("Type variable", "")).strip().lower()
        if col not in df.columns or col in ordered:
            continue
        if type_variable not in {"texte", "geographie", "identifiant", "derivee"}:
            continue
        serie = df[col]
        nunique = int(serie.dropna().astype("string").str.strip().replace("", pd.NA).dropna().nunique())
        if 1 < nunique <= 60:
            ordered.append(col)

    for col in df.columns:
        if col in ordered:
            continue
        serie = df[col]
        if not (
            pd.api.types.is_object_dtype(serie)
            or pd.api.types.is_string_dtype(serie)
            or isinstance(serie.dtype, pd.CategoricalDtype)
        ):
            continue
        cleaned = serie.astype("string").str.strip().replace("", pd.NA).dropna()
        nunique = int(cleaned.nunique())
        if 1 < nunique <= 40:
            ordered.append(col)

    return ordered


def _cousp_filter_value_options(serie: pd.Series) -> list[str]:
    if serie is None:
        return []
    cleaned = (
        serie.astype("string")
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
    )
    values = cleaned.tolist()
    return sorted(values, key=lambda value: str(value).casefold())


def _cousp_apply_column_filters(
    df: pd.DataFrame,
    filters: dict[str, list[str]],
) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    out = df.copy()
    for col, selected_values in filters.items():
        if col not in out.columns:
            continue
        serie = out[col].astype("string").str.strip()
        if selected_values:
            out = out.loc[serie.isin([str(v).strip() for v in selected_values])].copy()
    return out


def _cousp_apply_local_multiselect_filters(
    df: pd.DataFrame,
    filter_columns: list[str],
    *,
    key_prefix: str,
) -> pd.DataFrame:
    """Applique des filtres multiselect locaux sur un DataFrame déjà calculé."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    available_columns = [col for col in filter_columns if col in df.columns]
    if not available_columns:
        return df.copy()

    filter_widgets = st.columns(min(3, max(1, len(available_columns))))
    active_filters: dict[str, list[str]] = {}

    for idx, col in enumerate(available_columns):
        options = _cousp_filter_value_options(df[col])
        with filter_widgets[idx % len(filter_widgets)]:
            selected_values = st.multiselect(
                f"{col}",
                options=options,
                default=[],
                key=f"{key_prefix}_{col}",
            )
            active_filters[col] = selected_values

    return _cousp_apply_column_filters(df, active_filters)


def render_cousp_tab(ctx: dict) -> None:
    """Render the COUSP standard analytics tab."""
    globals().update(ctx)

    build_package, workbook_builder = _resolve_cousp_helpers()

    render_section_title(7, "Pack d'analyse COUSP standard")
    if bool(globals().get("IDSR_MODE", False)):
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

    df_f_local = globals().get("df_f")
    if df_f_local is None or not isinstance(df_f_local, pd.DataFrame) or df_f_local.empty:
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

    with st.expander("Filtres COUSP sur colonnes cles et categorielles", expanded=False):
        st.caption(
            "Ces filtres affinent le perimetre de l'onglet COUSP avant calcul des KPI, "
            "de la completude, des anomalies et de l'export."
        )
        st.caption("Aucune valeur selectionnee = toutes les valeurs.")
        filter_candidates = _cousp_candidate_filter_columns(df_f_local)
        selected_filter_cols = st.multiselect(
            "Colonnes a filtrer",
            options=filter_candidates,
            default=[],
            key="cousp_filter_columns",
            help="Choisissez les colonnes cles ou categorielles a utiliser comme filtres pour cet onglet.",
        )

        active_filters: dict[str, list[str]] = {}
        if selected_filter_cols:
            filter_columns = st.columns(2)
            for idx, col in enumerate(selected_filter_cols):
                options = _cousp_filter_value_options(df_f_local[col])
                with filter_columns[idx % 2]:
                    selected_values = st.multiselect(
                        f"{col} - valeurs",
                        options=options,
                        default=[],
                        key=f"cousp_filter_values_{col}",
                    )
                    active_filters[col] = selected_values
        else:
            active_filters = {}

    df_cousp_scope = _cousp_apply_column_filters(df_f_local, active_filters)
    st.caption(
        f"Perimetre COUSP courant : {len(df_cousp_scope)} ligne(s) sur {len(df_f_local)} "
        "apres application des filtres de l'onglet."
    )
    if df_cousp_scope.empty:
        st.warning("Aucune ligne ne correspond aux filtres COUSP selectionnes.")
        return

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

    sheets, error = build_package(
        df_cousp_scope,
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
    m2.metric("Lignes filtrees", int(len(df_cousp_scope)))
    m3.metric("Anomalies de dates", int(len(anomalies_df)))
    m4.metric("Cas a relancer", int(len(relances_df)))

    st.markdown("**Resume du pack genere**")
    st_dataframe_safe(summary_df, height=220)

    try:
        cousp_excel_bytes = workbook_builder(sheets)
    except Exception as exc:
        st.warning(f"Le telechargement du pack COUSP est indisponible : {exc}")
    else:
        export_base_name = f"{str(globals().get('disease_key', 'maladie')).strip().lower()}_filtre"
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
            st.caption("Filtres de synthese COUSP.")
            st.caption("Aucune valeur selectionnee = toutes les valeurs.")
            synthese_view = _cousp_apply_local_multiselect_filters(
                synthese_df,
                ["Section", "Indicateur"],
                key_prefix="cousp_synthese_filter",
            )

            if "Section" in synthese_view.columns:
                kpi_df = synthese_view.loc[synthese_view["Section"] == "KPI"].copy()
                delay_df = synthese_view.loc[synthese_view["Section"] != "KPI"].copy()
            else:
                kpi_df = synthese_view.copy()
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
            st.caption("Aucune valeur selectionnee = toutes les valeurs.")
            completeness_view = _cousp_apply_local_multiselect_filters(
                completeness_df,
                ["Bloc", "Priorite", "Type variable", "Decision / observation"],
                key_prefix="cousp_completude_filter",
            )
            k_m1, k_m2, k_m3, k_m4 = st.columns(4)
            k_m1.metric("Variables suivies", str(len(completeness_view)))
            k_m2.metric(
                "Variables prioritaires",
                str(int((completeness_view["Decision / observation"] == "Prioritaire").sum())),
            )
            k_m3.metric(
                "Variables sans missing",
                str(int((completeness_view["Decision / observation"] == "OK").sum())),
            )
            k_m4.metric(
                "Missing moyen (%)",
                f"{pd.to_numeric(completeness_view['% missing'], errors='coerce').mean():.1f}",
            )

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
            annotation_fn = globals().get("apply_plotly_value_annotations")
            if callable(annotation_fn):
                figc = annotation_fn(figc, bool(globals().get("annot_vals", False)))
            st.plotly_chart(figc, width="stretch")

    with tab_anomalies:
        if anomalies_df.empty:
            st.success("Aucune anomalie de dates detectee dans le perimetre filtre.")
        else:
            st.markdown("**Parametres du sous-onglet Anomalies de dates**")
            st.caption(
                "Filtres rapides sur les anomalies de dates pour cibler une province, une zone ou un type d'anomalie."
            )
            st.caption("Aucune valeur selectionnee = toutes les valeurs.")
            anomalies_view = _cousp_apply_local_multiselect_filters(
                anomalies_df,
                [
                    "Province_notification",
                    "Zone_de_sante_notification",
                    "Variable_anomalie",
                    "Type_anomalie",
                ],
                key_prefix="cousp_anomalies_filter",
            )
            st.caption(f"{len(anomalies_view)} ligne(s) d'anomalies affichee(s).")
            st_dataframe_safe(anomalies_view, height=540)
            st.download_button(
                "Telecharger les anomalies filtrees (CSV)",
                data=anomalies_view.to_csv(index=False).encode("utf-8"),
                file_name="cousp_anomalies_dates.csv",
                mime="text/csv",
                key="download_cousp_anomalies_csv",
            )

    with tab_relances:
        if relances_df.empty:
            st.success("Aucun cas a relancer detecte dans le perimetre filtre.")
        else:
            st.caption(
                "Filtres rapides sur les cas a relancer pour cibler une province, une zone ou un motif de relance."
            )
            st.caption("Aucune valeur selectionnee = toutes les valeurs.")
            relances_view = _cousp_apply_local_multiselect_filters(
                relances_df,
                [
                    "Province_notification",
                    "Zone_de_sante_notification",
                    "Motif_relance",
                ],
                key_prefix="cousp_relances_filter",
            )
            st.caption(f"{len(relances_view)} ligne(s) de relance affichee(s).")
            st_dataframe_safe(relances_view, height=540)

    with tab_recherche:
        recherche_df = sheets.get("Recherche_dataset", pd.DataFrame())
        if recherche_df.empty:
            st.info("Aucun dataset de recherche disponible.")
        else:
            recherche_view = recherche_df.copy()
            with st.expander("Dictionnaire des variables ajoutees", expanded=False):
                st.caption(
                    "Ce tableau documente les variables derivees ou harmonisees ajoutees dans le dataset de recherche COUSP."
                )
                dictionnaire_df = _cousp_added_variables_dictionary()
                st_dataframe_safe(dictionnaire_df, height=420)
            with st.expander("Filtres du dataset de recherche", expanded=False):
                st.caption("Apercu du dataset standardise COUSP utilise pour la recherche et l'export.")
                st.caption("Aucune valeur selectionnee = toutes les valeurs.")
                recherche_filter_columns = _cousp_candidate_filter_columns(recherche_df)[:8]
                recherche_view = _cousp_apply_local_multiselect_filters(
                    recherche_df,
                    recherche_filter_columns,
                    key_prefix="cousp_recherche_filter",
                )
                st.caption(f"{len(recherche_view)} ligne(s) de recherche affichee(s).")
                st.download_button(
                    "Telecharger le dataset de recherche filtre (CSV)",
                    data=recherche_view.to_csv(index=False).encode("utf-8"),
                    file_name="cousp_recherche_dataset_filtre.csv",
                    mime="text/csv",
                    key="download_cousp_recherche_csv",
                )
            with st.expander("Apercu du dataset de recherche", expanded=True):
                st_dataframe_safe(recherche_view.head(200), height=540)
