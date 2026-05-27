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


def _cousp_pick_reference_join_key(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
) -> str | None:
    """Choisit une cle stable pour enrichir une vue COUSP a partir du dataset recherche."""
    if (
        left_df is None
        or right_df is None
        or not isinstance(left_df, pd.DataFrame)
        or not isinstance(right_df, pd.DataFrame)
        or left_df.empty
        or right_df.empty
    ):
        return None

    candidates = ["N_alerte", "N_epid", "N_labo"]
    best_key: str | None = None
    best_score = -1

    for col in candidates:
        if col not in left_df.columns or col not in right_df.columns:
            continue
        right_keys = right_df[col].astype("string").str.strip().replace("", pd.NA).dropna()
        if right_keys.empty or right_keys.duplicated().any():
            continue
        left_keys = left_df[col].astype("string").str.strip().replace("", pd.NA)
        score = int(left_keys.notna().sum())
        if score > best_score:
            best_key = col
            best_score = score

    return best_key


def _cousp_enrich_anomalies_with_reference_data(
    anomalies_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Ajoute les colonnes temporelles utiles aux graphiques d'anomalies."""
    if (
        anomalies_df is None
        or not isinstance(anomalies_df, pd.DataFrame)
        or anomalies_df.empty
        or reference_df is None
        or not isinstance(reference_df, pd.DataFrame)
        or reference_df.empty
    ):
        return anomalies_df.copy() if isinstance(anomalies_df, pd.DataFrame) else pd.DataFrame()

    join_key = _cousp_pick_reference_join_key(anomalies_df, reference_df)
    if not join_key:
        return anomalies_df.copy()

    extra_cols = [
        col
        for col in [
            join_key,
            "Date_notification",
            "Semaine_epid",
            "Classification_investigation",
            "Resultat_final_labo",
            "Issue",
        ]
        if col in reference_df.columns
    ]
    if join_key not in extra_cols:
        return anomalies_df.copy()

    out = anomalies_df.copy()
    out["__join_key_cousp"] = out[join_key].astype("string").str.strip().replace("", pd.NA)

    ref = reference_df[extra_cols].copy()
    ref["__join_key_cousp"] = ref[join_key].astype("string").str.strip().replace("", pd.NA)
    ref = ref.loc[ref["__join_key_cousp"].notna()].drop_duplicates("__join_key_cousp")

    merged = out.merge(
        ref.drop(columns=[join_key]),
        on="__join_key_cousp",
        how="left",
    )
    return merged.drop(columns=["__join_key_cousp"], errors="ignore")


def _cousp_parse_epi_week_label(value) -> tuple[int, int] | None:
    """Parse des formats tels que S19-2026, SE19-2026 ou 2026-W19."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    patterns = [
        r"(?P<year>\d{4})\D+(?P<week>\d{1,2})$",
        r"^[A-Za-z]{0,3}(?P<week>\d{1,2})\D+(?P<year>\d{4})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        year = int(match.group("year"))
        week = int(match.group("week"))
        if 1 <= week <= 53:
            return year, week
    return None


def _cousp_build_temporal_series(
    df: pd.DataFrame,
    *,
    grain: str,
    date_col: str = "Date_notification",
    week_col: str = "Semaine_epid",
) -> pd.DataFrame:
    """Construit une serie de comptage par jour, semaine epi ou mois."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["Periode", "Cas", "Libelle"])

    if grain == "jour":
        if date_col not in df.columns:
            return pd.DataFrame(columns=["Periode", "Cas", "Libelle"])
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna().dt.normalize()
        if dates.empty:
            return pd.DataFrame(columns=["Periode", "Cas", "Libelle"])
        counts = dates.value_counts().sort_index()
        out = counts.rename_axis("Periode").reset_index(name="Cas")
        out["Libelle"] = out["Periode"].dt.strftime("%Y-%m-%d")
        return out

    if grain == "mois":
        if date_col not in df.columns:
            return pd.DataFrame(columns=["Periode", "Cas", "Libelle"])
        months = pd.to_datetime(df[date_col], errors="coerce").dropna().dt.to_period("M")
        if months.empty:
            return pd.DataFrame(columns=["Periode", "Cas", "Libelle"])
        counts = months.value_counts().sort_index()
        out = counts.rename_axis("Mois").reset_index(name="Cas")
        out["Periode"] = out["Mois"].dt.to_timestamp()
        out["Libelle"] = out["Mois"].astype("string")
        return out[["Periode", "Cas", "Libelle"]]

    if grain == "semaine":
        if week_col not in df.columns:
            return pd.DataFrame(columns=["Periode", "Cas", "Libelle"])
        weeks = df[week_col].astype("string").str.strip().replace("", pd.NA).dropna()
        if weeks.empty:
            return pd.DataFrame(columns=["Periode", "Cas", "Libelle"])

        rows = []
        for label, count in weeks.value_counts().items():
            parsed = _cousp_parse_epi_week_label(label)
            year = parsed[0] if parsed else 9999
            week = parsed[1] if parsed else 9999
            rows.append(
                {
                    "Periode": str(label),
                    "Cas": int(count),
                    "Libelle": str(label),
                    "__sort_year": year,
                    "__sort_week": week,
                }
            )

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["Periode", "Cas", "Libelle"])
        out = out.sort_values(
            ["__sort_year", "__sort_week", "Libelle"],
            ascending=[True, True, True],
            kind="stable",
        ).reset_index(drop=True)
        return out[["Periode", "Cas", "Libelle"]]

    return pd.DataFrame(columns=["Periode", "Cas", "Libelle"])


def _cousp_build_category_counts(
    df: pd.DataFrame,
    category_col: str,
    *,
    topn: int = 10,
) -> pd.DataFrame:
    """Compte les lignes par categorie en excluant les valeurs vides."""
    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
        or category_col not in df.columns
    ):
        return pd.DataFrame(columns=[category_col, "Cas"])

    values = (
        df[category_col]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )
    if values.empty:
        return pd.DataFrame(columns=[category_col, "Cas"])

    return (
        values.value_counts()
        .head(int(max(1, topn)))
        .rename_axis(category_col)
        .reset_index(name="Cas")
    )


def _cousp_render_temporal_evolution_chart(
    df: pd.DataFrame,
    *,
    title: str,
    key_prefix: str,
) -> None:
    """Affiche une courbe d'evolution dynamique J/S/M."""
    st.caption(
        "Granularite dynamique : `Jour` et `Mois` utilisent `Date_notification`, "
        "`Semaine` utilise `Semaine_epid`."
    )
    label_to_grain = {
        "Jour": "jour",
        "Semaine": "semaine",
        "Mois": "mois",
    }
    selected_label = st.radio(
        "Periode d'analyse",
        options=list(label_to_grain.keys()),
        index=1,
        horizontal=True,
        key=f"{key_prefix}_grain",
    )
    series_df = _cousp_build_temporal_series(df, grain=label_to_grain[selected_label])
    if series_df.empty:
        st.info("Les colonnes temporelles necessaires sont absentes ou vides pour ce filtre.")
        return

    x_col = "Periode"
    fig = px.line(
        series_df,
        x=x_col,
        y="Cas",
        markers=True,
        title=title,
    )
    fig.update_traces(
        hovertemplate="%{x}<br>Cas: %{y}<extra></extra>",
        line=dict(width=3),
        marker=dict(size=7),
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(t=60, r=20, b=40, l=20),
        xaxis_title="Periode",
        yaxis_title="Nombre de cas",
        hovermode="x unified",
    )
    if label_to_grain[selected_label] == "semaine":
        fig.update_xaxes(type="category", tickangle=-40)
    elif label_to_grain[selected_label] == "mois":
        fig.update_xaxes(tickformat="%Y-%m")

    annotation_fn = globals().get("apply_plotly_value_annotations")
    if callable(annotation_fn):
        fig = annotation_fn(fig, bool(globals().get("annot_vals", False)))
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_chart")


def render_cousp_tab(ctx: dict) -> None:
    """Render the COUSP standard analytics tab."""
    globals().update(ctx)

    build_package, workbook_builder = _resolve_cousp_helpers()

    render_section_title(4, "Pack d'analyse COUSP standard")
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
    recherche_df = sheets.get("Recherche_dataset", pd.DataFrame())

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
            with st.expander("Synthese operationnelle - tableau et filtres", expanded=True):
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

            with st.expander("Visualisations de la synthese operationnelle", expanded=True):
                with st.container():
                    scol1, scol2 = st.columns([1.1, 1.1])
                    with scol1:
                        if kpi_df.empty:
                            st.info("Aucun KPI exploitable pour la visualisation.")
                        else:
                            kpi_plot = kpi_df.copy()
                            if "Valeur" in kpi_plot.columns:
                                kpi_plot["Valeur"] = pd.to_numeric(kpi_plot["Valeur"], errors="coerce")
                                kpi_plot = kpi_plot.loc[kpi_plot["Valeur"].notna()].copy()
                            if kpi_plot.empty or "Indicateur" not in kpi_plot.columns or "Valeur" not in kpi_plot.columns:
                                st.info("Les KPI ne contiennent pas de valeurs numeriques a afficher.")
                            else:
                                fig_kpi = px.bar(
                                    kpi_plot.sort_values("Valeur", ascending=False),
                                    x="Indicateur",
                                    y="Valeur",
                                    title="Vue d'ensemble des KPI COUSP",
                                    color="Valeur",
                                    color_continuous_scale="Blues",
                                )
                                fig_kpi.update_layout(
                                    template="plotly_white",
                                    height=380,
                                    margin=dict(t=60, r=20, b=80, l=20),
                                    coloraxis_showscale=False,
                                    xaxis_tickangle=-35,
                                    xaxis_title="Indicateur",
                                    yaxis_title="Valeur",
                                )
                                annotation_fn = globals().get("apply_plotly_value_annotations")
                                if callable(annotation_fn):
                                    fig_kpi = annotation_fn(fig_kpi, bool(globals().get("annot_vals", False)))
                                st.plotly_chart(fig_kpi, width="stretch", key="cousp_synthese_kpi_chart")
                    with scol2:
                        if delay_df.empty:
                            st.info("Aucun delai prioritaire exploitable pour la visualisation.")
                        else:
                            delay_plot = delay_df.copy()
                            if "Mediane" in delay_plot.columns:
                                delay_plot["Mediane"] = pd.to_numeric(delay_plot["Mediane"], errors="coerce")
                            if "Proportion_retards_%" in delay_plot.columns:
                                delay_plot["Proportion_retards_%"] = pd.to_numeric(
                                    delay_plot["Proportion_retards_%"],
                                    errors="coerce",
                                )
                            delay_plot = delay_plot.loc[delay_plot.get("Mediane").notna()].copy()
                            if delay_plot.empty or "Indicateur" not in delay_plot.columns:
                                st.info("Les delais prioritaires ne contiennent pas de mediane exploitable.")
                            else:
                                color_col = (
                                    "Proportion_retards_%"
                                    if "Proportion_retards_%" in delay_plot.columns
                                    and delay_plot["Proportion_retards_%"].notna().any()
                                    else "Mediane"
                                )
                                fig_delay = px.bar(
                                    delay_plot.sort_values("Mediane", ascending=False),
                                    x="Indicateur",
                                    y="Mediane",
                                    color=color_col,
                                    title="Mediane des delais prioritaires",
                                    color_continuous_scale="OrRd",
                                )
                                fig_delay.update_layout(
                                    template="plotly_white",
                                    height=380,
                                    margin=dict(t=60, r=20, b=80, l=20),
                                    xaxis_tickangle=-35,
                                    xaxis_title="Indicateur",
                                    yaxis_title="Mediane (jours)",
                                )
                                annotation_fn = globals().get("apply_plotly_value_annotations")
                                if callable(annotation_fn):
                                    fig_delay = annotation_fn(fig_delay, bool(globals().get("annot_vals", False)))
                                st.plotly_chart(fig_delay, width="stretch", key="cousp_synthese_delay_chart")

    with tab_completude:
        if completeness_df.empty:
            st.info("Aucune analyse de completude disponible.")
        else:
            with st.expander("Completude - tableau et filtres", expanded=True):
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

            with st.expander("Visualisations de la completude", expanded=True):
                with st.container():
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
            with st.expander("Anomalies de dates - tableau et filtres", expanded=True):
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
            with st.expander("Visualisations des anomalies", expanded=True):
                with st.container():
                    anomalies_viz_df = _cousp_enrich_anomalies_with_reference_data(anomalies_view, recherche_df)
                    st.markdown("**Visualisations des anomalies**")
                    vcol1, vcol2 = st.columns([1.4, 1.0])
                    with vcol1:
                        _cousp_render_temporal_evolution_chart(
                            anomalies_viz_df,
                            title="Evolution des cas avec anomalies de dates",
                            key_prefix="cousp_anomalies_trend",
                        )
                    with vcol2:
                        anomaly_counts = _cousp_build_category_counts(
                            anomalies_view,
                            "Variable_anomalie",
                            topn=10,
                        )
                        if anomaly_counts.empty:
                            st.info("Aucune variable d'anomalie exploitable pour le graphique.")
                        else:
                            fig_anom = px.bar(
                                anomaly_counts.sort_values("Cas", ascending=True),
                                x="Cas",
                                y="Variable_anomalie",
                                orientation="h",
                                title="Top 10 des anomalies de dates",
                                color="Cas",
                                color_continuous_scale="Reds",
                            )
                            fig_anom.update_layout(
                                template="plotly_white",
                                height=360,
                                margin=dict(t=60, r=20, b=30, l=20),
                                coloraxis_showscale=False,
                                xaxis_title="Nombre de cas",
                                yaxis_title="Variable d'anomalie",
                            )
                            annotation_fn = globals().get("apply_plotly_value_annotations")
                            if callable(annotation_fn):
                                fig_anom = annotation_fn(fig_anom, bool(globals().get("annot_vals", False)))
                            st.plotly_chart(fig_anom, width="stretch", key="cousp_anomalies_top_chart")

    with tab_relances:
        if relances_df.empty:
            st.success("Aucun cas a relancer detecte dans le perimetre filtre.")
        else:
            with st.expander("Cas a relancer - tableau et filtres", expanded=True):
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
                st.download_button(
                    "Telecharger les cas a relancer filtres (CSV)",
                    data=relances_view.to_csv(index=False).encode("utf-8"),
                    file_name="cousp_cas_a_relancer.csv",
                    mime="text/csv",
                    key="download_cousp_relances_csv",
                )

            with st.expander("Visualisations des cas a relancer", expanded=True):
                with st.container():
                    rcol1, rcol2 = st.columns([1.1, 1.1])
                    with rcol1:
                        motif_counts = _cousp_build_category_counts(
                            relances_view,
                            "Motif_relance",
                            topn=10,
                        )
                        if motif_counts.empty:
                            st.info("Aucun motif de relance exploitable pour la visualisation.")
                        else:
                            fig_relance_motif = px.bar(
                                motif_counts.sort_values("Cas", ascending=True),
                                x="Cas",
                                y="Motif_relance",
                                orientation="h",
                                title="Top 10 des motifs de relance",
                                color="Cas",
                                color_continuous_scale="Oranges",
                            )
                            fig_relance_motif.update_layout(
                                template="plotly_white",
                                height=380,
                                margin=dict(t=60, r=20, b=30, l=20),
                                coloraxis_showscale=False,
                                xaxis_title="Nombre de cas",
                                yaxis_title="Motif de relance",
                            )
                            annotation_fn = globals().get("apply_plotly_value_annotations")
                            if callable(annotation_fn):
                                fig_relance_motif = annotation_fn(
                                    fig_relance_motif,
                                    bool(globals().get("annot_vals", False)),
                                )
                            st.plotly_chart(
                                fig_relance_motif,
                                width="stretch",
                                key="cousp_relances_motif_chart",
                            )
                    with rcol2:
                        relance_geo_options = [
                            col
                            for col in [
                                "Province_notification",
                                "Zone_de_sante_notification",
                                "Aire_de_sante_notification",
                            ]
                            if col in relances_view.columns
                        ]
                        if not relance_geo_options:
                            st.info("Aucune variable geographique exploitable pour les relances.")
                        else:
                            selected_relance_geo = st.selectbox(
                                "Repartition geographique des relances",
                                options=relance_geo_options,
                                index=min(1, len(relance_geo_options) - 1),
                                key="cousp_relances_geo_col",
                            )
                            relance_geo_counts = _cousp_build_category_counts(
                                relances_view,
                                selected_relance_geo,
                                topn=10,
                            )
                            if relance_geo_counts.empty:
                                st.info("Aucune donnee geographique exploitable pour ce filtre.")
                            else:
                                fig_relance_geo = px.bar(
                                    relance_geo_counts.sort_values("Cas", ascending=False),
                                    x=selected_relance_geo,
                                    y="Cas",
                                    title=f"Top 10 - {selected_relance_geo}",
                                    color="Cas",
                                    color_continuous_scale="Teal",
                                )
                                fig_relance_geo.update_layout(
                                    template="plotly_white",
                                    height=380,
                                    margin=dict(t=60, r=20, b=60, l=20),
                                    coloraxis_showscale=False,
                                    xaxis_tickangle=-35,
                                    xaxis_title=selected_relance_geo,
                                    yaxis_title="Nombre de cas",
                                )
                                annotation_fn = globals().get("apply_plotly_value_annotations")
                                if callable(annotation_fn):
                                    fig_relance_geo = annotation_fn(
                                        fig_relance_geo,
                                        bool(globals().get("annot_vals", False)),
                                    )
                                st.plotly_chart(
                                    fig_relance_geo,
                                    width="stretch",
                                    key="cousp_relances_geo_chart",
                                )

    with tab_recherche:
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
            with st.container():
                with st.expander("Apercu du dataset de recherche", expanded=False):
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
                    st_dataframe_safe(recherche_view.head(200), height=540)
            with st.expander("Visualisations du dataset de recherche", expanded=True):
                with st.container():
                    st.markdown("**Visualisations du dataset de recherche**")
                    rv1, rv2 = st.columns([1.4, 1.0])
                    with rv1:
                        _cousp_render_temporal_evolution_chart(
                            recherche_view,
                            title="Evolution des cas du dataset de recherche",
                            key_prefix="cousp_recherche_trend",
                        )
                    with rv2:
                        geo_options = [
                            col
                            for col in [
                                "Province_notification",
                                "Zone_de_sante_notification",
                                "Aire_de_sante_notification",
                            ]
                            if col in recherche_view.columns
                        ]
                        if not geo_options:
                            st.info("Aucune variable geographique exploitable pour la visualisation.")
                        else:
                            selected_geo = st.selectbox(
                                "Repartition geographique",
                                options=geo_options,
                                index=min(1, len(geo_options) - 1),
                                key="cousp_recherche_geo_col",
                            )
                            geo_counts = _cousp_build_category_counts(
                                recherche_view,
                                selected_geo,
                                topn=10,
                            )
                            if geo_counts.empty:
                                st.info("Aucune donnee geographique exploitable pour ce filtre.")
                            else:
                                fig_geo = px.bar(
                                    geo_counts.sort_values("Cas", ascending=False),
                                    x=selected_geo,
                                    y="Cas",
                                    title=f"Top 10 - {selected_geo}",
                                    color="Cas",
                                    color_continuous_scale="Blues",
                                )
                                fig_geo.update_layout(
                                    template="plotly_white",
                                    height=360,
                                    margin=dict(t=60, r=20, b=60, l=20),
                                    coloraxis_showscale=False,
                                    xaxis_tickangle=-35,
                                    xaxis_title=selected_geo,
                                    yaxis_title="Nombre de cas",
                                )
                                annotation_fn = globals().get("apply_plotly_value_annotations")
                                if callable(annotation_fn):
                                    fig_geo = annotation_fn(fig_geo, bool(globals().get("annot_vals", False)))
                                st.plotly_chart(fig_geo, width="stretch", key="cousp_recherche_geo_chart")

                    category_options = [
                        col
                        for col in [
                            "Classification_investigation",
                            "Resultat_final_labo",
                            "Issue",
                            "Sexe",
                            "Source_alerte",
                            "Tranche_age",
                        ]
                        if col in recherche_view.columns
                    ]
                    if category_options:
                        selected_category = st.selectbox(
                            "Profil du dataset de recherche",
                            options=category_options,
                            index=0,
                            key="cousp_recherche_profile_col",
                        )
                        category_counts = _cousp_build_category_counts(
                            recherche_view,
                            selected_category,
                            topn=12,
                        )
                        if not category_counts.empty:
                            fig_profile = px.pie(
                                category_counts,
                                names=selected_category,
                                values="Cas",
                                hole=0.45,
                                title=f"Distribution - {selected_category}",
                            )
                            fig_profile.update_layout(
                                template="plotly_white",
                                height=420,
                                margin=dict(t=60, r=20, b=20, l=20),
                            )
                            st.plotly_chart(fig_profile, width="stretch", key="cousp_recherche_profile_chart")
