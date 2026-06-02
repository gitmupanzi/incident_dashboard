"""Helpers internes pour construire le pack COUSP standard.

Ce module garde `incident_dashboard` autonome: aucun import runtime n'est fait
vers un projet frere.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Dict, Optional

import pandas as pd


SCHEMA_VARIABLES_COUSP = [
    {"Bloc": "Alerte", "Variable cle": "Provenance", "Priorite": "P1", "Type variable": "texte"},
    {"Bloc": "Alerte", "Variable cle": "Source_alerte", "Priorite": "P1", "Type variable": "texte"},
    {"Bloc": "Alerte", "Variable cle": "N_alerte", "Priorite": "P1", "Type variable": "identifiant"},
    {"Bloc": "Alerte", "Variable cle": "N_epid", "Priorite": "P1", "Type variable": "identifiant"},
    {"Bloc": "Alerte", "Variable cle": "N_labo", "Priorite": "P2", "Type variable": "identifiant"},
    {"Bloc": "Alerte", "Variable cle": "Semaine_epid", "Priorite": "P2", "Type variable": "derivee"},
    {"Bloc": "Alerte", "Variable cle": "Num_semaine_epid", "Priorite": "P1", "Type variable": "derivee"},
    {"Bloc": "Alerte", "Variable cle": "Annee_epid", "Priorite": "P1", "Type variable": "derivee"},
    {"Bloc": "Alerte", "Variable cle": "Province_notification", "Priorite": "P1", "Type variable": "geographie"},
    {"Bloc": "Alerte", "Variable cle": "Code_province", "Priorite": "P2", "Type variable": "geographie"},
    {"Bloc": "Alerte", "Variable cle": "Zone_de_sante_notification", "Priorite": "P1", "Type variable": "geographie"},
    {"Bloc": "Alerte", "Variable cle": "Aire_de_sante_notification", "Priorite": "P1", "Type variable": "geographie"},
    {"Bloc": "Alerte", "Variable cle": "Adresse", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Alerte", "Variable cle": "Localite", "Priorite": "P1", "Type variable": "geographie"},
    {"Bloc": "Alerte", "Variable cle": "Nom_complet", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Alerte", "Variable cle": "Sexe", "Priorite": "P1", "Type variable": "texte"},
    {"Bloc": "Alerte", "Variable cle": "Age_annee", "Priorite": "P1", "Type variable": "numerique"},
    {"Bloc": "Alerte", "Variable cle": "Age_mois", "Priorite": "P2", "Type variable": "numerique"},
    {"Bloc": "Alerte", "Variable cle": "Age", "Priorite": "P2", "Type variable": "derivee"},
    {"Bloc": "Alerte", "Variable cle": "Unite_age", "Priorite": "P2", "Type variable": "derivee"},
    {"Bloc": "Alerte", "Variable cle": "Age_en_ans", "Priorite": "P1", "Type variable": "derivee"},
    {"Bloc": "Alerte", "Variable cle": "Tranche_age", "Priorite": "P2", "Type variable": "derivee"},
    {"Bloc": "Alerte", "Variable cle": "Tranche_age_en_ans", "Priorite": "P1", "Type variable": "derivee"},
    {"Bloc": "Alerte", "Variable cle": "Profession", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Alerte", "Variable cle": "Autre_Profession", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Alerte", "Variable cle": "Statut_vaccinal", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Alerte", "Variable cle": "Signes_symptomes", "Priorite": "P1", "Type variable": "texte"},
    {"Bloc": "Alerte", "Variable cle": "Autres_Signes_symptomes", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Alerte", "Variable cle": "Date_notification", "Priorite": "P1", "Type variable": "date"},
    {"Bloc": "Alerte", "Variable cle": "Date_debut_symptomes", "Priorite": "P1", "Type variable": "date"},
    {"Bloc": "Investigation", "Variable cle": "Investigation", "Priorite": "P1", "Type variable": "derivee"},
    {"Bloc": "Investigation", "Variable cle": "Date_investigation", "Priorite": "P1", "Type variable": "date"},
    {"Bloc": "Investigation", "Variable cle": "Nom_investigateur", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Investigation", "Variable cle": "Cas_suspect", "Priorite": "P1", "Type variable": "texte"},
    {"Bloc": "Investigation", "Variable cle": "Classification_investigation", "Priorite": "P1", "Type variable": "texte"},
    {"Bloc": "Investigation", "Variable cle": "Malade_etait_il_un_contact_connu", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Investigation", "Variable cle": "Lien_epid_avec_un_cas", "Priorite": "P1", "Type variable": "texte"},
    {"Bloc": "Investigation", "Variable cle": "Facteur_exposition", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Investigation", "Variable cle": "Type_contact", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Investigation", "Variable cle": "Cas_source_id", "Priorite": "P2", "Type variable": "identifiant"},
    {"Bloc": "Investigation", "Variable cle": "Noms_cas_source", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Investigation", "Variable cle": "Type_de_lien", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Investigation", "Variable cle": "Narratif", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Prelevement", "Variable cle": "Prelevement", "Priorite": "P1", "Type variable": "texte"},
    {"Bloc": "Prelevement", "Variable cle": "Statut_patient_lors_du_prelevement", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Prelevement", "Variable cle": "Date_prelevement", "Priorite": "P1", "Type variable": "date"},
    {"Bloc": "Prelevement", "Variable cle": "Type_prelevement", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Laboratoire", "Variable cle": "Date_reception_echantillon_labo", "Priorite": "P1", "Type variable": "date"},
    {"Bloc": "Laboratoire", "Variable cle": "Date_analyse", "Priorite": "P1", "Type variable": "date"},
    {"Bloc": "Laboratoire", "Variable cle": "Date_confirmation", "Priorite": "P2", "Type variable": "date"},
    {"Bloc": "Laboratoire", "Variable cle": "Resultat_final_labo", "Priorite": "P1", "Type variable": "texte"},
    {"Bloc": "Laboratoire", "Variable cle": "GeneXpert_Ct_GP", "Priorite": "P2", "Type variable": "numerique"},
    {"Bloc": "Laboratoire", "Variable cle": "GeneXpert_Ct_NP", "Priorite": "P2", "Type variable": "numerique"},
    {"Bloc": "Laboratoire", "Variable cle": "Nom_laboratoire", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Prise en charge / Issue", "Variable cle": "Date_admission_au_CT", "Priorite": "P1", "Type variable": "date"},
    {"Bloc": "Prise en charge / Issue", "Variable cle": "Statut_avant_admission", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Prise en charge / Issue", "Variable cle": "Structure_de_prise_en_charge", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Prise en charge / Issue", "Variable cle": "Structure_de_prise_en_charge_2", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Prise en charge / Issue", "Variable cle": "Patient_en_isolement", "Priorite": "P1", "Type variable": "texte"},
    {"Bloc": "Prise en charge / Issue", "Variable cle": "Issue", "Priorite": "P1", "Type variable": "texte"},
    {"Bloc": "Prise en charge / Issue", "Variable cle": "Date_rapportage_Statut_final", "Priorite": "P2", "Type variable": "date"},
    {"Bloc": "Prise en charge / Issue", "Variable cle": "Date_issue", "Priorite": "P1", "Type variable": "derivee"},
    {"Bloc": "Prise en charge / Issue", "Variable cle": "Date_sortie_au_CT", "Priorite": "P2", "Type variable": "date"},
    {"Bloc": "Prise en charge / Issue", "Variable cle": "Type_sortie_CT", "Priorite": "P2", "Type variable": "texte"},
    {"Bloc": "Prise en charge / Issue", "Variable cle": "Duree_du_sejour_au_CT", "Priorite": "P1", "Type variable": "derivee"},
    {"Bloc": "Commentaire", "Variable cle": "Commentaire", "Priorite": "P2", "Type variable": "texte"},
]

VARIABLES_CIBLES_COUSP = [item["Variable cle"] for item in SCHEMA_VARIABLES_COUSP]

COLONNES_DATES_COUSP = [
    "Date_notification",
    "Date_debut_symptomes",
    "Date_investigation",
    "Date_prelevement",
    "Date_reception_echantillon_labo",
    "Date_analyse",
    "Date_confirmation",
    "Date_admission_au_CT",
    "Date_issue",
    "Date_sortie_au_CT",
    "Date_rapportage_Statut_final",
]

DELAIS_STANDARD_P1 = [
    ("delai_symptomes_notification", "Date_notification", "Date_debut_symptomes"),
    ("delai_notification_investigation", "Date_investigation", "Date_notification"),
    ("delai_notification_prelevement", "Date_prelevement", "Date_notification"),
    ("delai_prelevement_reception_labo", "Date_reception_echantillon_labo", "Date_prelevement"),
    ("delai_reception_analyse_labo", "Date_analyse", "Date_reception_echantillon_labo"),
    ("delai_notification_admission_ct", "Date_admission_au_CT", "Date_notification"),
    ("delai_admission_issue", "Date_issue", "Date_admission_au_CT"),
]

DELAIS_STANDARD_P2 = [
    ("delai_symptomes_investigation", "Date_investigation", "Date_debut_symptomes"),
    ("delai_symptomes_prelevement", "Date_prelevement", "Date_debut_symptomes"),
    ("delai_symptomes_reception_labo", "Date_reception_echantillon_labo", "Date_debut_symptomes"),
    ("delai_symptomes_analyse_labo", "Date_analyse", "Date_debut_symptomes"),
    ("delai_symptomes_admission_ct", "Date_admission_au_CT", "Date_debut_symptomes"),
    ("delai_symptomes_issue", "Date_issue", "Date_debut_symptomes"),
    ("delai_notification_reception_labo", "Date_reception_echantillon_labo", "Date_notification"),
    ("delai_notification_analyse_labo", "Date_analyse", "Date_notification"),
    ("delai_notification_issue", "Date_issue", "Date_notification"),
    ("delai_investigation_prelevement", "Date_prelevement", "Date_investigation"),
    ("delai_investigation_reception_labo", "Date_reception_echantillon_labo", "Date_investigation"),
    ("delai_investigation_analyse_labo", "Date_analyse", "Date_investigation"),
    ("delai_investigation_admission_ct", "Date_admission_au_CT", "Date_investigation"),
    ("delai_prelevement_analyse_labo", "Date_analyse", "Date_prelevement"),
    ("delai_analyse_confirmation", "Date_confirmation", "Date_analyse"),
    ("delai_admission_sortie_ct", "Date_sortie_au_CT", "Date_admission_au_CT"),
    ("delai_issue_rapportage_statut_final", "Date_rapportage_Statut_final", "Date_issue"),
    ("delai_sortie_rapportage_statut_final", "Date_rapportage_Statut_final", "Date_sortie_au_CT"),
]

DELAIS_STANDARD_COUSP = DELAIS_STANDARD_P1 + DELAIS_STANDARD_P2

OUI_NORMALISE = {"oui", "o", "yes", "y", "true", "vrai", "1"}
POSITIF_LAB_NORMALISE = {"positif", "positive", "confirme", "confirmee", "confirmee", "positive"}
DECES_NORMALISE = {"deces", "decede", "decedee", "mort", "deces hospitalier"}

ALIAS_SOURCE_VERS_COUSP = {
    "Date_debut_maladie": "Date_debut_symptomes",
    "Date_reception_labo": "Date_reception_echantillon_labo",
    "Date_resultat": "Date_analyse",
    "Resultat_labo": "Resultat_final_labo",
    "Classification_finale": "Classification_investigation",
    "Type_de_prelevement": "Type_prelevement",
}


def _normaliser_texte(valeur: Any) -> str:
    if valeur is None or pd.isna(valeur):
        return ""
    texte = str(valeur).strip().lower()
    texte = "".join(
        caractere
        for caractere in unicodedata.normalize("NFKD", texte)
        if not unicodedata.combining(caractere)
    )
    return " ".join(texte.replace("_", " ").split())


def _as_bool_oui(serie: pd.Series) -> pd.Series:
    return serie.map(lambda valeur: _normaliser_texte(valeur) in OUI_NORMALISE)


def _is_positive_labo(serie: pd.Series) -> pd.Series:
    return serie.map(lambda valeur: _normaliser_texte(valeur) in POSITIF_LAB_NORMALISE)


def _is_deces(serie: pd.Series) -> pd.Series:
    return serie.map(lambda valeur: _normaliser_texte(valeur) in DECES_NORMALISE)


def _is_suspect(serie_cas_suspect: pd.Series, serie_classification: pd.Series) -> pd.Series:
    cas_suspect = _as_bool_oui(serie_cas_suspect)
    classification = serie_classification.fillna("").map(_normaliser_texte)
    return cas_suspect | classification.str.contains("suspect|probable|compatible|confirme", regex=True)


def _choisir_date_reference(df: pd.DataFrame) -> pd.Series:
    if "Date_notification" in df.columns:
        reference = df["Date_notification"]
    elif "Date_debut_symptomes" in df.columns:
        reference = df["Date_debut_symptomes"]
    else:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    return pd.to_datetime(reference, errors="coerce")


def _calculer_tranche_age(age: Any) -> Optional[str]:
    if age is None or pd.isna(age):
        return pd.NA
    try:
        valeur = float(age)
    except (TypeError, ValueError):
        return pd.NA
    if valeur < 1:
        return "<1"
    if valeur < 5:
        return "1-4"
    if valeur < 15:
        return "5-14"
    if valeur < 25:
        return "15-24"
    if valeur < 35:
        return "25-34"
    if valeur < 45:
        return "35-44"
    if valeur < 60:
        return "45-59"
    return "60+"


def _apply_aliases(data: pd.DataFrame) -> pd.DataFrame:
    for source, target in ALIAS_SOURCE_VERS_COUSP.items():
        if source in data.columns and target not in data.columns:
            data[target] = data[source]
    return data


def normaliser_dataframe_cousp(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit etre un DataFrame pandas.")

    data = _apply_aliases(df.copy())

    for colonne in VARIABLES_CIBLES_COUSP:
        if colonne not in data.columns:
            data[colonne] = pd.NA

    for colonne in COLONNES_DATES_COUSP:
        data[colonne] = pd.to_datetime(data[colonne], errors="coerce")

    reference = _choisir_date_reference(data)
    reference_iso = reference.dt.isocalendar()

    data["Annee_epid"] = pd.to_numeric(data["Annee_epid"], errors="coerce").fillna(reference_iso.year).astype("Int64")
    data["Num_semaine_epid"] = pd.to_numeric(data["Num_semaine_epid"], errors="coerce").fillna(reference_iso.week).astype("Int64")

    semaine_num = data["Num_semaine_epid"].astype("string").str.zfill(2)
    semaine_calculee = "SE" + semaine_num
    data["Semaine_epid"] = data["Semaine_epid"].astype("string")
    semaine_vide = data["Semaine_epid"].isna() | data["Semaine_epid"].str.strip().eq("")
    data.loc[semaine_vide, "Semaine_epid"] = semaine_calculee.loc[semaine_vide]

    vide_invest = data["Investigation"].isna() | data["Investigation"].astype("string").str.strip().eq("")
    data.loc[vide_invest & data["Date_investigation"].notna(), "Investigation"] = "Oui"

    vide_prel = data["Prelevement"].isna() | data["Prelevement"].astype("string").str.strip().eq("")
    data.loc[vide_prel & data["Date_prelevement"].notna(), "Prelevement"] = "Oui"

    vide_classif = data["Classification_investigation"].isna() | data["Classification_investigation"].astype("string").str.strip().eq("")
    if "Classification_finale" in df.columns:
        data.loc[vide_classif, "Classification_investigation"] = df.loc[vide_classif, "Classification_finale"]

    vide_suspect = data["Cas_suspect"].isna() | data["Cas_suspect"].astype("string").str.strip().eq("")
    suspect_calc = _is_suspect(
        data["Cas_suspect"].fillna(pd.NA),
        data["Classification_investigation"].fillna(pd.NA),
    )
    data.loc[vide_suspect & suspect_calc, "Cas_suspect"] = "Oui"

    positif = _is_positive_labo(data["Resultat_final_labo"])
    data.loc[data["Date_confirmation"].isna() & positif, "Date_confirmation"] = data["Date_analyse"]

    data["Date_issue"] = data["Date_issue"].fillna(data["Date_sortie_au_CT"])
    deces = _is_deces(data["Issue"])
    data.loc[data["Date_issue"].isna() & deces, "Date_issue"] = data["Date_rapportage_Statut_final"]

    age_en_ans = pd.to_numeric(data["Age_en_ans"], errors="coerce")
    age_en_ans = age_en_ans.fillna(pd.to_numeric(data["Age_annee"], errors="coerce"))
    age_en_ans = age_en_ans.fillna(pd.to_numeric(data["Age_mois"], errors="coerce") / 12)
    if "Age" in data.columns:
        age_brut = pd.to_numeric(data["Age"], errors="coerce")
        unite = data["Unite_age"].astype("string").str.lower()
        age_calc = age_brut.where(unite.isna() | unite.str.contains("an", na=False), age_brut)
        age_calc = age_calc.where(~unite.str.contains("mois", na=False), age_brut / 12)
        age_calc = age_calc.where(~unite.str.contains("jour", na=False), age_brut / 365.25)
        age_calc = age_calc.where(~unite.str.contains("semaine", na=False), age_brut / 52.0)
        age_en_ans = age_en_ans.fillna(age_calc)
    data["Age_en_ans"] = age_en_ans.round(2).astype("Float64")
    data["Tranche_age_en_ans"] = data["Tranche_age_en_ans"].fillna(data["Age_en_ans"].map(_calculer_tranche_age))
    data["Tranche_age"] = data["Tranche_age"].fillna(data["Tranche_age_en_ans"])

    duree_calculee = (data["Date_issue"] - data["Date_admission_au_CT"]).dt.days
    data["Duree_du_sejour_au_CT"] = pd.to_numeric(data["Duree_du_sejour_au_CT"], errors="coerce").fillna(duree_calculee).astype("Float64")

    return data


def calculer_delais_standard_cousp(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for nom_delai, col_fin, col_debut in DELAIS_STANDARD_COUSP:
        fin = pd.to_datetime(data[col_fin], errors="coerce")
        debut = pd.to_datetime(data[col_debut], errors="coerce")
        # Les anomalies COUSP sont suivies au niveau date, sans tenir compte des heures.
        delai = (fin.dt.normalize() - debut.dt.normalize()).dt.days
        data[nom_delai] = delai.astype("Float64")
    return data


def calculer_kpis_operationnels(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    total_lignes = len(data)
    total_alertes = int(data["N_alerte"].notna().sum()) if "N_alerte" in data.columns else total_lignes
    total_cas_notifies = int(data["N_epid"].notna().sum()) if "N_epid" in data.columns else total_lignes

    investigation = _as_bool_oui(data["Investigation"]) | data["Date_investigation"].notna()
    suspects = _is_suspect(data["Cas_suspect"], data["Classification_investigation"])
    preleves = _as_bool_oui(data["Prelevement"]) | data["Date_prelevement"].notna()
    analyses = data["Resultat_final_labo"].fillna("").astype("string").str.strip().ne("")
    positifs = _is_positive_labo(data["Resultat_final_labo"])
    population_pec = suspects | positifs
    admis = data["Date_admission_au_CT"].notna()
    isoles = _as_bool_oui(data["Patient_en_isolement"])
    deces = _is_deces(data["Issue"])

    def _ligne(indicateur: str, numerateur: int, denominateur: Optional[int], utilite: str) -> Dict[str, Any]:
        valeur = pd.NA if denominateur in (None, 0) else round((numerateur / denominateur) * 100, 2)
        return {
            "Indicateur": indicateur,
            "Valeur": valeur,
            "Numerateur": numerateur,
            "Denominateur": denominateur,
            "Utilite": utilite,
        }

    duree_mediane = pd.to_numeric(data["Duree_du_sejour_au_CT"], errors="coerce").dropna()
    rows = [
        {"Indicateur": "Total alertes", "Valeur": total_alertes, "Numerateur": total_alertes, "Denominateur": pd.NA, "Utilite": "Mesure le volume de signalements."},
        {"Indicateur": "Total cas notifies", "Valeur": total_cas_notifies, "Numerateur": total_cas_notifies, "Denominateur": pd.NA, "Utilite": "Mesure les cas enregistres dans la liste lineaire."},
        _ligne("Taux d'investigation (%)", int(investigation.sum()), total_alertes if total_alertes > 0 else total_lignes, "Suit la reponse terrain."),
        _ligne("Taux de cas suspects (%)", int((suspects & investigation).sum()), int(investigation.sum()), "Mesure la validation des alertes."),
        _ligne("Taux de prelevement (%)", int((preleves & suspects).sum()), int(suspects.sum()), "Controle le passage vers le laboratoire."),
        _ligne("Taux de positivite labo (%)", int(positifs.sum()), int(analyses.sum()), "Oriente l'analyse epidemiologique."),
        _ligne("Taux d'admission (%)", int((admis & population_pec).sum()), int(population_pec.sum()), "Mesure du transfert vers la prise en charge."),
        _ligne("Taux d'isolement (%)", int((isoles & population_pec).sum()), int(population_pec.sum()), "Suit la reduction du risque de transmission."),
        _ligne("Letalite (%)", int((deces & positifs).sum()), int(positifs.sum()), "Suit la gravite et la qualite de prise en charge."),
        {"Indicateur": "Duree mediane de sejour", "Valeur": round(float(duree_mediane.median()), 2) if not duree_mediane.empty else pd.NA, "Numerateur": pd.NA, "Denominateur": pd.NA, "Utilite": "Aide a planifier les lits et les ressources."},
    ]
    return pd.DataFrame(rows)


def resumer_delais_operationnels(df: pd.DataFrame) -> pd.DataFrame:
    lignes = []
    for nom_delai, _, _ in DELAIS_STANDARD_P1:
        serie = pd.to_numeric(df[nom_delai], errors="coerce").dropna()
        if serie.empty:
            lignes.append({"Indicateur": nom_delai, "Mediane": pd.NA, "IQR_Q1": pd.NA, "IQR_Q3": pd.NA, "Minimum": pd.NA, "Maximum": pd.NA, "Proportion_retards_%": pd.NA})
            continue
        lignes.append(
            {
                "Indicateur": nom_delai,
                "Mediane": round(float(serie.median()), 2),
                "IQR_Q1": round(float(serie.quantile(0.25)), 2),
                "IQR_Q3": round(float(serie.quantile(0.75)), 2),
                "Minimum": round(float(serie.min()), 2),
                "Maximum": round(float(serie.max()), 2),
                "Proportion_retards_%": round(float((serie.gt(1)).mean() * 100), 2),
            }
        )
    return pd.DataFrame(lignes)


def analyser_qualite_donnees_cousp(
    df: pd.DataFrame,
    *,
    seuil_acceptable: float = 5.0,
    seuil_surveillance: float = 20.0,
) -> pd.DataFrame:
    if seuil_acceptable < 0 or seuil_surveillance < 0:
        raise ValueError("Les seuils de completude doivent etre positifs.")
    if seuil_acceptable > seuil_surveillance:
        raise ValueError("seuil_acceptable ne peut pas etre superieur a seuil_surveillance.")

    total_lignes = len(df)
    resultats = []
    for ligne in SCHEMA_VARIABLES_COUSP:
        variable = ligne["Variable cle"]
        colonne_absente = variable not in df.columns
        if colonne_absente:
            nb_renseignees = 0
        else:
            nb_renseignees = int(df[variable].notna().sum())
        nb_manquantes = int(total_lignes - nb_renseignees)
        pct_missing = round((nb_manquantes / total_lignes) * 100, 2) if total_lignes > 0 else 0.0

        if colonne_absente:
            decision = "Colonne absente"
        elif pct_missing == 0:
            decision = "OK"
        elif pct_missing <= float(seuil_acceptable):
            decision = "Acceptable"
        elif pct_missing <= float(seuil_surveillance):
            decision = "A surveiller"
        else:
            decision = "Prioritaire"

        resultats.append(
            {
                "Bloc": ligne["Bloc"],
                "Variable cle": variable,
                "Priorite": ligne["Priorite"],
                "Type variable": ligne["Type variable"],
                "Total lignes": total_lignes,
                "Renseignees": nb_renseignees,
                "Manquantes": nb_manquantes,
                "% missing": pct_missing,
                "Decision / observation": decision,
            }
        )
    return pd.DataFrame(resultats)


def detecter_anomalies_dates(df: pd.DataFrame) -> pd.DataFrame:
    colonnes_ids = [
        colonne
        for colonne in [
            "N_alerte",
            "N_epid",
            "N_labo",
            "Province_notification",
            "Zone_de_sante_notification",
            "Aire_de_sante_notification",
            "Nom_complet",
        ]
        if colonne in df.columns
    ]

    lignes = []
    for nom_delai, _, _ in DELAIS_STANDARD_COUSP:
        masque_negatif = pd.to_numeric(df[nom_delai], errors="coerce").lt(0)
        if not masque_negatif.any():
            continue
        sous_df = df.loc[masque_negatif, colonnes_ids].copy()
        sous_df["Variable_anomalie"] = nom_delai
        sous_df["Valeur"] = pd.to_numeric(df.loc[masque_negatif, nom_delai], errors="coerce").values
        sous_df["Type_anomalie"] = "Delai negatif"
        lignes.append(sous_df)

    if not lignes:
        return pd.DataFrame(columns=colonnes_ids + ["Variable_anomalie", "Valeur", "Type_anomalie"])
    return pd.concat(lignes, ignore_index=True)


def detecter_cas_a_relancer(df: pd.DataFrame) -> pd.DataFrame:
    index = df.index
    investigation = _as_bool_oui(df["Investigation"]) | df["Date_investigation"].notna()
    suspects = _is_suspect(df["Cas_suspect"], df["Classification_investigation"])
    preleves = _as_bool_oui(df["Prelevement"]) | df["Date_prelevement"].notna()
    reception_labo = df["Date_reception_echantillon_labo"].notna()
    analyse_labo = df["Date_analyse"].notna()
    positif_labo = _is_positive_labo(df["Resultat_final_labo"])
    deces = _is_deces(df["Issue"])

    motifs = pd.Series("", index=index, dtype="string")
    motifs = motifs.mask(df["Date_notification"].notna() & ~investigation, motifs + "Alerte sans investigation; ")
    motifs = motifs.mask(suspects & ~preleves, motifs + "Cas suspect sans prelevement; ")
    motifs = motifs.mask(df["Date_prelevement"].notna() & ~reception_labo, motifs + "Prelevement sans reception labo; ")
    motifs = motifs.mask(reception_labo & ~analyse_labo, motifs + "Reception sans analyse labo; ")
    motifs = motifs.mask(positif_labo & df["Date_confirmation"].isna(), motifs + "Positif sans date confirmation; ")
    motifs = motifs.mask(deces & df["Date_issue"].isna(), motifs + "Deces sans date issue; ")

    colonnes_sortie = [
        colonne
        for colonne in [
            "N_alerte",
            "N_epid",
            "N_labo",
            "Province_notification",
            "Zone_de_sante_notification",
            "Aire_de_sante_notification",
            "Localite",
            "Nom_complet",
            "Classification_investigation",
            "Resultat_final_labo",
            "Issue",
        ]
        if colonne in df.columns
    ]

    relances = df.loc[motifs.str.strip() != "", colonnes_sortie].copy()
    relances["Motif_relance"] = motifs.loc[relances.index].str.rstrip("; ").values
    return relances.reset_index(drop=True)


def construire_synthese_operationnelle(df: pd.DataFrame) -> pd.DataFrame:
    kpis = calculer_kpis_operationnels(df).copy()
    kpis.insert(0, "Section", "KPI")

    delais = resumer_delais_operationnels(df).copy()
    if not delais.empty:
        delais.insert(0, "Section", "Delais")

    return pd.concat([kpis, delais], ignore_index=True, sort=False)


def preparer_recherche_dataset(df: pd.DataFrame, *, anonymiser: bool = False) -> pd.DataFrame:
    data = df.copy()
    if anonymiser and "Nom_complet" in data.columns:
        data = data.drop(columns=["Nom_complet"])
    return data


def generer_feuilles_sortie_cousp_local(
    df: pd.DataFrame,
    *,
    anonymiser_recherche: bool = False,
    seuil_acceptable: float = 5.0,
    seuil_surveillance: float = 20.0,
) -> Dict[str, pd.DataFrame]:
    data = normaliser_dataframe_cousp(df)
    data = calculer_delais_standard_cousp(data)
    return {
        "LL_standard_nettoyee": data,
        "Synthese_operationnelle": construire_synthese_operationnelle(data),
        "Completeness_variables_cles": analyser_qualite_donnees_cousp(
            data,
            seuil_acceptable=seuil_acceptable,
            seuil_surveillance=seuil_surveillance,
        ),
        "Anomalies_dates": detecter_anomalies_dates(data),
        "Cas_a_relancer": detecter_cas_a_relancer(data),
        "Recherche_dataset": preparer_recherche_dataset(data, anonymiser=anonymiser_recherche),
    }
