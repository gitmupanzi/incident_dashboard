# -*- coding: utf-8 -*-
# dataminsante/colonne_valeur/colonne_nettoyage.py
# Notice : Pour le nettoyage de colonnes de df et son organisatison

import pandas as pd
import re
import unicodedata
import logging
from typing import Any, Optional, Sequence, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Définir le chemin racine du projet à partir de ce fichier
base_dir = Path(__file__).resolve().parents[2]
mapping_file_path = base_dir / "data" / "Rename_columns.xlsx"
# Vérification de conformité des colonnes
def verifier_colonnes(df: pd.DataFrame, colonnes_attendues: list[str], afficher: str = "toutes") -> None:
    """
    Vérifie la conformité des colonnes du DataFrame par rapport à une liste de référence.
    
    Affiche :
    - les colonnes manquantes (attendues mais absentes)
    - les colonnes non attendues (présentes mais non attendues)
    
    Paramètres :
    - df : DataFrame à vérifier
    - colonnes_attendues : Liste des noms de colonnes attendues
    - afficher : "toutes" | "manquantes" | "non_attendues" | "rien"
    """
    if afficher not in {"toutes", "manquantes", "non_attendues", "rien"}:
        raise ValueError("Le paramètre 'afficher' doit être 'toutes', 'manquantes', 'non_attendues' ou 'rien'.")

    if afficher == "rien":
        return  # Ne rien afficher du tout

    colonnes_df = df.columns.tolist()
    manquantes = [col for col in colonnes_attendues if col not in colonnes_df]
    non_attendues = [col for col in colonnes_df if col not in colonnes_attendues]

    if afficher in ("toutes", "manquantes"):
        if manquantes:
            logger.warning("⚠️ Colonnes manquantes (absentes du DataFrame mais attendues) :")
            for col in manquantes:
                logger.warning(f"  - {col}")
        else:
            logger.info("✅ Aucune colonne manquante.")

    if afficher in ("toutes", "non_attendues"):
        if non_attendues:
            logger.warning("⚠️ Colonnes non attendues (présentes dans le DataFrame mais non attendues) :")
            for col in non_attendues:
                logger.warning(f"  - {col}")
        else:
            logger.info("✅ Aucune colonne inattendue.")

# Nettoyage individuel d’un nom de colonne 
def standardiser_nom(nom_col) -> str:
    """
    Nettoie un nom de colonne :
    - Gère int, float, NaN, None
    - Supprime les accents
    - Minuscule + remplace tout caractère spécial/ponctuation par un underscore
    - Supprime les underscores multiples
    - Capitalise le premier mot
    """

    # 1) gérer None / NaN
    if nom_col is None or (isinstance(nom_col, float) and pd.isna(nom_col)):
        return ""

    # 2) conversion obligatoire en string (LA LIGNE QUI MANQUAIT)
    nom_col = str(nom_col)

    # 2bis) tentative légère de réparation de texte mal encodé (mojibake fréquent)
    if any(token in nom_col for token in ("Ã", "Â", "â€™", "â€")):
        try:
            nom_col = nom_col.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    # 3) normalisation unicode (suppression accents sans perdre les lettres)
    nom_col = "".join(
        ch for ch in unicodedata.normalize("NFKD", nom_col)
        if not unicodedata.combining(ch)
    )

    # 4) nettoyage
    nom_col = nom_col.strip().lower()
    nom_col = re.sub(r"[^\w\s]", "_", nom_col)
    nom_col = re.sub(r"\s+", "_", nom_col)
    nom_col = re.sub(r"_+", "_", nom_col)
    nom_col = nom_col.strip('_')

    # 5) capitaliser premier mot
    mots = nom_col.split('_')
    if mots and mots[0] != "":
        mots[0] = mots[0].capitalize()

    return '_'.join(mots)

# Renommer les colonnes à partir d’un fichier de mapping Excel
def renommer_colonnes_selon_mapping(df: pd.DataFrame, mapping_file: Union[str, Path] = mapping_file_path) -> pd.DataFrame:
    """
    Renomme les colonnes selon un mapping Excel (colonnes : ancien_nom, nouveau_nom).
    """
    mapping_file = Path(mapping_file)
    if not mapping_file.exists():
        raise FileNotFoundError(f"❌ Fichier de mapping introuvable : {mapping_file}")

    mapping_df = pd.read_excel(mapping_file, dtype=str).dropna()
    if mapping_df.shape[1] < 2:
        raise ValueError("❌ Le fichier de mapping doit avoir au moins deux colonnes.")

    mapping_dict = dict(zip(mapping_df.iloc[:, 0].str.strip(), mapping_df.iloc[:, 1].str.strip()))
    mapping_utilisable = {k: v for k, v in mapping_dict.items() if k in df.columns}

    if not mapping_utilisable:
        logger.warning("⚠️ Aucun nom de colonne du mapping ne correspond à celles du DataFrame.")
        return df

    logger.info(f"✅ Colonnes renommées selon mapping : {mapping_utilisable}")
    return df.rename(columns=mapping_utilisable)


# Standardisation globale des noms de colonnes (avec ou sans mapping)
def standardiser_noms_colonnes(
    df: pd.DataFrame,
    mapping_file: Optional[Union[str, Path]] = None,
    nom_col: Optional[str] = None
) -> Union[pd.DataFrame, str]:
    """
    Standardise les noms de colonnes d'un DataFrame, avec mapping optionnel.

    Comportement :
    - si `nom_col` est fourni : retourne uniquement la version nettoyee de ce nom ;
    - sinon : nettoie toutes les colonnes du DataFrame avec `standardiser_nom` ;
    - si `mapping_file` est fourni : applique ensuite
      `renommer_colonnes_selon_mapping`.

    Cette fonction est la brique de base du pipeline de preparation des colonnes
    dans `dataminsante`. Elle permet de separer deux usages :
    - standardisation seule avec `mapping_file=None` ;
    - standardisation suivie d'un renommage controle par un fichier de reference.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("❌ L'objet passé n'est pas un DataFrame.")

    if nom_col:
        if nom_col not in df.columns:
            raise ValueError(f"❌ Colonne '{nom_col}' absente du DataFrame.")
        return standardiser_nom(nom_col)

    df.columns = [standardiser_nom(col) for col in df.columns]

    if mapping_file:
        df = renommer_colonnes_selon_mapping(df, mapping_file)

    return df


def completer_liste_colonnes(
    colonnes_base: Sequence[str],
    colonnes_optionnelles: Optional[Sequence[str]] = None,
    activer: bool = True,
) -> list[str]:
    """
    Construit une liste de colonnes de reference a partir d'une base,
    avec ajout optionnel de colonnes supplementaires.

    L'ordre est conserve et les doublons sont retires.
    """
    if colonnes_base is None:
        raise ValueError("colonnes_base ne peut pas etre None.")

    resultat = list(colonnes_base)
    if activer and colonnes_optionnelles:
        resultat.extend(list(colonnes_optionnelles))

    return list(dict.fromkeys(resultat))


def transferer_valeurs_colonnes(
    df: pd.DataFrame,
    colonne_source: Union[str, Sequence[str], None] = None,
    colonne_cible: Union[str, Sequence[str], None] = None,
    mapping: Optional[dict[str, str]] = None,
    masque: Optional[pd.Series] = None,
    colonne_filtre: Optional[str] = None,
    valeurs_filtre: Optional[list[Any]] = None,
    normaliser_filtre_texte: bool = False,
    ecraser: bool = False,
    supprimer_source: bool = False,
    valeur_vide: Any = pd.NA,
    ignorer_colonnes_absentes: bool = False,
    retourner_rapport: bool = False,
    dossier_sortie: Optional[Union[str, Path]] = None,
    nom_fichier_rapport: Optional[str] = None,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict[str, Any]]]:
    """
    Copie ou deplace des valeurs d'une colonne source vers une colonne cible
    sur les memes lignes.

    Args:
        df: DataFrame source.
        colonne_source: Colonne a lire, ou liste de colonnes source.
        colonne_cible: Colonne a alimenter, ou liste de colonnes cible.
        mapping: Dictionnaire {colonne_source: colonne_cible}.
        masque: Serie booleenne optionnelle pour limiter les lignes a traiter.
        colonne_filtre: Colonne optionnelle utilisee pour construire un masque.
        valeurs_filtre: Liste de valeurs attendues dans `colonne_filtre`.
        normaliser_filtre_texte: Si True, normalise accents/casse/espaces avant comparaison.
        ecraser: Si True, remplace aussi les valeurs deja presentes dans la cible.
        supprimer_source: Si True, vide la source sur les lignes transferees.
        valeur_vide: Valeur utilisee pour creer la colonne cible ou vider la source.
        ignorer_colonnes_absentes: Si True, ignore les colonnes source ou filtre absentes
            avec un warning au lieu de lever une erreur.
        retourner_rapport: Si True, retourne aussi un dictionnaire de synthese.
        dossier_sortie: Dossier optionnel d'export du rapport si `retourner_rapport=True`.
        nom_fichier_rapport: Nom de base du rapport exporte (sans extension).

    Returns:
        pd.DataFrame, ou tuple (DataFrame, rapport) si `retourner_rapport=True`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit etre un DataFrame pandas.")

    def _est_serie_vide(serie: pd.Series) -> pd.Series:
        """
        Retourne un masque True pour les valeurs considerees comme vides:
        NaN/NA/None ou chaine vide apres strip.
        """
        masque = serie.isna()
        try:
            masque |= serie.astype("string").str.strip().eq("")
        except Exception:
            # Pour des types non convertibles proprement, on garde au minimum isna().
            pass
        return masque

    def _exporter_rapport_si_demande(rapport: dict[str, Any]) -> dict[str, Any]:
        if not retourner_rapport or not dossier_sortie:
            return rapport

        dossier = Path(dossier_sortie)
        dossier.mkdir(parents=True, exist_ok=True)

        base_nom = nom_fichier_rapport or f"rapport_transfert_colonnes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        fichier_excel = dossier / f"{base_nom}.xlsx"

        details = rapport.get("details")
        if isinstance(details, list) and details:
            df_rapport = pd.DataFrame(details)
        else:
            df_rapport = pd.DataFrame([rapport])

        df_rapport.to_excel(fichier_excel, index=False)
        rapport["fichier_rapport"] = str(fichier_excel)
        logger.info("Rapport de transfert exporte : %s", fichier_excel)
        return rapport

    if mapping is not None:
        if colonne_source is not None or colonne_cible is not None:
            raise ValueError(
                "Utiliser soit `mapping`, soit le couple `colonne_source` / `colonne_cible`, pas les deux."
            )
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("`mapping` doit etre un dictionnaire non vide.")
        if any(source == cible for source, cible in mapping.items()):
            raise ValueError("Le mapping contient au moins une correspondance source == cible, ce qui est ambigu.")

        sources = list(mapping.keys())
        cibles = list(mapping.values())
        if len(set(sources)) != len(sources):
            raise ValueError("Les colonnes source du mapping doivent etre uniques.")
        if len(set(cibles)) != len(cibles):
            raise ValueError("Les colonnes cible du mapping doivent etre uniques.")
        df_resultat = transferer_valeurs_colonnes(
            df,
            colonne_source=sources,
            colonne_cible=cibles,
            masque=masque,
            colonne_filtre=colonne_filtre,
            valeurs_filtre=valeurs_filtre,
            normaliser_filtre_texte=normaliser_filtre_texte,
            ecraser=ecraser,
            supprimer_source=supprimer_source,
            valeur_vide=valeur_vide,
            ignorer_colonnes_absentes=ignorer_colonnes_absentes,
            retourner_rapport=True,
        )
        df_resultat, details_rapport = df_resultat
        rapport = {
            "mode": "mapping",
            "correspondances": len(mapping),
            "total_transferts": sum(item.get("transferts", 0) for item in details_rapport.get("details", [])),
            "details": details_rapport.get("details", []),
        }
        nb_ignorees = sum(1 for item in rapport["details"] if item.get("ignoree"))
        nb_sans_transfert = sum(1 for item in rapport["details"] if item.get("transferts", 0) == 0 and not item.get("ignoree"))
        nb_avec_transfert = sum(1 for item in rapport["details"] if item.get("transferts", 0) > 0)
        rapport = _exporter_rapport_si_demande(rapport)
        if nb_ignorees:
            logger.warning(
                "Transfert via mapping : %s correspondance(s) ignoree(s) car colonne absente.",
                nb_ignorees,
            )
        if nb_sans_transfert:
            logger.info(
                "Transfert via mapping : %s correspondance(s) sans transfert effectif.",
                nb_sans_transfert,
            )
        logger.info(
            "Transfert via mapping termine : %s correspondance(s) | avec transfert=%s | total transferts=%s.",
            len(mapping),
            nb_avec_transfert,
            rapport["total_transferts"],
        )
        return (df_resultat, rapport) if retourner_rapport else df_resultat

    if colonne_source is None or colonne_cible is None:
        raise ValueError(
            "Vous devez fournir soit `mapping`, soit `colonne_source` et `colonne_cible`."
        )

    est_sequence_source = isinstance(colonne_source, Sequence) and not isinstance(colonne_source, str)
    est_sequence_cible = isinstance(colonne_cible, Sequence) and not isinstance(colonne_cible, str)

    if est_sequence_source or est_sequence_cible:
        if not (est_sequence_source and est_sequence_cible):
            raise TypeError(
                "colonne_source et colonne_cible doivent etre tous les deux des chaines "
                "ou tous les deux des sequences de meme longueur."
            )

        sources = list(colonne_source)
        cibles = list(colonne_cible)
        if len(sources) != len(cibles):
            raise ValueError("Les listes colonne_source et colonne_cible doivent avoir la meme longueur.")
        if any(str(source) == str(cible) for source, cible in zip(sources, cibles)):
            raise ValueError("Une correspondance source == cible a ete detectee dans les listes fournies.")
        if len({str(source) for source in sources}) != len(sources):
            raise ValueError("Les colonnes source doivent etre uniques dans le transfert groupe.")
        if len({str(cible) for cible in cibles}) != len(cibles):
            raise ValueError("Les colonnes cible doivent etre uniques dans le transfert groupe.")

        df_resultat = df.copy()
        details = []
        for source, cible in zip(sources, cibles):
            resultat = transferer_valeurs_colonnes(
                df_resultat,
                colonne_source=str(source),
                colonne_cible=str(cible),
                masque=masque,
                colonne_filtre=colonne_filtre,
                valeurs_filtre=valeurs_filtre,
                normaliser_filtre_texte=normaliser_filtre_texte,
                ecraser=ecraser,
                supprimer_source=supprimer_source,
                valeur_vide=valeur_vide,
                ignorer_colonnes_absentes=ignorer_colonnes_absentes,
                retourner_rapport=True,
            )
            df_resultat, rapport_item = resultat
            details.append(rapport_item)

        rapport = {
            "mode": "groupe",
            "correspondances": len(sources),
            "total_transferts": sum(item.get("transferts", 0) for item in details),
            "details": details,
        }
        nb_ignorees = sum(1 for item in details if item.get("ignoree"))
        nb_sans_transfert = sum(1 for item in details if item.get("transferts", 0) == 0 and not item.get("ignoree"))
        nb_avec_transfert = sum(1 for item in details if item.get("transferts", 0) > 0)
        rapport = _exporter_rapport_si_demande(rapport)
        if nb_ignorees:
            logger.warning(
                "Transfert groupe : %s correspondance(s) ignoree(s) car colonne absente.",
                nb_ignorees,
            )
        if nb_sans_transfert:
            logger.info(
                "Transfert groupe : %s correspondance(s) sans transfert effectif.",
                nb_sans_transfert,
            )
        logger.info(
            "Transfert groupe termine : %s correspondance(s) | avec transfert=%s | total transferts=%s.",
            len(sources),
            nb_avec_transfert,
            rapport["total_transferts"],
        )
        return (df_resultat, rapport) if retourner_rapport else df_resultat

    colonne_source = str(colonne_source)
    colonne_cible = str(colonne_cible)

    if colonne_source not in df.columns:
        if ignorer_colonnes_absentes:
            logger.warning(
                "Colonne source absente ignoree : '%s' -> '%s'.",
                colonne_source,
                colonne_cible,
            )
            rapport = {
                "mode": "simple",
                "colonne_source": colonne_source,
                "colonne_cible": colonne_cible,
                "colonne_cible_creee": False,
                "colonne_filtre": colonne_filtre,
                "valeurs_filtre": list(valeurs_filtre) if valeurs_filtre is not None else None,
                "normaliser_filtre_texte": normaliser_filtre_texte,
                "ecraser": ecraser,
                "supprimer_source": supprimer_source,
                "lignes_filtrees": 0,
                "sources_non_vides": 0,
                "transferts": 0,
                "ignoree": True,
                "raison": f"Colonne source absente : {colonne_source}",
            }
            rapport = _exporter_rapport_si_demande(rapport)
            return (df.copy(), rapport) if retourner_rapport else df.copy()
        raise KeyError(f"Colonne source absente : {colonne_source}")
    if colonne_source == colonne_cible:
        raise ValueError("colonne_source et colonne_cible ne peuvent pas etre identiques.")

    df_resultat = df.copy()
    colonne_cible_creee = False

    if colonne_cible not in df_resultat.columns:
        df_resultat[colonne_cible] = valeur_vide
        colonne_cible_creee = True
        logger.info("Colonne cible creee automatiquement : %s", colonne_cible)

    if masque is not None:
        if not isinstance(masque, pd.Series):
            raise TypeError("masque doit etre une serie booleenne pandas.")
        masque_lignes = masque.reindex(df_resultat.index, fill_value=False).astype(bool)
    elif colonne_filtre is not None and valeurs_filtre is not None:
        if colonne_filtre not in df_resultat.columns:
            if ignorer_colonnes_absentes:
                logger.warning(
                    "Colonne filtre absente ignoree : '%s' pour transfert '%s' -> '%s'.",
                    colonne_filtre,
                    colonne_source,
                    colonne_cible,
                )
                masque_lignes = pd.Series(False, index=df_resultat.index)
            else:
                raise KeyError(f"Colonne filtre absente : {colonne_filtre}")
        else:
            serie_filtre = df_resultat[colonne_filtre]
            if normaliser_filtre_texte:
                def _normaliser_texte(valeur: Any) -> str:
                    if valeur is None or pd.isna(valeur):
                        return ""
                    texte = str(valeur).strip().lower()
                    return "".join(
                        ch for ch in unicodedata.normalize("NFKD", texte)
                        if not unicodedata.combining(ch)
                    )

                valeurs_filtre_norm = {_normaliser_texte(valeur) for valeur in valeurs_filtre}
                masque_lignes = serie_filtre.map(_normaliser_texte).isin(valeurs_filtre_norm)
            else:
                masque_lignes = serie_filtre.isin(valeurs_filtre)
            masque_lignes = masque_lignes.reindex(df_resultat.index, fill_value=False).astype(bool)
            logger.info(
                "Filtre applique sur '%s' pour %s valeur(s) cible(s).",
                colonne_filtre,
                len(valeurs_filtre),
            )
    else:
        masque_lignes = pd.Series(True, index=df_resultat.index)

    masque_source_non_vide = ~_est_serie_vide(df_resultat[colonne_source])
    if ecraser:
        masque_cible = pd.Series(True, index=df_resultat.index)
    else:
        masque_cible = _est_serie_vide(df_resultat[colonne_cible])

    masque_transfert = masque_lignes & masque_source_non_vide & masque_cible
    nb_lignes_filtrees = int(masque_lignes.sum())
    nb_sources_non_vides = int((masque_lignes & masque_source_non_vide).sum())
    nb_transferts = int(masque_transfert.sum())

    df_resultat.loc[masque_transfert, colonne_cible] = df_resultat.loc[masque_transfert, colonne_source]

    if supprimer_source and nb_transferts > 0:
        df_resultat.loc[masque_transfert, colonne_source] = valeur_vide

    action = "deplacees" if supprimer_source else "copiees"
    logger.info(
        "Valeurs %s de '%s' vers '%s' : %s ligne(s) transferee(s) | lignes filtrees=%s | sources non vides=%s.",
        action,
        colonne_source,
        colonne_cible,
        nb_transferts,
        nb_lignes_filtrees,
        nb_sources_non_vides,
    )
    if nb_lignes_filtrees == 0:
        logger.warning(
            "Aucune ligne eligible pour le transfert '%s' -> '%s' (masque/filtre vide).",
            colonne_source,
            colonne_cible,
        )
    elif nb_sources_non_vides == 0:
        logger.warning(
            "Aucune valeur source exploitable pour le transfert '%s' -> '%s'.",
            colonne_source,
            colonne_cible,
        )
    elif nb_transferts == 0:
        if ecraser:
            logger.info(
                "Aucun transfert realise pour '%s' -> '%s' malgre ecrasement autorise.",
                colonne_source,
                colonne_cible,
            )
        else:
            logger.info(
                "Aucun transfert realise pour '%s' -> '%s' car la cible etait deja renseignee ou aucune ligne restante n'etait eligible.",
                colonne_source,
                colonne_cible,
            )

    rapport = {
        "mode": "simple",
        "colonne_source": colonne_source,
        "colonne_cible": colonne_cible,
        "colonne_cible_creee": colonne_cible_creee,
        "colonne_filtre": colonne_filtre,
        "valeurs_filtre": list(valeurs_filtre) if valeurs_filtre is not None else None,
        "normaliser_filtre_texte": normaliser_filtre_texte,
        "ecraser": ecraser,
        "supprimer_source": supprimer_source,
        "lignes_filtrees": nb_lignes_filtrees,
        "sources_non_vides": nb_sources_non_vides,
        "transferts": nb_transferts,
    }
    rapport = _exporter_rapport_si_demande(rapport)

    return (df_resultat, rapport) if retourner_rapport else df_resultat


def harmoniser_schema_colonnes(
    df: pd.DataFrame,
    colonnes_reference: list[str],
    alias_colonnes: Optional[dict[str, str]] = None,
    valeur_defaut: Any = None,
    garder_colonnes_en_trop: bool = True,
    ajouter_colonnes_absentes: bool = True,
    supprimer_colonnes_vides: bool = False,
    supprimer_lignes_vides: bool = False,
    trier_autres: bool = False,
    supprimer_alias_fusionnes: bool = True,
    config_fusion_colonnes: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Harmonise un DataFrame selon un schema de reference.

    La fonction peut :
    - pre-fusionner des colonnes similaires ou groupees ;
    - fusionner des colonnes alias vers une colonne cible de reference ;
    - ajouter les colonnes manquantes ;
    - conserver ou supprimer les colonnes en trop ;
    - reclasser les colonnes finales.

    Exemple d'usage pour des donnees API :
        alias = {
            "Province_de_notification": "Province_notification",
            "Zone_de_notification": "Zone_de_sante_notification",
            "Aire_de_sante_de_notification": "Aire_de_sante_notification",
            "Classification_finale": "Classification_investigation",
        }
        df = harmoniser_schema_colonnes(df, colonne_ebola, alias_colonnes=alias)

    Exemple avec pre-fusion des colonnes proches :
        config_fusion = {
            "method": "similarity",
            "type_fusion": "first_non_null",
            "seuil_similarite": 1,
            "drop": True,
        }
        df = harmoniser_schema_colonnes(
            df,
            colonne_ebola,
            config_fusion_colonnes=config_fusion,
        )
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit etre un DataFrame pandas.")

    if not colonnes_reference:
        return df.copy()

    df_resultat = df.copy()

    if config_fusion_colonnes:
        # Import local pour eviter une dependance circulaire :
        # colonne_fusion importe deja verifier_colonnes depuis ce module.
        from dataminsante.colonne_valeur.colonne_fusion import (
            fusionner_colonnes_similaires_ou_groupes,
        )

        config_fusion_colonnes = dict(config_fusion_colonnes)
        df_resultat = fusionner_colonnes_similaires_ou_groupes(
            df_resultat,
            **config_fusion_colonnes,
        )
        logger.info(
            "Pre-fusion des colonnes appliquee avant harmonisation avec config : %s",
            config_fusion_colonnes,
        )

    def _fusionner_colonnes_dupliquees_exactes(df_entree: pd.DataFrame) -> pd.DataFrame:
        """
        Fusionne les colonnes ayant exactement le meme nom en conservant,
        ligne par ligne, la premiere valeur non nulle de gauche a droite.
        """
        if not df_entree.columns.duplicated().any():
            return df_entree

        colonnes_uniques = list(dict.fromkeys(df_entree.columns.tolist()))
        series_fusionnees = []
        resume_doublons = {}

        for col in colonnes_uniques:
            selection = df_entree.loc[:, df_entree.columns == col]
            if selection.shape[1] == 1:
                serie = selection.iloc[:, 0].copy()
            else:
                serie = selection.iloc[:, 0].copy()
                for idx in range(1, selection.shape[1]):
                    serie = serie.combine_first(selection.iloc[:, idx])
                resume_doublons[str(col)] = int(selection.shape[1])

            serie.name = col
            series_fusionnees.append(serie)

        logger.warning(
            "Colonnes dupliquees fusionnees automatiquement avant harmonisation : %s",
            resume_doublons,
        )
        return pd.concat(series_fusionnees, axis=1)

    df_resultat = _fusionner_colonnes_dupliquees_exactes(df_resultat)

    # Dedoublonner la reference tout en conservant l'ordre.
    colonnes_reference_uniques = list(dict.fromkeys(colonnes_reference))
    reference_par_nom_standard = {
        standardiser_nom(col): col
        for col in colonnes_reference_uniques
        if isinstance(col, str) and str(col).strip()
    }

    def _resoudre_nom(nom_colonne: Any) -> str:
        if nom_colonne is None:
            return ""
        nom_colonne = str(nom_colonne).strip()
        if not nom_colonne:
            return ""
        if nom_colonne in df_resultat.columns:
            return nom_colonne

        nom_standard = standardiser_nom(nom_colonne)
        if nom_standard in reference_par_nom_standard:
            return reference_par_nom_standard[nom_standard]

        for col_existante in df_resultat.columns:
            if standardiser_nom(col_existante) == nom_standard:
                return col_existante

        return nom_standard or nom_colonne

    def _colonne_est_entierement_vide(nom_colonne: Any) -> bool:
        """
        Retourne True si la colonne ne contient que des valeurs manquantes.

        Avec des noms de colonnes dupliques, ``df_resultat[nom_colonne]`` peut
        renvoyer un DataFrame plutot qu'une Series. On reduit alors le test sur
        l'ensemble des colonnes correspondantes.
        """
        selection = df_resultat[nom_colonne]
        if isinstance(selection, pd.DataFrame):
            return bool(selection.isna().all().all())
        return bool(selection.isna().all())

    alias_colonnes = alias_colonnes or {}
    alias_resolus = {}
    for source, cible in alias_colonnes.items():
        source_resolu = _resoudre_nom(source)
        cible_resolue = _resoudre_nom(cible)
        if source_resolu and cible_resolue:
            alias_resolus[source_resolu] = cible_resolue

    if supprimer_colonnes_vides:
        colonnes_a_supprimer = []

        for col in df_resultat.columns:
            colonne_est_vide = _colonne_est_entierement_vide(col)
            est_unnamed_vide = str(col).startswith("Unnamed") and colonne_est_vide
            est_vide_hors_reference = (
                colonne_est_vide
                and col not in colonnes_reference_uniques
                and col not in alias_resolus
            )
            if est_unnamed_vide or est_vide_hors_reference:
                colonnes_a_supprimer.append(col)

        if colonnes_a_supprimer:
            df_resultat.drop(columns=colonnes_a_supprimer, inplace=True)
            logger.info("Colonnes vides supprimees avant harmonisation : %s", sorted(colonnes_a_supprimer))

    colonnes_fusionnees = []
    for source, cible in alias_resolus.items():
        if source == cible or source not in df_resultat.columns:
            continue

        if cible not in df_resultat.columns:
            df_resultat[cible] = valeur_defaut

        masque_source = df_resultat[source].notna()
        masque_cible_vide = df_resultat[cible].isna()
        n_completees = int((masque_source & masque_cible_vide).sum())

        if n_completees > 0:
            logger.info(
                "Fusion colonne alias '%s' -> '%s' : %s valeur(s) completee(s).",
                source,
                cible,
                n_completees,
            )

        df_resultat[cible] = df_resultat[cible].combine_first(df_resultat[source])
        colonnes_fusionnees.append((source, cible))

    if supprimer_alias_fusionnes:
        colonnes_a_supprimer = [
            source for source, cible in colonnes_fusionnees
            if source in df_resultat.columns and source != cible
        ]
        if colonnes_a_supprimer:
            df_resultat.drop(columns=colonnes_a_supprimer, inplace=True)
            logger.info("Colonnes alias supprimees apres fusion : %s", sorted(colonnes_a_supprimer))

    colonnes_manquantes = []
    if ajouter_colonnes_absentes:
        for col in colonnes_reference_uniques:
            if col not in df_resultat.columns:
                df_resultat[col] = valeur_defaut
                colonnes_manquantes.append(col)

        if colonnes_manquantes:
            logger.info("Colonnes ajoutees pour respecter le schema : %s", sorted(colonnes_manquantes))

    if not garder_colonnes_en_trop:
        colonnes_supprimees_hors_schema = [
            col for col in df_resultat.columns
            if col not in colonnes_reference_uniques
        ]
        if colonnes_supprimees_hors_schema:
            logger.info(
                "Colonnes supprimees hors schema de reference : %s",
                sorted(colonnes_supprimees_hors_schema),
            )
        colonnes_conservees = [col for col in colonnes_reference_uniques if col in df_resultat.columns]
        df_resultat = df_resultat[colonnes_conservees]

    colonnes_vues = set()
    colonnes_dupliquees = set()
    colonnes_absentes = []
    colonnes_prioritaires_uniques = []

    for col in colonnes_reference_uniques:
        if col not in df_resultat.columns:
            colonnes_absentes.append(col)
            continue
        if col in colonnes_vues:
            colonnes_dupliquees.add(col)
            continue
        colonnes_vues.add(col)
        colonnes_prioritaires_uniques.append(col)

    if colonnes_dupliquees:
        logger.info("Colonnes dupliquees ignorees : %s", sorted(colonnes_dupliquees))

    if colonnes_absentes and not ajouter_colonnes_absentes:
        logger.warning("Colonnes absentes ignorees : %s", sorted(colonnes_absentes))

    autres_colonnes = [col for col in df_resultat.columns if col not in colonnes_prioritaires_uniques]
    if trier_autres:
        autres_colonnes = sorted(autres_colonnes)

    df_resultat = df_resultat[colonnes_prioritaires_uniques + autres_colonnes]

    if supprimer_lignes_vides:
        lignes_avant = df_resultat.shape[0]
        df_resultat = df_resultat.dropna(how="all")
        lignes_supprimees = lignes_avant - df_resultat.shape[0]
        if lignes_supprimees > 0:
            logger.info("%s ligne(s) entierement vide(s) supprimee(s).", lignes_supprimees)

    return df_resultat


def verifier_alias_colonnes(
    df: pd.DataFrame,
    alias_colonnes: dict[str, str],
    colonnes_reference: Optional[list[str]] = None,
    afficher: bool = True,
) -> dict[str, Any]:
    """
    Verifie quels alias de colonnes sont effectivement exploitables
    dans un DataFrame avant harmonisation.

    La fonction identifie :
    - les alias dont la colonne source est presente ;
    - les alias dont la colonne source est absente ;
    - les cibles qui risquent de rester vides ;
    - les cibles qui ne figurent pas dans un schema de reference optionnel.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit etre un DataFrame pandas.")

    if alias_colonnes is None:
        alias_colonnes = {}

    if not isinstance(alias_colonnes, dict):
        raise TypeError("alias_colonnes doit etre un dictionnaire {source: cible}.")

    colonnes_df = set(df.columns)

    alias_sources_presentes = {
        source: cible
        for source, cible in alias_colonnes.items()
        if source in colonnes_df
    }

    alias_sources_absentes = {
        source: cible
        for source, cible in alias_colonnes.items()
        if source not in colonnes_df
    }

    cibles_probablement_non_alimentees = [
        cible
        for source, cible in alias_sources_absentes.items()
        if cible not in colonnes_df
    ]

    variables_cibles_invalides = []
    if colonnes_reference is not None:
        colonnes_reference_uniques = set(dict.fromkeys(colonnes_reference))
        variables_cibles_invalides = [
            cible
            for cible in alias_colonnes.values()
            if cible not in colonnes_reference_uniques
        ]

    rapport = {
        "alias_sources_presentes": alias_sources_presentes,
        "alias_sources_absentes": alias_sources_absentes,
        "cibles_probablement_non_alimentees": cibles_probablement_non_alimentees,
        "variables_cibles_invalides": variables_cibles_invalides,
    }

    if afficher:
        print("Alias effectivement utilisables :")
        for source, cible in alias_sources_presentes.items():
            print(f"  - {source} -> {cible}")

        if alias_sources_absentes:
            print("\nAlias non trouves dans les colonnes du DataFrame :")
            for source, cible in alias_sources_absentes.items():
                print(f"  - {source} -> {cible}")

        if cibles_probablement_non_alimentees:
            print(
                "\nAttention, ces colonnes cibles risquent de rester vides : "
                f"{cibles_probablement_non_alimentees}"
            )

        if variables_cibles_invalides:
            print(
                "\nAttention, certaines cibles d'alias sont absentes du schema de reference : "
                f"{variables_cibles_invalides}"
            )

    logger.info(
        "Verification alias colonnes : %s present(s), %s absent(s), %s cible(s) potentiellement non alimentee(s).",
        len(alias_sources_presentes),
        len(alias_sources_absentes),
        len(cibles_probablement_non_alimentees),
    )

    return rapport

# ------------------------------------------------
# Fonction principale à utiliser dans un pipeline
# ------------------------------------------------
def clean_all_column_names(
    df: pd.DataFrame,
    mapping_file: Optional[Union[str, Path]] = mapping_file_path,
) -> pd.DataFrame:
    """
    Point d'entree de pipeline pour nettoyer toutes les colonnes d'un DataFrame.

    Cette fonction reapplique `standardiser_noms_colonnes(...)` avec un fichier
    de mapping par defaut, tout en laissant la possibilite de fournir un autre
    fichier via `mapping_file`.

    Usage typique :
    - `clean_all_column_names(df)` : mapping par defaut du package ;
    - `clean_all_column_names(df, mapping_file=autre_mapping)` :
      meme pipeline avec un mapping personnalise.
    """
    return standardiser_noms_colonnes(df, mapping_file=mapping_file)
