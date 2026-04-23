"""Render the methodology and interpretation tab."""

from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


def render_methodology_tab(ctx: dict) -> None:
    """Render the methodology and interpretation tab."""
    globals().update(ctx)
    render_section_title(15, "Méthodologie, définitions et limites d'interprétation")
    render_reader_narrative(
        "Pourquoi cet onglet existe",
        "Cette page documente les règles de lecture du tableau de bord. Elle permet de rendre les résultats plus transparents "
        "pour le COUSP, les programmes, les DPS, les partenaires techniques et les lecteurs non spécialistes.",
        tone="decision",
    )

    current_disease_label = DISEASE_SPECS.get(disease_key, {}).get("label", disease_key)
    methodology_context = pd.DataFrame(
        [
            ("Maladie / source sélectionnée", current_disease_label),
            ("Mode analytique", "IDSR agrégé hebdomadaire" if IDSR_MODE else "Liste linéaire individuelle"),
            ("Périmètre actif", f"{len(df_f):,} lignes après filtres".replace(",", " ") if isinstance(df_f, pd.DataFrame) else "Non disponible"),
            ("Lecture recommandée", "Décrire, vérifier, prioriser ; ne pas conclure automatiquement sans validation terrain."),
        ],
        columns=["Élément", "Description"],
    )
    st.dataframe(methodology_context, width="stretch", hide_index=True)

    st.markdown("### 1. Principes généraux de lecture")
    st.markdown(
        """
        - Les résultats décrivent les **données disponibles dans le périmètre filtré** ; ils ne représentent pas nécessairement toute la population exposée.
        - Une augmentation de cas, une alerte ou un score élevé est un **signal de vérification**, pas une confirmation automatique d'épidémie.
        - Les indicateurs doivent être lus avec la **complétude**, la **promptitude**, les filtres actifs et la connaissance terrain.
        - Les résultats peuvent évoluer après rattrapage, correction ou consolidation des données.
        """
    )

    indicator_defs = pd.DataFrame(
        [
            ("Cas", "Nombre de lignes/cas dans la liste filtrée", "Toutes les lignes du périmètre filtré", "Dépend de la déduplication et des filtres."),
            ("Décès", "Cas dont l'issue est interprétée comme décès", "Tous les cas filtrés", "La qualité du champ Issue influence fortement l'indicateur."),
            ("Létalité / CFR (%)", "Décès / Cas × 100", "Tous les cas filtrés", "À interpréter avec prudence si le nombre de cas est faible."),
            ("Prélèvement (%)", "Cas avec prélèvement documenté Oui / Cas × 100", "Tous les cas filtrés", "Peut refléter la pratique terrain ou la complétude du champ."),
            ("TDR ou test réalisé (%)", "Cas avec test documenté Oui / Cas × 100", "Tous les cas filtrés", "Selon la maladie, le test attendu peut différer."),
            ("Positivité (%)", "Résultats positifs / Résultats valides positifs ou négatifs × 100", "Résultats valides", "Exclut les résultats invalides, en attente ou non interprétables."),
            ("Promptitude ≤ seuil", "Cas dont le délai est inférieur ou égal au seuil choisi", "Cas avec dates valides et délai non négatif", "Un délai manquant n'est pas classé comme rapide ou lent."),
            ("Complétude (%)", "Proportion moyenne de champs clés renseignés", "Champs standards disponibles", "Mesure la documentation, pas la qualité clinique intrinsèque."),
            ("Alerte hebdomadaire", "Cas récents comparés à une moyenne historique courte", "Groupe géographique et semaine", "Signal à investiguer, sensible au faible historique."),
            ("Score de risque", "Score composite 0-100 combinant volume, tendance, CFR, qualité et promptitude", "Groupe géographique sélectionné", "Priorise l'attention ; ne remplace pas l'analyse experte."),
            ("IREP", "Indice provincial composite de risque épidémique", "Province", "Dépend des poids choisis et des indicateurs disponibles."),
        ],
        columns=["Indicateur", "Règle de calcul", "Dénominateur", "Limite principale"],
    )
    st.markdown("### 2. Définitions des indicateurs")
    st.dataframe(indicator_defs, width="stretch", hide_index=True, height=420)

    st.markdown("### 3. Règles de standardisation")
    standardization_rules = pd.DataFrame(
        [
            ("Géographie", "Province, Zone de santé et Aire de santé sont harmonisées vers les colonnes standards du dashboard."),
            ("Temps", "Les semaines épidémiologiques sont construites à partir des colonnes année/semaine ou des dates disponibles."),
            ("Dates", "Les dates ISO sont lues en year-first ; les autres formats sont interprétés avec prudence en day-first."),
            ("Âge", "L'âge est converti en années lorsque l'unité est disponible : jours, semaines, mois ou ans."),
            ("Sexe", "Les variantes usuelles sont harmonisées vers Masculin/Feminin lorsque possible."),
            ("Issue", "Les libellés compatibles avec décès alimentent l'indicateur is_death."),
            ("Laboratoire", "Les résultats sont classés en positifs, négatifs, invalides ou non interprétables selon les valeurs disponibles."),
        ],
        columns=["Bloc", "Règle appliquée"],
    )
    st.dataframe(standardization_rules, width="stretch", hide_index=True)

    st.markdown("### 4. Contrôles qualité appliqués")
    qc_rules = pd.DataFrame(
        [
            ("Chronologie", "Dates incohérentes : notification avant début, résultat avant prélèvement, issue avant admission, etc."),
            ("Âge", "Âges négatifs ou supérieurs aux limites plausibles."),
            ("Géographie", "Zone de santé renseignée sans province, ou aire de santé renseignée sans zone."),
            ("Laboratoire", "Résultat renseigné alors que le prélèvement ou le test n'est pas documenté comme réalisé."),
            ("Issue", "Décès sans date d'issue lorsque la date est attendue."),
            ("Doublons", "Empreintes probables construites à partir de l'identité, de l'âge, des dates et de la géographie lorsque disponibles."),
        ],
        columns=["Contrôle", "Interprétation"],
    )
    st.dataframe(qc_rules, width="stretch", hide_index=True)

    st.markdown("### 5. Limites à mentionner dans les restitutions")
    limitations = pd.DataFrame(
        [
            ("Données incomplètes", "Une variable absente ou peu renseignée peut modifier l'interprétation des tendances, profils et délais."),
            ("Retards de notification", "Les semaines les plus récentes peuvent être sous-estimées si les données arrivent tardivement."),
            ("Petits effectifs", "Les pourcentages et CFR peuvent varier fortement avec peu de cas."),
            ("Doublons", "Les doublons potentiels doivent être revus avant toute conclusion définitive sur les volumes."),
            ("Changements de définition", "Toute modification de définition de cas ou de stratégie de dépistage peut modifier les tendances."),
            ("Cartographie", "Les résultats cartographiques dépendent de la qualité des libellés géographiques et des fichiers GeoJSON."),
            ("Alertes automatiques", "Les alertes statistiques ne remplacent pas l'investigation, la vérification terrain et la validation épidémiologique."),
        ],
        columns=["Limite", "Conséquence pratique"],
    )
    st.dataframe(limitations, width="stretch", hide_index=True)

    st.markdown("### 6. Confidentialité et diffusion")
    st.markdown(
        """
        - Les listes linéaires peuvent contenir des informations nominatives ou indirectement identifiantes.
        - Les exports destinés à un partage externe doivent être limités au strict nécessaire et, si possible, anonymisés.
        - Les résultats diffusés publiquement devraient privilégier les agrégats par semaine, province ou zone de santé.
        - Toute diffusion institutionnelle doit préciser la source, la période, les filtres appliqués et la date de génération.
        """
    )

    st.markdown("### 7. Checklist avant validation institutionnelle")
    checklist = pd.DataFrame(
        [
            ("Source vérifiée", "Le fichier chargé correspond à la maladie, à la période et au niveau attendu."),
            ("Filtres vérifiés", "Les filtres géographiques, temporels et de classification sont cohérents avec la question analysée."),
            ("Complétude revue", "Les champs clés sont suffisamment renseignés pour soutenir l'interprétation."),
            ("Doublons revus", "Les doublons potentiels majeurs ont été vérifiés ou documentés."),
            ("Délais interprétés avec prudence", "Les délais sont lus uniquement si les dates sources sont fiables."),
            ("Alertes validées", "Les signaux automatiques ont été confrontés à l'information terrain."),
            ("Export traçable", "L'export inclut la source, la période, les filtres et la date de génération."),
        ],
        columns=["Point de contrôle", "Critère attendu"],
    )
    st.dataframe(checklist, width="stretch", hide_index=True)


