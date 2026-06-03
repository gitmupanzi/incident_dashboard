"""Render the methodology and interpretation tab."""

from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


def _series_has_documented_values(series: pd.Series) -> bool:
    """Retourne True seulement si une colonne contient au moins une valeur réellement documentée."""
    if series is None:
        return False
    if isinstance(series, pd.DataFrame):
        return any(_series_has_documented_values(series[col]) for col in series.columns)
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return bool(series.notna().any())
    normalized = series.astype("string").str.strip()
    return bool((normalized.notna() & normalized.ne("")).any())


def _build_conservative_audit_scope(
    df_current: pd.DataFrame,
    df_source: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construit un périmètre d'audit prudent en réinjectant les colonnes documentées du frame source."""
    if not isinstance(df_current, pd.DataFrame):
        return pd.DataFrame()

    audit_scope = df_current.copy()
    if not isinstance(df_source, pd.DataFrame) or df_source.empty:
        return audit_scope

    source_scope = df_source.copy()
    if not source_scope.index.equals(audit_scope.index):
        source_scope = source_scope.reindex(audit_scope.index)

    for col in source_scope.columns:
        if col not in audit_scope.columns:
            audit_scope[col] = source_scope[col]
            continue
        try:
            current_has_data = _series_has_documented_values(audit_scope[col])
            source_has_data = _series_has_documented_values(source_scope[col])
        except Exception:
            continue
        if (not current_has_data) and source_has_data:
            audit_scope[col] = source_scope[col]

    return audit_scope


def _clear_builder_cache(builder) -> None:
    """Efface prudemment le cache Streamlit d'un builder quand il en expose un."""
    clear_fn = getattr(builder, "clear", None)
    if callable(clear_fn):
        try:
            clear_fn()
        except Exception:
            pass


def render_methodology_tab(ctx: dict) -> None:
    """Render the methodology and interpretation tab."""
    globals().update(ctx)
    render_section_title(2, "Méthodologie, définitions et limites d'interprétation")
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

    methodology_scope = _build_conservative_audit_scope(
        df_f if isinstance(df_f, pd.DataFrame) else pd.DataFrame(),
        ctx.get("df_f_source"),
    )

    st.markdown("### 0. Audit standard du fichier et analyses activables")
    structure_audit_builder = globals().get("build_standard_file_structure_audit")
    capability_matrix_builder = globals().get("build_standard_analysis_capability_matrix")
    if callable(structure_audit_builder) and callable(capability_matrix_builder) and isinstance(methodology_scope, pd.DataFrame):
        _clear_builder_cache(structure_audit_builder)
        _clear_builder_cache(capability_matrix_builder)
        structure_audit = structure_audit_builder(
            methodology_scope,
            source_name=current_disease_label,
            sheet_name="Perimetre filtre",
        )
        capability_matrix = capability_matrix_builder(methodology_scope)

        if not structure_audit.empty:
            audit_row = structure_audit.iloc[0]
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Lignes", f"{int(audit_row['Nombre_lignes']):,}".replace(",", " "))
            a2.metric("Colonnes", f"{int(audit_row['Nombre_colonnes'])}")
            a3.metric("Colonnes vides", f"{int(audit_row['Nb_colonnes_vides'])}")
            a4.metric(
                "Couverture champs standards",
                f"{float(audit_row['Taux_couverture_champs_standards_%']):.1f}%",
            )
            st.dataframe(structure_audit, width="stretch", hide_index=True)

        if not capability_matrix.empty:
            st.caption(
                "Cette matrice indique quelles briques analytiques standard sont pleinement disponibles, "
                "partielles ou indisponibles sur le perimetre actuellement filtre."
            )
            st.dataframe(capability_matrix, width="stretch", hide_index=True, height=380)

    classification_audit_builder = globals().get("build_standard_classification_audit")
    if callable(classification_audit_builder) and isinstance(methodology_scope, pd.DataFrame):
        _clear_builder_cache(classification_audit_builder)
        classification_audit = classification_audit_builder(methodology_scope)
        if not classification_audit.empty:
            st.markdown("### 0.b Classification standard active")
            st.caption(
                "Ce tableau separe, quand la source le permet, la lecture de qualification des alertes "
                "de la classification d'investigation et de la synthese finale / resultat."
            )
            st.dataframe(classification_audit, width="stretch", hide_index=True, height=260)

    care_issue_audit_builder = globals().get("build_standard_care_issue_audit")
    if callable(care_issue_audit_builder) and isinstance(methodology_scope, pd.DataFrame):
        _clear_builder_cache(care_issue_audit_builder)
        care_issue_audit = care_issue_audit_builder(methodology_scope)
        if not care_issue_audit.empty:
            st.markdown("### 0.c Prise en charge / issue standard")
            st.caption(
                "Ce resume transverse consolide admission, isolement, issue, deces et duree de sejour "
                "quand les variables correspondantes sont documentees."
            )
            st.dataframe(care_issue_audit, width="stretch", hide_index=True)

    symptom_audit_builder = globals().get("build_standard_symptom_audit")
    if callable(symptom_audit_builder) and isinstance(methodology_scope, pd.DataFrame):
        _clear_builder_cache(symptom_audit_builder)
        symptom_audit = symptom_audit_builder(methodology_scope)
        if not symptom_audit.empty:
            st.markdown("### 0.d Bloc clinique / symptomes")
            st.caption(
                "Ce bloc reste optionnel : il ne s'affiche que lorsque la source documente des symptomes exploitables."
            )
            st.dataframe(symptom_audit, width="stretch", hide_index=True)

    st.markdown("### 1. Principes généraux de lecture")
    st.markdown(
        """
        - Les résultats décrivent les **données disponibles dans le périmètre filtré** ; ils ne représentent pas nécessairement toute la population exposée.
        - Une augmentation de cas, une alerte ou un score élevé est un **signal de vérification**, pas une confirmation automatique d'épidémie.
        - Les indicateurs doivent être lus avec la **complétude**, la **promptitude**, les filtres actifs et la connaissance terrain.
        - Les résultats peuvent évoluer après rattrapage, correction ou consolidation des données.
        """
    )

    st.markdown("### 2. Chaîne analytique standard COUSP")
    st.caption(
        "Le dashboard suit une logique standard multi-maladies : `Alerte -> Notification -> Investigation -> Exposition -> "
        "Prélèvement -> Laboratoire -> Prise en charge / Issue`. Les blocs ne s'affichent pleinement que si les colonnes "
        "nécessaires existent dans la source chargée."
    )
    chain_method = pd.DataFrame(
        [
            ("Alerte", "N_alerte, Source_alerte, Localite si disponibles", "Identifier les alertes et leur origine dans la chaîne de surveillance."),
            ("Notification", "Lignes filtrées / N_epid si disponible", "Décrire les cas effectivement notifiés dans le périmètre actif."),
            ("Investigation", "Investigation, Date_investigation, Classification_investigation", "Vérifier que les cas ont fait l'objet d'une investigation documentée ou fortement inférée par une classification d'investigation exploitable."),
            ("Exposition", "Lien épidémiologique, cas source, facteur d'exposition", "Décrire les liens de transmission et les expositions documentées."),
            ("Prélèvement", "Prelevement, Date_prelevement, Type_de_prelevement", "Suivre la couverture de prélèvement parmi les cas éligibles."),
            ("Laboratoire", "Date_reception_labo, Resultat_labo / TDR_Resultat, Date_confirmation", "Suivre l'acheminement, l'analyse et l'interprétation des résultats."),
            ("Prise en charge / Issue", "Hospitalisation, Date_admission_au_CT, Issue, Date_issue", "Décrire l'accès aux soins, les décès et les issues documentées."),
        ],
        columns=["Bloc standard", "Variables cibles", "Lecture métier"],
    )
    st.dataframe(chain_method, width="stretch", hide_index=True, height=300)

    current_chain = build_standard_surveillance_chain_table(methodology_scope) if isinstance(methodology_scope, pd.DataFrame) else pd.DataFrame()
    if not current_chain.empty:
        with st.expander("Aperçu de la chaîne standard sur le périmètre filtré", expanded=False):
            st.caption(
                "Ce tableau reprend les indicateurs standard effectivement calculables avec la source actuellement filtrée."
            )
            st.dataframe(current_chain, width="stretch", hide_index=True, height=360)

    indicator_defs = pd.DataFrame(
        [
            ("Alertes documentées", "Nombre d'identifiants d'alerte non vides", "Toutes les lignes filtrées", "Disponible seulement si `N_alerte` existe."),
            ("Cas / notifications", "Nombre de lignes/cas dans la liste filtrée", "Toutes les lignes du périmètre filtré", "Dépend de la déduplication, de la définition de cas et des filtres."),
            ("Cas investigués", "Investigation=Oui ou investigation documentée par date/classification", "Alertes documentées si disponibles, sinon cas filtrés", "Par convention standard, une classification exploitable est traitée comme une forte preuve d'investigation, même si la colonne `Investigation` manque."),
            ("Cas suspects / probables / confirmés", "Effectif par classification standardisée", "Cas investigués si possible, sinon cas filtrés", "La qualité de `Classification_investigation` influence prioritairement ces indicateurs ; `Classification_finale` est lue séparément comme synthèse finale si elle existe."),
            ("Décès", "Cas dont l'issue est interprétée comme décès", "Tous les cas filtrés", "La qualité du champ Issue influence fortement l'indicateur."),
            ("Létalité / CFR (%)", "Décès / Cas × 100", "Tous les cas filtrés", "À interpréter avec prudence si le nombre de cas est faible."),
            ("Prélèvement (%)", "Cas avec prélèvement documenté Oui / Cas × 100", "Cas suspects si possible, sinon cas filtrés", "Peut refléter la pratique terrain ou la complétude du champ."),
            ("Réception labo documentée (%)", "Cas avec Date_reception_labo / Cas × 100", "Cas prélevés si possible, sinon cas filtrés", "Indique l'acheminement documenté des échantillons."),
            ("TDR ou test réalisé (%)", "Cas avec test documenté Oui / Cas × 100", "Tous les cas filtrés", "Selon la maladie, le test attendu peut différer."),
            ("Résultats labo disponibles (%)", "Résultats documentés / Cas × 100", "Réceptions labo si possible, sinon cas prélevés ou cas filtrés", "Exclut l'absence de résultat et dépend de la chaîne laboratoire."),
            ("Positivité (%)", "Résultats positifs / Résultats valides positifs ou négatifs × 100", "Résultats valides", "Exclut les résultats invalides, en attente ou non interprétables."),
            ("Résultats invalides (%)", "Résultats invalides / Tests ou résultats documentés × 100", "Tests ou résultats documentés", "Sert à repérer des difficultés analytiques ou de saisie."),
            ("Vivants documentés", "Issues standardisées contenant un statut vivant documenté, y compris les sorties avec patient vivant", "Tous les cas filtrés", "Dépend de la qualité du champ Issue."),
            ("Promptitude ≤ seuil", "Cas dont le délai est inférieur ou égal au seuil choisi", "Cas avec dates valides et délai non négatif", "Un délai manquant n'est pas classé comme rapide ou lent."),
            ("Complétude (%)", "Proportion moyenne de champs clés renseignés", "Champs standards disponibles", "Mesure la documentation, pas la qualité clinique intrinsèque."),
            ("Alerte hebdomadaire", "Cas récents comparés à une moyenne historique courte", "Groupe géographique et semaine", "Signal à investiguer, sensible au faible historique."),
            ("Score de risque", "Score composite 0-100 combinant volume, tendance, CFR, qualité et promptitude", "Groupe géographique sélectionné", "Priorise l'attention ; ne remplace pas l'analyse experte."),
            ("IREP", "Score composite 0-100 combinant tendance, incidence, létalité, promptitude et complétude", "Province ou zone de santé selon l'onglet IREP", "Dépend des poids choisis, du dénominateur population et de la disponibilité des composantes."),
        ],
        columns=["Indicateur", "Règle de calcul", "Dénominateur", "Limite principale"],
    )
    st.markdown("### 3. Définitions des indicateurs")
    st.dataframe(indicator_defs, width="stretch", hide_index=True, height=420)

    st.markdown("### 4. Dénominateurs standard à retenir")
    denominator_rules = pd.DataFrame(
        [
            ("Cas investigués", "Alertes documentées si `N_alerte` existe ; sinon cas filtrés", "Rester standard même quand la source ne sépare pas explicitement alerte et cas ; une classification exploitable compte comme preuve forte d'investigation."),
            ("Cas suspects / probables / confirmés", "Cas investigués si disponibles ; sinon cas filtrés", "Éviter d'imposer un dénominateur absent dans d'autres maladies."),
            ("Cas prélevés", "Cas suspects si la classification existe ; sinon cas filtrés", "Le plan standard COUSP privilégie les suspects quand l'information existe."),
            ("Réception labo documentée", "Cas prélevés", "Mesure la continuité de la chaîne d'acheminement."),
            ("Résultats labo disponibles", "Réceptions labo si disponibles ; sinon cas prélevés", "Évite de surévaluer la performance labo quand la réception manque."),
            ("Cas positifs / négatifs / invalides", "Résultats documentés ou valides selon l'indicateur", "Sépare la performance analytique de la couverture de prélèvement."),
            ("Décès, vivants, hospitalisés", "Cas filtrés", "Lecture simple et stable multi-maladies."),
            ("Promptitude", "Cas avec dates valides et délai non négatif", "Les cas sans dates comparables sont exclus du calcul."),
        ],
        columns=["Indicateur", "Dénominateur standard", "Pourquoi"],
    )
    st.dataframe(denominator_rules, width="stretch", hide_index=True, height=320)

    st.markdown("### 5. Délais standards de promptitude")
    delay_defs = pd.DataFrame(
        [
            ("Début -> notification", f"{DATE_NOTIF} - {DATE_ONSET}", "Mesure la rapidité de notification initiale."),
            ("Notification -> investigation", f"{DATE_INV} - {DATE_NOTIF}", "Mesure la réactivité de l'investigation après notification."),
            ("Notification -> prélèvement", f"{DATE_PREL} - {DATE_NOTIF}", "Mesure le passage de la notification au prélèvement."),
            ("Prélèvement -> réception labo", f"{DATE_RECEP} - {DATE_PREL}", "Mesure l'acheminement vers le laboratoire."),
            ("Réception labo -> résultat", f"{DATE_RES} - {DATE_RECEP}", "Mesure la durée analytique et de restitution du résultat."),
            ("Notification -> admission", f"{DATE_ADM} - {DATE_NOTIF}", "Mesure l'accès documenté à la prise en charge."),
            ("Admission -> issue", f"{DATE_ISSUE} - {DATE_ADM}", "Mesure la durée jusqu'à l'issue documentée."),
            ("Début -> admission", f"{DATE_ADM} - {DATE_ONSET}", "Mesure le délai global avant admission."),
            ("Début -> prélèvement", f"{DATE_PREL} - {DATE_ONSET}", "Mesure le délai global avant prélèvement."),
            ("Prélèvement -> résultat", f"{DATE_RES} - {DATE_PREL}", "Lecture synthétique de la chaîne laboratoire quand la réception manque."),
        ],
        columns=["Délai standard", "Formule source", "Interprétation"],
    )
    st.dataframe(delay_defs, width="stretch", hide_index=True, height=360)
    available_delays = list_available_standard_delays(methodology_scope) if isinstance(methodology_scope, pd.DataFrame) else []
    if available_delays:
        available_delay_tbl = pd.DataFrame(available_delays, columns=["Code délai", "Libellé disponible"])
        with st.expander("Délais actuellement disponibles dans le périmètre filtré", expanded=False):
            st.dataframe(available_delay_tbl, width="stretch", hide_index=True)

    st.markdown("### 6. IREP : formules et logique de calcul")
    st.caption(
        "L'IREP aide à hiérarchiser les territoires en combinant volume, risque relatif et qualité de l'information. "
        "Il sert à prioriser l'attention, pas à conclure seul à une flambée."
    )

    irep_components = pd.DataFrame(
        [
            ("Cas", "Somme des cas sur la fenêtre analysée", "Fenetre hebdomadaire, fenetre recente parametrable ou cumulee"),
            ("Tendance", "Cas fenêtre courante / (Cas fenêtre précédente + 1)", "Compare la dynamique récente au niveau géographique analysé"),
            ("Taux d'attaque (%)", "(Cas / Population exposée) × 100", "Lecture proportionnelle du poids de l'événement dans la population"),
            ("Incidence", "(Cas / Population exposée) × multiplicateur", "Multiplicateur paramétrable : 1 000, 10 000 ou 100 000"),
            ("Létalité (%)", "(Décès / Cas) × 100", "Mesure la gravité apparente parmi les cas rapportés"),
            ("Promptitude (%)", "Cas notifiés dans le délai seuil / Cas avec délai valide × 100", "Mesure la rapidité de notification"),
            ("Risque de promptitude", "100 - Promptitude (%)", "Plus la promptitude baisse, plus le risque augmente"),
            ("Complétude (%)", "Moyenne des champs critiques renseignés × 100", "Par défaut : Province_notification et Zone_de_sante_notification"),
            ("Risque de complétude", "100 - Complétude (%)", "Une faible complétude augmente le risque d'interprétation"),
            ("Scoring composante", "Transformation robuste en score 0-100", "Les composantes sont remises sur une échelle comparable"),
        ],
        columns=["Composante", "Formule / règle", "Interprétation opérationnelle"],
    )
    st.dataframe(irep_components, width="stretch", hide_index=True, height=380)

    st.markdown(
        """
```text
IREP = Somme des composantes disponibles après redistribution des poids
```

```text
IREP = w_tendance × Score_tendance
     + w_incidence × Score_incidence
     + w_létalité × Score_létalité
     + w_promptitude × Score_promptitude
     + w_complétude × Score_complétude
```
"""
    )
    st.caption(
        "Si une composante est indisponible, son poids est redistribué entre les composantes restantes afin d'éviter "
        "de pénaliser artificiellement le territoire."
    )

    irep_rules = pd.DataFrame(
        [
            ("Fenêtres de lecture", "Situation hebdomadaire, fenetre recente parametrable et situation cumulee."),
            ("Population par défaut", "Le référentiel `data/RDC_Zone_de_sante_OCHA.xlsx` est utilisé s'il n'y a pas de téléversement."),
            ("Dénominateur population", "Somme des populations uniques par ZS ou population maximale du groupe selon le paramétrage."),
            ("Lecture recommandée", "Toujours confronter l'IREP aux cas bruts, à l'incidence, à la complétude, aux zones silencieuses et à la promptitude."),
            ("Interprétation", "Un territoire peut avoir moins de cas mais rester plus prioritaire si son incidence ou ses risques de qualité sont plus élevés."),
        ],
        columns=["Règle", "Application dans le dashboard"],
    )
    st.dataframe(irep_rules, width="stretch", hide_index=True)

    st.markdown("### 7. Règles de standardisation")
    standardization_rules = pd.DataFrame(
        [
            ("Alerte / identification", "Les variantes proches de `N_alerte`, `N_epid`, `Localite` et `Source_alerte` sont conservées lorsqu'elles existent pour alimenter la chaîne standard."),
            ("Géographie", "Province, Zone de santé et Aire de santé sont harmonisées vers les colonnes standards du dashboard."),
            ("Temps", "Les semaines épidémiologiques sont construites à partir des colonnes année/semaine ou des dates disponibles."),
            ("Dates", "Les dates ISO sont lues en year-first ; les autres formats sont interprétés avec prudence en day-first."),
            ("Âge", "L'âge est converti en années lorsque l'unité est disponible : jours, semaines, mois ou ans."),
            ("Sexe", "Les variantes usuelles sont harmonisées vers Masculin/Feminin lorsque possible."),
            ("Investigation", "Une investigation est considérée comme documentée si l'un des éléments suivants existe : `Investigation=Oui`, `Date_investigation`, ou `Classification_investigation`. `Classification_finale` n'est plus utilisée comme substitut automatique de l'investigation."),
            ("Issue", "Les libellés compatibles avec décès alimentent l'indicateur `is_death` et les statuts vivants, y compris les sorties avec patient vivant, sont standardisés dans `Issue_std`."),
            ("Laboratoire", "Les résultats sont classés en positifs, négatifs, invalides ou non interprétables selon les valeurs disponibles ; `Resultat_labo` et `TDR_Resultat` sont rapprochés."),
            ("Promptitude", "Les délais standard sont calculés seulement quand les deux dates sources existent ; les délais négatifs sont conservés pour la qualité mais exclus des lectures opérationnelles."),
        ],
        columns=["Bloc", "Règle appliquée"],
    )
    st.dataframe(standardization_rules, width="stretch", hide_index=True)

    field_matrix = build_recommended_fields_matrix(methodology_scope) if isinstance(methodology_scope, pd.DataFrame) else pd.DataFrame()
    if not field_matrix.empty:
        with st.expander("Disponibilité des variables standards dans la source filtrée", expanded=False):
            st.caption(
                "Cette matrice montre quelles variables standards sont présentes et à quel niveau de complétude dans les données actuelles."
            )
            st.dataframe(field_matrix, width="stretch", hide_index=True, height=420)

    st.markdown("### 8. Contrôles qualité appliqués")
    qc_rules = pd.DataFrame(
        [
            ("Chronologie", "Dates incohérentes : notification avant début, résultat avant prélèvement, issue avant admission, etc."),
            ("Âge", "Âges négatifs ou supérieurs aux limites plausibles."),
            ("Géographie", "Zone de santé renseignée sans province, ou aire de santé renseignée sans zone."),
            ("Investigation", "Cas sans investigation documentée selon la règle standard (statut, date ou classification) ; cas classés suspects/probables sans prélèvement."),
            ("Laboratoire", "Résultat renseigné alors que le prélèvement ou le test n'est pas documenté comme réalisé ; prélèvement sans réception ; réception sans résultat ; positif sans date de confirmation."),
            ("Issue", "Décès sans date d'issue lorsque la date est attendue."),
            ("Doublons", "Empreintes probables construites à partir de l'identité, de l'âge, des dates et de la géographie lorsque disponibles."),
        ],
        columns=["Contrôle", "Interprétation"],
    )
    st.dataframe(qc_rules, width="stretch", hide_index=True)

    st.markdown("### 9. Limites à mentionner dans les restitutions")
    limitations = pd.DataFrame(
        [
            ("Données incomplètes", "Une variable absente ou peu renseignée peut modifier l'interprétation des tendances, profils et délais."),
            ("Alerte vs cas", "Certaines sources ne distinguent pas explicitement `N_alerte` et `N_epid` ; le dashboard retombe alors sur les cas filtrés comme base standard."),
            ("Retards de notification", "Les semaines les plus récentes peuvent être sous-estimées si les données arrivent tardivement."),
            ("Petits effectifs", "Les pourcentages et CFR peuvent varier fortement avec peu de cas."),
            ("Doublons", "Les doublons potentiels doivent être revus avant toute conclusion définitive sur les volumes."),
            ("Changements de définition", "Toute modification de définition de cas ou de stratégie de dépistage peut modifier les tendances."),
            ("Cartographie", "Les résultats cartographiques dépendent de la qualité des libellés géographiques et des fichiers GeoJSON."),
            ("Alertes automatiques", "Les alertes statistiques ne remplacent pas l'investigation, la vérification terrain et la validation épidémiologique."),
            ("IREP", "Le score dépend des poids, de la population de référence et de la disponibilité des composantes ; il doit être lu comme un outil de priorisation."),
        ],
        columns=["Limite", "Conséquence pratique"],
    )
    st.dataframe(limitations, width="stretch", hide_index=True)

    st.markdown("### 10. Confidentialité et diffusion")
    st.markdown(
        """
        - Les listes linéaires peuvent contenir des informations nominatives ou indirectement identifiantes.
        - Les exports destinés à un partage externe doivent être limités au strict nécessaire et, si possible, anonymisés.
        - Les résultats diffusés publiquement devraient privilégier les agrégats par semaine, province ou zone de santé.
        - Toute diffusion institutionnelle doit préciser la source, la période, les filtres appliqués et la date de génération.
        """
    )

    st.markdown("### 11. Checklist avant validation institutionnelle")
    checklist = pd.DataFrame(
        [
            ("Source vérifiée", "Le fichier chargé correspond à la maladie, à la période et au niveau attendu."),
            ("Filtres vérifiés", "Les filtres géographiques, temporels et de classification sont cohérents avec la question analysée."),
            ("Dénominateurs validés", "Le lecteur comprend bien si la base utilisée est cas filtrés, alertes documentées, cas suspects, cas prélevés ou résultats valides."),
            ("Complétude revue", "Les champs clés sont suffisamment renseignés pour soutenir l'interprétation."),
            ("Doublons revus", "Les doublons potentiels majeurs ont été vérifiés ou documentés."),
            ("Délais interprétés avec prudence", "Les délais sont lus uniquement si les dates sources sont fiables."),
            ("Chaîne standard revue", "Les ruptures entre investigation, prélèvement, réception labo, résultat et issue ont été vérifiées."),
            ("Alertes validées", "Les signaux automatiques ont été confrontés à l'information terrain."),
            ("Export traçable", "L'export inclut la source, la période, les filtres et la date de génération."),
        ],
        columns=["Point de contrôle", "Critère attendu"],
    )
    st.dataframe(checklist, width="stretch", hide_index=True)


