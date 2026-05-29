"""Render the statistical concepts tab."""

import html

from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


def _build_statistical_notions_catalog() -> pd.DataFrame:
    """Catalogue les principales notions statistiques et quasi-statistiques du projet."""
    rows = [
        {
            "Famille": "Mesure de base",
            "Notion": "Effectif / nombre de cas (n)",
            "Comment elle est utilisée ici": "Comptage simple des lignes, des alertes, des décès, des cas positifs ou des unités géographiques touchées.",
            "Où dans le projet": "Vue d'ensemble, Surveillance, COUSP, IREP, IDSR",
            "Pourquoi c'est important": "C'est la base de presque tous les autres indicateurs et de toute comparaison temporelle ou géographique.",
            "Point d'attention": "Le nombre dépend de la déduplication, du périmètre filtré et du niveau d'agrégation.",
        },
        {
            "Famille": "Mesure de base",
            "Notion": "Proportion / pourcentage (%)",
            "Comment elle est utilisée ici": "Partie ÷ tout × 100 pour la couverture, la complétude, la part des décès, la part des résultats positifs, etc.",
            "Où dans le projet": "Vue d'ensemble, Qualité, COUSP, IDSR",
            "Pourquoi c'est important": "Permet de comparer des situations de tailles différentes sur une même échelle.",
            "Point d'attention": "Un petit dénominateur peut faire varier fortement le pourcentage.",
        },
        {
            "Famille": "Mesure épidémiologique",
            "Notion": "Taux de létalité (CFR %)",
            "Comment elle est utilisée ici": "Décès ÷ cas × 100.",
            "Où dans le projet": "Vue d'ensemble, Surveillance, COUSP, IREP, IDSR",
            "Pourquoi c'est important": "Aide à apprécier la gravité apparente de l'événement et à prioriser la prise en charge.",
            "Point d'attention": "Dépend de la qualité de l'issue, des décès documentés et du niveau de confirmation des cas.",
        },
        {
            "Famille": "Mesure épidémiologique",
            "Notion": "Incidence",
            "Comment elle est utilisée ici": "Cas ÷ population × multiplicateur (1 000, 10 000 ou 100 000 selon le module).",
            "Où dans le projet": "IREP, IDSR",
            "Pourquoi c'est important": "Permet de comparer le poids d'un événement entre territoires de tailles différentes.",
            "Point d'attention": "La qualité du dénominateur population est déterminante.",
        },
        {
            "Famille": "Mesure épidémiologique",
            "Notion": "Taux d'attaque",
            "Comment elle est utilisée ici": "Cas ÷ population exposée × 100, surtout dans le flux IDSR.",
            "Où dans le projet": "IDSR",
            "Pourquoi c'est important": "Donne une lecture rapide de l'ampleur relative d'une flambée.",
            "Point d'attention": "Reste sensible au choix de la population exposée et à la qualité du rapportage.",
        },
        {
            "Famille": "Laboratoire",
            "Notion": "Positivité (%)",
            "Comment elle est utilisée ici": "Résultats positifs ÷ résultats valides (positifs + négatifs) × 100.",
            "Où dans le projet": "Vue d'ensemble, Surveillance, COUSP",
            "Pourquoi c'est important": "Aide à suivre l'intensité de la circulation ou la pertinence du ciblage des prélèvements.",
            "Point d'attention": "Les résultats invalides, indéterminés ou absents réduisent l'interprétabilité.",
        },
        {
            "Famille": "Qualité / performance",
            "Notion": "Complétude (%)",
            "Comment elle est utilisée ici": "Proportion de champs renseignés ou moyenne de présence sur un panier de variables clés.",
            "Où dans le projet": "Qualité, COUSP, IREP, IDSR",
            "Pourquoi c'est important": "Mesure la capacité des données à soutenir une interprétation fiable.",
            "Point d'attention": "Une bonne complétude ne garantit pas à elle seule l'exactitude des données.",
        },
        {
            "Famille": "Qualité / performance",
            "Notion": "Promptitude / % sous seuil",
            "Comment elle est utilisée ici": "Cas avec délai ≤ seuil ÷ cas avec délai valide × 100.",
            "Où dans le projet": "Vue d'ensemble, Surveillance, COUSP, IREP",
            "Pourquoi c'est important": "Évalue la rapidité opérationnelle de notification, investigation, prélèvement ou admission.",
            "Point d'attention": "Les délais négatifs ou manquants sont exclus du calcul opérationnel.",
        },
        {
            "Famille": "Résumé de distribution",
            "Notion": "Médiane",
            "Comment elle est utilisée ici": "Valeur centrale des délais lorsque les observations sont ordonnées.",
            "Où dans le projet": "Vue d'ensemble, Surveillance, COUSP",
            "Pourquoi c'est important": "Décrit le délai typique sans être trop influencée par des valeurs extrêmes.",
            "Point d'attention": "La médiane ne résume pas à elle seule toute la dispersion.",
        },
        {
            "Famille": "Résumé de distribution",
            "Notion": "Quartiles (P25, P75), min et max",
            "Comment elle est utilisée ici": "Le projet résume plusieurs délais par P25, médiane, P75, minimum et maximum.",
            "Où dans le projet": "Surveillance, Qualité, COUSP",
            "Pourquoi c'est important": "Permet de voir si les délais sont homogènes ou très dispersés.",
            "Point d'attention": "Les extrêmes peuvent refléter des erreurs de saisie autant qu'une réalité terrain.",
        },
        {
            "Famille": "Tendance",
            "Notion": "Variation (%)",
            "Comment elle est utilisée ici": "(Valeur courante - valeur précédente) ÷ valeur précédente × 100.",
            "Où dans le projet": "Surveillance, IDSR, briefing narratif",
            "Pourquoi c'est important": "Met en évidence une hausse, une baisse ou une stabilité d'une période à l'autre.",
            "Point d'attention": "Quand la période précédente est très faible, la variation peut devenir très instable.",
        },
        {
            "Famille": "Tendance",
            "Notion": "Baseline / moyenne des semaines précédentes",
            "Comment elle est utilisée ici": "Moyenne mobile des semaines antérieures, souvent sur 3 semaines.",
            "Où dans le projet": "Alertes hebdomadaires, COUSP, IDSR",
            "Pourquoi c'est important": "Sert de point de comparaison pour repérer une hausse inhabituelle.",
            "Point d'attention": "Une baseline faible ou peu renseignée fragilise l'alerte automatique.",
        },
        {
            "Famille": "Tendance",
            "Notion": "Ratio de tendance / ratio à la baseline",
            "Comment elle est utilisée ici": "Cas récents ÷ baseline ou cas S0 ÷ moyenne des semaines précédentes.",
            "Où dans le projet": "Alertes hebdomadaires, IREP, clusters",
            "Pourquoi c'est important": "Traduit la vitesse d'accélération d'un signal.",
            "Point d'attention": "Un ratio élevé sur de très petits nombres n'a pas le même poids qu'un ratio élevé sur un grand volume.",
        },
        {
            "Famille": "Signal statistique",
            "Notion": "Alerte hebdomadaire",
            "Comment elle est utilisée ici": "Signal si le volume dépasse un minimum et une baseline multipliée par un ratio paramétrable.",
            "Où dans le projet": "Surveillance, COUSP",
            "Pourquoi c'est important": "Aide à prioriser rapidement les zones ou semaines à vérifier.",
            "Point d'attention": "Ce n'est pas une confirmation d'épidémie, mais un signal de contrôle.",
        },
        {
            "Famille": "Signal statistique",
            "Notion": "Cluster spatio-temporel",
            "Comment elle est utilisée ici": "Concentration récente de cas + croissance par rapport aux semaines antérieures.",
            "Où dans le projet": "Surveillance, COUSP",
            "Pourquoi c'est important": "Repère des foyers récents qui méritent une vérification terrain.",
            "Point d'attention": "Le cluster dépend du découpage spatial, de la fenêtre temporelle et des seuils choisis.",
        },
        {
            "Famille": "Score composite",
            "Notion": "Score de risque opérationnel",
            "Comment elle est utilisée ici": "Score 0-100 combinant volume, tendance, létalité, qualité, promptitude, positivité et flags QC.",
            "Où dans le projet": "COUSP, exports standards",
            "Pourquoi c'est important": "Permet de hiérarchiser l'attention opérationnelle entre territoires.",
            "Point d'attention": "Comme tout score composite, il simplifie plusieurs dimensions et ne remplace pas l'analyse experte.",
        },
        {
            "Famille": "Score composite",
            "Notion": "IREP",
            "Comment elle est utilisée ici": "Score pondéré fondé sur la tendance, l'incidence, la létalité, la promptitude et la complétude.",
            "Où dans le projet": "Onglet IREP",
            "Pourquoi c'est important": "Aide à prioriser les provinces ou zones de santé à partir de plusieurs dimensions à la fois.",
            "Point d'attention": "Le résultat dépend des poids, des composantes disponibles et de la population de référence.",
        },
        {
            "Famille": "Transformation de score",
            "Notion": "Scoring par quantiles / min-max",
            "Comment elle est utilisée ici": "Plusieurs composantes sont ramenées sur une échelle 0-100 avant combinaison.",
            "Où dans le projet": "IREP, score de risque",
            "Pourquoi c'est important": "Rend comparables des mesures de natures différentes avant de les additionner.",
            "Point d'attention": "Le score dépend de la distribution observée dans les données chargées.",
        },
        {
            "Famille": "Qualité",
            "Notion": "Doublon potentiel",
            "Comment elle est utilisée ici": "Cas proches repérés via une empreinte d'identité, d'âge, de dates et de géographie.",
            "Où dans le projet": "Qualité, exports, COUSP",
            "Pourquoi c'est important": "Protège les analyses contre les surcomptes et certaines incohérences de saisie.",
            "Point d'attention": "Ce n'est pas une preuve certaine de doublon, mais un signal de revue.",
        },
        {
            "Famille": "Aide au mapping",
            "Notion": "Score de similarité / confiance",
            "Comment elle est utilisée ici": "Le projet attribue un score aux propositions de mapping de colonnes et aux rapprochements flous.",
            "Où dans le projet": "Mapping 'Autre', cartographie fuzzy",
            "Pourquoi c'est important": "Aide à décider si une correspondance automatique est suffisamment crédible.",
            "Point d'attention": "Il s'agit d'un score algorithmique de rapprochement, pas d'une validation métier définitive.",
        },
    ]
    return pd.DataFrame(rows)


def _fmt_stat_value(value: Any, *, decimals: int = 1, suffix: str = "") -> str:
    """Formate proprement une valeur statistique pour l'affichage pédagogique."""
    if value is None or pd.isna(value):
        return "Non disponible"
    try:
        if decimals <= 0:
            return f"{int(round(float(value))):,}".replace(",", " ") + suffix
        return f"{float(value):,.{decimals}f}".replace(",", " ") + suffix
    except Exception:
        return f"{value}{suffix}"


def _build_statistical_examples(df: pd.DataFrame, *, idsr_mode: bool = False) -> pd.DataFrame:
    """Construit quelques exemples concrets à partir du périmètre filtré."""
    columns = ["Notion", "Valeur observée", "Pourquoi cet exemple compte"]
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, str]] = []

    if not idsr_mode:
        kpi = compute_indicators(df)
        cfr_pct = kpi.get("cfr_pct")
        positivity_pct = safe_pct(kpi.get("pos_num"), kpi.get("pos_den"))
        promptitude_pct, promptitude_n = pct_under_threshold(df.get("delai_onset_to_adm"), get_session_int("seuil_jours", 2))
        delay_summary = build_standard_delay_summary(df)
        quality_summary = standard_data_quality_summary(df)
        weekly = build_weekly_overview_table(df)

        rows.extend(
            [
                {
                    "Notion": "Effectif de cas",
                    "Valeur observée": _fmt_stat_value(kpi.get("n_cases"), decimals=0),
                    "Pourquoi cet exemple compte": "C'est la taille du périmètre actif sur lequel reposent les autres indicateurs.",
                },
                {
                    "Notion": "Létalité (CFR %)",
                    "Valeur observée": _fmt_stat_value(cfr_pct, decimals=2, suffix="%"),
                    "Pourquoi cet exemple compte": "Aide à suivre la gravité apparente parmi les cas visibles.",
                },
                {
                    "Notion": "Positivité labo (%)",
                    "Valeur observée": _fmt_stat_value(positivity_pct, decimals=1, suffix="%"),
                    "Pourquoi cet exemple compte": "Montre la part de résultats positifs parmi les résultats interprétables.",
                },
                {
                    "Notion": f"Promptitude ≤ {get_session_int('seuil_jours', 2)} jours",
                    "Valeur observée": f"{_fmt_stat_value(promptitude_pct, decimals=1, suffix='%')} (n={int(promptitude_n or 0)})",
                    "Pourquoi cet exemple compte": "Donne la part des cas arrivés dans le délai opérationnel retenu.",
                },
            ]
        )

        if not delay_summary.empty:
            top_delay = delay_summary.sort_values("n", ascending=False).iloc[0]
            rows.append(
                {
                    "Notion": f"Médiane du délai {top_delay['Type_delai']}",
                    "Valeur observée": f"{_fmt_stat_value(top_delay['Médiane_j'], decimals=1, suffix=' j')} (P25={_fmt_stat_value(top_delay['P25_j'], decimals=1)}, P75={_fmt_stat_value(top_delay['P75_j'], decimals=1)})",
                    "Pourquoi cet exemple compte": "Résume un délai central et sa dispersion sur la relation la mieux documentée.",
                }
            )

        if not quality_summary.empty:
            comp_row = quality_summary.loc[quality_summary["Indicateur"].astype(str).eq("Complétude médiane champs clés (%)")]
            if not comp_row.empty:
                rows.append(
                    {
                        "Notion": "Complétude médiane",
                        "Valeur observée": _fmt_stat_value(comp_row["Valeur"].iloc[0], decimals=1, suffix="%"),
                        "Pourquoi cet exemple compte": "Rappelle que l'interprétation dépend aussi de la qualité documentaire.",
                    }
                )

        if not weekly.empty and "Cas" in weekly.columns:
            latest_cases = pd.to_numeric(weekly["Cas"], errors="coerce").dropna()
            if not latest_cases.empty:
                rows.append(
                    {
                        "Notion": "Moyenne hebdomadaire des cas",
                        "Valeur observée": _fmt_stat_value(latest_cases.mean(), decimals=1),
                        "Pourquoi cet exemple compte": "Donne un ordre de grandeur de la charge hebdomadaire sur la période visible.",
                    }
                )
    else:
        cases_col = next((c for c in ["Total_cas", "Cas", "TOTALCAS"] if c in df.columns), None)
        deaths_col = next((c for c in ["Total_deces", "Deces", "TOTALDECES"] if c in df.columns), None)
        attack_col = next((c for c in ["Taux_attaque", "Taux_attaque_%"] if c in df.columns), None)
        cfr_col = next((c for c in ["Taux_letalite", "CFR_calc_%", "CFR_recalc_pct"] if c in df.columns), None)

        if cases_col is not None:
            total_cases = pd.to_numeric(df[cases_col], errors="coerce").fillna(0).sum()
            rows.append(
                {
                    "Notion": "Cas agrégés",
                    "Valeur observée": _fmt_stat_value(total_cases, decimals=0),
                    "Pourquoi cet exemple compte": "Montre le volume total rapporté dans le fichier IDSR filtré.",
                }
            )
        if deaths_col is not None:
            total_deaths = pd.to_numeric(df[deaths_col], errors="coerce").fillna(0).sum()
            rows.append(
                {
                    "Notion": "Décès agrégés",
                    "Valeur observée": _fmt_stat_value(total_deaths, decimals=0),
                    "Pourquoi cet exemple compte": "Documente la gravité rapportée dans la fenêtre IDSR active.",
                }
            )
        if cfr_col is not None:
            cfr_series = pd.to_numeric(df[cfr_col], errors="coerce").dropna()
            if not cfr_series.empty:
                rows.append(
                    {
                        "Notion": "Létalité moyenne rapportée",
                        "Valeur observée": _fmt_stat_value(cfr_series.mean(), decimals=2, suffix="%"),
                        "Pourquoi cet exemple compte": "Illustre l'usage du taux de létalité sur les données agrégées.",
                    }
                )
        if attack_col is not None:
            attack_series = pd.to_numeric(df[attack_col], errors="coerce").dropna()
            if not attack_series.empty:
                rows.append(
                    {
                        "Notion": "Taux d'attaque moyen",
                        "Valeur observée": _fmt_stat_value(attack_series.mean(), decimals=2, suffix="%"),
                        "Pourquoi cet exemple compte": "Montre comment le projet compare des territoires sur une base populationnelle.",
                    }
                )

    return pd.DataFrame(rows, columns=columns)


def _render_statistics_styles() -> None:
    """Injecte quelques styles locaux pour rendre l'onglet plus pédagogique."""
    st.markdown(
        """
<style>
    .stats-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.7rem;
    }

    .stats-chip {
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.20);
        border-radius: 999px;
        padding: 0.36rem 0.72rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }

    .stats-lead-note {
        margin-top: 0.55rem;
        color: rgba(255,255,255,0.92);
        font-size: 0.92rem;
        line-height: 1.45;
        max-width: 60rem;
    }

    .stats-path {
        background: rgba(255,255,255,0.90);
        border: 1px solid rgba(18, 53, 106, 0.10);
        border-radius: 20px;
        padding: 0.85rem 0.95rem;
        box-shadow: 0 12px 26px rgba(11, 44, 99, 0.08);
        margin: 0.45rem 0 1rem 0;
    }

    .stats-path-title {
        color: #0b2c63;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }

    .stats-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.7rem;
        margin: 0.4rem 0 1rem 0;
    }

    .stats-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(18, 53, 106, 0.10);
        border-radius: 18px;
        padding: 0.92rem 0.95rem;
        box-shadow: 0 12px 26px rgba(11, 44, 99, 0.08);
    }

    .stats-card-label {
        color: #5b718d;
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
    }

    .stats-card-value {
        margin-top: 0.32rem;
        color: #1553a1;
        font-size: 1.42rem;
        line-height: 1.1;
        font-weight: 800;
    }

    .stats-card-sub {
        margin-top: 0.28rem;
        color: #35506f;
        font-size: 0.84rem;
        line-height: 1.35;
    }

    .stats-formula-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1rem;
    }

    .stats-formula-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(246,250,255,0.96) 100%);
        border-radius: 18px;
        border: 1px solid rgba(18, 53, 106, 0.10);
        padding: 0.95rem 1rem;
        box-shadow: 0 12px 26px rgba(11, 44, 99, 0.07);
    }

    .stats-formula-label {
        color: #0b2c63;
        font-size: 0.84rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
    }

    .stats-formula-code {
        margin-top: 0.52rem;
        color: #1553a1;
        font-family: "Consolas", "Courier New", monospace;
        font-size: 0.9rem;
        line-height: 1.45;
        white-space: pre-wrap;
    }

    .stats-formula-why {
        margin-top: 0.55rem;
        color: #3d5878;
        font-size: 0.84rem;
        line-height: 1.38;
    }

    .stats-family-title {
        color: #0b2c63;
        font-size: 0.95rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_stats_card_grid(cards: list[dict[str, str]]) -> None:
    """Affiche une grille légère de cartes d'information."""
    blocks: list[str] = []
    for card in cards:
        label = html.escape(str(card.get("label", "")), quote=False)
        value = html.escape(str(card.get("value", "")), quote=False)
        subtitle = html.escape(str(card.get("subtitle", "")), quote=False).replace("\n", "<br>")
        blocks.append(
            f"""
<div class="stats-card">
  <div class="stats-card-label">{label}</div>
  <div class="stats-card-value">{value}</div>
  <div class="stats-card-sub">{subtitle}</div>
</div>
"""
        )
    st.markdown(f"<div class='stats-card-grid'>{''.join(blocks)}</div>", unsafe_allow_html=True)


def _render_statistics_hero(
    *,
    current_disease_label: str,
    mode_label: str,
    notions_count: int,
    examples_count: int,
) -> None:
    """Affiche un hero harmonisé avec l'identité COUSP du dashboard."""
    st.markdown(
        f"""
<div class="cousp-hero">
  <div class="cousp-hero-grid">
    <div class="cousp-hero-flag">STAT</div>
    <div class="cousp-hero-copy">
      <div class="cousp-hero-badge">Guide de lecture</div>
      <h1>NOTIONS STATISTIQUES UTILISÉES DANS LE DASHBOARD</h1>
      <p>Lecture standardisée des indicateurs, des dénominateurs, des scores et des signaux opérationnels.</p>
      <div class="stats-chip-row">
        <span class="stats-chip">Maladie : {html.escape(str(current_disease_label), quote=False)}</span>
        <span class="stats-chip">Mode : {html.escape(mode_label, quote=False)}</span>
        <span class="stats-chip">Notions documentées : {notions_count}</span>
        <span class="stats-chip">Exemples dynamiques : {examples_count}</span>
      </div>
      <div class="stats-lead-note">
        Cet onglet rappelle ce que mesurent les indicateurs, comment ils sont calculés dans votre projet,
        et pourquoi ils doivent être relus avec le périmètre filtré, la qualité des données et le contexte terrain.
      </div>
    </div>
    <div class="cousp-hero-meta">
      <div class="cousp-hero-meta-label">Lecture recommandée</div>
      <div class="cousp-hero-meta-value">Décrire, comparer, prioriser</div>
      <div class="cousp-hero-meta-sub">Un indicateur ne vaut jamais seul. Il doit toujours être relu avec sa source, son dénominateur et ses limites.</div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_formula_cards() -> None:
    """Présente les formules principales sous forme de cartes lisibles."""
    formulas = [
        {
            "label": "Pourcentage",
            "formula": "Pourcentage (%) = (numérateur / dénominateur) × 100",
            "why": "Utile pour comparer des situations de tailles différentes sur une base commune.",
        },
        {
            "label": "Létalité",
            "formula": "CFR (%) = (décès / cas) × 100",
            "why": "Sert à apprécier la gravité apparente parmi les cas documentés.",
        },
        {
            "label": "Positivité",
            "formula": "Positivité (%) = (positifs / résultats valides) × 100",
            "why": "Aide à suivre la part de confirmations parmi les résultats interprétables.",
        },
        {
            "label": "Incidence",
            "formula": "Incidence = (cas / population) × multiplicateur",
            "why": "Rend les comparaisons plus justes entre zones de tailles différentes.",
        },
        {
            "label": "Variation",
            "formula": "Variation (%) = ((courant - précédent) / précédent) × 100",
            "why": "Montre si la situation augmente, baisse ou se stabilise d'une période à l'autre.",
        },
        {
            "label": "Promptitude",
            "formula": "% sous seuil = (cas <= seuil / cas avec délai valide) × 100",
            "why": "Mesure la rapidité opérationnelle sans confondre délais manquants et retards réels.",
        },
    ]
    blocks: list[str] = []
    for item in formulas:
        label = html.escape(item["label"], quote=False)
        formula = html.escape(item["formula"], quote=False)
        why = html.escape(item["why"], quote=False)
        blocks.append(
            f"""
<div class="stats-formula-card">
  <div class="stats-formula-label">{label}</div>
  <div class="stats-formula-code">{formula}</div>
  <div class="stats-formula-why">{why}</div>
</div>
"""
        )
    st.markdown(f"<div class='stats-formula-grid'>{''.join(blocks)}</div>", unsafe_allow_html=True)


def _render_notions_by_family(notions: pd.DataFrame) -> None:
    """Présente les notions regroupées par famille métier."""
    family_counts = (
        notions.groupby("Famille", dropna=False)["Notion"].count().reset_index(name="Notions documentées")
        if not notions.empty
        else pd.DataFrame(columns=["Famille", "Notions documentées"])
    )
    if not family_counts.empty:
        _render_stats_card_grid(
            [
                {
                    "label": str(row["Famille"]),
                    "value": str(int(row["Notions documentées"])),
                    "subtitle": "notions expliquées dans cette famille",
                }
                for _, row in family_counts.iterrows()
            ]
        )

    for family in notions["Famille"].dropna().unique().tolist():
        subset = notions.loc[notions["Famille"].astype(str).eq(str(family))].copy()
        with st.expander(f"{family} ({len(subset)})", expanded=False):
            st.markdown(
                f"<div class='stats-family-title'>{html.escape(str(family), quote=False)}</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                subset.drop(columns=["Famille"]),
                width="stretch",
                hide_index=True,
                height=min(120 + 44 * len(subset), 420),
            )


def _render_examples_highlight(examples: pd.DataFrame) -> None:
    """Met en avant les exemples les plus parlants avant le tableau détaillé."""
    if examples.empty:
        st.info("Aucun exemple statistique n'est disponible sur le périmètre filtré actuel.")
        return

    _render_stats_card_grid(
        [
            {
                "label": str(row["Notion"]),
                "value": str(row["Valeur observée"]),
                "subtitle": str(row["Pourquoi cet exemple compte"]),
            }
            for _, row in examples.head(6).iterrows()
        ]
    )
    with st.expander("Voir le tableau détaillé des exemples calculés", expanded=False):
        st.dataframe(examples, width="stretch", hide_index=True)


def _render_learning_path() -> None:
    """Affiche le parcours conseillé pour lire l'onglet de façon cohérente."""
    st.markdown(
        """
<div class="stats-path">
  <div class="stats-path-title">Parcours conseillé</div>
  <div class="cousp-chain-stepper">
    <div class="cousp-chain-step"><span class="dot blue"></span>Comprendre le périmètre</div>
    <div class="cousp-chain-step"><span class="dot orange"></span>Identifier le dénominateur</div>
    <div class="cousp-chain-step"><span class="dot amber"></span>Relire la formule</div>
    <div class="cousp-chain-step"><span class="dot green"></span>Examiner l'exemple actif</div>
    <div class="cousp-chain-step"><span class="dot purple"></span>Comparer entre onglets</div>
    <div class="cousp-chain-step"><span class="dot red"></span>Vérifier les limites</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_tab_importance_cards() -> None:
    """Explique quelles notions dominent dans chaque onglet métier."""
    cards = [
        {"label": "Vue d'ensemble", "value": "KPI clés", "subtitle": "Effectifs, pourcentages, CFR, positivité, délais médians, % sous seuil."},
        {"label": "Surveillance", "value": "Tendance", "subtitle": "Variation %, baseline, alertes hebdomadaires, clusters, dispersion des délais."},
        {"label": "Profil", "value": "Distribution", "subtitle": "Fréquences, âge-sexe, structure spatiale, profil personne-lieu-temps."},
        {"label": "Qualité et export", "value": "Fiabilité", "subtitle": "Complétude, incohérences, doublons potentiels, valeurs hors bornes."},
        {"label": "COUSP", "value": "Priorisation", "subtitle": "Chaîne standard, seuils opérationnels, signaux, score de risque."},
        {"label": "IREP", "value": "Score composite", "subtitle": "Incidence, létalité, promptitude, complétude et redistribution des poids."},
        {"label": "IDSR", "value": "Lecture agrégée", "subtitle": "Incidence, taux d'attaque, létalité agrégée, variations hebdomadaires."},
    ]
    _render_stats_card_grid(cards)


def _render_caveat_cards() -> None:
    """Rappelle les pièges d'interprétation les plus fréquents."""
    caveats = [
        {"constat": "Petit dénominateur", "interpretation": "Un pourcentage élevé sur peu de cas peut sembler spectaculaire sans être stable."},
        {"constat": "Données incomplètes", "interpretation": "Une tendance apparente peut venir d'un retard de saisie ou de consolidation."},
        {"constat": "Score composite", "interpretation": "Le score résume plusieurs dimensions, mais masque parfois la cause précise de la priorité."},
        {"constat": "Alerte automatique", "interpretation": "Une alerte statistique reste un signal de vérification, pas une confirmation."},
        {"constat": "Médiane seule", "interpretation": "Une médiane correcte peut cacher une queue de retards longs sur une partie des cas."},
        {"constat": "Comparaison spatiale", "interpretation": "Comparer deux zones sans même qualité de données ou sans population fiable peut induire en erreur."},
    ]
    col_left, col_right = st.columns(2)
    for idx, caveat in enumerate(caveats):
        target = col_left if idx % 2 == 0 else col_right
        with target:
            render_reader_narrative("Point de vigilance", caveat, tone="missing")


def render_statistics_tab(ctx: dict) -> None:
    """Render the statistical concepts tab."""
    globals().update(ctx)

    _render_statistics_styles()
    render_section_title(0, "Notions statistiques utilisées dans le projet")
    render_reader_narrative(
        "Pourquoi cet onglet existe",
        "Cette page explique les notions statistiques et les indicateurs utilisés dans le dashboard, afin que les résultats "
        "soient compris de manière homogène avant toute interprétation opérationnelle.",
        tone="decision",
    )

    current_disease_label = DISEASE_SPECS.get(disease_key, {}).get("label", disease_key)
    scope_rows = len(df_f) if isinstance(df_f, pd.DataFrame) else 0
    notions = _build_statistical_notions_catalog()
    examples = _build_statistical_examples(
        df_f if isinstance(df_f, pd.DataFrame) else pd.DataFrame(),
        idsr_mode=bool(IDSR_MODE),
    )
    mode_label = "IDSR agrégé hebdomadaire" if IDSR_MODE else "Liste linéaire individuelle"
    _render_statistics_hero(
        current_disease_label=current_disease_label,
        mode_label=mode_label,
        notions_count=len(notions),
        examples_count=len(examples),
    )
    render_standards_note()
    _render_stats_card_grid(
        [
            {
                "label": "Périmètre actif",
                "value": f"{scope_rows:,}".replace(",", " ") if scope_rows else "0",
                "subtitle": "lignes actuellement analysées après filtres",
            },
            {
                "label": "Mode analytique",
                "value": "IDSR" if IDSR_MODE else "Line list",
                "subtitle": mode_label,
            },
            {
                "label": "Familles couvertes",
                "value": str(notions["Famille"].nunique() if not notions.empty else 0),
                "subtitle": "groupes de notions expliqués dans cet onglet",
            },
            {
                "label": "Question directrice",
                "value": "Bien lire",
                "subtitle": "Que mesure exactement l'indicateur, comment est-il calculé ici, et jusqu'où peut-on l'interpréter ?",
            },
        ]
    )
    _render_learning_path()
    with st.expander("Contexte détaillé du périmètre actif", expanded=False):
        context_tbl = pd.DataFrame(
            [
                ("Maladie / source sélectionnée", current_disease_label),
                ("Mode analytique", mode_label),
                ("Périmètre actif", f"{scope_rows:,} lignes après filtres".replace(",", " ") if scope_rows else "Non disponible"),
                ("Question clé", "Que mesure exactement l'indicateur, comment est-il calculé ici, et comment l'interpréter sans surconclure ?"),
            ],
            columns=["Élément", "Description"],
        )
        st.dataframe(context_tbl, width="stretch", hide_index=True)

    st.markdown("### 1. Principes de lecture")
    col_a, col_b = st.columns(2)
    with col_a:
        render_reader_narrative(
            "Règle de lecture",
            {
                "constat": "Une mesure statistique résume une partie de l'information disponible.",
                "interpretation": "Elle aide à structurer la lecture, mais ne remplace jamais la validation terrain ni la discussion métier.",
                "action": "Relire chaque indicateur avec son dénominateur, son périmètre et la source des données.",
            },
            tone="decision",
        )
    with col_b:
        render_reader_narrative(
            "Règle de prudence",
            {
                "constat": "Une hausse, une alerte ou un score élevé est d'abord un signal.",
                "interpretation": "Dans ce projet, beaucoup de notions servent à la priorisation opérationnelle plus qu'à l'inférence académique.",
                "action": "Toujours confronter le signal à la complétude, à la promptitude et au contexte de collecte.",
            },
            tone="standard",
        )

    st.markdown("### 2. Catalogue des notions utilisées")
    _render_notions_by_family(notions)
    with st.expander("Voir le catalogue complet en tableau", expanded=False):
        st.dataframe(notions, width="stretch", hide_index=True, height=540)

    st.markdown("### 3. Formules simples à retenir")
    _render_formula_cards()
    st.caption(
        "Dans le dashboard, ces formules sont souvent combinées avec des règles métier: exclusions des délais négatifs, résultats valides seulement, "
        "ou inférence d'investigation à partir d'une classification exploitable."
    )

    st.markdown("### 4. Exemples calculés sur le périmètre actif")
    _render_examples_highlight(examples)

    st.markdown("### 5. Notions les plus importantes par onglet")
    _render_tab_importance_cards()

    st.markdown("### 6. Comment éviter les mauvaises interprétations")
    _render_caveat_cards()
