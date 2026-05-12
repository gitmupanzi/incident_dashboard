from dashboard_app.core import (
    AGE_UNIT_DAY_PATTERN,
    AGE_UNIT_MONTH_PATTERN,
    AGE_UNIT_WEEK_PATTERN,
    AGE_UNIT_YEAR_PATTERN,
    APP_BUILD_TAG,
    Any,
    BytesIO,
    COLOR_CASES,
    COLOR_CFR,
    COLOR_DEATHS,
    COLOR_FEMININ,
    COLOR_INCONNU,
    COLOR_MASCULIN,
    Dict,
    HAS_CUSTOM_VIZ,
    HAS_RAPIDFUZZ,
    Iterable,
    List,
    MAP_ANNOTATION_MODE_OPTIONS,
    METRIC_COLUMN_ALIASES,
    MISSING_LABEL,
    MISSING_LABEL_VERBOSE,
    MultiPolygon,
    Optional,
    Path,
    SEX_COLOR_MAP,
    SequenceMatcher,
    Tuple,
    Union,
    as_list,
    build_cases_deaths_cfr_pivot,
    build_weekly_cases_cfr_combo,
    build_weekly_cases_deaths_combo,
    build_weekly_multiline_by_group,
    carte_statique_matplotlib,
    choose_week_column,
    compter_par_categorie,
    ctx,
    date,
    datetime,
    df_to_csv_bytes,
    extraire_numero,
    extraire_ordre_tranche,
    flatten_columns,
    fmt_yw_label,
    glob,
    go,
    gpd,
    graphique_barres_facette,
    graphique_pyramide_age,
    hashlib,
    is_numeric_dtype,
    json,
    logger,
    logging,
    make_unique,
    np,
    ordered_weeks_from_weekly_sorted,
    orient,
    os,
    pd,
    plot_barres_pct_sous_seuil,
    plot_boxplot_delais_plotly,
    plot_camembert_interactif,
    plot_courbe_par_categories_plotly,
    plot_courbe_plotly,
    plot_evolution_multi_auto,
    plot_histogramme_groupe_interactif_empile,
    plot_pyramide_symetrique,
    plt,
    prepare_idsr_numeric,
    process,
    px,
    re,
    render_pivot_with_cfr,
    reorder_pivot_weeks,
    safe_pct,
    st,
    st_dataframe_safe,
    tempfile,
    unicodedata,
    verifier_presence_colonnes,
    fuzz,
)
from dashboard_app.core import (
    _normalize_name,
    _scale_marker_sizes,
    _strip_accents,
)

# --- Colonnes (tes noms exacts) ---
COL_PROV = "Province_notification"
COL_ZS   = "Zone_de_sante_notification"
COL_AS   = "Aire_de_sante_notification"

COL_YEAR = "Annee_epid"
COL_WNUM = "Num_semaine_epid"
COL_WEEK = "Semaine_epid"  # colonne semaine (résolue dynamiquement plus bas si besoin)
COL_SEX  = "Sexe"
COL_AGE  = "Age"
COL_UNIT = "Unite_age"
COL_AGEG = "Tranche_age_en_ans"
COL_AGEG2 = "Tranche_age"

COL_PREL = "Prelevement"
COL_TDR  = "TDR_realise"
COL_TDRR = "TDR_Resultat"
COL_HOSP = "Hospitalisation"
COL_DEHY = "Degre_deshydratation"
COL_ISSUE= "Issue"
COL_CLASS= "Classification_finale"

DATE_ONSET = "Date_debut_maladie"
DATE_NOTIF = "Date_notification"
DATE_ADM   = "Date_admission_au_CT"
DATE_PREL  = "Date_prelevement"
DATE_CONS  = "Date_consultation"
DATE_INV   = "Date_investigation"
DATE_RES   = "Date_resultat"
DATE_RECEP = "Date_reception_labo"
DATE_ISSUE = "Date_issue"

PROVINCE_PATTERNS = [
    (r"^\s*bas[\s_-]*uele\s*$", "Bas Uele"),
    (r"^\s*equateur\s*$", "Equateur"),
    (r"^\s*haut[\s_-]*katanga\s*$", "Haut Katanga"),
    (r"^\s*haut[\s_-]*lomami\s*$", "Haut Lomami"),
    (r"^\s*haut[\s_-]*uele\s*$", "Haut Uele"),
    (r"^\s*ituri\s*$", "Ituri"),
    (r"^\s*kasai[\s_-]*central\s*$", "Kasai Central"),
    (r"^\s*kasai\s*$", "Kasai"),
    (r"^\s*kinshasa\s*$", "Kinshasa"),
    (r"^\s*kongo[\s_-]*central\s*$", "Kongo Central"),
    (r"^\s*kasai[\s_-]*oriental\s*$", "Kasai Oriental"),
    (r"^\s*kwango\s*$", "Kwango"),
    (r"^\s*kwilu\s*$", "Kwilu"),
    (r"^\s*lomami\s*$", "Lomami"),
    (r"^\s*lualaba\s*$", "Lualaba"),
    (r"^\s*mai[\s_-]*ndombe\s*$", "Maindombe"),
    (r"^\s*maindombe\s*$", "Maindombe"),
    (r"^\s*maniema\s*$", "Maniema"),
    (r"^\s*mongala\s*$", "Mongala"),
    (r"^\s*nord[\s_-]*kivu\s*$", "Nord Kivu"),
    (r"^\s*nord[\s_-]*ubangi\s*$", "Nord Ubangi"),
    (r"^\s*sankuru\s*$", "Sankuru"),
    (r"^\s*sud[\s_-]*kivu\s*$", "Sud Kivu"),
    (r"^\s*sud[\s_-]*ubangi\s*$", "Sud Ubangi"),
    (r"^\s*tanganyika\s*$", "Tanganyika"),
    (r"^\s*tshuapa\s*$", "Tshuapa"),
    (r"^\s*tshopo\s*$", "Tshopo"),
]


# =========================
# SECTION: STANDARDISATION MULTI-MALADIES
# MALADIES (CONFIG / SPECS)
# Objectif: rendre le dashboard "multi-line list" (Choléra, Rougeole, Mpox, Ebola, Intox, IDSR)
# - Chaque maladie peut avoir des noms de colonnes différents -> on renomme vers le schéma commun
# - Le reste du pipeline (standardize_ll_core -> standardize_df -> KPI/graphs) reste identique
# =========================

DISEASE_SPECS: Dict[str, Dict[str, Any]] = {
    "cholera": {
        "label": "Choléra",
        "enabled": True,
        "default_sheet": "LL_Cholera",
        "rename_map": {
            # variations fréquentes
            "Nom_complet_cas_suspect": "Nom_complet",
        },
        # dates candidates (pour Date_debut_maladie si manquante)
        "onset_candidates": ["Date_debut_maladie"],
        "notif_candidates": ["Date_notification"],
        "adm_candidates": ["Date_admission_au_CT"],
        "prel_candidates": ["Date_prelevement"],
        "issue_candidates": ["Date_issue", "Date_sortie_au_CT", "Date_de_guerie"],
        "result_candidates": ["Date_resultat"],
        "receipt_candidates": ["Date_reception_labo"],
        "class_candidates": ["Classification_finale"],
    },
    "rougeole": {
        "label": "Rougeole (line list)",
        "enabled": True,
        "default_sheet": "LL_Rougeole_Rubeole",
        "rename_map": {
            "Evolution": "Issue",
            "Echantillon_preleve": "Prelevement",
            "Date_reception_echantillon_labo": "Date_reception_labo",
            "Date_envoi_resultat": "Date_resultat",
            "Date_partage_resultat_pcr": "Date_resultat",
            "Date_partage_resultat_tdr_surveillance_epi": "Date_resultat",
            "Date_partage_resultat_machd_surveillance_epi": "Date_resultat",
            "Nombre_doses_vaccin": "Nombre_dose_recues",
        },
        "onset_candidates": ["Date_debut_maladie", "Date_debut_symptomes"],
        "notif_candidates": ["Date_notification"],
        "adm_candidates": ["Date_admission_au_CT", "Date_admission"],
        "prel_candidates": [
            "Date_prelevement",
            "Date_prelevement_clean",
            "Date_prelevement_urine",
            "Date_prelevement_respiratoire",
            "Date_autre_prelevement",
        ],
        "issue_candidates": ["Date_issue"],
        "result_candidates": [
            "Date_resultat",
            "Date_reception_resultat",
            "Date_envoi_resultat",
            "Date_partage_resultat_pcr",
            "Date_partage_resultat_tdr_surveillance_epi",
            "Date_partage_resultat_machd_surveillance_epi",
            "Date_resultat_igm_labo_national",
            "Date_resultat_isolement_viral_regional",
        ],
        "receipt_candidates": [
            "Date_reception_labo",
            "Date_reception_echantillon",
            "Date_reception_echantillon_labo",
            "Date_acheminement",
        ],
        "class_candidates": ["Classification_finale", "Status_cas"],
    },
    "mpox": {
        "label": "Mpox (line list)",
        "enabled": True,
        "default_sheet": "",
        "rename_map": {
            # =========================
            # GÉOGRAPHIE / IDENTITÉ
            # =========================
            "Nom_complet_cas_suspect": "Nom_complet",
            "Nom_Cas": "Nom_complet",

            "Div_Prov": "Province_notification",
            "Zone_Sante": "Zone_de_sante_notification",
            "ZoneSante": "Zone_de_sante_notification",
            "Aire_Sante": "Aire_de_sante_notification",

            # =========================
            # DÉMOGRAPHIE
            # =========================
            "Age_Cas": "Age",
            "Age_Unite": "Unite_age",
            "Sexe_Cas": "Sexe",
            "AgeGroup": "Tranche_age_en_ans",
            "AgeGroup2": "Tranche_age",

            # =========================
            # CLASSIFICATION / ISSUE
            # =========================
            "Classification_finale_du_cas": "Classification_finale",
            "ClassificationFinale": "Classification_finale",
            "Statut_Cas": "Classification_finale",

            "DateDecharge": "Date_issue",
            "Date_Deces": "Date_issue",
            "Date_Décès": "Date_issue",

            # =========================
            # DATES CLÉS
            # =========================
            "Date_debut_symptomes": "Date_debut_maladie",
            "Date_Investigation": "Date_investigation",
            "DateHospit": "Date_admission_au_CT",

            # =========================
            # PRÉLÈVEMENT / LABO
            # =========================
            "Prelevement_realise_au_moment_de_investigation": "Prelevement",
            "Prelevement_investigation": "Prelevement",
            "PrelevementInvestigation": "Prelevement",
            "Prelevement_apres_investigation": "Prelevement_apres_investigation",
            "PrelevementApresInvestigation": "Prelevement_apres_investigation",

            "Si_oui_date_de_prelevement": "Date_prelevement",
            "Date_Prelevement": "Date_prelevement",

            "Date_Envoie_Echantillon": "Date_d_envoie_d_echantillons_au_laboratoire",
            "Date_Reception_Echantillon": "Date_reception_labo",

            "Date_Analyse": "Date_resultat",
            "Date_resultat_final_opx": "Date_resultat",

            "Quel_est_le_resultats": "Resultat_labo",
            "Resultat_final_opx": "Resultat_labo",
            "Resultats_Labo": "Resultat_labo",

            "Status_Analyse": "Statut_analyse",
        },
        "onset_candidates": [
            "Date_debut_maladie",
            "Date_debut_symptomes",
            "Date_Eruptions_Cutanee",
            "Date_Fievre",
            "Si_oui_des_eruption_cutanee_quelle_est_la_date_de_debut_de_leruption_cutanee",
            "Si_le_cas_suspect_a_eu_une_fievre_quelle_est_la_date_du_debut_de_la_fievre",
        ],
        "notif_candidates": [
            "Date_notification",
        ],
        "adm_candidates": [
            "Date_admission_au_CT",
            "DateHospit",
            "Date_d_hospitalisation_isolement",
            "Date_investigation",
            "Date_Investigation",
        ],
        "prel_candidates": [
            "Date_prelevement",
            "Date_Prelevement",
            "Date_d_envoie_d_echantillons_au_laboratoire",
            "Date_envoie_echantillon",
            "Date_Envoie_Echantillon",
        ],
    },
    "ebola": {
        "label": "Ebola / MVE (line list)",
        "enabled": True,
        "default_sheet": "LL_Ebola",
        "rename_map": {
            "Date_debut_symptomes": "Date_debut_maladie",
            "Date_issue": "Date_sortie_au_CT",
        },
        "onset_candidates": ["Date_debut_maladie", "Date_debut_symptomes"],
        "notif_candidates": ["Date_notification"],
        "adm_candidates": ["Date_admission_au_CT"],
        "prel_candidates": ["Date_prelevement"],
        "issue_candidates": ["Date_issue", "Date_sortie_au_CT", "Date_deces"],
        "result_candidates": ["Date_resultat"],
        "receipt_candidates": ["Date_reception_labo"],
        "class_candidates": ["Classification_finale"],
    },
    "intox": {
        "label": "Intoxication (line list)",
        "enabled": True,
        "default_sheet": "LL_Intox",
        "rename_map": {
            "Date_consultation": "Date_notification",
            "Date_apparition_signes": "Date_debut_maladie",
        },
        "onset_candidates": ["Date_debut_maladie", "Date_apparition_signes"],
        "notif_candidates": ["Date_notification", "Date_consultation"],
        "adm_candidates": ["Date_admission_au_CT"],
        "prel_candidates": ["Date_prelevement"],
        "issue_candidates": ["Date_issue", "Date_sortie_au_CT"],
        "result_candidates": ["Date_resultat"],
        "receipt_candidates": ["Date_reception_labo"],
        "class_candidates": ["Classification_finale"],
    },
    "idsr": {
        "label": "IDSR agrégé (hebdo)",
        "enabled": True,
        "default_sheet": "IDSR",
        "rename_map": {
            "Num": "Num",
            "NUM": "Num",
            "Pays": "Pays",
            "PAYS": "Pays",
            "Province": "Province_notification",
            "PROV": "Province_notification",
            "Zone_de_sante": "Zone_de_sante_notification",
            "ZS": "Zone_de_sante_notification",
            "POP": "Population",
            "NUMSEM": "Num_semaine_epid",
            "DEBUTSEM": "Date_debut_semaine",
            "Year": "Annee_epid",
            "MALADIE": "Maladie",
            "C328TNN": "Cas_tnn",
            "C011MOIS": "Cas_0_11mois",
            "C1259MOIS": "Cas_12_59mois",
            "C515ANS": "Cas_5_14ans",
            "CP15ANS": "Cas_15plus",
            "DTNN": "Deces_tnn",
            "D011MOIS": "Deces_0_11mois",
            "D1259MOIS": "Deces_12_59mois",
            "D515ANS": "Deces_5_14ans",
            "DP15ANS": "Deces_15plus",
            "TOTALCAS": "Total_cas",
            "TOTALDECES": "Total_deces",
            "LETAL": "Taux_letalite",
            "ATTAQ": "Taux_attaque",
        },
        "onset_candidates": ["Date_debut_semaine", "Date_notification"],
        "notif_candidates": ["Date_debut_semaine", "Date_notification"],
        "adm_candidates": [],
        "prel_candidates": [],
    },
    "meningite": {
        "label": "Méningite (line list)",
        "enabled": True,
        "default_sheet": "LL_Meningite",
        "rename_map": {
            "Date_de_reception": "Date_reception_labo",
            "Classification_investigation": "Classification_finale",
        },
        "onset_candidates": ["Date_debut_maladie", "Date_debut_symptomes"],
        "notif_candidates": ["Date_notification", "Date_consultation"],
        "adm_candidates": ["Date_admission_au_CT", "Date_admission"],
        "prel_candidates": ["Date_prelevement"],
        "issue_candidates": ["Date_issue", "Date_sortie_au_CT"],
        "result_candidates": ["Date_resultat"],
        "receipt_candidates": ["Date_reception_labo", "Date_de_reception"],
        "class_candidates": ["Classification_finale", "Classification_investigation"],
    },
    "autre": {
        "label": "Autre (line list générique)",
        "enabled": True,
        "default_sheet": "",
        "rename_map": {},
        "onset_candidates": [
            "Date_debut_maladie",
            "Date_debut_symptomes",
            "Date_apparition_signes",
        ],
        "notif_candidates": [
            "Date_notification",
            "Date_consultation",
        ],
        "adm_candidates": [
            "Date_admission_au_CT",
            "Date_admission",
        ],
        "prel_candidates": [
            "Date_prelevement",
            "Date_prelevement_clean",
        ],
        "issue_candidates": [
            "Date_issue",
            "Date_sortie_au_CT",
            "Date_deces",
            "Date_de_guerie",
        ],
        "result_candidates": [
            "Date_resultat",
            "Date_reception_resultat",
        ],
        "receipt_candidates": [
            "Date_reception_labo",
            "Date_reception_echantillon",
            "Date_de_reception",
        ],
        "class_candidates": [
            "Classification_finale",
            "Classification_investigation",
            "Status_cas",
        ],
    },
}


def is_disease_enabled(disease_key: str) -> bool:
    """Indique si la maladie sélectionnée est activée dans le dashboard."""
    spec = DISEASE_SPECS.get(disease_key, {})
    return bool(spec.get("enabled", True))

def _coalesce_first(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """Retourne la première colonne non-NA dans candidates (coalesce)."""
    if not candidates:
        return pd.Series([pd.NA] * len(df), index=df.index)
    out = None
    for c in candidates:
        if c in df.columns:
            s = df[c]
            out = s if out is None else out.combine_first(s)
    if out is None:
        out = pd.Series([pd.NA] * len(df), index=df.index)
    return out


def _rename_columns_by_alias_map(df: pd.DataFrame, alias_map: Dict[str, str]) -> pd.DataFrame:
    """Applique un renommage robuste aux accents, espaces et variantes de casse."""
    if not alias_map:
        return df

    real_cols_norm = {_normalize_name(c): c for c in df.columns}
    rename_dict: Dict[str, str] = {}
    reserved_targets = set(df.columns)
    for src, dst in alias_map.items():
        real_src = real_cols_norm.get(_normalize_name(src))
        if (real_src is not None) and (dst not in reserved_targets):
            rename_dict[real_src] = dst
            reserved_targets.add(dst)

    return df.rename(columns=rename_dict) if rename_dict else df

@st.cache_data(show_spinner=False)
def standardize_ll_by_disease(df: pd.DataFrame, disease_key: str) -> pd.DataFrame:
    """
    1) Renommage spécifique maladie (DISEASE_SPECS[disease_key]['rename_map'])
    2) Standardisation core (standardize_ll_core)
    3) Coalesce les dates et variables standards utiles aux onglets
       à partir des candidats de la maladie (si colonnes manquantes ou vides)
    """
    spec = DISEASE_SPECS.get(disease_key, DISEASE_SPECS["cholera"])
    df = _clean_colnames(df)

    # 1) Rename spécifique (robuste aux accents, espaces, ponctuation)
    df = _rename_columns_by_alias_map(df, spec.get("rename_map", {}) or {})

    # 2) Core
    df = standardize_ll_core(df)

    # 2a) Post-traitement spécifique Rougeole labo
    if disease_key == "rougeole":
        df = _enrich_rougeole_lab_columns(df)

    # 2b) Post-traitement spécifique Mpox
    if disease_key == "mpox":
        # Alias explicite pour la chaîne labo standard
        if "Resultat_labo" in df.columns and "TDR_Resultat" not in df.columns:
            df["TDR_Resultat"] = df["Resultat_labo"]

        if "TDR_realise" not in df.columns:
            df["TDR_realise"] = pd.NA

        if "Statut_analyse" in df.columns:
            s = df["Statut_analyse"].astype("string").str.strip().str.lower()
            yes_mask = s.isin([
                "fait", "réalisé", "realise", "réalisée",
                "complete", "complété", "completee", "termine", "terminé"
            ])
            no_mask = s.isin([
                "non fait", "non realise", "non réalisé",
                "en attente", "pending", "attente"
            ])
            df.loc[yes_mask, "TDR_realise"] = "Oui"
            df.loc[no_mask & df["TDR_realise"].isna(), "TDR_realise"] = "Non"

        if "Resultat_labo" in df.columns:
            has_result = df["Resultat_labo"].notna()
            df.loc[has_result & df["TDR_realise"].isna(), "TDR_realise"] = "Oui"

    # 3) Coalesce dates et variables standards (si vides)
    date_candidate_map = {
        "Date_debut_maladie": "onset_candidates",
        "Date_notification": "notif_candidates",
        "Date_admission_au_CT": "adm_candidates",
        "Date_prelevement": "prel_candidates",
        "Date_reception_labo": "receipt_candidates",
        "Date_resultat": "result_candidates",
        "Date_issue": "issue_candidates",
    }
    text_candidate_map = {
        "Classification_finale": "class_candidates",
    }

    # - On convertit toutes les candidates de date en datetime (robuste)
    for colset in date_candidate_map.values():
        for c in spec.get(colset, []) or []:
            if c in df.columns:
                df[c] = _to_dt(df[c])

    # On remplit les colonnes standard ligne par ligne pour combler les trous
    # sans ecraser les valeurs deja presentes.
    for target_col, spec_key in date_candidate_map.items():
        if target_col not in df.columns:
            df[target_col] = pd.NaT
        fallback_series = _coalesce_first(df, spec.get(spec_key, []))
        df[target_col] = _to_dt(df[target_col]).combine_first(fallback_series)

    for target_col, spec_key in text_candidate_map.items():
        if target_col not in df.columns:
            df[target_col] = pd.NA
        fallback_series = _coalesce_first(df, spec.get(spec_key, []))
        df[target_col] = df[target_col].combine_first(fallback_series)

    # Recalcul ISO si nécessaire après coalesce
    # (ex: Mpox où Date_debut_maladie était vide et vient d'être rempli)
    ref = df["Date_notification"].combine_first(df["Date_debut_maladie"])
    if ref.notna().any():
        iso = ref.dt.isocalendar()
        if "Annee_epid" not in df.columns:
            df["Annee_epid"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        if "Num_semaine_epid" not in df.columns:
            df["Num_semaine_epid"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

        year_current = pd.to_numeric(df["Annee_epid"], errors="coerce").astype("Int64")
        week_current = pd.to_numeric(df["Num_semaine_epid"], errors="coerce").astype("Int64")
        df["Annee_epid"] = year_current.combine_first(iso["year"].astype("Int64"))
        df["Num_semaine_epid"] = week_current.combine_first(iso["week"].astype("Int64"))

    if "Semaine_epid" not in df.columns:
        df["Semaine_epid"] = pd.NA
    week_label_current = df["Semaine_epid"].astype("string")
    missing_week_label = week_label_current.isna() | week_label_current.str.strip().eq("")
    if missing_week_label.any():
        y = pd.to_numeric(df["Annee_epid"], errors="coerce").astype("Int64")
        w = pd.to_numeric(df["Num_semaine_epid"], errors="coerce").astype("Int64")
        computed_week_label = y.astype("string") + "-W" + w.astype("string").str.zfill(2)
        valid_computed_week_label = y.notna() & w.notna()
        fill_mask = missing_week_label & valid_computed_week_label
        df.loc[fill_mask, "Semaine_epid"] = computed_week_label.loc[fill_mask]

    return df


# Provinces épidémiques (tes paramètres)
EPIDEMIE = {
    "Bas Uele": False, "Equateur": True, "Haut Katanga": True, "Haut Lomami": True,
    "Haut Uele": False, "Ituri": False, "Kasai Central": False, "Kasai": False,
    "Kinshasa": True, "Kongo Central": False, "Kasai Oriental": True, "Kwango": False,
    "Kwilu": False, "Lomami": True, "Lualaba": True, "Maindombe": True, "Maniema": True,
    "Mongala": True, "Nord Kivu": True, "Nord Ubangi": False, "Sankuru": True,
    "Sud Kivu": True, "Sud Ubangi": False, "Tanganyika": True, "Tshuapa": False,
    "Tshopo": True,
}
PROVINCES_EPID = [p for p, ok in EPIDEMIE.items() if ok]
PROVINCES_END  = [p for p, ok in EPIDEMIE.items() if not ok]


# =========================
# HELPERS (UI)
# =========================
def get_toggle_flag(flag_name: str, default: bool = False) -> bool:
    """Lecture sûre d'un booléen UI depuis st.session_state."""
    try:
        return bool(st.session_state.get(flag_name, default))
    except Exception:
        return bool(default)

def get_session_int(key: str, default: int) -> int:
    """Lecture sûre d'un entier depuis st.session_state."""
    try:
        return int(st.session_state.get(key, default))
    except Exception:
        return int(default)


def get_provinces_epid() -> List[str]:
    """Source unique de vérité pour la liste des provinces épidémiques."""
    return list(PROVINCES_EPID)


def call_optional_function(func_name: str, *args, default=None, **kwargs):
    """Appelle une fonction globale optionnelle si elle existe et est callable."""
    func = globals().get(func_name)
    if callable(func):
        try:
            return func(*args, **kwargs)
        except Exception:
            return default
    return default


def pct_change_safe(cur, prv):
    """Variation relative sûre, retourne np.nan si la base est nulle ou absente."""
    if pd.isna(prv) or prv == 0:
        return np.nan
    return (cur - prv) / prv * 100.0


def pct_change_metric_safe(cur, prv):
    """Variation relative sûre pour st.metric, retourne None si la base est non interprétable."""
    if pd.isna(cur):
        return None
    delta = pct_change_safe(cur, prv)
    return None if pd.isna(delta) else float(delta)


def _apply_compact_bar_spacing(fig: Optional[go.Figure], bargap: float = 0.0, bargroupgap: float = 0.0) -> Optional[go.Figure]:
    """Resserre les espacements pour tous les graphiques à barres / histogrammes."""
    if fig is None:
        return fig
    try:
        has_bar_like = any(isinstance(tr, (go.Bar, go.Histogram)) for tr in fig.data)
        if not has_bar_like:
            return fig

        current_bargap = fig.layout.bargap
        current_bargroupgap = fig.layout.bargroupgap
        fig.update_layout(
            bargap=bargap if current_bargap is None else min(float(current_bargap), float(bargap)),
            bargroupgap=bargroupgap if current_bargroupgap is None else min(float(current_bargroupgap), float(bargroupgap)),
        )
    except Exception:
        return fig
    return fig



def _json_safe_plotly_value(value):
    """Convertit les valeurs pandas/numpy non sérialisables en valeurs JSON compatibles Plotly/Streamlit."""
    try:
        if value is pd.NA or value is pd.NaT:
            return None
    except Exception:
        pass

    if isinstance(value, dict):
        return {k: _json_safe_plotly_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe_plotly_value(v) for v in value]

    if isinstance(value, (pd.Series, pd.Index)):
        return [_json_safe_plotly_value(v) for v in value.tolist()]

    if isinstance(value, np.ndarray):
        return [_json_safe_plotly_value(v) for v in value.tolist()]

    if isinstance(value, np.generic):
        return _json_safe_plotly_value(value.item())

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime()

    if isinstance(value, datetime):
        return value

    if isinstance(value, date):
        return value

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def sanitize_plotly_figure_for_streamlit(fig: Optional[go.Figure]) -> Optional[go.Figure]:
    """Nettoie une figure Plotly avant st.plotly_chart pour éviter les erreurs NAType/orjson."""
    if fig is None:
        return fig
    try:
        return go.Figure(_json_safe_plotly_value(fig.to_plotly_json()))
    except Exception:
        return fig

# =========================
# SECTION: VISUALISATIONS STREAMLIT/PLOTLY
# =========================
def st_plot(fig, key=None, height=None, stretch=True, annotate_values: Optional[bool] = None):
    """Affiche une figure Plotly de manière robuste et compatible Streamlit ≥ 1.31.

    - Remplace use_container_width (déprécié) par width
    - width='stretch'  -> pleine largeur
    - width='content'  -> largeur naturelle
    - N'envoie jamais height=None
    - Évite les dépendances implicites à globals()
    """
    if fig is None:
        st.info("Aucune visualisation disponible : données absentes ou variables requises manquantes.")
        return

    if annotate_values is None:
        annotate_values = get_toggle_flag("annot_vals", False)

    try:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=[COLOR_CASES, COLOR_DEATHS, COLOR_CFR, COLOR_MASCULIN, COLOR_FEMININ, "#3e8b5a", "#d97b16"],
            font=dict(color="#20344f", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=30, t=70, b=50),
        )
        fig.update_xaxes(showgrid=False, linecolor="rgba(9,37,79,0.10)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(9,37,79,0.08)", zeroline=False)
    except Exception:
        pass

    fig = _apply_compact_bar_spacing(fig, bargap=0.0, bargroupgap=0.0)
    fig = apply_plotly_value_annotations(fig, bool(annotate_values))
    fig = sanitize_plotly_figure_for_streamlit(fig)

    kwargs = {}

    # ✅ Nouveau standard Streamlit
    kwargs["width"] = "stretch" if stretch else "content"

    if key is not None:
        kwargs["key"] = key

    if height is not None:
        kwargs["height"] = height

    return st.plotly_chart(fig, **kwargs)


def render_section_title(section_number: int, title: str) -> None:
    """Affiche un titre de sous-section harmonisé dans les onglets compacts."""
    safe_title = str(title).upper()
    st.markdown(
        f"""
<div class="cousp-section-title">
  <span class="cousp-section-index">{int(section_number):02d}</span>
  <span>{safe_title}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_standards_note() -> None:
    """Affiche une note normative courte pour guider l'interprétation des indicateurs."""
    st.caption(
        "Lecture normative : les indicateurs présentés doivent être interprétés selon les définitions usuelles de la surveillance en santé publique, "
        "en tenant compte de la complétude, de la promptitude, de la qualité des données et du niveau de confirmation biologique disponible."
    )


def inject_professional_dashboard_css() -> None:
    """Applique une identité visuelle institutionnelle inspirée du tableau de bord COUSP."""
    st.markdown(
        """
<style>
    :root {
        --cousp-blue-dark: #0b2c63;
        --cousp-blue: #1553a1;
        --cousp-blue-soft: #eaf2ff;
        --cousp-green: #2d7d46;
        --cousp-green-soft: #eef8f1;
        --cousp-orange: #d97b16;
        --cousp-orange-soft: #fff4e8;
        --cousp-red: #b9353f;
        --cousp-slate: #44536a;
        --cousp-border: rgba(18, 53, 106, 0.10);
        --cousp-shadow: 0 14px 30px rgba(11, 44, 99, 0.10);
        --cousp-card-bg: rgba(255, 255, 255, 0.92);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(39, 107, 191, 0.12), transparent 30%),
            radial-gradient(circle at top right, rgba(45, 125, 70, 0.12), transparent 26%),
            linear-gradient(180deg, #f4f8fd 0%, #edf3f9 46%, #f6f8fc 100%);
    }

    .main .block-container {
        max-width: 1520px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f2f6fb 0%, #e8f0f7 100%);
        border-right: 1px solid rgba(11, 44, 99, 0.08);
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stCaption {
        color: #18365f;
    }

    .cousp-hero {
        position: relative;
        overflow: hidden;
        padding: 1.35rem 1.75rem;
        margin-bottom: 1.1rem;
        border-radius: 24px;
        color: #ffffff;
        background:
            linear-gradient(120deg, rgba(255,255,255,0.10), rgba(255,255,255,0.02)),
            linear-gradient(90deg, #072654 0%, #0f438f 56%, #2c75c8 100%);
        box-shadow: 0 18px 40px rgba(7, 38, 84, 0.24);
        border: 1px solid rgba(255,255,255,0.18);
    }

    .cousp-hero::before,
    .cousp-hero::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        background: rgba(255,255,255,0.10);
        filter: blur(6px);
    }

    .cousp-hero::before {
        width: 220px;
        height: 220px;
        top: -130px;
        right: -30px;
    }

    .cousp-hero::after {
        width: 160px;
        height: 160px;
        bottom: -90px;
        left: -20px;
    }

    .cousp-hero-badge {
        display: inline-block;
        padding: 0.28rem 0.7rem;
        margin-bottom: 0.8rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.16);
        font-size: 0.82rem;
        letter-spacing: 0.14em;
        font-weight: 700;
    }

    .cousp-hero h1 {
        margin: 0;
        font-size: clamp(1.6rem, 2.8vw, 2.25rem);
        line-height: 1.15;
        letter-spacing: 0.03em;
        font-weight: 800;
    }

    .cousp-hero p {
        margin: 0.45rem 0 0;
        font-size: 1rem;
        opacity: 0.94;
        font-weight: 500;
    }

    .cousp-context-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 0.8rem;
        margin: 0.8rem 0 1rem;
    }

    .cousp-context-chip {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid var(--cousp-border);
        box-shadow: var(--cousp-shadow);
        border-radius: 18px;
        padding: 0.85rem 1rem;
    }

    .cousp-context-chip .label {
        color: var(--cousp-slate);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .cousp-context-chip .value {
        color: var(--cousp-blue-dark);
        font-size: 1rem;
        font-weight: 800;
        margin-top: 0.28rem;
    }

    .cousp-kpi-card {
        min-height: 148px;
        border-radius: 20px;
        padding: 1rem 1.05rem;
        color: #ffffff;
        position: relative;
        overflow: hidden;
        box-shadow: 0 16px 34px rgba(13, 46, 93, 0.18);
        height: 100%;
    }

    .cousp-kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(205px, 1fr));
        gap: 0.9rem;
        margin-bottom: 0.75rem;
        align-items: stretch;
    }

    .cousp-kpi-grid .cousp-kpi-card.span-2 {
        grid-column: span 2;
    }

    .cousp-kpi-card::after {
        content: "";
        position: absolute;
        inset: auto -20px -25px auto;
        width: 86px;
        height: 86px;
        border-radius: 50%;
        background: rgba(255,255,255,0.10);
    }

    .cousp-kpi-card.blue { background: linear-gradient(160deg, #2f76d2 0%, #164fa7 100%); }
    .cousp-kpi-card.navy { background: linear-gradient(160deg, #314468 0%, #232b44 100%); }
    .cousp-kpi-card.orange { background: linear-gradient(160deg, #f09b39 0%, #d67310 100%); }
    .cousp-kpi-card.green { background: linear-gradient(160deg, #4e9864 0%, #2f6f44 100%); }

    .cousp-kpi-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        opacity: 0.90;
        margin-bottom: 0.35rem;
    }

    .cousp-kpi-value {
        font-size: clamp(1.6rem, 2vw, 2.35rem);
        line-height: 1.05;
        font-weight: 800;
        margin-bottom: 0.35rem;
        word-break: break-word;
    }

    .cousp-kpi-subtitle {
        font-size: 0.92rem;
        font-weight: 500;
        opacity: 0.95;
        word-break: break-word;
    }

    @media (max-width: 1280px) {
        .cousp-kpi-grid {
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        }

        .cousp-kpi-grid .cousp-kpi-card.span-2 {
            grid-column: span 1;
        }
    }

    .cousp-panel-title {
        margin: 0.35rem 0 0.55rem;
        color: var(--cousp-blue-dark);
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
    }

    .cousp-panel-subtitle {
        margin: -0.15rem 0 0.6rem;
        color: var(--cousp-slate);
        font-size: 0.9rem;
    }

    .cousp-summary-box {
        background: var(--cousp-card-bg);
        border-radius: 22px;
        border: 1px solid var(--cousp-border);
        box-shadow: var(--cousp-shadow);
        padding: 1.05rem 1.1rem;
        margin-bottom: 0.7rem;
    }

    .cousp-summary-box .summary-lead {
        color: var(--cousp-blue-dark);
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .cousp-summary-box ul {
        margin: 0;
        padding-left: 1rem;
        color: #2a3c57;
    }

    .cousp-summary-box li {
        margin-bottom: 0.35rem;
    }

    .cousp-section-title {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin: 0.4rem 0 0.75rem;
        color: var(--cousp-blue-dark);
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .cousp-section-index {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 12px;
        background: linear-gradient(160deg, #eff4fd 0%, #dfeaf9 100%);
        border: 1px solid rgba(21, 83, 161, 0.16);
        color: var(--cousp-blue);
        font-size: 0.88rem;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.45);
    }

    div[data-testid="stPlotlyChart"],
    div[data-testid="stImage"],
    div[data-testid="stDataFrame"],
    div[data-testid="stMetric"] {
        background: var(--cousp-card-bg);
        border: 1px solid var(--cousp-border);
        border-radius: 22px;
        box-shadow: var(--cousp-shadow);
    }

    div[data-testid="stPlotlyChart"] {
        padding: 0.4rem 0.5rem 0.2rem;
    }

    div[data-testid="stImage"],
    div[data-testid="stDataFrame"] {
        padding: 0.55rem;
    }

    div[data-testid="stMetric"] {
        padding: 0.9rem 1rem;
    }

    div[data-testid="stMetric"] label {
        color: var(--cousp-slate);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.7rem;
    }

    div[data-testid="stTabs"] button[role="tab"] {
        border-radius: 14px 14px 0 0;
        padding: 0.85rem 1rem;
        margin-right: 0.22rem;
        background: rgba(255,255,255,0.74);
        border: 1px solid rgba(21, 83, 161, 0.10);
        color: var(--cousp-blue-dark);
        font-weight: 700;
    }

    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #ffffff;
        background: linear-gradient(90deg, #11448d 0%, #2b74ca 100%);
        border-color: transparent;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        position: sticky;
        top: 0.5rem;
        z-index: 20;
        padding: 0.35rem 0.4rem 0;
        margin-bottom: 0.55rem;
        background: linear-gradient(
            180deg,
            rgba(246, 249, 255, 0.98) 0%,
            rgba(246, 249, 255, 0.94) 80%,
            rgba(246, 249, 255, 0.00) 100%
        );
        backdrop-filter: blur(6px);
    }

    div[data-testid="stTabs"] [data-baseweb="tab-panel"] {
        padding-top: 0.2rem;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--cousp-border);
        border-radius: 18px;
        background: rgba(255,255,255,0.82);
        box-shadow: var(--cousp-shadow);
    }

    div[data-testid="stExpander"] summary {
        min-height: 54px;
        padding-top: 0.2rem;
        padding-bottom: 0.2rem;
    }

    div[data-testid="stExpander"] summary p {
        color: var(--cousp-blue-dark);
        font-size: 0.88rem;
        font-weight: 700;
        line-height: 1.2;
    }

    div[data-testid="stExpander"] details[open] {
        border-radius: 18px;
        border: 1px solid rgba(21, 83, 161, 0.16);
        background: rgba(255,255,255,0.96);
        box-shadow: 0 18px 38px rgba(11, 44, 99, 0.12);
    }

    div[data-testid="stExpander"] details[open] summary {
        border-bottom: 1px solid rgba(21, 83, 161, 0.10);
        background: linear-gradient(90deg, rgba(234,242,255,0.95) 0%, rgba(245,249,255,0.95) 100%);
        border-radius: 18px 18px 0 0;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] div[data-testid="stExpander"] {
        min-height: 62px;
        border-radius: 16px;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] div[data-testid="stExpander"] summary {
        min-height: 62px;
        padding-left: 0.55rem;
        padding-right: 0.55rem;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] div[data-testid="stExpander"] summary p {
        font-size: 0.80rem;
        text-transform: none;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] div[data-testid="stButton"] > button {
        min-height: 64px;
        border-radius: 18px;
        padding: 0.55rem 0.7rem;
        white-space: normal;
        line-height: 1.18;
        font-size: 0.80rem;
        font-weight: 700;
        box-shadow: var(--cousp-shadow);
        border: 1px solid rgba(21, 83, 161, 0.14);
        background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(243,247,253,0.98) 100%);
        color: var(--cousp-blue-dark);
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] div[data-testid="stButton"] > button:hover {
        border-color: rgba(21, 83, 161, 0.28);
        color: var(--cousp-blue);
        transform: translateY(-1px);
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {
        color: #ffffff;
        border-color: transparent;
        background: linear-gradient(120deg, #103d82 0%, #2369be 100%);
        box-shadow: 0 18px 34px rgba(16, 61, 130, 0.24);
    }

    .cousp-detail-empty {
        margin-top: 0.95rem;
        margin-bottom: 0.5rem;
        padding: 1rem 1.15rem;
        border-radius: 20px;
        border: 1px dashed rgba(21, 83, 161, 0.28);
        background: linear-gradient(180deg, rgba(247, 250, 255, 0.98) 0%, rgba(239, 245, 253, 0.95) 100%);
        color: #2b4264;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.55);
    }

    .cousp-detail-empty strong {
        display: block;
        margin-bottom: 0.22rem;
        color: var(--cousp-blue-dark);
        letter-spacing: 0.02em;
    }

    .cousp-footer {
        margin-top: 2rem;
        padding: 1.15rem 1.35rem;
        border-radius: 22px;
        color: #ffffff;
        background:
            linear-gradient(120deg, rgba(255,255,255,0.10), rgba(255,255,255,0.04)),
            linear-gradient(90deg, #09254f 0%, #0b3d7d 56%, #1e5fae 100%);
        box-shadow: 0 16px 36px rgba(8, 38, 78, 0.20);
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
    }

    .cousp-footer strong {
        display: block;
        margin-bottom: 0.18rem;
        font-size: 1rem;
        letter-spacing: 0.03em;
    }

    .cousp-footer span {
        font-size: 0.92rem;
        opacity: 0.92;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def render_professional_header() -> None:
    """Affiche le bandeau principal du tableau de bord."""
    st.markdown(
        """
<div class="cousp-hero">
  <div class="cousp-hero-badge">COUSP RDC</div>
  <h1>TABLEAU DE BORD DES INCIDENTS ÉPIDÉMIOLOGIQUES</h1>
  <p>Situation épidémiologique hebdomadaire - COUSP RDC</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Affiche un pied de page professionnel et institutionnel."""
    st.markdown(
        """
<div class="cousp-footer">
  <div>
    <strong>COUSP RDC</strong>
    <span>Surveillance épidémiologique nationale</span>
  </div>
  <div>
    <strong>Données fiables pour décisions rapides</strong>
    <span>Protection des communautés et pilotage opérationnel</span>
  </div>
  <div>
    <strong>Ministere de la Sante</strong>
    <span>Tous droits reserves</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def apply_plotly_value_annotations(fig: Optional[go.Figure], enabled: bool) -> Optional[go.Figure]:
    """Harmonise les couleurs et, si demandé, affiche les annotations sur les graphiques Plotly."""
    if fig is None:
        return fig

    try:
        for tr in fig.data:
            name = str(getattr(tr, "name", "") or "").strip()
            name_lower = name.lower()

            # Harmonisation des couleurs entre graphiques
            if isinstance(tr, go.Bar):
                if "létalit" in name_lower or "letalit" in name_lower or "cfr" in name_lower:
                    tr.marker.color = COLOR_CFR
                elif "déc" in name_lower or "dece" in name_lower or "deat" in name_lower:
                    tr.marker.color = COLOR_DEATHS
                elif name_lower in {"masculin", "male", "m"}:
                    tr.marker.color = COLOR_MASCULIN
                elif name_lower in {"feminin", "féminin", "female", "f"}:
                    tr.marker.color = COLOR_FEMININ
                elif "cas" in name_lower:
                    tr.marker.color = COLOR_CASES

                if enabled:
                    if tr.text is None:
                        tr.text = tr.y
                    tr.texttemplate = "%{text}"
                    tr.textposition = "outside"
                    tr.cliponaxis = False

            elif isinstance(tr, go.Histogram):
                if enabled:
                    tr.texttemplate = "%{y}"
                    tr.textposition = "outside"
                    tr.cliponaxis = False

            elif isinstance(tr, go.Scatter):
                if "létalit" in name_lower or "letalit" in name_lower or "cfr" in name_lower:
                    tr.line.color = COLOR_CFR
                    tr.marker.color = COLOR_CFR
                elif "déc" in name_lower or "dece" in name_lower or "deat" in name_lower:
                    tr.line.color = COLOR_DEATHS
                    tr.marker.color = COLOR_DEATHS
                elif name_lower in {"masculin", "male", "m"}:
                    tr.line.color = COLOR_MASCULIN
                    tr.marker.color = COLOR_MASCULIN
                elif name_lower in {"feminin", "féminin", "female", "f"}:
                    tr.line.color = COLOR_FEMININ
                    tr.marker.color = COLOR_FEMININ
                elif "cas" in name_lower:
                    tr.line.color = COLOR_CASES
                    tr.marker.color = COLOR_CASES

                if enabled:
                    if tr.y is None:
                        continue
                    mode = tr.mode or ""
                    if "text" not in mode:
                        tr.mode = (mode + "+text") if mode else "lines+markers+text"
                    if tr.text is None:
                        tr.text = tr.y
                    tr.textposition = "top center"

            elif isinstance(tr, go.Pie):
                labels = [str(x) for x in (tr.labels or [])]
                if labels:
                    pie_colors = [SEX_COLOR_MAP.get(lbl, None) for lbl in labels]
                    if any(c is not None for c in pie_colors):
                        tr.marker.colors = [c or COLOR_INCONNU for c in pie_colors]
                if enabled:
                    tr.textinfo = "label+percent+value"

    except Exception:
        return fig

    fig = _apply_compact_bar_spacing(fig, bargap=0.0, bargroupgap=0.0)
    return fig

def pick_age_col(df):
    """Choisir automatiquement la meilleure colonne tranche d’âge disponible."""
    if COL_AGEG2 in df.columns and df[COL_AGEG2].notna().any():
        return COL_AGEG2
    if COL_AGEG in df.columns and df[COL_AGEG].notna().any():
        return COL_AGEG
    return None

def ensure_lower(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.lower()
    return df

# =========================
# HELPERS (DATA CLEAN)
# =========================
def clean_str(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .str.strip()
         .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )

def norm_yesno(x):
    if x is None or pd.isna(x):
        return None
    try:
        numeric_x = float(x)
        if numeric_x == 1.0:
            return "Oui"
        if numeric_x == 0.0:
            return "Non"
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in ["oui", "o", "y", "yes", "1", "1.0", "true", "vrai"]:
        return "Oui"
    if s in ["non", "n", "no", "0", "0.0", "false", "faux"]:
        return "Non"
    return str(x).strip()

def is_positive(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip().lower()
    return "posit" in s or s in ["pos", "+"]

def is_death(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip().lower()
    keys = ["deces", "décès", "decede", "décédé", "décéder", "deceder", "mort", "death", "dead", "dcd","Décéder"]
    return any(k in s for k in keys)

def safe_to_datetime(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def make_yw(df):
    # clé Year-Week ex: 2026-W01
    if COL_YEAR in df.columns and COL_WNUM in df.columns:
        y = pd.to_numeric(df[COL_YEAR], errors="coerce").astype("Int64")
        w = pd.to_numeric(df[COL_WNUM], errors="coerce").astype("Int64")
        df["YW"] = y.astype("string") + "-W" + w.astype("string").str.zfill(2)
    else:
        df["YW"] = pd.NA
    return df

def taux_binaire(df, col, positive="Oui"):
    if col not in df.columns or len(df) == 0:
        return np.nan, 0
    s = df[col].astype("string")
    denom = int(s.notna().sum())
    if denom == 0:
        return np.nan, 0
    num = int((s == positive).sum())
    return safe_pct(num, denom), denom

def group_rate(df, group_col, indicator_col, positive_value):
    if group_col not in df.columns or indicator_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "n", "n_pos", "taux_%"])
    tmp = df[[group_col, indicator_col]].copy()
    tmp = tmp[tmp[group_col].notna()]
    tmp["is_pos"] = tmp[indicator_col].astype("string") == positive_value
    g = tmp.groupby(group_col, as_index=False).agg(
        n=(indicator_col, lambda x: int(x.notna().sum())),
        n_pos=("is_pos", "sum"),
    )
    g["taux_%"] = [safe_pct(num, den) for num, den in zip(g["n_pos"], g["n"])]
    return g.sort_values(group_col)

def group_cfr(df, group_col):
    if group_col not in df.columns or COL_ISSUE not in df.columns:
        return pd.DataFrame(columns=[group_col, "cas", "deces", "cfr_%"])
    tmp = df[[group_col, COL_ISSUE]].copy()
    tmp = tmp[tmp[group_col].notna()]
    tmp["is_death"] = tmp[COL_ISSUE].apply(is_death)
    g = tmp.groupby(group_col, as_index=False).agg(
        cas=(COL_ISSUE, "size"),
        deces=("is_death", "sum"),
    )
    g["cfr_%"] = [safe_pct(num, den) for num, den in zip(g["deces"], g["cas"])]
    return g.sort_values(group_col)



# =========================
# INDICATEURS – DÉFINITIONS COHÉRENTES (utilisés partout)
# =========================
def _norm_txt_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna(pd.NA)
    return s.str.strip().str.lower()

YES_SET = {"oui", "o", "y", "yes", "1", "true", "vrai"}

TDR_POS_SET = {"positif", "positive", "pos", "+", "tdr positif"}
TDR_NEG_SET = {"negatif", "négatif", "negative", "neg", "-", "tdr negatif", "tdr négatif"}

def _is_yes_series(s: pd.Series) -> pd.Series:
    return _norm_txt_series(s).isin(YES_SET)

def _normalize_lab_result_value(value) -> object:
    if value is None or pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return pd.NA

    norm = _strip_accents(text).lower().strip()
    norm = re.sub(r"[_\-]+", " ", norm)
    norm = re.sub(r"\s+", " ", norm)

    code = None
    if re.fullmatch(r"[1-5](?:\.0+)?", norm):
        code = norm[0]
    else:
        m = re.match(r"^([1-5])\b", norm)
        if m:
            code = m.group(1)

    if "posit" in norm or code == "1":
        return "positif"
    if "negat" in norm or code == "2":
        return "negatif"
    if any(token in norm for token in ["pending", "attente", "en cours"]) or code == "5":
        return "en attente"
    if any(token in norm for token in ["not tested", "non teste", "non realise", "not done"]) or code == "4":
        return "non teste"
    if any(token in norm for token in ["indet", "equivo", "douteux", "inconclu"]) or code == "3":
        return "indetermine"

    return norm

def _tdr_result_norm(s: pd.Series) -> pd.Series:
    return s.apply(_normalize_lab_result_value).astype("string")

def _coalesce_series_list(series_list: List[pd.Series], index) -> pd.Series:
    out = pd.Series(pd.NA, index=index, dtype="string")
    for series in series_list:
        if series is None:
            continue
        s = pd.Series(series, index=index).astype("string")
        out = out.combine_first(s)
    return out

def _build_specimen_type_from_columns(df: pd.DataFrame) -> pd.Series:
    specimen_cols = [
        ("Prelevement_sang", "Sang"),
        ("Prelevement_urine", "Urine"),
        ("Prelevement_respiratoire", "Respiratoire"),
    ]
    available = [col for col, _ in specimen_cols if col in df.columns]
    if not available and "Autre_prelevement" not in df.columns and "Precision_autre_prelevement" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")

    def _row_value(row: pd.Series):
        labels: List[str] = []
        for col, label in specimen_cols:
            if col in row.index and norm_yesno(row[col]) == "Oui":
                labels.append(label)

        other_raw = row.get("Precision_autre_prelevement")
        other_precision = "" if other_raw is None or pd.isna(other_raw) else str(other_raw).strip()
        other_yes = "Autre_prelevement" in row.index and norm_yesno(row.get("Autre_prelevement")) == "Oui"
        if other_yes or other_precision:
            labels.append(f"Autre ({other_precision})" if other_precision else "Autre")

        if not labels:
            return pd.NA
        return " + ".join(labels)

    return df.apply(_row_value, axis=1).astype("string")

def _enrich_rougeole_lab_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Nombre_doses_vaccin" in df.columns and "Nombre_dose_recues" not in df.columns:
        df["Nombre_dose_recues"] = df["Nombre_doses_vaccin"]

    if "Type_de_prelevement" not in df.columns:
        df["Type_de_prelevement"] = pd.Series(pd.NA, index=df.index, dtype="string")
    specimen_type = _build_specimen_type_from_columns(df)
    df["Type_de_prelevement"] = df["Type_de_prelevement"].astype("string").combine_first(specimen_type)

    result_candidates = []
    for col in [
        "TDR_Resultat",
        "Resultat_pcr_labo_national",
        "Resultat_machd_labo_national",
        "Resultat_igm",
        "Resultat_igm_rubeole",
        "Resultat_igm_rougeole_fievre_jaune_regional",
    ]:
        if col in df.columns:
            result_candidates.append(_tdr_result_norm(df[col]))

    lab_result = _coalesce_series_list(result_candidates, df.index)

    if "Resultat_labo" not in df.columns:
        df["Resultat_labo"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["Resultat_labo"] = df["Resultat_labo"].astype("string").combine_first(lab_result)

    if "TDR_Resultat" not in df.columns:
        df["TDR_Resultat"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["TDR_Resultat"] = _tdr_result_norm(df["TDR_Resultat"]).combine_first(lab_result)

    if "TDR_realise" not in df.columns:
        df["TDR_realise"] = pd.Series(pd.NA, index=df.index, dtype="string")
    tdr_realise = df["TDR_realise"].apply(norm_yesno).astype("string")
    result_done = df["Resultat_labo"].isin(["positif", "negatif", "indetermine"])
    result_not_done = df["Resultat_labo"].isin(["en attente", "non teste"])
    tdr_realise.loc[result_done & tdr_realise.isna()] = "Oui"
    tdr_realise.loc[result_not_done & tdr_realise.isna()] = "Non"
    df["TDR_realise"] = tdr_realise

    if "Prelevement" not in df.columns:
        df["Prelevement"] = pd.Series(pd.NA, index=df.index, dtype="string")
    prelevement = df["Prelevement"].apply(norm_yesno).astype("string")
    lab_evidence_masks = []
    for col in [
        "Prelevement_sang",
        "Prelevement_urine",
        "Prelevement_respiratoire",
        "Autre_prelevement",
    ]:
        if col in df.columns:
            lab_evidence_masks.append(df[col].apply(norm_yesno).eq("Oui"))
    for col in [
        "Date_prelevement",
        "Date_prelevement_urine",
        "Date_prelevement_respiratoire",
        "Date_autre_prelevement",
        "Date_reception_echantillon_labo",
        "Date_envoi_resultat",
        "Date_partage_resultat_pcr",
        "Date_partage_resultat_tdr_surveillance_epi",
        "Date_partage_resultat_machd_surveillance_epi",
        "Resultat_igm",
        "Resultat_igm_rubeole",
        "Resultat_pcr_labo_national",
        "Resultat_machd_labo_national",
    ]:
        if col in df.columns:
            lab_evidence_masks.append(df[col].notna())

    if lab_evidence_masks:
        prelevement_evidence = pd.concat(lab_evidence_masks, axis=1).any(axis=1)
        prelevement.loc[prelevement_evidence & prelevement.isna()] = "Oui"
    df["Prelevement"] = prelevement

    return df

# =========================
# SECTION: INDICATEURS KPI
# =========================
@st.cache_data(show_spinner=False)
def compute_indicators(df_in: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcule les KPI avec des dénominateurs cohérents (= nombre de cas filtrés).

    Définitions:
    - CFR% = décès / tous cas filtrés
    - Prélèvement% = (Prelevement == Oui) / tous cas filtrés
    - Hospitalisation% = (Hospitalisation == Oui) / tous cas filtrés
    - TDR_réalisé% = (TDR_realise == Oui) / tous cas filtrés
    - Couverture TDR% = identique à TDR_réalisé% mais renvoyée explicitement (num/den)
    - Positivité TDR% = positifs / (positifs + négatifs) parmi:
        (TDR_realise == Oui) ET (TDR_Resultat ∈ {pos, neg})
      => on exclut invalide, non prélevé, en attente, etc.
    """

    df = df_in.copy()
    n_cases = int(len(df))

    # -----------------------------
    # Décès & CFR (sur tous cas)
    # -----------------------------
    n_deaths = int(df["is_death"].sum()) if "is_death" in df.columns else 0
    cfr_pct = safe_pct(n_deaths, n_cases)

    # -----------------------------
    # Helper taux binaire Oui/Non (den = n_cases)
    # -----------------------------
    def _rate_yes(col_name: str) -> Tuple[float, int, int]:
        """Retourne (taux%, num_oui, denom_cases)."""
        if col_name not in df.columns or n_cases == 0:
            return (np.nan, 0, n_cases)
        num = int(_is_yes_series(df[col_name]).sum())
        return (num / n_cases * 100.0, num, n_cases)

    prelev_pct, n_prelev_yes, den_cases = _rate_yes(COL_PREL)
    hosp_pct, n_hosp_yes, _ = _rate_yes(COL_HOSP)
    tdr_pct, n_tdr_yes, _ = _rate_yes(COL_TDR)
    if (
        ("Resultat_labo" in df.columns)
        and (n_cases > 0)
        and (
            pd.isna(tdr_pct)
            or (COL_TDR not in df.columns)
            or (not df[COL_TDR].notna().any())
        )
    ):
        n_tdr_yes = int(df["Resultat_labo"].notna().sum())
        tdr_pct = safe_pct(n_tdr_yes, n_cases)

    # -----------------------------
    # Couverture TDR (explicite)
    # -----------------------------
    # (équivalent de tdr_pct, mais renvoyé avec num/den dédiés)
    tdr_coverage_num = n_tdr_yes
    tdr_coverage_den = den_cases
    tdr_coverage_pct = tdr_pct

    # -----------------------------
    # Positivité TDR (sur TDR réalisés + résultats valides)
    # -----------------------------
    result_col = None
    if COL_TDRR in df.columns and df[COL_TDRR].notna().any():
        result_col = COL_TDRR
    elif "Resultat_labo" in df.columns and df["Resultat_labo"].notna().any():
        result_col = "Resultat_labo"

    if (result_col is not None) and (n_cases > 0):
        tdr_yes = _is_yes_series(df[COL_TDR]) if (COL_TDR in df.columns and df[COL_TDR].notna().any()) else pd.Series(True, index=df.index)
        res_n = _tdr_result_norm(df[result_col])

        # Résultats valides = pos/neg uniquement
        valid_res = res_n.isin(TDR_POS_SET.union(TDR_NEG_SET))

        # Dénominateur positivité = TDR=Oui & (pos/neg)
        pos_den = int((tdr_yes & valid_res).sum())

        # Numérateur = TDR=Oui & positif
        pos_num = int((tdr_yes & res_n.isin(TDR_POS_SET)).sum())

        pos_pct = safe_pct(pos_num, pos_den)
    else:
        pos_den, pos_num, pos_pct = 0, 0, np.nan

    # -----------------------------
    # Taux d'invalides (utile pour qualité test)
    # -----------------------------
    # On le calcule parmi TDR réalisés (=Oui). On inclut "invalide", "inba", etc.
    invalid_num = 0
    invalid_den = 0
    invalid_pct = np.nan

    if (result_col is not None) and (n_cases > 0):
        tdr_yes = _is_yes_series(df[COL_TDR]) if (COL_TDR in df.columns and df[COL_TDR].notna().any()) else pd.Series(True, index=df.index)
        res_n = _tdr_result_norm(df[result_col])

        # Définition "invalide" (ajuste au besoin)
        invalid_set = {"invalide", "invalid", "inba", "bande absente"}

        invalid_den = int(tdr_yes.sum())
        invalid_num = int((tdr_yes & res_n.isin(invalid_set)).sum())
        invalid_pct = safe_pct(invalid_num, invalid_den)

    # -----------------------------
    # Degré déshydratation (table d'effectifs)
    # -----------------------------
    if COL_DEHY in df.columns and n_cases > 0:
        dehy_tbl = (
            df[COL_DEHY]
            .astype("string")
            .fillna(MISSING_LABEL_VERBOSE)
            .str.strip()
            .replace({"": MISSING_LABEL_VERBOSE, "inconnu/non renseigne": MISSING_LABEL_VERBOSE,"inconnu": MISSING_LABEL_VERBOSE})
            .value_counts(dropna=False)
            .rename_axis(COL_DEHY)
            .reset_index(name="Nombre_de_cas")
        )
    else:
        dehy_tbl = pd.DataFrame(columns=[COL_DEHY, "Nombre_de_cas"])

    return {
        "n_cases": n_cases,
        "n_deaths": n_deaths,
        "cfr_pct": cfr_pct,

        "prelev_pct": prelev_pct,
        "prelev_num": n_prelev_yes,
        "prelev_den": den_cases,

        "hosp_pct": hosp_pct,
        "hosp_num": n_hosp_yes,
        "hosp_den": den_cases,

        "tdr_pct": tdr_pct,          # % TDR=Oui sur tous cas
        "tdr_num": n_tdr_yes,
        "tdr_den": den_cases,

        # Couverture TDR (explicite)
        "tdr_coverage_pct": tdr_coverage_pct,
        "tdr_coverage_num": tdr_coverage_num,
        "tdr_coverage_den": tdr_coverage_den,

        # Positivité (sur résultats valides)
        "pos_pct": pos_pct,
        "pos_num": pos_num,
        "pos_den": pos_den,

        # Invalides (qualité)
        "invalid_pct": invalid_pct,
        "invalid_num": invalid_num,
        "invalid_den": invalid_den,

        "dehy_tbl": dehy_tbl,
    }

@st.cache_data(show_spinner=False)
def compute_group_indicators(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Table d'indicateurs par groupe avec les mêmes définitions/denoms."""
    if df_in is None or df_in.empty or group_col not in df_in.columns:
        return pd.DataFrame(columns=[group_col, "Cas", "Décès", "CFR_%", "Prélèvement_%", "Hospitalisation_%", "TDR_réalisé_%", "Positivité_TDR_%"])

    df = df_in.copy()
    df = df[df[group_col].notna()]

    result_col = None
    if COL_TDRR in df.columns and df[COL_TDRR].notna().any():
        result_col = COL_TDRR
    elif "Resultat_labo" in df.columns and df["Resultat_labo"].notna().any():
        result_col = "Resultat_labo"
    has_tdr_chain = COL_TDR in df.columns and df[COL_TDR].notna().any()

    # Base (cas, décès)
    g = df.groupby(group_col, as_index=False).agg(
        Cas=(group_col, "size"),
        Décès=("is_death", "sum") if "is_death" in df.columns else (group_col, "size"),
    )
    if "is_death" not in df.columns:
        g["Décès"] = 0

    g["CFR_%"] = [safe_pct(num, den) for num, den in zip(g["Décès"], g["Cas"])]

    def _add_rate(col, new_name, yes_mask=None):
        if yes_mask is not None:
            tmp = df[[group_col]].copy()
            tmp["is_yes"] = yes_mask.astype(bool)
        else:
            if col not in df.columns:
                g[new_name] = np.nan
                return
            tmp = df[[group_col, col]].copy()
            tmp["is_yes"] = _is_yes_series(tmp[col])
        num = tmp.groupby(group_col)["is_yes"].sum()
        den = tmp.groupby(group_col).size()
        g[new_name] = [safe_pct(n, d) for n, d in zip(num.reindex(g[group_col]).fillna(0), den.reindex(g[group_col]).fillna(0))]

    _add_rate(COL_PREL, "Prélèvement_%")
    _add_rate(COL_HOSP, "Hospitalisation_%")
    if has_tdr_chain:
        _add_rate(COL_TDR, "TDR_réalisé_%")
    elif result_col == "Resultat_labo":
        _add_rate(None, "TDR_réalisé_%", yes_mask=df["Resultat_labo"].notna())
    else:
        g["TDR_réalisé_%"] = np.nan

    # Positivité (parmi TDR=Oui + résultat valide)
    if result_col is not None:
        tdr_yes = _is_yes_series(df[COL_TDR]) if has_tdr_chain else pd.Series(True, index=df.index)
        res_n = _tdr_result_norm(df[result_col])
        valid_res = res_n.isin(TDR_POS_SET.union(TDR_NEG_SET))
        df_pos = df[[group_col]].copy()
        df_pos["den_pos"] = (tdr_yes & valid_res).astype(int)
        df_pos["num_pos"] = (tdr_yes & res_n.isin(TDR_POS_SET)).astype(int)
        sums = df_pos.groupby(group_col, as_index=False).agg(den_pos=("den_pos", "sum"), num_pos=("num_pos", "sum"))
        g = g.merge(sums, on=group_col, how="left")
        g["Positivité_TDR_%"] = [safe_pct(n, d) for n, d in zip(g["num_pos"].fillna(0), g["den_pos"].fillna(0))]
        g = g.drop(columns=["den_pos", "num_pos"])
    else:
        g["Positivité_TDR_%"] = np.nan

    return g.sort_values(group_col)


def delay_days(df, date_end, date_start, new_col):
    if date_end in df.columns and date_start in df.columns:
        df[new_col] = (df[date_end] - df[date_start]).dt.days
    else:
        df[new_col] = np.nan
    return df

def pct_under_threshold(series, threshold=2):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return np.nan, 0
    n = len(series)
    under = int((series <= threshold).sum())
    return safe_pct(under, n), n

def compile_from_folder(folder, pattern, sheet=None):
    files = sorted(glob.glob(str(Path(folder) / pattern)))
    if not files:
        return pd.DataFrame(), []
    dfs = []
    for f in files:
        try:
            d = pd.read_excel(f, sheet_name=sheet) if sheet else pd.read_excel(f)
            d["__source_file__"] = os.path.basename(f)
            dfs.append(d)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(), files
    return pd.concat(dfs, ignore_index=True), files

@st.cache_data(show_spinner=False)
def load_data_from_excel(path):
    return pd.read_excel(path)

@st.cache_data(show_spinner=False)
def standardize_df(df):
    df = df.copy()

    # Strings
    for c in [COL_PROV, COL_ZS, COL_AS, COL_SEX, COL_ISSUE, COL_CLASS, COL_TDRR, COL_AGEG, COL_AGEG2, COL_DEHY]:
        if c in df.columns:
            df[c] = clean_str(df[c])

    # Yes/No
    for c in [COL_PREL, COL_TDR, COL_HOSP]:
        if c in df.columns:
            df[c] = df[c].apply(norm_yesno)

    # Dates
    date_candidates = [DATE_ONSET, DATE_NOTIF, DATE_ADM, DATE_PREL, DATE_CONS, DATE_INV, DATE_RES, DATE_RECEP, DATE_ISSUE, "Date_sortie_au_CT", "Date_confirmation", "Date_de_guerie"]
    df = safe_to_datetime(df, [c for c in date_candidates if c in df.columns])

    # Year/week
    df = make_yw(df)

    # décès bool
    df["is_death"] = df[COL_ISSUE].apply(is_death) if COL_ISSUE in df.columns else False

    # positivité
    if COL_TDRR in df.columns:
        df["is_tdr_pos"] = df[COL_TDRR].apply(is_positive)
    elif "Resultat_labo" in df.columns:
        df["is_tdr_pos"] = df["Resultat_labo"].apply(is_positive)
    else:
        df["is_tdr_pos"] = False

    # harmonisations standards
    if COL_CLASS in df.columns:
        class_norm = (
            df[COL_CLASS].astype("string").str.strip().str.lower()
            .apply(lambda v: _strip_accents(v) if pd.notna(v) else v)
        )
        df["Classification_finale_std"] = (
            class_norm
            .map({
                "confirme": "Confirmé", "confirmé": "Confirmé", "confirme par labo": "Confirmé",
                "probable": "Probable", "suspect": "Suspect", "compatible": "Compatible",
                "non cas": "Non cas", "non_cas": "Non cas", "discarded": "Non cas"
            })
            .fillna(df[COL_CLASS])
        )
        df["Classification_finale_std"] = df["Classification_finale_std"].replace({
            "confirmee": "Confirm\u00e9",
            "confirme au labo": "Confirm\u00e9",
            "positif": "Confirm\u00e9",
            "cas suspect": "Suspect",
        })
    if COL_ISSUE in df.columns:
        issue_norm = (
            df[COL_ISSUE].astype("string").str.strip().str.lower()
            .apply(lambda v: _strip_accents(v) if pd.notna(v) else v)
        )
        df["Issue_std"] = (
            issue_norm
            .map({
                "decede": "Décédé", "décédé": "Décédé", "deces": "Décédé", "décès": "Décédé",
                "gueri": "Guéri", "guéri": "Guéri", "sorti gueri": "Guéri",
                "en traitement": "En traitement", "transfere": "Transféré", "transféré": "Transféré"
            })
            .fillna(df[COL_ISSUE])
        )
        df["Issue_std"] = df["Issue_std"].replace({
            "deceder": "D\u00e9c\u00e9d\u00e9",
            "mort": "D\u00e9c\u00e9d\u00e9",
            "en cours": "En traitement",
            "traiter": "En traitement",
            "traite": "En traitement",
            "sorti": "Sorti",
        })

    # indicateurs standards
    df["preleve_oui_non"] = _is_yes_series(df[COL_PREL]) if COL_PREL in df.columns else False
    df["tdr_realise_oui_non"] = _is_yes_series(df[COL_TDR]) if COL_TDR in df.columns else False
    df["hospitalise_oui_non"] = _is_yes_series(df[COL_HOSP]) if COL_HOSP in df.columns else False
    if "Resultat_labo" in df.columns:
        df["confirme_labo_oui_non"] = df["Resultat_labo"].apply(is_positive)
    else:
        df["confirme_labo_oui_non"] = df["is_tdr_pos"]

    # délais standards
    df = delay_days(df, DATE_CONS, DATE_ONSET, "delai_onset_to_consult")
    df = delay_days(df, DATE_NOTIF, DATE_ONSET, "delai_onset_to_notif")
    df = delay_days(df, DATE_ADM, DATE_ONSET, "delai_onset_to_adm")
    df = delay_days(df, DATE_PREL, DATE_ONSET, "delai_onset_to_prel")
    df = delay_days(df, DATE_RES, DATE_PREL, "delai_prel_to_result")
    df = delay_days(df, DATE_INV, DATE_NOTIF, "delai_notif_to_invest")
    df = delay_days(df, DATE_ISSUE, DATE_ADM, "delai_adm_to_issue")

    # alias explicites demandés pour les analyses standards
    delay_aliases = {
        "delai_onset_to_consult": "Delai_debut_consultation_j",
        "delai_onset_to_notif": "Delai_debut_notification_j",
        "delai_onset_to_adm": "Delai_debut_admission_j",
        "delai_onset_to_prel": "Delai_debut_prelevement_j",
        "delai_prel_to_result": "Delai_prelevement_resultat_j",
        "delai_notif_to_invest": "Delai_notification_investigation_j",
        "delai_adm_to_issue": "Delai_admission_issue_j",
    }
    for src, dst in delay_aliases.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")

    # validations utiles pour dashboard
    if "Age_en_ans" in df.columns:
        agey = pd.to_numeric(df["Age_en_ans"], errors="coerce")
    elif COL_AGE in df.columns:
        agey = pd.to_numeric(df[COL_AGE], errors="coerce")
    else:
        agey = pd.Series(np.nan, index=df.index)
    df["age_valid"] = agey.between(0, 120, inclusive="both")

    if COL_SEX in df.columns:
        sx = df[COL_SEX].astype("string").str.strip().str.lower()
        sx = sx.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
        df["sexe_valid"] = sx.isin(["m", "masculin", "male", "homme", "h", "f", "feminin", "female", "femme"])
    else:
        df["sexe_valid"] = False

    df["geo_valid"] = True
    if COL_ZS in df.columns and COL_PROV in df.columns:
        df["geo_valid"] = ~(df[COL_ZS].notna() & df[COL_PROV].isna())
    if COL_AS in df.columns and COL_ZS in df.columns:
        df["geo_valid"] = df["geo_valid"] & ~(df[COL_AS].notna() & df[COL_ZS].isna())

    # flags de qualité explicites
    df["missing_parent_geo_flag"] = ~df["geo_valid"]

    df["chrono_valid"] = True
    for c in ["delai_onset_to_consult", "delai_onset_to_notif", "delai_onset_to_adm", "delai_onset_to_prel", "delai_prel_to_result", "delai_notif_to_invest", "delai_adm_to_issue"]:
        if c in df.columns:
            df["chrono_valid"] = df["chrono_valid"] & (pd.to_numeric(df[c], errors="coerce").fillna(0) >= 0)
    df["chronologie_invalide"] = ~df["chrono_valid"]
    df["age_hors_limites"] = ~pd.Series(df["age_valid"], index=df.index).fillna(False)

    if COL_ISSUE in df.columns and DATE_ISSUE in df.columns:
        df["deces_sans_date_issue"] = df[COL_ISSUE].apply(is_death) & df[DATE_ISSUE].isna()
    else:
        df["deces_sans_date_issue"] = False

    if (COL_PREL in df.columns) and (COL_TDR in df.columns):
        prelev_non = df[COL_PREL].astype("string").str.strip().str.lower().eq("non")
        tdr_oui = df[COL_TDR].astype("string").str.strip().str.lower().eq("oui")
        df["prelev_tdr_incoherent"] = prelev_non & tdr_oui
    else:
        df["prelev_tdr_incoherent"] = False

    # clé de doublon potentiel
    dup_parts = []
    for c in ["Nom_complet", COL_SEX, "Age_en_ans", DATE_ONSET, COL_PROV, COL_ZS]:
        if c in df.columns:
            ser = df[c]
            if "Date" in c:
                ser = pd.to_datetime(ser, errors="coerce").dt.strftime("%Y-%m-%d")
            dup_parts.append(ser.astype("string").fillna(""))
    if dup_parts:
        fp = pd.concat(dup_parts, axis=1).agg("|".join, axis=1)
        fp = fp.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
        df["duplicate_fingerprint"] = fp
        df["duplicate_potential"] = fp.duplicated(keep=False) & fp.ne("")
    else:
        df["duplicate_fingerprint"] = pd.NA
        df["duplicate_potential"] = False

    # alias booléens explicites pour le bloc qualité
    df["doublons_potentiels"] = df["duplicate_potential"]

    # score de complétude simple (champs clés)
    core_fields = [c for c in [COL_PROV, COL_ZS, COL_AS, COL_SEX, "Age_en_ans", DATE_ONSET, DATE_NOTIF, COL_PREL, COL_ISSUE, COL_CLASS] if c in df.columns]
    if core_fields:
        df["score_completude_core_%"] = df[core_fields].notna().mean(axis=1).mul(100).round(1)
    else:
        df["score_completude_core_%"] = np.nan

    return df

# =========================
# HELPERS (Qualité & Alertes)
# =========================
# =========================
# SECTION: QUALITE DES DONNEES
# =========================
@st.cache_data(show_spinner=False)
def qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne un tableau long des incohérences (1 ligne = 1 flag = 1 cas)."""
    out = []

    def _add(mask, flag_name):
        if mask is None:
            return
        idx = df.index[mask.fillna(False)].tolist() if hasattr(mask, "fillna") else df.index[mask].tolist()
        if len(idx):
            out.extend([{"row_id": int(i), "flag": flag_name} for i in idx])

    for c, lab in [
        ("delai_onset_to_adm", "Date admission < début maladie"),
        ("delai_onset_to_prel", "Date prélèvement < début maladie"),
        ("delai_onset_to_consult", "Date consultation < début maladie"),
        ("delai_onset_to_notif", "Date notification < début maladie"),
        ("delai_adm_to_issue", "Date issue < admission"),
        ("delai_prel_to_result", "Date résultat < prélèvement"),
        ("delai_notif_to_invest", "Date investigation < notification"),
    ]:
        if c in df.columns:
            _add(pd.to_numeric(df[c], errors="coerce") < 0, lab)

    if (COL_TDR in df.columns) and (COL_TDRR in df.columns):
        _add((df[COL_TDR].astype("string") == "Non") & (df[COL_TDRR].notna()), "TDR=Non mais résultat renseigné")
        _add((df[COL_TDR].isna()) & (df[COL_TDRR].notna()), "Résultat TDR sans statut TDR")

    if (COL_PREL in df.columns) and ("Resultat_labo" in df.columns):
        _add((df[COL_PREL].astype("string") == "Non") & (df["Resultat_labo"].notna()), "Prélèvement=Non mais résultat labo renseigné")
    if (COL_PREL in df.columns) and (DATE_PREL in df.columns):
        _add((df[COL_PREL].astype("string") == "Non") & (df[DATE_PREL].notna()), "Prélèvement=Non mais date prélèvement renseignée")

    if ("Femme_enceinte" in df.columns) and (COL_SEX in df.columns):
        s_sex = df[COL_SEX].astype("string").str.lower()
        s_preg = df["Femme_enceinte"].astype("string").str.lower()
        _add((s_preg == "oui") & (~s_sex.str.contains("fem")), "Femme_enceinte=Oui mais sexe ≠ féminin")

    if (COL_AGE in df.columns):
        age_num = pd.to_numeric(df[COL_AGE], errors="coerce")
        _add((age_num < 0) | (age_num > 120), "Âge hors limites (0–120)")
    if "Age_en_ans" in df.columns:
        age_num = pd.to_numeric(df["Age_en_ans"], errors="coerce")
        _add((age_num < 0) | (age_num > 120), "Âge_en_ans hors limites (0–120)")

    if (COL_ZS in df.columns) and (COL_PROV in df.columns):
        _add(df[COL_ZS].notna() & df[COL_PROV].isna(), "ZS renseignée mais province manquante")
    if (COL_AS in df.columns) and (COL_ZS in df.columns):
        _add(df[COL_AS].notna() & df[COL_ZS].isna(), "AS renseignée mais ZS manquante")

    if COL_ISSUE in df.columns and DATE_ISSUE in df.columns:
        _add(df[COL_ISSUE].apply(is_death) & df[DATE_ISSUE].isna(), "Décès sans date issue")

    if COL_ISSUE in df.columns:
        _add(df[COL_ISSUE].isna(), "Issue manquante")

    if "duplicate_potential" in df.columns:
        _add(df["duplicate_potential"] == True, "Doublon potentiel")

    if not out:
        return pd.DataFrame(columns=["row_id", "flag"])
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def completeness_table(df: pd.DataFrame, cols_required: list[str], by: str) -> pd.DataFrame:
    """Complétude (%) des champs clés par groupe. Robuste aux doublons de colonnes."""

    if df is None or df.empty:
        return pd.DataFrame(columns=[by, "n", "score_completude_%"])

    # ✅ Fix: colonnes dupliquées -> garde la première occurrence
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # 1) Colonnes existantes
    cols = [c for c in cols_required if c in df.columns]

    # 2) Évite que `by` soit dans `cols` (sinon doublon dans df[[by]+cols])
    cols = [c for c in cols if c != by]

    # 3) Dé-dup au cas où
    cols = list(dict.fromkeys(cols))

    if (by not in df.columns) or (len(cols) == 0):
        return pd.DataFrame(columns=[by, "n", "score_completude_%"])

    tmp = df[[by] + cols].copy()
    tmp = tmp[tmp[by].notna()]

    if tmp.empty:
        return pd.DataFrame(columns=[by, "n", "score_completude_%"])

    g = tmp.groupby(by, as_index=False).agg(n=(by, "size"))

    for c in cols:
        g[c] = tmp.groupby(by)[c].apply(lambda x: float(x.notna().mean() * 100)).values

    g["score_completude_%"] = g[cols].mean(axis=1).round(1)
    return g.sort_values("score_completude_%", ascending=True)

@st.cache_data(show_spinner=False)
def standard_data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Résumé standard de qualité des données sur les line lists filtrées."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Indicateur", "Valeur"])

    rows = [{"Indicateur": "Cas (n)", "Valeur": int(len(df))}]
    if "score_completude_core_%" in df.columns:
        ser = pd.to_numeric(df["score_completude_core_%"], errors="coerce")
        rows.append({"Indicateur": "Complétude médiane champs clés (%)", "Valeur": round(float(ser.median()), 1)})
        rows.append({"Indicateur": "Complétude moyenne champs clés (%)", "Valeur": round(float(ser.mean()), 1)})

    for col, label in [
        ("duplicate_potential", "Doublons potentiels (%)"),
        ("chronologie_invalide", "Chronologie invalide (%)"),
        ("chrono_valid", "Chronologie valide (%)"),
        ("age_hors_limites", "Âge hors limites (%)"),
        ("age_valid", "Âge valide (%)"),
        ("sexe_valid", "Sexe valide (%)"),
        ("missing_parent_geo_flag", "ZS/AS sans niveau supérieur (%)"),
        ("geo_valid", "Géographie valide (%)"),
        ("deces_sans_date_issue", "Décès sans date d’issue (%)"),
        ("prelev_tdr_incoherent", "Prélèvement/TDR incohérents (%)"),
        ("preleve_oui_non", "Prélèvement oui (%)"),
        ("tdr_realise_oui_non", "TDR réalisé oui (%)"),
        ("hospitalise_oui_non", "Hospitalisation oui (%)"),
        ("confirme_labo_oui_non", "Confirmation/positivité labo (%)"),
        ("is_death", "Décès (%)"),
    ]:
        if col in df.columns:
            rows.append({"Indicateur": label, "Valeur": round(float(pd.Series(df[col]).fillna(False).mean() * 100), 1)})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def duplicate_candidates_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "duplicate_fingerprint" not in df.columns or "duplicate_potential" not in df.columns:
        return pd.DataFrame()
    tmp = df[df["duplicate_potential"] == True].copy()
    if tmp.empty:
        return pd.DataFrame()
    grouped = tmp.groupby("duplicate_fingerprint", dropna=False)
    out = grouped.size().reset_index(name="Occurrences")
    if "Nom_complet" in tmp.columns:
        out["Nom_complet"] = grouped["Nom_complet"].apply(lambda x: " | ".join(pd.Series(x).dropna().astype(str).unique()[:3])).values
    if COL_PROV in tmp.columns:
        out[COL_PROV] = grouped[COL_PROV].apply(lambda x: " | ".join(pd.Series(x).dropna().astype(str).unique()[:3])).values
    if COL_ZS in tmp.columns:
        out[COL_ZS] = grouped[COL_ZS].apply(lambda x: " | ".join(pd.Series(x).dropna().astype(str).unique()[:3])).values
    return out.sort_values(["Occurrences"], ascending=False)

def build_standard_delay_summary(df: pd.DataFrame) -> pd.DataFrame:
    delay_map = {
        "delai_onset_to_consult": "Début → consultation",
        "delai_onset_to_notif": "Début → notification",
        "delai_onset_to_adm": "Début → admission",
        "delai_onset_to_prel": "Début → prélèvement",
        "delai_prel_to_result": "Prélèvement → résultat",
        "delai_notif_to_invest": "Notification → investigation",
        "delai_adm_to_issue": "Admission → issue",
    }
    rows = []
    for c, label in delay_map.items():
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            s = s[s >= 0].dropna()
            if len(s):
                rows.append({
                    "Type_delai": label,
                    "n": int(len(s)),
                    "Médiane_j": round(float(s.median()), 1),
                    "P25_j": round(float(s.quantile(0.25)), 1),
                    "P75_j": round(float(s.quantile(0.75)), 1),
                    "Min_j": round(float(s.min()), 1),
                    "Max_j": round(float(s.max()), 1),
                })
    return pd.DataFrame(rows)

STANDARD_DELAY_LABELS = {
    "delai_onset_to_consult": "Début -> consultation",
    "delai_onset_to_notif": "Début -> notification",
    "delai_onset_to_adm": "Début -> admission",
    "delai_onset_to_prel": "Début -> prélèvement",
    "delai_prel_to_result": "Prélèvement -> résultat",
    "delai_notif_to_invest": "Notification -> investigation",
    "delai_adm_to_issue": "Admission -> issue",
}

def list_available_standard_delays(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Retourne les délais standards exploitables dans le périmètre filtré."""
    available = []
    if df is None or df.empty:
        return available

    for col, label in STANDARD_DELAY_LABELS.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        s = s[s >= 0].dropna()
        if len(s):
            available.append((col, label))
    return available

@st.cache_data(show_spinner=False)
def build_delay_group_summary(df: pd.DataFrame, delay_col: str, group_col: str, threshold: float = 2) -> pd.DataFrame:
    """Resume un delai standard par groupe avec indicateurs robustes."""
    threshold_val = float(threshold)
    threshold_lab = int(threshold_val) if threshold_val.is_integer() else round(threshold_val, 1)
    pct_col = f"% <= {threshold_lab} j"
    out_cols = [group_col, "n", "Mediane_j", "P25_j", "P75_j", "Min_j", "Max_j", pct_col]

    if (
        df is None
        or df.empty
        or delay_col not in df.columns
        or group_col not in df.columns
    ):
        return pd.DataFrame(columns=out_cols)

    tmp = df[[group_col, delay_col]].copy()
    tmp[delay_col] = pd.to_numeric(tmp[delay_col], errors="coerce")
    tmp = tmp[tmp[group_col].notna() & tmp[delay_col].notna()].copy()
    tmp = tmp[tmp[delay_col] >= 0]

    if tmp.empty:
        return pd.DataFrame(columns=out_cols)

    grouped = (
        tmp.groupby(group_col, as_index=False)
        .agg(
            n=(delay_col, "size"),
            Mediane_j=(delay_col, "median"),
            P25_j=(delay_col, lambda x: x.quantile(0.25)),
            P75_j=(delay_col, lambda x: x.quantile(0.75)),
            Min_j=(delay_col, "min"),
            Max_j=(delay_col, "max"),
        )
    )
    pct_tbl = (
        tmp.groupby(group_col)[delay_col]
        .apply(lambda x: float((x <= threshold_val).mean() * 100.0))
        .reset_index(name=pct_col)
    )
    grouped = grouped.merge(pct_tbl, on=group_col, how="left")

    for c in ["Mediane_j", "P25_j", "P75_j", "Min_j", "Max_j", pct_col]:
        grouped[c] = pd.to_numeric(grouped[c], errors="coerce").round(1)

    return grouped.sort_values(["n", group_col], ascending=[False, True])

def build_recommended_fields_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matrice simple de disponibilité des champs standards recommandés."""
    field_groups = {
        "Identification": ["Nom_complet", "N_epid", "N", "N_labo"],
        "Géographie": [COL_PROV, COL_ZS, COL_AS],
        "Personne": [COL_SEX, "Age_en_ans", "Tranche_age"],
        "Temps": [DATE_ONSET, DATE_CONS, DATE_NOTIF, DATE_INV, DATE_ADM, DATE_PREL, DATE_RES, DATE_ISSUE],
        "Issue": [COL_ISSUE, DATE_ISSUE, "Date_sortie_au_CT"],
        "Labo": [COL_PREL, COL_TDR, COL_TDRR, "Resultat_labo", "Type_de_prelevement", "Nom_laboratoire", "Etat_echantillon"],
        "Vaccination": ["Statut_vaccinal", "Vaccin_precedemment", "Nombre_dose", "Nombre_dose_recues", "Date_derniere_vaccination"],
        "Lien épid / cluster": ["Lien_epid_avec_un_cas", "Cas_source_id", "Facteur_exposition", "Type_de_lien"],
    }
    rows = []
    for grp, cols in field_groups.items():
        for c in cols:
            rows.append({
                "Bloc": grp,
                "Variable": c,
                "Disponible": "Oui" if c in df.columns else "Non",
                "Complétude_%": round(float(df[c].notna().mean() * 100), 1) if c in df.columns else np.nan,
            })
    return pd.DataFrame(rows)

def cascade_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cascade prélèvement -> TDR -> résultat valide -> positif (sur les données filtrées).

    Améliorations vs version initiale :
    - Cascade "entonnoir" avec dénominateurs séquentiels (prélevé -> TDR -> résultat).
    - "Résultat disponible" = seulement résultats biologiques valides (positif/négatif),
      donc on exclut les statuts type "non réalisé/non prélevé", "en cours", etc.
    - Détection & quantification des incohérences (résultat renseigné mais TDR_realise != Oui).
    - Calcule le positif à partir de TDR_Resultat si is_tdr_pos absent.
    """

    n_all = int(len(df))

    # Colonnes (si absentes -> séries NA pour ne pas planter)
    prelev = df[COL_PREL].astype("string") if COL_PREL in df.columns else pd.Series([pd.NA] * n_all)
    tdr    = df[COL_TDR].astype("string")  if COL_TDR  in df.columns else pd.Series([pd.NA] * n_all)
    result_col = None
    if COL_TDRR in df.columns and df[COL_TDRR].notna().any():
        result_col = COL_TDRR
    elif "Resultat_labo" in df.columns and df["Resultat_labo"].notna().any():
        result_col = "Resultat_labo"
    tdr_res_raw = df[result_col].astype("string") if result_col is not None else pd.Series([pd.NA] * n_all)

    # Normalisation minimale (trim + lower) pour gérer variantes d’écriture
    def _norm(s: pd.Series) -> pd.Series:
        s = s.fillna(pd.NA).astype("string")
        return s.str.strip().str.lower()

    prelev_n = _norm(prelev)
    tdr_n    = _norm(tdr)
    res_n    = _norm(tdr_res_raw)

    # Valeurs "Oui" possibles (à élargir si nécessaire)
    YES = {"oui", "yes", "y", "1", "true", "vrai"}

    # Résultats biologiques valides (inclut variantes FR/EN)
    POS_SET = {"positif", "positive", "+"}
    NEG_SET = {"négatif", "negatif", "negative", "-"}

    # Statuts / non-résultats fréquents observés (à exclure du "résultat disponible")
    # ex: "non réalisé/non prélevé" est un statut, pas un résultat
    NON_RESULT_HINTS = {
        "non réalisé", "non realise", "non realisé", "non preleve", "non prélevé",
        "non réalisé/non prélevé", "non realise/non preleve",
        "en cours", "en attente"
    }

    def _is_yes(s: pd.Series) -> pd.Series:
        return s.isin(YES)

    def _is_valid_result(s: pd.Series) -> pd.Series:
        # Valide si positif ou négatif (strict)
        return s.isin(POS_SET.union(NEG_SET))

    def _is_non_result_status(s: pd.Series) -> pd.Series:
        # Heuristique : contient un des fragments de statut
        # (utile pour mesurer les cas où on a "rempli" TDR_Resultat avec un statut)
        patt = "|".join([p.replace("/", r"\/") for p in sorted(NON_RESULT_HINTS)])
        return s.str.contains(patt, case=False, na=False)

    prelev_yes = _is_yes(prelev_n)
    tdr_yes    = _is_yes(tdr_n) if COL_TDR in df.columns else pd.Series(True, index=df.index)

    # Comptes séquentiels (entonnoir)
    n_prelev = int(prelev_yes.sum())
    n_tdr    = int((prelev_yes & tdr_yes).sum())  # TDR parmi les prélevés
    valid_res_mask = (prelev_yes & tdr_yes & _is_valid_result(res_n))
    n_res = int(valid_res_mask.sum())

    # Positifs : priorité à is_tdr_pos si disponible, sinon via résultat
    if "is_tdr_pos" in df.columns:
        # On ne compte les positifs que parmi les résultats valides (entonnoir)
        is_pos = df["is_tdr_pos"].fillna(0).astype(int) == 1
        n_pos = int((valid_res_mask & is_pos).sum())
    else:
        n_pos = int((prelev_yes & tdr_yes & res_n.isin(POS_SET)).sum())

    # Qualité / incohérences (diagnostic)
    # 1) Résultat renseigné (non NA) alors que TDR_realise != Oui
    res_filled = tdr_res_raw.notna()
    incoh_res_without_tdr = int((res_filled & ~tdr_yes).sum())

    # 2) "TDR_Resultat" rempli avec un statut type "non réalisé/non prélevé"
    status_in_result = int(_is_non_result_status(res_n).sum())

    # 3) Résultats valides (pos/neg) alors que TDR_realise != Oui (plus grave)
    incoh_validres_without_tdr = int((_is_valid_result(res_n) & ~tdr_yes).sum())

    def _pct(num: int, den: int) -> float:
        return np.nan if den == 0 else (num / den * 100.0)

    rows = [
        ["Tous cas", n_all, n_all, 100.0],

        # Cascade séquentielle
        ["Prélèvement=Oui", n_prelev, n_all, _pct(n_prelev, n_all)],
        ["TDR réalisé=Oui (parmi prélevés)" if COL_TDR in df.columns else "Test documenté (parmi prélevés)", n_tdr, n_prelev, _pct(n_tdr, n_prelev)],
        ["Résultat valide (Positif/Négatif) (parmi tests)", n_res, n_tdr, _pct(n_res, n_tdr)],
        ["Positifs (parmi résultats valides)", n_pos, n_res, _pct(n_pos, n_res)],

        # Qualité des données (signaux)
        ["⚠ Résultat renseigné mais TDR_realise != Oui", incoh_res_without_tdr, n_all, _pct(incoh_res_without_tdr, n_all)],
        ["⚠ Statut saisi dans TDR_Resultat (ex: non réalisé/non prélevé)", status_in_result, n_all, _pct(status_in_result, n_all)],
        ["⚠ Résultat valide (Pos/Nég) mais TDR_realise != Oui", incoh_validres_without_tdr, n_all, _pct(incoh_validres_without_tdr, n_all)],
    ]

    return pd.DataFrame(rows, columns=["Étape", "n", "Dénominateur", "%"])

@st.cache_data(show_spinner=False)
def alerts_weekly_simple(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Alerte simple:
    - calcule cas par YW et groupe
    - % variation vs semaine précédente
    - baseline = moyenne des 3 semaines précédentes (t-1,t-2,t-3)
    """
    if ("YW" not in df.columns) or (group_col not in df.columns):
        return pd.DataFrame(columns=[group_col, "YW", "Cas", "Cas_prev", "var_%", "baseline_3w", "signal"])

    tmp = df[[group_col, "YW"]].dropna().copy()
    weekly = tmp.groupby([group_col, "YW"], as_index=False).size().rename(columns={"size": "Cas"})
    weekly = weekly.sort_values([group_col, "YW"])

    # Cas_prev
    weekly["Cas_prev"] = weekly.groupby(group_col)["Cas"].shift(1)

    # Variation %
    weekly["var_%"] = np.where(
        weekly["Cas_prev"].fillna(0) > 0,
        (weekly["Cas"] - weekly["Cas_prev"]) / weekly["Cas_prev"] * 100,
        np.nan
    )

    # baseline moyenne 3 semaines précédentes
    weekly["baseline_3w"] = (
        weekly.groupby(group_col)["Cas"]
              .shift(1)
              .rolling(3, min_periods=2)
              .mean()
              .reset_index(level=0, drop=True)
    )

    # signal: hausse forte ET dépasse baseline
    weekly["signal"] = (
        (weekly["Cas"] >= (weekly["baseline_3w"] * 1.5)) &
        (weekly["Cas"] >= 10)  # tu peux régler
    )

    return weekly

def _score_minmax_0_100(series: pd.Series, inverse: bool = False) -> pd.Series:
    """Score min-max robuste entre 0 et 100, avec option d'inversion du risque."""
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    valid = s.dropna()
    if valid.empty:
        return out
    lo = float(valid.min())
    hi = float(valid.max())
    if hi <= lo:
        out.loc[valid.index] = 50.0
    else:
        out.loc[valid.index] = ((valid - lo) / (hi - lo)) * 100.0
    if inverse:
        out = 100.0 - out
    return out


@st.cache_data(show_spinner=False)
def build_weekly_alerts(
    df: pd.DataFrame,
    group_col: str,
    *,
    week_col: str = "YW",
    cases_col: Optional[str] = None,
    baseline_weeks: int = 3,
    min_baseline_periods: int = 2,
    min_cases: int = 10,
    alert_ratio: float = 1.5,
    watch_ratio: float = 1.2,
) -> pd.DataFrame:
    """
    Alertes hebdomadaires configurables par groupe.

    - Cas = nombre de lignes, sauf si cases_col numerique est fourni.
    - baseline = moyenne des semaines precedentes dans le groupe.
    - signal_level distingue Alerte, Surveillance et Stable.
    """
    out_cols = [
        group_col,
        week_col,
        "Cas",
        "Cas_prev",
        "var_%",
        "baseline",
        "ratio_baseline",
        "signal_level",
        "signal",
    ]
    if df is None or df.empty or group_col not in df.columns or week_col not in df.columns:
        return pd.DataFrame(columns=out_cols)

    cols = [group_col, week_col]
    if cases_col and cases_col in df.columns:
        cols.append(cases_col)
    tmp = df[cols].copy()
    tmp = tmp[tmp[group_col].notna() & tmp[week_col].notna()].copy()
    if tmp.empty:
        return pd.DataFrame(columns=out_cols)

    tmp[group_col] = tmp[group_col].astype(str).str.strip()
    tmp[week_col] = tmp[week_col].astype(str).str.strip()
    tmp = tmp[(tmp[group_col] != "") & (tmp[week_col] != "")]
    if tmp.empty:
        return pd.DataFrame(columns=out_cols)

    if cases_col and cases_col in tmp.columns:
        tmp["_cases"] = pd.to_numeric(tmp[cases_col], errors="coerce")
        if tmp["_cases"].notna().mean() < 0.5:
            tmp["_cases"] = 1.0
        tmp["_cases"] = tmp["_cases"].fillna(0.0)
        weekly = tmp.groupby([group_col, week_col], as_index=False)["_cases"].sum()
        weekly = weekly.rename(columns={"_cases": "Cas"})
    else:
        weekly = tmp.groupby([group_col, week_col], as_index=False).size().rename(columns={"size": "Cas"})

    weekly = weekly.sort_values([group_col, week_col]).reset_index(drop=True)
    weekly["Cas_prev"] = weekly.groupby(group_col)["Cas"].shift(1)
    weekly["baseline"] = weekly.groupby(group_col)["Cas"].transform(
        lambda x: x.shift(1).rolling(
            int(baseline_weeks),
            min_periods=max(1, int(min_baseline_periods)),
        ).mean()
    )
    weekly["var_%"] = np.where(
        pd.to_numeric(weekly["Cas_prev"], errors="coerce").fillna(0) > 0,
        (weekly["Cas"] - weekly["Cas_prev"]) / weekly["Cas_prev"] * 100.0,
        np.nan,
    )
    weekly["ratio_baseline"] = np.where(
        pd.to_numeric(weekly["baseline"], errors="coerce").fillna(0) > 0,
        weekly["Cas"] / weekly["baseline"],
        np.nan,
    )

    weekly["signal_level"] = "Stable"
    watch_mask = (weekly["Cas"] >= int(min_cases)) & (weekly["ratio_baseline"] >= float(watch_ratio))
    alert_mask = (weekly["Cas"] >= int(min_cases)) & (weekly["ratio_baseline"] >= float(alert_ratio))
    weekly.loc[watch_mask, "signal_level"] = "Surveillance"
    weekly.loc[alert_mask, "signal_level"] = "Alerte"
    weekly["signal"] = weekly["signal_level"].eq("Alerte")

    return weekly[out_cols]


def build_spatiotemporal_cluster_table(
    df: pd.DataFrame,
    *,
    group_cols: Optional[List[str]] = None,
    week_col: str = "YW",
    cases_col: Optional[str] = None,
    recent_weeks: int = 2,
    previous_weeks: int = 4,
    min_recent_cases: int = 5,
    growth_ratio: float = 1.5,
) -> pd.DataFrame:
    """Repere les groupes avec concentration recente et croissance temporelle."""
    if df is None or df.empty or week_col not in df.columns:
        return pd.DataFrame()

    if group_cols is None:
        group_cols = [c for c in [COL_PROV, COL_ZS] if c in df.columns]
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return pd.DataFrame()

    tmp_cols = group_cols + [week_col]
    if cases_col and cases_col in df.columns:
        tmp_cols.append(cases_col)
    tmp = df[tmp_cols].copy()
    tmp = tmp.dropna(subset=group_cols + [week_col])
    if tmp.empty:
        return pd.DataFrame()

    tmp[week_col] = tmp[week_col].astype(str).str.strip()
    weeks = sorted([w for w in tmp[week_col].dropna().unique().tolist() if str(w).strip()])
    if not weeks:
        return pd.DataFrame()

    recent_list = weeks[-int(recent_weeks):]
    previous_start = max(0, len(weeks) - int(recent_weeks) - int(previous_weeks))
    previous_list = weeks[previous_start: max(0, len(weeks) - int(recent_weeks))]

    if cases_col and cases_col in tmp.columns:
        tmp["_cases"] = pd.to_numeric(tmp[cases_col], errors="coerce")
        if tmp["_cases"].notna().mean() < 0.5:
            tmp["_cases"] = 1.0
        tmp["_cases"] = tmp["_cases"].fillna(0.0)
        agg = tmp.groupby(group_cols + [week_col], as_index=False)["_cases"].sum().rename(columns={"_cases": "Cas"})
    else:
        agg = tmp.groupby(group_cols + [week_col], as_index=False).size().rename(columns={"size": "Cas"})

    recent = (
        agg[agg[week_col].isin(recent_list)]
        .groupby(group_cols, as_index=False)["Cas"].sum()
        .rename(columns={"Cas": "Cas_recents"})
    )
    previous = (
        agg[agg[week_col].isin(previous_list)]
        .groupby(group_cols, as_index=False)["Cas"].sum()
        .rename(columns={"Cas": "Cas_precedents"})
    )
    out = recent.merge(previous, on=group_cols, how="left")
    out["Cas_precedents"] = out["Cas_precedents"].fillna(0)
    prev_den = out["Cas_precedents"] / max(len(previous_list), 1)
    recent_den = out["Cas_recents"] / max(len(recent_list), 1)
    out["ratio_croissance"] = recent_den / (prev_den + 1.0)
    out["Semaines_recentes"] = ", ".join(recent_list)
    out["Semaines_reference"] = ", ".join(previous_list) if previous_list else ""
    out["cluster_signal"] = (
        (out["Cas_recents"] >= int(min_recent_cases))
        & (out["ratio_croissance"] >= float(growth_ratio))
    )
    out = out.sort_values(["cluster_signal", "Cas_recents", "ratio_croissance"], ascending=[False, False, False])
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_operational_risk_score(
    df: pd.DataFrame,
    *,
    group_col: Optional[str] = None,
    week_col: str = "YW",
    recent_weeks: int = 4,
    threshold_days: int = 2,
) -> pd.DataFrame:
    """
    Score operationnel de priorisation par zone/province.

    Combine volume, tendance recente, CFR, qualite, promptitude, positivite
    et signaux QC quand les colonnes sont disponibles.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if group_col is None:
        group_col = COL_ZS if COL_ZS in df.columns else (COL_PROV if COL_PROV in df.columns else None)
    if group_col is None or group_col not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d = d[d[group_col].notna()].copy()
    if d.empty:
        return pd.DataFrame()

    d["_risk_group"] = d[group_col].astype(str).str.strip()
    d = d[d["_risk_group"] != ""].copy()
    if d.empty:
        return pd.DataFrame()

    d["_risk_death"] = (
        pd.to_numeric(d["is_death"], errors="coerce").fillna(0)
        if "is_death" in d.columns
        else (d[COL_ISSUE].apply(is_death).astype(int) if COL_ISSUE in d.columns else 0)
    )

    base = (
        d.groupby("_risk_group", as_index=False)
        .agg(Cas=(group_col, "size"), Deces=("_risk_death", "sum"))
        .rename(columns={"_risk_group": group_col})
    )
    base["CFR_%"] = [safe_pct(num, den) for num, den in zip(base["Deces"], base["Cas"])]

    if week_col in d.columns and d[week_col].notna().any():
        weeks = sorted(d[week_col].dropna().astype(str).unique().tolist())
        recent_list = weeks[-int(recent_weeks):]
        previous_list = weeks[max(0, len(weeks) - (int(recent_weeks) * 2)): max(0, len(weeks) - int(recent_weeks))]
        recent = d[d[week_col].astype(str).isin(recent_list)].groupby("_risk_group", as_index=False).size()
        recent = recent.rename(columns={"_risk_group": group_col, "size": "Cas_recents"})
        previous = d[d[week_col].astype(str).isin(previous_list)].groupby("_risk_group", as_index=False).size()
        previous = previous.rename(columns={"_risk_group": group_col, "size": "Cas_reference"})
        base = base.merge(recent, on=group_col, how="left").merge(previous, on=group_col, how="left")
        base["Cas_recents"] = base["Cas_recents"].fillna(0)
        base["Cas_reference"] = base["Cas_reference"].fillna(0)
        recent_avg = base["Cas_recents"] / max(len(recent_list), 1)
        previous_avg = base["Cas_reference"] / max(len(previous_list), 1)
        base["Ratio_tendance"] = recent_avg / (previous_avg + 1.0)
    else:
        base["Cas_recents"] = np.nan
        base["Cas_reference"] = np.nan
        base["Ratio_tendance"] = np.nan

    key_cols = [c for c in [COL_PROV, COL_ZS, COL_AS, COL_SEX, COL_AGE, DATE_ONSET, DATE_NOTIF, COL_ISSUE] if c in d.columns]
    if "score_completude_core_%" in d.columns:
        comp = d.groupby("_risk_group")["score_completude_core_%"].mean().reset_index()
        comp = comp.rename(columns={"_risk_group": group_col, "score_completude_core_%": "Completude_%"})
    elif key_cols:
        comp_values = d.groupby("_risk_group")[key_cols].apply(lambda x: float(x.notna().mean().mean() * 100.0))
        comp = comp_values.reset_index(name="Completude_%").rename(columns={"_risk_group": group_col})
    else:
        comp = pd.DataFrame({group_col: base[group_col], "Completude_%": np.nan})
    base = base.merge(comp, on=group_col, how="left")

    if "delai_onset_to_notif" in d.columns:
        tmp = d[["_risk_group", "delai_onset_to_notif"]].copy()
        tmp["delai_onset_to_notif"] = pd.to_numeric(tmp["delai_onset_to_notif"], errors="coerce")
        tim = (
            tmp[tmp["delai_onset_to_notif"].notna() & (tmp["delai_onset_to_notif"] >= 0)]
            .groupby("_risk_group")["delai_onset_to_notif"]
            .apply(lambda x: float((x <= threshold_days).mean() * 100.0))
            .reset_index(name=f"Promptitude_<={threshold_days}j_%")
            .rename(columns={"_risk_group": group_col})
        )
        base = base.merge(tim, on=group_col, how="left")
    else:
        base[f"Promptitude_<={threshold_days}j_%"] = np.nan

    result_col = COL_TDRR if COL_TDRR in d.columns and d[COL_TDRR].notna().any() else ("Resultat_labo" if "Resultat_labo" in d.columns and d["Resultat_labo"].notna().any() else None)
    if result_col:
        tmp = d[["_risk_group", result_col]].copy()
        res = _tdr_result_norm(tmp[result_col])
        tmp["_valid"] = res.isin(TDR_POS_SET.union(TDR_NEG_SET))
        tmp["_pos"] = res.isin(TDR_POS_SET)
        pos = tmp.groupby("_risk_group", as_index=False).agg(
            _pos_num=("_pos", "sum"),
            _valid_den=("_valid", "sum"),
        )
        pos["Positivite_%"] = [
            safe_pct(num, den) for num, den in zip(pos["_pos_num"], pos["_valid_den"])
        ]
        pos = pos.rename(columns={"_risk_group": group_col}).drop(columns=["_pos_num", "_valid_den"])
        base = base.merge(pos, on=group_col, how="left")
    else:
        base["Positivite_%"] = np.nan

    flags = qc_flags(d)
    if not flags.empty and "row_id" in flags.columns:
        row_groups = d.reset_index().rename(columns={"index": "row_id"})[["row_id", "_risk_group"]]
        flag_counts = (
            flags.merge(row_groups, on="row_id", how="left")
            .dropna(subset=["_risk_group"])
            .groupby("_risk_group", as_index=False)
            .size()
            .rename(columns={"_risk_group": group_col, "size": "QC_flags"})
        )
        base = base.merge(flag_counts, on=group_col, how="left")
    else:
        base["QC_flags"] = 0
    base["QC_flags"] = pd.to_numeric(base["QC_flags"], errors="coerce").fillna(0)
    base["QC_flags_par_100_cas"] = np.where(base["Cas"] > 0, base["QC_flags"] / base["Cas"] * 100.0, np.nan)

    score_inputs = pd.DataFrame(index=base.index)
    score_inputs["Volume"] = _score_minmax_0_100(base["Cas"])
    score_inputs["Volume_recent"] = _score_minmax_0_100(base["Cas_recents"])
    score_inputs["Tendance"] = _score_minmax_0_100(base["Ratio_tendance"])
    score_inputs["CFR"] = _score_minmax_0_100(base["CFR_%"])
    score_inputs["Qualite"] = _score_minmax_0_100(base["Completude_%"], inverse=True)
    score_inputs["Promptitude"] = _score_minmax_0_100(base[f"Promptitude_<={threshold_days}j_%"], inverse=True)
    score_inputs["Positivite"] = _score_minmax_0_100(base["Positivite_%"])
    score_inputs["QC"] = _score_minmax_0_100(base["QC_flags_par_100_cas"])

    weights = {
        "Volume": 0.22,
        "Volume_recent": 0.18,
        "Tendance": 0.18,
        "CFR": 0.14,
        "Qualite": 0.10,
        "Promptitude": 0.08,
        "Positivite": 0.05,
        "QC": 0.05,
    }

    def _weighted_score(row):
        available = {k: v for k, v in weights.items() if pd.notna(row.get(k))}
        if not available:
            return np.nan
        denom = sum(available.values())
        return sum((available[k] / denom) * row[k] for k in available)

    base["Score_risque"] = score_inputs.apply(_weighted_score, axis=1).round(1)
    base["Priorite"] = pd.cut(
        base["Score_risque"],
        bins=[-np.inf, 30, 60, 80, np.inf],
        labels=["Faible", "Moderee", "Elevee", "Tres elevee"],
    ).astype("string")

    return base.sort_values(["Score_risque", "Cas"], ascending=[False, False]).reset_index(drop=True)

# =========================
# HELPERS (Sitrep)
# =========================
def export_sitrep_pdf(payload):
    """
    Exportation PDF robuste :
    - fonctionne même si certaines sections n'existent pas
    - imprime uniquement les sections disponibles
    - supporte l'ajout d'images (PNG bytes) via payload["images"]
      Formats acceptés:
        - [(title, png_bytes), ...]
        - [{"title": "...", "bytes": png_bytes}, ...]
    """
    from io import BytesIO

    # Import reportlab seulement si dispo
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except Exception as e:
        raise ModuleNotFoundError(
            "Le module 'reportlab' n'est pas installé. "
            "Installe-le via: pip install reportlab  (ou conda install -c conda-forge reportlab)"
        ) from e

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    # -------------------------
    # Helpers mise en page
    # -------------------------
    left = 50
    top = h - 50
    bottom_margin = 70
    y = top

    def new_page():
        nonlocal y
        c.showPage()
        y = top

    def ensure_space(required_height):
        """Saute de page si l'espace restant est insuffisant."""
        nonlocal y
        if y - required_height < bottom_margin:
            new_page()

    def draw_title(txt, size=11, gap=6):
        nonlocal y
        ensure_space(size + 10)
        c.setFont("Helvetica-Bold", size)
        c.drawString(left, y, str(txt))
        y -= (size + gap)

    def draw_line(txt, size=10, x=None, gap=3):
        nonlocal y
        x = left + 10 if x is None else x
        ensure_space(size + 10)
        c.setFont("Helvetica", size)
        # sécurité: éviter None
        c.drawString(x, y, str(txt) if txt is not None else "")
        y -= (size + gap)

    def add_list(title, items):
        """Ajoute une section liste si items non vide."""
        if not items:
            return
        if isinstance(items, (str, int, float)):
            items = [str(items)]
        draw_title(title, size=11)
        for it in items:
            draw_line(f"- {it}", size=10, x=left + 10)

    def add_table_simple(title, table_df, max_rows=25):
        """Table texte simple (robuste)."""
        draw_title(title, size=11)
        c.setFont("Helvetica", 9)

        if table_df is not None and hasattr(table_df, "empty") and (not table_df.empty):
            draw_line("ZS | Cas | Décès", size=9, x=left)
            for _, r in table_df.head(max_rows).iterrows():
                zs = str(r.iloc[0])
                cas = int(r.get("cas", 0)) if hasattr(r, "get") else 0
                dec = int(r.get("deces", 0)) if hasattr(r, "get") else 0
                draw_line(f"{zs} | {cas} | {dec}", size=9, x=left)
        else:
            draw_line("Données indisponibles.", size=9, x=left)

    def add_image(png_bytes, title=None, max_w=None, max_h=360):
        """
        Ajoute une image PNG (bytes) avec redimensionnement automatique.
        - max_w par défaut = largeur page - marges
        """
        nonlocal y
        if not png_bytes:
            return

        if title:
            draw_title(title, size=11)

        max_w = (w - 2 * left) if max_w is None else max_w

        img = ImageReader(BytesIO(png_bytes))
        iw, ih = img.getSize()

        # scale pour rentrer dans la page
        scale = min(max_w / float(iw), max_h / float(ih), 1.0)
        dw, dh = iw * scale, ih * scale

        ensure_space(dh + 20)
        c.drawImage(
            img,
            left,
            y - dh,
            width=dw,
            height=dh,
            preserveAspectRatio=True,
            mask="auto",
        )
        y -= dh + 15

    # -------------------------
    # HEADER
    # -------------------------
    meta = payload.get("meta", {})
    semaine = meta.get("semaine", "-")
    annee = meta.get("annee", "-")
    date_pub = meta.get("date_publication", "")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, y, f"SITREP épidémiologique CHOLERA - Semaine {semaine} / {annee}")
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(left, y, f"Date de publication: {date_pub}")
    y -= 25

    # -------------------------
    # POINTS SAILLANTS
    # -------------------------
    draw_title("Points saillants", size=11)

    kpi = payload.get("kpi", {})
    bullets = payload.get("points_saillants") or [
        f"Cas (SE): {kpi.get('cas_semaine', 0)}",
        f"Décès (SE): {kpi.get('deces_semaine', 0)}",
        f"CFR (SE): {float(kpi.get('cfr_semaine', 0.0)):.2f}%",
    ]

    for b in bullets:
        draw_line(f"- {b}", size=10, x=left + 10)

    # -------------------------
    # TABLE EPIDEMIOLOGIQUE
    # -------------------------
    table = payload.get("table_epi")
    add_table_simple("Situation épidémiologique (par ZS)", table, max_rows=25)

    # -------------------------
    # SECTIONS OPTIONNELLES
    # -------------------------
    add_list("Défis et besoins", payload.get("defis_besoins"))
    add_list("Perspectives", payload.get("perspectives"))

    # -------------------------
    # ANNEXES: IMAGES
    # -------------------------
    images = payload.get("images", [])
    if images:
        draw_title("Annexes graphiques", size=12)

        for item in images:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                t, b = item
                add_image(b, title=t)
            elif isinstance(item, dict):
                add_image(item.get("bytes"), title=item.get("title", ""))
            else:
                # si juste bytes
                add_image(item, title=None)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# =========================
# HELPERS (GEO + FUZZY JOIN)
# =========================
def _norm_key(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = "".join(c for c in unicodedata.normalize("NFD", x) if unicodedata.category(c) != "Mn")
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[’'`]", " ", x)
    x = re.sub(r"[^a-z0-9\s\-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _normalize_province_name_for_matching(x: str) -> str:
    if x is None:
        return ""
    cleaned = str(x).strip()
    cleaned = "".join(c for c in unicodedata.normalize("NFD", cleaned) if unicodedata.category(c) != "Mn")
    cleaned = re.sub(r"\s+", " ", cleaned)
    for pattern, province_name in PROVINCE_PATTERNS:
        if re.match(pattern, cleaned, flags=re.IGNORECASE):
            return province_name
    return cleaned

def _fuzzy_best_match(query, choices):
    """Retourne (best_choice, score_0_1)."""
    if not choices:
        return None, 0.0
    if HAS_RAPIDFUZZ:
        best = process.extractOne(query, choices, scorer=fuzz.token_sort_ratio)
        if best is None:
            return None, 0.0
        return best[0], best[1] / 100.0
    best_choice, best_score = None, 0.0
    for c in choices:
        sc = SequenceMatcher(None, query, c).ratio()
        if sc > best_score:
            best_score, best_choice = sc, c
    return best_choice, best_score

def joindre_donnees_fuzzy_geo(
    carte_gdf,
    df_donnees,
    colonne_cle_geo="name",
    colonne_cle_data=COL_PROV,
    colonne_valeurs="nb_cas",
    seuil=0.90,
):
    gdf = carte_gdf.copy()
    use_province_cleaning = str(colonne_cle_data).strip() == COL_PROV

    if use_province_cleaning:
        gdf["_key_geo"] = gdf[colonne_cle_geo].astype(str).map(_normalize_province_name_for_matching).map(_norm_key)
    else:
        gdf["_key_geo"] = gdf[colonne_cle_geo].astype(str).map(_norm_key)
    d = df_donnees.copy()
    if use_province_cleaning:
        d["_key_data"] = d[colonne_cle_data].astype(str).map(_normalize_province_name_for_matching).map(_norm_key)
    else:
        d["_key_data"] = d[colonne_cle_data].astype(str).map(_norm_key)

    if colonne_valeurs in d.columns:
        df_grouped = d.groupby("_key_data", as_index=False)[colonne_valeurs].sum()
    else:
        df_grouped = d.groupby("_key_data", as_index=False).size().rename(columns={"size": colonne_valeurs})

    choices = gdf["_key_geo"].dropna().unique().tolist()
    unique_data_keys = df_grouped["_key_data"].dropna().unique().tolist()

    mapping_rows = []
    for dk in unique_data_keys:
        best, score = _fuzzy_best_match(dk, choices)
        ok = (score >= seuil)
        mapping_rows.append({"key_data": dk, "key_geo": best if ok else None, "score": score, "matched": ok})

    df_map = pd.DataFrame(mapping_rows)

    df_grouped2 = df_grouped.merge(
        df_map[df_map["matched"]][["key_data", "key_geo"]],
        left_on="_key_data", right_on="key_data", how="left"
    )
    df_grouped2 = (
        df_grouped2.dropna(subset=["key_geo"])
                   .groupby("key_geo", as_index=False)[colonne_valeurs].sum()
    )

    gdf = gdf.merge(df_grouped2, left_on="_key_geo", right_on="key_geo", how="left")
    gdf[colonne_valeurs] = gdf[colonne_valeurs].fillna(0)

    match_rate = float(df_map["matched"].mean()) if len(df_map) else 0.0
    return gdf, df_map.sort_values("score", ascending=True), match_rate

def gdf_to_plotly_geojson(gdf, fid_col="fid"):
    g = gdf.copy()
    try:
        g = g.to_crs(epsg=4326)
    except Exception:
        pass
    if "geometry" in g.columns:
        try:
            g["geometry"] = g.geometry.apply(_orient_geometry_for_plotly)
        except Exception:
            pass
    g[fid_col] = g.index.astype(str)
    geojson = json.loads(g.to_json())
    return g, geojson


def _orient_geometry_for_plotly(geometry):
    if orient is None or geometry is None or geometry.is_empty:
        return geometry
    try:
        if geometry.geom_type == "Polygon":
            return orient(geometry, sign=-1.0)
        if MultiPolygon is not None and geometry.geom_type == "MultiPolygon":
            return MultiPolygon([orient(poly, sign=-1.0) for poly in geometry.geoms])
    except Exception:
        return geometry
    return geometry


def _resolve_map_filter_value(selected_label, available_values):
    if selected_label is None or pd.isna(selected_label):
        return None
    selected_text = str(selected_label).strip()
    if not selected_text:
        return None

    candidates = [
        value
        for value in available_values
        if value is not None and not pd.isna(value) and str(value).strip()
    ]
    for value in candidates:
        if str(value).strip() == selected_text:
            return value

    selected_key = _norm_key(selected_text)
    for value in candidates:
        if _norm_key(value) == selected_key:
            return value
    return None


def enrich_fuzzy_geo_map_labels(
    gdf_join,
    df_map,
    df_source,
    source_label_col: str,
    geo_label_col: str = "name",
    output_col: str = "_map_label",
):
    gdf_enriched = gdf_join.copy()
    fallback = (
        gdf_enriched[geo_label_col].astype(str)
        if geo_label_col in gdf_enriched.columns
        else pd.Series("", index=gdf_enriched.index, dtype="object")
    )
    gdf_enriched[output_col] = fallback

    if (
        df_map is None
        or df_map.empty
        or "matched" not in df_map.columns
        or "key_data" not in df_map.columns
        or "key_geo" not in df_map.columns
        or source_label_col not in df_source.columns
        or "_key_geo" not in gdf_enriched.columns
    ):
        return gdf_enriched

    labels = df_source[[source_label_col]].dropna().copy()
    if labels.empty:
        return gdf_enriched

    labels["_key_data"] = labels[source_label_col].astype(str).map(_norm_key)
    labels = labels.drop_duplicates("_key_data")

    matched = df_map[df_map["matched"]].merge(
        labels,
        left_on="key_data",
        right_on="_key_data",
        how="left",
    )
    matched = matched.dropna(subset=["key_geo", source_label_col])
    if matched.empty:
        return gdf_enriched

    matched_labels = (
        matched.groupby("key_geo", as_index=False)[source_label_col]
        .agg(lambda s: s.dropna().astype(str).iloc[0] if len(s.dropna()) else None)
        .rename(columns={source_label_col: "_mapped_label"})
    )
    gdf_enriched = gdf_enriched.merge(
        matched_labels,
        left_on="_key_geo",
        right_on="key_geo",
        how="left",
    )
    gdf_enriched[output_col] = gdf_enriched["_mapped_label"].fillna(gdf_enriched[output_col])
    return gdf_enriched.drop(columns=["key_geo", "_mapped_label"], errors="ignore")


def build_interactive_geo_map(
    gdf,
    value_col: str,
    label_col: str = "_map_label",
    hover_metric_label: str = "Cas",
    height: int = 520,
):
    if gdf is None or gdf.empty or value_col not in gdf.columns:
        return None, None

    label_col_eff = label_col if label_col in gdf.columns else ("name" if "name" in gdf.columns else None)
    if label_col_eff is None:
        return None, None

    columns_to_keep = [label_col_eff, value_col, "geometry"]
    gdf_polygons, geojson = gdf_to_plotly_geojson(gdf[columns_to_keep].copy(), fid_col="_map_fid")
    gdf_polygons[value_col] = pd.to_numeric(gdf_polygons[value_col], errors="coerce").fillna(0)

    gdf_points = gdf[columns_to_keep].copy()
    gdf_points[value_col] = pd.to_numeric(gdf_points[value_col], errors="coerce").fillna(0)
    gdf_points = gdf_points[gdf_points.geometry.notna() & ~gdf_points.geometry.is_empty].copy()
    if gdf_points.empty:
        return None, None

    try:
        gdf_points = gdf_points.to_crs(epsg=3857)
        gdf_points["geometry"] = gdf_points.geometry.centroid
        gdf_points = gdf_points.to_crs(epsg=4326)
    except Exception:
        try:
            gdf_points["geometry"] = gdf_points.geometry.centroid
        except Exception:
            return None, None

    gdf_points = gdf_points[gdf_points.geometry.notna() & ~gdf_points.geometry.is_empty].copy()
    gdf_points["_lon"] = gdf_points.geometry.x
    gdf_points["_lat"] = gdf_points.geometry.y
    gdf_points["_marker_size"] = _scale_marker_sizes(gdf_points[value_col], min_size=10, max_size=34)
    if (gdf_points["_marker_size"] > 0).any():
        gdf_points = gdf_points[gdf_points["_marker_size"] > 0].copy()
    else:
        gdf_points["_marker_size"] = 12

    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            geojson=geojson,
            locations=gdf_polygons["_map_fid"],
            z=np.ones(len(gdf_polygons)),
            featureidkey="properties._map_fid",
            colorscale=[[0.0, "rgba(255,255,255,0.03)"], [1.0, "rgba(255,255,255,0.03)"]],
            showscale=False,
            marker_line_color="#94a3b8",
            marker_line_width=0.8,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lon=gdf_points["_lon"],
            lat=gdf_points["_lat"],
            mode="markers",
            customdata=np.column_stack([gdf_points[label_col_eff], gdf_points[value_col]]),
            marker=dict(
                size=gdf_points["_marker_size"],
                color="#c2410c",
                opacity=0.85,
                line=dict(color="white", width=1),
            ),
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>{hover_metric_label}: %{{customdata[1]}}<extra></extra>",
            showlegend=False,
            name=hover_metric_label,
        )
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        domain=dict(x=[0.02, 0.98], y=[0.02, 0.98]),
        bgcolor="#f8fafc",
        showland=False,
        showocean=False,
        showlakes=False,
        showcountries=False,
        showcoastlines=False,
        showframe=False,
    )
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=6, b=6),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        dragmode="pan",
        uirevision=f"map-{label_col_eff}-{value_col}",
    )
    return fig, gdf_points


def get_selected_map_point(selection_state):
    if not selection_state:
        return None

    selection = selection_state.get("selection", {}) if hasattr(selection_state, "get") else {}
    points = selection.get("points", []) if isinstance(selection, dict) else []
    return points[0] if points else None


def get_clicked_map_label(point, gdf_map, label_col: str = "_map_label", fid_col: str = "_map_fid"):
    if not point:
        return None

    customdata = point.get("customdata") or []
    if len(customdata) > 0 and customdata[0]:
        return customdata[0]

    location = point.get("location")
    if location is not None and gdf_map is not None and fid_col in gdf_map.columns:
        match = gdf_map[gdf_map[fid_col] == str(location)]
        if not match.empty and label_col in match.columns:
            return match.iloc[0][label_col]

    point_index = point.get("pointIndex", point.get("point_number"))
    if point_index is not None and gdf_map is not None and 0 <= point_index < len(gdf_map):
        row = gdf_map.iloc[point_index]
        if label_col in row:
            return row[label_col]

    return None

# =========================
# CORE STANDARDISATION (LINE LIST) — commun Rougeole/Choléra/…
# Objectif: garantir les colonnes clés (dates, âge, sexe, semaines ISO, geo)
# =========================
def __strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')

def _clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _to_dt(s: pd.Series) -> pd.Series:

    if s is None:
        return pd.Series(dtype="datetime64[ns]")

    # 1) déjà datetime
    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    # 2) tentative texte directe. Les dates ISO YYYY-MM-DD doivent rester
    # year-first, sinon dayfirst=True peut lire 2026-01-05 comme 1er mai.
    raw_text = s.astype("string").str.strip()
    iso_mask = raw_text.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\D|$)", na=False)
    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso_mask.any():
        dt.loc[iso_mask] = pd.to_datetime(raw_text.loc[iso_mask], errors="coerce", yearfirst=True)
    rest_text = ~iso_mask
    if rest_text.any():
        dt.loc[rest_text] = pd.to_datetime(raw_text.loc[rest_text], errors="coerce", dayfirst=True)

    # si >60% converti -> c'était bien du texte
    if dt.notna().mean() > 0.6:
        return dt

    # 3) conversion numérique intelligente
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # ---- timestamps millisecondes (Android/Kobo) ----
    ms_mask = x.between(1e11, 1e14)
    out.loc[ms_mask] = pd.to_datetime(x.loc[ms_mask], unit="ms", errors="coerce")

    # ---- timestamps secondes ----
    sec_mask = x.between(1e9, 1e11)
    out.loc[sec_mask] = pd.to_datetime(x.loc[sec_mask], unit="s", errors="coerce")

    # ---- Excel serial dates (vrai cas attendu) ----
    excel_mask = x.between(1, 60000)
    out.loc[excel_mask] = pd.to_datetime(
        x.loc[excel_mask],
        unit="D",
        origin="1899-12-30",
        errors="coerce"
    )

    # fallback final texte
    rest = out.isna()
    if rest.any():
        rest_raw = raw_text.loc[rest]
        rest_iso = rest_raw.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\D|$)", na=False)
        if rest_iso.any():
            out.loc[rest_raw.loc[rest_iso].index] = pd.to_datetime(
                rest_raw.loc[rest_iso],
                errors="coerce",
                yearfirst=True,
            )
        if (~rest_iso).any():
            out.loc[rest_raw.loc[~rest_iso].index] = pd.to_datetime(
                rest_raw.loc[~rest_iso],
                errors="coerce",
                dayfirst=True,
            )

    return out


def standardize_ll_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise une line list (toutes maladies) vers les colonnes communes attendues
    par les analyses (ou par des fonctions downstream).
    - Crée les colonnes si absentes
    - Convertit Date_notification / Date_debut_maladie
    - Calcule Annee_epid / Num_semaine_epid / Semaine_epid (ISO) à partir de la meilleure date dispo
    - Harmonise Sexe (Masculin/Feminin)
    - Calcule Age_en_ans et tranches si manquants
    """
    df = _clean_colnames(df)

    # --- Rename léger (variantes fréquentes -> standard)
    rename_map = {
        # Dates
        "date_notif": "Date_notification",
        "daterep": "Date_notification",
        "date_rep": "Date_notification",
        "date_rapportage": "Date_notification",
        "date_notification_cas": "Date_notification",
        "date_de_notification": "Date_notification",

        "date_debut": "Date_debut_maladie",
        "date_onset": "Date_debut_maladie",
        "date_debut_symptomes": "Date_debut_maladie",
        "date_symptomes": "Date_debut_maladie",
        "date_issue": "Date_issue",
        "date_reception_labo": "Date_reception_labo",
        "date_reception_echantillon": "Date_reception_labo",
        "date_de_reception": "Date_reception_labo",
        "date_resultat": "Date_resultat",
        "date_reception_resultat": "Date_resultat",

        # Geo
        "province": "Province_notification",
        "prov": "Province_notification",
        "province_notif": "Province_notification",

        "zs": "Zone_de_sante_notification",
        "zone_sante": "Zone_de_sante_notification",
        "zone_sante_notification": "Zone_de_sante_notification",
        "zs_notif": "Zone_de_sante_notification",

        "as": "Aire_de_sante_notification",
        "aire_sante": "Aire_de_sante_notification",
        "aire_sante_notification": "Aire_de_sante_notification",
        "as_notif": "Aire_de_sante_notification",

        # Mpox / autres line lists (variantes fréquentes)
        "Div_Prov": "Province_notification",
        "div_prov": "Province_notification",
        "Province_notif": "Province_notification",
        "Zone_Sante": "Zone_de_sante_notification",
        "zone_sante": "Zone_de_sante_notification",
        "ZoneSante": "Zone_de_sante_notification",
        "Aire_Sante": "Aire_de_sante_notification",
        "aire_sante": "Aire_de_sante_notification",
        "Age_Cas": "Age",
        "age_cas": "Age",
        "Age_Unite": "Unite_age",
        "age_unite": "Unite_age",
        "Sexe_Cas": "Sexe",
        "sexe_cas": "Sexe",
        "Statut_Cas": "Issue",
        "statut_cas": "Issue",
        "Date_Décès": "Date_deces",
        "Date_Deces": "Date_deces",

        # Temps
        "year": "Annee_epid",
        "annee": "Annee_epid",
        "epi_year": "Annee_epid",
        "week": "Num_semaine_epid",
        "numsem": "Num_semaine_epid",
        "num_sem": "Num_semaine_epid",
        "epi_week": "Num_semaine_epid",

        # Sexe / âge
        "sex": "Sexe",
        "gender": "Sexe",
        "age_unit": "Unite_age",
        "unite": "Unite_age",
        "unite_d_age": "Unite_age",
        "age_annee": "Age",
        "age_mois": "Age",
        "age_years": "Age_en_ans",
        "age_annees": "Age_en_ans",

        # Outcome / classif
        "outcome": "Issue",
        "evolution": "Issue",
        "classification": "Classification_finale",
        "classif": "Classification_finale",
        "statut_cas": "Classification_finale",
        "classification_investigation": "Classification_finale",

        # Labo / prelevement
        "echantillon_preleve": "Prelevement",
        "type_prelevement": "Type_de_prelevement",
        "nombre_doses_vaccin": "Nombre_dose_recues",
        "laboratoire": "Nom_laboratoire",
        "lab_name": "Nom_laboratoire",
        "numero_labo": "N_labo",
        "num_labo": "N_labo",
    }
    df = _rename_columns_by_alias_map(df, rename_map)

    # --- Colonnes attendues (création si absentes)
    required = [
        "Date_notification", "Date_debut_maladie", "Date_issue", "Date_resultat", "Date_reception_labo",
        "Province_notification", "Zone_de_sante_notification", "Aire_de_sante_notification",
        "Semaine_epid", "Num_semaine_epid", "Annee_epid",
        "Sexe", "Age", "Unite_age", "Age_en_ans",
        "Tranche_age", "Tranche_age_en_ans",
        "Issue", "Classification_finale", "Prelevement", "TDR_realise", "TDR_Resultat", "Hospitalisation", "Resultat_labo",
    ]
    for c in required:
        if c not in df.columns:
            df[c] = pd.NA

    # --- Dates
    df["Date_notification"] = _to_dt(df["Date_notification"])
    df["Date_debut_maladie"] = _to_dt(df["Date_debut_maladie"])

    # --- Age brut: fallback utile pour certaines line lists labo
    age_numeric = pd.to_numeric(df["Age"], errors="coerce")
    unit_text = df["Unite_age"].astype("string").str.strip()
    unit_missing = unit_text.isna() | unit_text.eq("")
    if "Age_annee" in df.columns:
        age_annee = pd.to_numeric(df["Age_annee"], errors="coerce")
        mask = age_numeric.isna() & age_annee.notna()
        df.loc[mask, "Age"] = age_annee.loc[mask]
        df.loc[mask & unit_missing, "Unite_age"] = "ans"
        age_numeric = pd.to_numeric(df["Age"], errors="coerce")
        unit_text = df["Unite_age"].astype("string").str.strip()
        unit_missing = unit_text.isna() | unit_text.eq("")
    if "Age_mois" in df.columns:
        age_mois = pd.to_numeric(df["Age_mois"], errors="coerce")
        mask = age_numeric.isna() & age_mois.notna()
        df.loc[mask, "Age"] = age_mois.loc[mask]
        df.loc[mask & unit_missing, "Unite_age"] = "mois"

    # --- Année/Semaine ISO: si manquantes, calculer depuis la meilleure date
    need_year = df["Annee_epid"].isna().all()
    need_week = df["Num_semaine_epid"].isna().all()

    if need_year or need_week:
        ref = None
        if df["Date_notification"].notna().any():
            ref = df["Date_notification"]
        elif df["Date_debut_maladie"].notna().any():
            ref = df["Date_debut_maladie"]

        if ref is not None:
            iso = ref.dt.isocalendar()  # year, week, day
            if need_year:
                df["Annee_epid"] = iso["year"].astype("Int64")
            if need_week:
                df["Num_semaine_epid"] = iso["week"].astype("Int64")

    # Semaine_epid = YYYY-Www
    y = pd.to_numeric(df["Annee_epid"], errors="coerce").astype("Int64")
    w = pd.to_numeric(df["Num_semaine_epid"], errors="coerce").astype("Int64")
    df["Semaine_epid"] = y.astype("string") + "-W" + w.astype("string").str.zfill(2)

    # --- Sexe (harmonisation simple)
    s = df["Sexe"].astype("string").str.strip().str.lower()
    s = s.apply(lambda v: _strip_accents(v) if pd.notna(v) else v)
    df["Sexe"] = np.where(s.isin(["m", "masculin", "male", "homme", "h"]), "Masculin",
                   np.where(s.isin(["f", "feminin", "féminin", "female", "femme"]), "Feminin", df["Sexe"]))

    # --- Age_en_ans depuis Age + Unite_age (si manquant)
    df["Age_en_ans"] = pd.to_numeric(df["Age_en_ans"], errors="coerce")

    age = pd.to_numeric(df["Age"], errors="coerce")
    unit = df["Unite_age"].astype("string").str.strip().str.lower()

    # retirer accents + normaliser (robuste)
    unit = unit.apply(lambda v: _strip_accents(v) if pd.notna(v) else v)

    # valeurs vides -> on suppose "ans"
    unit = unit.fillna("ans")

    # normalisation large des unités (tu peux enrichir)
    unit = unit.replace({
        # années
        "a": "ans", "an": "ans", "ans": "ans", "annee": "ans", "annees": "ans",
        "year": "ans", "years": "ans", "yr": "ans", "yrs": "ans", "y": "ans",

        # mois
        "m": "mois", "mo": "mois", "mos": "mois", "mois": "mois",
        "month": "mois", "months": "mois",

        # semaines
        "s": "semaines", "sem": "semaines", "semaine": "semaines", "semaines": "semaines",
        "week": "semaines", "weeks": "semaines", "w": "semaines",

        # jours
        "j": "jours", "jr": "jours", "jour": "jours", "jours": "jours",
        "day": "jours", "days": "jours", "d": "jours",
    })

    # calcul âge en années selon unité
    age_years = np.where(unit.eq("ans"), age,
                np.where(unit.eq("mois"), age / 12.0,
                np.where(unit.eq("semaines"), age / 52.0,
                np.where(unit.eq("jours"), age / 365.25, np.nan))))

    # optionnel: nettoyer âges aberrants (tu peux commenter si tu ne veux pas)
    age_years = pd.Series(age_years, index=df.index)
    age_years = age_years.where((age_years >= 0) & (age_years <= 120), np.nan)

    # fill safe (pas d’erreur de longueur)
    df["Age_en_ans"] = df["Age_en_ans"].fillna(age_years)

    # --- Tranches d'âge (si manquantes)
    df["Tranche_age"] = df["Tranche_age"].astype("string")
    df["Tranche_age_en_ans"] = df["Tranche_age_en_ans"].astype("string")

    a = pd.to_numeric(df["Age_en_ans"], errors="coerce")

    conds = [
        a.notna() & (a < (1/12)),
        a.notna() & (a >= (1/12)) & (a < 5),
        a.notna() & (a >= 5) & (a < 15),
        a.notna() & (a >= 15),
    ]
    lab_txt  = ["<1 mois", "1–59 mois", "5–14 ans", "≥15 ans"]
    lab_year = ["<0.083", "0.083–4.999", "5–14", "≥15"]

    # Remplir seulement si NA (sans problème de longueur)
    df["Tranche_age"] = df["Tranche_age"].fillna(pd.Series(np.select(conds, lab_txt, default=pd.NA), index=df.index))
    df["Tranche_age_en_ans"] = df["Tranche_age_en_ans"].fillna(pd.Series(np.select(conds, lab_year, default=pd.NA), index=df.index))

    return df

# =========================
# THEME DASHBOARD
# =========================
