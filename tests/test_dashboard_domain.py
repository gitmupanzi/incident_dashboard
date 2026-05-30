import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard_app.app_loader import validate_read_only_sql_query
from dashboard_app.advanced import clean_week, compute_irep_province
from dashboard_app.core import safe_pct
from dashboard_app.domain import (
    COL_AS,
    COL_AGE,
    COL_AGEG,
    COL_AGEG2,
    COL_CLASS,
    COL_DEHY,
    COL_PROV,
    COL_HOSP,
    COL_INVEST,
    COL_ISSUE,
    COL_PREL,
    COL_SEX,
    COL_TDR,
    COL_TDRR,
    COL_UNIT,
    COL_WEEK,
    COL_WNUM,
    COL_YEAR,
    COL_ZS,
    DATE_ADM,
    DATE_CONS,
    DATE_INV,
    DATE_ISSUE,
    DATE_NOTIF,
    DATE_ONSET,
    DATE_PREL,
    DATE_RECEP,
    DATE_RES,
    build_cousp_standard_export_package,
    build_operational_risk_score,
    cascade_metrics,
    build_standard_followup_tables,
    build_standard_action_tracker_template,
    build_standard_surveillance_chain_table,
    build_standard_signal_table,
    build_spatiotemporal_cluster_table,
    build_weekly_alerts,
    compute_indicators,
    is_death,
    merge_standard_action_tracker_template,
    standard_data_quality_summary,
    standardize_df,
    standardize_ll_by_disease,
    standardize_ll_core,
    workbook_bytes_from_sheet_dict,
)
from dashboard_app.narratives import _build_scope_overview_text
from dashboard_app.overview import (
    _build_delay_focus_metrics,
    _build_quality_focus_metrics,
    build_dashboard_kpi_payload,
    build_geo_pair_key,
    build_geo_pair_label,
    build_simple_lab_table,
    format_range_label_for_display,
    split_geo_pair_label,
)
from dashboard_app.tabs.idsr import (
    _build_idsr_completeness_matrices,
    _build_idsr_period_labels,
    _idsr_build_attack_threshold_tables,
    _idsr_build_hotspot_tables,
    _idsr_build_standard_geo_table,
    _idsr_build_year_week_key_series,
    _idsr_build_year_week_label_series,
    _idsr_fill_missing_year_from_week_consensus,
    _load_idsr_workbook,
    _idsr_recent_weeks,
    harmonize_idsr_columns,
    idsr_frame_looks_valid,
    idsr_year_from_debutsem,
    list_available_idsr_files,
    normalize_idsr_debutsem_column,
)
from dashboard_app.tabs.irep import (
    _irep_clean_province_series,
    _irep_build_silence_table,
    _irep_build_window_risk_table,
    _irep_prepare_analysis_scope,
    _irep_prepare_population_reference_frame,
)
from dashboard_app.tabs.statistics import (
    _build_statistical_examples,
    _build_statistical_notions_catalog,
)


CORE_STANDARD_COLUMNS = [
    DATE_NOTIF,
    DATE_ONSET,
    DATE_ADM,
    DATE_PREL,
    DATE_RECEP,
    DATE_RES,
    DATE_ISSUE,
    COL_PROV,
    COL_ZS,
    COL_AS,
    COL_WEEK,
    COL_WNUM,
    COL_YEAR,
    COL_SEX,
    COL_AGE,
    COL_UNIT,
    "Age_en_ans",
    COL_AGEG2,
    COL_AGEG,
    COL_ISSUE,
    COL_CLASS,
    COL_PREL,
    COL_TDR,
    COL_TDRR,
    COL_HOSP,
    "Resultat_labo",
]

ANALYTIC_STANDARD_COLUMNS = [
    "YW",
    "is_death",
    "is_tdr_pos",
    "preleve_oui_non",
    "tdr_realise_oui_non",
    "hospitalise_oui_non",
    "delai_onset_to_notif",
    "score_completude_core_%",
    "duplicate_potential",
]


class CoreHelpersTest(unittest.TestCase):
    def test_safe_pct_handles_zero_and_valid_denominators(self):
        self.assertTrue(math.isnan(safe_pct(1, 0)))
        self.assertEqual(safe_pct(1, 4), 25.0)

    def test_clean_week_keeps_only_valid_epidemiological_weeks(self):
        result = clean_week(pd.Series(["SE01", "53", "54", "x", None]))
        self.assertEqual(result.iloc[0], 1)
        self.assertEqual(result.iloc[1], 53)
        self.assertTrue(pd.isna(result.iloc[2]))
        self.assertTrue(pd.isna(result.iloc[3]))
        self.assertTrue(pd.isna(result.iloc[4]))

    def test_statistical_notions_catalog_covers_core_concepts(self):
        catalog = _build_statistical_notions_catalog()
        notions = set(catalog["Notion"].astype(str).tolist())

        self.assertIn("Taux de létalité (CFR %)", notions)
        self.assertIn("Incidence", notions)
        self.assertIn("Médiane", notions)
        self.assertIn("IREP", notions)
        self.assertIn("Alerte hebdomadaire", notions)


class StandardizationTest(unittest.TestCase):
    def test_standardize_ll_core_renames_aliases_and_derives_common_fields(self):
        raw = pd.DataFrame(
            {
                "date_notif": ["2026-01-05", "2026-01-12"],
                "province": ["Kinshasa", "Nord Kivu"],
                "zs": ["ZS A", "ZS B"],
                "sex": ["M", "female"],
                "Age_Cas": [6, 14],
                "Age_Unite": ["mois", "jours"],
            }
        )

        out = standardize_ll_core(raw)

        self.assertIn("Province_notification", out.columns)
        self.assertIn("Zone_de_sante_notification", out.columns)
        self.assertIn("Semaine_epid", out.columns)
        self.assertEqual(out.loc[0, "Province_notification"], "Kinshasa")
        self.assertEqual(out.loc[0, "Sexe"], "Masculin")
        self.assertEqual(out.loc[1, "Sexe"], "Feminin")
        self.assertAlmostEqual(float(out.loc[0, "Age_en_ans"]), 0.5, places=3)
        self.assertAlmostEqual(float(out.loc[1, "Age_en_ans"]), 14 / 365.25, places=3)
        self.assertEqual(out.loc[0, "Semaine_epid"], "2026-W02")

    def test_standardize_ll_by_disease_enriches_rougeole_lab_fields(self):
        raw = pd.DataFrame(
            {
                "Date_debut_symptomes": ["2026-01-03", "2026-01-04", "2026-01-05"],
                "Province_notification": ["Kinshasa", "Kinshasa", "Sud Kivu"],
                "Zone_de_sante_notification": ["ZS A", "ZS B", "ZS C"],
                "Sexe": ["Masculin", "Feminin", "Masculin"],
                "Age": [4, 7, 2],
                "Unite_age": ["annees", "annees", "annees"],
                "Prelevement_sang": [1, 1, 1],
                "Date_prelevement": ["2026-01-04", "2026-01-05", "2026-01-06"],
                "Date_reception_echantillon_labo": ["2026-01-06", "2026-01-07", "2026-01-08"],
                "Date_envoi_resultat": ["2026-01-10", "2026-01-11", None],
                "Resultat_igm": [1, 2, 5],
                "Nombre_doses_vaccin": [2, 1, 99],
                "Nom_laboratoire": ["INRB", "INRB", "INRB"],
                "N_labo": ["LAB001", "LAB002", "LAB003"],
                "Date_derniere_vaccination": ["2025-12-01", None, "2025-11-15"],
            }
        )

        core_df = standardize_ll_by_disease(raw, "rougeole")

        self.assertEqual(core_df.loc[0, COL_PREL], "Oui")
        self.assertEqual(core_df.loc[0, "Type_de_prelevement"], "Sang")
        self.assertEqual(core_df.loc[0, "Resultat_labo"], "positif")
        self.assertEqual(core_df.loc[1, COL_TDRR], "negatif")
        self.assertEqual(core_df.loc[2, COL_TDRR], "en attente")
        self.assertEqual(core_df.loc[0, DATE_RECEP].strftime("%Y-%m-%d"), "2026-01-06")
        self.assertEqual(core_df.loc[0, DATE_RES].strftime("%Y-%m-%d"), "2026-01-10")
        self.assertEqual(core_df.loc[0, "Nombre_dose_recues"], 2)
        self.assertEqual(core_df.loc[0, "Nom_laboratoire"], "INRB")
        self.assertEqual(core_df.loc[0, "N_labo"], "LAB001")
        self.assertEqual(core_df.loc[0, "Date_derniere_vaccination"], "2025-12-01")

        analytic_df = standardize_df(core_df)
        self.assertTrue(bool(analytic_df.loc[0, "is_tdr_pos"]))
        self.assertFalse(bool(analytic_df.loc[1, "is_tdr_pos"]))

    def test_standardize_ll_by_disease_backfills_missing_dates_row_by_row(self):
        raw = pd.DataFrame(
            {
                "Date_notification": [None, "2026-01-08"],
                "Date_consultation": ["2026-01-05", None],
                "Province_notification": ["Kinshasa", "Kinshasa"],
                "Zone_de_sante_notification": ["Gombe", "Gombe"],
            }
        )

        out = standardize_ll_by_disease(raw, "intox")

        self.assertEqual(out.loc[0, DATE_NOTIF].strftime("%Y-%m-%d"), "2026-01-05")
        self.assertEqual(int(out.loc[0, COL_YEAR]), 2026)
        self.assertEqual(int(out.loc[0, COL_WNUM]), 2)
        self.assertEqual(out.loc[0, COL_WEEK], "2026-W02")

    def test_standardize_df_derives_investigation_from_date_and_classification(self):
        raw = pd.DataFrame(
            {
                COL_INVEST: [None, None, "Non", "Oui"],
                DATE_INV: ["2026-01-03", None, "2026-01-04", None],
                COL_CLASS: ["Suspect", "Non cas", "Probable", None],
                COL_PROV: ["Kinshasa"] * 4,
                COL_ZS: ["Gombe"] * 4,
            }
        )

        out = standardize_df(raw)

        self.assertEqual(out.loc[0, COL_INVEST], "Oui")
        self.assertEqual(out.loc[1, COL_INVEST], "Oui")
        self.assertEqual(out.loc[2, COL_INVEST], "Non")
        self.assertEqual(out.loc[3, COL_INVEST], "Oui")
        self.assertTrue(bool(out.loc[2, "investigated_oui_non"]))
        self.assertEqual(int(out["investigated_oui_non"].sum()), 4)

    def test_standardize_df_derives_extended_promptitude_delays(self):
        raw = pd.DataFrame(
            {
                DATE_NOTIF: ["2026-01-02"],
                DATE_INV: ["2026-01-03"],
                DATE_PREL: ["2026-01-05"],
                DATE_RECEP: ["2026-01-06"],
                DATE_RES: ["2026-01-08"],
                DATE_ADM: ["2026-01-04"],
                DATE_ISSUE: ["2026-01-10"],
                COL_PROV: ["Kinshasa"],
                COL_ZS: ["Gombe"],
            }
        )

        out = standardize_df(raw)

        self.assertEqual(float(out.loc[0, "delai_notif_to_invest"]), 1.0)
        self.assertEqual(float(out.loc[0, "delai_notif_to_prel"]), 3.0)
        self.assertEqual(float(out.loc[0, "delai_prel_to_receipt"]), 1.0)
        self.assertEqual(float(out.loc[0, "delai_receipt_to_result"]), 2.0)
        self.assertEqual(float(out.loc[0, "delai_notif_to_adm"]), 2.0)
        self.assertEqual(float(out.loc[0, "delai_adm_to_issue"]), 6.0)

    def test_build_cousp_standard_export_package_returns_expected_sheets(self):
        raw = pd.DataFrame(
            {
                DATE_NOTIF: ["2026-01-02", "2026-01-05"],
                DATE_ONSET: ["2026-01-01", "2026-01-04"],
                DATE_INV: ["2026-01-03", None],
                DATE_PREL: ["2026-01-03", None],
                DATE_RECEP: ["2026-01-04", None],
                DATE_RES: ["2026-01-05", None],
                COL_PREL: ["Oui", "Non"],
                COL_INVEST: ["Oui", None],
                COL_ISSUE: ["Gueri", None],
                COL_CLASS: ["Suspect", "Suspect"],
                "Resultat_labo": ["Positif", None],
                COL_PROV: ["Kinshasa", "Nord Kivu"],
                COL_ZS: ["Gombe", "Goma"],
                COL_AS: ["A1", "A2"],
                COL_SEX: ["Masculin", "Feminin"],
                COL_AGE: [25, 12],
                COL_UNIT: ["annees", "annees"],
            }
        )

        standardized = standardize_df(raw)
        sheets, error = build_cousp_standard_export_package(standardized)

        self.assertIsNone(error)
        self.assertIn("LL_standard_nettoyee", sheets)
        self.assertIn("Synthese_operationnelle", sheets)
        self.assertIn("Cas_a_relancer", sheets)

        excel_bytes = workbook_bytes_from_sheet_dict(sheets)
        self.assertGreater(len(excel_bytes), 0)

    def test_build_cousp_standard_export_package_applies_dynamic_thresholds(self):
        raw = pd.DataFrame(
            {
                DATE_NOTIF: ["2026-01-02"] * 10,
                DATE_ONSET: ["2026-01-01"] * 10,
                COL_PROV: ["Kinshasa"] * 10,
                COL_ZS: ["Gombe"] * 10,
                COL_AS: ["A1"] * 10,
                COL_SEX: ["Masculin"] * 9 + [None],
                COL_AGE: [25] * 10,
                COL_UNIT: ["annees"] * 10,
            }
        )

        standardized = standardize_df(raw)
        sheets_default, error_default = build_cousp_standard_export_package(standardized)
        sheets_custom, error_custom = build_cousp_standard_export_package(
            standardized,
            seuil_acceptable=15.0,
            seuil_surveillance=30.0,
        )

        self.assertIsNone(error_default)
        self.assertIsNone(error_custom)

        completeness_default = sheets_default["Completeness_variables_cles"]
        completeness_custom = sheets_custom["Completeness_variables_cles"]

        decision_default = completeness_default.loc[
            completeness_default["Variable cle"] == COL_SEX,
            "Decision / observation",
        ].iloc[0]
        decision_custom = completeness_custom.loc[
            completeness_custom["Variable cle"] == COL_SEX,
            "Decision / observation",
        ].iloc[0]

        self.assertEqual(decision_default, "A surveiller")
        self.assertEqual(decision_custom, "Acceptable")

    def test_build_dashboard_kpi_payload_includes_investigated_cases(self):
        raw = pd.DataFrame(
            {
                COL_PROV: ["Kinshasa", "Kinshasa", "Nord Kivu"],
                COL_ZS: ["Gombe", "Gombe", "Beni"],
                DATE_NOTIF: ["2026-01-05", "2026-01-06", "2026-01-07"],
                COL_INVEST: ["Oui", None, "Non"],
                DATE_INV: [None, "2026-01-08", None],
                COL_CLASS: ["Suspect", "Confirmé", "Non cas"],
            }
        )

        payload = build_dashboard_kpi_payload(standardize_df(raw))
        chain = {str(item["label"]): item.get("value") for item in payload.get("surveillance_chain", [])}

        self.assertEqual(chain.get("Cas investigues"), 3)


class IndicatorTest(unittest.TestCase):
    def test_compute_indicators_uses_expected_denominators(self):
        df = pd.DataFrame(
            {
                "is_death": [1, 0, 0, 0],
                COL_PREL: ["Oui", "Non", "Oui", None],
                COL_HOSP: ["Non", "Oui", "Non", "Non"],
                COL_TDR: ["Oui", "Non", "Oui", "Oui"],
                COL_TDRR: ["positif", "negatif", "invalide", "positif"],
            }
        )

        indicators = compute_indicators(df)

        self.assertEqual(indicators["n_cases"], 4)
        self.assertEqual(indicators["n_deaths"], 1)
        self.assertEqual(indicators["cfr_pct"], 25.0)
        self.assertEqual(indicators["prelev_pct"], 50.0)
        self.assertEqual(indicators["tdr_pct"], 100.0)
        self.assertEqual(indicators["pos_num"], 2)
        self.assertEqual(indicators["pos_den"], 3)
        self.assertEqual(indicators["invalid_num"], 1)
        self.assertEqual(indicators["invalid_den"], 4)

    def test_compute_indicators_accepts_documentary_lab_evidence(self):
        df = standardize_df(
            pd.DataFrame(
                {
                    COL_PREL: ["Non", None, "Oui"],
                    DATE_PREL: [None, None, "2026-01-05"],
                    DATE_RECEP: ["2026-01-07", None, None],
                    DATE_RES: ["2026-01-08", "2026-01-09", None],
                    "Resultat_labo": ["positif", "negatif", None],
                    COL_HOSP: ["Non", None, "Non"],
                    DATE_ADM: [None, "2026-01-10", None],
                    "is_death": [0, 0, 0],
                }
            )
        )

        indicators = compute_indicators(df)

        self.assertEqual(indicators["prelev_num"], 3)
        self.assertEqual(indicators["tdr_num"], 2)
        self.assertEqual(indicators["hosp_num"], 1)
        self.assertEqual(indicators["pos_num"], 1)
        self.assertEqual(indicators["pos_den"], 2)

    def test_build_simple_lab_table_uses_documentary_evidence_labels(self):
        df = standardize_df(
            pd.DataFrame(
                {
                    COL_PREL: ["Non", None],
                    DATE_RECEP: ["2026-01-07", None],
                    DATE_RES: ["2026-01-08", None],
                    "Resultat_labo": ["positif", None],
                    "N_labo": ["LAB-1", None],
                }
            )
        )

        lab_tbl = build_simple_lab_table(df)
        labels = set(lab_tbl["Indicateur labo"].astype(str).tolist())
        values = dict(zip(lab_tbl["Indicateur labo"], lab_tbl["n"]))

        self.assertIn("Prélèvement documenté", labels)
        self.assertIn("Test documenté", labels)
        self.assertIn("Résultat labo documenté", labels)
        self.assertEqual(values["Prélèvement documenté"], 1)
        self.assertEqual(values["Test documenté"], 1)
        self.assertEqual(values["Résultat labo documenté"], 1)

    def test_build_quality_focus_metrics_uses_documentary_contact_and_care_evidence(self):
        df = standardize_df(
            pd.DataFrame(
                {
                    COL_CLASS: ["Suspect", "Probable", "Confirmé"],
                    DATE_RES: ["2026-01-08", None, None],
                    "Resultat_labo": ["positif", None, None],
                    "Lien_epid_avec_un_cas": ["Oui", None, "Non"],
                    DATE_ADM: [None, "2026-01-10", None],
                    COL_ISSUE: [None, None, "gueri"],
                }
            )
        )

        metrics = {str(item["label"]): float(item["value"]) for item in _build_quality_focus_metrics(df)}

        self.assertEqual(metrics["Suspects sans prélèvement"], 50.0)
        self.assertEqual(metrics["Prélèvements sans résultat"], 0.0)
        self.assertEqual(metrics["Contacts sans détail"], 33.3)
        self.assertEqual(metrics["PEC sans issue"], 33.3)

    def test_build_delay_focus_metrics_prioritizes_operational_chain(self):
        df = standardize_df(
            pd.DataFrame(
                {
                    DATE_ONSET: ["2026-01-01", "2026-01-01"],
                    DATE_NOTIF: ["2026-01-02", "2026-01-05"],
                    DATE_INV: ["2026-01-03", "2026-01-10"],
                    DATE_PREL: ["2026-01-04", "2026-01-08"],
                    DATE_RECEP: ["2026-01-05", "2026-01-09"],
                    DATE_RES: ["2026-01-06", "2026-01-11"],
                    DATE_ADM: ["2026-01-03", "2026-01-07"],
                    DATE_ISSUE: ["2026-01-07", "2026-01-11"],
                }
            )
        )

        delays = _build_delay_focus_metrics(df)
        labels = [str(item["label"]) for item in delays]
        within_target = {str(item["label"]): float(item["pct_within_target"]) for item in delays}

        self.assertEqual(
            labels,
            ["Notification", "Investigation", "Prélèvement", "Réception", "Résultat", "Admission", "Issue"],
        )
        self.assertEqual(within_target["Notification"], 50.0)
        self.assertEqual(within_target["Issue"], 0.0)

    def test_build_statistical_examples_returns_core_line_list_examples(self):
        df = standardize_df(
            pd.DataFrame(
                {
                    DATE_ONSET: ["2026-01-01", "2026-01-02"],
                    DATE_NOTIF: ["2026-01-02", "2026-01-05"],
                    DATE_ADM: ["2026-01-03", "2026-01-06"],
                    DATE_RES: ["2026-01-06", "2026-01-07"],
                    COL_ISSUE: ["deces", "gueri"],
                    COL_TDRR: ["positif", "negatif"],
                    COL_PROV: ["Kinshasa", "Kinshasa"],
                    COL_ZS: ["Gombe", "Gombe"],
                }
            )
        )

        examples = _build_statistical_examples(df, idsr_mode=False)
        notions = set(examples["Notion"].astype(str).tolist())

        self.assertIn("Effectif de cas", notions)
        self.assertIn("Létalité (CFR %)", notions)
        self.assertIn("Positivité labo (%)", notions)

    def test_is_death_recognizes_common_labels(self):
        self.assertTrue(is_death("deces"))
        self.assertTrue(is_death("mort"))
        self.assertFalse(is_death("gueri"))


class IrepTest(unittest.TestCase):
    def test_compute_irep_province_returns_current_week_ranked_rows(self):
        rows = []
        case_plan = {
            "Kinshasa": [2, 3, 4, 12],
            "Nord Kivu": [1, 1, 2, 3],
            "Tshopo": [4, 4, 5, 5],
        }
        death_plan = {
            "Kinshasa": [0, 0, 0, 1],
            "Nord Kivu": [0, 0, 0, 0],
            "Tshopo": [0, 1, 0, 0],
        }
        for province, counts in case_plan.items():
            for week_idx, cases in enumerate(counts, start=1):
                rows.append(
                    {
                        "Province_notification": province,
                        "Semaine_epid": f"2026-W{week_idx:02d}",
                        "Total_cas": cases,
                        "Total_deces": death_plan[province][week_idx - 1],
                        "Zone_de_sante_notification": "ZS",
                        "Sexe": "Masculin",
                        "Age": 20,
                        "Date_debut_maladie": "2026-01-01",
                        "Date_notification": "2026-01-02",
                        "Issue": "vivant",
                    }
                )

        out = compute_irep_province(
            pd.DataFrame(rows),
            current_week="2026-W04",
            population_map={
                "Kinshasa": 17_000_000,
                "Nord Kivu": 8_000_000,
                "Tshopo": 3_000_000,
            },
        )

        self.assertEqual(len(out), 3)
        self.assertEqual(set(out["Province_notification"]), {"Kinshasa", "Nord Kivu", "Tshopo"})
        kinshasa = out.loc[out["Province_notification"] == "Kinshasa"].iloc[0]
        self.assertEqual(kinshasa["cas_S0"], 12)
        self.assertTrue(out["IREP"].notna().any())


class AdvancedAnalyticsTest(unittest.TestCase):
    def _sample_line_list(self):
        rows = []
        counts = {
            "ZS Forte": [2, 3, 4, 12],
            "ZS Stable": [3, 3, 3, 3],
        }
        for zone, week_counts in counts.items():
            for week_idx, n_cases in enumerate(week_counts, start=1):
                for case_idx in range(n_cases):
                    rows.append(
                        {
                            COL_PROV: "Kinshasa",
                            COL_ZS: zone,
                            "YW": f"2026-W{week_idx:02d}",
                            "is_death": 1 if zone == "ZS Forte" and week_idx == 4 and case_idx == 0 else 0,
                            "score_completude_core_%": 90 if zone == "ZS Stable" else 65,
                            "delai_onset_to_notif": 1 if zone == "ZS Stable" else 4,
                            COL_TDRR: "positif" if zone == "ZS Forte" and week_idx == 4 else "negatif",
                            "Age": 20,
                            "Sexe": "Masculin",
                            "Issue": "deces" if zone == "ZS Forte" and week_idx == 4 and case_idx == 0 else "vivant",
                            "Date_debut_maladie": "2026-01-01",
                            "Date_notification": "2026-01-02",
                        }
                    )
        return pd.DataFrame(rows)

    def test_build_weekly_alerts_flags_growth_against_baseline(self):
        df = self._sample_line_list()

        alerts = build_weekly_alerts(
            df,
            COL_ZS,
            week_col="YW",
            baseline_weeks=3,
            min_baseline_periods=2,
            min_cases=5,
            alert_ratio=1.5,
        )

        latest = alerts[(alerts[COL_ZS] == "ZS Forte") & (alerts["YW"] == "2026-W04")].iloc[0]
        stable = alerts[(alerts[COL_ZS] == "ZS Stable") & (alerts["YW"] == "2026-W04")].iloc[0]
        self.assertTrue(bool(latest["signal"]))
        self.assertEqual(latest["signal_level"], "Alerte")
        self.assertFalse(bool(stable["signal"]))

    def test_cluster_table_identifies_recent_concentration(self):
        df = self._sample_line_list()

        clusters = build_spatiotemporal_cluster_table(
            df,
            group_cols=[COL_PROV, COL_ZS],
            week_col="YW",
            recent_weeks=1,
            previous_weeks=3,
            min_recent_cases=5,
            growth_ratio=1.5,
        )

        row = clusters[clusters[COL_ZS] == "ZS Forte"].iloc[0]
        self.assertEqual(row["Cas_recents"], 12)
        self.assertTrue(bool(row["cluster_signal"]))


class StandardAnalyticsTest(unittest.TestCase):
    def test_build_standard_surveillance_chain_table_summarizes_standard_blocks(self):
        df = standardize_df(
            pd.DataFrame(
                {
                    "N_alerte": ["AL-1", "AL-2", None],
                    "N_epid": ["EP-1", "EP-2", "EP-3"],
                    COL_INVEST: ["Oui", None, "Non"],
                    DATE_INV: [None, "2026-01-04", None],
                    COL_CLASS: ["Suspect", "Confirmé", "Probable"],
                    COL_PREL: ["Oui", "Oui", "Non"],
                    DATE_PREL: ["2026-01-03", "2026-01-04", None],
                    DATE_RECEP: ["2026-01-05", None, None],
                    "Resultat_labo": ["positif", "negatif", None],
                    COL_ISSUE: ["gueri", "deces", "vivant"],
                    DATE_ISSUE: ["2026-01-10", "2026-01-09", None],
                    COL_PROV: ["Kinshasa"] * 3,
                    COL_ZS: ["Gombe"] * 3,
                }
            )
        )

        chain = build_standard_surveillance_chain_table(df)
        values = dict(zip(chain["Indicateur"], chain["Valeur"]))

        self.assertEqual(values["Alertes documentées"], 2)
        self.assertEqual(values["Cas notifiés"], 3)
        self.assertEqual(values["Cas investigués"], 3)
        self.assertEqual(values["Cas suspects"], 1)
        self.assertEqual(values["Cas probables"], 1)
        self.assertEqual(values["Cas confirmés"], 1)
        self.assertEqual(values["Cas prélevés"], 2)
        self.assertEqual(values["Réceptions labo documentées"], 2)
        self.assertEqual(values["Cas positifs"], 1)
        self.assertEqual(values["Décès documentés"], 1)

    def test_build_standard_surveillance_chain_table_includes_prevention_and_exposure_evidence(self):
        df = standardize_df(
            pd.DataFrame(
                {
                    "Lien_epid_avec_un_cas": ["Oui", None, "Non"],
                    "Cas_source_id": [None, "SRC-2", None],
                    "Facteur_exposition": [None, "Soins", None],
                    "Type_de_lien": [None, None, "Familial"],
                    "Statut_vaccinal": ["Oui", None, "Non"],
                    "Nombre_dose_recues": [2, None, None],
                    "Date_derniere_vaccination": [None, "2025-12-01", None],
                    "Profession": ["Infirmier", None, "Élève"],
                }
            )
        )

        chain = build_standard_surveillance_chain_table(df)
        values = dict(zip(chain["Indicateur"], chain["Valeur"]))

        self.assertEqual(values["Exposition ou lien épid documenté"], 3)
        self.assertEqual(values["Lien épid / contact connu déclaré"], 3)
        self.assertEqual(values["Vaccination documentée"], 3)
        self.assertEqual(values["Antécédent vaccinal positif"], 2)
        self.assertEqual(values["Profession documentée"], 2)

    def test_build_standard_followup_tables_detects_standard_relance_cases(self):
        df = standardize_df(
            pd.DataFrame(
                {
                    "N_alerte": ["AL-1", "AL-2", "AL-3"],
                    COL_INVEST: ["Non", "Oui", "Oui"],
                    COL_CLASS: ["Suspect", "Probable", "Confirmé"],
                    COL_PREL: ["Non", "Oui", "Oui"],
                    DATE_PREL: [None, "2026-01-05", "2026-01-06"],
                    DATE_RECEP: [None, None, "2026-01-07"],
                    DATE_RES: [None, None, None],
                    "Resultat_labo": [None, None, "positif"],
                    "Date_confirmation": [None, None, None],
                    COL_ISSUE: ["vivant", "deces", "vivant"],
                    DATE_ISSUE: [None, None, None],
                    COL_PROV: ["Kinshasa", "", "Kinshasa"],
                    COL_ZS: ["Gombe", "Gombe", ""],
                }
            )
        )

        summary, detail = build_standard_followup_tables(df)
        counts = dict(zip(summary["Règle"], summary["Cas à relancer"]))

        self.assertEqual(counts["Cas sans investigation documentée"], 0)
        self.assertEqual(counts["Cas suspects ou probables sans prélèvement"], 1)
        self.assertEqual(counts["Cas prélevés sans réception labo"], 1)
        self.assertEqual(counts["Réceptions labo sans résultat documenté"], 0)
        self.assertEqual(counts["Cas positifs sans date de confirmation"], 1)
        self.assertEqual(counts["Décès sans date d'issue"], 1)
        self.assertEqual(counts["Localisation incomplète"], 2)
        self.assertFalse(detail.empty)

    def test_standard_data_quality_summary_includes_vaccination_and_exposure_metrics(self):
        df = standardize_df(
            pd.DataFrame(
                {
                    "Statut_vaccinal": ["Oui", None],
                    "Date_derniere_vaccination": [None, "2025-12-01"],
                    "Lien_epid_avec_un_cas": ["Oui", None],
                    "Facteur_exposition": [None, "Soins"],
                    "Profession": ["Infirmier", None],
                }
            )
        )

        summary = standard_data_quality_summary(df)
        values = dict(zip(summary["Indicateur"], summary["Valeur"]))

        self.assertEqual(values["Vaccination documentée (%)"], 100.0)
        self.assertEqual(values["Antécédent vaccinal positif (%)"], 100.0)
        self.assertEqual(values["Exposition documentée (%)"], 100.0)
        self.assertEqual(values["Lien épid / contact connu (%)"], 50.0)
        self.assertEqual(values["Profession documentée (%)"], 50.0)

    def test_build_standard_followup_tables_uses_documentary_lab_evidence(self):
        df = standardize_df(
            pd.DataFrame(
                {
                    "N_alerte": ["AL-1", "AL-2"],
                    COL_INVEST: ["Oui", "Oui"],
                    COL_CLASS: ["Probable", "Suspect"],
                    COL_PREL: ["Non", "Oui"],
                    DATE_PREL: [None, "2026-01-06"],
                    DATE_RECEP: [None, None],
                    DATE_RES: ["2026-01-08", None],
                    "Resultat_labo": ["negatif", None],
                    COL_PROV: ["Kinshasa", "Kinshasa"],
                    COL_ZS: ["Gombe", "Gombe"],
                }
            )
        )

        summary, _ = build_standard_followup_tables(df)
        counts = dict(zip(summary["Règle"], summary["Cas à relancer"]))

        self.assertEqual(counts["Cas suspects ou probables sans prélèvement"], 0)
        self.assertEqual(counts["Cas prélevés sans réception labo"], 1)
        self.assertEqual(counts["Réceptions labo sans résultat documenté"], 0)

    def test_cascade_metrics_keeps_filtered_indexes_aligned_without_futurewarning(self):
        df = standardize_df(
            pd.DataFrame(
                {
                    COL_PREL: ["Oui", "Oui", "Non"],
                    DATE_PREL: ["2026-01-05", "2026-01-06", None],
                    DATE_RECEP: ["2026-01-07", "2026-01-08", None],
                    "Resultat_labo": ["positif", "negatif", None],
                },
                index=[10, 20, 30],
            ).loc[[10, 30]]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cascade = cascade_metrics(df)

        future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
        self.assertEqual(future_warnings, [])
        values = dict(zip(cascade["Étape"], cascade["n"]))
        self.assertEqual(values["Tous cas"], 2)
        self.assertEqual(values["Prélèvement documenté"], 1)
        self.assertEqual(values["Test documenté (parmi prélèvements)"], 1)
        self.assertEqual(values["Résultat valide (Positif/Négatif) (parmi tests)"], 1)
        self.assertEqual(values["Positifs (parmi résultats valides)"], 1)

    def test_build_standard_followup_tables_detects_contact_and_care_gaps(self):
        df = standardize_df(
            pd.DataFrame(
                {
                    "Lien_epid_avec_un_cas": ["Oui", "Non"],
                    DATE_ADM: ["2026-01-10", None],
                    COL_ISSUE: [None, "gueri"],
                    DATE_ISSUE: [None, "2026-01-11"],
                }
            )
        )

        summary, _ = build_standard_followup_tables(df)
        counts = dict(zip(summary["Règle"], summary["Cas à relancer"]))

        self.assertEqual(counts["Lien épid / contact connu sans détail"], 1)
        self.assertEqual(counts["Prise en charge documentée sans issue"], 1)

    def test_build_standard_signal_table_flags_core_operational_triggers(self):
        rows = []
        weekly_counts = {
            ("Kinshasa", "ZS Forte"): [2, 3, 4, 12],
            ("Nord Kivu", "ZS Silence"): [2, 2, 2, 0],
        }
        for (province, zone), counts in weekly_counts.items():
            for week_idx, n_cases in enumerate(counts, start=1):
                for case_idx in range(n_cases):
                    rows.append(
                        {
                            COL_PROV: province,
                            COL_ZS: zone,
                            "YW": f"2026-W{week_idx:02d}",
                            "is_death": 1 if province == "Kinshasa" and week_idx == 4 and case_idx == 0 else 0,
                            "delai_onset_to_notif": 5 if week_idx == 4 else 1,
                            DATE_INV: None if week_idx == 4 and case_idx < 6 else "2026-01-10",
                            COL_TDR: "Oui",
                            COL_TDRR: "positif" if week_idx == 4 else "negatif",
                            DATE_ONSET: "2026-01-01",
                            DATE_NOTIF: "2026-01-06" if week_idx == 4 else "2026-01-02",
                            COL_ISSUE: "deces" if province == "Kinshasa" and week_idx == 4 and case_idx == 0 else "vivant",
                        }
                    )

        signals = build_standard_signal_table(
            pd.DataFrame(rows),
            week_col="YW",
            completeness_threshold=80.0,
            timeliness_threshold_days=2.0,
            timeliness_target_pct=80.0,
            investigation_target_pct=90.0,
            positivity_high_threshold=40.0,
            cfr_high_threshold=3.0,
            min_alert_cases=5,
            alert_ratio=1.5,
        )

        self.assertFalse(signals.empty)

        share_row = signals.loc[signals["Indicateur"] == "Provinces ayant transmis les données récentes"].iloc[0]
        self.assertEqual(share_row["Statut"], "Alerte")
        self.assertIn("silencieuse", share_row["Ce qu'on observe"])

        alert_row = signals.loc[signals["Indicateur"] == "Hausse inhabituelle des cas"].iloc[0]
        self.assertEqual(alert_row["Statut"], "Alerte")
        self.assertEqual(alert_row["À surveiller"], "Oui")

        investigation_row = signals.loc[signals["Indicateur"] == "Cas investigués"].iloc[0]
        self.assertEqual(investigation_row["Statut"], "Alerte")

        timeliness_row = signals.loc[signals["Indicateur"] == "Notification faite à temps"].iloc[0]
        self.assertEqual(timeliness_row["Statut"], "Alerte")

    def test_build_standard_signal_table_marks_controls_ok_when_recent_week_is_stable(self):
        rows = []
        for week_idx, n_cases in enumerate([4, 4, 5, 5], start=1):
            for case_idx in range(n_cases):
                rows.append(
                    {
                        COL_PROV: "Kinshasa",
                        COL_ZS: "ZS Stable",
                        "YW": f"2026-W{week_idx:02d}",
                        "is_death": 0,
                        "delai_onset_to_notif": 1,
                        DATE_INV: "2026-01-03",
                        COL_TDR: "Oui",
                        COL_TDRR: "negatif",
                        DATE_ONSET: "2026-01-01",
                        DATE_NOTIF: "2026-01-02",
                        COL_ISSUE: "vivant",
                    }
                )

        signals = build_standard_signal_table(
            pd.DataFrame(rows),
            week_col="YW",
            completeness_threshold=80.0,
            timeliness_threshold_days=2.0,
            timeliness_target_pct=80.0,
            investigation_target_pct=90.0,
            positivity_high_threshold=40.0,
            cfr_high_threshold=3.0,
            min_alert_cases=5,
            alert_ratio=1.5,
        )

        self.assertFalse(signals.empty)
        status_by_indicator = dict(zip(signals["Indicateur"], signals["Statut"]))
        self.assertEqual(status_by_indicator["Cas investigués"], "OK")
        self.assertEqual(status_by_indicator["Notification faite à temps"], "OK")
        self.assertEqual(status_by_indicator["Part des tests positifs"], "OK")

    def test_build_standard_signal_table_uses_classification_when_investigation_date_is_missing(self):
        rows = []
        for week_idx, n_cases in enumerate([3, 3, 4, 5], start=1):
            for _ in range(n_cases):
                rows.append(
                    {
                        COL_PROV: "Kinshasa",
                        COL_ZS: "ZS Classification",
                        "YW": f"2026-W{week_idx:02d}",
                        COL_CLASS: "Suspect",
                        DATE_INV: None,
                        DATE_ONSET: "2026-01-01",
                        DATE_NOTIF: "2026-01-02",
                        COL_ISSUE: "vivant",
                    }
                )

        signals = build_standard_signal_table(
            pd.DataFrame(rows),
            week_col="YW",
            completeness_threshold=80.0,
            timeliness_threshold_days=2.0,
            timeliness_target_pct=80.0,
            investigation_target_pct=90.0,
            positivity_high_threshold=40.0,
            cfr_high_threshold=3.0,
            min_alert_cases=5,
            alert_ratio=1.5,
        )

        investigation_row = signals.loc[signals["Indicateur"] == "Cas investigués"].iloc[0]
        self.assertEqual(investigation_row["Statut"], "OK")

    def test_build_standard_action_tracker_template_uses_only_active_triggers(self):
        signal_table = pd.DataFrame(
            [
                {
                    "Bloc": "Promptitude",
                    "Indicateur": "Notification faite à temps",
                    "Statut": "Alerte",
                    "À surveiller": "Oui",
                    "Ce qu'on observe": "Promptitude insuffisante",
                    "Action proposée": "Relancer les zones en retard",
                },
                {
                    "Bloc": "Biologie",
                    "Indicateur": "Part des tests positifs",
                    "Statut": "OK",
                    "À surveiller": "Non",
                    "Ce qu'on observe": "RAS",
                    "Action proposée": "Maintenir le suivi",
                },
            ]
        )

        tracker = build_standard_action_tracker_template(
            signal_table,
            disease_label="Choléra",
            analysis_label="SE01 à SE04",
            generated_on="2026-05-14",
        )

        self.assertEqual(len(tracker), 1)
        row = tracker.iloc[0]
        self.assertEqual(row["Signal_ID"], "Promptitude::Notification faite à temps")
        self.assertEqual(row["Maladie_source"], "Choléra")
        self.assertEqual(row["Priorite_action"], "Urgent")
        self.assertEqual(row["Statut_action"], "À démarrer")

    def test_merge_standard_action_tracker_template_preserves_manual_fields(self):
        template = pd.DataFrame(
            [
                {
                    "Signal_ID": "Promptitude::Notification faite à temps",
                    "Priorite_action": "Urgent",
                    "Niveau_reponse": "Surveillance",
                    "Action_a_suivre": "Relancer les zones",
                    "Responsable": "",
                    "Echeance": "",
                    "Statut_action": "À démarrer",
                    "Commentaire": "",
                }
            ]
        )
        existing = pd.DataFrame(
            [
                {
                    "Signal_ID": "Promptitude::Notification faite à temps",
                    "Priorite_action": "Cette semaine",
                    "Niveau_reponse": "National",
                    "Action_a_suivre": "Action reformulée",
                    "Responsable": "INFOSAN",
                    "Echeance": "2026-05-20",
                    "Statut_action": "En cours",
                    "Commentaire": "Déjà transmis",
                }
            ]
        )

        merged = merge_standard_action_tracker_template(template, existing)

        self.assertEqual(len(merged), 1)
        row = merged.iloc[0]
        self.assertEqual(row["Responsable"], "INFOSAN")
        self.assertEqual(row["Echeance"], "2026-05-20")
        self.assertEqual(row["Statut_action"], "En cours")
        self.assertEqual(row["Commentaire"], "Déjà transmis")
        self.assertEqual(row["Action_a_suivre"], "Action reformulée")


class IrepDecisionSupportTest(unittest.TestCase):
    def _sample_line_list(self):
        rows = []
        counts = {
            "ZS Forte": [2, 3, 4, 12],
            "ZS Stable": [3, 3, 3, 3],
        }
        for zone, week_counts in counts.items():
            for week_idx, n_cases in enumerate(week_counts, start=1):
                for case_idx in range(n_cases):
                    rows.append(
                        {
                            COL_PROV: "Kinshasa",
                            COL_ZS: zone,
                            "YW": f"2026-W{week_idx:02d}",
                            "is_death": 1 if zone == "ZS Forte" and week_idx == 4 and case_idx == 0 else 0,
                            "score_completude_core_%": 90 if zone == "ZS Stable" else 65,
                            "delai_onset_to_notif": 1 if zone == "ZS Stable" else 4,
                            COL_TDRR: "positif" if zone == "ZS Forte" and week_idx == 4 else "negatif",
                            "Age": 20,
                            "Sexe": "Masculin",
                            "Issue": "deces" if zone == "ZS Forte" and week_idx == 4 and case_idx == 0 else "vivant",
                            "Date_debut_maladie": "2026-01-01",
                            "Date_notification": "2026-01-02",
                        }
                    )
        return pd.DataFrame(rows)

    def test_irep_population_reference_supports_zone_and_province_denominators(self):
        pop_ref = _irep_prepare_population_reference_frame(
            pd.DataFrame(
                {
                    "PROVINCE": ["Kinshasa", "Kinshasa", "Nord Kivu"],
                    "Nom": ["ZS A", "ZS B", "ZS C"],
                    "ZSCode": ["CD1", "CD2", "CD3"],
                    "Population": [1000, 2000, 3000],
                }
            )
        )

        self.assertEqual(len(pop_ref), 3)
        self.assertEqual(int(pop_ref["Population_reference"].sum()), 6000)
        self.assertIn("Kinshasa", set(pop_ref["Province_reference"]))
        self.assertIn("ZS B", set(pop_ref["Zone_de_sante_reference"]))

    def test_irep_clean_province_series_normalizes_ocha_labels(self):
        cleaned = _irep_clean_province_series(pd.Series(["Haut-Katanga", "Maī-Ndombe", "Kasaī", "Sud-Ubangi"]))
        self.assertEqual(cleaned.tolist(), ["Haut Katanga", "Maindombe", "Kasai", "Sud Ubangi"])

    def test_irep_window_table_uses_population_and_flags_silent_units(self):
        pop_ref = _irep_prepare_population_reference_frame(
            pd.DataFrame(
                {
                    "PROVINCE": ["Kinshasa", "Kinshasa", "Kinshasa"],
                    "Nom": ["ZS Forte", "ZS Stable", "ZS Silence"],
                    "ZSCode": ["CD1", "CD2", "CD3"],
                    "Population": [1000, 2000, 1500],
                }
            )
        )

        rows = []
        for week, cases_forte, cases_stable in [(1, 2, 3), (2, 3, 3), (3, 4, 3), (4, 12, 0)]:
            for idx in range(cases_forte):
                rows.append(
                    {
                        COL_PROV: "Kinshasa",
                        COL_ZS: "ZS Forte",
                        "YW": f"2026-W{week:02d}",
                        DATE_ONSET: "2026-01-01",
                        DATE_NOTIF: "2026-01-03",
                        COL_ISSUE: "deces" if week == 4 and idx == 0 else "vivant",
                    }
                )
            for _ in range(cases_stable):
                rows.append(
                    {
                        COL_PROV: "Kinshasa",
                        COL_ZS: "ZS Stable",
                        "YW": f"2026-W{week:02d}",
                        DATE_ONSET: "2026-01-01",
                        DATE_NOTIF: "2026-01-02",
                        COL_ISSUE: "vivant",
                    }
                )

        scoped, reference = _irep_prepare_analysis_scope(pd.DataFrame(rows))
        latest_order = reference["order"].iloc[-1]
        previous_orders = reference.tail(2).head(1)["order"].tolist()
        latest_df = scoped[scoped["_surv_order"] == latest_order].copy()

        table, meta = _irep_build_window_risk_table(
            latest_df,
            scoped,
            group_cols=[COL_PROV, COL_ZS],
            geography_level="zone",
            pop_ref=pop_ref,
            trend_current_orders=[latest_order],
            trend_previous_orders=previous_orders,
            completeness_required=[COL_PROV, COL_ZS, DATE_ONSET, DATE_NOTIF, COL_ISSUE],
            threshold_days=2,
            weights={
                "trend": 0.30,
                "incidence": 0.25,
                "cfr": 0.20,
                "timeliness": 0.15,
                "completeness": 0.10,
            },
        )

        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0][COL_ZS], "ZS Forte")
        self.assertEqual(int(table.iloc[0]["Population_exposée"]), 1000)
        self.assertAlmostEqual(float(table.iloc[0]["Incidence_pour_100000"]), 1200.0, places=2)
        self.assertEqual(meta["population_coverage"], 1)

        silence = _irep_build_silence_table(
            latest_df,
            scoped,
            group_cols=[COL_PROV, COL_ZS],
            geography_level="zone",
            pop_ref=pop_ref,
        )
        self.assertEqual(set(silence[COL_ZS]), {"ZS Stable", "ZS Silence"})

    def test_irep_window_table_supports_custom_incidence_multiplier(self):
        pop_ref = _irep_prepare_population_reference_frame(
            pd.DataFrame(
                {
                    "PROVINCE": ["Kinshasa"],
                    "Nom": ["ZS Forte"],
                    "ZSCode": ["CD1"],
                    "Population": [1000],
                }
            )
        )
        rows = [
            {
                COL_PROV: "Kinshasa",
                COL_ZS: "ZS Forte",
                "YW": "2026-W04",
                DATE_ONSET: "2026-01-01",
                DATE_NOTIF: "2026-01-02",
                COL_ISSUE: "vivant",
            }
            for _ in range(10)
        ]
        scoped, reference = _irep_prepare_analysis_scope(pd.DataFrame(rows))
        latest_order = reference["order"].iloc[-1]
        latest_df = scoped[scoped["_surv_order"] == latest_order].copy()

        table, meta = _irep_build_window_risk_table(
            latest_df,
            scoped,
            group_cols=[COL_PROV, COL_ZS],
            geography_level="zone",
            pop_ref=pop_ref,
            trend_current_orders=[latest_order],
            trend_previous_orders=[],
            completeness_required=[COL_PROV, COL_ZS, DATE_ONSET, DATE_NOTIF, COL_ISSUE],
            threshold_days=2,
            weights={
                "trend": 0.30,
                "incidence": 0.25,
                "cfr": 0.20,
                "timeliness": 0.15,
                "completeness": 0.10,
            },
            denominator_mode="zs_sum",
            incidence_multiplier=1000,
        )

        self.assertIn("Incidence_pour_1000", table.columns)
        self.assertAlmostEqual(float(table.iloc[0]["Incidence_pour_1000"]), 10.0, places=2)
        self.assertAlmostEqual(float(meta["global_incidence"]), 10.0, places=2)

    def test_irep_total_population_falls_back_to_reference_when_zone_match_is_missing(self):
        pop_ref = _irep_prepare_population_reference_frame(
            pd.DataFrame(
                {
                    "PROVINCE": ["Kinshasa", "Kinshasa"],
                    "Nom": ["ZS Forte", "ZS Stable"],
                    "ZSCode": ["CD1", "CD2"],
                    "Population": [1000, 2000],
                }
            )
        )
        rows = [
            {
                COL_PROV: "Kinshasa",
                COL_ZS: "ZS Inconnue",
                "YW": "2026-W04",
                DATE_ONSET: "2026-01-01",
                DATE_NOTIF: "2026-01-02",
                COL_ISSUE: "vivant",
            }
            for _ in range(5)
        ]
        scoped, reference = _irep_prepare_analysis_scope(pd.DataFrame(rows))
        latest_order = reference["order"].iloc[-1]
        latest_df = scoped[scoped["_surv_order"] == latest_order].copy()

        table, meta = _irep_build_window_risk_table(
            latest_df,
            scoped,
            group_cols=[COL_PROV, COL_ZS],
            geography_level="zone",
            pop_ref=pop_ref,
            trend_current_orders=[latest_order],
            trend_previous_orders=[],
            completeness_required=[COL_PROV, COL_ZS, DATE_ONSET, DATE_NOTIF, COL_ISSUE],
            threshold_days=2,
            weights={
                "trend": 0.30,
                "incidence": 0.25,
                "cfr": 0.20,
                "timeliness": 0.15,
                "completeness": 0.10,
            },
            denominator_mode="zs_sum",
            incidence_multiplier=100000,
        )

        self.assertTrue(pd.isna(table.iloc[0]["Population_exposée"]))
        self.assertEqual(int(meta["total_population"]), 3000)

    def test_operational_risk_score_ranks_groups(self):
        df = self._sample_line_list()

        risk = build_operational_risk_score(
            df,
            group_col=COL_ZS,
            week_col="YW",
            recent_weeks=2,
            threshold_days=2,
        )

        self.assertEqual(set(risk[COL_ZS]), {"ZS Forte", "ZS Stable"})
        self.assertIn("Score_risque", risk.columns)
        self.assertGreaterEqual(
            float(risk.loc[risk[COL_ZS] == "ZS Forte", "Score_risque"].iloc[0]),
            float(risk.loc[risk[COL_ZS] == "ZS Stable", "Score_risque"].iloc[0]),
        )


class NarrativePeriodTest(unittest.TestCase):
    def test_display_range_formatter_uses_single_visual_convention(self):
        self.assertEqual(
            format_range_label_for_display("02/03/2026 -> 22/03/2026"),
            "02/03/2026 → 22/03/2026",
        )
        self.assertEqual(
            format_range_label_for_display("SE10-2026 à SE12-2026"),
            "SE10-2026 → SE12-2026",
        )
        self.assertEqual(
            format_range_label_for_display("Période indisponible"),
            "Période indisponible",
        )

    def test_zone_map_helpers_use_province_plus_zone_key(self):
        self.assertEqual(
            build_geo_pair_label("Kinshasa", "Damas"),
            "Kinshasa / Damas",
        )
        self.assertNotEqual(
            build_geo_pair_key("Kinshasa", "Damas"),
            build_geo_pair_key("Sud-Kivu", "Damas"),
        )
        self.assertEqual(
            split_geo_pair_label("Kinshasa / Damas"),
            ("Kinshasa", "Damas"),
        )

    def test_cumulative_surveillance_summary_prefers_iso_week_bounds(self):
        df = pd.DataFrame(
            {
                COL_YEAR: [2026, 2026],
                COL_WNUM: [10, 12],
                DATE_NOTIF: ["2026-03-05", "2026-03-09"],
                COL_PROV: ["Kinshasa", "Tshopo"],
                COL_ZS: ["ZS A", "ZS B"],
                "is_death": [0, 1],
            }
        )

        latest_week_df = df[df[COL_WNUM] == 12].copy()
        summary = _build_scope_overview_text(
            df,
            scope_kind="cumulative",
            latest_week_df=latest_week_df,
            latest_label="SE12-2026",
        )

        self.assertIsNotNone(summary)
        self.assertIn("du 02/03/2026 au 22/03/2026", summary)
        self.assertNotIn("du 05/03/2026 au 09/03/2026", summary)

    def test_recent_window_summary_uses_selected_week_count(self):
        df = pd.DataFrame(
            {
                COL_YEAR: [2026, 2026],
                COL_WNUM: [10, 12],
                DATE_NOTIF: ["2026-03-05", "2026-03-09"],
                COL_PROV: ["Kinshasa", "Tshopo"],
                COL_ZS: ["ZS A", "ZS B"],
                "is_death": [0, 1],
            }
        )

        summary = _build_scope_overview_text(
            df,
            scope_kind="recent_window",
            recent_window_weeks=4,
        )

        self.assertIsNotNone(summary)
        self.assertIn("Au cours des 4 dernieres semaines", summary)

    def test_idsr_period_labels_follow_iso_week_bounds_and_sorted_window(self):
        df = pd.DataFrame(
            {
                COL_YEAR: [2026, 2026, 2026],
                COL_WNUM: [10, 12, 11],
                "TIME_KEY": [202610, 202612, 202611],
                "TIME_LAB": ["SE10-2026", "SE12-2026", "SE11-2026"],
                "Date_debut_semaine_iso": ["2026-03-02", "2026-03-16", "2026-03-09"],
            }
        )

        period_label, time_span = _build_idsr_period_labels(df)

        self.assertEqual(period_label, "02/03/2026 -> 22/03/2026")
        self.assertEqual(time_span, "SE10-2026 -> SE12-2026")

    def test_idsr_completeness_matrices_fill_missing_iso_weeks(self):
        df = pd.DataFrame(
            {
                COL_PROV: ["Kinshasa", "Kinshasa"],
                COL_ZS: ["ZS A", "ZS A"],
                COL_YEAR: [2026, 2026],
                COL_WNUM: [10, 12],
            }
        )

        count_pivot, _, _, _ = _build_idsr_completeness_matrices(df, COL_PROV, COL_ZS)

        self.assertEqual(count_pivot.columns.tolist(), ["2026-W10", "2026-W11", "2026-W12"])
        self.assertEqual(int(count_pivot.loc["Kinshasa", "2026-W11"]), 0)


class IdsrLoadingTest(unittest.TestCase):
    def _load_real_idsr_bulletin_scope(self):
        fixture_path = PROJECT_ROOT / "tests" / "IDSR.xlsx"
        if not fixture_path.exists():
            self.skipTest("Le fichier de test IDSR.xlsx n'est pas disponible dans tests/.")

        df = _load_idsr_workbook(fixture_path)
        scope = harmonize_idsr_columns(df.copy())
        scope = normalize_idsr_debutsem_column(scope)
        scope["Num_semaine_epid"] = pd.to_numeric(scope.get("Num_semaine_epid"), errors="coerce").astype("Int64")

        year_num = pd.to_numeric(pd.Series(scope.get("Annee_epid"), index=scope.index), errors="coerce")
        if year_num.isna().all():
            scope["Annee_epid"] = idsr_year_from_debutsem(scope.get("DEBUTSEM"))
            year_num = pd.to_numeric(pd.Series(scope.get("Annee_epid"), index=scope.index), errors="coerce")

        week_num = pd.to_numeric(pd.Series(scope.get("Num_semaine_epid"), index=scope.index), errors="coerce")
        scope["TIME_LAB"] = _idsr_build_year_week_label_series(year_num, week_num)
        scope["TIME_KEY"] = _idsr_build_year_week_key_series(year_num, week_num)
        return scope

    def test_idsr_frame_validator_accepts_aggregated_shape(self):
        df = pd.DataFrame(
            {
                "PROV": ["Kinshasa"],
                "ZS": ["Gombe"],
                "NUMSEM": [10],
                "DEBUTSEM": ["lundi 03 mars 2026"],
                "TOTALCAS": [12],
                "POP": [100000],
            }
        )

        self.assertTrue(idsr_frame_looks_valid(df))

    def test_missing_year_is_filled_when_a_week_maps_to_one_observed_year(self):
        df = pd.DataFrame(
            {
                "Annee_epid": pd.Series([2026, pd.NA, 2025, pd.NA], dtype="Int64"),
                "Num_semaine_epid": pd.Series([18, 18, 17, 19], dtype="Int64"),
            }
        )

        out = _idsr_fill_missing_year_from_week_consensus(df)

        self.assertEqual(int(out.loc[1, "Annee_epid"]), 2026)
        self.assertTrue(pd.isna(out.loc[3, "Annee_epid"]))

    def test_load_idsr_workbook_rejects_non_idsr_workbook(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ll_cholera.xlsx"
            pd.DataFrame(
                {
                    "Province_notification": ["Kinshasa"],
                    "Zone_de_sante_notification": ["Gombe"],
                    "Date_notification": ["2026-01-05"],
                }
            ).to_excel(path, sheet_name="LL_Cholera", index=False)

            with self.assertRaises(ValueError):
                _load_idsr_workbook(path)

    def test_list_available_idsr_files_filters_non_idsr_workbooks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            idsr_path = root / "surveillance_idsr.xlsx"
            ll_path = root / "line_list.xlsx"

            pd.DataFrame(
                {
                    "PROV": ["Kinshasa"],
                    "ZS": ["Gombe"],
                    "NUMSEM": [10],
                    "DEBUTSEM": ["lundi 03 mars 2026"],
                    "TOTALCAS": [12],
                    "POP": [100000],
                }
            ).to_excel(idsr_path, sheet_name="Feuil1", index=False)
            pd.DataFrame(
                {
                    "Province_notification": ["Kinshasa"],
                    "Zone_de_sante_notification": ["Gombe"],
                    "Date_notification": ["2026-01-05"],
                }
            ).to_excel(ll_path, sheet_name="LL_Cholera", index=False)

            out = list_available_idsr_files([idsr_path, ll_path])

            self.assertEqual(out, [idsr_path])

    def test_load_idsr_workbook_accepts_real_fixture_from_tests_folder(self):
        fixture_path = PROJECT_ROOT / "tests" / "IDSR.xlsx"
        if not fixture_path.exists():
            self.skipTest("Le fichier de test IDSR.xlsx n'est pas disponible dans tests/.")

        df = _load_idsr_workbook(fixture_path)

        self.assertFalse(df.empty)
        self.assertTrue(idsr_frame_looks_valid(df))
        self.assertGreaterEqual(len(df), 1)
        self.assertTrue(
            {"PROV", "ZS", "NUMSEM", "DEBUTSEM", "TOTALCAS", "TOTALDECES"}.issubset(df.columns)
        )

    def test_real_fixture_supports_bulletin_standard_tables(self):
        scope = self._load_real_idsr_bulletin_scope()

        recent_weeks = _idsr_recent_weeks(scope, last_n=3)
        self.assertFalse(recent_weeks.empty)
        self.assertIn("TIME_LAB", recent_weeks.columns)
        self.assertIn("TIME_KEY", recent_weeks.columns)

        province_table = _idsr_build_standard_geo_table(
            scope,
            group_cols=["Province_notification"],
            recent_weeks=recent_weeks,
            zs_col="Zone_de_sante_notification",
        )

        self.assertFalse(province_table.empty)
        latest_label = str(recent_weeks.iloc[-1]["TIME_LAB"])
        self.assertTrue(
            {
                "Province_notification",
                "Cas_cumul",
                "Deces_cumul",
                f"Cas {latest_label}",
                f"Deces {latest_label}",
            }.issubset(province_table.columns)
        )

    def test_real_fixture_supports_bulletin_attack_and_hotspot_helpers(self):
        scope = self._load_real_idsr_bulletin_scope()

        weekly_threshold, mean_3w, latest_threshold = _idsr_build_attack_threshold_tables(
            scope,
            province_col="Province_notification",
            zs_col="Zone_de_sante_notification",
            last_n=3,
        )

        self.assertFalse(weekly_threshold.empty)
        self.assertFalse(mean_3w.empty)
        self.assertFalse(latest_threshold.empty)
        self.assertTrue({"TIME_LAB", "ZS_au_seuil", "ZS_evaluees", "Cas"}.issubset(weekly_threshold.columns))
        self.assertTrue(
            {
                "Province_notification",
                "Zone_de_sante_notification",
                "Incidence_moy_3_semaines",
                "Semaines_au_seuil",
            }.issubset(mean_3w.columns)
        )

        province_value = str(scope["Province_notification"].dropna().astype(str).iloc[0])
        summary, latest_zs, above_avg, silent_zs, detail = _idsr_build_hotspot_tables(
            scope,
            province_value=province_value,
            province_col="Province_notification",
            zs_col="Zone_de_sante_notification",
            last_n=3,
        )

        self.assertIn("latest_label", summary)
        self.assertIn("reporting_zs_latest", summary)
        self.assertFalse(latest_zs.empty)
        self.assertFalse(detail.empty)
        self.assertIn("Zone_de_sante_notification", latest_zs.columns)
        self.assertIn("Zone_de_sante_notification", detail.columns)
        self.assertIsInstance(above_avg, pd.DataFrame)
        self.assertIsInstance(silent_zs, pd.DataFrame)


class AppLoaderValidationTest(unittest.TestCase):
    def test_validate_read_only_sql_query_accepts_select_and_with(self):
        self.assertTrue(validate_read_only_sql_query("SELECT * FROM public.line_list"))
        self.assertTrue(validate_read_only_sql_query("WITH a AS (SELECT 1) SELECT * FROM a"))

    def test_validate_read_only_sql_query_rejects_multi_statement_or_write_queries(self):
        self.assertFalse(validate_read_only_sql_query("SELECT * FROM a; DELETE FROM a"))
        self.assertFalse(validate_read_only_sql_query("DELETE FROM public.line_list"))


class StandardSchemaContractTest(unittest.TestCase):
    def _standard_schema_sample(self):
        rows = []
        weekly_counts = {
            "ZS Forte": [2, 3, 4, 12],
            "ZS Stable": [3, 3, 3, 3],
        }
        for zone, counts in weekly_counts.items():
            for week_idx, n_cases in enumerate(counts, start=1):
                for case_idx in range(n_cases):
                    rows.append(
                        {
                            DATE_NOTIF: f"2026-01-{week_idx + 1:02d}",
                            DATE_ONSET: f"2026-01-{week_idx:02d}",
                            DATE_ADM: f"2026-01-{week_idx + 1:02d}",
                            DATE_PREL: f"2026-01-{week_idx + 2:02d}",
                            DATE_RECEP: f"2026-01-{week_idx + 3:02d}",
                            DATE_RES: f"2026-01-{week_idx + 4:02d}",
                            DATE_ISSUE: f"2026-01-{week_idx + 5:02d}",
                            COL_PROV: "Kinshasa",
                            COL_ZS: zone,
                            COL_AS: f"AS {zone}",
                            COL_YEAR: 2026,
                            COL_WNUM: week_idx,
                            COL_WEEK: f"2026-W{week_idx:02d}",
                            COL_SEX: "Masculin" if case_idx % 2 == 0 else "Feminin",
                            COL_AGE: 20 + case_idx,
                            COL_UNIT: "ans",
                            COL_ISSUE: "deces" if zone == "ZS Forte" and week_idx == 4 and case_idx == 0 else "vivant",
                            COL_CLASS: "Confirme",
                            COL_PREL: "Oui",
                            COL_TDR: "Oui",
                            COL_TDRR: "positif" if zone == "ZS Forte" and week_idx == 4 else "negatif",
                            COL_HOSP: "Oui" if zone == "ZS Forte" else "Non",
                            COL_DEHY: "Aucune",
                            "Resultat_labo": "positif" if zone == "ZS Forte" and week_idx == 4 else "negatif",
                        }
                    )
        return pd.DataFrame(rows)

    def test_standard_common_columns_feed_standard_analytics(self):
        raw = self._standard_schema_sample()
        core_df = standardize_ll_core(raw)

        missing_core = [col for col in CORE_STANDARD_COLUMNS if col not in core_df.columns]
        self.assertEqual(missing_core, [])

        df = standardize_df(core_df)
        missing_analytics = [col for col in ANALYTIC_STANDARD_COLUMNS if col not in df.columns]
        self.assertEqual(missing_analytics, [])

        indicators = compute_indicators(df)
        self.assertEqual(indicators["n_cases"], len(df))
        self.assertGreater(indicators["n_deaths"], 0)

        risk = build_operational_risk_score(
            df,
            group_col=COL_ZS,
            week_col="YW",
            recent_weeks=2,
            threshold_days=2,
        )
        self.assertFalse(risk.empty)
        self.assertIn("Score_risque", risk.columns)

        alerts = build_weekly_alerts(
            df,
            COL_ZS,
            week_col="YW",
            min_cases=5,
            alert_ratio=1.5,
        )
        self.assertIn("signal_level", alerts.columns)

        clusters = build_spatiotemporal_cluster_table(
            df,
            group_cols=[COL_PROV, COL_ZS],
            week_col="YW",
            min_recent_cases=5,
        )
        self.assertIsInstance(clusters, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
