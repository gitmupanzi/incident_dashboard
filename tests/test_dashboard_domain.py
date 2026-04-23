import math
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    build_operational_risk_score,
    build_spatiotemporal_cluster_table,
    build_weekly_alerts,
    compute_indicators,
    is_death,
    standardize_df,
    standardize_ll_core,
)
from dashboard_app.narratives import _build_scope_overview_text
from dashboard_app.overview import format_range_label_for_display
from dashboard_app.tabs.idsr import _build_idsr_period_labels


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
        self.assertEqual(indicators["tdr_pct"], 75.0)
        self.assertEqual(indicators["pos_num"], 2)
        self.assertEqual(indicators["pos_den"], 2)
        self.assertEqual(indicators["invalid_num"], 1)
        self.assertEqual(indicators["invalid_den"], 3)

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
