import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard_app.column_mapping import (
    AUTO_APPLY_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DERIVED_COLUMNS,
    IMPORTANT_COLUMNS,
    OPTIONAL_COLUMNS,
    SOURCE_COLUMNS,
    add_derived_columns_after_mapping,
    apply_auto_prefill_to_selection_state,
    auto_map_columns,
    build_auto_applied_mapping,
    build_mapping_preview_table,
    build_mapping_quality_report,
    dataframe_to_standardized_excel_bytes,
    describe_mapping_candidate,
    extract_profile_mapping,
    list_mapping_profiles,
    load_mapping_profile,
    normalize_column_name,
    rename_dataframe_to_standard,
    resolve_mapping_selection_metadata,
    save_mapping_profile,
    validate_mapping,
)


class ColumnMappingNormalizationTest(unittest.TestCase):
    def test_normalize_column_name_removes_accents_spaces_and_case(self):
        self.assertEqual(normalize_column_name(" Zone de santé "), "zone_de_sante")
        self.assertEqual(normalize_column_name("Date-Notif"), "date_notif")

    def test_source_and_derived_columns_are_strictly_separated(self):
        for derived_name in DERIVED_COLUMNS:
            self.assertNotIn(derived_name, SOURCE_COLUMNS)
            self.assertNotIn(derived_name, IMPORTANT_COLUMNS)
            self.assertNotIn(derived_name, OPTIONAL_COLUMNS)


class ColumnMappingDetectionTest(unittest.TestCase):
    def test_describe_mapping_candidate_detects_exact_match(self):
        meta = describe_mapping_candidate("Province_notification", "Province_notification")

        self.assertEqual(meta["method"], "exact_match")
        self.assertEqual(meta["confidence"], 100)
        self.assertTrue(meta["accepted"])

    def test_describe_mapping_candidate_detects_normalized_match(self):
        meta = describe_mapping_candidate("Province_notification", "Province-notification")

        self.assertEqual(meta["method"], "normalized_match")
        self.assertTrue(meta["accepted"])

    def test_describe_mapping_candidate_detects_variant_match(self):
        meta = describe_mapping_candidate("Zone_de_sante_notification", "ZS")

        self.assertEqual(meta["method"], "variant_match")
        self.assertEqual(meta["confidence"], 100)
        self.assertTrue(meta["accepted"])

    def test_describe_mapping_candidate_detects_fuzzy_match(self):
        meta = describe_mapping_candidate("Province_notification", "Provnce", threshold=80)

        self.assertEqual(meta["method"], "fuzzy_match")
        self.assertGreaterEqual(meta["confidence"], 80)

    def test_auto_map_columns_returns_confidence_metadata(self):
        mapping, metadata = auto_map_columns(
            ["Province", "ZS", "date_notif", "Age_Cas"],
            threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )

        self.assertEqual(mapping["Province_notification"], "Province")
        self.assertEqual(metadata["Province_notification"]["method"], "variant_match")
        self.assertEqual(metadata["Zone_de_sante_notification"]["source_column"], "ZS")
        self.assertEqual(metadata["Date_notification"]["method"], "variant_match")
        self.assertEqual(metadata["Age"]["source_column"], "Age_Cas")

    def test_auto_map_columns_keeps_low_confidence_candidate_out_of_mapping(self):
        mapping, metadata = auto_map_columns(
            ["province x", "zs locale", "date approximative"],
            threshold=95,
        )

        self.assertNotIn("Date_notification", mapping)
        self.assertEqual(metadata["Date_notification"]["method"], "fuzzy_match")
        self.assertFalse(metadata["Date_notification"]["accepted"])

    def test_resolve_mapping_selection_metadata_marks_manual_override(self):
        _, metadata = auto_map_columns(["Province", "ZS", "Date_notification", "Age_en_ans"])

        manual_meta = resolve_mapping_selection_metadata(
            "Province_notification",
            "Province terrain",
            metadata,
        )

        self.assertEqual(manual_meta["method"], "manual")
        self.assertTrue(manual_meta["accepted"])


class ColumnMappingAutoApplyThresholdTest(unittest.TestCase):
    def test_build_auto_applied_mapping_accepts_score_100(self):
        auto_mapping = build_auto_applied_mapping(
            {
                "Province_notification": {
                    "source_column": "Province",
                    "confidence": 100,
                    "method": "exact_match",
                    "accepted": True,
                }
            },
            threshold=AUTO_APPLY_CONFIDENCE_THRESHOLD,
        )

        self.assertEqual(auto_mapping["Province_notification"], "Province")

    def test_build_auto_applied_mapping_accepts_score_92(self):
        auto_mapping = build_auto_applied_mapping(
            {
                "Age": {
                    "source_column": "Age_Cas",
                    "confidence": 92,
                    "method": "fuzzy_match",
                    "accepted": True,
                }
            },
            threshold=AUTO_APPLY_CONFIDENCE_THRESHOLD,
        )

        self.assertEqual(auto_mapping["Age"], "Age_Cas")

    def test_build_auto_applied_mapping_rejects_score_89(self):
        auto_mapping = build_auto_applied_mapping(
            {
                "Issue": {
                    "source_column": "Evolution",
                    "confidence": 89,
                    "method": "fuzzy_match",
                    "accepted": False,
                }
            },
            threshold=AUTO_APPLY_CONFIDENCE_THRESHOLD,
        )

        self.assertNotIn("Issue", auto_mapping)

    def test_low_confidence_suggestion_remains_visible_but_not_applied(self):
        auto_mapping = build_auto_applied_mapping(
            {
                "Classification_finale": {
                    "source_column": "Statut",
                    "confidence": 76,
                    "method": "fuzzy_match",
                    "accepted": False,
                }
            },
            threshold=AUTO_APPLY_CONFIDENCE_THRESHOLD,
        )

        preview = build_mapping_preview_table(
            {"Province_notification": "Province", "Zone_de_sante_notification": "ZS"},
            {
                "Classification_finale": {
                    "source_column": "Statut",
                    "confidence": 76,
                    "method": "fuzzy_match",
                    "accepted": False,
                }
            },
            threshold=AUTO_APPLY_CONFIDENCE_THRESHOLD,
        )

        classification_row = preview.loc[preview["Variable standard"] == "Classification_finale"].iloc[0]
        self.assertEqual(auto_mapping, {})
        self.assertEqual(classification_row["Colonne source proposée"], "Statut")
        self.assertEqual(classification_row["Statut"], "À vérifier")

    def test_manual_mapping_can_replace_non_auto_applied_suggestion(self):
        auto_metadata = {
            "Issue": {
                "source_column": "Evolution",
                "confidence": 82,
                "method": "fuzzy_match",
                "accepted": False,
            }
        }
        auto_mapping = build_auto_applied_mapping(
            auto_metadata,
            threshold=AUTO_APPLY_CONFIDENCE_THRESHOLD,
        )
        manual_meta = resolve_mapping_selection_metadata("Issue", "Evolution", auto_metadata)

        self.assertNotIn("Issue", auto_mapping)
        self.assertEqual(manual_meta["method"], "manual")
        self.assertTrue(manual_meta["accepted"])

    def test_build_auto_applied_mapping_ignores_derived_columns(self):
        auto_mapping = build_auto_applied_mapping(
            {
                "Semaine_epid": {
                    "source_column": "Semaine source",
                    "confidence": 100,
                    "method": "variant_match",
                    "accepted": True,
                },
                "Age_en_ans": {
                    "source_column": "Age standardise",
                    "confidence": 95,
                    "method": "variant_match",
                    "accepted": True,
                },
            },
            threshold=AUTO_APPLY_CONFIDENCE_THRESHOLD,
            include_derived=False,
        )

        self.assertEqual(auto_mapping, {})

    def test_apply_auto_prefill_to_selection_state_updates_placeholders_only(self):
        updated_state = apply_auto_prefill_to_selection_state(
            current_state={
                "Province_notification": "-- Non associee --",
                "Zone_de_sante_notification": "Choix manuel",
                "Date_notification": "",
            },
            auto_prefill_mapping={
                "Province_notification": "Province",
                "Zone_de_sante_notification": "ZS",
                "Date_notification": "Date_notif",
            },
            placeholder="-- Non associee --",
        )

        self.assertEqual(updated_state["Province_notification"], "Province")
        self.assertEqual(updated_state["Date_notification"], "Date_notif")
        self.assertEqual(updated_state["Zone_de_sante_notification"], "Choix manuel")


class ColumnMappingValidationTest(unittest.TestCase):
    def test_validate_mapping_does_not_treat_derived_columns_as_source_requirements(self):
        is_valid, errors = validate_mapping(
            {
                "Province_notification": "Province",
                "Zone_de_sante_notification": "ZS",
                "Date_notification": "Date_notification",
                "Age": "Age",
                "Unite_age": "Unite_age",
            }
        )

        self.assertTrue(is_valid)
        self.assertFalse(any("Semaine_epid" in err for err in errors))
        self.assertFalse(any("Age_en_ans" in err for err in errors))

    def test_validate_mapping_requires_critical_columns(self):
        is_valid, errors = validate_mapping(
            {
                "Date_notification": "Date notification",
                "Age_en_ans": "Age en ans",
            }
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("Province_notification" in err for err in errors))
        self.assertTrue(any("Zone_de_sante_notification" in err for err in errors))

    def test_validate_mapping_rejects_duplicate_source_assignments(self):
        is_valid, errors = validate_mapping(
            {
                "Province_notification": "Province",
                "Zone_de_sante_notification": "Province",
                "Date_notification": "Date notification",
                "Age_en_ans": "Age en ans",
            }
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("plusieurs fois" in err for err in errors))

    def test_validate_mapping_rejects_missing_time_columns(self):
        is_valid, errors = validate_mapping(
            {
                "Province_notification": "Province",
                "Zone_de_sante_notification": "ZS",
                "Age_en_ans": "Age en ans",
            }
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("temporelle" in err for err in errors))

    def test_validate_mapping_accepts_year_and_week_without_date_notification(self):
        is_valid, errors = validate_mapping(
            {
                "Province_notification": "Province",
                "Zone_de_sante_notification": "ZS",
                "Annee_epid": "Year",
                "Num_semaine_epid": "Week",
                "Age_en_ans": "Age en ans",
            }
        )

        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_validate_mapping_accepts_age_and_unit_without_age_en_ans(self):
        is_valid, errors = validate_mapping(
            {
                "Province_notification": "Province",
                "Zone_de_sante_notification": "ZS",
                "Date_notification": "Date notification",
                "Age": "Age",
                "Unite_age": "Unite_age",
            }
        )

        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_validate_mapping_accepts_tranche_age_without_age_columns(self):
        is_valid, errors = validate_mapping(
            {
                "Province_notification": "Province",
                "Zone_de_sante_notification": "ZS",
                "Date_debut_maladie": "Date_debut_maladie",
                "Tranche_age": "Tranche_age",
            }
        )

        self.assertTrue(is_valid)
        self.assertEqual(errors, [])


class ColumnMappingPreviewTest(unittest.TestCase):
    def test_build_mapping_preview_table_reports_manual_and_derived_statuses(self):
        _, auto_metadata = auto_map_columns(["Province", "ZS", "Date_notification", "Age", "Unite_age"])
        preview = build_mapping_preview_table(
            {
                "Province_notification": "Province terrain",
                "Zone_de_sante_notification": "ZS",
                "Date_notification": "Date_notification",
                "Age": "Age",
                "Unite_age": "Unite_age",
            },
            auto_metadata,
        )

        province_row = preview.loc[preview["Variable standard"] == "Province_notification"].iloc[0]
        week_row = preview.loc[preview["Variable standard"] == "Semaine_epid"].iloc[0]

        self.assertEqual(province_row["Méthode de détection"], "manual")
        self.assertEqual(province_row["Statut"], "À vérifier")
        self.assertEqual(week_row["Statut"], "Calculable automatiquement")


class ColumnMappingProfileTest(unittest.TestCase):
    def test_save_load_and_list_mapping_profiles(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mapping_dir = Path(tmp_dir)
            save_path = save_mapping_profile(
                {
                    "Province_notification": "Province",
                    "Zone_de_sante_notification": "ZS",
                },
                profile_name="autre_cholera_labo",
                metadata={"disease_key": "autre"},
                mapping_dir=mapping_dir,
            )

            self.assertTrue(save_path.exists())

            payload = load_mapping_profile("autre_cholera_labo", mapping_dir=mapping_dir)
            self.assertEqual(payload["profile_name"], "autre_cholera_labo")
            self.assertEqual(payload["mapping"]["Province_notification"], "Province")

            profiles = list_mapping_profiles(mapping_dir=mapping_dir)
            self.assertEqual(profiles, ["autre_cholera_labo"])

            saved_json = json.loads(save_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_json["metadata"]["disease_key"], "autre")

    def test_extract_profile_mapping_keeps_only_source_columns_by_default(self):
        profile_mapping = extract_profile_mapping(
            {
                "Province_notification": "Province",
                "Zone_de_sante_notification": "ZS",
                "Date_notification": "Date_notif",
                "Age": "Age",
                "Unite_age": "Unite_age",
                "Semaine_epid": "Semaine_fichier",
                "Age_en_ans": "Age_standardise",
            }
        )

        self.assertIn("Province_notification", profile_mapping)
        self.assertNotIn("Semaine_epid", profile_mapping)
        self.assertNotIn("Age_en_ans", profile_mapping)

    def test_extract_profile_mapping_keeps_derived_fallbacks_when_sources_missing(self):
        profile_mapping = extract_profile_mapping(
            {
                "Province_notification": "Province",
                "Zone_de_sante_notification": "ZS",
                "Semaine_epid": "Semaine_fichier",
                "Age_en_ans": "Age_standardise",
            }
        )

        self.assertEqual(profile_mapping["Semaine_epid"], "Semaine_fichier")
        self.assertEqual(profile_mapping["Age_en_ans"], "Age_standardise")


class ColumnMappingRenameTest(unittest.TestCase):
    def test_rename_dataframe_to_standard_renames_expected_columns(self):
        raw = pd.DataFrame(
            {
                "Province": ["Kinshasa"],
                "ZS": ["ZS Gombe"],
                "Date notif": ["2026-01-10"],
            }
        )

        renamed = rename_dataframe_to_standard(
            raw,
            {
                "Province_notification": "Province",
                "Zone_de_sante_notification": "ZS",
                "Date_notification": "Date notif",
            },
        )

        self.assertIn("Province_notification", renamed.columns)
        self.assertIn("Zone_de_sante_notification", renamed.columns)
        self.assertIn("Date_notification", renamed.columns)
        self.assertNotIn("Province", renamed.columns)
        self.assertEqual(renamed.loc[0, "Province_notification"], "Kinshasa")

    def test_rename_dataframe_to_standard_preserves_existing_standard_column(self):
        raw = pd.DataFrame(
            {
                "Province_notification": [pd.NA, "Nord Kivu"],
                "Province": ["Kinshasa", "Nord Kivu"],
            }
        )

        renamed = rename_dataframe_to_standard(
            raw,
            {"Province_notification": "Province"},
        )

        self.assertEqual(renamed.loc[0, "Province_notification"], "Kinshasa")
        self.assertEqual(renamed.loc[1, "Province_notification"], "Nord Kivu")
        self.assertNotIn("Province", renamed.columns)

    def test_rename_dataframe_to_standard_keeps_unmapped_columns(self):
        raw = pd.DataFrame(
            {
                "Province": ["Kinshasa"],
                "ZS": ["ZS Gombe"],
                "Commentaire_libre": ["Texte libre"],
            }
        )

        renamed = rename_dataframe_to_standard(
            raw,
            {
                "Province_notification": "Province",
                "Zone_de_sante_notification": "ZS",
            },
            keep_unmapped_columns=True,
        )

        self.assertIn("Commentaire_libre", renamed.columns)


class ColumnMappingDerivedColumnsTest(unittest.TestCase):
    def test_add_derived_columns_after_mapping_builds_epi_columns_from_date(self):
        raw = pd.DataFrame(
            {
                "Date_notification": ["2026-01-05", "2026-01-12"],
                "Province_notification": ["Kinshasa", "Kinshasa"],
                "Zone_de_sante_notification": ["Gombe", "Gombe"],
                "Age_en_ans": [20, 25],
            }
        )

        out = add_derived_columns_after_mapping(raw)

        self.assertIn("Annee_epid", out.columns)
        self.assertIn("Num_semaine_epid", out.columns)
        self.assertIn("Semaine_epid", out.columns)
        self.assertEqual(int(out.loc[0, "Annee_epid"]), 2026)
        self.assertEqual(int(out.loc[0, "Num_semaine_epid"]), 2)
        self.assertEqual(out.loc[0, "Semaine_epid"], "2026-W02")

    def test_add_derived_columns_after_mapping_builds_age_columns_from_age_and_unit(self):
        raw = pd.DataFrame(
            {
                "Province_notification": ["Kinshasa", "Kinshasa"],
                "Zone_de_sante_notification": ["Gombe", "Gombe"],
                "Date_notification": ["2026-01-05", "2026-01-12"],
                "Age": [24, 10],
                "Unite_age": ["mois", "ans"],
            }
        )

        out = add_derived_columns_after_mapping(raw)

        self.assertAlmostEqual(float(out.loc[0, "Age_en_ans"]), 2.0, places=3)
        self.assertAlmostEqual(float(out.loc[1, "Age_en_ans"]), 10.0, places=3)
        self.assertEqual(out.loc[0, "Tranche_age"], "1-59 mois")
        self.assertEqual(out.loc[1, "Tranche_age"], "5-14 ans")

    def test_add_derived_columns_after_mapping_preserves_existing_values_by_default(self):
        raw = pd.DataFrame(
            {
                "Date_notification": ["2026-01-05"],
                "Province_notification": ["Kinshasa"],
                "Zone_de_sante_notification": ["Gombe"],
                "Age_en_ans": [7.0],
                "Tranche_age": ["Deja calculee"],
            }
        )

        out = add_derived_columns_after_mapping(raw)

        self.assertEqual(out.loc[0, "Tranche_age"], "Deja calculee")
        self.assertEqual(float(out.loc[0, "Age_en_ans"]), 7.0)

    def test_add_derived_columns_after_mapping_parses_existing_week_labels(self):
        raw = pd.DataFrame(
            {
                "Province_notification": ["Kinshasa"],
                "Zone_de_sante_notification": ["Gombe"],
                "Semaine_epid": ["S02-2026"],
                "Age_en_ans": [7.0],
            }
        )

        out = add_derived_columns_after_mapping(raw)

        self.assertEqual(int(out.loc[0, "Annee_epid"]), 2026)
        self.assertEqual(int(out.loc[0, "Num_semaine_epid"]), 2)
        self.assertEqual(out.loc[0, "Semaine_epid"], "S02-2026")

    def test_add_derived_columns_after_mapping_can_return_derived_info(self):
        raw = pd.DataFrame(
            {
                "Date_notification": ["2026-01-05"],
                "Province_notification": ["Kinshasa"],
                "Zone_de_sante_notification": ["Gombe"],
                "Age": [24],
                "Unite_age": ["mois"],
            }
        )

        out, info = add_derived_columns_after_mapping(raw, return_info=True)

        self.assertIn("Semaine_epid", out.columns)
        self.assertGreaterEqual(info["weeks_calculated"], 1)
        self.assertGreaterEqual(info["age_years_calculated"], 1)
        self.assertGreaterEqual(info["age_groups_calculated"], 1)


class ColumnMappingQualityReportTest(unittest.TestCase):
    def test_build_mapping_quality_report_returns_expected_summary(self):
        df = pd.DataFrame(
            {
                "Province_notification": ["Kinshasa", "Kinshasa"],
                "Zone_de_sante_notification": ["Gombe", "Kintambo"],
                "Date_notification": pd.to_datetime(["2026-01-05", None]),
                "Age_en_ans": [10, None],
                "Tranche_age": ["5-14 ans", None],
                "Semaine_epid": ["2026-W02", None],
                "Commentaire_libre": ["a", "b"],
            }
        )

        report = build_mapping_quality_report(
            df,
            mapping={
                "Province_notification": "Province",
                "Zone_de_sante_notification": "ZS",
                "Date_notification": "Date_notif",
                "Age_en_ans": "Age_ans",
            },
            derived_info={
                "original_columns": ["Province", "ZS", "Date_notif", "Age_ans", "Commentaire_libre", "Autre_col"],
                "weeks_calculated": 1,
                "age_groups_calculated": 1,
            },
        )

        self.assertEqual(report["Nombre de lignes"], 2)
        self.assertEqual(report["Nombre de colonnes sources"], 3)
        self.assertEqual(report["Nombre de colonnes non reconnues"], 2)
        self.assertEqual(report["Dates valides"]["valid"], 1)
        self.assertEqual(report["Âges valides"]["valid"], 1)
        self.assertIn("Date_debut_maladie", report["Colonnes importantes absentes"])
        self.assertNotIn("Semaine_epid", report["Colonnes importantes absentes"])


class ColumnMappingExportTest(unittest.TestCase):
    def test_dataframe_to_standardized_excel_bytes_returns_non_empty_workbook(self):
        df = pd.DataFrame(
            {
                "Province_notification": ["Kinshasa"],
                "Zone_de_sante_notification": ["Gombe"],
                "Semaine_epid": ["2026-W02"],
            }
        )

        payload = dataframe_to_standardized_excel_bytes(df)

        self.assertGreater(len(payload), 100)
        self.assertTrue(payload.startswith(b"PK"))


if __name__ == "__main__":
    unittest.main()
