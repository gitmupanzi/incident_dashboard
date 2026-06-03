import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard_app.domain import (
    build_standard_analysis_capability_matrix,
    build_standard_classification_audit,
    build_standard_care_issue_audit,
    build_standard_file_structure_audit,
    build_standard_symptom_audit,
    compute_indicators,
    build_standard_followup_tables,
    build_standard_signal_table,
    build_standard_surveillance_chain_table,
    standard_data_quality_summary,
    standardize_df,
    standardize_ll_by_disease,
)
from dashboard_app.overview import build_dashboard_kpi_payload, build_simple_lab_table


def _guess_disease_key(path: Path) -> str:
    name = path.name.lower()
    if "ebola" in name:
        return "ebola"
    if "cholera" in name:
        return "cholera"
    if "meningite" in name:
        return "meningite"
    if "rougeole" in name or "rubeole" in name:
        return "rougeole"
    return "autre"


class RealLineListRegressionTest(unittest.TestCase):
    def test_standard_chain_keeps_valid_rates_on_real_line_lists(self):
        line_list_dir = PROJECT_ROOT / "line_list"
        files = sorted(
            path for path in line_list_dir.glob("*.xlsx")
            if "sem21_rdc" not in path.name.lower()
        )
        self.assertTrue(files, "Aucune line list .xlsx n'a ete trouvee dans le dossier line_list.")

        checks_passed = 0
        checks_total = 0
        failures: list[str] = []

        for path in files:
            disease_key = _guess_disease_key(path)
            with self.subTest(file=path.name, disease=disease_key):
                with pd.ExcelFile(path) as xls:
                    raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
                standardized = standardize_df(standardize_ll_by_disease(raw, disease_key))

                chain = build_standard_surveillance_chain_table(standardized)
                followup_summary, followup_detail = build_standard_followup_tables(standardized)
                signals = build_standard_signal_table(standardized)
                indicators = compute_indicators(standardized)
                lab_table = build_simple_lab_table(standardized)
                quality_summary = standard_data_quality_summary(standardized)
                payload = build_dashboard_kpi_payload(standardized)

                checks_total += 8

                if not chain.empty:
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: chaine standard vide")

                rate_col = next((col for col in chain.columns if "Taux" in str(col)), None)
                invalid_rates = pd.DataFrame()
                if rate_col is not None and not chain.empty:
                    rate_series = pd.to_numeric(chain[rate_col], errors="coerce")
                    invalid_rates = chain[(rate_series < 0) | (rate_series > 100)].copy()
                if invalid_rates.empty:
                    checks_passed += 1
                else:
                    failures.append(
                        f"{path.name}: taux hors bornes -> "
                        + ", ".join(
                            f"{row['Indicateur']}={row[rate_col]}"
                            for _, row in invalid_rates.iterrows()
                        )
                    )

                if isinstance(followup_summary, pd.DataFrame) and isinstance(followup_detail, pd.DataFrame):
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: tables de relance indisponibles")

                if isinstance(signals, pd.DataFrame):
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: table de signaux indisponible")

                indicator_values = [
                    indicators.get("prelev_pct"),
                    indicators.get("tdr_pct"),
                    indicators.get("hosp_pct"),
                    indicators.get("cfr_pct"),
                    indicators.get("invalid_pct"),
                ]
                invalid_indicator_values = [
                    value for value in indicator_values
                    if pd.notna(value) and (float(value) < 0 or float(value) > 100)
                ]
                if not invalid_indicator_values:
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: KPI hors bornes -> {invalid_indicator_values}")

                if isinstance(lab_table, pd.DataFrame):
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: table labo profil indisponible")

                if isinstance(quality_summary, pd.DataFrame):
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: résumé qualité indisponible")

                quality_focus = payload.get("quality_focus", []) if isinstance(payload, dict) else []
                delay_focus = payload.get("delay_focus", []) if isinstance(payload, dict) else []
                quality_ok = isinstance(payload, dict) and all(
                    pd.notna(item.get("value")) and 0.0 <= float(item.get("value", 0.0)) <= 100.0
                    for item in quality_focus
                )
                delay_ok = isinstance(payload, dict) and all(
                    float(item.get("median_days", 0.0)) >= 0.0
                    and 0.0 <= float(item.get("pct_within_target", 0.0)) <= 100.0
                    for item in delay_focus
                )
                if quality_ok and delay_ok:
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: payload accueil incohérent")

        success_rate = checks_passed / checks_total if checks_total else 0.0
        self.assertGreaterEqual(
            success_rate,
            0.90,
            f"Taux de reussite insuffisant sur les line lists reelles: {success_rate:.1%}. "
            f"Echecs: {' | '.join(failures)}",
        )

    def test_standard_analysis_audit_stays_usable_on_real_line_lists(self):
        line_list_dir = PROJECT_ROOT / "line_list"
        files = sorted(
            path for path in line_list_dir.glob("*.xlsx")
            if "sem21_rdc" not in path.name.lower()
        )
        self.assertTrue(files, "Aucune line list .xlsx n'a ete trouvee dans le dossier line_list.")

        checks_passed = 0
        checks_total = 0
        failures: list[str] = []

        for path in files:
            disease_key = _guess_disease_key(path)
            with self.subTest(file=path.name, disease=disease_key):
                with pd.ExcelFile(path) as xls:
                    raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
                    sheet_name = xls.sheet_names[0]
                standardized = standardize_df(standardize_ll_by_disease(raw, disease_key))

                audit = build_standard_file_structure_audit(
                    standardized,
                    source_name=path.name,
                    sheet_name=sheet_name,
                )
                matrix = build_standard_analysis_capability_matrix(standardized)
                classification_audit = build_standard_classification_audit(standardized)
                care_issue_audit = build_standard_care_issue_audit(standardized)
                symptom_audit = build_standard_symptom_audit(standardized)

                checks_total += 7

                if len(audit) == 1:
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: audit structure indisponible")

                if not audit.empty and int(audit.iloc[0]["Nombre_lignes"]) == len(standardized):
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: audit structure incoherent sur le nombre de lignes")

                if not matrix.empty and {"Bloc analytique", "Statut", "Score activation (%)"}.issubset(matrix.columns):
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: matrice de capacites indisponible")

                available_or_partial = matrix["Statut"].isin(["Disponible", "Partiel"]).sum() if not matrix.empty else 0
                if available_or_partial >= 4:
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: trop peu de briques activables ({available_or_partial})")

                if isinstance(classification_audit, pd.DataFrame) and not classification_audit.empty:
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: audit classification indisponible")

                if isinstance(care_issue_audit, pd.DataFrame) and not care_issue_audit.empty:
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: audit PEC/issue indisponible")

                if isinstance(symptom_audit, pd.DataFrame):
                    checks_passed += 1
                else:
                    failures.append(f"{path.name}: audit symptomes indisponible")

        success_rate = checks_passed / checks_total if checks_total else 0.0
        self.assertGreaterEqual(
            success_rate,
            0.80,
            f"Taux de reussite insuffisant pour l'audit standard: {success_rate:.1%}. "
            f"Echecs: {' | '.join(failures)}",
        )


if __name__ == "__main__":
    unittest.main()
