import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard_app.domain import (
    build_standard_followup_tables,
    build_standard_signal_table,
    build_standard_surveillance_chain_table,
    standardize_df,
    standardize_ll_by_disease,
)


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
        files = sorted(line_list_dir.glob("*.xlsx"))
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

                checks_total += 4

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

        success_rate = checks_passed / checks_total if checks_total else 0.0
        self.assertGreaterEqual(
            success_rate,
            0.90,
            f"Taux de reussite insuffisant sur les line lists reelles: {success_rate:.1%}. "
            f"Echecs: {' | '.join(failures)}",
        )


if __name__ == "__main__":
    unittest.main()
