import sys
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
    standardize_df,
    standardize_ll_by_disease,
)


def guess_disease_key(path: Path) -> str:
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


def main() -> int:
    files = sorted(
        path for path in (PROJECT_ROOT / "line_list").glob("*.xlsx")
        if "sem21_rdc" not in path.name.lower()
    )
    if not files:
        print("Aucune line list .xlsx n'a ete trouvee dans line_list.")
        return 1

    checks_passed = 0
    checks_total = 0
    failures: list[str] = []

    print("Audit standard des analyses activables")
    print("-" * 60)

    for path in files:
        disease_key = guess_disease_key(path)
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
        ok_count = 0

        if len(audit) == 1:
            checks_passed += 1
            ok_count += 1
        else:
            failures.append(f"{path.name}: audit structure indisponible")

        if not audit.empty and int(audit.iloc[0]["Nombre_lignes"]) == len(standardized):
            checks_passed += 1
            ok_count += 1
        else:
            failures.append(f"{path.name}: audit incoherent sur le nombre de lignes")

        if not matrix.empty and {"Bloc analytique", "Statut", "Score activation (%)"}.issubset(matrix.columns):
            checks_passed += 1
            ok_count += 1
        else:
            failures.append(f"{path.name}: matrice de capacites indisponible")

        available_or_partial = matrix["Statut"].isin(["Disponible", "Partiel"]).sum() if not matrix.empty else 0
        if available_or_partial >= 4:
            checks_passed += 1
            ok_count += 1
        else:
            failures.append(f"{path.name}: trop peu de briques activables ({available_or_partial})")

        if isinstance(classification_audit, pd.DataFrame) and not classification_audit.empty:
            checks_passed += 1
            ok_count += 1
        else:
            failures.append(f"{path.name}: audit classification indisponible")

        if isinstance(care_issue_audit, pd.DataFrame) and not care_issue_audit.empty:
            checks_passed += 1
            ok_count += 1
        else:
            failures.append(f"{path.name}: audit PEC/issue indisponible")

        if isinstance(symptom_audit, pd.DataFrame):
            checks_passed += 1
            ok_count += 1
        else:
            failures.append(f"{path.name}: audit symptomes indisponible")

        status = "OK" if ok_count == 7 else "PARTIEL"
        print(f"{status:8} {path.name} | controles valides: {ok_count}/7 | briques actives ou partielles: {available_or_partial}")

    success_rate = checks_passed / checks_total if checks_total else 0.0
    print("-" * 60)
    print(f"Taux de reussite global: {success_rate:.1%}")

    if failures:
        print("Echecs:")
        for item in failures:
            print(f"- {item}")

    return 0 if success_rate >= 0.80 else 1


if __name__ == "__main__":
    raise SystemExit(main())
