import unittest

import pandas as pd

from dashboard_app.tabs.cousp import (
    _cousp_build_temporal_series,
    _cousp_enrich_anomalies_with_reference_data,
)


class TestCouspTabHelpers(unittest.TestCase):
    def test_build_temporal_series_groups_by_day_and_month(self):
        df = pd.DataFrame(
            {
                "Date_notification": [
                    "2026-05-10 09:30:00",
                    "2026-05-10 14:20:00",
                    "2026-05-11 08:10:00",
                    "2026-06-01 10:00:00",
                ]
            }
        )

        by_day = _cousp_build_temporal_series(df, grain="jour")
        self.assertEqual(by_day["Cas"].tolist(), [2, 1, 1])
        self.assertEqual(by_day["Libelle"].tolist(), ["2026-05-10", "2026-05-11", "2026-06-01"])

        by_month = _cousp_build_temporal_series(df, grain="mois")
        self.assertEqual(by_month["Cas"].tolist(), [3, 1])
        self.assertEqual(by_month["Libelle"].tolist(), ["2026-05", "2026-06"])

    def test_build_temporal_series_sorts_epi_weeks(self):
        df = pd.DataFrame(
            {
                "Semaine_epid": [
                    "S10-2026",
                    "S02-2026",
                    "S02-2026",
                    "2026-W03",
                ]
            }
        )

        by_week = _cousp_build_temporal_series(df, grain="semaine")

        self.assertEqual(by_week["Libelle"].tolist(), ["S02-2026", "2026-W03", "S10-2026"])
        self.assertEqual(by_week["Cas"].tolist(), [2, 1, 1])

    def test_enrich_anomalies_with_reference_data_uses_alert_id(self):
        anomalies_df = pd.DataFrame(
            {
                "N_alerte": ["ALT-001", "ALT-002"],
                "Variable_anomalie": [
                    "delai_symptomes_notification",
                    "delai_notification_investigation",
                ],
            }
        )
        reference_df = pd.DataFrame(
            {
                "N_alerte": ["ALT-001", "ALT-002"],
                "Date_notification": ["2026-05-10", "2026-05-11"],
                "Semaine_epid": ["S19-2026", "S19-2026"],
            }
        )

        enriched = _cousp_enrich_anomalies_with_reference_data(anomalies_df, reference_df)

        self.assertIn("Date_notification", enriched.columns)
        self.assertIn("Semaine_epid", enriched.columns)
        self.assertEqual(enriched.loc[0, "Date_notification"], "2026-05-10")
        self.assertEqual(enriched.loc[1, "Semaine_epid"], "S19-2026")


if __name__ == "__main__":
    unittest.main()
