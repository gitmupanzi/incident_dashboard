"""API canonique transverse et standard multi-maladies.

Ce module centralise les points d'entrée publics utilisés par les onglets,
la vue d'ensemble et les tests pour les audits transversaux et les résumés
sémantiques standards. Les implémentations vivent encore dans
``dashboard_app.domain`` pour limiter les changements structurels, mais les
nouveaux appelants doivent désormais importer depuis ce module plutôt que
depuis ``domain`` directement.
"""

from dashboard_app.domain import (
    build_recommended_fields_matrix,
    build_standard_analysis_capability_matrix,
    build_standard_capability_note,
    build_standard_care_issue_audit,
    build_standard_classification_audit,
    build_standard_disease_profile,
    build_standard_file_structure_audit,
    build_standard_semantic_status_summary,
    build_standard_symptom_audit,
    summarize_standard_analysis_capabilities,
)

__all__ = [
    "build_recommended_fields_matrix",
    "build_standard_analysis_capability_matrix",
    "build_standard_capability_note",
    "build_standard_care_issue_audit",
    "build_standard_classification_audit",
    "build_standard_disease_profile",
    "build_standard_file_structure_audit",
    "build_standard_semantic_status_summary",
    "build_standard_symptom_audit",
    "summarize_standard_analysis_capabilities",
]
