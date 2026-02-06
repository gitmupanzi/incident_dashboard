# DASHBOARD DE GESTION DES DONNÉES ÉPIDÉMIOLOGIQUES (Streamlit)

Ce projet est un **dashboard Streamlit** destiné à la gestion, la standardisation, l’analyse et la visualisation des **données épidémiologiques de surveillance COUSP**, incluant :

- **Listes linéaires (patient-level)** : Choléra, Rougeole, Mpox, Ebola, Intoxication, etc.
- **Données agrégées IDSR** (hebdomadaires).

Le dashboard vise à fournir un **cadre unique et standard** pour toutes les maladies à potentiel épidémique, avec des indicateurs cohérents, reproductibles et exploitables pour la décision opérationnelle et stratégique.

---

## Fonctionnalités principales

- Nettoyage et **standardisation automatique** des variables
- Création des **variables dérivées communes** :
  - Année et semaine épidémiologique (ISO)
  - Âge harmonisé et tranches d’âge
  - Provenance / localisation
- Tableaux analytiques :
  - Province / Zone de santé / Aire de santé
  - Semaine épidémiologique
- Visualisations :
  - Courbes épidémiologiques
  - Répartition par âge et sexe
  - Indicateurs clés (cas, décès, létalité, positivité)
- Export des résultats :
  - CSV
  - Excel
- **Cartographie statique (optionnelle)** :
  - GeoJSON
  - Jointure classique ou fuzzy (correction orthographique)

---

## Technologies utilisées

- **Python 3.9+**
- **Streamlit** – Interface interactive
- **Pandas / NumPy** – Manipulation des données
- **Plotly** – Visualisations interactives
- **OpenPyXL** – Export Excel
- **RapidFuzz** _(optionnel)_ – Jointure fuzzy
- **GeoPandas** _(optionnel)_ – Cartographie

---

## Installation

### 1. Créer un environnement virtuel (recommandé)

```bash
python -m venv .venv
```
