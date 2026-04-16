# Dashboard de Gestion des Données Épidémiologiques

Ce projet est un **dashboard Streamlit** destiné à la **gestion, la standardisation, l’analyse et la visualisation** des données épidémiologiques de surveillance.

Il prend en charge :

- les **listes linéaires** (données individuelles) : choléra, rougeole, mpox, Ebola, intoxication, etc.
- les **données agrégées IDSR** (hebdomadaires)

L’objectif est de fournir un **cadre unique, standardisé et réutilisable** pour le suivi des maladies à potentiel épidémique, avec des indicateurs cohérents et directement exploitables pour la décision opérationnelle.

---

## Fonctionnalités principales

- nettoyage et **standardisation automatique** des variables
- création de **variables dérivées communes**
  - année et semaine épidémiologique
  - âge harmonisé et tranches d’âge
  - variables géographiques standardisées
- production de **tableaux analytiques**
  - province, zone de santé, aire de santé
  - semaine épidémiologique
  - indicateurs clés de surveillance
- production de **visualisations interactives**
- export des résultats en **CSV**, **Excel** et selon les cas en **PDF**
- cartographie avec fichiers **GeoJSON** lorsque les données géographiques sont disponibles

---

## Visualisations disponibles

Le dashboard propose plusieurs visualisations pour explorer les données de manière complémentaire :

- **Situation globale des cas et des décès**  
  Vue synthétique des indicateurs principaux pour apprécier rapidement le niveau de gravité et le volume de cas observés.

- **Évolution hebdomadaire des cas** 📈  
  Suivi temporel de la dynamique épidémique afin d’identifier les pics, les ralentissements et les changements de tendance.

- **Taux de létalité par semaine** ⚠️  
  Analyse de la gravité au fil du temps à partir du rapport entre les décès et les cas notifiés.

- **Répartition des cas par province** 🗺️  
  Comparaison des provinces pour repérer rapidement les zones les plus contributrices au total de cas.

- **Analyse par zone de santé** 🏥  
  Lecture plus fine de la distribution spatiale pour appuyer le pilotage opérationnel local.

- **Tableaux croisés province × semaine** 📋  
  Vue combinée géographique et temporelle, utile pour la restitution, le suivi et l’interprétation des tendances.

- **Cartographie des cas** 🌍  
  Représentation spatiale des notifications lorsque les fichiers géographiques sont disponibles, avec possibilité de jointure fuzzy pour gérer les écarts d’écriture.

---

## Technologies utilisées

- **Python 3.9+**
- **Streamlit** - interface interactive
- **Pandas / NumPy** - manipulation des données
- **Plotly** - visualisations interactives
- **OpenPyXL** - lecture et export Excel
- **RapidFuzz** _(optionnel)_ - jointure fuzzy
- **GeoPandas** _(optionnel)_ - cartographie

---

## Structure du projet

Le code applicatif est organisé autour du package [`dashboard_app`](./dashboard_app) :

- `dashboard_app/core.py` : constantes globales, helpers communs et socle technique
- `dashboard_app/domain.py` : logique métier, standardisation, KPI, qualité et cartes
- `dashboard_app/overview.py` : synthèse, vues d’accueil et helpers transversaux
- `dashboard_app/advanced.py` : chargement avancé, IDSR, IREP et calculs spécifiques
- `incident_dashboard.py` : point d’entrée Streamlit

---

## Installation

### 1. Créer un environnement virtuel

```bash
python -m venv .venv
```

### 2. Activer l’environnement

Sous Windows PowerShell :

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Lancer l’application

```bash
streamlit run .\incident_dashboard.py
```

Une fois lancée, l’application permet de charger :

- un fichier **Excel** ou **CSV** de type line list
- un fichier **IDSR agrégé** pour les vues hebdomadaires dédiées

---

## Données géographiques

Les cartes sont disponibles si les fichiers géographiques nécessaires sont présents dans le dossier [`data`](./data).

Selon la qualité des libellés géographiques, le dashboard peut utiliser :

- une **jointure directe**
- une **jointure fuzzy** pour rapprocher des noms proches mais orthographiés différemment

---

## Cas d’usage

Ce dashboard peut être utilisé pour :

- le suivi hebdomadaire des flambées
- la surveillance multi-maladies
- l’analyse descriptive des cas
- le contrôle qualité des données
- la préparation de tableaux de bord et de restitutions opérationnelles

