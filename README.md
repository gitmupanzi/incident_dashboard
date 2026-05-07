# Incident Dashboard

Application Streamlit pour la standardisation, l'analyse et la visualisation de donnees epidemiologiques de surveillance.

Le projet cible principalement trois besoins operationnels :

- analyser des line lists de surveillance
- exploiter des donnees de laboratoire integrees aux line lists
- produire des lectures hebdomadaires a partir des donnees IDSR agregees

L'objectif est de fournir un cadre unique, reutilisable et plus robuste pour transformer des fichiers heterogenes en sorties analytiques directement exploitables pour la decision.

## Ce que fait l'application

- standardise les colonnes issues de plusieurs formats de line lists
- derive des variables communes :
  `Annee_epid`, `Num_semaine_epid`, `Semaine_epid`, `Age_en_ans`, `Tranche_age`
- calcule des indicateurs standards de surveillance :
  cas, deces, letalite, completude, promptitude, positivite labo, alertes hebdomadaires
- propose des vues dediees pour :
  surveillance, profil epidemiologique, qualite des donnees, SITREP, IDSR, IREP et cartographie
- exporte des resultats en CSV, Excel et, selon les modules, en PDF

## Perimetre analytique

### 1. Line lists de surveillance

Le dashboard prend en charge plusieurs types de listes lineaires, notamment :

- cholera
- rougeole
- mpox
- Ebola / MVE
- meningite
- intoxication
- autre line list via mapping manuel des colonnes

### 2. Donnees de laboratoire

Les champs de laboratoire sont integres au pipeline analytique quand ils existent dans les fichiers sources :

- prelevement
- date de prelevement
- date de reception labo
- date de resultat
- resultat TDR
- resultat labo
- type de prelevement

### 3. Donnees IDSR agregees

L'onglet IDSR est dedie aux donnees hebdomadaires agregees par maladie et par niveau geographique. Il permet notamment :

- l'analyse des cas et deces par semaine
- la lecture de la completude de rapportage
- le calcul du taux d'attaque et de l'incidence
- la production de tableaux de synthese par province et zone de sante

## Fonctionnalites principales

- chargement des donnees depuis un fichier local, un fichier inclus dans le projet ou PostgreSQL
- normalisation des noms de colonnes et des valeurs courantes
- calcul automatique des semaines epidemiologiques ISO
- harmonisation des variables geographiques
- controle qualite des donnees
- visualisations interactives Plotly
- cartographie a partir de fichiers GeoJSON
- outils d'analyse avancee : alertes, clusters spatio-temporels, score de risque operationnel, IREP

## Structure du projet

```text
incident_dashboard/
|-- incident_dashboard.py        # point d'entree Streamlit principal
|-- dashboard_app/
|   |-- app_loader.py           # chargement des sources et PostgreSQL
|   |-- column_mapping.py       # mapping automatique / manuel des colonnes
|   |-- core.py                 # helpers transverses, graphiques, export
|   |-- domain.py               # logique metier et standardisation
|   |-- advanced.py             # calculs avances et cache Excel
|   |-- overview.py             # synthese et aides a la lecture
|   |-- narratives.py           # generation de textes d'interpretation
|   |-- runtime_support.py      # contexte partage
|   |-- tabs/                   # rendu des onglets Streamlit
|-- data/                       # referentiels et geometries
|-- line_list/                  # fichiers de demonstration / travail local
|-- tests/                      # tests unitaires
|-- requirements.txt
```

## Modes de chargement

### Line lists

Pour les maladies de type line list, l'application permet de charger les donnees de trois manieres :

- televersement d'un fichier `.xlsx`, `.xls` ou `.csv`
- selection d'un fichier deja present dans `line_list/`
- lecture depuis PostgreSQL

### IDSR

Pour l'onglet IDSR, le chargement est volontairement plus strict :

- televersement d'un classeur Excel IDSR
- selection d'un classeur IDSR valide present dans `line_list/`

Le chargeur IDSR verifie maintenant que le classeur ressemble bien a un fichier IDSR agrege avant de lancer l'analyse.

## Prerequis

- Python 3.9 ou plus recent
- pip
- environnement virtuel recommande

## Installation

### 1. Creer un environnement virtuel

```bash
python -m venv .venv
```

### 2. Activer l'environnement

Sous Windows PowerShell :

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Installer les dependances

```bash
pip install -r requirements.txt
```

## Lancer l'application

```bash
streamlit run .\incident_dashboard.py
```

## Executer les tests

```bash
python -m unittest discover -s tests -v
```

## Donnees attendues

### Pour les line lists

Les analyses sont plus fiables si les fichiers contiennent au minimum :

- une localisation de notification :
  `Province_notification`, `Zone_de_sante_notification`
- un repere temporel :
  `Date_notification` ou le couple `Annee_epid` + `Num_semaine_epid`

Selon les analyses, d'autres variables sont utiles :

- `Date_debut_maladie`
- `Age`, `Unite_age`
- `Sexe`
- `Issue`
- `Classification_finale`
- variables de laboratoire

### Pour l'option `Autre`

Si le fichier ne suit pas un schema connu, l'application propose :

- une detection automatique des colonnes
- une validation manuelle des correspondances
- la sauvegarde de profils de mapping reutilisables

### Pour IDSR

Un fichier IDSR valide doit fournir, selon le format disponible, des informations de type :

- province
- zone de sante
- semaine epidemiologique ou debut de semaine
- cas et/ou deces
- population

## Donnees geographiques

Les cartes detaillees sont disponibles si les fichiers necessaires sont presents dans `data/`, notamment :

- `geometry_rdc_provinces.geojson`
- `geometry_rdc_zones_sante.geojson`
- fichiers de correspondance geographique

Le dashboard prend en charge :

- la jointure directe sur les libelles
- la jointure fuzzy pour rapprocher des noms proches

## Cas d'usage

- suivi hebdomadaire des flambees
- surveillance multi-maladies
- analyses descriptives par age, sexe, lieu et temps
- lecture rapide de la situation pour les reunions de coordination
- verification de la qualite des donnees avant diffusion
- preparation de tableaux, graphiques et exports pour SITREP

## Qualite et fiabilite

Le projet contient des tests unitaires sur les briques critiques :

- standardisation des schemas de donnees
- mapping de colonnes
- calcul des indicateurs
- logique IDSR
- fonctions d'aide a la decision comme IREP et alertes

Si vous modifiez les regles de standardisation ou les conventions de colonnes, il est recommande d'executer la suite de tests avant toute mise en production.

## Limites actuelles

- la qualite des sorties depend directement de la qualite des colonnes sources
- certaines analyses avancées supposent des champs bien renseignes sur les dates et la geographie
- les mappings multi-maladies reposent encore sur des conventions de noms qui doivent etre entretenues

## Point d'entree principal

- application principale : [incident_dashboard.py](./incident_dashboard.py)
- logique metier : [dashboard_app/domain.py](./dashboard_app/domain.py)
- IDSR : [dashboard_app/tabs/idsr.py](./dashboard_app/tabs/idsr.py)
- mapping de colonnes : [dashboard_app/column_mapping.py](./dashboard_app/column_mapping.py)
