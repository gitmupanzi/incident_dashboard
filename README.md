# Incident Dashboard

Application Streamlit pour la standardisation, l'analyse et la visualisation de donnees epidemiologiques de surveillance.

Le projet sert de cadre unique pour :

- charger des line lists heterogenes de surveillance
- standardiser les colonnes et les valeurs vers un schema commun
- produire des lectures analytiques multi-maladies directement utilisables pour la decision
- exploiter des donnees de laboratoire integrees aux line lists
- analyser des donnees IDSR agregees hebdomadaires

## Ce que fait l'application

L'application combine plusieurs briques dans une meme interface :

- standardisation multi-maladies des line lists
- calcul automatique de variables communes :
  `Annee_epid`, `Num_semaine_epid`, `Semaine_epid`, `Age_en_ans`, `Tranche_age`,
  `Classification_finale_std`, `Issue_std`, `investigated_oui_non`
- calcul de delais standards COUSP :
  `delai_onset_to_notif`, `delai_notif_to_invest`, `delai_notif_to_prel`,
  `delai_prel_to_receipt`, `delai_receipt_to_result`, `delai_notif_to_adm`,
  `delai_adm_to_issue`
- analyses descriptives Temps-Lieu-Personne
- suivi standardise de la chaine alerte -> investigation -> prelevement -> laboratoire -> prise en charge
- controle qualite, detection de doublons et revue de coherence
- tableaux standard de cas a relancer dans la chaine de surveillance
- generation d'une synthese COUSP orientee decision
- calcul d'un indice composite IREP
- cartographie statique et interactive
- lecture IDSR agregee avec completude, incidence, taux d'attaque et tableaux standards

## Logique standard COUSP

Le projet vise un cadre de lecture standard multi-maladies pour les listes lineaires de surveillance. La logique analytique privilegie les blocs suivants :

- `Alerte` : `N_alerte`, `Source_alerte`, `Localite` quand ils existent
- `Notification` : cas effectivement presents dans le perimetre filtre
- `Investigation` : `Investigation`, `Date_investigation`, `Classification_investigation` ou, a defaut, une `Classification_finale` exploitable
- `Exposition` : lien epidemiologique, cas source, facteur d'exposition
- `Prelevement` : `Prelevement`, `Date_prelevement`, `Type_de_prelevement`
- `Laboratoire` : `Date_reception_labo`, `Resultat_labo`, `Resultat_final_labo` ou `TDR_Resultat`, `Date_confirmation`
- `Prise en charge / Issue` : `Hospitalisation`, `Date_admission_au_CT`, `Issue`, `Date_issue`

Cette organisation permet de rester standard meme quand certaines maladies ont moins de variables que d'autres. Les blocs absents dans la source ne cassent pas le dashboard : ils sont simplement reduits ou ignores.
Pour l'investigation, le dashboard applique une convention standard importante : une classification exploitable est consideree comme une forte preuve qu'une investigation a eu lieu, meme si la colonne `Investigation` est absente ou vide.

## Espaces analytiques de l'interface

L'interface principale expose aujourd'hui les onglets suivants :

- `Vue d'ensemble` : synthese d'accueil, KPI, cartes rapides, tendance hebdomadaire et lecture contextuelle
- `Surveillance` : dynamique epidemiologique, completude de surveillance, alertes, clusters, chaine analytique standard COUSP et delais de promptitude
- `Profil` : lecture descriptive selon le modele Temps-Lieu-Personne, pyramides, distributions age/sexe, detail labo
- `Qualite et export` : qualite des donnees, promptitude, coherence, cas a relancer, exports CSV/Excel, QC flags, doublons, tracker d'actions
- `COUSP` : synthese orientee decision, foyers prioritaires, signaux utiles a la decision et export standard
- `IREP` : indice composite de risque epidemiologique avec sorties telechargeables
- `Cartographie` : cartes detaillees par province et zone de sante
- `Methodologie` : definitions, chaines standards, denominateurs, delais, conventions analytiques et limites d'interpretation
- `IDSR` : analyse des donnees agregees hebdomadaires, distincte du flux line list

## Perimetre analytique

### 1. Line lists de surveillance

Les configurations maladies actuellement supportees incluent :

- cholera
- rougeole / rubeole
- mpox
- Ebola / MVE
- meningite
- intoxication
- autre line list via mapping manuel ou semi-automatique

Le pipeline essaie de standardiser automatiquement les variantes frequentes de colonnes, puis enrichit les champs derives necessaires aux analyses.

### 2. Donnees de laboratoire

Quand elles existent dans les fichiers sources, les variables biologiques sont integrees au pipeline :

- prelevement
- date de prelevement
- date de reception labo
- date de resultat
- resultat TDR
- resultat labo
- resultat final labo
- type de prelevement
- nom et numero de laboratoire

Les tableaux de bord calculent notamment les volumes de prelevements, les resultats positifs, negatifs, invalides, les receptions labo documentees, certains delais de chaine et plusieurs indicateurs de coherence.

### 3. Donnees IDSR agregees

L'onglet `IDSR` est dedie aux donnees hebdomadaires agregees par maladie et par niveau geographique. Il permet notamment :

- la lecture des cas et deces par semaine
- la completude de rapportage
- les tableaux hebdomadaires et mensuels
- le calcul du taux d'attaque et de l'incidence
- des controles qualite sur semaines, dates et doublons
- des exports CSV et Excel

## Sources de donnees supportees

### Pour les line lists

Le mode line list permet de charger les donnees depuis :

- un televersement local `.xlsx`, `.xls` ou `.csv`
- un fichier present dans `line_list/`
- une table ou une requete `SELECT` depuis PostgreSQL

### Pour l'option `Autre`

Si le fichier ne suit pas un schema deja connu, l'application propose :

- la detection automatique des correspondances de colonnes
- une validation ou correction manuelle des mappings
- un rapport de qualite du mapping
- l'export d'une version standardisee du fichier
- la sauvegarde et le rechargement de profils de mapping reutilisables

Les profils de mapping sont geres par `dashboard_app/column_mapping.py` et sont destines a etre persistes dans `data/mappings/` lors de leur sauvegarde.

### Pour IDSR

Le mode IDSR est volontairement plus strict :

- l'utilisateur charge un classeur Excel IDSR dans l'onglet `IDSR`
- un fichier IDSR peut aussi etre selectionne parmi les fichiers locaux disponibles
- la validation du format est effectuee avant l'analyse

En mode `idsr`, les onglets line list restent visibles mais leurs analyses detaillees ne remplacent pas le flux IDSR dedie.

## Donnees attendues

### Line lists

Les analyses sont plus robustes si les fichiers contiennent au minimum :

- une geographie de notification :
  `Province_notification`, `Zone_de_sante_notification`
- un repere temporel :
  `Date_notification` ou le couple `Annee_epid` + `Num_semaine_epid`

Selon les modules, d'autres variables sont tres utiles :

- `N_alerte`, `N_epid`, `Source_alerte`, `Localite`
- `Date_debut_maladie`
- `Date_investigation`
- `Age`, `Unite_age`
- `Sexe`
- `Issue`
- `Classification_finale`
- `Investigation` ou, a defaut, une `Classification_finale` ou une `Classification_investigation` exploitable
- `Date_prelevement`, `Date_reception_labo`, `Date_resultat`, `Date_confirmation`
- variables de laboratoire

### IDSR

Un fichier IDSR exploitable doit fournir, selon le format disponible, des informations du type :

- province
- zone de sante
- semaine epidemiologique ou debut de semaine
- cas et/ou deces
- population
- maladie

## Fonctionnalites principales

- harmonisation automatique des noms de colonnes et des valeurs usuelles
- standardisation des schemas multi-maladies
- calcul des semaines epidemiologiques ISO
- harmonisation geographique avec referentiels et jointures fuzzy
- calcul d'indicateurs standards :
  alertes documentees, cas, cas investigues, suspects, probables, confirmes,
  prelevements, receptions labo, positivite labo, deces, gueris, promptitude, completude
- calcul de la chaine standard COUSP avec denominateurs adaptes selon les colonnes disponibles
- prise en compte standard de l'investigation documentee a partir du statut, de la date ou d'une classification exploitable
- alertes hebdomadaires et clusters spatio-temporels
- tableaux standard de relance :
  cas sans investigation, suspects/probables sans prelevement, prelevements sans reception,
  receptions sans resultat, positifs sans date de confirmation, deces sans date d'issue
- score de risque operationnel et IREP
- generation de tableaux de synthese, graphiques Plotly et exports

## Denominateurs et conventions standards

Pour rester multi-maladies, le dashboard n'impose pas un seul denominateur partout. Il utilise une logique standard adaptee aux colonnes reellement disponibles :

- `Cas investigues` :
  alertes documentees si `N_alerte` existe, sinon cas filtres
- `Cas suspects / probables / confirmes` :
  cas investigues si possible, sinon cas filtres
- `Cas preleves` :
  cas suspects si la classification existe, sinon cas filtres
- `Receptions labo documentees` :
  cas preleves
- `Resultats labo disponibles` :
  receptions labo si disponibles, sinon cas preleves
- `Positifs / negatifs / invalides` :
  resultats documentes ou resultats valides selon l'indicateur
- `Promptitude` :
  uniquement les cas avec dates valides et delai non negatif

Cette convention est aussi documentee dans l'onglet `Methodologie`.

## Delais standards suivis

Le dashboard suit en priorite les delais suivants quand les dates existent :

- debut maladie -> notification
- notification -> investigation
- notification -> prelevement
- prelevement -> reception labo
- reception labo -> resultat
- notification -> admission
- admission -> issue
- debut maladie -> admission
- debut maladie -> prelevement
- prelevement -> resultat

## Exports disponibles

Selon les onglets, l'application propose des sorties comme :

- CSV line list filtree
- Excel line list standardisee
- CSV de QC flags
- CSV de doublons
- CSV/Excel de tableaux analytiques
- CSV des cas a relancer dans la chaine standard
- CSV du suivi d'actions et du score de risque

## Structure actuelle du projet

```text
incident_dashboard/
|-- incident_dashboard.py              # point d'entree Streamlit principal
|-- call_center.py                     # composant auxiliaire present dans le depot
|-- dashboard_app/
|   |-- app_loader.py                  # chargement des sources et validation PostgreSQL
|   |-- advanced.py                    # aides avancees et wrappers analytiques
|   |-- column_mapping.py              # mapping auto/manu, profils, export standardise
|   |-- core.py                        # helpers transverses, rendu, export
|   |-- domain.py                      # logique metier, standardisation, indicateurs, chaine COUSP, QC
|   |-- narratives.py                  # textes de lecture et narration
|   |-- overview.py                    # synthese, KPI, cartes et composants d'accueil
|   |-- runtime_support.py             # contexte partage entre onglets
|   |-- tabs/
|   |   |-- overview_detail.py         # vue d'ensemble detaillee
|   |   |-- surveillance.py            # surveillance, chaine standard, delais, alertes, clusters
|   |   |-- profile.py                 # analyses Temps-Lieu-Personne
|   |   |-- quality.py                 # qualite, coherence, cas a relancer, extraction et export
|   |   |-- cousp.py                   # synthese COUSP orientee decision
|   |   |-- irep.py                    # indice composite IREP
|   |   |-- maps.py                    # cartographie detaillee
|   |   |-- methodology.py             # definitions, chaines standard, denominateurs et limites
|   |   |-- idsr.py                    # flux IDSR hebdomadaire agrege
|-- data/
|   |-- geometry_rdc_provinces.geojson
|   |-- geometry_rdc_zones_sante.geojson
|   |-- RDC_Zone_de_sante_OCHA.xlsx
|-- line_list/                         # fichiers locaux d'exemple / travail
|-- tests/                             # tests unitaires et fixture IDSR
|-- requirements.txt
```

## Jeux de donnees presentes dans le depot

### `line_list/`

Le depot contient deja plusieurs exemples de travail local, notamment :

- `rdc_compilation_LL_Cholera_...xlsx`
- `rdc_compilation_LL_Meningite_...xlsx`
- `rdc_compilation_LL_Rougeole_Rubeole_...xlsx`
- `SEM20.xlsx`

### `data/`

Les ressources geographiques presentes servent a la cartographie et aux rapprochements geographiques :

- `geometry_rdc_provinces.geojson`
- `geometry_rdc_zones_sante.geojson`
- `RDC_Zone_de_sante_OCHA.xlsx`

## Tests et fiabilite

Le projet contient une suite de tests orientee sur les briques critiques :

- standardisation des line lists et schema commun
- mapping de colonnes et profils de mapping
- indicateurs epidemiologiques
- chaine analytique standard COUSP et cas a relancer
- chargement et validation IDSR
- calculs IREP
- alertes, clusters et contrats de schema standards

Principaux fichiers de test :

- `tests/test_dashboard_domain.py`
- `tests/test_column_mapping.py`
- `tests/IDSR.xlsx`

Execution :

```bash
python -m unittest discover -s tests -v
```

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

## Cas d'usage couverts

- suivi hebdomadaire de flambees et d'incidents de sante publique
- surveillance multi-maladies basee sur line lists
- lecture standardisee COUSP pour revues de surveillance multi-maladies
- preparation de lectures pour reunions de coordination
- production de tableaux et exports pour diffusion
- identification rapide des ruptures de chaine investigation -> prelevement -> labo -> issue
- verification de la qualite des donnees avant partage
- generation rapide d'outils d'aide a la decision multi-maladies

## Limites et points d'attention

- la qualite des sorties depend directement de la qualite des colonnes sources
- certaines analyses exigent des dates et une geographie suffisamment renseignees
- certaines sources ne distinguent pas explicitement `N_alerte` et `N_epid`, ce qui limite la lecture purement orientee alerte
- le mode `Autre` reste dependant de la qualite du mapping utilisateur
- les analyses IDSR et les analyses line list suivent des logiques differentes et ne doivent pas etre confondues
- les conventions de mapping multi-maladies doivent etre maintenues au fil des nouveaux formats sources

## Fichiers d'entree principaux

- application principale : [incident_dashboard.py](./incident_dashboard.py)
- logique metier : [dashboard_app/domain.py](./dashboard_app/domain.py)
- synthese et accueil : [dashboard_app/overview.py](./dashboard_app/overview.py)
- mapping de colonnes : [dashboard_app/column_mapping.py](./dashboard_app/column_mapping.py)
- onglet IDSR : [dashboard_app/tabs/idsr.py](./dashboard_app/tabs/idsr.py)
