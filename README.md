# Incident Dashboard

Application Streamlit pour la standardisation, l'analyse et la visualisation de donnees epidemiologiques de surveillance.

Le projet sert de cadre unique pour :

- charger des line lists heterogenes de surveillance
- standardiser les colonnes et les valeurs vers un schema commun
- produire des lectures analytiques multi-maladies directement exploitables
- exploiter des donnees de laboratoire integrees aux line lists
- analyser des donnees IDSR agregees hebdomadaires

## Demarrage rapide

### Prerequis

- Python 3.10+ recommande
- dependances de `requirements.txt`
- fichiers de travail dans `line_list/` pour les tests sur donnees reelles

### Installation

```bash
python -m venv .venv
```

Sous Windows PowerShell :

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Lancer l'application

```bash
streamlit run .\incident_dashboard.py
```

### Lancer les tests

Suite complete :

```bash
python -m unittest discover -s tests -v
```

Tests utiles sur la logique metier :

```bash
python -m unittest tests.test_dashboard_domain -v
python -m unittest tests.test_column_mapping -v
```

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
- audit standard du fichier, matrice des capacites analytiques et profil standard multi-maladies
- separation semantique standard entre investigation, classification finale, resultat laboratoire et issue clinique
- controle qualite, detection de doublons et revue de coherence
- tableaux standard de cas a relancer dans la chaine de surveillance
- synthese COUSP orientee decision
- calcul d'un indice composite IREP
- cartographie statique et interactive
- lecture IDSR agregee avec completude, incidence, taux d'attaque et tableaux standards

## Logique standard COUSP

Le dashboard applique un cadre de lecture standard multi-maladies organise autour des blocs suivants :

- `Alerte` : `N_alerte`, `Source_alerte`, `Localite` quand ils existent
- `Notification` : cas presents dans le perimetre filtre
- `Investigation` : `Investigation`, `Date_investigation`, `Classification_investigation`
- `Exposition` : lien epidemiologique, cas source, facteur d'exposition
- `Prelevement` : `Prelevement`, `Date_prelevement`, `Type_de_prelevement`
- `Laboratoire` : `Date_reception_labo`, `Resultat_labo`, `Resultat_final_labo` ou `TDR_Resultat`, `Date_confirmation`
- `Prise en charge / Issue` : `Hospitalisation`, `Date_admission_au_CT`, `Issue`, `Date_issue`

Cette organisation permet de rester robuste meme quand certaines maladies ont moins de variables que d'autres. Les blocs absents dans la source ne cassent pas le dashboard : ils sont reduits ou ignores.

## Conventions importantes sur les colonnes

### Investigation

- `Classification_investigation` est la source prioritaire pour documenter l'etape d'investigation
- une classification d'investigation exploitable est consideree comme une forte preuve qu'une investigation a eu lieu
- `Classification_finale` n'est plus utilisee comme substitut automatique de l'investigation

### Classification finale

- `Classification_finale` represente une synthese finale de cas
- elle peut contenir des statuts comme `Suspect`, `Probable`, `Confirme`, `Non cas`
- elle peut aussi, selon la source, porter des valeurs proches d'un resultat biologique

### Resultat laboratoire

Le projet doit gerer des sources heterogenes. Selon le fichier, le resultat labo peut etre porte par :

- `Resultat_labo`
- `Resultat_final_labo`
- `Resultat_final`
- `TDR_Resultat`
- parfois `Classification_finale` quand les valeurs ressemblent clairement a des resultats biologiques

### Filtres dynamiques dans l'interface

Les filtres de l'application ne supposent pas un seul nom de colonne :

- le filtre de classification peut utiliser `Classification_finale` ou `Classification_investigation` selon la colonne documentee disponible
- le filtre de resultat labo peut utiliser `Resultat_labo`, `Resultat_final_labo`, `Resultat_final` ou `TDR_Resultat`

Cette logique evite les filtres vides sur des fichiers reels qui n'emploient pas tous les memes noms de variables.

## Espaces analytiques de l'interface

L'interface principale expose aujourd'hui les onglets suivants :

- `Vue d'ensemble` : synthese d'accueil, KPI, cartes rapides, tendance hebdomadaire et lecture contextuelle
- `Notions statistiques` : explication pedagogique des notions quantitatives utilisees
- `Surveillance` : dynamique epidemiologique, completude, alertes, clusters, chaine COUSP et delais
- `Profil` : lecture descriptive selon le modele Temps-Lieu-Personne, detail labo
- `Qualite et export` : qualite des donnees, promptitude, coherence, doublons et exports
- `COUSP` : synthese orientee decision, foyers prioritaires et export standard
- `IREP` : indice composite de risque epidemiologique
- `Cartographie` : cartes detaillees par province et zone de sante
- `Methodologie` : definitions, conventions analytiques, denominateurs et limites
- `IDSR` : analyse des donnees agregees hebdomadaires

## Sources de donnees supportees

### Line lists

Le mode line list permet de charger les donnees depuis :

- un televersement local `.xlsx`, `.xls` ou `.csv`
- un fichier present dans `line_list/`
- une extraction `DHIS2 Tracker`
- une table ou une requete `SELECT` depuis PostgreSQL

### IDSR

L'onglet `IDSR` est dedie aux donnees agregees hebdomadaires. Il permet notamment :

- la lecture des cas et deces par semaine
- la completude de rapportage
- les tableaux hebdomadaires et mensuels
- le calcul du taux d'attaque et de l'incidence
- des controles qualite sur semaines, dates et doublons
- des exports CSV et Excel

## Donnees attendues

### Pour les line lists

Les analyses sont plus robustes si les fichiers contiennent au minimum :

- une geographie de notification : `Province_notification`, `Zone_de_sante_notification`
- un repere temporel : `Date_notification` ou le couple `Annee_epid` + `Num_semaine_epid`

Variables tres utiles selon les modules :

- `N_alerte`, `N_epid`, `Source_alerte`, `Localite`
- `Date_debut_maladie`
- `Date_investigation`
- `Age`, `Unite_age`
- `Sexe`
- `Issue`
- `Classification_investigation`
- `Classification_finale`
- `Investigation`
- `Date_prelevement`, `Date_reception_labo`, `Date_resultat`, `Date_confirmation`
- variables de laboratoire

### Pour IDSR

Un fichier IDSR exploitable doit fournir, selon le format disponible, des informations du type :

- province
- zone de sante
- semaine epidemiologique ou debut de semaine
- cas et/ou deces
- population
- maladie

## Denominateurs et conventions standards

Pour rester multi-maladies, le dashboard adapte les denominateurs aux colonnes reellement disponibles :

- `Cas investigues` : alertes documentees si `N_alerte` existe, sinon cas filtres
- `Cas suspects / probables / confirmes` : cas investigues si possible, sinon cas filtres
- `Cas preleves` : cas suspects si la classification existe, sinon cas filtres
- `Receptions labo documentees` : cas preleves
- `Resultats labo disponibles` : receptions labo si disponibles, sinon cas preleves
- `Positifs / negatifs / invalides` : resultats documentes ou resultats valides selon l'indicateur
- `Promptitude` : uniquement les cas avec dates valides et delai non negatif

## Delais standards suivis

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

## Structure du projet

```text
incident_dashboard/
|-- incident_dashboard.py              # point d'entree Streamlit principal
|-- call_center.py                     # composant auxiliaire present dans le depot
|-- dashboard_app/
|   |-- app_loader.py                  # chargement des sources et pre-standardisation
|   |-- advanced.py                    # wrappers et aides analytiques
|   |-- colonne_nettoyage.py           # nettoyage et renommage via reference
|   |-- column_mapping.py              # mapping auto/manu, profils, export standardise
|   |-- core.py                        # helpers transverses
|   |-- domain.py                      # logique metier, standardisation, indicateurs, QC
|   |-- narratives.py                  # narration et textes de lecture
|   |-- overview.py                    # synthese, KPI, cartes et composants d'accueil
|   |-- runtime_support.py             # contexte partage entre onglets
|   |-- tabs/
|   |   |-- overview_detail.py
|   |   |-- statistics.py
|   |   |-- surveillance.py
|   |   |-- profile.py
|   |   |-- quality.py
|   |   |-- cousp.py
|   |   |-- irep.py
|   |   |-- maps.py
|   |   |-- methodology.py
|   |   |-- idsr.py
|-- data/
|   |-- geometry_rdc_provinces.geojson
|   |-- geometry_rdc_zones_sante.geojson
|   |-- RDC_Zone_de_sante_OCHA.xlsx
|   |-- Rename_columns.xlsx
|-- line_list/
|-- tests/
|-- requirements.txt
```

## Jeux de donnees presents dans le depot

### `line_list/`

Le depot contient deja plusieurs exemples de travail local, notamment :

- `rdc_compilation_LL_Cholera_...xlsx`
- `rdc_compilation_LL_Meningite_...xlsx`
- `rdc_compilation_LL_Rougeole_Rubeole_...xlsx`
- `SEM20.xlsx`

### `data/`

Ressources geographiques principales :

- `geometry_rdc_provinces.geojson`
- `geometry_rdc_zones_sante.geojson`
- `RDC_Zone_de_sante_OCHA.xlsx`

## Tests et fiabilite

Le projet contient des tests sur les briques critiques :

- standardisation des line lists et schema commun
- mapping de colonnes et profils de mapping
- indicateurs epidemiologiques
- chaine analytique standard COUSP et cas a relancer
- chargement et validation IDSR
- calculs IREP
- alertes, clusters et contrats de schema standards

Principaux fichiers :

- `tests/test_dashboard_domain.py`
- `tests/test_column_mapping.py`
- `tests/test_line_list_regression.py`
- `tests/IDSR.xlsx`

## Cas d'usage couverts

- suivi hebdomadaire de flambees et d'incidents de sante publique
- surveillance multi-maladies basee sur line lists
- lecture standardisee COUSP pour revues de surveillance
- preparation de lectures pour reunions de coordination
- production de tableaux et exports pour diffusion
- identification rapide des ruptures de chaine investigation -> prelevement -> labo -> issue
- verification de la qualite des donnees avant partage
- generation rapide d'outils d'aide a la decision multi-maladies

## Limites et points d'attention

- la qualite des sorties depend directement de la qualite des colonnes sources
- certaines analyses exigent des dates et une geographie suffisamment renseignees
- certaines sources ne distinguent pas explicitement `N_alerte` et `N_epid`
- le mode `Autre` reste dependant de la qualite du mapping utilisateur
- les extractions DHIS2 tres riches peuvent produire des colonnes proches ou redondantes
- les analyses IDSR et les analyses line list suivent des logiques differentes
- les conventions de mapping multi-maladies doivent etre maintenues au fil des nouveaux formats

## Depannage

### Warning Streamlit / Plotly

Si un warning de ce type apparait :

```text
The keyword arguments have been deprecated and will be removed in a future release.
Use `config` instead to specify Plotly configuration options.
```

la cause la plus frequente est l'envoi d'arguments non supportes directement a `st.plotly_chart`.
Dans ce projet, la hauteur des graphiques doit etre appliquee sur la figure Plotly elle-meme, pas via `st.plotly_chart(height=...)`.

### GDAL_DATA non defini

Si vous voyez :

```text
Cannot find gdalvrt.xsd (GDAL_DATA is not defined)
```

cela vient de l'environnement geospatial local. Le dashboard peut continuer a fonctionner, mais certaines operations cartographiques dependront d'une installation GDAL correctement configuree.

## Fichiers d'entree principaux

- application principale : [incident_dashboard.py](./incident_dashboard.py)
- chargement et pipeline des sources : [dashboard_app/app_loader.py](./dashboard_app/app_loader.py)
- nettoyage / renommage des colonnes : [dashboard_app/colonne_nettoyage.py](./dashboard_app/colonne_nettoyage.py)
- logique metier : [dashboard_app/domain.py](./dashboard_app/domain.py)
- synthese et accueil : [dashboard_app/overview.py](./dashboard_app/overview.py)
- mapping de colonnes : [dashboard_app/column_mapping.py](./dashboard_app/column_mapping.py)
- onglet IDSR : [dashboard_app/tabs/idsr.py](./dashboard_app/tabs/idsr.py)
