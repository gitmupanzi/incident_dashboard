"""
Socle technique du dashboard épidémiologique Streamlit.

Principes de maintenance:
- Concentrer ici les helpers transverses réutilisés par plusieurs modules
- Préserver des fonctions pures autant que possible pour faciliter les tests
- Limiter les dépendances implicites à l'état global quand un paramètre suffit

Ancres de recherche utiles dans l'éditeur:
- SECTION: CONSTANTES
- SECTION: INDICATEURS KPI
- SECTION: QUALITE DES DONNEES
- SECTION: VISUALISATIONS STREAMLIT/PLOTLY
"""

# =========================
# Incident RDC – Dashboard (Streamlit UI) + VISUALISATIONS CUSTOM + CARTES FIX (Plotly + fuzzy join)
# =========================
# Héritage du refactor depuis la version monolithique :
# - les fonctions "custom" ont été conservées dans le package applicatif
# - la logique UI détaillée est désormais répartie entre plusieurs modules de tabs
# =========================
# SECTION INDEX
# 1) Imports et constantes
# 2) Cartes et visualisations
# 3) Helpers de nettoyage / standardisation
# 4) KPI et qualité des données
# 5) Interface Streamlit
# =========================

import os
import glob
import json
import re
import unicodedata
import logging
import hashlib
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype
from datetime import date,datetime
from io import BytesIO

# =========================================================
# UTILITAIRES GÉNÉRAUX (Harmonisation & Robustesse)
# - Centralise les fonctions réutilisées dans les onglets
# - Évite les divergences entre tabs
# =========================================================

import tempfile


# =========================
# SECTION: CONSTANTES
# =========================
MISSING_LABEL = "Inconnu"
MISSING_LABEL_VERBOSE = "Inconnu/Non renseigné"
APP_BUILD_TAG = "single-file-refined-v2"
AGE_UNIT_MONTH_PATTERN = r"\b(?:mois|month)s?\b"
AGE_UNIT_WEEK_PATTERN = r"\b(?:semaine|semaines|week|weeks)\b|\bsem\b"
AGE_UNIT_DAY_PATTERN = r"\b(?:jour|jours|day|days)\b"
AGE_UNIT_YEAR_PATTERN = r"\b(?:an|ans|annee|annees|ann[eé]e?s|year|yr|yrs)\b"
METRIC_COLUMN_ALIASES: Dict[str, List[str]] = {
    "Décès": ["Décès", "Deces"],
    "Létalité (%)": ["Létalité (%)", "Letalite (%)"],
    "Positivité (%)": ["Positivité (%)", "Positivite (%)"],
    "Prélèvement (%)": ["Prélèvement (%)", "Prelevement (%)"],
    "Prélèvement_%": ["Prélèvement_%", "Prelevement_%"],
    "TDR réalisé (%)": ["TDR réalisé (%)", "TDR realise (%)"],
    "TDR_réalisé_%": ["TDR_réalisé_%", "TDR_realise_%", "TDR realise_%"],
    "Positivité TDR (%)": ["Positivité TDR (%)", "Positivite TDR (%)"],
    "Positivité_TDR_%": ["Positivité_TDR_%", "Positivite_TDR_%"],
}


def as_list(value: Union[str, List[str], Tuple[str, ...]]) -> List[str]:
    """Normalise un argument colonnes en liste."""
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def _normalize_metric_alias_columns(
    df: Optional[pd.DataFrame],
    alias_map: Optional[Dict[str, List[str]]] = None,
) -> Optional[pd.DataFrame]:
    """Renomme les variantes de colonnes métiers vers une forme canonique."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    alias_map = alias_map or METRIC_COLUMN_ALIASES
    cols = set(df.columns)
    rename_map: Dict[str, str] = {}

    for preferred, aliases in alias_map.items():
        if preferred in cols:
            continue
        for alias in aliases:
            if alias in cols:
                rename_map[alias] = preferred
                break

    return df.rename(columns=rename_map) if rename_map else df


def _is_week_axis_identifier(value: Optional[str]) -> bool:
    """Détecte si un libellé/nom de colonne correspond à une dimension hebdomadaire."""
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return any(
        token in text
        for token in ["semaine", "week", "yw", "time_key", "time_lab", "num_semaine_epid", "semaine_epid"]
    )


def _resolve_weekly_bar_spacing(
    x_identifier: Optional[str] = None,
    x_title: Optional[str] = None,
    bargap: float = 0.2,
    bargroupgap: float = 0.1,
) -> tuple[float, float]:
    """Resserre automatiquement les barres si l'axe X est hebdomadaire."""
    if _is_week_axis_identifier(x_identifier) or _is_week_axis_identifier(x_title):
        return min(float(bargap), 0.0), min(float(bargroupgap), 0.0)
    return float(bargap), float(bargroupgap)


def _scale_marker_sizes(
    values: Union[pd.Series, np.ndarray, list],
    min_size: float = 18,
    max_size: float = 220,
) -> pd.Series:
    """Calcule des tailles de marqueurs proportionnelles aux valeurs positives."""
    s = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0).clip(lower=0)
    sizes = pd.Series(0.0, index=s.index, dtype="float64")
    positive = s[s > 0]
    if positive.empty:
        return sizes
    if float(positive.max()) == float(positive.min()):
        sizes.loc[positive.index] = (min_size + max_size) / 2
        return sizes
    scaled = np.sqrt(positive / positive.max())
    sizes.loc[positive.index] = min_size + (max_size - min_size) * scaled
    return sizes


def safe_pct(num: Union[int, float], den: Union[int, float]) -> float:
    """Retourne un pourcentage en gérant les dénominateurs nuls."""
    try:
        den = float(den)
        num = float(num)
    except Exception:
        return np.nan
    if den <= 0:
        return np.nan
    return num / den * 100.0


# =========================================================
# ✅ BLOC CARTE (INTÉGRÉ) – AUTONOME
# =========================================================
MAP_ANNOTATION_MODE_OPTIONS = {
    "Aucune annotation": "aucun",
    "Noms uniquement": "nom",
    "Valeurs uniquement": "valeur",
    "Noms + valeurs": "nom_valeur",
}


def carte_statique_matplotlib(
    gdf,
    colonne_valeurs: str,
    titre: str,
    annoter: bool = True,
    mode_annotation: str = "nom_valeur",
    nom_zone: str = "name",
    fmt_valeurs: str = "{:.0f}",
    seuil_affichage: float = 1,
    cmap: str = "Reds",
    afficher_fond_carte: bool = False,
    titre_fontsize: int = 11,
    legend_titre: str = "Nombre de cas",
    legend_taille_ticks: int = 7,
    legend_taille_titre: int = 8,
    cb_height: float = 0.12,
    cb_width: float = 0.25,
    cb_shift_up: float = 0.05,
    afficher_barre_echelle: bool = True,
    longueur_barre_km: float = 50,
    afficher_boussole: bool = True,
    afficher_legende_taille: bool = True,
    figsize=(12, 10),
):
    """
    IMPORTANT Streamlit:
    - Retourne une figure Matplotlib (fig)
    - L'appelant fait st.pyplot(fig) puis plt.close(fig)
    """
    if gdf is None or gdf.empty:
        return None
    if colonne_valeurs not in gdf.columns:
        return None

    # ---- helpers ----
    def _ajouter_barre_echelle(ax, longueur_km=50, loc=(0.90, 0.04), largeur_ligne=0.8, taille_police=7):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        x_debut = x_min + (x_max - x_min) * loc[0]
        y = y_min + (y_max - y_min) * loc[1]

        longueur_m = longueur_km * 1000
        h = (y_max - y_min) * 0.005

        ax.plot([x_debut, x_debut + longueur_m], [y, y], linewidth=largeur_ligne, color="black")
        ax.plot([x_debut, x_debut], [y - h, y + h], linewidth=largeur_ligne, color="black")
        ax.plot([x_debut + longueur_m, x_debut + longueur_m], [y - h, y + h], linewidth=largeur_ligne, color="black")

        ax.text(
            x_debut + longueur_m / 2,
            y + h * 2,
            f"{longueur_km:.0f} km",
            ha="center",
            va="bottom",
            fontsize=taille_police,
        )

    def _ajouter_boussole(ax, loc=(0.95, 0.95), offset=0.08, taille_police=11):
        ax.annotate(
            "N",
            xy=loc,
            xytext=(loc[0], loc[1] - offset),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=taille_police,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", linewidth=1.2),
        )

    def _ajouter_legende_taille(ax, valeurs, tailles_points):
        valeurs_num = pd.to_numeric(pd.Series(valeurs), errors="coerce").fillna(0)
        tailles_num = pd.to_numeric(pd.Series(tailles_points), errors="coerce").fillna(0)
        masque = (valeurs_num > 0) & (tailles_num > 0)
        if not masque.any():
            return

        valeurs_pos = valeurs_num[masque]
        if valeurs_pos.empty:
            return

        quantiles = [0.25, 0.6, 1.0] if len(valeurs_pos) >= 3 else [0.5, 1.0]
        valeurs_legende = []
        for q in quantiles:
            val = float(valeurs_pos.quantile(q))
            if val <= 0:
                continue
            val_arrondi = int(round(val))
            if val_arrondi <= 0:
                val_arrondi = max(1, int(np.ceil(val)))
            valeurs_legende.append(val_arrondi)

        valeurs_legende = sorted(set(valeurs_legende))
        if not valeurs_legende:
            return

        tailles_legende = _scale_marker_sizes(valeurs_legende, min_size=24, max_size=360)
        handles = []
        for val, taille in zip(valeurs_legende, tailles_legende):
            try:
                label = fmt_valeurs.format(val)
            except Exception:
                label = str(val)
            handles.append(
                plt.scatter(
                    [],
                    [],
                    s=float(taille),
                    color="#c2410c",
                    edgecolors="white",
                    linewidths=0.8,
                    alpha=0.82,
                    label=label,
                )
            )

        if not handles:
            return

        leg = ax.legend(
            handles=handles,
            title=f"{legend_titre}\n(taille du point)",
            loc="lower left",
            bbox_to_anchor=(0.12, 0.02),
            frameon=True,
            fontsize=legend_taille_ticks,
            title_fontsize=legend_taille_titre,
            borderpad=0.6,
            labelspacing=1.0,
            scatterpoints=1,
            borderaxespad=0.0,
        )
        if leg is not None:
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_alpha(0.92)
            leg.get_frame().set_edgecolor("#d4d4d8")

    # ---- reprojection (pour échelle en mètres) ----
    try:
        if gdf.crs is None or (hasattr(gdf.crs, "to_epsg") and gdf.crs.to_epsg() != 3857):
            gdf = gdf.to_crs(epsg=3857)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=figsize)

    # ---- plot : contours legers + centroïdes proportionnels ----
    gdf_plot = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf_plot.empty:
        plt.close(fig)
        return None

    geom_types = gdf_plot.geometry.geom_type.unique()
    valeurs_num = pd.to_numeric(gdf_plot[colonne_valeurs], errors="coerce").fillna(0)

    if set(geom_types) != {"Point"}:
        try:
            gdf_plot.boundary.plot(ax=ax, color="#94a3b8", linewidth=0.8, alpha=0.9, zorder=1)
        except Exception:
            gdf_plot.plot(ax=ax, facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=0.8, alpha=0.9, zorder=1)
        centroides = gdf_plot.copy()
        centroides["geometry"] = gdf_plot.geometry.centroid
    else:
        centroides = gdf_plot.copy()

    centroides[colonne_valeurs] = valeurs_num.values
    centroides = centroides[centroides.geometry.notna() & ~centroides.geometry.is_empty].copy()
    tailles = _scale_marker_sizes(centroides[colonne_valeurs], min_size=24, max_size=360)
    centroides["_marker_size"] = tailles.values
    centroides_visibles = centroides[centroides["_marker_size"] > 0].copy()
    if centroides_visibles.empty:
        centroides_visibles = centroides.copy()
        centroides_visibles["_marker_size"] = 28

    centroides_visibles.plot(
        ax=ax,
        color="#c2410c",
        markersize=centroides_visibles["_marker_size"],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.82,
        zorder=3,
    )

    ax.set_title(titre, fontsize=titre_fontsize)
    ax.axis("off")

    # ---- fond (optionnel) ----
    if afficher_fond_carte and ctx is not None:
        try:
            ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
        except Exception:
            pass

    # ---- labels ----
    if annoter:
        mode_annotation_normalise = str(mode_annotation or "nom_valeur").strip().lower()
        inclure_nom = mode_annotation_normalise in {"nom", "noms", "nom_valeur", "noms_valeurs", "both", "label"}
        inclure_valeur = mode_annotation_normalise in {"valeur", "valeurs", "nom_valeur", "noms_valeurs", "both", "label"}
        for _, row in centroides.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                continue

            val = row[colonne_valeurs]
            if pd.isna(val) or val <= seuil_affichage:
                continue

            x, y = row.geometry.x, row.geometry.y

            parts = []
            if inclure_nom and nom_zone in gdf.columns and pd.notna(row[nom_zone]):
                zone_label = str(row[nom_zone]).strip()
                if zone_label:
                    parts.append(zone_label)
            if inclure_valeur:
                try:
                    parts.append(fmt_valeurs.format(val))
                except Exception:
                    parts.append(str(val))
            if not parts:
                continue

            ax.text(
                x, y, "\n".join(parts),
                ha="center", va="center",
                fontsize=7, color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, boxstyle="round,pad=0.15"),
            )

    # ---- échelle + boussole ----
    if afficher_barre_echelle:
        _ajouter_barre_echelle(ax, longueur_km=longueur_barre_km)
    if afficher_boussole:
        _ajouter_boussole(ax)
    if afficher_legende_taille:
        _ajouter_legende_taille(ax, centroides_visibles[colonne_valeurs], centroides_visibles["_marker_size"])

    plt.tight_layout()

    return fig


def choose_week_column(df: pd.DataFrame) -> Tuple[pd.Series, Optional[str]]:
    """Choisit la meilleure colonne semaine disponible (YW > TIME_KEY > TIME_LAB).
    Retourne: (series_label, order_key_col)
    - series_label: labels de semaine (string)
    - order_key_col: colonne qui permet un tri chronologique (ex: TIME_KEY)
    """
    if "YW" in df.columns:
        s = df["YW"].astype(str)
        order_key = "YW_KEY" if "YW_KEY" in df.columns else None
        return s, order_key
    if "TIME_KEY" in df.columns:
        return df["TIME_KEY"].astype(str), "TIME_KEY"
    if "TIME_LAB" in df.columns:
        return df["TIME_LAB"].astype(str), None
    return pd.Series(dtype="object"), None


def ordered_weeks_from_weekly_sorted(weekly_sorted: pd.DataFrame, fmt=None) -> List[str]:
    """Construit une liste ordonnée de semaines (au format normalisé) depuis weekly_sorted."""
    if fmt is None:
        fmt = fmt_yw_label

    if (weekly_sorted is None) or (not isinstance(weekly_sorted, pd.DataFrame)) or weekly_sorted.empty:
        return []

    if "YW" in weekly_sorted.columns:
        return (
            weekly_sorted[["YW"]]
            .dropna()
            .drop_duplicates()["YW"]
            .astype(str)
            .map(fmt)
            .tolist()
        )

    if "TIME_KEY" in weekly_sorted.columns:
        return (
            weekly_sorted[["TIME_KEY"]]
            .dropna()
            .drop_duplicates()
            .sort_values("TIME_KEY")["TIME_KEY"]
            .astype(str)
            .map(fmt)
            .tolist()
        )

    if "TIME_LAB" in weekly_sorted.columns:
        return (
            weekly_sorted[["TIME_LAB"]]
            .dropna()
            .drop_duplicates()["TIME_LAB"]
            .astype(str)
            .map(fmt)
            .tolist()
        )

    return []


def build_cases_deaths_cfr_pivot(df: pd.DataFrame,
                                *,
                                idx_cols: List[str],
                                week_series: pd.Series,
                                col_cases: str = "Total_cas",
                                col_deaths: str = "Total_deces",
                                week_name: str = "_YW_COL",
                                cfr_label: str = "Létalité (%)") -> pd.DataFrame:
    """Construit un pivot MultiIndex: (Cas/Décès/CFR) x semaine, index = idx_cols."""
    tmp = df.copy()
    tmp[week_name] = week_series.astype(str).map(fmt_yw_label)

    pw = tmp.groupby(idx_cols + [week_name], as_index=False).agg(
        Cas=(col_cases, "sum"),
        Décès=(col_deaths, "sum"),
    )
    pw[cfr_label] = np.where(pw["Cas"] > 0, (pw["Décès"] / pw["Cas"]) * 100.0, np.nan)

    pivot = pw.pivot_table(
        index=idx_cols,
        columns=week_name,
        values=["Cas", "Décès", cfr_label],
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )

    # Sécurité : normaliser aussi les labels semaine du pivot
    pivot = pivot.copy()
    pivot.columns = pd.MultiIndex.from_tuples(
        [(lvl0, fmt_yw_label(lvl1)) for (lvl0, lvl1) in pivot.columns],
        names=pivot.columns.names
    )
    return pivot

def fmt_yw_label(v):
    """Normalise un label Année-Semaine en format 'YYYYWww'.
    Exemples:
      - 202601 -> 2026W01
      - 2026-W1 -> 2026W01
      - 2026.1 / 2026-1 / 2026 1 -> 2026W01
    """
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if not s:
        return ""

    # Cas numérique compact: 202601 / 202604 / 202502
    if re.fullmatch(r"\d{5,6}", s):
        year = s[:4]
        week = int(s[4:])
        return f"{year}W{week:02d}"

    # Cas déjà avec W
    m = re.search(r"(\d{4}).*?W(\d{1,2})", s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}W{int(m.group(2)):02d}"

    # Cas 2026-1 / 2026.1 / 2026 1
    m = re.search(r"(\d{4})\D+(\d{1,2})$", s)
    if m:
        return f"{m.group(1)}W{int(m.group(2)):02d}"

    return s

# Compatibilité: garder l'ancien nom utilisé dans certaines sections
_fmt_yw_label = fmt_yw_label

def make_unique(cols: Iterable[str]) -> List[str]:
    """Rend une liste de colonnes unique (pyarrow/Streamlit n'aime pas les doublons)."""
    seen = {}
    out = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out

def flatten_columns(cols) -> List[str]:
    """Aplatit un MultiIndex en 'lvl0 | lvl1' pour un affichage clair dans st.dataframe."""
    flat = []
    for c in cols:
        if isinstance(c, tuple):
            flat.append(" | ".join([str(x) for x in c]))
        else:
            flat.append(str(c))
    return flat

def st_dataframe_safe(df, *, height: int = 520):
    """Affichage Streamlit robuste:
    - Aplatit MultiIndex
    - Rend colonnes uniques
    - Rend Arrow-compatible (convertit object mixtes -> string)
    """
    _df = df.copy()

    # 1) Colonnes uniques + lisibles
    _df.columns = make_unique(flatten_columns(_df.columns))

    # 2) ✅ Arrow-safe: convertir les colonnes object mixtes
    for c in _df.columns:
        if _df[c].dtype == "object":
            # bytes -> str
            _df[c] = _df[c].map(
                lambda x: x.decode("utf-8", "ignore")
                if isinstance(x, (bytes, bytearray))
                else x
            )

            # si types mixtes (int + str + ...) -> tout en string
            # (Arrow n'aime pas les colonnes object hétérogènes)
            _df[c] = _df[c].astype("string")

    # 3) Affichage
    st.dataframe(_df, width="stretch", height=height)



def render_pivot_with_cfr(pivot: pd.DataFrame,
                          *,
                          idx_reset: bool = True,
                          cfr_label: str = "Létalité (%)",
                          cfr_decimals: int = 2,
                          height: int = 520) -> None:
    """Rendu standard des pivots Cas/Décès/CFR:
    - arrondit CFR
    - reset_index pour rendre visibles Province/ZS
    - aplatit/unique les colonnes pour Streamlit
    """
    if pivot is None or pivot.empty:
        st.info("Aucune donnée disponible pour afficher ce tableau.")
        return

    pivot_display = pivot.copy()
    try:
        if (cfr_label in pivot_display.columns.get_level_values(0)):
            pivot_display = pivot_display.apply(
                lambda s: s.round(cfr_decimals)
                if (isinstance(s.name, tuple) and s.name[0] == cfr_label)
                else s
            )
    except Exception:
        # Si les colonnes ne sont pas en MultiIndex, on ignore
        pass

    if idx_reset:
        pivot_display = pivot_display.reset_index()

    st_dataframe_safe(pivot_display, height=height)


def reorder_pivot_weeks(pivot: pd.DataFrame,
                        ordered_weeks: list,
                        *,
                        fill_value: float = 0) -> pd.DataFrame:
    """Réordonne un pivot MultiIndex (lvl0=mesure, lvl1=semaine) selon ordered_weeks.
    - filtre automatiquement les semaines absentes
    - garde l'ordre des mesures
    """
    if pivot is None or pivot.empty:
        return pivot

    if not isinstance(pivot.columns, pd.MultiIndex) or pivot.columns.nlevels < 2:
        return pivot

    weeks_present = set(pivot.columns.levels[1])
    order = [w for w in ordered_weeks if w in weeks_present]
    if not order:
        return pivot

    lvl0 = list(pivot.columns.levels[0])
    return pivot.reindex(columns=pd.MultiIndex.from_product([lvl0, order]), fill_value=fill_value)


@st.cache_data(show_spinner=False)
def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

@st.cache_data(show_spinner=False)
def prepare_idsr_numeric(df: pd.DataFrame,
                         col_cases: str = "Total_cas",
                         col_deaths: str = "Total_deces") -> pd.DataFrame:
    """Prépare une DF IDSR avec colonnes numériques standardisées.
    N'altère pas la logique métier: uniquement coercition + NA -> 0.
    """
    out = df.copy()
    if col_cases in out.columns:
        out[col_cases] = _to_numeric_series(out[col_cases]).fillna(0)
    if col_deaths in out.columns:
        out[col_deaths] = _to_numeric_series(out[col_deaths]).fillna(0)
    return out

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")



def _strip_accents(text):
    if text is None or pd.isna(text):
        return text
    text = str(text)
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


def _normalize_name(s: str) -> str:
    """Normalise un nom de colonne pour des correspondances robustes.
    Gère accents, espaces, ponctuation et variations de casse.
    """
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


# Cartes
try:
    import geopandas as gpd
except Exception:
    gpd = None

# Optionnel (meilleur fuzzy)
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False
    
# matplotlib pour les couleurs


try:
    import contextily as ctx
except Exception:
    ctx = None

try:
    from shapely.geometry import MultiPolygon
    from shapely.geometry.polygon import orient
except Exception:
    MultiPolygon = None
    orient = None

# -------------------------
# LOGGER
# -------------------------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# Palette harmonisée (mêmes couleurs d'un onglet à l'autre)
COLOR_CASES = "#1F77B4"
COLOR_DEATHS = "#7F7F7F"
COLOR_CFR = "#D62728"
COLOR_MASCULIN = "#1A1E2B"
COLOR_FEMININ = "#E70B0B"
COLOR_INCONNU = "#9E9E9E"

SEX_COLOR_MAP = {
    "Masculin": COLOR_MASCULIN,
    "Feminin": COLOR_FEMININ,
    "Féminin": COLOR_FEMININ,
    "Inconnu": COLOR_INCONNU,
    "Inconue": COLOR_INCONNU,
    "Autre": "#8C564B",
}


# =========================================================
# ✅ BLOC VISUALISATIONS (INTÉGRÉ) – AUTONOME
# =========================================================

def verifier_presence_colonnes(
    df: pd.DataFrame,
    colonnes: Union[str, List[str], Tuple[str, ...]]
) -> bool:
    colonnes = as_list(colonnes)
    if not colonnes:
        logger.error("[ERREUR] Aucun nom de colonne valide fourni.")
        return False

    if df is None or not isinstance(df, pd.DataFrame):
        logger.error("[ERREUR] df n'est pas un DataFrame pandas valide.")
        return False

    for col in colonnes:
        if col not in df.columns:
            logger.error(f"[ERREUR] Colonne '{col}' non trouvée dans le DataFrame.")
            return False
    return True

def extraire_numero(x: Any) -> int:
    match = re.search(r"\d+", str(x))
    return int(match.group()) if match else -1

def compter_par_categorie(
    df: pd.DataFrame,
    colonne: str,
    seuil_min: int = 0
) -> pd.DataFrame:
    if colonne not in df.columns:
        raise ValueError(f"[ERREUR] Colonne '{colonne}' non trouvée dans le DataFrame.")
    counts = df[colonne].fillna(MISSING_LABEL).value_counts(dropna=False)
    filtered = counts[counts >= seuil_min].reset_index()
    filtered.columns = [colonne, "Nombre de cas"]
    return filtered

def plot_histogramme_groupe_interactif_empile(
    df: pd.DataFrame,
    x_col: str,
    x_titre: str,
    hue_col: str,
    y_titre: str = "Nombre de cas",
    titre: Optional[str] = None,
    rotation: int = 45,
    annot: bool = False,
    pas_x: Optional[int] = None,
    bargap: float = 0.2,
    bargroupgap: float = 0.1,
    taille_fig: Tuple[int, int] = (1500, 500),
    x_trier: bool = False,
    ordre: str = "asc",
    y_col: Optional[str] = None,
    aggfunc: str = "sum"
) -> Optional[go.Figure]:
    bargap, bargroupgap = _resolve_weekly_bar_spacing(x_col, x_titre, bargap, bargroupgap)

    if not all(col in df.columns for col in [x_col, hue_col]):
        logger.error("❌ Colonnes manquantes dans le DataFrame")
        return None

    if y_col is not None and y_col not in df.columns:
        logger.error(f"❌ La colonne de valeurs '{y_col}' n'existe pas dans le DataFrame")
        return None

    ordre = str(ordre).lower().strip()
    if ordre not in {"asc", "desc"}:
        logger.warning("[WARN] Paramètre 'ordre' invalide. Utilisation de 'asc'.")
        ordre = "asc"

    categories_x = sorted(df[x_col].dropna().unique(), key=extraire_numero)

    # 1) Mode simple
    if not x_trier:
        if y_col is None:
            fig = px.histogram(
                df,
                x=x_col,
                color=hue_col,
                barmode="stack",
                title=titre or f"Histogramme empilé de '{x_col}' par '{hue_col}'",
                labels={x_col: x_titre, hue_col: hue_col},
                category_orders={x_col: categories_x},
                histfunc="count",
            )
        else:
            fig = px.histogram(
                df,
                x=x_col,
                color=hue_col,
                y=y_col,
                barmode="stack",
                title=titre or f"Barres empilées de '{y_col}' par '{x_col}' et '{hue_col}'",
                labels={x_col: x_titre, hue_col: hue_col, y_col: y_titre},
                category_orders={x_col: categories_x},
                histfunc=aggfunc,
            )

        if annot:
            fig.update_traces(texttemplate="%{y}", textposition="outside", cliponaxis=False)

        fig.update_layout(
            xaxis_title=x_titre,
            yaxis_title=y_titre,
            bargap=bargap,
            bargroupgap=bargroupgap,
            template="plotly_white",
            xaxis_tickangle=-rotation,
            width=taille_fig[0],
            height=taille_fig[1],
        )

        if pas_x is not None:
            try:
                tickvals = [categories_x[i] for i in range(0, len(categories_x), pas_x)]
                fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=tickvals)
            except Exception:
                pass

        return fig

    # 2) Mode tri segments
    if y_col is None:
        df_agg = df.groupby([x_col, hue_col], observed=True).size().reset_index(name="valeur")
    else:
        df_agg = (
            df.groupby([x_col, hue_col], observed=True)[y_col]
              .agg(aggfunc)
              .reset_index(name="valeur")
        )

    if df_agg.empty:
        logger.info("[INFO] Aucun résultat après agrégation.")
        return None

    fig = go.Figure()
    ascending = True if ordre == "asc" else False
    first_x = categories_x[0] if categories_x else None

    for x_val in categories_x:
        sous_df = df_agg[df_agg[x_col] == x_val].copy()
        if sous_df.empty:
            continue

        sous_df = sous_df.sort_values("valeur", ascending=ascending)
        cumul = 0
        for _, row in sous_df.iterrows():
            fig.add_trace(go.Bar(
                x=[x_val],
                y=[row["valeur"]],
                name=str(row[hue_col]),
                offsetgroup=str(x_val),
                base=cumul,
                text=[row["valeur"]] if annot else None,
                textposition="inside" if annot else "none",
                showlegend=bool(x_val == first_x),
            ))
            cumul += row["valeur"]

    fig.update_layout(
        barmode="stack",
        bargap=bargap,
        bargroupgap=bargroupgap,
        xaxis_title=x_titre,
        yaxis_title=y_titre,
        title=titre,
        template="plotly_white",
        width=taille_fig[0],
        height=taille_fig[1],
        xaxis_tickangle=-rotation,
    )

    if pas_x is not None:
        try:
            tickvals = [categories_x[i] for i in range(0, len(categories_x), pas_x)]
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=tickvals)
        except Exception:
            pass

    return fig

def graphique_barres_facette(
    df: pd.DataFrame,
    x_col: str = "Num_semaine_epid",
    x_titre: str = "Semaine épidémiologique",
    y_col: str = "Cases",
    y_titre: str = "Nombre de cas",
    facette_col: str = "Province",
    titre: Optional[str] = "Répartition des cas",
    taille_fig: Tuple[int, int] = (1600, 600),
    rotation: int = 45,
    couleurs_personnalisees: Optional[Union[str, dict]] = None,
    bargap: float = 0.2,
    bargroupgap: float = 0.1,
    annot: bool = False,
    pas_x: Optional[int] = None,
    auto_aggregate: bool = True,
    filtre_valeur: Optional[str] = None,
    return_fig: bool = False,
    encadrer_facettes: bool = True,
    couleur_contour_facette: str = "#E6E6DD",
) -> Optional[go.Figure]:

    df = df.copy()
    bargap, bargroupgap = _resolve_weekly_bar_spacing(x_col, x_titre, bargap, bargroupgap)

    if not verifier_presence_colonnes(df, [x_col, facette_col]):
        return None

    # Si y_col absent, on force un compteur
    if y_col not in df.columns:
        df["_tmp_count_"] = 1
        y_col = "_tmp_count_"
        y_titre = "Nombre d’occurrences"
        auto_aggregate = True

    # Filtrage facette
    if filtre_valeur is not None:
        df = df[df[facette_col] == filtre_valeur]
        facet_col = None
    else:
        facet_col = facette_col

    # ✅ PROTECTION: y_col est aussi une clé de groupby (ex: y_col == x_col)
    # Dans ce cas, pandas casse au reset_index -> on fait un count() au lieu d’un sum()
    group_cols = [facette_col, x_col]
    y_is_group_key = (y_col in group_cols)

    # Auto aggregate
    if auto_aggregate:
        if (not pd.api.types.is_numeric_dtype(df[y_col])) or y_is_group_key:
            # -> comptage d’occurrences
            df = df.groupby(group_cols, observed=True).size().reset_index(name="Nb_occurrences")
            y_col = "Nb_occurrences"
            y_titre = "Nombre de cas"
        else:
            # -> somme sur la variable numérique
            df = df.groupby(group_cols, observed=True)[y_col].sum().reset_index()

    if df.empty:
        logger.info("[INFO] Aucune donnée à afficher.")
        return None

    # Ordonner les facettes
    categories = sorted(df[facette_col].dropna().unique())
    df[facette_col] = pd.Categorical(df[facette_col], categories=categories, ordered=True)

    # Couleurs
    if isinstance(couleurs_personnalisees, dict):
        color_map = couleurs_personnalisees
        color_col = facette_col
    elif isinstance(couleurs_personnalisees, str):
        df["Couleur_unique"] = "Unique"
        color_col = "Couleur_unique"
        color_map = {"Unique": couleurs_personnalisees}
    else:
        color_col = facette_col if facet_col is not None else None
        color_map = None

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        facet_col=facet_col,
        facet_col_wrap=4 if facet_col is not None else None,
        color_discrete_map=color_map,
        labels={x_col: "", y_col: "", facette_col: facette_col},
        title=titre,
        height=taille_fig[1],
        width=taille_fig[0],
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        bargap=bargap,
        bargroupgap=bargroupgap,
        xaxis_tickangle=rotation,
        title_x=0.5,
        margin=dict(t=80, b=80, l=80),
    )

    if pas_x is not None:
        fig.update_xaxes(tickmode="linear", dtick=pas_x)

    if facet_col is not None:
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="")

    fig.add_annotation(
        x=0.5, y=-0.12, xref="paper", yref="paper",
        showarrow=False, text=x_titre,
        font=dict(size=14), xanchor="center", yanchor="top",
    )
    fig.add_annotation(
        x=-0.07, y=0.5, xref="paper", yref="paper",
        showarrow=False, text=y_titre,
        font=dict(size=14), textangle=-90,
        xanchor="center", yanchor="middle",
    )

    if annot:
        fig.update_traces(texttemplate="%{y}", textposition="outside", cliponaxis=False)

    if encadrer_facettes:
        for axis in fig.layout:
            if isinstance(fig.layout[axis], go.layout.XAxis) and "domain" in fig.layout[axis]:
                yaxis_name = axis.replace("xaxis", "yaxis")
                if yaxis_name in fig.layout and "domain" in fig.layout[yaxis_name]:
                    x0, x1 = fig.layout[axis].domain
                    y0, y1 = fig.layout[yaxis_name].domain
                    fig.add_shape(
                        type="rect",
                        x0=x0, x1=x1, y0=y0, y1=y1,
                        xref="paper", yref="paper",
                        line=dict(color=couleur_contour_facette, width=1),
                        fillcolor="rgba(0,0,0,0)",
                    )

    return fig if return_fig else fig

def plot_courbe_plotly(
    df: pd.DataFrame,
    colonne: str,
    titre: Optional[str] = None,
    annot: bool = False,
    rotation: int = 0,
    marker_size: int = 8,
    pas_x: Optional[int] = None,
    taille_fig: Tuple[int, int] = (1500, 500),
) -> Optional[go.Figure]:

    if not verifier_presence_colonnes(df, colonne):
        return None

    cas = df.groupby(colonne, observed=True).size().reset_index(name="Nombre de cas")
    categories_x = sorted(cas[colonne].dropna().unique(), key=extraire_numero)
    cas[colonne] = pd.Categorical(cas[colonne], categories=categories_x, ordered=True)

    fig = px.line(
        cas,
        x=colonne,
        y="Nombre de cas",
        title=titre or f"Courbe par '{colonne}'",
        markers=True,
        labels={colonne: colonne, "Nombre de cas": "Nombre de cas"},
    )

    if annot:
        fig.add_trace(go.Scatter(
            x=cas[colonne],
            y=cas["Nombre de cas"],
            mode="text",
            text=cas["Nombre de cas"],
            textposition="top center",
            showlegend=False,
        ))

    if rotation != 0:
        fig.update_layout(xaxis_tickangle=-rotation)

    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(width=taille_fig[0], height=taille_fig[1], template="plotly_white")

    if pas_x is not None:
        try:
            tickvals = [categories_x[i] for i in range(0, len(categories_x), pas_x)]
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=tickvals)
        except Exception:
            pass

    return fig

def plot_courbe_par_categories_plotly(
    df: pd.DataFrame,
    colonne_x: str,
    colonne_y: str,
    titre: Optional[str] = None,
    rotation: int = 45,
    annot: bool = False,
    pas_x: Optional[int] = None,
    taille_fig: Tuple[int, int] = (700, 500),
) -> Optional[go.Figure]:

    if not verifier_presence_colonnes(df, [colonne_x, colonne_y]):
        logger.info("Colonnes manquantes")
        return None

    cas = df.groupby([colonne_x, colonne_y], observed=True).size().reset_index(name="Nombre de cas")
    if cas.empty:
        logger.info("[INFO] Aucun point à afficher.")
        return None

    ordre_x = sorted(cas[colonne_x].unique(), key=extraire_numero)
    cas[colonne_x] = pd.Categorical(cas[colonne_x], categories=ordre_x, ordered=True)

    fig_args = {
        "data_frame": cas,
        "x": colonne_x,
        "y": "Nombre de cas",
        "color": colonne_y,
        "markers": True,
        "title": titre or f"Courbe de 'Nombre de cas' par '{colonne_x}' et '{colonne_y}'",
        "labels": {colonne_x: colonne_x, "Nombre de cas": "Nombre de cas", colonne_y: colonne_y},
        "category_orders": {colonne_x: ordre_x},
        "color_discrete_sequence": px.colors.qualitative.Set1,
    }
    if annot:
        fig_args["text"] = "Nombre de cas"

    fig = px.line(**fig_args)

    fig.update_layout(
        xaxis_tickangle=-rotation,
        template="plotly_white",
        xaxis_title=colonne_x,
        yaxis_title="Nombre de cas",
        width=taille_fig[0],
        height=taille_fig[1],
    )

    if annot:
        fig.update_traces(textposition="top center")

    if pas_x is not None:
        try:
            tickvals = [ordre_x[i] for i in range(0, len(ordre_x), pas_x)]
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=tickvals)
        except Exception as e:
            logger.info(f"Erreur ticks personnalisés : {e}")

    return fig

def plot_camembert_interactif(
    df: pd.DataFrame,
    colonne: Union[str, List[str]],
    titre: Optional[str] = None,
    seuil_min: int = 0,
    afficher_legende: bool = True,
    annot: bool = True,
    taille_fig: Tuple[int, int] = (700, 500),
    palette_couleurs: Optional[Union[List[str], Dict[str, str]]] = None,
) -> Optional[go.Figure]:

    if isinstance(colonne, list):
        for col in colonne:
            if col not in df.columns:
                logger.error(f"[ERREUR] Colonne '{col}' absente du DataFrame")
                return None
        serie_travail = df[colonne].fillna(MISSING_LABEL).astype(str).apply(" - ".join, axis=1)
        nom_analyse = " - ".join(colonne)
    else:
        if colonne not in df.columns:
            logger.error(f"[ERREUR] Colonne '{colonne}' absente du DataFrame")
            return None
        serie_travail = df[colonne].fillna(MISSING_LABEL)
        nom_analyse = colonne

    counts = serie_travail.value_counts(dropna=False)
    counts = counts[counts >= seuil_min]
    if counts.empty:
        logger.info("[INFO] Aucune catégorie ne correspond au seuil minimal.")
        return None

    labels = counts.index.tolist()
    valeurs = counts.values.tolist()

    couleurs_pie = None
    if isinstance(palette_couleurs, dict):
        couleurs_pie = [palette_couleurs.get(lbl, COLOR_INCONNU) for lbl in labels]
    elif palette_couleurs:
        couleurs_pie = palette_couleurs
    elif (isinstance(colonne, str) and colonne == COL_SEX) or nom_analyse.lower() == str(COL_SEX).lower():
        couleurs_pie = [SEX_COLOR_MAP.get(str(lbl), COLOR_INCONNU) for lbl in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=valeurs,
        hole=0.4,
        textinfo="label+percent+value" if annot else "label",
        hoverinfo="label+value+percent",
        marker=dict(
            line=dict(color="#FFFFFF", width=2),
            colors=couleurs_pie,
        ),
    )])

    fig.update_layout(
        title=titre or f"Répartition par {nom_analyse}",
        legend=dict(
            orientation="v",
            y=0.5,
            yanchor="middle",
            x=1.05,
            xanchor="left",
        ) if afficher_legende else dict(visible=False),
        width=taille_fig[0],
        height=taille_fig[1],
        margin=dict(l=20, r=150 if afficher_legende else 20, t=60, b=20),
        template="plotly_white",
    )
    return fig

def extraire_ordre_tranche(tranche: str) -> float:
    tranche = str(tranche).lower().strip()

    if "mois" in tranche:
        facteur = 1
    elif "semaine" in tranche:
        facteur = 1 / 4
    else:
        facteur = 12

    if tranche.startswith("<"):
        match = re.search(r"(\d+)", tranche)
        if match:
            return int(match.group(1)) * facteur - 0.5
        return 0

    if tranche.startswith(">"):
        match = re.search(r"(\d+)", tranche)
        if match:
            return int(match.group(1)) * facteur + 1000
        return 9999

    match = re.match(r"(\d+)[^\d]+(\d+)", tranche)
    if match:
        debut = int(match.group(1))
        return debut * facteur

    match = re.search(r"(\d+)", tranche)
    if match:
        return int(match.group(1)) * facteur

    return 9999

def plot_pyramide_symetrique(
    df: pd.DataFrame,
    col_categorie: str,
    col_groupe: str,
    valeurs_neg: Optional[List[str]] = None,
    titre: Optional[str] = "Pyramide Symétrique",
    seuil_min: int = 0,
    afficher_signe_negatif: bool = True,
    afficher_signe_negatif_dans_label: bool = True,
    croissant: bool = True,
    hauteur: int = 430,
) -> Optional[go.Figure]:

    if not verifier_presence_colonnes(df, [col_categorie, col_groupe]):
        return None

    counts = df.groupby([col_categorie, col_groupe], observed=True).size().reset_index(name="Nombre de cas")
    counts = counts[counts["Nombre de cas"] >= seuil_min]
    if counts.empty:
        logger.info("[INFO] Aucun groupe ne correspond au seuil minimal.")
        return None

    if not afficher_signe_negatif_dans_label:
        counts["label_text"] = counts["Nombre de cas"].abs().astype(str)
    else:
        counts["label_text"] = counts["Nombre de cas"].astype(str)

    try:
        ordre_categories = sorted(
            counts[col_categorie].unique(),
            key=extraire_ordre_tranche,
            reverse=not croissant,
        )
    except Exception as e:
        logger.warning(f"[WARN] Échec du tri logique: {e}")
        ordre_categories = sorted(counts[col_categorie].unique(), reverse=not croissant)

    counts[col_categorie] = pd.Categorical(counts[col_categorie], categories=ordre_categories, ordered=True)

    sexe_display_map = {
        "feminin": "Feminin",
        "féminin": "Feminin",
        "f": "Feminin",
        "female": "Feminin",
        "femme": "Feminin",
        "masculin": "Masculin",
        "m": "Masculin",
        "male": "Masculin",
        "homme": "Masculin",
    }
    counts["_groupe_display"] = counts[col_groupe].apply(
        lambda value: sexe_display_map.get(str(value).strip().lower(), str(value).strip())
    )

    categorie_label = "Tranche d'âge" if "tranche" in str(col_categorie).lower() else str(col_categorie)
    groupe_label = "Sexe" if str(col_groupe).strip().lower() in {"sexe", "sex"} else str(col_groupe)
    color_map = {
        "Feminin": "#E70B0B",
        "Masculin": "#1a1e2b",
    }

    # Force explicit zero-filled traces per sex so bars stay perfectly aligned per age band.
    pivot = (
        counts.pivot_table(
            index=col_categorie,
            columns="_groupe_display",
            values="Nombre de cas",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reindex(ordre_categories)
        .fillna(0)
    )

    preferred_order = [label for label in ["Feminin", "Masculin"] if label in pivot.columns]
    remaining = [label for label in pivot.columns.tolist() if label not in preferred_order]
    trace_order = preferred_order + remaining

    fig = go.Figure()
    negative_groups = {sexe_display_map.get(str(v).strip().lower(), str(v).strip()) for v in (valeurs_neg or [])}
    for group_name in trace_order:
        values = pd.to_numeric(pivot[group_name], errors="coerce").fillna(0).abs()
        signed_values = -values if (afficher_signe_negatif and group_name in negative_groups) else values
        text_values = values.astype(int).astype(str) if not afficher_signe_negatif_dans_label else signed_values.astype(int).astype(str)
        fig.add_trace(
            go.Bar(
                y=ordre_categories,
                x=signed_values.tolist(),
                orientation="h",
                name=str(group_name),
                marker=dict(color=color_map.get(str(group_name), SEX_COLOR_MAP.get(str(group_name), None))),
                text=text_values.tolist(),
                texttemplate="%{text}",
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"{groupe_label}: {group_name}<br>{categorie_label}: %{{y}}<br>Nombre de cas: %{{customdata}}<extra></extra>",
                customdata=values.astype(int).tolist(),
            )
        )

    max_val = int(max(abs(v) for tr in fig.data for v in tr.x)) if fig.data else 0
    axis_max = max(1, int(np.ceil(max_val * 1.08)))
    fig.update_layout(
        barmode="relative",
        title_text=titre or "",
        xaxis=dict(
            tickvals=[-max_val, 0, max_val],
            ticktext=[str(max_val), "0", str(max_val)],
            automargin=True,
            zeroline=True,
            zerolinewidth=1.2,
            zerolinecolor="rgba(26,30,43,0.28)",
            range=[-axis_max, axis_max],
        ),
        bargap=0.1,
        template="plotly_white",
        yaxis=dict(categoryorder="array", categoryarray=ordre_categories, title=categorie_label),
        xaxis_title="Nombre de cas",
        height=int(hauteur),
        margin=dict(t=52, b=44, l=72, r=56),
        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )
    fig.update_yaxes(automargin=True)
    return fig

def graphique_pyramide_age(
    df: pd.DataFrame,
    col_tranche: str = "Tranche_age",
    col_sexe: str = "Sexe",
    col_valeur: str = "Nombre",
    valeurs_neg: Optional[List[str]] = None,
    titre: Optional[str] = "Pyramide des âges",
    seuil_min: int = 0,
    afficher_signe_negatif: bool = True,
    afficher_signe_negatif_dans_label: bool = True,
    croissant: bool = True,
    couleurs_personnalisees: Optional[Dict[str, str]] = None,
    annot: bool = False,
    facette_col: Optional[str] = None,
    taille_fig: Tuple[int, int] = (1200, 700),
    return_fig: bool = False,
    couleur_contour_facette: str = "#777772",
) -> Optional[go.Figure]:

    df = df.copy()

    for c in [col_tranche, col_sexe, col_valeur]:
        if c not in df.columns:
            logger.error(f"[ERROR] Colonne '{c}' absente dans le DataFrame")
            return None

    if facette_col is not None and facette_col not in df.columns:
        logger.error(f"[ERROR] Colonne de facettage '{facette_col}' absente dans le DataFrame")
        return None

    df = df.dropna(subset=[col_tranche, col_sexe])
    if facette_col:
        df = df.dropna(subset=[facette_col])

    group_cols = [col_tranche, col_sexe]
    if facette_col is not None:
        group_cols.append(facette_col)

    if pd.api.types.is_numeric_dtype(df[col_valeur]):
        agg_df = df.groupby(group_cols, observed=True)[col_valeur].sum().reset_index()
    else:
        agg_df = df.groupby(group_cols, observed=True).size().reset_index(name=col_valeur)

    agg_df = agg_df[agg_df[col_valeur] >= seuil_min]
    if agg_df.empty:
        logger.info("[INFO] Aucune donnée après filtrage avec seuil_min")
        return None

    if valeurs_neg is not None and afficher_signe_negatif:
        valeurs_neg_set = {v.lower() for v in valeurs_neg}
        agg_df[col_valeur] = agg_df.apply(
            lambda row: -row[col_valeur] if str(row[col_sexe]).lower() in valeurs_neg_set else row[col_valeur],
            axis=1,
        )

    if afficher_signe_negatif_dans_label:
        agg_df["label_text"] = agg_df[col_valeur].astype(str)
    else:
        agg_df["label_text"] = agg_df[col_valeur].abs().astype(str)

    try:
        categories = sorted(agg_df[col_tranche].unique(), key=extraire_ordre_tranche, reverse=not croissant)
    except Exception:
        categories = sorted(agg_df[col_tranche].unique(), reverse=not croissant)
    agg_df[col_tranche] = pd.Categorical(agg_df[col_tranche], categories=categories, ordered=True)

    if couleurs_personnalisees is None:
        couleurs_personnalisees = {"Masculin": "#1a1e2b", "Feminin": "#E70B0B"}
    for cat in agg_df[col_sexe].unique():
        if cat not in couleurs_personnalisees:
            couleurs_personnalisees[cat] = None

    # --- Construction de la pyramide ---
    # Objectif: 2 barres (homme/femme) EXACTEMENT sur la même ligne de tranche d'âge
    # => on évite le "grouped bar" et on force une superposition relative autour de 0.
    if facette_col is None:
        # Séparer les groupes (ex: Masculin/Feminin) pour avoir 2 traces alignées sur le même y
        sexes = list(agg_df[col_sexe].dropna().unique())

        fig = go.Figure()
        for sx in sexes:
            d = agg_df[agg_df[col_sexe] == sx].copy()
            fig.add_trace(go.Bar(
                y=d[col_tranche],
                x=d[col_valeur],
                orientation="h",
                name=str(sx),
                marker=dict(color=couleurs_personnalisees.get(str(sx))),
                text=d["label_text"] if annot else None,
                textposition="outside" if annot else "none",
                cliponaxis=False,
            ))

        # IMPORTANT: "relative" = même ligne (y) + valeurs négatives à gauche, positives à droite
        fig.update_layout(
            barmode="relative",
            title=titre,
            width=taille_fig[0],
            height=taille_fig[1],
            template="plotly_white",
            bargap=0.1,
            bargroupgap=0,
            title_x=0.5,
            margin=dict(t=80, b=80, l=80, r=80),
            yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=categories),
            xaxis_title="Nombre",
            yaxis_title="Tranche d'âge",
        )
    else:
        # Avec facettes: on garde px.bar, mais on force l'alignement par tranche
        fig = px.bar(
            agg_df,
            y=col_tranche,
            x=col_valeur,
            color=col_sexe,
            orientation="h",
            text="label_text" if annot else None,
            color_discrete_map=couleurs_personnalisees,
            facet_col=facette_col,
            facet_col_wrap=4 if facette_col else None,
            title=titre,
            labels={col_valeur: "Nombre", col_tranche: "Tranche d'âge", col_sexe: "Sexe"},
            category_orders={col_tranche: agg_df[col_tranche].cat.categories.tolist()},
            width=taille_fig[0],
            height=taille_fig[1],
        )
        fig.update_layout(barmode="relative")


    max_val = max(abs(agg_df[col_valeur])) if not agg_df.empty else 0
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            tickvals=[-max_val, 0, max_val],
            ticktext=[str(int(max_val)), "0", str(int(max_val))],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="LightGrey",
        ),
        yaxis=dict(autorange="reversed"),
        bargap=0.1,
        bargroupgap=0,
        title_x=0.5,
        margin=dict(t=80, b=80, l=80, r=80),
    )

    if annot:
        fig.update_traces(textposition="outside", cliponaxis=False)

    if facette_col:
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        # Encadrer uniquement les vraies facettes. Sur une figure simple,
        # certains axes peuvent avoir domain=None, ce qui provoque une erreur.
        for axis_name in fig.layout:
            axis_obj = fig.layout[axis_name]
            if not isinstance(axis_obj, go.layout.XAxis):
                continue

            x_domain = getattr(axis_obj, "domain", None)
            if x_domain is None or len(x_domain) != 2:
                continue

            yaxis_name = axis_name.replace("xaxis", "yaxis")
            if yaxis_name not in fig.layout:
                continue

            yaxis_obj = fig.layout[yaxis_name]
            y_domain = getattr(yaxis_obj, "domain", None)
            if y_domain is None or len(y_domain) != 2:
                continue

            x0, x1 = x_domain
            y0, y1 = y_domain
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1, y0=y0, y1=y1,
                xref="paper", yref="paper",
                line=dict(color=couleur_contour_facette, width=1),
                fillcolor="rgba(0,0,0,0)",
            )

    return fig if return_fig else fig

def plot_boxplot_delais_plotly(
    df: pd.DataFrame,
    colonnes_delais: List[str],
    col_groupe: Optional[str] = None,
    titre: Optional[str] = None,
    taille_fig: Tuple[int, int] = (1000, 600),
    rotation: int = 45,
) -> Optional[go.Figure]:

    colonnes_manquantes = [c for c in colonnes_delais if c not in df.columns]
    if colonnes_manquantes:
        logger.error(f"[ERREUR] Colonnes délais manquantes : {colonnes_manquantes}")
        return None

    id_vars = [col_groupe] if col_groupe and col_groupe in df.columns else []
    df_long = df[id_vars + colonnes_delais].melt(
        id_vars=id_vars,
        value_vars=colonnes_delais,
        var_name="Delai",
        value_name="Jours",
    ).dropna(subset=["Jours"])

    if df_long.empty:
        logger.info("[INFO] Aucun délai non manquant à afficher.")
        return None

    if col_groupe and col_groupe in df.columns:
        fig = px.box(
            df_long, x=col_groupe, y="Jours", color="Delai",
            points="outliers",
            title=titre or "Distribution des délais observés (en jours) par groupe",
        )
    else:
        fig = px.box(
            df_long, x="Delai", y="Jours",
            points="outliers",
            title=titre or "Distribution des délais observés (en jours)",
        )

    fig.update_layout(
        template="plotly_white",
        width=taille_fig[0],
        height=taille_fig[1],
        xaxis_tickangle=rotation,
        yaxis_title="Délai (jours)",
    )
    return fig

def plot_barres_pct_sous_seuil(
    df_resume_groupe: pd.DataFrame,
    col_groupe: str = "Province_notification",
    col_n: str = "n",
    col_sous_seuil: str = "sous_seuil",
    col_pct: str = "pct_sous_seuil_%",
    titre: Optional[str] = None,
    seuil: int = 2,
    taille_fig: Tuple[int, int] = (1200, 600),
    rotation: int = 45,
    annot: bool = True,
    tri_desc: bool = True,
) -> Optional[go.Figure]:

    colonnes_requises = [col_groupe, col_n, col_sous_seuil, col_pct]
    manquantes = [c for c in colonnes_requises if c not in df_resume_groupe.columns]
    if manquantes:
        logger.error(f"[ERREUR] Colonnes manquantes dans df_resume_groupe : {manquantes}")
        return None

    df_plot = df_resume_groupe.copy()
    df_plot[col_pct] = pd.to_numeric(df_plot[col_pct], errors="coerce").fillna(0)

    if tri_desc:
        df_plot = df_plot.sort_values(col_pct, ascending=False)

    fig = px.bar(
        df_plot,
        x=col_groupe,
        y=col_pct,
        text=col_pct if annot else None,
        title=titre or f"% de cas avec délai ≤ {seuil} jours par {col_groupe}",
        labels={col_groupe: col_groupe, col_pct: f"% sous {seuil} jours"},
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside" if annot else "none",
        cliponaxis=False,
    )

    y_max = df_plot[col_pct].max()
    if pd.isna(y_max):
        y_max = 5
    y_max = min(105, max(5, y_max + 5))

    fig.update_layout(
        template="plotly_white",
        width=taille_fig[0],
        height=taille_fig[1],
        xaxis_tickangle=rotation,
        yaxis_title=f"% sous {seuil} jours",
        yaxis=dict(range=[0, y_max]),
    )
    return fig

def plot_evolution_multi_auto(
    df: pd.DataFrame,
    col_x: str = "Semaine_epid",
    courbe_col: List[str] = [],
    valeurs_courbe_col: Optional[Dict[str, Union[str, bool, int]]] = None,
    titre: Optional[str] = None,
    taille_fig: Tuple[int, int] = (1000, 600),
    couleurs: Optional[Dict[str, str]] = None,
    annot_x: bool = False,
    annot_y: bool = False,
    rotation: int = 0,
    marker_size: int = 8,
    pas_x: Optional[int] = None,
    afficher_legende: bool = True,
    seuil_min: int = 0,
    bargap: float = 0.2,
    bargroupgap: float = 0.1,
) -> Optional[go.Figure]:

    valeurs_courbe_col = valeurs_courbe_col or {}
    bargap, bargroupgap = _resolve_weekly_bar_spacing(col_x, col_x, bargap, bargroupgap)

    colonnes_absentes = [col for col in [col_x] + courbe_col if col not in df.columns]
    if colonnes_absentes:
        logger.error(f"[ERREUR] Colonnes absentes du DataFrame : {colonnes_absentes}")
        return None

    couleurs = couleurs or {"cas": "rgba(0, 123, 255, 0.6)"}
    for col in courbe_col:
        if col not in couleurs:
            couleurs[col] = None

    df_clean = df[[col_x] + courbe_col].copy().dropna(subset=[col_x])

    cas_par_x = df_clean[col_x].value_counts().sort_index()
    cas_par_x = cas_par_x[cas_par_x >= seuil_min]
    if cas_par_x.empty:
        logger.info("[INFO] Aucun groupe ne dépasse le seuil minimal.")
        return None

    cas_par_x = cas_par_x.sort_index()

    data_courbes = pd.DataFrame(index=cas_par_x.index)

    for col in courbe_col:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            tmp = df_clean.groupby(col_x, observed=True)[col].sum()
            tmp = tmp.reindex(cas_par_x.index, fill_value=0)
            data_courbes[col] = tmp
        else:
            val_pos = valeurs_courbe_col.get(col)
            if val_pos is None:
                tmp = df_clean.groupby(col_x, observed=True)[col].apply(lambda x: x.notna().sum())
                tmp = tmp.reindex(cas_par_x.index, fill_value=0)
                data_courbes[col] = tmp
            else:
                tmp = df_clean[df_clean[col] == val_pos].groupby(col_x, observed=True)[col].count()
                tmp = tmp.reindex(cas_par_x.index, fill_value=0)
                data_courbes[col] = tmp

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=cas_par_x.index,
        y=cas_par_x.values,
        name="Cas",
        marker_color=couleurs.get("cas"),
        yaxis="y1",
        text=cas_par_x.values if annot_x else None,
        textposition="auto" if annot_x else None,
    ))

    for col in courbe_col:
        fig.add_trace(go.Scatter(
            x=data_courbes.index,
            y=data_courbes[col],
            name=col,
            mode="lines+markers+text" if annot_y else "lines+markers",
            marker=dict(size=marker_size, color=couleurs.get(col)),
            yaxis="y2",
            text=[f"{v}" for v in data_courbes[col]] if annot_y else None,
            textposition="top center" if annot_y else None,
        ))

    fig.update_layout(
        title=titre or f"Évolution par '{col_x}'",
        xaxis=dict(
            title=col_x,
            tickangle=rotation,
            tickmode="linear",
            dtick=pas_x if pas_x else None,
            showgrid=True,
            gridcolor="LightGray",
            gridwidth=1,
        ),
        yaxis=dict(
            title="Nombre de cas",
            showgrid=True,
            gridcolor="LightGray",
            gridwidth=1,
        ),
        yaxis2=dict(
            title="Valeurs des courbes",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(
            x=1.02, y=1, xanchor="left", yanchor="top",
            traceorder="normal",
            font=dict(size=12),
            borderwidth=1,
        ) if afficher_legende else dict(visible=False),
        barmode="group",
        bargap=bargap,
        bargroupgap=bargroupgap,
        width=taille_fig[0],
        height=taille_fig[1],
        margin=dict(l=60, r=100 if afficher_legende else 20, t=60, b=60),
        template="plotly_white",
    )

    return fig


def build_weekly_cases_deaths_combo(
    weekly_df: pd.DataFrame,
    x_col: str,
    cases_col: str = "Cas",
    deaths_col: str = "Deces",
    titre: str = "Evolution hebdomadaire des cas et deces",
    x_titre: str = "Semaine épidémiologique",
    y_titre_cas: str = "Nombre de cas",
    y_titre_deces: str = "Nombre de deces",
    rotation: int = 45,
    annot_bars: bool = False,
    annot_line: bool = False,
    pas_x: Optional[int] = None,
    taille_fig: Tuple[int, int] = (1400, 550),
) -> Optional[go.Figure]:
    """Graphique combiné : Cas en histogramme et Décès en courbe."""
    if weekly_df is None or weekly_df.empty:
        return None
    weekly_df = _normalize_metric_alias_columns(weekly_df)

    required_cols = [x_col, cases_col, deaths_col]
    if any(col not in weekly_df.columns for col in required_cols):
        return None

    data = weekly_df[required_cols].copy().dropna(subset=[x_col])
    if data.empty:
        return None

    data[cases_col] = pd.to_numeric(data[cases_col], errors="coerce").fillna(0)
    data[deaths_col] = pd.to_numeric(data[deaths_col], errors="coerce").fillna(0)
    data[x_col] = data[x_col].astype(str)
    bargap, bargroupgap = _resolve_weekly_bar_spacing(x_col, x_titre, 0.04, 0.02)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=data[x_col],
            y=data[cases_col],
            name="Cas",
            marker_color=COLOR_CASES,
            yaxis="y1",
            text=data[cases_col] if annot_bars else None,
            textposition="outside" if annot_bars else None,
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data[x_col],
            y=data[deaths_col],
            name="Décès",
            mode="lines+markers+text" if annot_line else "lines+markers",
            line=dict(color=COLOR_DEATHS, width=3),
            marker=dict(color=COLOR_DEATHS, size=8),
            yaxis="y2",
            text=data[deaths_col] if annot_line else None,
            textposition="top center" if annot_line else None,
        )
    )
    fig.update_layout(
        title=titre,
        template="plotly_white",
        width=taille_fig[0],
        height=taille_fig[1],
        bargap=bargap,
        bargroupgap=bargroupgap,
        xaxis=dict(title=x_titre, tickangle=-rotation),
        yaxis=dict(title=y_titre_cas, rangemode="tozero"),
        yaxis2=dict(title=y_titre_deces, overlaying="y", side="right", rangemode="tozero"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(t=80, b=60, l=70, r=70),
    )
    if pas_x is not None:
        try:
            tickvals = data[x_col].iloc[:: max(int(pas_x), 1)]
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=tickvals)
        except Exception:
            pass
    return fig


def build_weekly_multiline_by_group(
    df: pd.DataFrame,
    week_col: str,
    group_col: str,
    selected_groups: Optional[List[str]] = None,
    titre: str = "Courbe épidémiologique par province",
    x_titre: str = "Semaine épidémiologique",
    y_titre: str = "Nombre de cas",
    rotation: int = 45,
    pas_x: Optional[int] = None,
    annot: bool = False,
    taille_fig: Tuple[int, int] = (1500, 650),
) -> Optional[go.Figure]:
    """Construit une courbe multi-séries par groupe avec légende interactive Plotly."""
    if not verifier_presence_colonnes(df, [week_col, group_col]):
        return None

    tmp = df[[week_col, group_col]].copy().dropna(subset=[week_col, group_col])
    if tmp.empty:
        return None

    tmp[group_col] = tmp[group_col].astype(str).str.strip()
    tmp = tmp[tmp[group_col] != ""]
    if tmp.empty:
        return None

    weekly = (
        tmp.groupby([week_col, group_col], observed=True)
        .size()
        .reset_index(name="Cas")
    )
    if weekly.empty:
        return None

    if selected_groups:
        selected_set = {str(x) for x in selected_groups}
        weekly = weekly[weekly[group_col].astype(str).isin(selected_set)]
        if weekly.empty:
            return None
        group_order = [str(x) for x in selected_groups if str(x) in weekly[group_col].astype(str).unique().tolist()]
    else:
        group_order = (
            weekly.groupby(group_col, observed=True)["Cas"]
            .sum()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )

    categories_x = sorted(weekly[week_col].dropna().unique(), key=extraire_numero)
    weekly[week_col] = pd.Categorical(weekly[week_col], categories=categories_x, ordered=True)
    weekly[group_col] = pd.Categorical(weekly[group_col].astype(str), categories=group_order, ordered=True)
    weekly = weekly.sort_values([group_col, week_col])

    fig = px.line(
        weekly,
        x=week_col,
        y="Cas",
        color=group_col,
        markers=True,
        labels={week_col: x_titre, "Cas": y_titre, group_col: group_col},
        title=titre,
        height=taille_fig[1],
        width=taille_fig[0],
    )
    if annot:
        fig.update_traces(
            mode="lines+markers+text",
            text=weekly["Cas"],
            textposition="top center",
            line=dict(width=2),
            marker=dict(size=7),
        )
    else:
        fig.update_traces(mode="lines+markers", line=dict(width=2), marker=dict(size=7))
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(title=x_titre, tickangle=-rotation),
        yaxis=dict(title=y_titre, rangemode="tozero"),
        legend=dict(
            title=group_col,
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(t=80, b=60, l=70, r=180),
    )
    if pas_x is not None:
        try:
            tickvals = [categories_x[i] for i in range(0, len(categories_x), max(int(pas_x), 1))]
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=tickvals)
        except Exception:
            pass
    return fig


def build_weekly_cases_cfr_combo(
    df: pd.DataFrame,
    week_col: str,
    death_col: str = "is_death",
    titre: str = "Cas et létalité par semaine épidémiologique",
    rotation: int = 45,
    annot_bars: bool = False,
    annot_line: bool = False,
    pas_x: Optional[int] = None,
    taille_fig: Tuple[int, int] = (1500, 600),
) -> Optional[go.Figure]:
    """Graphique combiné: barres = cas hebdomadaires, courbe = létalité (%)."""
    if not verifier_presence_colonnes(df, week_col):
        return None
    bargap, bargroupgap = _resolve_weekly_bar_spacing(week_col, "Semaine épidémiologique", 0.15, 0.1)

    tmp = df.copy()
    tmp = tmp[tmp[week_col].notna()].copy()
    if tmp.empty:
        return None

    if death_col in tmp.columns:
        death_vals = pd.to_numeric(tmp[death_col], errors="coerce").fillna(0)
    elif COL_ISSUE in tmp.columns:
        death_vals = tmp[COL_ISSUE].astype("string").str.lower().isin(["decede", "décédé", "decede(e)", "décès", "deces"]).astype(int)
    else:
        death_vals = pd.Series(0, index=tmp.index, dtype="int64")

    tmp["_death_tmp_"] = death_vals

    weekly = (
        tmp.groupby(week_col, observed=True, as_index=False)
           .agg(Cas=(week_col, "count"), Deces=("_death_tmp_", "sum"))
    )
    if weekly.empty:
        return None

    weekly["Létalité (%)"] = np.where(weekly["Cas"] > 0, (weekly["Deces"] / weekly["Cas"]) * 100.0, np.nan)
    categories_x = sorted(weekly[week_col].dropna().unique(), key=extraire_numero)
    weekly[week_col] = pd.Categorical(weekly[week_col], categories=categories_x, ordered=True)
    weekly = weekly.sort_values(week_col)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weekly[week_col],
        y=weekly["Cas"],
        name="Cas",
        marker_color=COLOR_CASES,
        yaxis="y1",
        text=weekly["Cas"] if annot_bars else None,
        textposition="outside" if annot_bars else None,
        cliponaxis=False,
    ))
    fig.add_trace(go.Scatter(
        x=weekly[week_col],
        y=weekly["Létalité (%)"],
        name="Létalité (%)",
        mode="lines+markers+text" if annot_line else "lines+markers",
        line=dict(color=COLOR_CFR),
        marker=dict(color=COLOR_CFR),
        yaxis="y2",
        text=weekly["Létalité (%)"].round(2).astype(str) + "%" if annot_line else None,
        textposition="top center" if annot_line else None,
    ))

    fig.update_layout(
        title=titre,
        template="plotly_white",
        width=taille_fig[0],
        height=taille_fig[1],
        bargap=bargap,
        bargroupgap=bargroupgap,
        xaxis=dict(title="Semaine épidémiologique", tickangle=-rotation),
        yaxis=dict(title="Nombre de cas", rangemode="tozero"),
        yaxis2=dict(title="Létalité (%)", overlaying="y", side="right", rangemode="tozero"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(t=80, b=60, l=70, r=70),
    )

    if pas_x is not None:
        try:
            tickvals = [categories_x[i] for i in range(0, len(categories_x), pas_x)]
            fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=tickvals)
        except Exception:
            pass

    return fig


# ✅ Comme les fonctions sont dans CE fichier, on force:
HAS_CUSTOM_VIZ = True


# =========================================================
# APP STREAMLIT (TON SCRIPT, TABS + PARAMÈTRES CONSERVÉS)
# =========================================================

# =========================
# CONFIG
# =========================
