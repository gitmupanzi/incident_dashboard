# =========================
# Incident RDC – Dashboard (Streamlit UI) + VISUALISATIONS CUSTOM + CARTES FIX (Plotly + fuzzy join)
# =========================
# ✅ Version AUTONOME (1 seul fichier)
# - Plus d'import depuis dataminsante.visualisation.visualisation_graphique
# - Les fonctions "custom" sont intégrées directement dans ce script
# - ✅ Tabs + paramètres + logique UI conservés exactement comme ta version
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
from typing import Optional, Union, List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype
from datetime import date


# =========================================================
# UTILITAIRES GÉNÉRAUX (Harmonisation & Robustesse)
# - Centralise les fonctions réutilisées dans les onglets
# - Évite les divergences entre tabs
# =========================================================

from typing import Iterable, List, Tuple, Any, Optional
import io
from functools import lru_cache
import tempfile



# =========================================================
# ✅ BLOC CARTE (INTÉGRÉ) – AUTONOME
# =========================================================
def carte_statique_matplotlib(
    gdf,
    colonne_valeurs: str,
    titre: str,
    annoter: bool = True,
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
    def _ajouter_barre_echelle(ax, longueur_km=50, loc=(0.10, 0.06), largeur_ligne=0.8, taille_police=7):
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

    # ---- reprojection (pour échelle en mètres) ----
    try:
        if gdf.crs is None or (hasattr(gdf.crs, "to_epsg") and gdf.crs.to_epsg() != 3857):
            gdf = gdf.to_crs(epsg=3857)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=figsize)

    # ---- plot ----
    geom_types = gdf.geometry[~gdf.geometry.is_empty].geom_type.unique()
    if set(geom_types) == {"Point"}:
        gdf.plot(
            column=colonne_valeurs,
            cmap=cmap,
            ax=ax,
            legend=True,
            markersize=40,
            edgecolor="k",
            linewidth=0.5,
        )
    else:
        gdf.plot(
            column=colonne_valeurs,
            cmap=cmap,
            ax=ax,
            legend=True,
            edgecolor="0.75",
            linewidth=0.8,
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
        for _, row in gdf.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                continue

            val = row[colonne_valeurs]
            if pd.isna(val) or val <= seuil_affichage:
                continue

            if row.geometry.geom_type in ["Polygon", "MultiPolygon"]:
                c = row.geometry.centroid
                x, y = c.x, c.y
            else:
                x, y = row.geometry.x, row.geometry.y

            parts = []
            if nom_zone in gdf.columns:
                parts.append(str(row[nom_zone]))
            try:
                parts.append(fmt_valeurs.format(val))
            except Exception:
                parts.append(str(val))

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

    plt.tight_layout()

    # ---- colorbar compacte ----
    if len(fig.axes) > 1:
        cb_ax = fig.axes[-1]
        pos = cb_ax.get_position()
        cb_ax.set_position([
            pos.x0 + 0.01,
            pos.y0 + cb_shift_up,
            pos.width * cb_width,
            pos.height * cb_height,
        ])
        cb_ax.tick_params(labelsize=legend_taille_ticks)
        if legend_titre:
            cb_ax.set_title(legend_titre, fontsize=legend_taille_titre, pad=4)

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
    """Affichage Streamlit robuste: colonnes uniques + pleine largeur."""
    _df = df.copy()
    _df.columns = make_unique(flatten_columns(_df.columns))
    st.dataframe(_df, width='stretch', height=height)


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
        st.info("Aucune donnée pour afficher le tableau.")
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

# -------------------------
# LOGGER
# -------------------------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


# =========================================================
# ✅ BLOC VISUALISATIONS (INTÉGRÉ) – AUTONOME
# =========================================================

def verifier_presence_colonnes(
    df: pd.DataFrame,
    colonnes: Union[str, List[str], Tuple[str, ...]]
) -> bool:
    if isinstance(colonnes, tuple):
        colonnes = list(colonnes)
    elif isinstance(colonnes, str):
        colonnes = [colonnes]
    elif isinstance(colonnes, list):
        pass
    else:
        logger.error(f"[ERREUR] Format non supporté pour les colonnes : {type(colonnes)}")
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
    counts = df[colonne].fillna("Inconnu").value_counts(dropna=False)
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
    palette_couleurs: Optional[List[str]] = None,
) -> Optional[go.Figure]:

    if isinstance(colonne, list):
        for col in colonne:
            if col not in df.columns:
                logger.error(f"[ERREUR] Colonne '{col}' absente du DataFrame")
                return None
        serie_travail = df[colonne].fillna("Inconnu").astype(str).apply(" - ".join, axis=1)
        nom_analyse = " - ".join(colonne)
    else:
        if colonne not in df.columns:
            logger.error(f"[ERREUR] Colonne '{colonne}' absente du DataFrame")
            return None
        serie_travail = df[colonne].fillna("Inconnu")
        nom_analyse = colonne

    counts = serie_travail.value_counts(dropna=False)
    counts = counts[counts >= seuil_min]
    if counts.empty:
        logger.info("[INFO] Aucune catégorie ne correspond au seuil minimal.")
        return None

    labels = counts.index.tolist()
    valeurs = counts.values.tolist()

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=valeurs,
        hole=0.4,
        textinfo="percent+label" if annot else "label",
        hoverinfo="label+value+percent",
        marker=dict(
            line=dict(color="#FFFFFF", width=2),
            colors=palette_couleurs if palette_couleurs else None,
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
) -> Optional[go.Figure]:

    if not verifier_presence_colonnes(df, [col_categorie, col_groupe]):
        return None

    counts = df.groupby([col_categorie, col_groupe], observed=True).size().reset_index(name="Nombre de cas")
    counts = counts[counts["Nombre de cas"] >= seuil_min]
    if counts.empty:
        logger.info("[INFO] Aucun groupe ne correspond au seuil minimal.")
        return None

    if valeurs_neg is not None and afficher_signe_negatif:
        valeurs_neg_lower = {v.lower() for v in valeurs_neg}
        counts["Nombre de cas"] = counts.apply(
            lambda row: -row["Nombre de cas"]
            if str(row[col_groupe]).lower() in valeurs_neg_lower
            else row["Nombre de cas"],
            axis=1,
        )

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

    fig = px.bar(
        counts,
        y=col_categorie,
        x="Nombre de cas",
        color=col_groupe,
        orientation="h",
        title=titre,
        labels={col_categorie: col_categorie, "Nombre de cas": "Nombre de cas", col_groupe: col_groupe},
        text="label_text",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )

    fig.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)

    max_val = max(abs(counts["Nombre de cas"]))
    fig.update_layout(
        xaxis=dict(
            tickvals=[-max_val, 0, max_val],
            ticktext=[str(max_val), "0", str(max_val)],
        ),
        bargap=0.1,
        template="plotly_white",
        yaxis=dict(categoryorder="array", categoryarray=ordre_categories),
    )
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


    max_val =    max_val = max(abs(agg_df[col_valeur])) if not agg_df.empty else 0
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
            title=titre or "Distribution des délais (en jours) par groupe",
        )
    else:
        fig = px.box(
            df_long, x="Delai", y="Jours",
            points="outliers",
            title=titre or "Distribution des délais (en jours)",
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


# ✅ Comme les fonctions sont dans CE fichier, on force:
HAS_CUSTOM_VIZ = True


# =========================================================
# APP STREAMLIT (TON SCRIPT, TABS + PARAMÈTRES CONSERVÉS)
# =========================================================

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="LL RDC – Dashboard", layout="wide")

# --- Colonnes (tes noms exacts) ---
COL_PROV = "Province_notification"
COL_ZS   = "Zone_de_sante_notification"
COL_AS   = "Aire_de_sante_notification"

COL_YEAR = "Annee_epid"
COL_WNUM = "Num_semaine_epid"
COL_SEX  = "Sexe"
COL_AGE  = "Age"
COL_UNIT = "Unite_age"
COL_AGEG = "Tranche_age_en_ans"
COL_AGEG2 = "Tranche_age"

COL_PREL = "Prelevement"
COL_TDR  = "TDR_realise"
COL_TDRR = "TDR_Resultat"
COL_HOSP = "Hospitalisation"
COL_DEHY = "Degre_deshydratation"
COL_ISSUE= "Issue"
COL_CLASS= "Classification_finale"

DATE_ONSET = "Date_debut_maladie"
DATE_ADM   = "Date_admission_au_CT"
DATE_PREL  = "Date_prelevement"

# =========================
# MALADIES (CONFIG / SPECS)
# Objectif: rendre le dashboard "multi-line list" (Choléra, Rougeole, Mpox, Ebola, Intox, IDSR)
# - Chaque maladie peut avoir des noms de colonnes différents -> on renomme vers le schéma commun
# - Le reste du pipeline (standardize_ll_core -> standardize_df -> KPI/graphs) reste identique
# =========================

DISEASE_SPECS: Dict[str, Dict[str, Any]] = {
    "cholera": {
        "label": "Choléra",
        "default_sheet": "LL_Cholera",
        "rename_map": {
            # variations fréquentes
            "Nom_complet_cas_suspect": "Nom_complet",
        },
        # dates candidates (pour Date_debut_maladie si manquante)
        "onset_candidates": ["Date_debut_maladie"],
        "notif_candidates": ["Date_notification"],
        "adm_candidates": ["Date_admission_au_CT"],
        "prel_candidates": ["Date_prelevement"],
    },
    "rougeole": {
        "label": "Rougeole (line list)",
        "default_sheet": "LL_Rougeole",
        "rename_map": {},
        "onset_candidates": ["Date_debut_maladie", "Date_debut_symptomes"],
        "notif_candidates": ["Date_notification"],
        "adm_candidates": ["Date_admission_au_CT", "Date_admission"],
        "prel_candidates": ["Date_prelevement", "Date_prelevement_clean"],
    },
    "mpox": {
        "label": "Mpox (line list)",
        "default_sheet": "LL_Mpox",
        "rename_map": {
            # Identité
            "Nom_complet_cas_suspect": "Nom_complet",
            # Classification
            "Classification_finale_du_cas": "Classification_finale",
            # Prélevement / labo (harmonisation minimale)
            "Prelevement_realise_au_moment_de_investigation": "Prelevement",
            "Si_oui_date_de_prelevement": "Date_prelevement",
            "Quel_est_le_resultats": "Resultat_labo",
        },
        # Mpox: onset "unique" rarement présent; on choisit une meilleure approximation si dispo
        "onset_candidates": [
            "Si_oui_des_eruption_cutanee_quelle_est_la_date_de_debut_de_leruption_cutanee",
            "Si_le_cas_suspect_a_eu_une_fievre_quelle_est_la_date_du_debut_de_la_fievre",
            "Date_debut_symptomes",
            "Date_debut_maladie",
        ],
        "notif_candidates": ["Date_notification"],
        "adm_candidates": ["Date_d_hospitalisation_isolement", "Date_admission_au_CT"],
        "prel_candidates": ["Date_prelevement", "Date_d_envoie_d_echantillons_au_laboratoire"],
    },
    "ebola": {
        "label": "Ebola / MVE (line list)",
        "default_sheet": "LL_Ebola",
        "rename_map": {
            "Date_debut_symptomes": "Date_debut_maladie",
            "Date_issue": "Date_sortie_au_CT",
        },
        "onset_candidates": ["Date_debut_maladie", "Date_debut_symptomes"],
        "notif_candidates": ["Date_notification"],
        "adm_candidates": ["Date_admission_au_CT"],
        "prel_candidates": ["Date_prelevement"],
    },
    "intox": {
        "label": "Intoxication (line list)",
        "default_sheet": "LL_Intox",
        "rename_map": {
            "Date_consultation": "Date_notification",
            "Date_apparition_signes": "Date_debut_maladie",
        },
        "onset_candidates": ["Date_debut_maladie", "Date_apparition_signes"],
        "notif_candidates": ["Date_notification", "Date_consultation"],
        "adm_candidates": ["Date_admission_au_CT"],
        "prel_candidates": ["Date_prelevement"],
    },
    "idsr": {
        "label": "IDSR agrégé (hebdo)",
        "default_sheet": "IDSR",
        "rename_map": {
            "Num": "Num",
            "Pays": "Pays",
            "Province": "Province_notification",
            "Zone_de_sante": "Zone_de_sante_notification",
            "NUMSEM": "Num_semaine_epid",
            "Year": "Annee_epid",
            "MALADIE": "Maladie",
            "TOTALCAS": "Total_cas",
            "TOTALDECES": "Total_deces",
        },
        "onset_candidates": ["Date_debut_semaine", "Date_notification"],
        "notif_candidates": ["Date_debut_semaine", "Date_notification"],
        "adm_candidates": [],
        "prel_candidates": [],
    },
}

def _coalesce_first(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """Retourne la première colonne non-NA dans candidates (coalesce)."""
    if not candidates:
        return pd.Series([pd.NA] * len(df), index=df.index)
    out = None
    for c in candidates:
        if c in df.columns:
            s = df[c]
            out = s if out is None else out.combine_first(s)
    if out is None:
        out = pd.Series([pd.NA] * len(df), index=df.index)
    return out

def standardize_ll_by_disease(df: pd.DataFrame, disease_key: str) -> pd.DataFrame:
    """
    1) Renommage spécifique maladie (DISEASE_SPECS[disease_key]['rename_map'])
    2) Standardisation core (standardize_ll_core)
    3) Coalesce dates: Date_debut_maladie / Date_notification / Date_admission_au_CT / Date_prelevement
       à partir des candidats de la maladie (si colonnes manquantes ou vides)
    """
    spec = DISEASE_SPECS.get(disease_key, DISEASE_SPECS["cholera"])
    df = _clean_colnames(df)

    # 1) Rename spécifique
    rmap = spec.get("rename_map", {}) or {}
    # Renommage seulement si la colonne source existe ET la cible n'existe pas déjà
    for src, dst in rmap.items():
        if (src in df.columns) and (dst not in df.columns):
            df = df.rename(columns={src: dst})

    # 2) Core
    df = standardize_ll_core(df)

    # 3) Coalesce dates (si vides)
    # - On convertit toutes les candidates en datetime (robuste)
    for colset in ["onset_candidates", "notif_candidates", "adm_candidates", "prel_candidates"]:
        for c in spec.get(colset, []) or []:
            if c in df.columns:
                df[c] = _to_dt(df[c])

    # On remplit les colonnes standard si elles sont totalement vides
    if ("Date_debut_maladie" in df.columns) and df["Date_debut_maladie"].isna().all():
        df["Date_debut_maladie"] = _coalesce_first(df, spec.get("onset_candidates", []))
    if ("Date_notification" in df.columns) and df["Date_notification"].isna().all():
        df["Date_notification"] = _coalesce_first(df, spec.get("notif_candidates", []))
    if ("Date_admission_au_CT" in df.columns):
        if df["Date_admission_au_CT"].isna().all():
            df["Date_admission_au_CT"] = _coalesce_first(df, spec.get("adm_candidates", []))
    if ("Date_prelevement" in df.columns):
        if df["Date_prelevement"].isna().all():
            df["Date_prelevement"] = _coalesce_first(df, spec.get("prel_candidates", []))

    # Recalcul ISO si nécessaire après coalesce
    # (ex: Mpox où Date_debut_maladie était vide et vient d'être rempli)
    need_year = df["Annee_epid"].isna().all()
    need_week = df["Num_semaine_epid"].isna().all()
    if need_year or need_week:
        ref = None
        if df["Date_notification"].notna().any():
            ref = df["Date_notification"]
        elif df["Date_debut_maladie"].notna().any():
            ref = df["Date_debut_maladie"]
        if ref is not None:
            iso = ref.dt.isocalendar()
            if need_year:
                df["Annee_epid"] = iso["year"].astype("Int64")
            if need_week:
                df["Num_semaine_epid"] = iso["week"].astype("Int64")
            y = pd.to_numeric(df["Annee_epid"], errors="coerce").astype("Int64")
            w = pd.to_numeric(df["Num_semaine_epid"], errors="coerce").astype("Int64")
            df["Semaine_epid"] = y.astype("string") + "-W" + w.astype("string").str.zfill(2)

    return df


# Provinces épidémiques (tes paramètres)
EPIDEMIE = {
    "Bas Uele": False, "Equateur": True, "Haut Katanga": True, "Haut Lomami": True,
    "Haut Uele": False, "Ituri": False, "Kasai Central": False, "Kasai": False,
    "Kinshasa": True, "Kongo Central": False, "Kasai Oriental": True, "Kwango": False,
    "Kwilu": False, "Lomami": True, "Lualaba": True, "Maindombe": True, "Maniema": True,
    "Mongala": True, "Nord Kivu": True, "Nord Ubangi": False, "Sankuru": True,
    "Sud Kivu": True, "Sud Ubangi": False, "Tanganyika": True, "Tshuapa": False,
    "Tshopo": True,
}
PROVINCES_EPID = [p for p, ok in EPIDEMIE.items() if ok]
PROVINCES_END  = [p for p, ok in EPIDEMIE.items() if not ok]


# =========================
# HELPERS (UI)
# =========================
def st_plot(fig, key=None, height=None, stretch=True):
    """Affiche une figure Plotly de manière robuste et compatible Streamlit ≥ 1.31.

    - Remplace use_container_width (déprécié) par width
    - width='stretch'  -> pleine largeur
    - width='content'  -> largeur naturelle
    - N'envoie jamais height=None
    """
    if fig is None:
        st.info("Aucune donnée à afficher (figure vide / colonnes manquantes).")
        return

    kwargs = {}

    # ✅ Nouveau standard Streamlit
    kwargs["width"] = "stretch" if stretch else "content"

    if key is not None:
        kwargs["key"] = key

    if height is not None:
        kwargs["height"] = height

    return st.plotly_chart(fig, **kwargs)



def apply_plotly_value_annotations(fig: Optional[go.Figure], enabled: bool) -> Optional[go.Figure]:
    """Ajoute des annotations (valeurs) sur les graphiques Plotly, de façon générique.
    - Bar: valeurs au-dessus des barres
    - Line/Scatter: valeurs au-dessus des points (si markers)
    """
    if fig is None or not enabled:
        return fig

    try:
        for tr in fig.data:
            # Bar charts
            if isinstance(tr, go.Bar):
                # Si text déjà présent, on le respecte
                if tr.text is None:
                    tr.text = tr.y
                # Position/format
                tr.texttemplate = "%{text}"
                tr.textposition = "outside"
                tr.cliponaxis = False

            # Line charts / scatter
            elif isinstance(tr, go.Scatter):
                # N'annoter que si on a des y numériques
                if tr.y is None:
                    continue
                # Ajoute le texte uniquement si pas déjà en mode text
                mode = tr.mode or ""
                if "text" not in mode:
                    tr.mode = (mode + "+text") if mode else "lines+markers+text"
                if tr.text is None:
                    tr.text = tr.y
                tr.textposition = "top center"
    except Exception:
        # On ne casse jamais l'app si Plotly refuse une propriété
        return fig

    return fig

def pick_age_col(df):
    """Choisir automatiquement la meilleure colonne tranche d’âge disponible."""
    if COL_AGEG2 in df.columns and df[COL_AGEG2].notna().any():
        return COL_AGEG2
    if COL_AGEG in df.columns and df[COL_AGEG].notna().any():
        return COL_AGEG
    return None

def ensure_lower(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.lower()
    return df

# =========================
# HELPERS (DATA CLEAN)
# =========================
def clean_str(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .str.strip()
         .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )

def norm_yesno(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in ["oui", "o", "y", "yes", "1", "true", "vrai"]:
        return "Oui"
    if s in ["non", "n", "no", "0", "false", "faux"]:
        return "Non"
    return str(x).strip()

def is_positive(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip().lower()
    return "posit" in s or s in ["pos", "+"]

def is_death(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip().lower()
    keys = ["deces", "décès", "decede", "décédé", "mort", "death", "dead", "dcd"]
    return any(k in s for k in keys)

def safe_to_datetime(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def make_yw(df):
    # clé Year-Week ex: 2026-W01
    if COL_YEAR in df.columns and COL_WNUM in df.columns:
        y = pd.to_numeric(df[COL_YEAR], errors="coerce").astype("Int64")
        w = pd.to_numeric(df[COL_WNUM], errors="coerce").astype("Int64")
        df["YW"] = y.astype("string") + "-W" + w.astype("string").str.zfill(2)
    else:
        df["YW"] = pd.NA
    return df

def taux_binaire(df, col, positive="Oui"):
    if col not in df.columns or len(df) == 0:
        return np.nan, 0
    s = df[col].astype("string")
    denom = int(s.notna().sum())
    if denom == 0:
        return np.nan, 0
    num = int((s == positive).sum())
    return (num / denom * 100.0), denom

def group_rate(df, group_col, indicator_col, positive_value):
    if group_col not in df.columns or indicator_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "n", "n_pos", "taux_%"])
    tmp = df[[group_col, indicator_col]].copy()
    tmp = tmp[tmp[group_col].notna()]
    tmp["is_pos"] = tmp[indicator_col].astype("string") == positive_value
    g = tmp.groupby(group_col, as_index=False).agg(
        n=(indicator_col, lambda x: int(x.notna().sum())),
        n_pos=("is_pos", "sum"),
    )
    g["taux_%"] = np.where(g["n"] > 0, g["n_pos"] / g["n"] * 100, np.nan)
    return g.sort_values(group_col)

def group_cfr(df, group_col):
    if group_col not in df.columns or COL_ISSUE not in df.columns:
        return pd.DataFrame(columns=[group_col, "cas", "deces", "cfr_%"])
    tmp = df[[group_col, COL_ISSUE]].copy()
    tmp = tmp[tmp[group_col].notna()]
    tmp["is_death"] = tmp[COL_ISSUE].apply(is_death)
    g = tmp.groupby(group_col, as_index=False).agg(
        cas=(COL_ISSUE, "size"),
        deces=("is_death", "sum"),
    )
    g["cfr_%"] = np.where(g["cas"] > 0, g["deces"] / g["cas"] * 100, np.nan)
    return g.sort_values(group_col)



# =========================
# INDICATEURS – DÉFINITIONS COHÉRENTES (utilisés partout)
# =========================
def _norm_txt_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna(pd.NA)
    return s.str.strip().str.lower()

YES_SET = {"oui", "o", "y", "yes", "1", "true", "vrai"}

TDR_POS_SET = {"positif", "positive", "pos", "+", "tdr positif"}
TDR_NEG_SET = {"negatif", "négatif", "negative", "neg", "-", "tdr negatif", "tdr négatif"}

def _is_yes_series(s: pd.Series) -> pd.Series:
    return _norm_txt_series(s).isin(YES_SET)

def _tdr_result_norm(s: pd.Series) -> pd.Series:
    return _norm_txt_series(s)

def compute_indicators(df_in: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcule les KPI avec des dénominateurs cohérents (= nombre de cas filtrés).

    Définitions:
    - CFR% = décès / tous cas filtrés
    - Prélèvement% = (Prelevement == Oui) / tous cas filtrés
    - Hospitalisation% = (Hospitalisation == Oui) / tous cas filtrés
    - TDR_réalisé% = (TDR_realise == Oui) / tous cas filtrés
    - Couverture TDR% = identique à TDR_réalisé% mais renvoyée explicitement (num/den)
    - Positivité TDR% = positifs / (positifs + négatifs) parmi:
        (TDR_realise == Oui) ET (TDR_Resultat ∈ {pos, neg})
      => on exclut invalide, non prélevé, en attente, etc.
    """

    df = df_in.copy()
    n_cases = int(len(df))

    # -----------------------------
    # Décès & CFR (sur tous cas)
    # -----------------------------
    n_deaths = int(df["is_death"].sum()) if "is_death" in df.columns else 0
    cfr_pct = (n_deaths / n_cases * 100.0) if n_cases > 0 else np.nan

    # -----------------------------
    # Helper taux binaire Oui/Non (den = n_cases)
    # -----------------------------
    def _rate_yes(col_name: str) -> Tuple[float, int, int]:
        """Retourne (taux%, num_oui, denom_cases)."""
        if col_name not in df.columns or n_cases == 0:
            return (np.nan, 0, n_cases)
        num = int(_is_yes_series(df[col_name]).sum())
        return (num / n_cases * 100.0, num, n_cases)

    prelev_pct, n_prelev_yes, den_cases = _rate_yes(COL_PREL)
    hosp_pct, n_hosp_yes, _ = _rate_yes(COL_HOSP)
    tdr_pct, n_tdr_yes, _ = _rate_yes(COL_TDR)

    # -----------------------------
    # Couverture TDR (explicite)
    # -----------------------------
    # (équivalent de tdr_pct, mais renvoyé avec num/den dédiés)
    tdr_coverage_num = n_tdr_yes
    tdr_coverage_den = den_cases
    tdr_coverage_pct = tdr_pct

    # -----------------------------
    # Positivité TDR (sur TDR réalisés + résultats valides)
    # -----------------------------
    if (COL_TDR in df.columns) and (COL_TDRR in df.columns) and (n_cases > 0):
        tdr_yes = _is_yes_series(df[COL_TDR])
        res_n = _tdr_result_norm(df[COL_TDRR])

        # Résultats valides = pos/neg uniquement
        valid_res = res_n.isin(TDR_POS_SET.union(TDR_NEG_SET))

        # Dénominateur positivité = TDR=Oui & (pos/neg)
        pos_den = int((tdr_yes & valid_res).sum())

        # Numérateur = TDR=Oui & positif
        pos_num = int((tdr_yes & res_n.isin(TDR_POS_SET)).sum())

        pos_pct = (pos_num / pos_den * 100.0) if pos_den > 0 else np.nan
    else:
        pos_den, pos_num, pos_pct = 0, 0, np.nan

    # -----------------------------
    # Taux d'invalides (utile pour qualité test)
    # -----------------------------
    # On le calcule parmi TDR réalisés (=Oui). On inclut "invalide", "inba", etc.
    invalid_num = 0
    invalid_den = 0
    invalid_pct = np.nan

    if (COL_TDR in df.columns) and (COL_TDRR in df.columns) and (n_cases > 0):
        tdr_yes = _is_yes_series(df[COL_TDR])
        res_n = _tdr_result_norm(df[COL_TDRR])

        # Définition "invalide" (ajuste au besoin)
        invalid_set = {"invalide", "invalid", "inba", "bande absente"}

        invalid_den = int(tdr_yes.sum())
        invalid_num = int((tdr_yes & res_n.isin(invalid_set)).sum())
        invalid_pct = (invalid_num / invalid_den * 100.0) if invalid_den > 0 else np.nan

    # -----------------------------
    # Degré déshydratation (table d'effectifs)
    # -----------------------------
    if COL_DEHY in df.columns and n_cases > 0:
        dehy_tbl = (
            df[COL_DEHY]
            .astype("string")
            .fillna("Inconnu")
            .str.strip()
            .replace({"": "Inconnu"})
            .value_counts(dropna=False)
            .rename_axis(COL_DEHY)
            .reset_index(name="Nombre_de_cas")
        )
    else:
        dehy_tbl = pd.DataFrame(columns=[COL_DEHY, "Nombre_de_cas"])

    return {
        "n_cases": n_cases,
        "n_deaths": n_deaths,
        "cfr_pct": cfr_pct,

        "prelev_pct": prelev_pct,
        "prelev_num": n_prelev_yes,
        "prelev_den": den_cases,

        "hosp_pct": hosp_pct,
        "hosp_num": n_hosp_yes,
        "hosp_den": den_cases,

        "tdr_pct": tdr_pct,          # % TDR=Oui sur tous cas
        "tdr_num": n_tdr_yes,
        "tdr_den": den_cases,

        # Couverture TDR (explicite)
        "tdr_coverage_pct": tdr_coverage_pct,
        "tdr_coverage_num": tdr_coverage_num,
        "tdr_coverage_den": tdr_coverage_den,

        # Positivité (sur résultats valides)
        "pos_pct": pos_pct,
        "pos_num": pos_num,
        "pos_den": pos_den,

        # Invalides (qualité)
        "invalid_pct": invalid_pct,
        "invalid_num": invalid_num,
        "invalid_den": invalid_den,

        "dehy_tbl": dehy_tbl,
    }

def compute_group_indicators(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Table d'indicateurs par groupe avec les mêmes définitions/denoms."""
    if df_in is None or df_in.empty or group_col not in df_in.columns:
        return pd.DataFrame(columns=[group_col, "Cas", "Décès", "CFR_%", "Prélèvement_%", "Hospitalisation_%", "TDR_réalisé_%", "Positivité_TDR_%"])

    df = df_in.copy()
    df = df[df[group_col].notna()]

    # Base (cas, décès)
    g = df.groupby(group_col, as_index=False).agg(
        Cas=(group_col, "size"),
        Décès=("is_death", "sum") if "is_death" in df.columns else (group_col, "size"),
    )
    if "is_death" not in df.columns:
        g["Décès"] = 0

    g["CFR_%"] = np.where(g["Cas"] > 0, g["Décès"] / g["Cas"] * 100.0, np.nan)

    def _add_rate(col, new_name):
        if col not in df.columns:
            g[new_name] = np.nan
            return
        tmp = df[[group_col, col]].copy()
        tmp["is_yes"] = _is_yes_series(tmp[col])
        num = tmp.groupby(group_col)["is_yes"].sum()
        den = tmp.groupby(group_col).size()
        g[new_name] = (num / den * 100.0).reindex(g[group_col]).to_numpy()

    _add_rate(COL_PREL, "Prélèvement_%")
    _add_rate(COL_HOSP, "Hospitalisation_%")
    _add_rate(COL_TDR, "TDR_réalisé_%")

    # Positivité (parmi TDR=Oui + résultat valide)
    if (COL_TDR in df.columns) and (COL_TDRR in df.columns):
        tdr_yes = _is_yes_series(df[COL_TDR])
        res_n = _tdr_result_norm(df[COL_TDRR])
        valid_res = res_n.isin(TDR_POS_SET.union(TDR_NEG_SET))
        df_pos = df[[group_col]].copy()
        df_pos["den_pos"] = (tdr_yes & valid_res).astype(int)
        df_pos["num_pos"] = (tdr_yes & res_n.isin(TDR_POS_SET)).astype(int)
        sums = df_pos.groupby(group_col, as_index=False).agg(den_pos=("den_pos", "sum"), num_pos=("num_pos", "sum"))
        g = g.merge(sums, on=group_col, how="left")
        g["Positivité_TDR_%"] = np.where(g["den_pos"] > 0, g["num_pos"] / g["den_pos"] * 100.0, np.nan)
        g = g.drop(columns=["den_pos", "num_pos"])
    else:
        g["Positivité_TDR_%"] = np.nan

    return g.sort_values(group_col)


def delay_days(df, date_end, date_start, new_col):
    if date_end in df.columns and date_start in df.columns:
        df[new_col] = (df[date_end] - df[date_start]).dt.days
    else:
        df[new_col] = np.nan
    return df

def pct_under_threshold(series, threshold=2):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return np.nan, 0
    n = len(series)
    under = int((series <= threshold).sum())
    return (under / n * 100.0), n

def compile_from_folder(folder, pattern, sheet=None):
    files = sorted(glob.glob(str(Path(folder) / pattern)))
    if not files:
        return pd.DataFrame(), []
    dfs = []
    for f in files:
        try:
            d = pd.read_excel(f, sheet_name=sheet) if sheet else pd.read_excel(f)
            d["__source_file__"] = os.path.basename(f)
            dfs.append(d)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame(), files
    return pd.concat(dfs, ignore_index=True), files

@st.cache_data(show_spinner=False)
def load_data_from_excel(path):
    return pd.read_excel(path)

def standardize_df(df):
    df = df.copy()

    # Strings
    for c in [COL_PROV, COL_ZS, COL_AS, COL_SEX, COL_ISSUE, COL_CLASS, COL_TDRR, COL_AGEG, COL_AGEG2, COL_DEHY]:
        if c in df.columns:
            df[c] = clean_str(df[c])

    # Yes/No
    for c in [COL_PREL, COL_TDR, COL_HOSP]:
        if c in df.columns:
            df[c] = df[c].apply(norm_yesno)

    # Dates
    df = safe_to_datetime(df, [DATE_ONSET, DATE_ADM, DATE_PREL])

    # Year/week
    df = make_yw(df)

    # décès bool
    df["is_death"] = df[COL_ISSUE].apply(is_death) if COL_ISSUE in df.columns else False

    # positivité
    df["is_tdr_pos"] = df[COL_TDRR].apply(is_positive) if COL_TDRR in df.columns else False

    # délais
    df = delay_days(df, DATE_ADM, DATE_ONSET, "delai_onset_to_adm")
    df = delay_days(df, DATE_PREL, DATE_ONSET, "delai_onset_to_prel")

    return df


# =========================
# HELPERS (Qualité & Alertes)
# =========================
def qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne un tableau long des incohérences (1 ligne = 1 flag = 1 cas)."""
    out = []

    def _add(mask, flag_name):
        if mask is None:
            return
        idx = df.index[mask.fillna(False)].tolist() if hasattr(mask, "fillna") else df.index[mask].tolist()
        if len(idx):
            out.extend([{"row_id": int(i), "flag": flag_name} for i in idx])

    # Dates incohérentes (délais négatifs)
    if "delai_onset_to_adm" in df.columns:
        _add(df["delai_onset_to_adm"] < 0, "Date admission < début maladie")
    if "delai_onset_to_prel" in df.columns:
        _add(df["delai_onset_to_prel"] < 0, "Date prélèvement < début maladie")

    # TDR non réalisé mais résultat rempli
    if (COL_TDR in df.columns) and (COL_TDRR in df.columns):
        _add((df[COL_TDR].astype("string") == "Non") & (df[COL_TDRR].notna()), "TDR=Non mais résultat renseigné")

    # TDR résultat mais TDR réalisé manquant
    if (COL_TDR in df.columns) and (COL_TDRR in df.columns):
        _add((df[COL_TDR].isna()) & (df[COL_TDRR].notna()), "Résultat TDR sans statut TDR")

    # Grossesse incohérente (si colonne existe)
    if ("Femme_enceinte" in df.columns) and (COL_SEX in df.columns):
        s_sex = df[COL_SEX].astype("string").str.lower()
        s_preg = df["Femme_enceinte"].astype("string").str.lower()
        _add((s_preg == "oui") & (~s_sex.str.contains("fem")), "Femme_enceinte=Oui mais sexe ≠ féminin")

    # Âge impossible (si Age existe)
    if (COL_AGE in df.columns):
        age_num = pd.to_numeric(df[COL_AGE], errors="coerce")
        _add((age_num < 0) | (age_num > 120), "Âge hors limites (0–120)")

    # Issue manquante (utile pour CFR fiable)
    if COL_ISSUE in df.columns:
        _add(df[COL_ISSUE].isna(), "Issue manquante")

    if not out:
        return pd.DataFrame(columns=["row_id", "flag"])
    return pd.DataFrame(out)

def completeness_table(df: pd.DataFrame, cols_required: list[str], by: str) -> pd.DataFrame:
    """Complétude (%) des champs clés par groupe. Robuste aux doublons de colonnes."""

    if df is None or df.empty:
        return pd.DataFrame(columns=[by, "n", "score_completude_%"])

    # ✅ Fix: colonnes dupliquées -> garde la première occurrence
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # 1) Colonnes existantes
    cols = [c for c in cols_required if c in df.columns]

    # 2) Évite que `by` soit dans `cols` (sinon doublon dans df[[by]+cols])
    cols = [c for c in cols if c != by]

    # 3) Dé-dup au cas où
    cols = list(dict.fromkeys(cols))

    if (by not in df.columns) or (len(cols) == 0):
        return pd.DataFrame(columns=[by, "n", "score_completude_%"])

    tmp = df[[by] + cols].copy()
    tmp = tmp[tmp[by].notna()]

    if tmp.empty:
        return pd.DataFrame(columns=[by, "n", "score_completude_%"])

    g = tmp.groupby(by, as_index=False).agg(n=(by, "size"))

    for c in cols:
        g[c] = tmp.groupby(by)[c].apply(lambda x: float(x.notna().mean() * 100)).values

    g["score_completude_%"] = g[cols].mean(axis=1).round(1)
    return g.sort_values("score_completude_%", ascending=True)

def cascade_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cascade prélèvement -> TDR -> résultat valide -> positif (sur les données filtrées).

    Améliorations vs version initiale :
    - Cascade "entonnoir" avec dénominateurs séquentiels (prélevé -> TDR -> résultat).
    - "Résultat disponible" = seulement résultats biologiques valides (positif/négatif),
      donc on exclut les statuts type "non réalisé/non prélevé", "en cours", etc.
    - Détection & quantification des incohérences (résultat renseigné mais TDR_realise != Oui).
    - Calcule le positif à partir de TDR_Resultat si is_tdr_pos absent.
    """

    n_all = int(len(df))

    # Colonnes (si absentes -> séries NA pour ne pas planter)
    prelev = df[COL_PREL].astype("string") if COL_PREL in df.columns else pd.Series([pd.NA] * n_all)
    tdr    = df[COL_TDR].astype("string")  if COL_TDR  in df.columns else pd.Series([pd.NA] * n_all)
    tdr_res_raw = df[COL_TDRR].astype("string") if COL_TDRR in df.columns else pd.Series([pd.NA] * n_all)

    # Normalisation minimale (trim + lower) pour gérer variantes d’écriture
    def _norm(s: pd.Series) -> pd.Series:
        s = s.fillna(pd.NA).astype("string")
        return s.str.strip().str.lower()

    prelev_n = _norm(prelev)
    tdr_n    = _norm(tdr)
    res_n    = _norm(tdr_res_raw)

    # Valeurs "Oui" possibles (à élargir si nécessaire)
    YES = {"oui", "yes", "y", "1", "true", "vrai"}

    # Résultats biologiques valides (inclut variantes FR/EN)
    POS_SET = {"positif", "positive", "+"}
    NEG_SET = {"négatif", "negatif", "negative", "-"}

    # Statuts / non-résultats fréquents observés (à exclure du "résultat disponible")
    # ex: "non réalisé/non prélevé" est un statut, pas un résultat
    NON_RESULT_HINTS = {
        "non réalisé", "non realise", "non realisé", "non preleve", "non prélevé",
        "non réalisé/non prélevé", "non realise/non preleve",
        "en cours", "en attente"
    }

    def _is_yes(s: pd.Series) -> pd.Series:
        return s.isin(YES)

    def _is_valid_result(s: pd.Series) -> pd.Series:
        # Valide si positif ou négatif (strict)
        return s.isin(POS_SET.union(NEG_SET))

    def _is_non_result_status(s: pd.Series) -> pd.Series:
        # Heuristique : contient un des fragments de statut
        # (utile pour mesurer les cas où on a "rempli" TDR_Resultat avec un statut)
        patt = "|".join([p.replace("/", r"\/") for p in sorted(NON_RESULT_HINTS)])
        return s.str.contains(patt, case=False, na=False)

    prelev_yes = _is_yes(prelev_n)
    tdr_yes    = _is_yes(tdr_n)

    # Comptes séquentiels (entonnoir)
    n_prelev = int(prelev_yes.sum())
    n_tdr    = int((prelev_yes & tdr_yes).sum())  # TDR parmi les prélevés
    valid_res_mask = (prelev_yes & tdr_yes & _is_valid_result(res_n))
    n_res = int(valid_res_mask.sum())

    # Positifs : priorité à is_tdr_pos si disponible, sinon via résultat
    if "is_tdr_pos" in df.columns:
        # On ne compte les positifs que parmi les résultats valides (entonnoir)
        is_pos = df["is_tdr_pos"].fillna(0).astype(int) == 1
        n_pos = int((valid_res_mask & is_pos).sum())
    else:
        n_pos = int((prelev_yes & tdr_yes & res_n.isin(POS_SET)).sum())

    # Qualité / incohérences (diagnostic)
    # 1) Résultat renseigné (non NA) alors que TDR_realise != Oui
    res_filled = tdr_res_raw.notna()
    incoh_res_without_tdr = int((res_filled & ~tdr_yes).sum())

    # 2) "TDR_Resultat" rempli avec un statut type "non réalisé/non prélevé"
    status_in_result = int(_is_non_result_status(res_n).sum())

    # 3) Résultats valides (pos/neg) alors que TDR_realise != Oui (plus grave)
    incoh_validres_without_tdr = int((_is_valid_result(res_n) & ~tdr_yes).sum())

    def _pct(num: int, den: int) -> float:
        return np.nan if den == 0 else (num / den * 100.0)

    rows = [
        ["Tous cas", n_all, n_all, 100.0],

        # Cascade séquentielle
        ["Prélèvement=Oui", n_prelev, n_all, _pct(n_prelev, n_all)],
        ["TDR réalisé=Oui (parmi prélevés)", n_tdr, n_prelev, _pct(n_tdr, n_prelev)],
        ["Résultat TDR valide (Positif/Négatif) (parmi TDR)", n_res, n_tdr, _pct(n_res, n_tdr)],
        ["TDR positif (parmi résultats valides)", n_pos, n_res, _pct(n_pos, n_res)],

        # Qualité des données (signaux)
        ["⚠ Résultat renseigné mais TDR_realise != Oui", incoh_res_without_tdr, n_all, _pct(incoh_res_without_tdr, n_all)],
        ["⚠ Statut saisi dans TDR_Resultat (ex: non réalisé/non prélevé)", status_in_result, n_all, _pct(status_in_result, n_all)],
        ["⚠ Résultat valide (Pos/Nég) mais TDR_realise != Oui", incoh_validres_without_tdr, n_all, _pct(incoh_validres_without_tdr, n_all)],
    ]

    return pd.DataFrame(rows, columns=["Étape", "n", "Dénominateur", "%"])

def alerts_weekly_simple(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Alerte simple:
    - calcule cas par YW et groupe
    - % variation vs semaine précédente
    - baseline = moyenne des 3 semaines précédentes (t-1,t-2,t-3)
    """
    if ("YW" not in df.columns) or (group_col not in df.columns):
        return pd.DataFrame(columns=[group_col, "YW", "Cas", "Cas_prev", "var_%", "baseline_3w", "signal"])

    tmp = df[[group_col, "YW"]].dropna().copy()
    weekly = tmp.groupby([group_col, "YW"], as_index=False).size().rename(columns={"size": "Cas"})
    weekly = weekly.sort_values([group_col, "YW"])

    # Cas_prev
    weekly["Cas_prev"] = weekly.groupby(group_col)["Cas"].shift(1)

    # Variation %
    weekly["var_%"] = np.where(
        weekly["Cas_prev"].fillna(0) > 0,
        (weekly["Cas"] - weekly["Cas_prev"]) / weekly["Cas_prev"] * 100,
        np.nan
    )

    # baseline moyenne 3 semaines précédentes
    weekly["baseline_3w"] = (
        weekly.groupby(group_col)["Cas"]
              .shift(1)
              .rolling(3, min_periods=2)
              .mean()
              .reset_index(level=0, drop=True)
    )

    # signal: hausse forte ET dépasse baseline
    weekly["signal"] = (
        (weekly["Cas"] >= (weekly["baseline_3w"] * 1.5)) &
        (weekly["Cas"] >= 10)  # tu peux régler
    )

    return weekly

# =========================
# HELPERS (Sitrep)
# =========================
def export_sitrep_pdf(payload):
    """
    Export PDF robuste :
    - fonctionne même si certaines sections n'existent pas
    - imprime uniquement les sections disponibles
    - supporte l'ajout d'images (PNG bytes) via payload["images"]
      Formats acceptés:
        - [(title, png_bytes), ...]
        - [{"title": "...", "bytes": png_bytes}, ...]
    """
    from io import BytesIO

    # Import reportlab seulement si dispo
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except Exception as e:
        raise ModuleNotFoundError(
            "Le module 'reportlab' n'est pas installé. "
            "Installe-le via: pip install reportlab  (ou conda install -c conda-forge reportlab)"
        ) from e

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    # -------------------------
    # Helpers mise en page
    # -------------------------
    left = 50
    top = h - 50
    bottom_margin = 70
    y = top

    def new_page():
        nonlocal y
        c.showPage()
        y = top

    def ensure_space(required_height):
        """Saute de page si l'espace restant est insuffisant."""
        nonlocal y
        if y - required_height < bottom_margin:
            new_page()

    def draw_title(txt, size=11, gap=6):
        nonlocal y
        ensure_space(size + 10)
        c.setFont("Helvetica-Bold", size)
        c.drawString(left, y, str(txt))
        y -= (size + gap)

    def draw_line(txt, size=10, x=None, gap=3):
        nonlocal y
        x = left + 10 if x is None else x
        ensure_space(size + 10)
        c.setFont("Helvetica", size)
        # sécurité: éviter None
        c.drawString(x, y, str(txt) if txt is not None else "")
        y -= (size + gap)

    def add_list(title, items):
        """Ajoute une section liste si items non vide."""
        if not items:
            return
        if isinstance(items, (str, int, float)):
            items = [str(items)]
        draw_title(title, size=11)
        for it in items:
            draw_line(f"- {it}", size=10, x=left + 10)

    def add_table_simple(title, table_df, max_rows=25):
        """Table texte simple (robuste)."""
        draw_title(title, size=11)
        c.setFont("Helvetica", 9)

        if table_df is not None and hasattr(table_df, "empty") and (not table_df.empty):
            draw_line("ZS | Cas | Décès", size=9, x=left)
            for _, r in table_df.head(max_rows).iterrows():
                zs = str(r.iloc[0])
                cas = int(r.get("cas", 0)) if hasattr(r, "get") else 0
                dec = int(r.get("deces", 0)) if hasattr(r, "get") else 0
                draw_line(f"{zs} | {cas} | {dec}", size=9, x=left)
        else:
            draw_line("Données indisponibles.", size=9, x=left)

    def add_image(png_bytes, title=None, max_w=None, max_h=360):
        """
        Ajoute une image PNG (bytes) avec redimensionnement automatique.
        - max_w par défaut = largeur page - marges
        """
        nonlocal y
        if not png_bytes:
            return

        if title:
            draw_title(title, size=11)

        max_w = (w - 2 * left) if max_w is None else max_w

        img = ImageReader(BytesIO(png_bytes))
        iw, ih = img.getSize()

        # scale pour rentrer dans la page
        scale = min(max_w / float(iw), max_h / float(ih), 1.0)
        dw, dh = iw * scale, ih * scale

        ensure_space(dh + 20)
        c.drawImage(
            img,
            left,
            y - dh,
            width=dw,
            height=dh,
            preserveAspectRatio=True,
            mask="auto",
        )
        y -= dh + 15

    # -------------------------
    # HEADER
    # -------------------------
    meta = payload.get("meta", {})
    semaine = meta.get("semaine", "-")
    annee = meta.get("annee", "-")
    date_pub = meta.get("date_publication", "")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, y, f"SITREP CHOLERA - Semaine {semaine} / {annee}")
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(left, y, f"Date de publication: {date_pub}")
    y -= 25

    # -------------------------
    # POINTS SAILLANTS
    # -------------------------
    draw_title("Points saillants", size=11)

    kpi = payload.get("kpi", {})
    bullets = payload.get("points_saillants") or [
        f"Cas (SE): {kpi.get('cas_semaine', 0)}",
        f"Décès (SE): {kpi.get('deces_semaine', 0)}",
        f"CFR (SE): {float(kpi.get('cfr_semaine', 0.0)):.2f}%",
    ]

    for b in bullets:
        draw_line(f"- {b}", size=10, x=left + 10)

    # -------------------------
    # TABLE EPIDEMIOLOGIQUE
    # -------------------------
    table = payload.get("table_epi")
    add_table_simple("Situation épidémiologique (par ZS)", table, max_rows=25)

    # -------------------------
    # SECTIONS OPTIONNELLES
    # -------------------------
    add_list("Défis et besoins", payload.get("defis_besoins"))
    add_list("Perspectives", payload.get("perspectives"))

    # -------------------------
    # ANNEXES: IMAGES
    # -------------------------
    images = payload.get("images", [])
    if images:
        draw_title("Annexes graphiques", size=12)

        for item in images:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                t, b = item
                add_image(b, title=t)
            elif isinstance(item, dict):
                add_image(item.get("bytes"), title=item.get("title", ""))
            else:
                # si juste bytes
                add_image(item, title=None)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# =========================
# HELPERS (GEO + FUZZY JOIN)
# =========================
def _norm_key(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = "".join(c for c in unicodedata.normalize("NFD", x) if unicodedata.category(c) != "Mn")
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[’'`]", " ", x)
    x = re.sub(r"[^a-z0-9\s\-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def _fuzzy_best_match(query, choices):
    """Retourne (best_choice, score_0_1)."""
    if not choices:
        return None, 0.0
    if HAS_RAPIDFUZZ:
        best = process.extractOne(query, choices, scorer=fuzz.token_sort_ratio)
        if best is None:
            return None, 0.0
        return best[0], best[1] / 100.0
    best_choice, best_score = None, 0.0
    for c in choices:
        sc = SequenceMatcher(None, query, c).ratio()
        if sc > best_score:
            best_score, best_choice = sc, c
    return best_choice, best_score

def joindre_donnees_fuzzy_geo(
    carte_gdf,
    df_donnees,
    colonne_cle_geo="name",
    colonne_cle_data=COL_PROV,
    colonne_valeurs="nb_cas",
    seuil=0.90,
):
    gdf = carte_gdf.copy()

    gdf["_key_geo"] = gdf[colonne_cle_geo].astype(str).map(_norm_key)
    d = df_donnees.copy()
    d["_key_data"] = d[colonne_cle_data].astype(str).map(_norm_key)

    if colonne_valeurs in d.columns:
        df_grouped = d.groupby("_key_data", as_index=False)[colonne_valeurs].sum()
    else:
        df_grouped = d.groupby("_key_data", as_index=False).size().rename(columns={"size": colonne_valeurs})

    choices = gdf["_key_geo"].dropna().unique().tolist()
    unique_data_keys = df_grouped["_key_data"].dropna().unique().tolist()

    mapping_rows = []
    for dk in unique_data_keys:
        best, score = _fuzzy_best_match(dk, choices)
        ok = (score >= seuil)
        mapping_rows.append({"key_data": dk, "key_geo": best if ok else None, "score": score, "matched": ok})

    df_map = pd.DataFrame(mapping_rows)

    df_grouped2 = df_grouped.merge(
        df_map[df_map["matched"]][["key_data", "key_geo"]],
        left_on="_key_data", right_on="key_data", how="left"
    )
    df_grouped2 = (
        df_grouped2.dropna(subset=["key_geo"])
                   .groupby("key_geo", as_index=False)[colonne_valeurs].sum()
    )

    gdf = gdf.merge(df_grouped2, left_on="_key_geo", right_on="key_geo", how="left")
    gdf[colonne_valeurs] = gdf[colonne_valeurs].fillna(0)

    match_rate = float(df_map["matched"].mean()) if len(df_map) else 0.0
    return gdf, df_map.sort_values("score", ascending=True), match_rate

def gdf_to_plotly_geojson(gdf, fid_col="fid"):
    g = gdf.copy()
    try:
        g = g.to_crs(epsg=4326)
    except Exception:
        pass
    g[fid_col] = g.index.astype(str)
    geojson = json.loads(g.to_json())
    return g, geojson

# =========================
# CORE STANDARDISATION (LINE LIST) — commun Rougeole/Choléra/…
# Objectif: garantir les colonnes clés (dates, âge, sexe, semaines ISO, geo)
# =========================
def __strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')

def _clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _to_dt(s: pd.Series) -> pd.Series:
    # robuste: excel numeric, texte, etc.
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def standardize_ll_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise une line list (toutes maladies) vers les colonnes communes attendues
    par les analyses (ou par des fonctions downstream).
    - Crée les colonnes si absentes
    - Convertit Date_notification / Date_debut_maladie
    - Calcule Annee_epid / Num_semaine_epid / Semaine_epid (ISO) à partir de la meilleure date dispo
    - Harmonise Sexe (Masculin/Feminin)
    - Calcule Age_en_ans et tranches si manquants
    """
    df = _clean_colnames(df)

    # --- Rename léger (variantes fréquentes -> standard)
    rename_map = {
        # Dates
        "date_notif": "Date_notification",
        "daterep": "Date_notification",
        "date_rep": "Date_notification",
        "date_rapportage": "Date_notification",
        "date_notification_cas": "Date_notification",
        "date_de_notification": "Date_notification",

        "date_debut": "Date_debut_maladie",
        "date_onset": "Date_debut_maladie",
        "date_debut_symptomes": "Date_debut_maladie",
        "date_symptomes": "Date_debut_maladie",

        # Geo
        "province": "Province_notification",
        "prov": "Province_notification",
        "province_notif": "Province_notification",

        "zs": "Zone_de_sante_notification",
        "zone_sante": "Zone_de_sante_notification",
        "zone_sante_notification": "Zone_de_sante_notification",
        "zs_notif": "Zone_de_sante_notification",

        "as": "Aire_de_sante_notification",
        "aire_sante": "Aire_de_sante_notification",
        "aire_sante_notification": "Aire_de_sante_notification",
        "as_notif": "Aire_de_sante_notification",

        # Temps
        "year": "Annee_epid",
        "annee": "Annee_epid",
        "epi_year": "Annee_epid",
        "week": "Num_semaine_epid",
        "numsem": "Num_semaine_epid",
        "num_sem": "Num_semaine_epid",
        "epi_week": "Num_semaine_epid",

        # Sexe / âge
        "sex": "Sexe",
        "gender": "Sexe",
        "age_unit": "Unite_age",
        "unite": "Unite_age",
        "unite_d_age": "Unite_age",
        "age_years": "Age_en_ans",
        "age_annees": "Age_en_ans",

        # Outcome / classif
        "outcome": "Issue",
        "evolution": "Issue",
        "classification": "Classification_finale",
        "classif": "Classification_finale",
        "statut_cas": "Classification_finale",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # --- Colonnes attendues (création si absentes)
    required = [
        "Date_notification", "Date_debut_maladie",
        "Province_notification", "Zone_de_sante_notification", "Aire_de_sante_notification",
        "Semaine_epid", "Num_semaine_epid", "Annee_epid",
        "Sexe", "Age", "Unite_age", "Age_en_ans",
        "Tranche_age", "Tranche_age_en_ans",
        "Issue", "Classification_finale",
    ]
    for c in required:
        if c not in df.columns:
            df[c] = pd.NA

    # --- Dates
    df["Date_notification"] = _to_dt(df["Date_notification"])
    df["Date_debut_maladie"] = _to_dt(df["Date_debut_maladie"])

    # --- Année/Semaine ISO: si manquantes, calculer depuis la meilleure date
    need_year = df["Annee_epid"].isna().all()
    need_week = df["Num_semaine_epid"].isna().all()

    if need_year or need_week:
        ref = None
        if df["Date_notification"].notna().any():
            ref = df["Date_notification"]
        elif df["Date_debut_maladie"].notna().any():
            ref = df["Date_debut_maladie"]

        if ref is not None:
            iso = ref.dt.isocalendar()  # year, week, day
            if need_year:
                df["Annee_epid"] = iso["year"].astype("Int64")
            if need_week:
                df["Num_semaine_epid"] = iso["week"].astype("Int64")

    # Semaine_epid = YYYY-Www
    y = pd.to_numeric(df["Annee_epid"], errors="coerce").astype("Int64")
    w = pd.to_numeric(df["Num_semaine_epid"], errors="coerce").astype("Int64")
    df["Semaine_epid"] = y.astype("string") + "-W" + w.astype("string").str.zfill(2)

    # --- Sexe (harmonisation simple)
    s = df["Sexe"].astype("string").str.strip().str.lower()
    s = s.apply(lambda v: _strip_accents(v) if pd.notna(v) else v)
    df["Sexe"] = np.where(s.isin(["m", "masculin", "male", "homme", "h"]), "Masculin",
                   np.where(s.isin(["f", "feminin", "féminin", "female", "femme"]), "Feminin", df["Sexe"]))

    # --- Age_en_ans depuis Age + Unite_age (si manquant)
    df["Age_en_ans"] = pd.to_numeric(df["Age_en_ans"], errors="coerce")

    age = pd.to_numeric(df["Age"], errors="coerce")
    unit = df["Unite_age"].astype("string").str.strip().str.lower()

    # retirer accents + normaliser (robuste)
    unit = unit.apply(lambda v: _strip_accents(v) if pd.notna(v) else v)

    # valeurs vides -> on suppose "ans"
    unit = unit.fillna("ans")

    # normalisation large des unités (tu peux enrichir)
    unit = unit.replace({
        # années
        "a": "ans", "an": "ans", "ans": "ans", "annee": "ans", "annees": "ans",
        "year": "ans", "years": "ans", "yr": "ans", "yrs": "ans", "y": "ans",

        # mois
        "m": "mois", "mo": "mois", "mos": "mois", "mois": "mois",
        "month": "mois", "months": "mois",

        # semaines
        "s": "semaines", "sem": "semaines", "semaine": "semaines", "semaines": "semaines",
        "week": "semaines", "weeks": "semaines", "w": "semaines",

        # jours
        "j": "jours", "jr": "jours", "jour": "jours", "jours": "jours",
        "day": "jours", "days": "jours", "d": "jours",
    })

    # calcul âge en années selon unité
    age_years = np.where(unit.eq("ans"), age,
                np.where(unit.eq("mois"), age / 12.0,
                np.where(unit.eq("semaines"), age / 52.0,
                np.where(unit.eq("jours"), age / 365.25, np.nan))))

    # optionnel: nettoyer âges aberrants (tu peux commenter si tu ne veux pas)
    age_years = pd.Series(age_years, index=df.index)
    age_years = age_years.where((age_years >= 0) & (age_years <= 120), np.nan)

    # fill safe (pas d’erreur de longueur)
    df["Age_en_ans"] = df["Age_en_ans"].fillna(age_years)

    # --- Tranches d'âge (si manquantes)
    df["Tranche_age"] = df["Tranche_age"].astype("string")
    df["Tranche_age_en_ans"] = df["Tranche_age_en_ans"].astype("string")

    a = pd.to_numeric(df["Age_en_ans"], errors="coerce")

    conds = [
        a.notna() & (a < (1/12)),
        a.notna() & (a >= (1/12)) & (a < 5),
        a.notna() & (a >= 5) & (a < 15),
        a.notna() & (a >= 15),
    ]
    lab_txt  = ["<1 mois", "1–59 mois", "5–14 ans", "≥15 ans"]
    lab_year = ["<0.083", "0.083–4.999", "5–14", "≥15"]

    # Remplir seulement si NA (sans problème de longueur)
    df["Tranche_age"] = df["Tranche_age"].fillna(pd.Series(np.select(conds, lab_txt, default=pd.NA), index=df.index))
    df["Tranche_age_en_ans"] = df["Tranche_age_en_ans"].fillna(pd.Series(np.select(conds, lab_year, default=pd.NA), index=df.index))

    return df

# =========================
# SIDEBAR: INPUT
# =========================
st.sidebar.header("Source des données")

# ✅ Choix maladie (pour renommer/standardiser correctement)
disease_key = st.sidebar.selectbox(
    "Maladie / type de line list",
    options=list(DISEASE_SPECS.keys()),
    format_func=lambda k: DISEASE_SPECS.get(k, {}).get("label", k),
    index=0,
)

mode = "Téléverser (upload)"  # Déploiement en ligne : upload uniquement

# --- Upload (line list ou IDSR selon la sélection)
# NOTE: on garde le fonctionnement historique pour les line lists.
#       En mode IDSR, on propose un upload IDSR séparé (2 façons: sidebar OU onglet 9).

# Par défaut: feuille selon la maladie (modifiable)
default_sheet = DISEASE_SPECS.get(disease_key, DISEASE_SPECS["cholera"]).get("default_sheet", "")

if disease_key != "idsr":
    # --- Upload line list (toutes maladies sauf IDSR)
    upl = st.sidebar.file_uploader(
        "📤 Téléverser une line list (xlsx/csv)",
        type=["xlsx", "xls", "csv"],
        key="ll_upload"
    )
    sheet_upl = st.sidebar.text_input("Nom feuille (si Excel upload)", value=default_sheet)

    # En mode line list, on ne propose pas l'upload IDSR ici (il reste disponible dans l'onglet 9)
    idsr_upl_side = None
else:
    st.sidebar.info("Mode **IDSR agrégé (hebdo)** : l'analyse IDSR se fait dans l'onglet **9) IDSR**.")

    # Upload IDSR en sidebar (optionnel) = 1ère façon
    idsr_upl_side = st.sidebar.file_uploader(
        "📤 Téléverser un IDSR agrégé (xlsx)",
        type=["xlsx", "xls"],
        key="idsr_upload_side"
    )

    # En mode IDSR, on ne force pas une line list
    upl = None
    sheet_upl = default_sheet



supp_doublons = st.sidebar.checkbox("Supprimer les doublons (simple)", value=True)
show_maps = st.sidebar.checkbox("Activer cartes (GeoJSON)", value=False)

st.sidebar.header("Période")
use_week_filter = st.sidebar.checkbox("Filtrer sur Num_semaine_epid", value=True)
week_min = st.sidebar.number_input("Semaine min", min_value=1, max_value=53, value=1, step=1)
week_max = st.sidebar.number_input("Semaine max", min_value=1, max_value=53, value=1, step=1)

st.sidebar.header("Seuil timeliness")
seuil_jours = st.sidebar.number_input("Seuil (jours) pour % sous seuil", min_value=0, max_value=30, value=2, step=1)

# ✅ Options visualisations
st.sidebar.header("Options visualisations")
use_custom_viz = st.sidebar.checkbox(
    "Utiliser visualisations custom (dataminsante)",
    value=True,
    help="Ici, les fonctions custom sont intégrées dans ce fichier (autonome)."
)
annot_vals = st.sidebar.checkbox("Afficher annotations (valeurs)", value=False, key="annot_vals")
pas_x = st.sidebar.number_input("Pas X (ticks)", min_value=1, max_value=10, value=1, step=1)
seuil_min_count = st.sidebar.number_input("Seuil minimal (filtrer petits groupes)", min_value=0, max_value=100, value=0, step=1)


# =========================
# LOAD
# =========================
IDSR_MODE = (disease_key == "idsr")

if not IDSR_MODE:
    # Déploiement en ligne : source unique = upload (xlsx/csv)
    if upl is None:
        st.info("📂 Téléverse un fichier (xlsx ou csv) pour démarrer.")

        st.info(
            """
            📊 **Visualisations disponibles :**
            - Situation globale des cas et décès
            - Évolution hebdomadaire des cas 📈
            - Taux de létalité par semaine ⚠️
            - Répartition des cas par province 🗺️
            - Analyse par zone de santé 🏥
            - Tableaux croisés province × semaine 📋
            - Cartographie des cas (si fichiers géographiques disponibles) 🌍
            """
        )

        st.stop()

    # --- Cache session : éviter de recharger/relire le fichier à chaque interaction (ex: changement d’onglet) ---
    try:
        _bytes = upl.getvalue() if hasattr(upl, "getvalue") else None
        _md5 = hashlib.md5(_bytes).hexdigest() if _bytes is not None else None
        _cache_key = (upl.name, getattr(upl, "size", None), _md5, str(sheet_upl).strip() if sheet_upl is not None else "")

        if st.session_state.get("_ll_cache_key") == _cache_key and isinstance(st.session_state.get("_ll_cache_raw"), pd.DataFrame):
            raw = st.session_state["_ll_cache_raw"]
        else:
            if upl.name.lower().endswith(".csv"):
                raw = pd.read_csv(upl)
            else:
                sh = sheet_upl.strip() if isinstance(sheet_upl, str) else ""
                raw = pd.read_excel(upl, sheet_name=sh if sh else 0)

            st.session_state["_ll_cache_key"] = _cache_key
            st.session_state["_ll_cache_raw"] = raw

        files_used = [f"upload:{upl.name}"]

    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier téléversé : {e}")
        st.stop()

    # ✅ 1) Standardisation commune (Rougeole/Choléra/…)
    raw = standardize_ll_by_disease(raw, disease_key)

    # ✅ 2) Standardisation spécifique choléra (les indicateurs/timeliness/etc.)
    df = standardize_df(raw)

    # Filtre semaine
    if use_week_filter and COL_WNUM in df.columns:
        df = df[df[COL_WNUM].between(week_min, week_max)]

    # Doublons (simple)
    if supp_doublons:
        key_cols = [c for c in ["Semaine_epid","Nom_complet", COL_PROV, COL_ZS, COL_AS, COL_SEX, COL_AGE, COL_UNIT, "Profession"] if c in df.columns]
        if key_cols:
            df = df.drop_duplicates(subset=key_cols, keep="first")

    df = df.copy()
    age_col_auto = pick_age_col(df)


    # =========================
    # UI: TITLE + KPIs
    # =========================
    st.title("Incident RDC – Dashboard")
    with st.expander("Fichiers utilisés"):
        st.write(files_used[:200])

    with st.expander("Indicateurs clés de la semaine"):
        kpi_all = compute_indicators(df)
        cases  = kpi_all["n_cases"]
        deaths = kpi_all["n_deaths"]
        cfr    = kpi_all["cfr_pct"] if not np.isnan(kpi_all["cfr_pct"]) else 0.0

        # --- Provinces (définitions) ---
        n_provinces_global = len(EPIDEMIE)  # toutes provinces RDC
        provinces_epid = PROVINCES_EPID if "PROVINCES_EPID" in globals() else [p for p, ok in EPIDEMIE.items() if ok]
        n_provinces_attendues = len(provinces_epid)  # provinces True

        # Provinces trouvées dans les données (df)
        n_provinces = df[COL_PROV].nunique() if (COL_PROV in df.columns and cases) else 0

        # Provinces épidémiques réellement rapportées = intersection (df ∩ provinces True)
        if (COL_PROV in df.columns) and cases:
            n_provinces_epid = df.loc[df[COL_PROV].isin(provinces_epid), COL_PROV].nunique()
        else:
            n_provinces_epid = 0

        # --- % complétude ---
        compl_epidem_pct = (n_provinces_epid / n_provinces_attendues * 100.0) if n_provinces_attendues > 0 else 0.0
        compl_nat_pct    = (n_provinces / n_provinces_global * 100.0) if n_provinces_global > 0 else 0.0

        # plafonner à 100%
        compl_epidem_pct = min(compl_epidem_pct, 100.0)
        compl_nat_pct    = min(compl_nat_pct, 100.0)

        # --- Textes ratio ---
        prov_ratio_epidem = f"{n_provinces_epid} / {n_provinces_attendues}"
        prov_ratio_nat    = f"{n_provinces} / {n_provinces_global}"

        # --- ZS ---
        n_zs = df[COL_ZS].nunique() if (COL_ZS in df.columns and cases) else 0

        # --- KPIs UI ---
        k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
        k1.metric("Cas (lignes)", f"{cases:,}".replace(",", " "))
        k2.metric("Décès", f"{deaths:,}".replace(",", " "))
        k3.metric("CFR (%)", f"{cfr:.2f}" if cases else "0.00")
        k4.metric("Semaine min", str(df[COL_WNUM].min()) if (COL_WNUM in df.columns and cases) else "-")
        k5.metric("Semaine max", str(df[COL_WNUM].max()) if (COL_WNUM in df.columns and cases) else "-")
        k6.metric("Provinces épidémiques", prov_ratio_epidem, f"{compl_epidem_pct:.1f}%")
        k7.metric("Couverture nationale", prov_ratio_nat, f"{compl_nat_pct:.1f}%")
        k8.metric("ZS touchées", f"{n_zs}")


    # =========================
    # FILTERS (UI) - MULTISELECT DÉPENDANTS AVEC "Toutes" PAR DÉFAUT
    # =========================
    st.sidebar.header("Filtres géographiques")

    # ---- Init state ----
    if "prov_sel" not in st.session_state:
        st.session_state["prov_sel"] = ["Toutes"]
    if "zs_sel" not in st.session_state:
        st.session_state["zs_sel"] = ["Toutes"]
    if "as_sel" not in st.session_state:
        st.session_state["as_sel"] = ["Toutes"]
    if "class_sel" not in st.session_state:
        st.session_state["class_sel"] = ["Toutes"]

    # ---- Bouton reset ----
    if st.sidebar.button("Réinitialiser les filtres"):
        st.session_state["prov_sel"] = ["Toutes"]
        st.session_state["zs_sel"] = ["Toutes"]
        st.session_state["as_sel"] = ["Toutes"]
        st.session_state["class_sel"] = ["Toutes"]
        st.rerun()

    df0 = df.copy()  # base (non filtré)

    def normalize_sel(state_key: str, options: list[str]):
        """
        - Garde seulement les valeurs valides
        - Si l'utilisateur a des choix spécifiques -> enlève "Toutes"
        - Si vide -> remet ["Toutes"]
        """
        sel = st.session_state.get(state_key, ["Toutes"])
        sel = [x for x in sel if x in options]

        if any(x != "Toutes" for x in sel):
            sel = [x for x in sel if x != "Toutes"]

        if len(sel) == 0:
            sel = ["Toutes"]

        st.session_state[state_key] = sel
        return sel

    # =========================
    # Province (multiselect)
    # =========================
    df1 = df0.copy()
    if COL_PROV in df0.columns:
        prov_options = ["Toutes"] + sorted([x for x in df0[COL_PROV].dropna().unique().tolist() if x])
        normalize_sel("prov_sel", prov_options)

        prov_sel = st.sidebar.multiselect(
            "Province (notification)",
            options=prov_options,
            default=st.session_state["prov_sel"],
            key="prov_sel",
        )

        if prov_sel and ("Toutes" not in prov_sel):
            df1 = df1[df1[COL_PROV].isin(prov_sel)]

    # =========================
    # Zone de santé (multiselect, dépend de Province)
    # =========================
    df2 = df1.copy()
    if COL_ZS in df1.columns:
        zs_options = ["Toutes"] + sorted([x for x in df1[COL_ZS].dropna().unique().tolist() if x])
        normalize_sel("zs_sel", zs_options)

        zs_sel = st.sidebar.multiselect(
            "Zone de santé (notification)",
            options=zs_options,
            default=st.session_state["zs_sel"],
            key="zs_sel",
        )

        if zs_sel and ("Toutes" not in zs_sel):
            df2 = df2[df2[COL_ZS].isin(zs_sel)]

    # =========================
    # Aire de santé (multiselect, dépend de Province + ZS)
    # =========================
    df3 = df2.copy()
    if COL_AS in df2.columns:
        as_options = ["Toutes"] + sorted([x for x in df2[COL_AS].dropna().unique().tolist() if x])
        normalize_sel("as_sel", as_options)

        as_sel = st.sidebar.multiselect(
            "Aire de santé (notification)",
            options=as_options,
            default=st.session_state["as_sel"],
            key="as_sel",
        )

        if as_sel and ("Toutes" not in as_sel):
            df3 = df3[df3[COL_AS].isin(as_sel)]

    # df_f = dataframe filtré géographiquement
    df_f = df3

    # =========================
    # Classification finale (multiselect, "Toutes" par défaut, dépend de df_f)
    # =========================
    if COL_CLASS in df_f.columns:
        class_values = sorted([x for x in df_f[COL_CLASS].dropna().unique().tolist() if x])
        class_options = ["Toutes"] + class_values
        normalize_sel("class_sel", class_options)

        class_sel = st.sidebar.multiselect(
            "Classification finale",
            options=class_options,
            default=st.session_state["class_sel"],
            key="class_sel",
        )

        if class_sel and ("Toutes" not in class_sel):
            df_f = df_f[df_f[COL_CLASS].isin(class_sel)]

    age_col = pick_age_col(df_f)
    # =========================
    # KPIs (après filtres géographiques)
    # =========================
    with st.expander("Indicateurs clés de la semaine correspondant aux filtres géographiques sélectionnés",expanded=True):
        st.info("Ces indicateurs reflètent uniquement les données correspondant aux filtres géographiques sélectionnés.")

        kpi_f = compute_indicators(df_f)
        cases  = kpi_f["n_cases"]
        deaths = kpi_f["n_deaths"]
        cfr    = kpi_f["cfr_pct"] if not np.isnan(kpi_f["cfr_pct"]) else 0.0

        # --- Provinces (définitions) ---
        n_provinces_global = len(EPIDEMIE)  # toutes les provinces (RDC)

        # si PROVINCES_EPID existe déjà, on l'utilise, sinon on le reconstruit
        provinces_epid = PROVINCES_EPID if "PROVINCES_EPID" in globals() else [p for p, ok in EPIDEMIE.items() if ok]
        n_provinces_attendues = len(provinces_epid)  # provinces True

        n_provinces_f = df_f[COL_PROV].nunique() if (COL_PROV in df_f.columns and cases) else 0

        # --- % complétude ---
        compl_epidem_pct = (n_provinces_f / n_provinces_attendues * 100.0) if n_provinces_attendues > 0 else 0.0
        compl_nat_pct    = (n_provinces_f / n_provinces_global   * 100.0) if n_provinces_global > 0 else 0.0

        # plafonner à 100% (optionnel)
        compl_epidem_pct = min(compl_epidem_pct, 100.0)
        compl_nat_pct    = min(compl_nat_pct, 100.0)

        # --- Textes ratio ---
        prov_ratio_epidem = f"{n_provinces_f} / {n_provinces_attendues}"
        prov_ratio_nat    = f"{n_provinces_f} / {n_provinces_global}"

        # --- ZS ---
        n_zs_f = df_f[COL_ZS].nunique() if (COL_ZS in df_f.columns and cases) else 0

        # --- KPIs UI ---
        k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
        k1.metric("Cas (lignes)", f"{cases:,}".replace(",", " "))
        k2.metric("Décès", f"{deaths:,}".replace(",", " "))
        k3.metric("CFR (%)", f"{cfr:.2f}" if cases else "0.00")
        k4.metric("Semaine min", str(df_f[COL_WNUM].min()) if (COL_WNUM in df_f.columns and cases) else "-")
        k5.metric("Semaine max", str(df_f[COL_WNUM].max()) if (COL_WNUM in df_f.columns and cases) else "-")
        k6.metric("Provinces épidémiques", prov_ratio_epidem, f"{compl_epidem_pct:.1f}%")
        k7.metric("Couverture nationale", prov_ratio_nat, f"{compl_nat_pct:.1f}%")
        k8.metric("ZS touchées", f"{n_zs_f}")



else:
    # Mode IDSR: on ne charge pas de line list ici. Les analyses IDSR sont dans l'onglet 9.
    raw = pd.DataFrame()
    df = pd.DataFrame()
    df_f = pd.DataFrame()
    files_used = []
    st.title("Incident RDC – Dashboard")
    st.info("🧭 Mode **IDSR agrégé (hebdo)** : va dans l'onglet **9) IDSR** pour téléverser et analyser le fichier.")

# =========================
# TABS
# =========================
def tab_help(title: str, md: str, expanded: bool = False):
    with st.expander(f"ℹ️ {title}", expanded=expanded):
        st.markdown(md)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "1) Évolution", "2) Taux & CFR", "3) Délais (timeliness)", "4) Démographie",
    "5) Complétude", "6) Données & Export", "7) Qualité & Alertes", "8) SITREP",
    "9) IDSR"
])


# =========================
# TAB 1: EVOLUTION
# =========================
with tab1:
    if IDSR_MODE:
        st.info("🧭 Mode **IDSR agrégé (hebdo)** : les analyses line list ne sont pas actives. Va dans l'onglet **9) IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Suivre l’évolution des cas et des décès dans le temps.
        
            **📖 Interprétation**
            - Une hausse progressive des cas peut indiquer une propagation ou un élargissement du dépistage.
            - Un pic soudain peut correspondre à un foyer actif ou à un rattrapage de notification.
            - La courbe **CFR (%)** aide à suivre l’évolution de la létalité dans le temps.
        
            **⚠️ Points d’attention**
            - Une hausse peut venir d’une meilleure complétude de rapportage, pas forcément d’une vraie augmentation.
            - Toujours lire la tendance sur 3–4 semaines, pas une seule semaine.
            """,
            expanded=False
        )
        
        st.subheader("Évolution hebdomadaire")
        
        # --- Graphes existants (YW) ---
        if "YW" in df_f.columns and df_f["YW"].notna().any():
            weekly = df_f.groupby("YW", as_index=False).agg(
                Cas=("YW", "count"),
                Deces=("is_death", "sum")
            )
            weekly["CFR_%"] = np.where(weekly["Cas"] > 0, weekly["Deces"] / weekly["Cas"] * 100, np.nan)
        
            cA, cB = st.columns([2, 1])
            with cA:
                fig = px.line(weekly, x="YW", y="Cas", markers=True, title="Cas par semaine (YW)")
                fig = apply_plotly_value_annotations(fig, annot_vals)
                st.plotly_chart(fig, width="stretch")
            with cB:
                fig2 = px.bar(weekly, x="YW", y="Deces", title="Décès par semaine")
                fig2 = apply_plotly_value_annotations(fig2, annot_vals)
                st.plotly_chart(fig2, width="stretch")
        
            fig3 = px.line(weekly, x="YW", y="CFR_%", markers=True, title="CFR (%) par semaine")
            fig3 = apply_plotly_value_annotations(fig3, annot_vals)
            st.plotly_chart(fig3, width="stretch")
        else:
            st.info("Pas de clé YW disponible (Annee_epid / Num_semaine_epid manquants).")
        
        st.divider()
        st.subheader("Visualisations")
        
        c1, c2 = st.columns(2)
        with c1:
            x_axis_choice = st.selectbox("Axe X", [COL_WNUM, "YW", COL_PROV], index=0)
        with c2:
            hue_choice = st.selectbox("Couleur (hue)", [c for c in [age_col, COL_SEX, COL_PROV] if c], index=0 if age_col else 0)
        
        # ✅ Histogramme empilé (count des cas)
        if use_custom_viz and HAS_CUSTOM_VIZ and x_axis_choice in df_f.columns and hue_choice in df_f.columns:
            fig = plot_histogramme_groupe_interactif_empile(
                df=df_f,
                x_col=x_axis_choice,
                x_titre=x_axis_choice,
                hue_col=hue_choice,
                y_titre="Nombre de cas",
                titre=f"Histogramme empilé: {x_axis_choice} x {hue_choice}",
                rotation=45,
                annot=annot_vals,
                pas_x=int(pas_x) if x_axis_choice in [COL_WNUM, "YW"] else None,
                bargap=0,
                bargroupgap=0.05,
                taille_fig=(1500, 600),
                x_trier=True,
                ordre="desc",
                y_col=None
            )
        
            st_plot(fig, key="stacked_hist")
        else:
            st.info("Activer 'Utiliser visualisations custom' et vérifier colonnes disponibles.")
        
        # ✅ Facettes par province : cas par semaine par province
        st.subheader("Cas par semaine, facetté par province")
        if use_custom_viz and HAS_CUSTOM_VIZ and COL_PROV in df_f.columns and COL_WNUM in df_f.columns:
            df_fac = df_f.copy()
            fig = graphique_barres_facette(
                df=df_fac,
                x_col=COL_WNUM,
                x_titre="Semaine épidémiologique",
                y_col=COL_WNUM,  # non numérique => comptage (occurrences)
                y_titre="Nombre de cas",
                facette_col=COL_PROV,
                titre="Répartition hebdomadaire des cas par province (facettes)",
                taille_fig=(1500, 700),
                rotation=45,
                couleurs_personnalisees="black",
                bargap=0,
                bargroupgap=0.02,
                annot=annot_vals,
                pas_x=int(pas_x),
                return_fig=True,
                encadrer_facettes=True,
                couleur_contour_facette="#777772"
            )
            st_plot(fig, key="facet_week_prov")
        else:
            st.info("Facettes: nécessite Province_notification + Num_semaine_epid + visualisations custom.")
        
        # ✅ Courbes multi-catégories
        st.subheader("Courbe multi-séries")
        if use_custom_viz and HAS_CUSTOM_VIZ and COL_WNUM in df_f.columns and COL_PROV in df_f.columns:
            fig = plot_courbe_par_categories_plotly(
                df=df_f,
                colonne_x=COL_WNUM,
                colonne_y=COL_PROV,
                titre="Évolution des cas par semaine et province",
                rotation=45,
                annot=annot_vals,
                pas_x=int(pas_x),
                taille_fig=(1500, 600)
            )
            st_plot(fig, key="line_week_prov")
        else:
            st.info("Courbes multi-séries: nécessite Province + Num_semaine_epid + visualisations custom.")
        
    # =========================
    # TAB 2: Taux & CFR
    # =========================
with tab2:
    if IDSR_MODE:
        st.info("🧭 Mode **IDSR agrégé (hebdo)** : les analyses line list ne sont pas actives. Va dans l'onglet **9) IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Évaluer la performance diagnostic/prise en charge et la létalité.
        
            **📖 Indicateurs**
            - **Taux prélèvement (%)** : disponibilité/collecte des échantillons.
            - **Taux TDR réalisé (%)** : capacité de test.
            - **Positivité TDR (%)** : circulation probable de Vibrio cholerae (attention au biais de test).
            - **Taux hospitalisation (%)** : gravité ou stratégie de prise en charge.
            - **CFR (%)** : létalité observée.
        
            **⚠️ Points d’attention**
            - CFR élevé peut refléter : retard de consultation, sous-détection des cas bénins, ou qualité des soins.
            - Positivité élevée + faible couverture TDR = confirmation insuffisante.
            """,
            expanded=False
        )
        
        st.subheader("Taux (qualité / process) et létalité")
        
        # ===== KPI GLOBAUX (définitions harmonisées) =====
        kpi = compute_indicators(df_f)
        
        a0, a1, a2, a3, a4, a5, a6 = st.columns(7)
        
        a0.metric(
            "Cas (n)",
            f"{kpi['n_cases']:,}".replace(",", " "),
            help="Nombre total de cas après application des filtres."
        )
        
        a1.metric(
            "Taux prélèvement (%)",
            "-" if np.isnan(kpi["prelev_pct"]) else f"{kpi['prelev_pct']:.1f}",
            help=(
                "Prélèvement=Oui / Tous cas filtrés.\n"
                f"n={kpi.get('prelev_num', 0)}/{kpi.get('prelev_den', kpi.get('n_cases', 0))}"
            )
        )
        
        a2.metric(
            "Couverture TDR (%)",
            "-" if np.isnan(kpi.get("tdr_coverage_pct", np.nan)) else f"{kpi['tdr_coverage_pct']:.1f}",
            help=(
                "TDR réalisés (Oui) / Tous cas filtrés.\n"
                f"n={kpi.get('tdr_coverage_num', 0)}/{kpi.get('tdr_coverage_den', kpi.get('n_cases', 0))}"
            )
        )
        
        a3.metric(
            "Positivité TDR (%)",
            "-" if np.isnan(kpi["pos_pct"]) else f"{kpi['pos_pct']:.1f}",
            help=(
                "Positifs / (Positifs + Négatifs) parmi TDR interprétables.\n"
                "Interprétable = TDR=Oui & résultat valide (Pos/Nég).\n"
                f"n={kpi.get('pos_num', 0)}/{kpi.get('pos_den', 0)}"
            )
        )
        
        a4.metric(
            "Taux hospitalisation (%)",
            "-" if np.isnan(kpi["hosp_pct"]) else f"{kpi['hosp_pct']:.1f}",
            help=(
                "Hospitalisation=Oui / Tous cas filtrés.\n"
                f"n={kpi.get('hosp_num', 0)}/{kpi.get('hosp_den', kpi.get('n_cases', 0))}"
            )
        )
        
        a5.metric(
            "CFR global (%)",
            "-" if np.isnan(kpi["cfr_pct"]) else f"{kpi['cfr_pct']:.2f}",
            help=f"Décès / Tous cas filtrés. n={kpi.get('n_deaths', 0)}/{kpi.get('n_cases', 0)}"
        )
        
        a6.metric(
            "% TDR invalides",
            "-" if np.isnan(kpi.get("invalid_pct", np.nan)) else f"{kpi['invalid_pct']:.1f}",
            help=(
                "Invalides (ex: INBA/bande absente) / TDR réalisés (TDR=Oui).\n"
                f"n={kpi.get('invalid_num', 0)}/{kpi.get('invalid_den', 0)}"
            )
        )
        
        
        
        
        # Degré de déshydratation (liste catégorielle)
        with st.expander("📋 Degré de déshydratation (répartition)", expanded=False):
            st_dataframe_safe(kpi["dehy_tbl"])
        
        st.divider()
        
        group_col = st.selectbox(
            "Grouper par",
            [c for c in [COL_PROV, COL_ZS, "YW", COL_WNUM] if c in df_f.columns],
            index=0
        )
        
        if group_col not in df_f.columns:
            st.warning(f"Colonne {group_col} absente dans les données filtrées.")
        else:
            g_ind = compute_group_indicators(df_f, group_col)
        
            st.markdown("**Table d'indicateurs (définitions cohérentes)**")
            st_dataframe_safe(g_ind)
        
            ind_to_plot = st.selectbox(
                "Indicateur à visualiser",
                options=[
                    "Cas",
                    "Décès",
                    "CFR_%",
                    "Prélèvement_%",
                    "Hospitalisation_%",
                    "TDR_réalisé_%",
                    "Positivité_TDR_%"
                ],
                index=3
            )
        
            fig = px.bar(g_ind, x=group_col, y=ind_to_plot, title=f"{ind_to_plot} par {group_col}")
            fig.update_layout(xaxis_tickangle=-45)
            fig = apply_plotly_value_annotations(fig, annot_vals)
            st.plotly_chart(fig, width="stretch")
        
        st.divider()
        st.subheader("Évolution multi-indicateurs (cas + courbes)")
        
        if use_custom_viz and HAS_CUSTOM_VIZ:
            df_tmp = df_f.copy()
        
            if "Femme_enceinte" in df_tmp.columns:
                df_tmp["Femme_enceinte"] = df_tmp["Femme_enceinte"].astype("string").str.lower()
            if COL_HOSP in df_tmp.columns:
                df_tmp[COL_HOSP] = df_tmp[COL_HOSP].astype("string").str.lower()
        
            curves = []
            valeurs_pos = {}
        
            if "Femme_enceinte" in df_tmp.columns:
                curves.append("Femme_enceinte")
                valeurs_pos["Femme_enceinte"] = "oui"
        
            if COL_HOSP in df_tmp.columns:
                curves.append(COL_HOSP)
                valeurs_pos[COL_HOSP] = "oui"
        
            if COL_WNUM in df_tmp.columns and curves:
                fig = plot_evolution_multi_auto(
                    df=df_tmp,
                    col_x=COL_WNUM,
                    courbe_col=curves,
                    valeurs_courbe_col=valeurs_pos,
                    titre="Cas (barres) + femmes enceintes / hospitalisation (courbes)",
                    annot_x=False,
                    annot_y=annot_vals,
                    rotation=0,
                    seuil_min=0,
                    taille_fig=(1500, 600),
                    bargap=0,
                    bargroupgap=0.0
                )
                st_plot(fig, key="multi_auto")
            else:
                st.info("Multi-indicateurs: nécessite Num_semaine_epid + (Femme_enceinte / Hospitalisation).")
        else:
            st.info("Active les visualisations custom pour afficher ce bloc.")
        
    # =========================
    # TAB 3: DELAIS
    # =========================
with tab3:
    if IDSR_MODE:
        st.info("🧭 Mode **IDSR agrégé (hebdo)** : les analyses line list ne sont pas actives. Va dans l'onglet **9) IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            f"""
            **🎯 Objectif** : Mesurer la rapidité de détection et d’accès aux soins.
        
            **📖 Indicateurs**
            - Délai **début maladie → admission**
            - Délai **début maladie → prélèvement**
            - **% ≤ {seuil_jours} jours** : proportion de cas pris en charge rapidement.
        
            **⚠️ Points d’attention**
            - Des délais longs augmentent le risque de transmission communautaire.
            - Des délais négatifs ou extrêmes = erreurs de saisie ou dates incorrectes.
            """,
            expanded=False
        )
        
        st.subheader("Délais (timeliness)")
        
        delais_cols = [c for c in ["delai_onset_to_adm", "delai_onset_to_prel"] if c in df_f.columns]
        
        if not delais_cols:
            st.info("Colonnes de délais indisponibles (dates manquantes).")
        else:
            df_del = df_f.copy()
            for c in delais_cols:
                df_del.loc[df_del[c] < 0, c] = np.nan
        
            st.markdown("**Distribution des délais**")
            if use_custom_viz and HAS_CUSTOM_VIZ:
                fig = plot_boxplot_delais_plotly(
                    df=df_del,
                    colonnes_delais=delais_cols,
                    col_groupe=COL_PROV if COL_PROV in df_del.columns else None,
                    titre="Distribution des délais (jours)",
                    taille_fig=(1500, 600),
                    rotation=45
                )
                st_plot(fig, key="boxplot_delais_custom")
            else:
                long = df_del.melt(value_vars=delais_cols, var_name="Type_delai", value_name="Jours").dropna()
                fig = px.box(long, x="Type_delai", y="Jours", points="outliers", title="Boxplot des délais (global)")
                fig = apply_plotly_value_annotations(fig, annot_vals)
                st.plotly_chart(fig, width="stretch")
        
            st.divider()
        
            st.markdown(f"**% sous seuil (≤ {seuil_jours} jours)**")
            c1, c2= st.columns(2)
            with c1:
                p1, n1 = pct_under_threshold(df_del.get("delai_onset_to_adm"), seuil_jours)
                st.metric("Admission ≤ seuil (%)", "-" if np.isnan(p1) else f"{p1:.1f}", help=f"n = {n1}")
            with c2:
                p2, n2 = pct_under_threshold(df_del.get("delai_onset_to_prel"), seuil_jours)
                st.metric("Prélèvement ≤ seuil (%)", "-" if np.isnan(p2) else f"{p2:.1f}", help=f"n = {n2}")        
        
            if use_custom_viz and HAS_CUSTOM_VIZ and COL_PROV in df_del.columns:
                st.subheader("Timeliness par province (% sous seuil)")
        
                rows = []
                for prov, sub in df_del.groupby(COL_PROV):
                    s = pd.to_numeric(sub.get("delai_onset_to_adm"), errors="coerce").dropna()
                    n = int(len(s))
                    sous = int((s <= seuil_jours).sum()) if n else 0
                    pct = (sous / n * 100) if n else np.nan
                    rows.append([prov, n, sous, pct])
        
                df_resume = pd.DataFrame(rows, columns=[COL_PROV, "n", "sous_seuil", "pct_sous_seuil_%"])
        
                fig = plot_barres_pct_sous_seuil(
                    df_resume_groupe=df_resume,
                    col_groupe=COL_PROV,
                    col_n="n",
                    col_sous_seuil="sous_seuil",
                    col_pct="pct_sous_seuil_%",
                    titre=f"% admission ≤ {seuil_jours} jours par province",
                    seuil=seuil_jours,
                    taille_fig=(1500, 600),
                    rotation=45,
                    annot=True,
                    tri_desc=True
                )
                st_plot(fig, key="timeliness_pct_prov")
        
                with st.expander("Table timeliness (résumé)"):
                    st.dataframe(df_resume.sort_values("pct_sous_seuil_%", ascending=False), width="stretch")
        
    # =========================
    # TAB 4: Démographie
    # =========================
with tab4:
    if IDSR_MODE:
        st.info("🧭 Mode **IDSR agrégé (hebdo)** : les analyses line list ne sont pas actives. Va dans l'onglet **9) IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Identifier les groupes les plus touchés.
        
            **📖 Interprétation**
            - Répartition **sexe** : différences d’exposition ou d’accès aux soins.
            - Répartition **âge** : identifie les groupes vulnérables/à risque.
            - **Pyramide âge/sexe** : profil de transmission (domicile, école, activités, etc.).
        
            **⚠️ Points d’attention**
            - Vérifier la complétude de l’âge et du sexe : beaucoup de “Inconnu” biaise la lecture.
            """,
            expanded=False
        )
        
        st.subheader("Démographie")
        
        cA, cB = st.columns(2)
        
        with cA:
            if COL_SEX in df_f.columns:
                sex_counts = df_f[COL_SEX].fillna("Inconnu").astype(str).str.strip().value_counts().reset_index()
                sex_counts.columns = [COL_SEX, "Cas"]
                fig = px.bar(sex_counts, x=COL_SEX, y="Cas", title="Cas par sexe")
                fig = apply_plotly_value_annotations(fig, annot_vals)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Colonne Sexe absente.")
        
        with cB:
            if age_col:
                age_counts = df_f[age_col].fillna("Inconnu").astype(str).str.strip().value_counts().reset_index()
                age_counts.columns = [age_col, "Cas"]
                fig = px.bar(age_counts, x=age_col, y="Cas", title=f"Cas par {age_col}")
                fig.update_layout(xaxis_tickangle=-45)
                fig = apply_plotly_value_annotations(fig, annot_vals)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Colonnes tranche âge absentes (Tranche_age_en_ans / Tranche_age).")
        
        st.divider()
        
        st.subheader("Proportion des cas (camembert)")
        if use_custom_viz and HAS_CUSTOM_VIZ and age_col:
            fig = plot_camembert_interactif(
                df=df_f,
                colonne=[COL_UNIT, age_col] if COL_UNIT in df_f.columns else [age_col],
                titre="Proportion des cas par tranche d'âge",
                seuil_min=int(seuil_min_count),
                afficher_legende=False,
                annot=True,
                taille_fig=(700, 500)
            )
            st_plot(fig, key="pie_age")
        else:
            st.info("Camembert: nécessite tranche d’âge + visualisations custom.")
        
        st.divider()
        
        st.subheader("Pyramide âge / sexe")
        if use_custom_viz and HAS_CUSTOM_VIZ and age_col and COL_SEX in df_f.columns:
            fig = plot_pyramide_symetrique(
                df=df_f,
                col_categorie=age_col,
                col_groupe=COL_SEX,
                valeurs_neg=["Masculin", "Homme", "M"],
                titre="Pyramide des âges (Masculin à gauche, Féminin à droite)",
                seuil_min=int(seuil_min_count),
                croissant=False,
                afficher_signe_negatif_dans_label=False
            )
            st_plot(fig, key="pyr_global")
        else:
            st.info("Pyramide: nécessite Tranche âge + Sexe + visualisations custom.")
        
        st.subheader("Pyramides par province (facettes)")
        if use_custom_viz and HAS_CUSTOM_VIZ and age_col and COL_SEX in df_f.columns and COL_PROV in df_f.columns:
            fig = graphique_pyramide_age(
                df=df_f,
                col_tranche=age_col,
                col_sexe=COL_SEX,
                col_valeur=COL_UNIT if COL_UNIT in df_f.columns else COL_SEX,  # si non numérique => comptage
                valeurs_neg=["Masculin", "Homme", "M"],
                titre="Pyramides âge/sex par province",
                seuil_min=10,
                croissant=False,
                afficher_signe_negatif_dans_label=False,
                facette_col=COL_PROV,
                annot=annot_vals,
                taille_fig=(1500, 900),
                return_fig=True,
                couleur_contour_facette="#777772"
            )
            st_plot(fig, key="pyr_fac_prov")
        else:
            st.info("Pyramides facettées: nécessite Province + Sexe + tranche âge + visualisations custom.")
        
    # =========================
    # TAB 5: Complétude
    # =========================
with tab5:
    if IDSR_MODE:
        st.info("🧭 Mode **IDSR agrégé (hebdo)** : les analyses line list ne sont pas actives. Va dans l'onglet **9) IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Vérifier si les provinces attendues notifient (complétude géographique).
        
            **📖 Interprétation**
            - **Manquantes** : silence épidémiologique ou problème de remontée/rapportage.
            - Le tableau croisé aide à repérer les zones/provinces dominantes ou sous-notifiantes.
        
            **⚠️ Points d’attention**
            - Une province silencieuse pendant une épidémie = signal d’alerte système à investiguer.
            """,
            expanded=False
        )
        
        st.subheader("Complétude (provinces attendues vs reçues)")
        
        if COL_PROV not in df_f.columns:
            st.info("Province_notification absente.")
        else:
            if COL_WNUM in df_f.columns and df_f[COL_WNUM].notna().any():
                last_w = int(df_f[COL_WNUM].max())
                present = sorted(df_f.loc[df_f[COL_WNUM] == last_w, COL_PROV].dropna().unique().tolist())
                st.caption(f"Calcul sur la semaine max filtrée: SE{last_w:02d}")
            else:
                present = sorted(df_f[COL_PROV].dropna().unique().tolist())
                st.caption("Calcul sur l’ensemble filtré (pas de Num_semaine_epid exploitable).")
        
            missing = [p for p in PROVINCES_EPID if p not in present]
            nb_att = len(PROVINCES_EPID)
            nb_rec = len([p for p in PROVINCES_EPID if p in present])
            compl = (nb_rec / nb_att * 100) if nb_att > 0 else np.nan
        
            c1, c2, c3 = st.columns(3)
            c1.metric("Provinces attendues", str(nb_att))
            c2.metric("Provinces trouvées", str(nb_rec))
            c3.metric("Complétude (%)", f"{compl:.1f}")
            if missing:
                st.warning("Manquantes: " + ", ".join(missing))
        
            with st.expander("Tableau provinces attendues vs reçues"):
                df_comp = pd.DataFrame({
                    "Province attendue": PROVINCES_EPID,
                    "Présente": [p in present for p in PROVINCES_EPID],
                    "Manquante": [p if p in missing else "" for p in PROVINCES_EPID],
                })
                st_dataframe_safe(df_comp)
        
            with st.expander("Cas par province (complétude / volume)", expanded=True):
                prov_counts = df_f[COL_PROV].fillna("Inconnu").value_counts().reset_index()
                prov_counts.columns = [COL_PROV, "Cas"]
                figp = px.bar(prov_counts, x=COL_PROV, y="Cas", title="Volume des cas par province (filtrés)")
                figp.update_layout(xaxis_tickangle=-45)
                figp = apply_plotly_value_annotations(figp, annot_vals)
                st.plotly_chart(figp, width="stretch")
        
            # TCD
            with st.expander("Tableau croisé dynamique – occurrences", expanded=False):
                # --- Scope: même logique que ton calcul "semaine max filtrée"
                scope_last_week = st.checkbox(
                    "Calculer uniquement sur la semaine max filtrée (même scope que la complétude)",
                    value=True,
                    key="ct_scope_last_week"
                )
                df_scope = df_f.copy()
                if scope_last_week and (COL_WNUM in df_scope.columns) and df_scope[COL_WNUM].notna().any():
                    last_w = int(df_scope[COL_WNUM].max())
                    df_scope = df_scope.loc[df_scope[COL_WNUM] == last_w].copy()
                    st.caption(f"Scope: SE{last_w:02d}")
                else:
                    st.caption("Scope: ensemble filtré")
        
                # --- Outils UX (global)
                cUX1, cUX2, cUX3, cUX4 = st.columns([1.1, 1.1, 1.1, 0.9])
                with cUX1:
                    show_pct = st.checkbox("Afficher %", value=False, key="ct_show_pct")
                with cUX2:
                    show_bar = st.checkbox("Barres (datatable)", value=True, key="ct_show_bar")
                with cUX3:
                    tbl_height = st.number_input("Hauteur tableau", min_value=250, max_value=1200, value=520, step=50, key="ct_tbl_height")
                with cUX4:
                    do_download = st.checkbox("Activer export", value=True, key="ct_export_on")
        
                # --- Choix du niveau d’agrégation (on maintient les 3 options)
                level = st.radio(
                    "Niveau d’agrégation",
                    ["Province (occurrences)", "Province + Zone de santé", "Tableau croisé Province × Zone"],
                    index=0,
                    horizontal=True,
                    key="ct_level"
                )
        
                # Helper: affiche tableau + option export
                def _show_table(df_to_show: pd.DataFrame, name: str):
                    st.dataframe(
                        df_to_show, width='stretch', height=int(tbl_height),
                        hide_index=False,
                        column_config=None
                    )
                    if do_download:
                        csv = df_to_show.to_csv(index=True).encode("utf-8")
                        st.download_button(
                            f"Télécharger {name} (CSV)",
                            data=csv,
                            file_name=f"{name}.csv".replace(" ", "_").lower(),
                            mime="text/csv",
                            key=f"dl_{name}"
                        )
        
                # 1) Province (occurrences)
                if level == "Province (occurrences)":
                    if COL_PROV not in df_scope.columns:
                        st.info("Colonne Province_notification absente.")
                    else:
                        piv = (
                            df_scope.assign(_prov=df_scope[COL_PROV].fillna("Inconnu"))
                            .groupby("_prov", dropna=False)
                            .size()
                            .reset_index(name="Occurrences")
                            .sort_values("Occurrences", ascending=False)
                            .rename(columns={"_prov": COL_PROV})
                        )
        
                        if show_pct:
                            total = int(piv["Occurrences"].sum()) if len(piv) else 0
                            piv["%"] = (piv["Occurrences"] / total * 100).round(1) if total > 0 else 0.0
        
                        if show_bar:
                            st.dataframe(
                                piv, width='stretch', height=int(tbl_height),
                                column_config={
                                    "Occurrences": st.column_config.ProgressColumn(
                                        "Occurrences",
                                        help="Occurrences (barres)",
                                        format="%d",
                                        min_value=0,
                                        max_value=int(piv["Occurrences"].max()) if len(piv) else 1,
                                    )
                                },
                            )
                            if do_download:
                                csv = df_to_csv_bytes(piv)
                                st.download_button(
                                    "Télécharger province_occurrences (CSV)",
                                    data=csv,
                                    file_name="province_occurrences.csv",
                                    mime="text/csv",
                                    key="dl_prov_occ"
                                )
                        else:
                            _show_table(piv, "province_occurrences")
        
                    with st.expander("Graphique (top provinces)"):
                        topk = st.number_input("Top K", min_value=5, max_value=30, value=15, step=1, key="ct_topk_prov")
                        figp = px.bar(piv.head(int(topk)), x=COL_PROV, y="Occurrences", title="Top provinces – occurrences")
                        figp.update_layout(xaxis_tickangle=-45)
                        figp = apply_plotly_value_annotations(figp, annot_vals)
                        st.plotly_chart(figp, width="stretch")
        
                # 2) Province + Zone de santé
                elif level == "Province + Zone de santé":
                    if (COL_PROV not in df_scope.columns) or (COL_ZS not in df_scope.columns):
                        st.info("Colonnes Province_notification / Zone_de_sante_notification absentes.")
                    else:
                        colA, colB, colC = st.columns([1.2, 1.2, 1.6])
                        with colA:
                            view_mode = st.radio(
                                "Vue",
                                ["Top N (table longue)", "Déroulable Province → Zone"],
                                index=1,
                                horizontal=True,
                                key="ct_view_mode_pz"
                            )
                        with colB:
                            limit_zones = st.checkbox("Limiter zones (perf)", value=True, key="ct_limit_zones_pz")
                        with colC:
                            top_z = st.number_input("Top zones (si limitation)", min_value=10, max_value=2000, value=250, step=25, key="ct_top_z_pz")
        
                        df_scope2 = df_scope.copy()
                        if limit_zones:
                            zones_top = (
                                df_scope2[COL_ZS].fillna("Inconnu")
                                .value_counts()
                                .head(int(top_z))
                                .index.tolist()
                            )
                            df_scope2 = df_scope2[df_scope2[COL_ZS].fillna("Inconnu").isin(zones_top)].copy()
        
                        piv = (
                            df_scope2.assign(
                                _prov=df_scope2[COL_PROV].fillna("Inconnu"),
                                _zs=df_scope2[COL_ZS].fillna("Inconnu"),
                            )
                            .groupby(["_prov", "_zs"], dropna=False)
                            .size()
                            .reset_index(name="Occurrences")
                            .sort_values("Occurrences", ascending=False)
                            .rename(columns={"_prov": COL_PROV, "_zs": COL_ZS})
                        )
        
                        if show_pct:
                            tot_prov = piv.groupby(COL_PROV, as_index=False)["Occurrences"].sum().rename(columns={"Occurrences": "Total_province"})
                            piv = piv.merge(tot_prov, on=COL_PROV, how="left")
                            piv["%_dans_province"] = (piv["Occurrences"] / piv["Total_province"] * 100).round(1)
                            piv = piv.drop(columns=["Total_province"])
        
                        tot_prov = (
                            piv.groupby(COL_PROV, as_index=False)["Occurrences"].sum()
                            .sort_values("Occurrences", ascending=False)
                        )
        
                        if view_mode == "Top N (table longue)":
                            top_n = st.number_input("Afficher Top N lignes", min_value=10, max_value=20000, value=500, step=50, key="ct_topn_long")
                            df_show = piv.head(int(top_n)).copy()
        
                            if show_bar:
                                st.dataframe(
                                    df_show, width='stretch', height=int(tbl_height),
                                    column_config={
                                        "Occurrences": st.column_config.ProgressColumn(
                                            "Occurrences",
                                            format="%d",
                                            min_value=0,
                                            max_value=int(piv["Occurrences"].max()) if len(piv) else 1,
                                        )
                                    },
                                )
                            else:
                                _show_table(df_show, "province_zone_topN")
        
                        else:
                            tcd = (
                                piv.set_index([COL_PROV, COL_ZS])[["Occurrences"]]
                                .sort_values("Occurrences", ascending=False)
                            )
                            tcd = tcd.reindex(tot_prov[COL_PROV].tolist(), level=0)
        
                            st.caption("Clique sur les triangles à gauche pour dérouler/replier Province → Zone.")
                            st.dataframe(tcd, width='stretch', height=int(tbl_height))
        
                            if do_download:
                                csv = tcd.reset_index().to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Télécharger province_zone_deroulable (CSV)",
                                    data=csv,
                                    file_name="province_zone_deroulable.csv",
                                    mime="text/csv",
                                    key="dl_pz_deroulable"
                                )
        
                        with st.expander("Totaux par province (somme des zones)"):
                            if show_bar:
                                st.dataframe(
                                    tot_prov, width='stretch', height=450,
                                    column_config={
                                        "Occurrences": st.column_config.ProgressColumn(
                                            "Occurrences",
                                            format="%d",
                                            min_value=0,
                                            max_value=int(tot_prov["Occurrences"].max()) if len(tot_prov) else 1,
                                        )
                                    },
                                )
                            else:
                                st_dataframe_safe(tot_prov)
        
                        with st.expander("Graphique (top provinces)"):
                            topk = st.number_input("Top K", min_value=5, max_value=30, value=15, step=1, key="ct_topk_pz")
                            figp = px.bar(tot_prov.head(int(topk)), x=COL_PROV, y="Occurrences", title="Top provinces – occurrences (scope)")
                            figp.update_layout(xaxis_tickangle=-45)
                            figp = apply_plotly_value_annotations(figp, annot_vals)
                            st.plotly_chart(figp, width="stretch")
        
                # 3) Tableau croisé Province × Zone
                else:
                    if (COL_PROV not in df_scope.columns) or (COL_ZS not in df_scope.columns):
                        st.info("Colonnes Province_notification / Zone_de_sante_notification absentes.")
                    else:
                        cA, cB, cC = st.columns([1.1, 1.3, 1.6])
                        with cA:
                            limit_zones = st.checkbox("Limiter aux zones les plus fréquentes", value=True, key="ct_limit_zones_wide")
                        with cB:
                            top_z = st.number_input("Top zones", min_value=10, max_value=1500, value=120, step=10, key="ct_topz_wide")
                        with cC:
                            show_heatmap = st.checkbox("Afficher en heatmap", value=False, key="ct_show_heatmap")
        
                        if limit_zones:
                            zones_top = (
                                df_scope[COL_ZS].fillna("Inconnu")
                                .value_counts()
                                .head(int(top_z))
                                .index.tolist()
                            )
                            df_ct = df_scope[df_scope[COL_ZS].fillna("Inconnu").isin(zones_top)].copy()
                        else:
                            df_ct = df_scope.copy()
        
                        ct = pd.crosstab(
                            index=df_ct[COL_PROV].fillna("Inconnu"),
                            columns=df_ct[COL_ZS].fillna("Inconnu"),
                            margins=True,
                            margins_name="Total",
                            dropna=False
                        )
        
                        sort_totals = st.checkbox("Trier par total décroissant", value=True, key="ct_sort_totals")
                        if sort_totals and "Total" in ct.columns and "Total" in ct.index:
                            rows = ct.drop(index="Total", errors="ignore").sort_values("Total", ascending=False)
                            cols_tot = ct.drop(columns="Total", errors="ignore").loc["Total"].sort_values(ascending=False).index.tolist() \
                                if "Total" in ct.index else ct.drop(columns="Total", errors="ignore").columns.tolist()
                            ct = rows[cols_tot]
                            ct.loc["Total"] = ct.sum(axis=0)
                            ct["Total"] = ct.sum(axis=1)
                            ct = ct.fillna(0).astype(int)
        
                        st.dataframe(ct, width='stretch', height=int(tbl_height))
        
                        if do_download:
                            csv = ct.to_csv(index=True).encode("utf-8")
                            st.download_button(
                                "Télécharger province_x_zone (CSV)",
                                data=csv,
                                file_name="province_x_zone.csv",
                                mime="text/csv",
                                key="dl_ct_wide"
                            )
        
                        if show_heatmap:
                            ct_heat = ct.drop(index="Total", errors="ignore").drop(columns="Total", errors="ignore")
                            fig_hm = px.imshow(
                                ct_heat,
                                aspect="auto",
                                labels=dict(x="Zone de santé", y="Province", color="Occurrences"),
                                title="Heatmap – Occurrences Province × Zone"
                            )
                            fig_hm.update_layout(height=700)
                            st.plotly_chart(fig_hm, width="stretch")
        
    # =========================
    # TAB 6: DATA & EXPORT
    # =========================
with tab6:
    if IDSR_MODE:
        st.info("🧭 Mode **IDSR agrégé (hebdo)** : les analyses line list ne sont pas actives. Va dans l'onglet **9) IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Consulter et exporter les données filtrées pour analyses/partage.
        
            **📖 Utilisation**
            - Export **CSV/Excel** pour analyses complémentaires (R/Python/DHIS2).
            - Vérifier les filtres actifs avant export.
        
            **⚠️ Points d’attention**
            - Les exports reflètent exactement le périmètre filtré (province/ZS/AS/semaine/classification).
            """,
            expanded=False
        )
        
        st.subheader("Données filtrées & export")
        
        st_dataframe_safe(df_f, height=420)
        
        csv = df_to_csv_bytes(df_f)
        st.download_button(
            "Télécharger CSV (filtré)",
            data=csv,
            file_name="cholera_filtre.csv",
            mime="text/csv"
        )
        
        try:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_f.to_excel(writer, sheet_name="LL_Cholera", index=False)
        
            st.download_button(
                "Télécharger Excel (filtré)",
                data=buffer.getvalue(),
                file_name="cholera_filtre.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception:
            st.info("Export Excel indisponible (openpyxl ?).")
        
    # =========================
    # TAB 7 — Labo / qualité / signaux
    # =========================
with tab7:
    if IDSR_MODE:
        st.info("🧭 Mode **IDSR agrégé (hebdo)** : les analyses line list ne sont pas actives. Va dans l'onglet **9) IDSR**.")
    else:
        tab_help(
            "Comment lire cet onglet",
            """
            **🎯 Objectif** : Détecter incohérences, problèmes de complétude, goulots labo, et signaux d’alerte.
        
            **📖 Sections**
            - **Indicateurs rapides** : 3–5 KPI qualité/action
            - **QC Flags** : incohérences (dates, TDR, âge…)
            - **Complétude champs clés** : % remplissage par site
            - **Cascade labo** : cas → prélèvement → TDR → résultat valide → positif
            - **Alertes tendance** : hausse inhabituelle vs baseline simple
        
            **⚠️ Points d’attention**
            - Un signal ≠ confirmation d’épidémie : déclenche une investigation terrain.
            - Les % de cascade sont calculés sur une logique *entonnoir* (séquentielle).
            """,
            expanded=False
        )
        
        st.subheader("Qualité des données & alertes opérationnelles")
        
        # -------- Helpers (robustes) --------
        def _get_pct_from_cascade(casc: pd.DataFrame, key: str) -> float:
            """Récupère le % de la première ligne dont Étape contient key (robuste aux libellés)."""
            if casc is None or casc.empty or "Étape" not in casc.columns or "%" not in casc.columns:
                return np.nan
            m = casc.loc[casc["Étape"].astype(str).str.contains(key, regex=False, na=False), "%"]
            return float(m.iloc[0]) if len(m) else np.nan
        
        def _safe_num(x):
            try:
                return float(x)
            except Exception:
                return np.nan
        
        # ==========================================================
        # 0) Indicateurs rapides (KPI)
        # ==========================================================
        n_total = len(df_f)
        
        kpi = compute_indicators(df_f)
        casc_global = cascade_metrics(df_f) if n_total else pd.DataFrame()
        
        # KPI “qualité TDR” (sur cascade)
        kpi_incoh_res_wo_tdr = _get_pct_from_cascade(casc_global, "Résultat renseigné mais TDR_realise != Oui")
        kpi_status_in_result = _get_pct_from_cascade(casc_global, "Statut saisi dans TDR_Resultat")
        
        # ✅ 7 colonnes (ajout hospitalisation)
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        
        c1.metric(
            "Cas (n)",
            f"{kpi['n_cases']:,}".replace(",", " "),
            help="Nombre total de cas après application des filtres (Province/ZS/SE, etc.)."
        )
        
        c2.metric(
            "% prélèvement",
            "-" if np.isnan(kpi["prelev_pct"]) else f"{kpi['prelev_pct']:.1f}",
            help=f"Prélèvement=Oui / Tous les cas filtrés. n={kpi.get('prelev_num', 0)}/{kpi.get('prelev_den', kpi.get('n_cases', 0))}"
        )
        
        c3.metric(
            "Couverture TDR (%)",
            "-" if np.isnan(kpi["tdr_pct"]) else f"{kpi['tdr_pct']:.1f}",
            help=f"TDR_realise=Oui / Tous les cas filtrés. n={kpi.get('tdr_num', 0)}/{kpi.get('tdr_den', kpi.get('n_cases', 0))}"
        )
        
        # ✅ Positivité
        pos_label = "-"
        if not np.isnan(kpi["pos_pct"]):
            pos_label = f"{kpi['pos_pct']:.1f}"
        c4.metric(
            "Positivité TDR",
            pos_label,
            help=(
                "Positifs / (Positifs + Négatifs) parmi les TDR interprétables "
                "(TDR_realise=Oui ET résultat valide Pos/Nég). "
                f"n={kpi.get('pos_num', 0)}/{kpi.get('pos_den', 0)}"
            )
        )
        
        # 🆕 Taux hospitalisation
        c5.metric(
            "Hospitalisation (%)",
            "-" if np.isnan(kpi["hosp_pct"]) else f"{kpi['hosp_pct']:.1f}",
            help=f"Hospitalisation=Oui / Tous les cas filtrés. n={kpi.get('hosp_num', 0)}/{kpi.get('hosp_den', kpi.get('n_cases', 0))}"
        )
        
        c6.metric(
            "CFR (%)",
            "-" if np.isnan(kpi["cfr_pct"]) else f"{kpi['cfr_pct']:.2f}",
            help=f"Décès / Tous les cas filtrés. n={kpi.get('n_deaths', 0)}/{kpi.get('n_cases', 0)}"
        )
        
        # % invalides
        inv_label = "-"
        if "invalid_pct" in kpi and not np.isnan(kpi["invalid_pct"]):
            inv_label = f"{kpi['invalid_pct']:.1f}"
        c7.metric(
            "% TDR invalides",
            inv_label,
            help=(
                "Invalides (ex: INBA/bande absente) / TDR réalisés (TDR_realise=Oui). "
                f"n={kpi.get('invalid_num', 0)}/{kpi.get('invalid_den', 0)}"
            )
        )
        
        # Alertes qualité TDR (si dispo)
        if not np.isnan(kpi_incoh_res_wo_tdr) or not np.isnan(kpi_status_in_result):
            with st.expander("📌 Signaux qualité TDR (données)", expanded=False):
                if not np.isnan(kpi_incoh_res_wo_tdr):
                    st.write(f"- **% Résultat renseigné mais TDR_realise ≠ Oui**: **{kpi_incoh_res_wo_tdr:.1f}%**")
                if not np.isnan(kpi_status_in_result):
                    st.write(f"- **% Statut saisi dans TDR_Resultat** (ex: non réalisé/non prélevé): **{kpi_status_in_result:.1f}%**")
        
        with st.expander("🔎 Détail cascade labo (entonnoir) + incohérences", expanded=False):
            st_dataframe_safe(casc_global)
        
        
        # ==========================================================
        # 1) QC Flags (incohérences)
        # ==========================================================
        with st.expander("🔎 Incohérences (QC Flags)", expanded=False):
        
        
            flags = qc_flags(df_f)
            if flags.empty:
                st.success("Aucune incohérence détectée selon les règles actuelles.")
            else:
                # Résumé
                resume = flags["flag"].value_counts().reset_index()
                resume.columns = ["Flag", "Occurrences"]
                st_dataframe_safe(resume)
        
                # Filtre par flag
                flag_list = sorted(flags["flag"].dropna().unique().tolist())
                flag_sel = st.selectbox("Filtrer le détail par flag", ["Tous"] + flag_list, index=0)
        
                # Détail (merge + colonnes utiles)
                cols_show = [c for c in [
                    "Nom_complet", COL_PROV, COL_ZS, COL_AS, COL_SEX, COL_AGE, COL_UNIT,
                    "YW", COL_WNUM, DATE_ONSET, DATE_ADM, DATE_PREL,
                    COL_PREL, COL_TDR, COL_TDRR, COL_HOSP, COL_ISSUE, COL_CLASS
                ] if c in df_f.columns]
        
                detail = flags.merge(df_f.reset_index().rename(columns={"index": "row_id"}), on="row_id", how="left")
        
                if flag_sel != "Tous":
                    detail = detail[detail["flag"] == flag_sel]
        
                st.caption("Détail des lignes concernées (filtré) — max 500 lignes")
                st.dataframe(detail[["flag"] + cols_show].head(500), width="stretch", height=420)
        
        # ==========================================================
        # 2) Complétude des champs clés
        # ==========================================================
        with st.expander("🔎 Complétude des champs clés", expanded=False):
        
            champs_cles = [
                COL_PROV, COL_ZS, COL_AS, "YW", COL_WNUM,
                COL_SEX, COL_AGE, COL_UNIT, DATE_ONSET,
                COL_PREL, COL_TDR, COL_TDRR, COL_HOSP,
                COL_ISSUE, COL_CLASS
            ]
        
            group_choices = [c for c in [COL_PROV, COL_ZS, "YW", COL_WNUM] if c in df_f.columns]
            group_for_comp = st.selectbox("Complétude par", group_choices, index=0 if group_choices else 0)
        
            comp = completeness_table(df_f, champs_cles, by=group_for_comp) if group_choices else pd.DataFrame()
        
            if comp.empty:
                st.info("Impossible de calculer la complétude (colonne group ou champs absents).")
            else:
                st_dataframe_safe(comp, height=520)
        
                # Bar chart plus lisible: top N pires scores
                topn = st.slider("Afficher les N groupes les moins complets", min_value=10, max_value=80, value=25, step=5)
                comp_plot = comp.sort_values("score_completude_%").head(topn)
        
                figc = px.bar(
                    comp_plot,
                    x=group_for_comp,
                    y="score_completude_%",
                    title=f"Score complétude (%) – {topn} groupes les moins complets (par {group_for_comp})"
                )
                figc.update_layout(xaxis_tickangle=-45, yaxis=dict(range=[0, 100]))
                figc = apply_plotly_value_annotations(figc, annot_vals)
                st.plotly_chart(figc, width="stretch")
        
        
        # ==========================================================
        # 3) Cascade prélèvement → TDR → résultat → positif
        # ==========================================================
        with st.expander("🔎 Cascade prélèvement → TDR → résultat → positif", expanded=False):
        
            cascad = cascade_metrics(df_f) if n_total else pd.DataFrame()
            if cascad.empty:
                st.info("Cascade indisponible (aucune donnée après filtres).")
            else:
                st_dataframe_safe(cascad)
        
            # Cascade par province (résumé robuste)
            if COL_PROV in df_f.columns and n_total:
                st.caption("Cascade par province (résumé)")
        
                rows = []
                for prov, sub in df_f.groupby(COL_PROV, dropna=False):
                    c = cascade_metrics(sub)
                    rows.append([
                        prov,
                        len(sub),
                        _get_pct_from_cascade(c, "Prélèvement=Oui"),
                        _get_pct_from_cascade(c, "TDR réalisé=Oui"),
                        _get_pct_from_cascade(c, "Résultat TDR valide"),
                        _get_pct_from_cascade(c, "TDR positif"),
                        _get_pct_from_cascade(c, "Résultat renseigné mais TDR_realise != Oui"),
                    ])
        
                df_cas = pd.DataFrame(
                    rows,
                    columns=[COL_PROV, "n", "% prélèvement", "% TDR", "% résultat valide", "% positif", "% incoh TDR"]
                )
        
                sort_col = st.selectbox(
                    "Trier par",
                    ["n", "% prélèvement", "% TDR", "% résultat valide", "% positif", "% incoh TDR"],
                    index=0
                )
                df_cas_sorted = df_cas.sort_values(sort_col, ascending=False if sort_col == "n" else True)
        
                st_dataframe_safe(df_cas_sorted, height=420)
        
        
        # ==========================================================
        # 4) Alertes tendance (hausse vs baseline simple)
        # ==========================================================
        with st.expander("🔎 Alertes tendance (hausse vs baseline simple)", expanded=False):
            alert_group_choices = [c for c in [COL_PROV, COL_ZS] if c in df_f.columns]
            alert_group = st.selectbox("Grouper les alertes par", alert_group_choices, index=0 if alert_group_choices else 0)
        
            alerts = alerts_weekly_simple(df_f, alert_group) if alert_group_choices else pd.DataFrame()
        
            if alerts.empty:
                st.info("Alertes indisponibles (YW manquant, groupe absent, ou pas assez de semaines).")
            else:
                # Dernière semaine observée
                last_yw = alerts["YW"].dropna().max()
                st.caption(f"Dernière semaine observée: {last_yw}")
        
                last = alerts[alerts["YW"] == last_yw].copy()
        
                # sécurité var_% (éviter inf)
                if "Cas_prev" in last.columns and "Cas" in last.columns:
                    last["Cas_prev"] = last["Cas_prev"].fillna(0)
                    last["var_%"] = np.where(
                        last["Cas_prev"] > 0,
                        (last["Cas"] - last["Cas_prev"]) / last["Cas_prev"] * 100.0,
                        np.nan
                    )
        
                # classement: signal d’abord, puis plus gros volumes
                last["signal"] = last["signal"].fillna(False)
                last = last.sort_values(["signal", "Cas"], ascending=[False, False])
        
                cols_out = [c for c in [alert_group, "YW", "Cas", "Cas_prev", "var_%", "baseline_3w", "signal"] if c in last.columns]
                st_dataframe_safe(last[cols_out], height=520)
        
                # Top signaux
                sig = last[last["signal"] == True].head(30)
                if len(sig):
                    figa = px.bar(sig, x=alert_group, y="Cas", title=f"Signaux (semaine {last_yw}) – top 30")
                    figa.update_layout(xaxis_tickangle=-45)
                    figa = apply_plotly_value_annotations(figa, annot_vals)
                    st.plotly_chart(figa, width="stretch")
                else:
                    st.success("Aucun signal détecté avec les seuils actuels (baseline*1.5 et Cas≥10).")
        
    # =========================
    # TAB 8: Sitrep automatique
    # =========================
with tab8:
    if IDSR_MODE:
        st.info("🧭 Mode **IDSR agrégé (hebdo)** : les analyses line list ne sont pas actives. Va dans l'onglet **9) IDSR**.")
    else:
        import pandas as pd
        import numpy as np
        from datetime import date
        
        st.markdown("## SITREP")
        tab_help(
            "Comment lire cet onglet",
            """
            ### 📰 Objectif du SITREP automatique
            Cet onglet génère un **rapport épidémiologique hebdomadaire** à partir des données actuellement filtrées dans le tableau de bord.
        
            ---
        
            ### ⚙️ Comment ça fonctionne
            - Le SITREP utilise **les données filtrées (df_f)** : provinces, ZS, période, classification, etc.
            - Les indicateurs sont recalculés **automatiquement** selon la **SE** et l’**année** sélectionnées.
            - Si tu changes les filtres du dashboard, le SITREP se met à jour.
        
            ---
        
            ### 📌 Sections du rapport
            **1️⃣ Points saillants**  
            Résumé automatique de la situation :
            - nombre de cas et décès de la semaine
            - évolution par rapport aux semaines précédentes
            - zones de santé les plus affectées
        
            **2️⃣ Situation épidémiologique**  
            Indicateurs clés :
            - Cas et décès de la semaine
            - Taux de létalité (CFR)
            - Cas cumulés de l’année
            - Tableau des zones de santé les plus touchées
        
            **3️⃣ Labo / qualité / signaux**  
            Indicateurs de surveillance :
            - Cascade prélèvement → TDR → résultat (si données disponibles)
            - Alertes statistiques basées sur l’évolution récente des cas
        
            ---
        
            ### 📤 Export
            Tu peux télécharger le SITREP généré automatiquement au format **PDF** en bas de page.
            Le document exporté reflète exactement les données visibles dans cet onglet.
        
            ---
            ℹ️ **Astuce :** Pour produire le SITREP officiel de la semaine, règle d’abord les filtres du tableau de bord (période, province, etc.), puis viens ici pour exporter.
            """,
            expanded=False
        )
        
        # =========================================================
        # 1) UI: SE / Année / Date de publication dépendants de df_f
        # =========================================================
        # Bornes SE
        if (COL_WNUM in df_f.columns) and df_f[COL_WNUM].notna().any():
            w_series = pd.to_numeric(df_f[COL_WNUM], errors="coerce").dropna()
            w_min, w_max = int(w_series.min()), int(w_series.max())
        else:
            w_min, w_max = 1, 53
        
        # Bornes Année
        if (COL_YEAR in df_f.columns) and df_f[COL_YEAR].notna().any():
            y_series = pd.to_numeric(df_f[COL_YEAR], errors="coerce").dropna()
            y_min, y_max = int(y_series.min()), int(y_series.max())
        else:
            y_min, y_max = 2020, date.today().year
        
        auto_last = st.checkbox(
            "Auto: utiliser la dernière SE/Année du filtrage",
            value=True,
            key="sitrep_auto_last"
        )
        
        colA, colB, colC = st.columns(3)
        
        with colA:
            semaine = st.number_input(
                "Semaine épidémiologique (SE)",
                min_value=int(w_min),
                max_value=int(w_max),
                value=int(w_max),
                step=1,
                key="sitrep_se",
            )
        
        with colB:
            annee = st.number_input(
                "Année",
                min_value=int(y_min),
                max_value=int(y_max),
                value=int(y_max),
                step=1,
                key="sitrep_year",
            )
        
        with colC:
            date_pub = st.date_input(
                "Date de publication",
                value=date.today(),
                key="sitrep_pubdate",
            )
        
        # Forcer aux valeurs "dernière SE/Année" si auto_last
        if auto_last:
            semaine = int(w_max)
            annee = int(y_max)
        
        st.caption(
            f"Scope SITREP: df_f (filtré). SE disponibles: {w_min}–{w_max}. "
            f"Années disponibles: {y_min}–{y_max}."
        )
        
        # =========================================================
        # 2) Build payload (défini ici pour que Tab8 soit autonome)
        # =========================================================
        def _build_sitrep_payload_from_df(df_scope, se, annee, date_pub):
            """
            Build un payload SITREP à partir de df_scope (ici df_f filtré).
        
            - Filtre SE/Année pour les indicateurs de la semaine (d_se)
            - Calcule les cumuls année jusqu'à la SE (d_cum)
            - Produit un tableau ZS (top cas) et des points saillants
            """
            d = df_scope.copy()
        
            # Fix: colonnes dupliquées => garder 1ère occurrence
            if d.columns.duplicated().any():
                d = d.loc[:, ~d.columns.duplicated()].copy()
        
            # Filtre SE/Année
            d_se = d.copy()
            if COL_WNUM in d_se.columns:
                d_se = d_se[pd.to_numeric(d_se[COL_WNUM], errors="coerce") == int(se)]
            if COL_YEAR in d_se.columns:
                d_se = d_se[pd.to_numeric(d_se[COL_YEAR], errors="coerce") == int(annee)]
        
            # Cumul année <= SE
            d_cum = d.copy()
            if COL_YEAR in d_cum.columns:
                d_cum = d_cum[pd.to_numeric(d_cum[COL_YEAR], errors="coerce") == int(annee)]
            if COL_WNUM in d_cum.columns:
                d_cum = d_cum[pd.to_numeric(d_cum[COL_WNUM], errors="coerce") <= int(se)]
        
            def _kpi(df_):
                cases = int(len(df_))
                deaths = int(df_["is_death"].sum()) if "is_death" in df_.columns else 0
                cfr = (deaths / cases * 100) if cases > 0 else 0.0
                return cases, deaths, cfr
        
            cas_se, dec_se, cfr_se = _kpi(d_se)
            cas_cum, dec_cum, cfr_cum = _kpi(d_cum)
        
            # ---------------------------------------------------------
            # Table épidémiologique par ZS (sur la SE sélectionnée)
            # -> Ajout de la colonne "Province de notification"
            # ---------------------------------------------------------
            table_epi = pd.DataFrame()
        
            prov_notif_col = None
            if "COL_PROV_NOTIF" in globals() and globals()["COL_PROV_NOTIF"] in d_se.columns:
                prov_notif_col = globals()["COL_PROV_NOTIF"]
            elif "COL_PROV" in globals() and globals()["COL_PROV"] in d_se.columns:
                prov_notif_col = globals()["COL_PROV"]
        
            if (COL_ZS in d_se.columns) and len(d_se):
                tmp = d_se.copy()
                tmp["_cas_"] = 1
                tmp["_deces_"] = tmp["is_death"].astype(int) if "is_death" in tmp.columns else 0
        
                group_cols = []
                if prov_notif_col is not None:
                    group_cols.append(prov_notif_col)
                group_cols.append(COL_ZS)
        
                table_epi = (
                    tmp.groupby(group_cols, as_index=False)
                       .agg(cas=("_cas_", "sum"), deces=("_deces_", "sum"))
                       .sort_values("cas", ascending=False)
                )
        
                if prov_notif_col is not None:
                    table_epi = table_epi.rename(columns={prov_notif_col: "Province de notification"})
        
            points = [
                f"SE{int(se):02d}/{int(annee)} : {cas_se} cas, {dec_se} décès (CFR {cfr_se:.2f}%).",
                f"Cumul année (SE01→SE{int(se):02d}) : {cas_cum} cas, {dec_cum} décès (CFR {cfr_cum:.2f}%).",
            ]
        
            if not table_epi.empty:
                top5 = table_epi.head(5)
                if "Province de notification" in table_epi.columns:
                    points.append(
                        "Top 5 ZS (cas) : " + ", ".join(
                            [f"{r['Province de notification']} / {r[COL_ZS]}={int(r['cas'])}"
                             for _, r in top5.iterrows()]
                        )
                    )
                else:
                    points.append(
                        "Top 5 ZS (cas) : " + ", ".join(
                            [f"{r[COL_ZS]}={int(r['cas'])}" for _, r in top5.iterrows()]
                        )
                    )
        
            payload = {
                "meta": {"semaine": int(se), "annee": int(annee), "date_publication": date_pub},
                "kpi": {
                    "cas_semaine": cas_se,
                    "deces_semaine": dec_se,
                    "cfr_semaine": cfr_se,
                    "cas_cumul": cas_cum,
                    "deces_cumul": dec_cum,
                    "cfr_cumul": cfr_cum,
                },
                "table_epi": table_epi,
                "points_saillants": points,
            }
        
            if "cascade_metrics" in globals() and callable(globals()["cascade_metrics"]):
                try:
                    payload["cascade"] = globals()["cascade_metrics"](d_se)
                except Exception:
                    payload["cascade"] = pd.DataFrame()
            else:
                payload["cascade"] = pd.DataFrame()
        
            if "alerts_weekly_simple" in globals() and callable(globals()["alerts_weekly_simple"]):
                try:
                    payload["alertes_last"] = globals()["alerts_weekly_simple"](d, COL_PROV) if COL_PROV in d.columns else pd.DataFrame()
                except Exception:
                    payload["alertes_last"] = pd.DataFrame()
            else:
                payload["alertes_last"] = pd.DataFrame()
        
            return payload
        
        sitrep_payload = _build_sitrep_payload_from_df(df_f, semaine, annee, date_pub)
        
        # =========================================================
        # 3) Affichage (pliable)
        # =========================================================
        with st.expander("1) Points saillants", expanded=True):
            if sitrep_payload["points_saillants"]:
                for b in sitrep_payload["points_saillants"]:
                    st.markdown(f"- {b}")
            else:
                st.caption("Aucun point saillant (données insuffisantes pour le scope).")
        
        with st.expander("2) Situation épidémiologique", expanded=True):
            k = sitrep_payload["kpi"]
        
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Cas (SE)", f"{k['cas_semaine']:,}".replace(",", " "))
            k2.metric("Décès (SE)", f"{k['deces_semaine']:,}".replace(",", " "))
            k3.metric("CFR (SE) %", f"{k['cfr_semaine']:.2f}")
            k4.metric(
                "Semaine min (filtré)",
                str(df_f[COL_WNUM].min()) if (COL_WNUM in df_f.columns and len(df_f)) else "-"
            )
            k5.metric(
                "Semaine max (filtré)",
                str(df_f[COL_WNUM].max()) if (COL_WNUM in df_f.columns and len(df_f)) else "-"
            )
        
            st.caption(
                f"Cumul année (SE01→SE{int(semaine):02d}) : "
                f"{k['cas_cumul']:,} cas, {k['deces_cumul']:,} décès (CFR {k['cfr_cumul']:.2f}%)."
                .replace(",", " ")
            )
        
            if sitrep_payload["table_epi"] is not None and not sitrep_payload["table_epi"].empty:
                st_dataframe_safe(sitrep_payload["table_epi"], height=520)
            else:
                st.caption("Table ZS indisponible (pas de données sur la SE/année ou colonne ZS manquante).")
        
        with st.expander("3) Labo / qualité / signaux", expanded=False):
            cascad = sitrep_payload.get("cascade")
            if cascad is not None and isinstance(cascad, pd.DataFrame) and not cascad.empty:
                st.markdown("### Cascade prélèvement → TDR → résultat")
                st_dataframe_safe(cascad, height=320)
            else:
                st.caption("Cascade indisponible (fonction/colonnes manquantes ou pas de données sur la SE).")
        
            al = sitrep_payload.get("alertes_last")
            if al is not None and isinstance(al, pd.DataFrame) and not al.empty:
                st.markdown("### Alertes (dernière semaine disponible)")
                cols = [c for c in ["YW", "Cas", "Cas_prev", "var_%", "baseline_3w", "signal"] if c in al.columns]
                st_dataframe_safe(al[cols] if cols else al, height=420)
            else:
                st.caption("Alertes indisponibles (fonction absente ou pas assez d’historique).")
        
        # =========================================================
        # 4) Export PDF
        # =========================================================
        st.divider()
        st.markdown("### Export")
        
        if "export_sitrep_pdf" in globals() and callable(export_sitrep_pdf):
            pdf_bytes = export_sitrep_pdf(sitrep_payload)
            st.download_button(
                "⬇️ Télécharger le SITREP (PDF)",
                data=pdf_bytes,
                file_name=f"SITREP_CHOLERA_SE{int(semaine):02d}_{int(annee)}.pdf",
                mime="application/pdf",
                type="primary",
                key="sitrep_dl_pdf",
            )
        else:
            st.error("La fonction export_sitrep_pdf(payload) n'est pas définie dans ce script.")
        
    # =========================
    # TAB 9 — IDSR
    # =========================
    # ----------------------------
    # Cache (Streamlit) : lecture Excel
    # ----------------------------
    @st.cache_data(show_spinner=False)
    def load_excel_cached(file, sheet_name=None):
            """Lecture Excel avec cache pour accélérer l'app (supporte UploadedFile ou chemin)."""
            return pd.read_excel(file, sheet_name=sheet_name) if sheet_name else pd.read_excel(file)
        
    # ----------------------------
    # Helpers robustes
    # ----------------------------
    def clean_week(series: pd.Series) -> pd.Series:
            """Nettoie une colonne semaine (extrait digits) -> Int64, bornée 1..53."""
            s = series.astype("string").str.extract(r"(\d+)", expand=False)
            w = pd.to_numeric(s, errors="coerce").astype("Int64")
            return w.where((w >= 1) & (w <= 53), pd.NA)
        
    def clean_year(series: pd.Series) -> pd.Series:
            """Nettoie une colonne année (extrait YYYY) -> Int64."""
            s = series.astype("string").str.extract(r"((?:19|20)\d{2})", expand=False)
            y = pd.to_numeric(s, errors="coerce").astype("Int64")
            return y.where((y >= 2000) & (y <= 2100), pd.NA)
        
    def parse_year_from_filename(path_or_name: str):
            """Extrait une année YYYY depuis un nom de fichier si disponible."""
            if not path_or_name:
                return None
            m = re.search(r"(19|20)\d{2}", str(path_or_name))
            return int(m.group()) if m else None
        
    def iso_monday_from_year_week(y, w):
            """Construit le lundi ISO depuis (année ISO, semaine ISO). Renvoie NaT si invalide."""
            try:
                return pd.Timestamp(date.fromisocalendar(int(y), int(w), 1))
            except Exception:
                return pd.NaT
        
    def norm_text(series: pd.Series) -> pd.Series:
            """Normalise du texte pour réduire les doublons (espaces, casse)."""
            s = series.astype("string")
            s = s.str.replace(r"\s+", " ", regex=True).str.strip()
            return s
        
    def to_numeric_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
            """Convertit une liste de colonnes en numeric si elles existent."""
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        
with tab9:
    st.markdown("## IDSR – Analyses")

    tab_help(
        "Comment lire cet onglet",
        """
    **🎯 Objectif** : analyser les tendances IDSR (cas/décès/CFR) par maladie et par niveau géographique (province/ZS),
    à partir d’un fichier agrégé par semaine.

    **✅ Inclus**
    - Évolution des cas/décès par semaine
    - CFR recalculé et comparaison avec Taux_letalite (si disponible)
    - Top provinces / ZS
    - Contrôles de cohérence (totaux vs tranches d’âge)
    - Mode secours si Année-Semaine (YW) non exploitable : filtre sur Numéro de semaine uniquement
    """,
        expanded=False
    )

    # -------------------------------------------------------------------------
    # 1) Chargement fichier IDSR
    # -------------------------------------------------------------------------
    st.caption("📥 Charger un fichier IDSR agrégé (.xlsx).")
    # 2 façons de téléverser :
    #  - Sidebar (si la source sélectionnée est IDSR)
    #  - Ici dans l’onglet 9 (toujours disponible)
    up_from_sidebar = idsr_upl_side if ('idsr_upl_side' in globals() or 'idsr_upl_side' in locals()) else None
    if up_from_sidebar is not None:
        st.caption("✅ Fichier IDSR détecté depuis la sidebar (mode IDSR).")
    up = up_from_sidebar or st.file_uploader("Fichier IDSR agrégé", type=["xlsx"], key="idsr_upl")

    default_path = "rdc_compilation_IDS_RDC_SE01_SE03_25_01_2026_00_07_33.xlsx"

    if up is not None:
        # priorité à la feuille IDS_RDC; sinon première feuille
        try:
            df_idsr = load_excel_cached(up, sheet_name="IDS_RDC")
        except Exception:
            df_idsr = load_excel_cached(up)
        src = "upload"
    else:
        try:
            df_idsr = load_excel_cached(default_path, sheet_name="IDS_RDC")
            src = default_path
        except Exception:
            try:
                df_idsr = load_excel_cached(default_path)
                src = default_path
            except Exception:
                df_idsr = pd.DataFrame()
                src = None

    if df_idsr.empty:
        st.info("Charge un fichier IDSR agrégé (xlsx) pour afficher les analyses.")
    else:
        st.success(f"Fichier chargé: {src} | Lignes: {len(df_idsr):,}")

        # ---------------------------------------------------------------------
        # 2) Harmonisation colonnes (BRUT vs COMPILÉ)
        # ---------------------------------------------------------------------
        df_idsr = df_idsr.copy()

        rename_map = {
            # Identifiants
            "NUM": "Num",
            "PAYS": "Pays",
            "PROV": "Province_notification",
            "Province": "Province_notification",
            "ZS": "Zone_de_sante_notification",
            "Zone_de_sante": "Zone_de_sante_notification",
            "POP": "Population",

            # GIS (si disponible)
            "prov_GIS": "Province_GIS",
            "Prov_GIS": "Province_GIS",
            "Province_GIS": "Province_GIS",
            "zs_GIS": "ZS_GIS",
            "ZS_GIS": "ZS_GIS",
            "ZoneSante_GIS": "ZS_GIS",


            # Temps
            "NUMSEM": "Num_semaine_epid",
            "Semaine": "Num_semaine_epid",
            # DEBUTSEM reste inchangé

            # Maladie
            "MALADIE": "Maladie",
            "disease": "Maladie",

            # Tranches âge (cas)
            "C328TNN": "Cas_tnn",
            "C011MOIS": "Cas_0_11mois",
            "C1259MOIS": "Cas_12_59mois",
            "C515ANS": "Cas_5_14ans",
            "CP15ANS": "Cas_15plus",

            # Tranches âge (décès)
            "DTNN": "Deces_tnn",
            "D011MOIS": "Deces_0_11mois",
            "D1259MOIS": "Deces_12_59mois",
            "D515ANS": "Deces_5_14ans",
            "DP15ANS": "Deces_15plus",

            # Totaux & indicateurs
            "TOTALCAS": "Total_cas",
            "TOTALDECES": "Total_deces",
            "LETAL": "Taux_letalite",
            "ATTAQ": "Taux_attaque",

            # Statut & clé
            "RecStatus": "Recstatus",
            "UniqueKey": "Cle_unique",

            # Année / semaine compilées
            "Year": "Annee_epid",
            "year": "Annee_epid",
            "Annee": "Annee_epid",
        }

        df_idsr = df_idsr.rename(columns={k: v for k, v in rename_map.items() if k in df_idsr.columns})
        # ---------------------------------------------------------
        # ✅ Détecteur automatique BRUT vs COMPILÉ
        # ---------------------------------------------------------
        # BRUT: contient DEBUTSEM + NUMSEM (après rename NUMSEM -> Num_semaine_epid)
        is_brut = ("DEBUTSEM" in df_idsr.columns) and ("Num_semaine_epid" in df_idsr.columns)

        # COMPILÉ: a déjà Date_debut_semaine et/ou Annee_epid / Semaine_epid
        is_compiled = (
            ("Date_debut_semaine" in df_idsr.columns)
            or ("Annee_epid" in df_idsr.columns)
            or ("Semaine_epid" in df_idsr.columns)
        )

        # Petit diagnostic (optionnel, utile)
        with st.expander("🧩 Diagnostic colonnes (dérouler)", expanded=False):
            st.write({
                "version_detectee": "BRUTE (DEBUTSEM/NUMSEM)" if is_brut else "COMPILÉE",
                "colonnes_temps": [
                    c for c in ["DEBUTSEM", "Date_debut_semaine", "Annee_epid", "Num_semaine_epid", "Semaine_epid", "YW"]
                    if c in df_idsr.columns
                ]
            })


        # Colonnes standard
        COL_MAL = "Maladie"
        COL_PROV_ID = "Province_notification"
        COL_ZS_ID = "Zone_de_sante_notification"

        
        # ---------------------------------------------------------------------
        # 2.b) Normalisation texte (Province/ZS/Maladie) pour éviter les doublons
        # ---------------------------------------------------------------------
        for _c in ["Maladie", "Province_notification", "Zone_de_sante_notification", "Province_GIS", "ZS_GIS"]:
            if _c in df_idsr.columns:
                df_idsr[_c] = norm_text(df_idsr[_c])

        # ---------------------------------------------------------------------
        # 3) Standardisation TEMPS (robuste sur semaine)
        # ---------------------------------------------------------------------
        # 3.1 Semaine
        if "Num_semaine_epid" in df_idsr.columns:
            df_idsr["Num_semaine_epid"] = clean_week(df_idsr["Num_semaine_epid"])
        else:
            df_idsr["Num_semaine_epid"] = pd.NA

        # 3.2 Année
        if "Annee_epid" in df_idsr.columns:
            df_idsr["Annee_epid"] = clean_year(df_idsr["Annee_epid"])
        else:
            df_idsr["Annee_epid"] = pd.NA

        # si Annee_epid vide -> essayer depuis Semaine_epid
        if df_idsr["Annee_epid"].isna().all() and "Semaine_epid" in df_idsr.columns:
            df_idsr["Annee_epid"] = clean_year(df_idsr["Semaine_epid"])

        # si semaine vide -> essayer depuis Semaine_epid (dernier nombre)
        if df_idsr["Num_semaine_epid"].isna().all() and "Semaine_epid" in df_idsr.columns:
            wk = df_idsr["Semaine_epid"].astype("string").str.extract(r"(\d{1,2})\s*$", expand=False)
            df_idsr["Num_semaine_epid"] = clean_week(wk)

        # dernier recours: année depuis nom du fichier
        if df_idsr["Annee_epid"].isna().all():
            y_guess = parse_year_from_filename(src)
            if y_guess is not None:
                df_idsr["Annee_epid"] = pd.Series([y_guess] * len(df_idsr), dtype="Int64")

        
        # -----------------------------------------------------------------
        # 3.3 Si fichier COMPILÉ et dates disponibles : dériver Année/Semaine
        # -----------------------------------------------------------------
        # Si l'utilisateur a un fichier compilé avec Date_debut_semaine mais sans Annee/Num_semaine,
        # on reconstruit Annee_epid et Num_semaine_epid depuis la date (ISO year/week).
        if (("Date_debut_semaine" in df_idsr.columns) or ("DEBUTSEM" in df_idsr.columns)) and (
            df_idsr["Annee_epid"].isna().all() or df_idsr["Num_semaine_epid"].isna().all()
        ):
            _dt_src = None
            if "Date_debut_semaine" in df_idsr.columns:
                _dt_src = pd.to_datetime(df_idsr["Date_debut_semaine"], errors="coerce")
            elif "DEBUTSEM" in df_idsr.columns:
                _dt_src = pd.to_datetime(df_idsr["DEBUTSEM"], errors="coerce")

            if _dt_src is not None and _dt_src.notna().any():
                _iso = _dt_src.dt.isocalendar()
                if df_idsr["Annee_epid"].isna().all():
                    df_idsr["Annee_epid"] = pd.to_numeric(_iso["year"], errors="coerce").astype("Int64")
                if df_idsr["Num_semaine_epid"].isna().all():
                    df_idsr["Num_semaine_epid"] = pd.to_numeric(_iso["week"], errors="coerce").astype("Int64")

        # YW & YW_KEY (si année + semaine)
        df_idsr["YW"] = (
            df_idsr["Annee_epid"].astype("string")
            + "-W"
            + df_idsr["Num_semaine_epid"].astype("string").str.zfill(2)
        )
        df_idsr["YW_KEY"] = (
            df_idsr["Annee_epid"].astype("Int64") * 100
            + df_idsr["Num_semaine_epid"].astype("Int64")
        )

        # Date ISO reconstruite pour affichage (basée sur Année+Semaine)
        df_idsr["Date_debut_semaine_iso"] = [
            iso_monday_from_year_week(y, w)
            for y, w in zip(df_idsr["Annee_epid"].tolist(), df_idsr["Num_semaine_epid"].tolist())
        ]

        # ---------------------------------------------------------------------
        # 4) QC date vs semaine (si date source disponible)
        # IMPORTANT : comparaison faite en numpy float64 (évite pd.NA bool ambigu)
        # ---------------------------------------------------------------------
        if "Date_debut_semaine" in df_idsr.columns:
            src_dt = pd.to_datetime(df_idsr["Date_debut_semaine"], errors="coerce")
        elif "DEBUTSEM" in df_idsr.columns:
            src_dt = pd.to_datetime(df_idsr["DEBUTSEM"], errors="coerce")
            df_idsr["Date_debut_semaine"] = df_idsr["DEBUTSEM"]  # copie visible
        else:
            src_dt = pd.Series(pd.NaT, index=df_idsr.index)

        has_date = src_dt.notna()

        if has_date.any():
            iso = src_dt.dt.isocalendar()

            iso_year = pd.to_numeric(iso["year"], errors="coerce").to_numpy(dtype="float64")
            iso_week = pd.to_numeric(iso["week"], errors="coerce").to_numpy(dtype="float64")

            y = pd.to_numeric(df_idsr["Annee_epid"], errors="coerce").to_numpy(dtype="float64")
            w = pd.to_numeric(df_idsr["Num_semaine_epid"], errors="coerce").to_numpy(dtype="float64")

            ok_mask = has_date.to_numpy() & (iso_year == y) & (iso_week == w)

            df_idsr["QC_Date_vs_Semaine"] = np.where(
                ~has_date.to_numpy(), "NA",
                np.where(ok_mask, "✅ OK", "❌ KO")
            )
        else:
            df_idsr["QC_Date_vs_Semaine"] = "NA"

        # ---------------------------------------------------------------------
        # 5) Axe temps UNIQUE pour tri/plots (gère mode secours)
        # ---------------------------------------------------------------------
        # TIME_KEY : tri stable (priorité YW_KEY sinon Num_semaine)
        yw_key_num = pd.to_numeric(df_idsr.get("YW_KEY"), errors="coerce")
        wnum_num = pd.to_numeric(df_idsr.get("Num_semaine_epid"), errors="coerce")

        df_idsr["TIME_KEY"] = np.where(yw_key_num.notna(), yw_key_num, wnum_num)

        # TIME_LAB : affichage (priorité YW sinon W##)
        df_idsr["TIME_LAB"] = np.where(
            df_idsr.get("YW", pd.Series([""] * len(df_idsr), index=df_idsr.index)).astype("string").str.contains(r"-W", na=False),
            df_idsr["YW"].astype("string"),
            "W" + wnum_num.astype("Int64").astype("string").str.zfill(2)
        )

        # ⚠️ Confort utilisateur (sans changer la logique) :
        # - Si base BRUTE (ou année indisponible), afficher W## plutôt que YYYY-W##
        _wlab = "W" + wnum_num.astype("Int64").astype("string").str.zfill(2)
        if "is_brut" in locals() and is_brut:
            df_idsr["TIME_LAB"] = _wlab
        else:
            if "Annee_epid" in df_idsr.columns and df_idsr["Annee_epid"].isna().all():
                df_idsr["TIME_LAB"] = _wlab

        # ---------------------------------------------------------------------
        # 6) Conversions numériques (variables d'analyse)

        # ---------------------------------------------------------------------
        
        # ---------------------------------------------------------------------
        # 6.a) Somme des tranches d’âge (cas/décès) + reconstruction prudente des totaux
        # ---------------------------------------------------------------------
        # On calcule toujours la somme des tranches (utile pour QC/écarts),
        # puis on ne reconstruit Total_cas/Total_deces QUE s'ils sont absents
        # ou très majoritairement manquants.
        cas_parts = [c for c in ["Cas_tnn", "Cas_0_11mois", "Cas_12_59mois", "Cas_5_14ans", "Cas_15plus"] if c in df_idsr.columns]
        dec_parts = [c for c in ["Deces_tnn", "Deces_0_11mois", "Deces_12_59mois", "Deces_5_14ans", "Deces_15plus"] if c in df_idsr.columns]

        if cas_parts:
            df_idsr["Total_cas_age"] = df_idsr[cas_parts].sum(axis=1, min_count=1)
        else:
            df_idsr["Total_cas_age"] = pd.NA

        if dec_parts:
            df_idsr["Total_deces_age"] = df_idsr[dec_parts].sum(axis=1, min_count=1)
        else:
            df_idsr["Total_deces_age"] = pd.NA

        # Reconstruction / complétion prudente des totaux (ne pas écraser les totaux valides)
        if "Total_cas" not in df_idsr.columns:
            df_idsr["Total_cas"] = df_idsr["Total_cas_age"]
        else:
            if df_idsr["Total_cas"].isna().mean() > 0.5:
                df_idsr["Total_cas"] = df_idsr["Total_cas"].fillna(df_idsr["Total_cas_age"])

        if "Total_deces" not in df_idsr.columns:
            df_idsr["Total_deces"] = df_idsr["Total_deces_age"]
        else:
            if df_idsr["Total_deces"].isna().mean() > 0.5:
                df_idsr["Total_deces"] = df_idsr["Total_deces"].fillna(df_idsr["Total_deces_age"])

        # ---------------------------------------------------------------------
        df_idsr = to_numeric_cols(df_idsr, [
            "Population",
            "Total_cas", "Total_deces", "Taux_letalite", "Taux_attaque",
            "Cas_tnn", "Cas_0_11mois", "Cas_12_59mois", "Cas_5_14ans", "Cas_15plus",
            "Deces_tnn", "Deces_0_11mois", "Deces_12_59mois", "Deces_5_14ans", "Deces_15plus"
        ])

        # Diagnostic rapide
        with st.expander("🧩 Diagnostic (temps & QC) – déplier", expanded=False):
            st.write({
                "colonnes_temps": [c for c in [
                    "Annee_epid", "Num_semaine_epid", "YW", "YW_KEY",
                    "TIME_LAB", "TIME_KEY", "Date_debut_semaine_iso", "QC_Date_vs_Semaine"
                ] if c in df_idsr.columns],
                "qc_date_vs_semaine": df_idsr["QC_Date_vs_Semaine"].value_counts(dropna=False).to_dict()
            })

        # ---------------------------------------------------------------------
        # 6.b) QC actionnable : afficher & exporter les lignes KO (si existent)
        # ---------------------------------------------------------------------
        if "QC_Date_vs_Semaine" in df_idsr.columns:
            df_qc_ko = df_idsr[df_idsr["QC_Date_vs_Semaine"] == "❌ KO"].copy()
            if not df_qc_ko.empty:
                with st.expander("🚩 Top lignes QC KO (Date vs Année-Semaine) – déplier", expanded=False):
                    show_cols = [c for c in [
                        "Maladie", "Province_notification", "Zone_de_sante_notification",
                        "DEBUTSEM", "Date_debut_semaine",
                        "Annee_epid", "Num_semaine_epid", "Semaine_epid", "YW",
                        "QC_Date_vs_Semaine"
                    ] if c in df_qc_ko.columns]
                    st.dataframe(df_qc_ko[show_cols].head(20), width="stretch")

                    csv_ko = df_qc_ko[show_cols].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Télécharger QC_KO.csv",
                        data=csv_ko,
                        file_name="QC_KO.csv",
                        mime="text/csv",
                        key="tab9_dl_qc_ko"
                    )


        # ---------------------------------------------------------------------
        # 7) Filtres : maladie, province, ZS, semaines (mode normal ou secours)
        # ---------------------------------------------------------------------
        # ---- Filtres (sur une seule ligne) : Maladie / Province / ZS / Année(DEBUTSEM) / Temps
        cA, cB, cC, fD, cD = st.columns(5)

        with cA:
            maladies = sorted([
                x for x in df_idsr.get(COL_MAL, pd.Series(dtype="object")).dropna().unique().tolist()
                if str(x).strip() != ""
            ])
            mal_sel = st.multiselect(
                "Maladie",
                options=maladies,
                default=[],
                help="Laisse vide pour toutes les maladies",
                key="tab9_mal_sel"
            )

        with cB:
            provs = sorted([
                x for x in df_idsr.get(COL_PROV_ID, pd.Series(dtype="object")).dropna().unique().tolist()
                if str(x).strip() != ""
            ])
            prov_sel = st.multiselect(
                "Province",
                options=provs,
                default=[],
                help="Laisse vide pour toutes les provinces",
                key="tab9_prov_sel"
            )

        with cC:
            if COL_ZS_ID in df_idsr.columns:
                if prov_sel and (COL_PROV_ID in df_idsr.columns):
                    zs_pool = df_idsr[df_idsr[COL_PROV_ID].isin(prov_sel)]
                else:
                    zs_pool = df_idsr

                zss = sorted([
                    x for x in zs_pool.get(COL_ZS_ID, pd.Series(dtype="object")).dropna().unique().tolist()
                    if str(x).strip() != ""
                ])

                zs_sel = st.multiselect(
                    "Zone de santé",
                    options=zss,
                    default=[],
                    help="Vide = toutes les ZS (filtrées par province si province sélectionnée)",
                    key="tab9_zs_sel"
                )
            else:
                zs_sel = []
                st.info("Colonne Zone_de_sante_notification absente (filtre ZS indisponible).")

        # Filtre Année (DEBUTSEM) — choix multiple
        years_selected = None  # utilisé plus loin pour messages/contrôles
        with fD:
            if "DEBUTSEM" in df_idsr.columns:
                _year_pool = df_idsr.copy()

                if mal_sel and (COL_MAL in _year_pool.columns):
                    _year_pool = _year_pool[_year_pool[COL_MAL].isin(mal_sel)]

                if prov_sel and (COL_PROV_ID in _year_pool.columns):
                    _year_pool = _year_pool[_year_pool[COL_PROV_ID].isin(prov_sel)]

                if zs_sel and (COL_ZS_ID in _year_pool.columns):
                    _year_pool = _year_pool[_year_pool[COL_ZS_ID].isin(zs_sel)]

                _debutsem = _year_pool["DEBUTSEM"]
                if pd.api.types.is_numeric_dtype(_debutsem):
                    _debutsem_dt = pd.to_datetime(_debutsem, unit="D", origin="1899-12-30", errors="coerce")
                else:
                    _debutsem_dt = pd.to_datetime(_debutsem, errors="coerce")

                years_available = sorted(_debutsem_dt.dt.year.dropna().astype(int).unique().tolist())

                if years_available:
                    years_selected = st.multiselect(
                        "Année (DEBUTSEM)",
                        options=years_available,
                        default=years_available,
                        key="tab9_years_debutsem",
                        help="Très utile en mode BRUT (WNUM) pour éviter de mélanger plusieurs années."
                    )
                else:
                    years_selected = []
                    st.info("Aucune année exploitable trouvée dans DEBUTSEM.")
            else:
                years_selected = []
                st.info("Colonne DEBUTSEM absente (filtre Année indisponible).")


        # Filtre semaines : logique robuste BRUT vs COMPILÉ
        with cD:
            # Badge BRUT / COMPILÉ (aide visuelle)
            _tag = "BRUT" if is_brut else "COMPILÉ"
            _bg = "#ffecb5" if is_brut else "#d1e7dd"
            _border = "#d39e00" if is_brut else "#0f5132"
            _txt = "#111" if is_brut else "#0f5132"
            st.markdown(
                f"""<div style='display:inline-block;padding:2px 10px;border-radius:999px;
                background:{_bg};border:1px solid {_border};color:{_txt};font-weight:700;font-size:12px'>
                IDS {_tag}
                </div>""",
                unsafe_allow_html=True
            )
            # st.caption("BRUT : filtre par numéro de semaine. COMPILÉ : filtre Année–Semaine (YW) si disponible.")

            # ---- utilitaire local : liste de semaines exploitables
            def _get_weeks_list(_df: pd.DataFrame) -> list:
                w = pd.to_numeric(_df.get("Num_semaine_epid"), errors="coerce")
                weeks = (
                    w.dropna()
                    .astype(int)
                    .sort_values()
                    .unique()
                    .tolist()
                )
                # fallback : tenter depuis Semaine_epid si Num_semaine_epid vide
                if (not weeks) and ("Semaine_epid" in _df.columns):
                    wk = _df["Semaine_epid"].astype("string").str.extract(r"(\d{1,2})\s*$", expand=False)
                    weeks = (
                        pd.to_numeric(wk, errors="coerce")
                        .dropna()
                        .astype(int)
                        .sort_values()
                        .unique()
                        .tolist()
                    )
                return weeks

            # ---- Détection capacité YW
            yw_key_series = pd.to_numeric(df_idsr.get("YW_KEY"), errors="coerce")
            has_yw = ("YW_KEY" in df_idsr.columns) and yw_key_series.notna().any()

            # ---- Cas BRUT : on force le filtre Numéro de semaine (plus sûr en opérationnel)
            if is_brut:
                # st.info("Base IDS BRUTE détectée : filtre temporel par **Numéro de semaine**.")
                week_filter_mode = "WNUM"

                weeks = _get_weeks_list(df_idsr)

                if weeks:
                    col_min, col_max = st.columns(2)
                    with col_min:
                        w_min = st.selectbox(
                            "Semaine min (Numéro semaine)",
                            options=weeks,
                            index=0,
                            key="tab9_w_min",
                        )
                    with col_max:
                        w_max = st.selectbox(
                            "Semaine max (Numéro semaine)",
                            options=weeks,
                            index=len(weeks) - 1,
                            key="tab9_w_max",
                        )

                    if weeks.index(w_min) > weeks.index(w_max):
                        w_min, w_max = w_max, w_min
                else:
                    st.warning("Aucune semaine exploitable (Num_semaine_epid / Semaine_epid).")
                    week_filter_mode = None

            # ---- Cas COMPILÉ : proposer Année-Semaine (YW) si dispo, sinon Numéro de semaine
            else:
                week_filter_mode = None

                if has_yw:
                    # Mode normal : Année+Semaine
                    yw_table = df_idsr[["YW", "YW_KEY"]].copy()
                    yw_table["YW_KEY"] = pd.to_numeric(yw_table["YW_KEY"], errors="coerce")
                    yw_table = yw_table.dropna().drop_duplicates().sort_values("YW_KEY")

                    yws = yw_table["YW"].astype(str).tolist()
                    if yws:
                        col_min, col_max = st.columns(2)
                        with col_min:
                            yw_min = st.selectbox(
                                "Semaine min (Année-Semaine)",
                                options=yws,
                                index=0,
                                key="tab9_yw_min",
                            )
                        with col_max:
                            yw_max = st.selectbox(
                                "Semaine max (Année-Semaine)",
                                options=yws,
                                index=len(yws) - 1,
                                key="tab9_yw_max",
                            )

                        if yws.index(yw_min) > yws.index(yw_max):
                            yw_min, yw_max = yw_max, yw_min

                        min_key = float(yw_table.loc[yw_table["YW"] == yw_min, "YW_KEY"].iloc[0])
                        max_key = float(yw_table.loc[yw_table["YW"] == yw_max, "YW_KEY"].iloc[0])
                        week_filter_mode = "YW"

                # Fallback / option : Numéro de semaine (toujours utile)
                weeks = _get_weeks_list(df_idsr)
                if weeks:
                    col_min, col_max = st.columns(2)
                    with col_min:
                        w_min = st.selectbox(
                            "Semaine min (Numéro semaine)",
                            options=weeks,
                            index=0,
                            key="tab9_w_min",
                        )
                    with col_max:
                        w_max = st.selectbox(
                            "Semaine max (Numéro semaine)",
                            options=weeks,
                            index=len(weeks) - 1,
                            key="tab9_w_max",
                        )

                    if weeks.index(w_min) > weeks.index(w_max):
                        w_min, w_max = w_max, w_min

                    if week_filter_mode is None:
                        week_filter_mode = "WNUM"
                else:
                    if week_filter_mode is None:
                        st.warning("Aucune semaine exploitable (YW_KEY / Num_semaine_epid).")
                        week_filter_mode = None

        # 8) Appliquer filtres
        # ---------------------------------------------------------------------
        df9 = df_idsr.copy()

        if mal_sel and COL_MAL in df9.columns:
            df9 = df9[df9[COL_MAL].isin(mal_sel)]

        if prov_sel and COL_PROV_ID in df9.columns:
            df9 = df9[df9[COL_PROV_ID].isin(prov_sel)]

        if zs_sel and COL_ZS_ID in df9.columns:
            df9 = df9[df9[COL_ZS_ID].isin(zs_sel)]

        # Filtre Année (DEBUTSEM) si sélection disponible
        if years_selected and ("DEBUTSEM" in df9.columns):
            _debutsem = df9["DEBUTSEM"]
            if pd.api.types.is_numeric_dtype(_debutsem):
                _debutsem_dt = pd.to_datetime(_debutsem, unit="D", origin="1899-12-30", errors="coerce")
            else:
                _debutsem_dt = pd.to_datetime(_debutsem, errors="coerce")
            _yrs = _debutsem_dt.dt.year
            df9 = df9[_yrs.isin([int(y) for y in years_selected])]
        elif years_selected and ("Annee_epid" in df9.columns):
            # fallback si DEBUTSEM absent
            df9 = df9[pd.to_numeric(df9["Annee_epid"], errors="coerce").isin([int(y) for y in years_selected])]


        # Filtre semaines selon mode
        # Copie avant filtre semaines: utile pour 'Situation – dernière semaine' (focus sur semaine max)
        df9_base = df9.copy()

        # Filtre semaines selon mode
        if week_filter_mode == "YW":
            df9["YW_KEY"] = pd.to_numeric(df9["YW_KEY"], errors="coerce")
            df9 = df9[df9["YW_KEY"].between(min_key, max_key, inclusive="both")]

        elif week_filter_mode == "WNUM":
            df9["Num_semaine_epid"] = pd.to_numeric(df9["Num_semaine_epid"], errors="coerce")
            df9 = df9[df9["Num_semaine_epid"].between(w_min, w_max, inclusive="both")]

        st.caption(f"📌 Périmètre filtré : {len(df9):,} lignes")
        # -------------------------------------------------------------
        # Plusieurs années en mode WNUM → pas de deltas interprétables
        # -------------------------------------------------------------
        disable_deltas = False
        if week_filter_mode == "WNUM" and "Annee_epid" in df9.columns:
            _yrs_scope = pd.to_numeric(df9["Annee_epid"], errors="coerce").dropna().unique().tolist()
            if len(_yrs_scope) > 1:
                disable_deltas = True
                st.info(
                    "ℹ️ Plusieurs années détectées (mode BRUT / WNUM). "
                    "Les variations vs semaine-1 sont désactivées."
                )



        # ---------------------------------------------------------------------
        # 8.b) Résumé – période filtrée (confort utilisateur)
        # ---------------------------------------------------------------------
        if not df9.empty:
            _tot_cas = pd.to_numeric(df9.get("Total_cas"), errors="coerce").sum(skipna=True) if "Total_cas" in df9.columns else np.nan
            _tot_dec = pd.to_numeric(df9.get("Total_deces"), errors="coerce").sum(skipna=True) if "Total_deces" in df9.columns else np.nan
            _cfr = (float(_tot_dec) / float(_tot_cas) * 100.0) if (pd.notna(_tot_cas) and _tot_cas > 0 and pd.notna(_tot_dec)) else np.nan

            _n_prov = df9[COL_PROV_ID].nunique(dropna=True) if COL_PROV_ID in df9.columns else 0
            _n_zs = df9[COL_ZS_ID].nunique(dropna=True) if COL_ZS_ID in df9.columns else 0

            st.markdown("### Résumé – période filtrée")
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Cas (total)", f"{int(_tot_cas):,}" if pd.notna(_tot_cas) else "NA")
            r2.metric("Décès (total)", f"{int(_tot_dec):,}" if pd.notna(_tot_dec) else "NA")
            r3.metric("CFR (recalculé)", f"{_cfr:.2f}%" if pd.notna(_cfr) else "NA")
            r4.metric("Provinces", f"{_n_prov:,}")
            r5.metric("Zones de santé", f"{_n_zs:,}")


        if df9.empty:
            st.info("Aucune donnée après filtrage.")
        else:
            st.divider()

            # -----------------------------------------------------------------
            # 9) Série temporelle : cas/décès/CFR (robuste sur TIME_KEY/LAB)
            # -----------------------------------------------------------------
            required_cols = ["Total_cas", "Total_deces"]
            missing = [c for c in required_cols if c not in df9.columns]
            if missing:
                st.error(f"Colonnes manquantes pour l'analyse temporelle : {', '.join(missing)}")
            else:
                # Agrégation hebdo
                weekly = df9.groupby(["TIME_LAB", "TIME_KEY"], as_index=False).agg(
                    Cas=("Total_cas", "sum"),
                    Deces=("Total_deces", "sum"),
                    Taux_letalite_moy=("Taux_letalite", "mean") if "Taux_letalite" in df9.columns else ("Total_cas", "size"),
                    Taux_attaque_moy=("Taux_attaque", "mean") if "Taux_attaque" in df9.columns else ("Total_cas", "size"),
                )

                # CFR recalculé (en %) : plus fiable que moyenne LETAL
                weekly["CFR_recalc_pct"] = np.where(
                    weekly["Cas"] > 0,
                    (weekly["Deces"] / weekly["Cas"]) * 100.0,
                    np.nan
                )

                # si LETAL existe, garder une version en % (supposée déjà en %)
                if "Taux_letalite_moy" in weekly.columns:
                    weekly["LETAL_moy_pct"] = weekly["Taux_letalite_moy"]
                else:
                    weekly["LETAL_moy_pct"] = np.nan

                # si taux_ n'existent pas, on met NaN
                if "Taux_letalite" not in df9.columns:
                    weekly["Taux_letalite_moy"] = np.nan
                if "Taux_attaque" not in df9.columns:
                    weekly["Taux_attaque_moy"] = np.nan

                weekly["CFR_calc_%"] = np.where(
                    weekly["Cas"] > 0, (weekly["Deces"] / weekly["Cas"]) * 100, np.nan
                )

                weekly_sorted = weekly.sort_values("TIME_KEY").reset_index(drop=True)

                # -------------------------------------------------------------
                # 9.b) Comparaison "Tranches d’âge" vs "Totaux" (visualisation)
                # -------------------------------------------------------------
                # Objectif : afficher 2 lignes de KPI (Cas/Décès/CFR) :
                # - Ligne 1 : Somme tranches d’âge (Cas_* / Deces_*) => détecte incohérences
                # - Ligne 2 : Totaux (Total_cas / Total_deces) => référence opérationnelle

                age_case_cols = [c for c in ["Cas_tnn", "Cas_0_11mois", "Cas_12_59mois", "Cas_5_14ans", "Cas_15plus"] if c in df9.columns]
                age_death_cols = [c for c in ["Deces_tnn", "Deces_0_11mois", "Deces_12_59mois", "Deces_5_14ans", "Deces_15plus"] if c in df9.columns]

                weekly_age_sorted = None
                if age_case_cols or age_death_cols:
                    _tmp = df9.copy()

                    if age_case_cols:
                        _tmp["Cas_age_sum"] = _tmp[age_case_cols].sum(axis=1, min_count=1)
                    else:
                        _tmp["Cas_age_sum"] = np.nan

                    if age_death_cols:
                        _tmp["Deces_age_sum"] = _tmp[age_death_cols].sum(axis=1, min_count=1)
                    else:
                        _tmp["Deces_age_sum"] = np.nan

                    weekly_age = _tmp.groupby(["TIME_LAB", "TIME_KEY"], as_index=False).agg(
                        Cas=("Cas_age_sum", "sum"),
                        Deces=("Deces_age_sum", "sum"),
                    )
                    weekly_age["CFR_calc_%"] = np.where(
                        weekly_age["Cas"] > 0, (weekly_age["Deces"] / weekly_age["Cas"]) * 100, np.nan
                    )
                    weekly_age_sorted = weekly_age.sort_values("TIME_KEY").reset_index(drop=True)
                # KPI dernière semaine + variati# KPI dernière semaine (focus sur semaine max) + variation vs semaine-1
                def pct_change(cur, prv):
                    if prv is None or pd.isna(prv) or prv == 0 or pd.isna(cur):
                        return None
                    return (cur - prv) / prv * 100

                last = None
                prev = None

                # On calcule sur df9_base (filtres maladie/province/ZS), sans dépendre de week_min.
                if "df9_base" in locals() and not df9_base.empty:
                    if week_filter_mode == "YW" and "YW_KEY" in df9_base.columns:
                        _b = df9_base.copy()
                        _b["YW_KEY"] = pd.to_numeric(_b["YW_KEY"], errors="coerce")
                        last_key = max_key if max_key is not None else _b["YW_KEY"].dropna().max()

                        df_last_kpi = _b[_b["YW_KEY"] == last_key]
                        keys = _b["YW_KEY"].dropna().drop_duplicates().sort_values().tolist()
                        prev_key = keys[-2] if len(keys) >= 2 else None
                        df_prev_kpi = _b[_b["YW_KEY"] == prev_key] if prev_key is not None else pd.DataFrame()

                        cas_last = pd.to_numeric(df_last_kpi.get("Total_cas"), errors="coerce").sum(skipna=True)
                        dec_last = pd.to_numeric(df_last_kpi.get("Total_deces"), errors="coerce").sum(skipna=True)
                        cfr_last = (float(dec_last) / float(cas_last) * 100.0) if (pd.notna(cas_last) and cas_last > 0 and pd.notna(dec_last)) else np.nan

                        cas_prev = pd.to_numeric(df_prev_kpi.get("Total_cas"), errors="coerce").sum(skipna=True) if not df_prev_kpi.empty else np.nan
                        dec_prev = pd.to_numeric(df_prev_kpi.get("Total_deces"), errors="coerce").sum(skipna=True) if not df_prev_kpi.empty else np.nan
                        cfr_prev = (float(dec_prev) / float(cas_prev) * 100.0) if (pd.notna(cas_prev) and cas_prev > 0 and pd.notna(dec_prev)) else np.nan

                        lab_last = df_last_kpi["TIME_LAB"].iloc[0] if ("TIME_LAB" in df_last_kpi.columns and not df_last_kpi.empty) else str(int(last_key) if pd.notna(last_key) else "NA")

                        last = {"TIME_LAB": lab_last, "Cas": cas_last, "Deces": dec_last, "CFR_calc_%": cfr_last}
                        prev = {"Cas": cas_prev, "Deces": dec_prev, "CFR_calc_%": cfr_prev} if not df_prev_kpi.empty else None

                    elif week_filter_mode == "WNUM" and "Num_semaine_epid" in df9_base.columns and "Annee_epid" in df9_base.columns:
                        _b = df9_base.copy()
                        _b["Num_semaine_epid"] = pd.to_numeric(_b["Num_semaine_epid"], errors="coerce")
                        _b["Annee_epid"] = pd.to_numeric(_b["Annee_epid"], errors="coerce")

                        year_candidates = _b.loc[_b["Num_semaine_epid"] == w_max, "Annee_epid"].dropna()
                        last_year = int(year_candidates.max()) if not year_candidates.empty else None

                        df_last_kpi = _b[(_b["Annee_epid"] == last_year) & (_b["Num_semaine_epid"] == w_max)] if last_year is not None else pd.DataFrame()
                        if last_year is not None and int(w_max) > 1:
                            df_prev_kpi = _b[(_b["Annee_epid"] == last_year) & (_b["Num_semaine_epid"] == (int(w_max) - 1))]
                        elif last_year is not None:
                            df_prev_kpi = _b[(_b["Annee_epid"] == (last_year - 1)) & (_b["Num_semaine_epid"].isin([52, 53]))]
                            if not df_prev_kpi.empty:
                                prev_week_num = int(df_prev_kpi["Num_semaine_epid"].max())
                                df_prev_kpi = df_prev_kpi[df_prev_kpi["Num_semaine_epid"] == prev_week_num]
                        else:
                            df_prev_kpi = pd.DataFrame()

                        if not df_last_kpi.empty:
                            cas_last = pd.to_numeric(df_last_kpi.get("Total_cas"), errors="coerce").sum(skipna=True)
                            dec_last = pd.to_numeric(df_last_kpi.get("Total_deces"), errors="coerce").sum(skipna=True)
                            cfr_last = (float(dec_last) / float(cas_last) * 100.0) if (pd.notna(cas_last) and cas_last > 0 and pd.notna(dec_last)) else np.nan

                            cas_prev = pd.to_numeric(df_prev_kpi.get("Total_cas"), errors="coerce").sum(skipna=True) if not df_prev_kpi.empty else np.nan
                            dec_prev = pd.to_numeric(df_prev_kpi.get("Total_deces"), errors="coerce").sum(skipna=True) if not df_prev_kpi.empty else np.nan
                            cfr_prev = (float(dec_prev) / float(cas_prev) * 100.0) if (pd.notna(cas_prev) and cas_prev > 0 and pd.notna(dec_prev)) else np.nan

                            last = {"TIME_LAB": f"W{int(w_max):02d}", "Cas": cas_last, "Deces": dec_last, "CFR_calc_%": cfr_last}
                            prev = {"Cas": cas_prev, "Deces": dec_prev, "CFR_calc_%": cfr_prev} if not df_prev_kpi.empty else None

                # Fallback: si on n'a rien trouvé, on garde l'ancien comportement
                if last is None and len(weekly_sorted) >= 1:
                    last = weekly_sorted.iloc[-1]
                    prev = weekly_sorted.iloc[-2] if len(weekly_sorted) >= 2 else None

                d_cas = None if ("disable_deltas" in locals() and disable_deltas) else (pct_change(last["Cas"], prev["Cas"]) if (last is not None and prev is not None) else None)
                d_dec = None if ("disable_deltas" in locals() and disable_deltas) else (pct_change(last["Deces"], prev["Deces"]) if (last is not None and prev is not None) else None)
                d_cfr = None if ("disable_deltas" in locals() and disable_deltas) else (pct_change(last["CFR_calc_%"], prev["CFR_calc_%"]) if (last is not None and prev is not None) else None)

                st.markdown("### Situation – dernière semaine")

                # Préparer la série "tranches d’âge" (Cas_* / Deces_*) pour comparer avec les totaux
                age_case_cols = [c for c in ["Cas_tnn", "Cas_0_11mois", "Cas_12_59mois", "Cas_5_14ans", "Cas_15plus"] if c in df9.columns]
                age_death_cols = [c for c in ["Deces_tnn", "Deces_0_11mois", "Deces_12_59mois", "Deces_5_14ans", "Deces_15plus"] if c in df9.columns]

                weekly_age_sorted = None
                if age_case_cols and age_death_cols:
                    _tmp = df9.copy()
                    _tmp["Cas_age_sum"] = _tmp[age_case_cols].sum(axis=1, skipna=True)
                    _tmp["Deces_age_sum"] = _tmp[age_death_cols].sum(axis=1, skipna=True)

                    weekly_age = _tmp.groupby(["TIME_LAB", "TIME_KEY"], as_index=False).agg(
                        Cas_age=("Cas_age_sum", "sum"),
                        Deces_age=("Deces_age_sum", "sum"),
                    )
                    weekly_age["CFR_age_%"] = np.where(
                        weekly_age["Cas_age"] > 0,
                        (weekly_age["Deces_age"] / weekly_age["Cas_age"]) * 100,
                        np.nan
                    )
                    weekly_age_sorted = weekly_age.sort_values("TIME_KEY").reset_index(drop=True)

                def pct_change(cur, prv):
                    if prv is None or pd.isna(prv) or prv == 0 or pd.isna(cur):
                        return None
                    return (cur - prv) / prv * 100

                # Ligne 1 — Somme tranches d’âge (Cas_* / Deces_*) — focus sur la semaine max
                df_last_week = pd.DataFrame()
                df_prev_week = pd.DataFrame()
                last_lab_focus = None

                if "df9_base" in locals() and not df9_base.empty:
                    # 1) Déterminer la "dernière semaine" = borne haute du filtre (w_max ou max_key)
                    if week_filter_mode == "YW" and "YW_KEY" in df9_base.columns:
                        _base = df9_base.copy()
                        _base["YW_KEY"] = pd.to_numeric(_base["YW_KEY"], errors="coerce")

                        last_key = max_key if max_key is not None else _base["YW_KEY"].dropna().max()

                        df_last_week = _base[_base["YW_KEY"] == last_key]  # focus semaine max (YW)
                        uniq_keys = (
                            _base["YW_KEY"].dropna().drop_duplicates().sort_values().tolist()
                        )
                        prev_key = uniq_keys[-2] if len(uniq_keys) >= 2 else None
                        df_prev_week = _base[_base["YW_KEY"] == prev_key] if prev_key is not None else pd.DataFrame()

                        if not df_last_week.empty:
                            last_lab_focus = (
                                df_last_week["TIME_LAB"].iloc[0]
                                if "TIME_LAB" in df_last_week.columns
                                else str(int(max_key) if pd.notna(max_key) else "NA")
                            )

                    elif week_filter_mode == "WNUM" and "Num_semaine_epid" in df9_base.columns:
                        _base = df9_base.copy()
                        _base["Num_semaine_epid"] = pd.to_numeric(_base["Num_semaine_epid"], errors="coerce")
                        if "Annee_epid" in _base.columns:
                            _base["Annee_epid"] = pd.to_numeric(_base["Annee_epid"], errors="coerce")

                            # choisir l'année la plus récente qui contient la semaine w_max
                            year_candidates = _base.loc[_base["Num_semaine_epid"] == w_max, "Annee_epid"].dropna()
                            last_year = int(year_candidates.max()) if not year_candidates.empty else None

                            if last_year is not None:
                                df_last_week = _base[(_base["Annee_epid"] == last_year) & (_base["Num_semaine_epid"] == w_max)]

                                # semaine précédente (dans la même année si possible)
                                if int(w_max) > 1:
                                    df_prev_week = _base[(_base["Annee_epid"] == last_year) & (_base["Num_semaine_epid"] == (int(w_max) - 1))]
                                else:
                                    # Si w_max == 1 : chercher semaine 52/53 de l'année précédente
                                    df_prev_week = _base[(_base["Annee_epid"] == (last_year - 1)) & (_base["Num_semaine_epid"].isin([52, 53]))]
                                    if not df_prev_week.empty:
                                        prev_week_num = int(df_prev_week["Num_semaine_epid"].max())
                                        df_prev_week = df_prev_week[df_prev_week["Num_semaine_epid"] == prev_week_num]

                                last_lab_focus = f"W{int(w_max):02d}"

                    # Fallback si on n'a pas réussi à isoler la semaine max
                    if df_last_week.empty:
                        df_last_week = df9.copy()
                        df_prev_week = pd.DataFrame()
                        last_lab_focus = df_last_week["TIME_LAB"].iloc[0] if ("TIME_LAB" in df_last_week.columns and not df_last_week.empty) else "NA"

                # Note opérationnelle : si plusieurs années sont incluses en mode WNUM,
                # les deltas vs semaine-1 sont désactivés (comparaison non interprétable).
                if week_filter_mode == "WNUM" and "Annee_epid" in df_last_week.columns:
                    _yrs = pd.to_numeric(df_last_week["Annee_epid"], errors="coerce").dropna().unique().tolist()
                    if len(_yrs) > 1:
                        st.info("ℹ️ Plusieurs années détectées pour cette semaine (mode BRUT / WNUM) : les variations vs semaine-1 sont désactivées.")
                        disable_deltas = True

                # 2) Affichage métriques "tranches d'âge" pour la semaine max
                if not df_last_week.empty and age_case_cols and age_death_cols:
                    cas_age_last = df_last_week[age_case_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0, skipna=True).sum()
                    dec_age_last = df_last_week[age_death_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0, skipna=True).sum()
                    cfr_age_last = (float(dec_age_last) / float(cas_age_last) * 100.0) if (pd.notna(cas_age_last) and cas_age_last > 0 and pd.notna(dec_age_last)) else np.nan

                    cas_age_prev = df_prev_week[age_case_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0, skipna=True).sum() if (not df_prev_week.empty) else np.nan
                    dec_age_prev = df_prev_week[age_death_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0, skipna=True).sum() if (not df_prev_week.empty) else np.nan
                    cfr_age_prev = (float(dec_age_prev) / float(cas_age_prev) * 100.0) if (pd.notna(cas_age_prev) and cas_age_prev > 0 and pd.notna(dec_age_prev)) else np.nan

                    d_cas_a = None if ("disable_deltas" in locals() and disable_deltas) else pct_change(cas_age_last, cas_age_prev)
                    d_dec_a = None if ("disable_deltas" in locals() and disable_deltas) else pct_change(dec_age_last, dec_age_prev)
                    d_cfr_a = None if ("disable_deltas" in locals() and disable_deltas) else pct_change(cfr_age_last, cfr_age_prev)

                    st.caption("Ligne 1 : Somme tranches d’âge (Cas_* / Deces_*)")
                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("Semaine", str(last_lab_focus))
                    a2.metric("Cas (tranches)", f"{int(cas_age_last):,}" if pd.notna(cas_age_last) else "NA", delta=None if d_cas_a is None else f"{d_cas_a:.1f}% vs semaine-1")
                    a3.metric("Décès (tranches)", f"{int(dec_age_last):,}" if pd.notna(dec_age_last) else "NA", delta=None if d_dec_a is None else f"{d_dec_a:.1f}% vs semaine-1")
                    a4.metric("CFR (tranches)", f"{cfr_age_last:.2f}%" if pd.notna(cfr_age_last) else "NA", delta=None if d_cfr_a is None else f"{d_cfr_a:.1f}% vs semaine-1")
                else:
                    st.caption("Ligne 1 : Somme tranches d’âge (Cas_* / Deces_*) — indisponible (colonnes manquantes ou aucune donnée)")

                
                # -----------------------------------------------------------------
                # Ligne 2 — Totaux (TOTALCAS / TOTALDECES) — focus sur la semaine max
                # Objectif: comparer directement avec la Ligne 1 (sommes des tranches d’âge)
                # -----------------------------------------------------------------
                tot_cas_lastwk = pd.to_numeric(df_last_week.get("Total_cas"), errors="coerce").sum(skipna=True) if (("Total_cas" in df9.columns) and (not df_last_week.empty)) else np.nan
                tot_dec_lastwk = pd.to_numeric(df_last_week.get("Total_deces"), errors="coerce").sum(skipna=True) if (("Total_deces" in df9.columns) and (not df_last_week.empty)) else np.nan
                cfr_tot_lastwk = (float(tot_dec_lastwk) / float(tot_cas_lastwk) * 100.0) if (pd.notna(tot_cas_lastwk) and tot_cas_lastwk > 0 and pd.notna(tot_dec_lastwk)) else np.nan

                tot_cas_prevwk = pd.to_numeric(df_prev_week.get("Total_cas"), errors="coerce").sum(skipna=True) if (("Total_cas" in df9.columns) and (not df_prev_week.empty)) else np.nan
                tot_dec_prevwk = pd.to_numeric(df_prev_week.get("Total_deces"), errors="coerce").sum(skipna=True) if (("Total_deces" in df9.columns) and (not df_prev_week.empty)) else np.nan
                cfr_tot_prevwk = (float(tot_dec_prevwk) / float(tot_cas_prevwk) * 100.0) if (pd.notna(tot_cas_prevwk) and tot_cas_prevwk > 0 and pd.notna(tot_dec_prevwk)) else np.nan

                d_cas_t = None if ("disable_deltas" in locals() and disable_deltas) else pct_change(tot_cas_lastwk, tot_cas_prevwk)
                d_dec_t = None if ("disable_deltas" in locals() and disable_deltas) else pct_change(tot_dec_lastwk, tot_dec_prevwk)
                d_cfr_t = None if ("disable_deltas" in locals() and disable_deltas) else pct_change(cfr_tot_lastwk, cfr_tot_prevwk)

                st.caption("Ligne 2 : Totaux (TOTALCAS / TOTALDECES)")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Semaine", str(last_lab_focus) if last_lab_focus is not None else "NA")
                k2.metric("Cas (totaux)", f"{int(tot_cas_lastwk):,}" if pd.notna(tot_cas_lastwk) else "NA", delta=None if d_cas_t is None else f"{d_cas_t:.1f}% vs semaine-1")
                k3.metric("Décès (totaux)", f"{int(tot_dec_lastwk):,}" if pd.notna(tot_dec_lastwk) else "NA", delta=None if d_dec_t is None else f"{d_dec_t:.1f}% vs semaine-1")
                k4.metric("CFR (totaux)", f"{cfr_tot_lastwk:.2f}%" if pd.notna(cfr_tot_lastwk) else "NA", delta=None if d_cfr_t is None else f"{d_cfr_t:.1f}% vs semaine-1")

                # -----------------------------------------------------------------
                # Écarts Totaux vs Tranches (semaine max)
                # -----------------------------------------------------------------
                diff_cas = (tot_cas_lastwk - cas_age_last) if ("cas_age_last" in locals() and pd.notna(tot_cas_lastwk) and pd.notna(cas_age_last)) else np.nan
                diff_dec = (tot_dec_lastwk - dec_age_last) if ("dec_age_last" in locals() and pd.notna(tot_dec_lastwk) and pd.notna(dec_age_last)) else np.nan

                if pd.notna(diff_cas) and pd.notna(diff_dec):
                    if (diff_cas == 0) and (diff_dec == 0):
                        st.success("✅ Aucun écart : TOTALCAS/TOTALDECES = somme des tranches d’âge (semaine max).")
                    else:
                        pct_cas = (diff_cas / cas_age_last * 100.0) if ("cas_age_last" in locals() and pd.notna(cas_age_last) and cas_age_last != 0) else np.nan
                        pct_dec = (diff_dec / dec_age_last * 100.0) if ("dec_age_last" in locals() and pd.notna(dec_age_last) and dec_age_last != 0) else np.nan
                        st.error(
                            "❌ Écart détecté (Totaux − Tranches) – semaine max : "
                            f"Cas={diff_cas:+,} ({pct_cas:.1f}%) | Décès={diff_dec:+,} ({pct_dec:.1f}%)"
                        )
                else:
                    st.info("Écart non calculable (colonnes manquantes ou données insuffisantes).")

                # Note: cette section est volontairement centrée sur la semaine max,
                # même si l'utilisateur change semaine min.
                st.divider()
                with st.expander("### Qualité des dates (date vs semaine)", expanded=False):
                    if "QC_Date_vs_Semaine" in df9.columns:
                        st.write(df9["QC_Date_vs_Semaine"].value_counts(dropna=False))
                    else:
                        st.info("QC indisponible (pas de dates source).")

                # -----------------------------------------------------------------
                # 10) Signaux – Top en hausse (dernière semaine vs précédente)
                # -----------------------------------------------------------------
                with st.expander("### Signaux – Top en hausse (dernière semaine vs semaine précédente)", expanded=False):

                    if (COL_PROV_ID in df9.columns) and (len(weekly_sorted) >= 2):
                        last_t = weekly_sorted.iloc[-1]["TIME_LAB"]
                        prev_t = weekly_sorted.iloc[-2]["TIME_LAB"]

                        df_last = df9[df9["TIME_LAB"] == last_t]
                        df_prev = df9[df9["TIME_LAB"] == prev_t]

                        prov_last = df_last.groupby(COL_PROV_ID, as_index=False).agg(Cas=("Total_cas", "sum"))
                        prov_prev = df_prev.groupby(COL_PROV_ID, as_index=False).agg(Cas_prev=("Total_cas", "sum"))

                        prov_delta = prov_last.merge(prov_prev, on=COL_PROV_ID, how="outer").fillna(0)
                        prov_delta["Delta_cas"] = prov_delta["Cas"] - prov_delta["Cas_prev"]
                        prov_delta["Delta_%"] = np.where(
                            prov_delta["Cas_prev"] > 0,
                            (prov_delta["Delta_cas"] / prov_delta["Cas_prev"]) * 100,
                            np.nan
                        )

                        min_cases = st.slider(
                            "Seuil cas (dernière semaine) pour afficher",
                            0, 1000, 5, step=5, key="tab9_min_cases_up"
                        )
                        prov_delta = prov_delta[prov_delta["Cas"] >= min_cases].sort_values("Delta_cas", ascending=False)

                        with st.expander("📈 Top provinces en hausse (dérouler)", expanded=False):
                            n_up = st.slider("Nombre à afficher", 5, 50, 15, step=5, key="tab9_n_up_prov")
                            st.dataframe(prov_delta.head(n_up), width="stretch", height=420, hide_index=True)
                    else:
                        st.info("Top en hausse indisponible (colonne Province ou semaines insuffisantes).")

                    # -----------------------------------------------------------------
                    # 11) Top provinces / ZS sur la période
                    # -----------------------------------------------------------------
                    c3, c4 = st.columns(2)

                    with c3:
                        if COL_PROV_ID in df9.columns and "Total_cas" in df9.columns and "Total_deces" in df9.columns:
                            top_prov = df9.groupby(COL_PROV_ID, as_index=False).agg(
                                Cas=("Total_cas", "sum"),
                                Deces=("Total_deces", "sum")
                            )
                            top_prov["CFR_%"] = np.where(top_prov["Cas"] > 0, (top_prov["Deces"] / top_prov["Cas"]) * 100, np.nan)
                            top_prov = top_prov.sort_values("Cas", ascending=False)

                            with st.expander("🏥 Top provinces (dérouler)", expanded=False):
                                n_prov = st.slider("Nombre de provinces à afficher", 10, 200, 20, step=10, key="tab9_n_top_prov")
                                st.dataframe(top_prov.head(n_prov), width="stretch", height=420, hide_index=True)
                        else:
                            top_prov = None
                            st.info("Top provinces indisponible (colonnes manquantes).")

                    with c4:
                        if (COL_PROV_ID in df9.columns) and (COL_ZS_ID in df9.columns) and ("Total_cas" in df9.columns) and ("Total_deces" in df9.columns):
                            top_zs = df9.groupby([COL_PROV_ID, COL_ZS_ID], as_index=False).agg(
                                Cas=("Total_cas", "sum"),
                                Deces=("Total_deces", "sum")
                            )
                            top_zs["CFR_%"] = np.where(top_zs["Cas"] > 0, (top_zs["Deces"] / top_zs["Cas"]) * 100, np.nan)
                            top_zs = top_zs.sort_values("Cas", ascending=False)

                            with st.expander("🗺️ Top zones de santé (dérouler)", expanded=False):
                                n_zs = st.slider("Nombre de ZS à afficher", 10, 300, 20, step=10, key="tab9_n_top_zs")
                                st.dataframe(top_zs.head(n_zs), width="stretch", height=420, hide_index=True)
                        else:
                            top_zs = None
                            st.info("Top ZS indisponible (colonnes manquantes).")

            # -----------------------------------------------------------------
            # 12) Contrôles cohérence totaux vs tranches d’âge
            # -----------------------------------------------------------------
            with st.expander("### Contrôle cohérence des totaux (tranches d’âge vs Total)", expanded=False):
                show_qc_tables = st.checkbox(
                    "Afficher les tableaux détaillés QC (peut être lourd)",
                    value=False,
                    key="tab9_show_qc_tables"
                )


                age_case_cols = [c for c in ["Cas_tnn", "Cas_0_11mois", "Cas_12_59mois", "Cas_5_14ans", "Cas_15plus"] if c in df9.columns]
                age_death_cols = [c for c in ["Deces_tnn", "Deces_0_11mois", "Deces_12_59mois", "Deces_5_14ans", "Deces_15plus"] if c in df9.columns]

                qc = df9.copy()

                if age_case_cols and "Total_cas" in qc.columns:
                    qc["sum_cas_age"] = qc[age_case_cols].sum(axis=1, skipna=True)
                    qc["diff_cas"] = qc["Total_cas"] - qc["sum_cas_age"]

                if age_death_cols and "Total_deces" in qc.columns:
                    qc["sum_deces_age"] = qc[age_death_cols].sum(axis=1, skipna=True)
                    qc["diff_deces"] = qc["Total_deces"] - qc["sum_deces_age"]

                qc_view = qc.copy()
                qc_view["QC_Cas"] = np.where(qc_view.get("diff_cas", 0).fillna(0) == 0, "✅ OK", "❌ KO") if "diff_cas" in qc_view.columns else "NA"
                qc_view["QC_Deces"] = np.where(qc_view.get("diff_deces", 0).fillna(0) == 0, "✅ OK", "❌ KO") if "diff_deces" in qc_view.columns else "NA"

                if ("diff_cas" in qc_view.columns) and ("diff_deces" in qc_view.columns):
                    qc_view["QC_Global"] = np.where(
                        (qc_view["diff_cas"].fillna(0) == 0) & (qc_view["diff_deces"].fillna(0) == 0),
                        "✅ OK", "❌ KO"
                    )
                elif "diff_cas" in qc_view.columns:
                    qc_view["QC_Global"] = np.where(qc_view["diff_cas"].fillna(0) == 0, "✅ OK", "❌ KO")
                elif "diff_deces" in qc_view.columns:
                    qc_view["QC_Global"] = np.where(qc_view["diff_deces"].fillna(0) == 0, "✅ OK", "❌ KO")
                else:
                    qc_view["QC_Global"] = "NA"

                # Colonnes QC à afficher
                cols_show = [c for c in [
                    "TIME_LAB", "TIME_KEY", "Date_debut_semaine_iso",
                    COL_MAL, COL_PROV_ID, COL_ZS_ID,
                    "Total_cas", "sum_cas_age", "diff_cas",
                    "Total_deces", "sum_deces_age", "diff_deces"
                ] if c in qc_view.columns]

                def style_qc(row):
                    """Style cellule: surligner seulement les écarts et QC_Global."""
                    styles = [""] * len(row)
                    cols = list(row.index)

                    def set_cell(col, bg=None, fg=None, weight=None):
                        if col in cols:
                            i = cols.index(col)
                            css = []
                            if bg is not None:
                                css.append(f"background-color: {bg}")
                            if fg is not None:
                                css.append(f"color: {fg}")
                            if weight is not None:
                                css.append(f"font-weight: {weight}")
                            styles[i] = "; ".join(css)

                    if row.get("diff_cas", 0) != 0:
                        set_cell("diff_cas", bg="#fff3cd", fg="#111", weight="700")
                    if row.get("diff_deces", 0) != 0:
                        set_cell("diff_deces", bg="#ffe5e5", fg="#111", weight="700")

                    if row.get("QC_Global") == "❌ KO":
                        set_cell("QC_Global", bg="#f2f2f2", fg="#111", weight="700")
                    else:
                        set_cell("QC_Global", fg="#111", weight="700")

                    if "QC_Cas" in cols:
                        set_cell("QC_Cas", fg="#111", weight="700")
                    if "QC_Deces" in cols:
                        set_cell("QC_Deces", fg="#111", weight="700")

                    return styles

                # Filtres QC
                st.markdown("#### Filtres QC")
                f1, f2, f3, f4 = st.columns(4)

                with f1:
                    qc_global_sel = st.selectbox("QC global", options=["Tous", "✅ OK", "❌ KO"], index=0, key="tab9_qc_global_sel")
                with f2:
                    qc_cas_sel = st.selectbox("QC cas", options=["Tous", "✅ OK", "❌ KO"], index=0, key="tab9_qc_cas_sel")
                with f3:
                    qc_deces_sel = st.selectbox("QC décès", options=["Tous", "✅ OK", "❌ KO"], index=0, key="tab9_qc_deces_sel")
                with f4:
                    
                    abs_diff_min = st.number_input(
                        "|diff| minimum",
                        min_value=0,
                        value=0,
                        step=1,
                        help="Filtre sur l'écart absolu (cas ou décès). Mets 1 pour exclure les diff = 0.",
                        key="tab9_qc_abs_diff_min"
                    )

                show_all = st.checkbox(
                    "Afficher toutes les lignes (sinon seulement incohérences)",
                    value=False,
                    key="tab9_qc_show_all"
                )

                # Base: toutes lignes vs seulement incohérences
                if show_all:
                    base_tbl = qc_view.copy()
                else:
                    base_tbl = qc_view.copy()
                    if "diff_cas" in base_tbl.columns:
                        base_tbl = base_tbl[base_tbl["diff_cas"].fillna(0) != 0]
                    if "diff_deces" in base_tbl.columns:
                        base_tbl = base_tbl[base_tbl["diff_deces"].fillna(0) != 0]

                # Appliquer filtres
                table_to_show = base_tbl.copy()

                if qc_global_sel != "Tous" and "QC_Global" in table_to_show.columns:
                    table_to_show = table_to_show[table_to_show["QC_Global"] == qc_global_sel]

                if qc_cas_sel != "Tous" and "QC_Cas" in table_to_show.columns:
                    table_to_show = table_to_show[table_to_show["QC_Cas"] == qc_cas_sel]

                if qc_deces_sel != "Tous" and "QC_Deces" in table_to_show.columns:
                    table_to_show = table_to_show[table_to_show["QC_Deces"] == qc_deces_sel]

                # Seuil sur diff
                if abs_diff_min > 0:
                    cond = False
                    if "diff_cas" in table_to_show.columns:
                        cond = cond | (table_to_show["diff_cas"].fillna(0).abs() >= abs_diff_min)
                    if "diff_deces" in table_to_show.columns:
                        cond = cond | (table_to_show["diff_deces"].fillna(0).abs() >= abs_diff_min)
                    table_to_show = table_to_show[cond]

                st.caption(f"📌 Lignes après filtres QC : {len(table_to_show):,}")

                # Colonnes QC à afficher
                qc_cols = ["QC_Global", "QC_Cas", "QC_Deces"]
                qc_cols = [c for c in qc_cols if c in table_to_show.columns]
                final_cols = qc_cols + cols_show

                if show_qc_tables:
                    with st.expander("🧾 Tableau QC (OK/KO) – cas & décès (dérouler)", expanded=False):
                        st.dataframe(
                        table_to_show[final_cols].style.apply(style_qc, axis=1),
                        width="stretch",
                        height=520,
                        hide_index=True
                    )
                
            # ---------------------------------------------------------------------
            # 14) IDSR – Spécifications des sorties
            # ---------------------------------------------------------------------
            # 14.1) Histogramme des cas + courbe de létalité (CFR%)
            with st.expander("📈 Histogramme des cas avec courbe de létalité (par semaine)", expanded=True):
                if 'weekly_sorted' in locals() and isinstance(weekly_sorted, pd.DataFrame) and not weekly_sorted.empty:
                    _wk = weekly_sorted.copy()
                    # Libellé unique Année-Semaine (évite doublons W01/W02 quand plusieurs années)
                    if "YW" in _wk.columns:
                        _wk["_X_LAB"] = _wk["YW"].astype(str)
                    elif "TIME_KEY" in _wk.columns:
                        _wk["_X_LAB"] = _wk["TIME_KEY"].astype(str)
                    else:
                        _wk["_X_LAB"] = _wk.get("TIME_LAB", pd.Series(dtype="object")).astype(str)
                    # (_fmt_yw_label est centralisée en haut du script)
                    _wk["_X_LAB"] = _wk["_X_LAB"].map(_fmt_yw_label)

                    # Sécurité sur colonnes
                    if ("_X_LAB" in _wk.columns) and ("Cas" in _wk.columns):
                        _wk["CFR_calc_%"] = pd.to_numeric(_wk.get("CFR_calc_%"), errors="coerce").astype(float)
                        # Plotly n'accepte pas pd.NA (NAType) -> forcer np.nan
                        _wk = _wk.replace({pd.NA: np.nan})

                        # Texte CFR (évite "NA %" et évite pd.NA)
                        _wk["_cfr_text"] = _wk["CFR_calc_%"].map(lambda x: "" if pd.isna(x) else f"{x:.2f} %")

                        fig_cas_cfr = go.Figure()
                        fig_cas_cfr.add_trace(go.Bar(
                            x=_wk["_X_LAB"],
                            y=pd.to_numeric(_wk["Cas"], errors="coerce").fillna(0).astype(float),
                            name="Cas",
                            yaxis="y1",
                        ))
                        fig_cas_cfr.add_trace(go.Scatter(
                            x=_wk["_X_LAB"],
                            y=_wk["CFR_calc_%"].astype(float),
                            name="Létalité (CFR%)",
                            mode="lines+markers+text",
                            yaxis="y2",
                            text=_wk["_cfr_text"],
                            textposition="top center",
                        ))
                        fig_cas_cfr.update_layout(
                            template="plotly_white",
                            xaxis_title="Semaine épidémiologique",
                            yaxis=dict(title="Nombre de cas"),
                            yaxis2=dict(title="Létalité (%)", overlaying="y", side="right", rangemode="tozero"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                            margin=dict(t=70, b=60, l=60, r=60),
                            height=420,
                        )
                        fig_cas_cfr = apply_plotly_value_annotations(fig_cas_cfr, annot_vals)
                        st.plotly_chart(fig_cas_cfr, width="stretch", key="idsr_hist_cas_cfr")
                    else:
                        st.info("Colonnes insuffisantes pour tracer l'évolution hebdomadaire (TIME_LAB/Cas).")
                else:
                    st.info("Aucune donnée agrégée par semaine disponible après filtrage.")

            # 14.2) Camembert par tranche d'âge + tableau associé
            with st.expander("📈 Camembert – répartition des cas par tranche d’âge", expanded=True):
                if not df9.empty:
                    # Colonnes attendues (agrégé IDSR) : Cas_* et Deces_* par tranche
                    age_cases_map = {
                        "Cas_tnn": "<1 mois",
                        "Cas_0_11mois": "0–11 mois",
                        "Cas_12_59mois": "12–59 mois",
                        "Cas_5_14ans": "5–14 ans",
                        "Cas_15plus": "≥15 ans",
                    }
                    age_deaths_map = {
                        "Deces_tnn": "<1 mois",
                        "Deces_0_11mois": "0–11 mois",
                        "Deces_12_59mois": "12–59 mois",
                        "Deces_5_14ans": "5–14 ans",
                        "Deces_15plus": "≥15 ans",
                    }
                    rows_age = []
                    for c_col, label in age_cases_map.items():
                        if c_col in df9.columns:
                            cas = pd.to_numeric(df9[c_col], errors="coerce").sum(skipna=True)
                            d_col = [k for k, v in age_deaths_map.items() if v == label and k in df9.columns]
                            dec = pd.to_numeric(df9[d_col[0]], errors="coerce").sum(skipna=True) if d_col else np.nan
                            rows_age.append({"Tranche d'âge": label, "Cas": cas, "Décès": dec})
                    df_age = pd.DataFrame(rows_age)
                    if not df_age.empty:
                        df_age["Cas"] = pd.to_numeric(df_age["Cas"], errors="coerce").fillna(0).astype(int)
                        df_age["Décès"] = pd.to_numeric(df_age["Décès"], errors="coerce")
                        df_age["Décès"] = df_age["Décès"].fillna(0).astype(int)
                        total_cas_age = int(df_age["Cas"].sum())
                        df_age["Létalité (%)"] = np.where(df_age["Cas"] > 0, (df_age["Décès"] / df_age["Cas"]) * 100.0, np.nan)
                        df_age["Proportion des cas (%)"] = np.where(total_cas_age > 0, (df_age["Cas"] / total_cas_age) * 100.0, np.nan)
                
                        # Ordre logique
                        ordre_age = ["<1 mois", "0–11 mois", "12–59 mois", "5–14 ans", "≥15 ans"]
                        df_age["Tranche d'âge"] = pd.Categorical(df_age["Tranche d'âge"], categories=ordre_age, ordered=True)
                        df_age = df_age.sort_values("Tranche d'âge")
                
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            fig_pie_age = go.Figure(data=[go.Pie(
                                labels=df_age["Tranche d'âge"].astype(str),
                                values=df_age["Cas"],
                                hole=0.45,
                                textinfo="percent+label",
                                hovertemplate="%{label}<br>Cas=%{value}<br>%{percent}<extra></extra>",
                            )])
                            fig_pie_age.update_layout(template="plotly_white", height=420, margin=dict(t=30, b=10, l=10, r=10))
                            st.plotly_chart(fig_pie_age, width="stretch", key="idsr_pie_age")
                
                        with c2:
                            st.dataframe(
                                df_age[["Tranche d'âge", "Cas", "Décès", "Létalité (%)", "Proportion des cas (%)"]]
                                .assign(**{
                                    "Létalité (%)": df_age["Létalité (%)"].round(2),
                                    "Proportion des cas (%)": df_age["Proportion des cas (%)"].round(2),
                                }),
                                width="stretch",
                                height=420,
                                hide_index=True
                            )
                    else:
                        st.info("Aucune colonne 'Cas_*' par tranche d’âge trouvée dans les données IDSR.")
                else:
                    st.info("Aucune donnée après filtrage (impossible de produire la répartition par âge).")


            # 14.3) Tableau d’évolution par province et semaine épidémiologique
            with st.expander("Tableau croisé – évolution par province et semaine", expanded=False):

                # Objectif : Provinces en lignes, Année-Semaine en colonnes, sous-colonnes Cas/Décès/Létalité (%)
                if (not df9.empty) and (COL_PROV_ID in df9.columns):

                    # Préparer les colonnes numériques (robuste)
                    tmp_pw = prepare_idsr_numeric(df9, col_cases="Total_cas", col_deaths="Total_deces")

                    # Choix du niveau d’affichage
                    level_pw = st.radio(
                        "Niveau d’affichage",
                        ["Province de notification", "Province + Zone de notification"],
                        horizontal=True,
                        key="idsr_pw_level",
                    )

                    # Colonnes de lignes (index) selon le niveau
                    zs_col = None
                    if ("COL_ZS_ID" in globals()) and (globals()["COL_ZS_ID"] in tmp_pw.columns):
                        zs_col = globals()["COL_ZS_ID"]
                    elif ("COL_ZS" in globals()) and (globals()["COL_ZS"] in tmp_pw.columns):
                        zs_col = globals()["COL_ZS"]

                    idx_cols = [COL_PROV_ID]
                    if (level_pw == "Province + Zone de notification") and (zs_col is not None):
                        idx_cols = [COL_PROV_ID, zs_col]

                    # Colonne semaine (unique) : privilégier Année-Semaine si dispo, sinon TIME_KEY, sinon TIME_LAB
                    week_series, _order_key_col = choose_week_column(tmp_pw)
                    if week_series.empty:
                        st.info("Colonnes manquantes pour produire le tableau province × semaine (YW/TIME_KEY/TIME_LAB).")
                    else:
                        # Construire pivot Cas/Décès/Létalité (%)
                        pivot = build_cases_deaths_cfr_pivot(
                            tmp_pw,
                            idx_cols=idx_cols,
                            week_series=week_series,
                            col_cases="Total_cas",
                            col_deaths="Total_deces",
                            week_name="_YW_COL",
                            cfr_label="Létalité (%)",
                        )

                        # Ordonner les semaines chronologiquement si possible
                        if "weekly_sorted" in locals() and isinstance(weekly_sorted, pd.DataFrame) and (not weekly_sorted.empty):
                            ordre_w = ordered_weeks_from_weekly_sorted(weekly_sorted, fmt=fmt_yw_label)
                            pivot = reorder_pivot_weeks(pivot, ordre_w, fill_value=0)
                        else:
                            # Fallback : ordre lexical sur YYYYWww (chronologique)
                            ordre_w = sorted(list(pivot.columns.levels[1]))
                            pivot = reorder_pivot_weeks(pivot, ordre_w, fill_value=0)

                        # Rendu standard : CFR arrondi + reset_index + affichage safe
                        render_pivot_with_cfr(pivot, cfr_label="Létalité (%)", cfr_decimals=2, height=520)

                else:
                    st.info("Aucune donnée après filtrage (impossible de produire le tableau province × semaine).")


# =========================
# MAPS
# =========================
if show_maps:
    st.divider()
    st.header("Cartes (statique)")

    if gpd is None:
        st.warning("geopandas n'est pas installé. Ajoute 'geopandas' dans requirements.txt si tu veux les cartes.")
    else:
        st.caption("Cartes statiques (provinces / zones). Jointure fuzzy tolérante sur 'name'.")

        # --- GeoJSON (déploiement en ligne)
        # Par défaut: utiliser les fichiers présents dans le repo
        # Si l'utilisateur téléverse: le fichier uploadé remplace le défaut
        geo_prov_upl = st.file_uploader("📍 GeoJSON provinces (optionnel)", type=["geojson", "json"], key="geojson_prov")
        geo_zs_upl   = st.file_uploader("📍 GeoJSON zones de santé (optionnel)", type=["geojson", "json"], key="geojson_zs")

        col_reset1, col_reset2 = st.columns([1, 3])
        with col_reset1:
            if st.button("↩️ Réinitialiser"):
                st.session_state["geojson_prov"] = None
                st.session_state["geojson_zs"] = None
                st.rerun()
        with col_reset2:
            st.caption("Réinitialise les uploads et revient aux GeoJSON par défaut du dépôt (si présents).")

        # Fichiers par défaut (dans le repo)
        geo_prov_default = "geometry_rdc_provinces.geojson"
        geo_zs_default   = "geometry_rdc_zones_sante.geojson"

        def _upl_to_tmp_path(upl_obj, suffix=".geojson"):
            if upl_obj is None:
                return None
            data = upl_obj.getvalue()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data)
            tmp.flush()
            tmp.close()
            return tmp.name

        # Priorité : upload → sinon défaut local (repo) → sinon None
        geo_prov = _upl_to_tmp_path(geo_prov_upl) or (geo_prov_default if Path(geo_prov_default).exists() else None)
        geo_zs   = _upl_to_tmp_path(geo_zs_upl)   or (geo_zs_default   if Path(geo_zs_default).exists() else None)

        seuil_match = st.slider("Seuil de matching (fuzzy)", 0.70, 1.00, 0.90, 0.01)

        # Options d'affichage
        annoter_map = st.checkbox("Annoter (nom + valeur)", value=True)
        seuil_aff = st.number_input("Seuil affichage annotation (valeur >)", min_value=0, max_value=100000, value=1, step=1)
        afficher_fond = st.checkbox("Afficher fond de carte (contextily)", value=False)
        longueur_km = st.number_input("Longueur barre échelle (km)", min_value=5, max_value=300, value=50, step=5)

        # ---------- Provinces ----------
        st.subheader("Carte Provinces (cas)")
        if geo_prov and Path(geo_prov).exists() and COL_PROV in df_f.columns:
            gdfp = gpd.read_file(geo_prov)

            df_carte = df_f[[COL_PROV]].dropna().copy()
            df_carte["nb_cas_prov"] = 1
            df_carte = df_carte.groupby(COL_PROV, as_index=False)["nb_cas_prov"].sum()

            gdf_join, df_map, match_rate = joindre_donnees_fuzzy_geo(
                carte_gdf=gdfp,
                df_donnees=df_carte,
                colonne_cle_geo="name",
                colonne_cle_data=COL_PROV,
                colonne_valeurs="nb_cas_prov",
                seuil=seuil_match
            )

            st.caption(f"Taux de correspondance (données→carte) : {match_rate:.1%}")
            with st.expander("Diagnostic matching provinces (pire en haut)"):
                st.dataframe(df_map.head(50), width="stretch")

            fig = carte_statique_matplotlib(
                gdf=gdf_join,
                colonne_valeurs="nb_cas_prov",
                titre="RDC - Cas Cholera cumulés par province",
                annoter=annoter_map,
                nom_zone="name",
                fmt_valeurs="{:.0f}",
                seuil_affichage=float(seuil_aff),
                cmap="Reds",
                afficher_fond_carte=afficher_fond,
                longueur_barre_km=float(longueur_km),
            )

            if fig:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.error("Impossible de générer la carte provinces.")
        else:
            st.info("Carte provinces: charge un GeoJSON provinces et assure-toi que la colonne Province est présente.")

        st.divider()

        # ---------- Zones de santé ----------
        st.subheader("Carte Zones de santé (cas)")
        if geo_zs and Path(geo_zs).exists() and COL_ZS in df_f.columns:
            gdfz = gpd.read_file(geo_zs)

            df_carte = df_f[[COL_ZS]].dropna().copy()
            df_carte["nb_cas_zs"] = 1
            df_carte = df_carte.groupby(COL_ZS, as_index=False)["nb_cas_zs"].sum()

            gdf_join, df_map, match_rate = joindre_donnees_fuzzy_geo(
                carte_gdf=gdfz,
                df_donnees=df_carte,
                colonne_cle_geo="name",
                colonne_cle_data=COL_ZS,
                colonne_valeurs="nb_cas_zs",
                seuil=seuil_match
            )

            st.caption(f"Taux de correspondance (données→carte) : {match_rate:.1%}")
            with st.expander("Diagnostic matching ZS (pire en haut)"):
                st.dataframe(df_map.head(50), width="stretch")

            fig = carte_statique_matplotlib(
                gdf=gdf_join,
                colonne_valeurs="nb_cas_zs",
                titre="RDC - Cas Cholera cumulés par zone",
                annoter=annoter_map,
                nom_zone="name",
                fmt_valeurs="{:.0f}",
                seuil_affichage=float(seuil_aff),
                cmap="Reds",
                afficher_fond_carte=afficher_fond,
                longueur_barre_km=float(longueur_km),
            )

            if fig:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.error("Impossible de générer la carte ZS.")
        else:
            st.info("Carte ZS: charge un GeoJSON ZS et assure-toi que la colonne Zone de santé est présente.")
