"""Render the weekly IDSR analytics tab."""

import re
import unicodedata
import warnings
from pathlib import Path

from dashboard_app.app_loader import (
    get_line_list_bundle_caption,
    guess_preferred_included_file,
    list_available_line_list_files,
)
from dashboard_app.overview import format_range_label_for_display
from dashboard_app.runtime_support import inject_runtime_support

inject_runtime_support(globals())


IDSR_RENAME_MAP = {
    "NUM": "Num",
    "PAYS": "Pays",
    "PROV": "Province_notification",
    "Province": "Province_notification",
    "ZS": "Zone_de_sante_notification",
    "Zone_de_sante": "Zone_de_sante_notification",
    "POP": "Population",
    "prov_GIS": "Province_GIS",
    "Prov_GIS": "Province_GIS",
    "Province_GIS": "Province_GIS",
    "zs_GIS": "ZS_GIS",
    "ZS_GIS": "ZS_GIS",
    "ZoneSante_GIS": "ZS_GIS",
    "NUMSEM": "Num_semaine_epid",
    "Semaine": "Num_semaine_epid",
    "MALADIE": "Maladie",
    "disease": "Maladie",
    "C328TNN": "Cas_tnn",
    "C011MOIS": "Cas_0_11mois",
    "C1259MOIS": "Cas_12_59mois",
    "C515ANS": "Cas_5_14ans",
    "CP15ANS": "Cas_15plus",
    "DTNN": "Deces_tnn",
    "D011MOIS": "Deces_0_11mois",
    "D1259MOIS": "Deces_12_59mois",
    "D515ANS": "Deces_5_14ans",
    "DP15ANS": "Deces_15plus",
    "TOTALCAS": "Total_cas",
    "TOTALDECES": "Total_deces",
    "LETAL": "Taux_letalite",
    "ATTAQ": "Taux_attaque",
    "RecStatus": "Recstatus",
    "UniqueKey": "Cle_unique",
    "Year": "Annee_epid",
    "year": "Annee_epid",
    "Annee": "Annee_epid",
}


def harmonize_idsr_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_idsr_column_names(df)
    return df.rename(columns={k: v for k, v in IDSR_RENAME_MAP.items() if k in df.columns})


def idsr_frame_looks_valid(df: pd.DataFrame) -> bool:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False

    work = harmonize_idsr_columns(df.copy())
    has_geography = {
        "Province_notification",
        "Zone_de_sante_notification",
    }.issubset(set(work.columns))
    has_time = any(
        col in work.columns
        for col in ["Num_semaine_epid", "DEBUTSEM", "Date_debut_semaine", "Semaine_epid"]
    )
    strong_signals = [
        "Population",
        "Total_cas",
        "Total_deces",
        "Taux_letalite",
        "Taux_attaque",
        "Maladie",
        "Cas_tnn",
        "Cas_0_11mois",
        "Deces_tnn",
        "DEBUTSEM",
    ]
    signal_count = sum(1 for col in strong_signals if col in work.columns)
    return bool(has_geography and has_time and signal_count >= 2)


def _seek_excel_source(file_source: object) -> None:
    if hasattr(file_source, "seek"):
        try:
            file_source.seek(0)
        except Exception:
            pass


def list_available_idsr_files(available_files: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for path in available_files:
        if path.suffix.lower() not in {".xlsx", ".xls"}:
            continue

        try:
            with pd.ExcelFile(path) as xls:
                prioritized = [name for name in ("IDS_RDC", "IDSR", "idsr") if name in xls.sheet_names]
                remaining = [name for name in xls.sheet_names if name not in prioritized]
            for sheet_name in [*prioritized, *remaining]:
                sample = pd.read_excel(path, sheet_name=sheet_name, nrows=10)
                if idsr_frame_looks_valid(sample):
                    candidates.append(path)
                    break
        except Exception:
            continue

    return sorted(candidates, key=lambda p: p.name.lower())



def _idsr_fmt_int(_value: Any) -> str:
    """Format entier robuste utilisable par toutes les rubriques IDSR."""
    if pd.isna(_value):
        return "NA"
    try:
        return f"{int(round(float(_value))):,}"
    except Exception:
        return str(_value)


def _idsr_fmt_pct(_value: Any, decimals: int = 1) -> str:
    """Format pourcentage robuste utilisable par toutes les rubriques IDSR."""
    if pd.isna(_value):
        return "NA"
    try:
        return f"{float(_value):.{int(decimals)}f}%"
    except Exception:
        return str(_value)


def _strip_accents_idsr(value: object) -> str:
    """Retourne un texte en minuscules sans accents, utile pour lire les mois français."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", str(value))
        if unicodedata.category(ch) != "Mn"
    ).lower().strip()


def _parse_idsr_french_date_value(value: object) -> object:
    """Parse une date française IDSR du type 'lundi 29 décembre 2025'."""
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value.normalize()

    text = _strip_accents_idsr(value)
    text = re.sub(r"[,.;:/\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    mois = {
        "janvier": 1, "janv": 1,
        "fevrier": 2, "fevr": 2, "fev": 2,
        "mars": 3,
        "avril": 4, "avr": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7, "juil": 7,
        "aout": 8,
        "septembre": 9, "sept": 9,
        "octobre": 10, "oct": 10,
        "novembre": 11, "nov": 11,
        "decembre": 12, "dec": 12,
    }

    match = re.search(
        r"(?:^|\s)(?P<jour>\d{1,2})\s+(?P<mois>[a-z]+)\s+(?P<annee>\d{4})(?:\s|$)",
        text,
    )
    if not match:
        return pd.NaT

    mois_num = mois.get(match.group("mois"))
    if mois_num is None:
        return pd.NaT

    try:
        return pd.Timestamp(
            year=int(match.group("annee")),
            month=mois_num,
            day=int(match.group("jour")),
        ).normalize()
    except Exception:
        return pd.NaT


def parse_idsr_date_series(values: pd.Series) -> pd.Series:
    """Convertit une colonne date IDSR en vraie date pandas, y compris les mois français."""
    src = pd.Series(values).copy()

    if pd.api.types.is_datetime64_any_dtype(src):
        return pd.to_datetime(src, errors="coerce").dt.normalize()

    if pd.api.types.is_numeric_dtype(src):
        return pd.to_datetime(src, unit="D", origin="1899-12-30", errors="coerce").dt.normalize()

    parsed = pd.Series(pd.NaT, index=src.index, dtype="datetime64[ns]")

    numeric_values = pd.to_numeric(src, errors="coerce")
    serial_mask = numeric_values.between(20000, 80000)
    if serial_mask.any():
        parsed.loc[serial_mask] = pd.to_datetime(
            numeric_values.loc[serial_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        ).dt.normalize()

    fr_mask = parsed.isna() & src.notna()
    if fr_mask.any():
        parsed.loc[fr_mask] = src.loc[fr_mask].map(_parse_idsr_french_date_value)

    standard_mask = parsed.isna() & src.notna()
    if standard_mask.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed.loc[standard_mask] = pd.to_datetime(
                src.loc[standard_mask],
                errors="coerce",
                dayfirst=True,
            ).dt.normalize()

    return pd.to_datetime(parsed, errors="coerce").dt.normalize()


def normalize_idsr_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les en-têtes IDSR sans changer leur logique métier."""
    df = df.copy()
    cleaned_cols = []
    for col in df.columns:
        col_txt = str(col).replace("\ufeff", "").replace("\xa0", " ")
        col_txt = re.sub(r"\s+", " ", col_txt).strip()
        cleaned_cols.append(col_txt)
    df.columns = cleaned_cols
    return df


def normalize_idsr_debutsem_column(df: pd.DataFrame) -> pd.DataFrame:
    """Crée/alimente Date_debut_semaine à partir de DEBUTSEM et convertit DEBUTSEM en vraie date."""
    if "DEBUTSEM" not in df.columns:
        return df

    parsed = parse_idsr_date_series(df["DEBUTSEM"])
    if parsed.notna().any():
        df = df.copy()
        df["DEBUTSEM"] = parsed
        df["Date_debut_semaine"] = parsed
    return df


def idsr_year_from_debutsem(values: pd.Series, mode: str = "iso") -> pd.Series:
    """Retourne l'année exploitable depuis DEBUTSEM.

    mode='iso' : année épidémiologique ISO. Exemple : lundi 29/12/2025 = 2026-W01.
    mode='calendar' : année civile de la date.
    """
    src = pd.Series(values)
    dt = parse_idsr_date_series(src)
    if not dt.notna().any():
        return pd.Series(pd.NA, index=src.index, dtype="Int64")

    if mode == "calendar":
        return pd.to_numeric(dt.dt.year, errors="coerce").astype("Int64")

    return pd.to_numeric(dt.dt.isocalendar()["year"], errors="coerce").astype("Int64")


def _idsr_format_year_week_label(year: object, week: object) -> str:
    """Formate une semaine pour l'affichage analytique : 2026-W17 si l'année est connue."""
    try:
        if pd.notna(year) and pd.notna(week):
            return f"{int(float(year))}-W{int(float(week)):02d}"
    except Exception:
        pass
    try:
        if pd.notna(week):
            return f"W{int(float(week)):02d}"
    except Exception:
        pass
    return "NA"


def _idsr_build_year_week_label_series(year_values: Any, week_values: Any) -> pd.Series:
    """Construit des libellés année-semaine pour les sorties IDSR, avec fallback W##."""
    week = pd.to_numeric(pd.Series(week_values), errors="coerce")
    idx = week.index
    if year_values is None:
        year = pd.Series(np.nan, index=idx)
    else:
        year = pd.to_numeric(pd.Series(year_values, index=idx), errors="coerce")

    label = pd.Series(pd.NA, index=idx, dtype="string")
    has_year_week = year.notna() & week.notna()
    has_week_only = (~has_year_week) & week.notna()

    if has_year_week.any():
        label.loc[has_year_week] = (
            year.loc[has_year_week].astype("Int64").astype("string")
            + "-W"
            + week.loc[has_year_week].astype("Int64").astype("string").str.zfill(2)
        )
    if has_week_only.any():
        label.loc[has_week_only] = (
            "W" + week.loc[has_week_only].astype("Int64").astype("string").str.zfill(2)
        )
    return label


def _idsr_build_year_week_key_series(year_values: Any, week_values: Any) -> pd.Series:
    """Clé de tri cohérente avec les libellés année-semaine."""
    week = pd.to_numeric(pd.Series(week_values), errors="coerce")
    idx = week.index
    if year_values is None:
        year = pd.Series(np.nan, index=idx)
    else:
        year = pd.to_numeric(pd.Series(year_values, index=idx), errors="coerce")
    return pd.to_numeric(np.where(year.notna() & week.notna(), year * 100 + week, week), errors="coerce")


def render_idsr_reading_guide() -> None:
    """Affiche un plan de lecture simple pour éviter que l'utilisateur se perde dans l'onglet IDSR."""
    with st.expander("🧭 Guide de lecture de l’onglet IDSR", expanded=True):
        st.markdown(
            """
**Parcours conseillé**

1. **Filtres + résumé** : vérifier la période, le nombre de cas, décès, maladies, provinces et ZS.
2. **Complétude** : contrôler d’abord quelles provinces/ZS ont effectivement partagé les données.
3. **Taux d’attaque / incidence** : comparer le risque entre territoires avec la population comme dénominateur.
4. **Cartographie** : localiser rapidement les territoires qui concentrent les cas.
5. **Situation hebdomadaire** : lire la dernière semaine, les hausses et les territoires prioritaires.
6. **Qualité des données** : vérifier dates, doublons et cohérence des totaux avant diffusion.
7. **Analyses détaillées et exports** : approfondir par âge, maladie, province, semaine et mois.

**Règle pratique** : lire de haut en bas. Les blocs fermés sont des analyses complémentaires ; ouvrez-les selon le besoin.
"""
        )


def render_idsr_phase_header(title: str, description: str = "") -> None:
    """Affiche un intertitre compact pour organiser les blocs IDSR."""
    st.markdown(f"### {title}")
    if description:
        st.caption(description)





def _idsr_week_label_for_completeness(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Retourne un libellé semaine et une clé de tri pour les analyses IDSR hebdomadaires."""
    if df is None or df.empty:
        return pd.Series(dtype="string"), pd.Series(dtype="float64")
    idx = df.index

    if "Num_semaine_epid" in df.columns:
        week_num = pd.to_numeric(df["Num_semaine_epid"], errors="coerce")
        year_num = pd.to_numeric(df.get("Annee_epid"), errors="coerce") if "Annee_epid" in df.columns else pd.Series(np.nan, index=idx)
        week_label = _idsr_build_year_week_label_series(year_num, week_num)
        week_key = _idsr_build_year_week_key_series(year_num, week_num)
        return week_label.astype("string"), pd.to_numeric(week_key, errors="coerce")

    if "YW" in df.columns:
        week_label = df["YW"].astype("string")
        week_key = pd.to_numeric(df.get("YW_KEY"), errors="coerce") if "YW_KEY" in df.columns else pd.Series(range(len(df)), index=idx)
        return week_label, week_key

    if "TIME_LAB" in df.columns:
        week_label = df["TIME_LAB"].astype("string")
        week_key = pd.to_numeric(df.get("TIME_KEY"), errors="coerce") if "TIME_KEY" in df.columns else pd.Series(range(len(df)), index=idx)
        return week_label, week_key

    return pd.Series(pd.NA, index=idx, dtype="string"), pd.Series(np.nan, index=idx)


def _build_idsr_completeness_matrices(
    df: pd.DataFrame,
    province_col: str,
    zs_col: str,
    *,
    expected_mode: str = "union",
    fill_continuous_weeks: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Construit les matrices de complétude IDSR province × semaine."""
    if df is None or df.empty or province_col not in df.columns or zs_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype="float64"), pd.DataFrame()

    work = df.copy()
    work["_idsr_week_label"], work["_idsr_week_key"] = _idsr_week_label_for_completeness(work)
    work["_idsr_province"] = work[province_col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    work["_idsr_zs"] = work[zs_col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    work = work.dropna(subset=["_idsr_province", "_idsr_zs", "_idsr_week_label", "_idsr_week_key"])
    work = work[(work["_idsr_province"] != "") & (work["_idsr_zs"] != "") & (work["_idsr_week_label"] != "")]
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype="float64"), pd.DataFrame()

    unique_reporting = work[["_idsr_province", "_idsr_zs", "_idsr_week_label", "_idsr_week_key"]].drop_duplicates()
    week_ref = unique_reporting[["_idsr_week_label", "_idsr_week_key"]].drop_duplicates().sort_values("_idsr_week_key")
    week_labels = week_ref["_idsr_week_label"].astype(str).tolist()

    if fill_continuous_weeks and week_ref["_idsr_week_key"].notna().any():
        keys = week_ref["_idsr_week_key"].dropna()
        if np.all(np.isclose(keys.astype(float), np.round(keys.astype(float)))):
            k_min, k_max = int(keys.min()), int(keys.max())
            if 1 <= k_min <= k_max <= 53:
                week_labels = [str(i) for i in range(k_min, k_max + 1)]
            elif k_min > 53:
                try:
                    start_year, start_week = divmod(k_min, 100)
                    end_year, end_week = divmod(k_max, 100)
                    start_date = pd.Timestamp.fromisocalendar(start_year, start_week, 1)
                    end_date = pd.Timestamp.fromisocalendar(end_year, end_week, 1)
                    if start_date <= end_date:
                        continuous_labels = []
                        cursor = start_date
                        while cursor <= end_date:
                            iso_cursor = cursor.isocalendar()
                            continuous_labels.append(f"{int(iso_cursor.year)}-W{int(iso_cursor.week):02d}")
                            cursor += pd.Timedelta(days=7)
                        if continuous_labels:
                            week_labels = continuous_labels
                except Exception:
                    pass

    counts = (
        unique_reporting
        .groupby(["_idsr_province", "_idsr_week_label"], as_index=False)
        .agg(Nombre_ZS=("_idsr_zs", "nunique"))
    )
    count_pivot = (
        counts
        .pivot_table(index="_idsr_province", columns="_idsr_week_label", values="Nombre_ZS", aggfunc="sum", fill_value=0, observed=False)
        .astype(int)
    )
    prov_order = sorted(count_pivot.index.astype(str).tolist())
    count_pivot = count_pivot.reindex(index=prov_order, columns=week_labels, fill_value=0)

    if expected_mode == "max_week":
        expected = count_pivot.max(axis=1).replace(0, np.nan)
    else:
        expected = (
            unique_reporting
            .groupby("_idsr_province")["_idsr_zs"]
            .nunique()
            .reindex(count_pivot.index)
            .replace(0, np.nan)
        )

    rel_pivot = count_pivot.div(expected, axis=0).clip(lower=0, upper=1).fillna(0)
    summary = pd.DataFrame({
        "Province": count_pivot.index.astype(str),
        "ZS attendues": expected.fillna(0).astype(int).values,
        "Semaines analysées": int(len(count_pivot.columns)),
        "Moyenne ZS rapportantes": count_pivot.mean(axis=1).round(1).values,
        "Complétude moyenne (%)": (rel_pivot.mean(axis=1) * 100).round(1).values,
        "Minimum ZS rapportantes": count_pivot.min(axis=1).astype(int).values,
        "Semaine la plus faible": count_pivot.idxmin(axis=1).astype(str).values,
        "Dernière semaine": str(count_pivot.columns[-1]) if len(count_pivot.columns) else "",
        "ZS dernière semaine": count_pivot.iloc[:, -1].astype(int).values if len(count_pivot.columns) else 0,
        "Complétude dernière semaine (%)": (rel_pivot.iloc[:, -1] * 100).round(1).values if len(rel_pivot.columns) else 0,
    })
    return count_pivot, rel_pivot, expected, summary


def _make_idsr_completeness_heatmap(count_pivot: pd.DataFrame, rel_pivot: pd.DataFrame) -> object:
    """Construit la figure Plotly du tableau de complétude IDSR."""
    if count_pivot is None or count_pivot.empty or rel_pivot is None or rel_pivot.empty:
        return None

    y_labels = [str(x).upper() for x in count_pivot.index.tolist()]
    x_labels = [str(x) for x in count_pivot.columns.tolist()]
    z_values = rel_pivot.to_numpy(dtype=float)
    text_values = count_pivot.astype(int).astype(str).to_numpy()

    hover = []
    for prov_idx, prov in enumerate(y_labels):
        row = []
        for week_idx, week in enumerate(x_labels):
            row.append(
                f"Province: {prov}<br>"
                f"Semaine: {week}<br>"
                f"ZS rapportantes: {int(count_pivot.iloc[prov_idx, week_idx])}<br>"
                f"Complétude relative: {float(rel_pivot.iloc[prov_idx, week_idx]) * 100:.1f}%"
            )
        hover.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "black"},
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            zmin=0,
            zmax=1,
            colorscale=[
                [0.00, "#ff0000"],
                [0.25, "#ff9900"],
                [0.50, "#ffff00"],
                [0.75, "#99ff00"],
                [1.00, "#00ee00"],
            ],
            colorbar={"title": "Complétude<br>(relative)", "tickformat": ".0%"},
        )
    )
    height = min(max(480, 26 * len(y_labels) + 180), 1050)
    fig.update_layout(
        title="Tableau de complétude IDSR (nombre de ZS par province)",
        xaxis_title="Semaines épidémiologiques de notification",
        yaxis_title="Divisions provinciales de la santé (DPS)",
        height=height,
        margin=dict(l=120, r=80, t=80, b=70),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(type="category", tickangle=0, showgrid=True, gridcolor="rgba(0,0,0,0.10)")
    fig.update_yaxes(type="category", autorange="reversed", showgrid=True, gridcolor="rgba(0,0,0,0.10)")
    return fig


def render_idsr_completeness_section(df: pd.DataFrame, province_col: str, zs_col: str) -> None:
    """Affiche l'analyse de complétude IDSR province × semaine dans une liste déroulante."""
    with st.expander("02 · 📊 Complétude IDSR — ZS rapportantes par province et semaine", expanded=False):
        st.markdown("### Tableau de complétude IDSR")
        st.caption(
            "Lecture : chaque cellule indique le nombre de zones de santé ayant rapporté au moins une ligne IDSR. "
            "La couleur représente la complétude relative par rapport au nombre de ZS attendues/observées dans la province."
        )

        if df is None or df.empty:
            st.info("Aucune donnée disponible pour calculer la complétude IDSR.")
            return

        missing = [c for c in [province_col, zs_col, "Num_semaine_epid"] if c not in df.columns]
        if missing:
            st.info("Complétude IDSR indisponible : colonne(s) manquante(s) : " + ", ".join(missing) + ".")
            return

        c1, c2, c3 = st.columns([1.25, 1, 1])
        with c1:
            expected_mode_label = st.selectbox(
                "Référence de complétude",
                options=["Total ZS observées sur la période", "Maximum hebdomadaire observé"],
                index=0,
                key="idsr_completeness_expected_mode",
                help=(
                    "Total ZS observées : dénominateur stable par province sur toute la période filtrée. "
                    "Maximum hebdomadaire : dénominateur basé sur la meilleure semaine observée."
                ),
            )
        with c2:
            completeness_threshold = st.slider(
                "Seuil d'alerte (%)",
                min_value=0,
                max_value=100,
                value=80,
                step=5,
                key="idsr_completeness_threshold",
            )
        with c3:
            show_completion_table = st.checkbox(
                "Afficher le tableau source",
                value=False,
                key="idsr_completeness_show_table",
            )

        expected_mode = "max_week" if expected_mode_label == "Maximum hebdomadaire observé" else "union"
        count_pivot, rel_pivot, expected, summary = _build_idsr_completeness_matrices(
            df,
            province_col,
            zs_col,
            expected_mode=expected_mode,
            fill_continuous_weeks=True,
        )

        if count_pivot.empty:
            st.info("Aucune combinaison Province / Zone de santé / Semaine exploitable pour produire la complétude.")
            return

        latest_col = count_pivot.columns[-1]
        total_expected = float(expected.fillna(0).sum())
        latest_count = float(count_pivot[latest_col].sum())
        latest_pct = (latest_count / total_expected * 100.0) if total_expected > 0 else np.nan
        mean_pct = float(rel_pivot.mean().mean() * 100.0) if not rel_pivot.empty else np.nan
        alert_cells = int((rel_pivot < (float(completeness_threshold) / 100.0)).sum().sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Provinces", f"{count_pivot.shape[0]:,}")
        k2.metric("Semaines", f"{count_pivot.shape[1]:,}")
        k3.metric("Complétude moyenne", "NA" if pd.isna(mean_pct) else f"{mean_pct:.1f}%")
        k4.metric(f"Dernière semaine ({latest_col})", "NA" if pd.isna(latest_pct) else f"{latest_pct:.1f}%")

        if alert_cells > 0:
            st.caption(
                f"Cellules sous le seuil de {int(completeness_threshold)}% : {alert_cells:,}. "
                "Elles méritent une vérification avec les DPS/ZS concernées."
            )

        def _join_weeks(_weeks: list[str], max_items: int = 18) -> str:
            _weeks = [str(w) for w in _weeks]
            if len(_weeks) <= max_items:
                return ", ".join(_weeks)
            return ", ".join(_weeks[:max_items]) + f", +{len(_weeks) - max_items} autre(s)"

        zero_mask = count_pivot.eq(0)
        zero_by_province_rows = []
        for _province in count_pivot.index.tolist():
            _weeks_zero = [str(w) for w in count_pivot.columns[zero_mask.loc[_province]].tolist()]
            if _weeks_zero:
                zero_by_province_rows.append({
                    "Province": str(_province),
                    "Nombre de semaines sans partage": int(len(_weeks_zero)),
                    "Semaines sans partage": _join_weeks(_weeks_zero),
                })

        zero_by_week_rows = []
        for _week in count_pivot.columns.tolist():
            _provinces_zero = [str(p) for p in count_pivot.index[count_pivot[_week].eq(0)].tolist()]
            if _provinces_zero:
                zero_by_week_rows.append({
                    "Semaine": str(_week),
                    "Nombre de provinces sans partage": int(len(_provinces_zero)),
                    "Provinces sans partage": ", ".join(_provinces_zero),
                })

        zero_by_province = pd.DataFrame(zero_by_province_rows)
        zero_by_week = pd.DataFrame(zero_by_week_rows)

        st.markdown("**Résumé automatique des provinces sans partage**")
        if zero_by_province.empty:
            st.success("Toutes les provinces observées ont partagé au moins une ZS pour chaque semaine affichée.")
        else:
            _n_prov_zero = int(zero_by_province["Province"].nunique())
            _n_week_zero = int(zero_by_week["Semaine"].nunique()) if not zero_by_week.empty else 0
            st.warning(
                f"{_n_prov_zero} province(s) présentent au moins une semaine sans partage IDSR. "
                f"Le problème concerne {_n_week_zero} semaine(s) dans la fenêtre analysée."
            )
            col_zero_1, col_zero_2 = st.columns([1.1, 1])
            with col_zero_1:
                st.markdown("**Par province**")
                zero_by_province = zero_by_province.sort_values(
                    ["Nombre de semaines sans partage", "Province"],
                    ascending=[False, True],
                )
                st.dataframe(zero_by_province, width="stretch", height=260, hide_index=True)
            with col_zero_2:
                st.markdown("**Par semaine**")
                if zero_by_week.empty:
                    st.info("Aucune semaine totalement manquante pour les provinces observées.")
                else:
                    st.dataframe(zero_by_week, width="stretch", height=260, hide_index=True)

            st.caption(
                "Note : ce résumé détecte les provinces présentes dans le fichier mais sans aucune ZS rapportante sur certaines semaines. "
                "Une province totalement absente du fichier nécessite une table de référence complète des provinces/ZS."
            )

        fig = _make_idsr_completeness_heatmap(count_pivot, rel_pivot)
        if fig is not None:
            try:
                st_plot(fig, key="idsr_completeness_heatmap")
            except Exception:
                st.plotly_chart(fig, width="stretch", key="idsr_completeness_heatmap")

        if show_completion_table:
            st.markdown("---")
            st.markdown("**📋 Données de complétude IDSR**")
            st.markdown("**Nombre de ZS rapportantes par province et semaine**")
            count_view = count_pivot.reset_index().rename(columns={"_idsr_province": "Province"})
            st.dataframe(count_view, width="stretch", height=420, hide_index=True)

            st.markdown("**Résumé par province**")
            summary_view = summary.sort_values(
                ["Complétude dernière semaine (%)", "Complétude moyenne (%)", "Province"],
                ascending=[True, True, True],
            )
            st.dataframe(summary_view, width="stretch", height=420, hide_index=True)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Télécharger complétude IDSR par semaine (CSV)",
                    data=count_view.to_csv(index=False).encode("utf-8"),
                    file_name="idsr_completude_zs_par_province_semaine.csv",
                    mime="text/csv",
                    key="idsr_download_completeness_matrix",
                )
            with dl2:
                st.download_button(
                    "⬇️ Télécharger résumé complétude IDSR (CSV)",
                    data=summary_view.to_csv(index=False).encode("utf-8"),
                    file_name="idsr_completude_resume_province.csv",
                    mime="text/csv",
                    key="idsr_download_completeness_summary",
                )
            if not zero_by_province.empty:
                st.download_button(
                    "⬇️ Télécharger provinces sans partage (CSV)",
                    data=zero_by_province.to_csv(index=False).encode("utf-8"),
                    file_name="idsr_provinces_sans_partage_par_semaine.csv",
                    mime="text/csv",
                    key="idsr_download_no_reporting_provinces",
                )


def _idsr_clean_group_cols_for_rates(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Nettoie les colonnes de regroupement utilisées pour les taux IDSR."""
    work = df.copy()
    for col in group_cols:
        if col in work.columns:
            work[col] = work[col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
            work.loc[work[col].isin(["", "<NA>", "nan", "None"]), col] = pd.NA
    return work


def _idsr_population_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    pop_col: str = "Population",
    zs_col: Optional[str] = None,
    denominator_mode: str = "zs_sum",
) -> pd.DataFrame:
    """Calcule un dénominateur population robuste par groupe."""
    if df is None or df.empty or pop_col not in df.columns:
        return pd.DataFrame(columns=[*group_cols, "Population_reference"])

    work = _idsr_clean_group_cols_for_rates(df, group_cols)
    work[pop_col] = pd.to_numeric(work[pop_col], errors="coerce")
    work = work.dropna(subset=[pop_col, *group_cols])
    work = work[work[pop_col] > 0]
    if work.empty:
        return pd.DataFrame(columns=[*group_cols, "Population_reference"])

    if denominator_mode == "zs_sum" and zs_col and zs_col in work.columns and zs_col not in group_cols:
        work[zs_col] = work[zs_col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
        work.loc[work[zs_col].isin(["", "<NA>", "nan", "None"]), zs_col] = pd.NA
        work_zs = work.dropna(subset=[zs_col])
        if not work_zs.empty:
            pop_by_zs = (
                work_zs
                .groupby([*group_cols, zs_col], dropna=False, as_index=False)
                .agg(Population_ZS=(pop_col, "max"))
            )
            return pop_by_zs.groupby(group_cols, dropna=False, as_index=False).agg(Population_reference=("Population_ZS", "sum"))

    return work.groupby(group_cols, dropna=False, as_index=False).agg(Population_reference=(pop_col, "max"))


def _idsr_population_total(
    df: pd.DataFrame,
    *,
    pop_col: str = "Population",
    province_col: Optional[str] = None,
    zs_col: Optional[str] = None,
    denominator_mode: str = "zs_sum",
) -> float:
    """Calcule une population de référence globale sur le périmètre filtré."""
    if df is None or df.empty or pop_col not in df.columns:
        return np.nan

    work = df.copy()
    work[pop_col] = pd.to_numeric(work[pop_col], errors="coerce")
    work = work[work[pop_col].notna() & (work[pop_col] > 0)]
    if work.empty:
        return np.nan

    if denominator_mode == "zs_sum" and zs_col and zs_col in work.columns:
        key_cols = [c for c in [province_col, zs_col] if c and c in work.columns]
        if key_cols:
            for col in key_cols:
                work[col] = work[col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
            pop_ref = work.dropna(subset=key_cols).groupby(key_cols, dropna=False)[pop_col].max().sum()
            return float(pop_ref) if pd.notna(pop_ref) else np.nan

    pop_ref = work[pop_col].max()
    return float(pop_ref) if pd.notna(pop_ref) else np.nan


def _build_idsr_attack_incidence_table(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    cases_col: str = "Total_cas",
    pop_col: str = "Population",
    province_col: Optional[str] = None,
    zs_col: Optional[str] = None,
    denominator_mode: str = "zs_sum",
    incidence_multiplier: int = 100000,
) -> pd.DataFrame:
    """Construit une table IDSR cas/population/taux d'attaque/incidence par groupe."""
    if df is None or df.empty or cases_col not in df.columns or pop_col not in df.columns:
        return pd.DataFrame()

    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return pd.DataFrame()

    work = _idsr_clean_group_cols_for_rates(df, group_cols)
    work[cases_col] = pd.to_numeric(work[cases_col], errors="coerce").fillna(0)
    work = work.dropna(subset=group_cols)
    if work.empty:
        return pd.DataFrame()

    cases_tbl = work.groupby(group_cols, dropna=False, as_index=False).agg(Cas=(cases_col, "sum"), Lignes=(cases_col, "size"))
    pop_tbl = _idsr_population_by_group(work, group_cols, pop_col=pop_col, zs_col=zs_col, denominator_mode=denominator_mode)

    out = cases_tbl.merge(pop_tbl, on=group_cols, how="left")
    out["Population_reference"] = pd.to_numeric(out["Population_reference"], errors="coerce")
    out["Cas"] = pd.to_numeric(out["Cas"], errors="coerce").fillna(0)
    out["Taux_attaque_%"] = np.where(out["Population_reference"] > 0, (out["Cas"] / out["Population_reference"]) * 100.0, np.nan)
    out[f"Incidence_pour_{int(incidence_multiplier)}"] = np.where(
        out["Population_reference"] > 0,
        (out["Cas"] / out["Population_reference"]) * float(incidence_multiplier),
        np.nan,
    )
    return out


def _build_idsr_attack_incidence_weekly(
    df: pd.DataFrame,
    *,
    cases_col: str = "Total_cas",
    pop_col: str = "Population",
    province_col: Optional[str] = None,
    zs_col: Optional[str] = None,
    denominator_mode: str = "zs_sum",
    incidence_multiplier: int = 100000,
) -> pd.DataFrame:
    """Construit la tendance hebdomadaire du taux d'attaque et de l'incidence."""
    if df is None or df.empty or cases_col not in df.columns or pop_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["_idsr_week_label"], work["_idsr_week_key"] = _idsr_week_label_for_completeness(work)
    work[cases_col] = pd.to_numeric(work[cases_col], errors="coerce").fillna(0)
    work = work.dropna(subset=["_idsr_week_label", "_idsr_week_key"])
    if work.empty:
        return pd.DataFrame()

    population_ref = _idsr_population_total(
        work,
        pop_col=pop_col,
        province_col=province_col,
        zs_col=zs_col,
        denominator_mode=denominator_mode,
    )

    weekly = (
        work
        .groupby(["_idsr_week_label", "_idsr_week_key"], as_index=False)
        .agg(Cas=(cases_col, "sum"))
        .sort_values("_idsr_week_key")
    )
    weekly["Population_reference"] = population_ref
    weekly["Taux_attaque_%"] = np.where(weekly["Population_reference"] > 0, (weekly["Cas"] / weekly["Population_reference"]) * 100.0, np.nan)
    weekly[f"Incidence_pour_{int(incidence_multiplier)}"] = np.where(
        weekly["Population_reference"] > 0,
        (weekly["Cas"] / weekly["Population_reference"]) * float(incidence_multiplier),
        np.nan,
    )
    return weekly.rename(columns={"_idsr_week_label": "Semaine", "_idsr_week_key": "Semaine_key"})


def render_idsr_attack_incidence_section(
    df: pd.DataFrame,
    *,
    province_col: str,
    zs_col: Optional[str],
    mal_col: str,
) -> None:
    """Affiche la rubrique IDSR taux d'attaque / incidence avec explication et calculs."""
    with st.expander("03 · 📈 Taux d’attaque et incidence — calculs et interprétation", expanded=False):
        st.markdown("### Taux d’attaque et incidence")
        st.caption(
            "Cette rubrique recalcule les indicateurs à partir des cas et de la population du périmètre filtré. "
            "Elle complète les tendances cas/décès et aide à comparer les provinces ou maladies malgré des tailles de population différentes."
        )

        with st.expander("03.a · 📘 Comprendre les indicateurs", expanded=False):
            st.markdown(
                """
**Taux d’attaque**  
C’est la proportion de personnes qui deviennent malades parmi une population exposée pendant une période donnée, souvent pendant une flambée ou une situation épidémique.

Formule opérationnelle :

```text
Taux d’attaque (%) = (Nombre de nouveaux cas / Population exposée ou à risque) × 100
```

**Incidence**  
C’est la fréquence des nouveaux cas dans une population pendant une période donnée. Pour comparer les territoires, on l’exprime souvent pour 1 000, 10 000 ou 100 000 habitants.

Formule opérationnelle :

```text
Incidence = (Nombre de nouveaux cas / Population à risque) × multiplicateur
```

**Différence pratique**  
Le taux d’attaque s’exprime généralement en **%** et est très utilisé en contexte de flambée. L’incidence est généralement exprimée **pour 100 000 habitants** afin de comparer des provinces ou zones de santé de tailles différentes.

**Point de vigilance**  
Le résultat dépend fortement de la qualité de la colonne `Population`. Si la population est répétée par maladie ou par semaine, le calcul doit éviter de sommer plusieurs fois la même population.
"""
            )

        required = ["Total_cas", "Population"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.warning("Calcul indisponible : colonne(s) manquante(s) : " + ", ".join(missing) + ".")
            return

        available_groupings = {}
        if province_col in df.columns:
            available_groupings["Province"] = [province_col]
        if mal_col in df.columns:
            available_groupings["Maladie"] = [mal_col]
        if province_col in df.columns and mal_col in df.columns:
            available_groupings["Province + maladie"] = [province_col, mal_col]
        if province_col in df.columns and zs_col and zs_col in df.columns:
            available_groupings["Zone de santé"] = [province_col, zs_col]

        if not available_groupings:
            st.info("Aucune variable de regroupement exploitable pour calculer les taux.")
            return

        c1, c2, c3, c4 = st.columns([1.2, 1.1, 1.0, 0.8])
        with c1:
            grouping_label = st.selectbox(
                "Niveau d’analyse",
                options=list(available_groupings.keys()),
                index=0,
                key="idsr_attack_incidence_grouping",
            )
        with c2:
            denominator_label = st.selectbox(
                "Dénominateur population",
                options=["Somme des populations uniques par ZS", "Population maximale du groupe"],
                index=0,
                key="idsr_attack_incidence_denominator",
                help=(
                    "Utilise la somme par ZS si `Population` est au niveau zone de santé. "
                    "Utilise la population maximale si `Population` est déjà une population provinciale répétée."
                ),
            )
        with c3:
            incidence_multiplier = st.selectbox(
                "Incidence pour",
                options=[1000, 10000, 100000],
                index=2,
                key="idsr_attack_incidence_multiplier",
            )
        with c4:
            top_n = st.slider(
                "Top",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                key="idsr_attack_incidence_topn",
            )

        denominator_mode = "group_max" if denominator_label == "Population maximale du groupe" else "zs_sum"
        group_cols = available_groupings[grouping_label]
        incidence_col = f"Incidence_pour_{int(incidence_multiplier)}"

        result_tbl = _build_idsr_attack_incidence_table(
            df,
            group_cols,
            cases_col="Total_cas",
            pop_col="Population",
            province_col=province_col,
            zs_col=zs_col,
            denominator_mode=denominator_mode,
            incidence_multiplier=int(incidence_multiplier),
        )

        if result_tbl.empty:
            st.info("Aucune donnée exploitable pour calculer le taux d’attaque et l’incidence.")
            return

        total_cases = pd.to_numeric(df.get("Total_cas"), errors="coerce").sum(skipna=True)
        total_population = _idsr_population_total(
            df,
            pop_col="Population",
            province_col=province_col,
            zs_col=zs_col,
            denominator_mode=denominator_mode,
        )
        global_attack = (float(total_cases) / float(total_population) * 100.0) if pd.notna(total_population) and total_population > 0 else np.nan
        global_incidence = (float(total_cases) / float(total_population) * float(incidence_multiplier)) if pd.notna(total_population) and total_population > 0 else np.nan

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cas analysés", "NA" if pd.isna(total_cases) else f"{int(round(float(total_cases))):,}")
        m2.metric("Population référence", "NA" if pd.isna(total_population) else f"{int(round(float(total_population))):,}")
        m3.metric("Taux d’attaque global", "NA" if pd.isna(global_attack) else f"{global_attack:.3f}%")
        m4.metric(f"Incidence / {int(incidence_multiplier):,}", "NA" if pd.isna(global_incidence) else f"{global_incidence:.2f}")

        valid_tbl = result_tbl.dropna(subset=["Population_reference", "Taux_attaque_%", incidence_col]).copy()
        if not valid_tbl.empty:
            top_attack = valid_tbl.sort_values("Taux_attaque_%", ascending=False).iloc[0]
            top_incidence = valid_tbl.sort_values(incidence_col, ascending=False).iloc[0]
            label_cols = [c for c in group_cols if c in valid_tbl.columns]

            def _row_label(row) -> str:
                return " / ".join(str(row[c]) for c in label_cols)

            st.info(
                f"Lecture automatique : le taux d’attaque le plus élevé est observé pour **{_row_label(top_attack)}** "
                f"({_idsr_fmt_pct(top_attack['Taux_attaque_%'], 3)}). "
                f"L’incidence la plus élevée est observée pour **{_row_label(top_incidence)}** "
                f"({float(top_incidence[incidence_col]):.2f} pour {int(incidence_multiplier):,} habitants)."
            )
        else:
            st.warning(
                "Les cas sont disponibles, mais la population de référence est insuffisante pour interpréter les taux. "
                "Vérifie la colonne Population."
            )

        display_tbl = result_tbl.copy()
        display_tbl["Population_reference"] = pd.to_numeric(display_tbl["Population_reference"], errors="coerce")
        display_tbl["Taux_attaque_%"] = pd.to_numeric(display_tbl["Taux_attaque_%"], errors="coerce")
        display_tbl[incidence_col] = pd.to_numeric(display_tbl[incidence_col], errors="coerce")
        display_tbl = display_tbl.sort_values(incidence_col, ascending=False, na_position="last")
        display_tbl["Groupe"] = display_tbl[group_cols].astype("string").fillna("Non renseigné").agg(" / ".join, axis=1)
        top_plot = display_tbl.head(int(top_n)).sort_values(incidence_col, ascending=True)

        if not top_plot.empty:
            fig = px.bar(
                top_plot,
                x=incidence_col,
                y="Groupe",
                orientation="h",
                text=incidence_col,
                title=f"Top {int(top_n)} — incidence pour {int(incidence_multiplier):,} habitants",
                hover_data={
                    "Cas": ":,.0f",
                    "Population_reference": ":,.0f",
                    "Taux_attaque_%": ":.3f",
                    incidence_col: ":.2f",
                    "Groupe": False,
                },
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
            fig.update_layout(xaxis_title=f"Incidence pour {int(incidence_multiplier):,} habitants", yaxis_title="")
            try:
                st_plot(fig, key="idsr_attack_incidence_top_chart")
            except Exception:
                st.plotly_chart(fig, width="stretch", key="idsr_attack_incidence_top_chart")

        with st.expander("03.b · 📋 Tableau détaillé des taux", expanded=False):
            table_cols = [*group_cols, "Cas", "Population_reference", "Taux_attaque_%", incidence_col, "Lignes"]
            table_view = display_tbl[[c for c in table_cols if c in display_tbl.columns]].copy()
            st.dataframe(table_view, width="stretch", height=420, hide_index=True)
            st.download_button(
                "⬇️ Télécharger taux d’attaque et incidence (CSV)",
                data=table_view.to_csv(index=False).encode("utf-8"),
                file_name="idsr_taux_attaque_incidence.csv",
                mime="text/csv",
                key="idsr_download_attack_incidence",
            )

        weekly_tbl = _build_idsr_attack_incidence_weekly(
            df,
            cases_col="Total_cas",
            pop_col="Population",
            province_col=province_col,
            zs_col=zs_col,
            denominator_mode=denominator_mode,
            incidence_multiplier=int(incidence_multiplier),
        )
        if not weekly_tbl.empty and incidence_col in weekly_tbl.columns:
            with st.expander("03.c · 📉 Tendance hebdomadaire de l’incidence", expanded=False):
                fig_week = px.line(
                    weekly_tbl,
                    x="Semaine",
                    y=incidence_col,
                    markers=True,
                    title=f"Incidence hebdomadaire pour {int(incidence_multiplier):,} habitants",
                    hover_data={"Cas": ":,.0f", "Population_reference": ":,.0f", "Taux_attaque_%": ":.3f"},
                )
                fig_week.update_layout(xaxis_title="Semaine épidémiologique", yaxis_title=f"Incidence / {int(incidence_multiplier):,}")
                try:
                    st_plot(fig_week, key="idsr_attack_incidence_weekly_chart")
                except Exception:
                    st.plotly_chart(fig_week, width="stretch", key="idsr_attack_incidence_weekly_chart")
                st.dataframe(weekly_tbl, width="stretch", height=300, hide_index=True)


def _build_idsr_period_labels(df_scope: pd.DataFrame) -> tuple[str, str]:
    """Construit des libellés de période cohérents pour le résumé IDSR."""
    period_label = compute_analysis_period_value(df_scope)
    if not period_label or period_label == "-":
        period_start = (
            pd.to_datetime(df_scope.get("Date_debut_semaine_iso"), errors="coerce")
            if "Date_debut_semaine_iso" in df_scope.columns
            else pd.Series(dtype="datetime64[ns]")
        )
        if not period_start.empty and period_start.notna().any():
            period_min = period_start.min()
            period_max = period_start.max() + pd.Timedelta(days=6)
            period_label = f"{period_min:%d/%m/%Y} -> {period_max:%d/%m/%Y}"
        else:
            period_label = "Période indisponible"

    time_span = "Fenêtre hebdo indisponible"
    if "TIME_KEY" in df_scope.columns and "TIME_LAB" in df_scope.columns:
        week_ref = (
            df_scope[["TIME_KEY", "TIME_LAB"]]
            .copy()
            .dropna(subset=["TIME_KEY", "TIME_LAB"])
            .drop_duplicates()
            .sort_values("TIME_KEY")
        )
        if not week_ref.empty:
            time_span = f"{week_ref['TIME_LAB'].iloc[0]} -> {week_ref['TIME_LAB'].iloc[-1]}"
    elif "TIME_LAB" in df_scope.columns:
        time_values = df_scope["TIME_LAB"].dropna().astype(str).tolist()
        if time_values:
            time_span = f"{min(time_values)} -> {max(time_values)}"

    return period_label, time_span


def _load_idsr_workbook(file_source: object) -> pd.DataFrame:
    """Charge un fichier IDSR en priorisant les feuilles usuelles."""
    last_exc = None
    candidate_sheets: list[str] = []

    try:
        _seek_excel_source(file_source)
        with pd.ExcelFile(file_source) as xls:
            candidate_sheets = list(xls.sheet_names)
    except Exception as exc:
        last_exc = exc

    prioritized = [sheet for sheet in ("IDS_RDC", "IDSR", "idsr") if sheet in candidate_sheets]
    remaining = [sheet for sheet in candidate_sheets if sheet not in prioritized]

    for sheet_name in [*prioritized, *remaining]:
        try:
            _seek_excel_source(file_source)
            df_loaded = load_excel_cached(file_source, sheet_name=sheet_name)
            if isinstance(df_loaded, pd.DataFrame) and not df_loaded.empty and idsr_frame_looks_valid(df_loaded):
                return df_loaded
        except Exception as exc:
            last_exc = exc

    try:
        _seek_excel_source(file_source)
        df_loaded = load_excel_cached(file_source)
        if isinstance(df_loaded, pd.DataFrame) and not df_loaded.empty and idsr_frame_looks_valid(df_loaded):
            return df_loaded
    except Exception as exc:
        last_exc = exc

    if last_exc is not None and not candidate_sheets:
        raise last_exc

    raise ValueError(
        "Le classeur selectionne ne ressemble pas a un fichier IDSR agrege valide "
        "(colonnes geographiques, temporelles et indicateurs IDSR introuvables)."
    )


def render_idsr_tab(ctx: dict) -> None:
    """Render the weekly IDSR analytics tab."""
    globals().update(ctx)
    render_section_title(16, "IDSR — Surveillance agrégée hebdomadaire")
    render_tab_narrative("idsr")

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
    source_options = {
        "Téléverser un fichier": "upload",
        "Charger un fichier inclus": "local",
    }
    source_label = st.selectbox(
        "Source de données IDSR",
        options=list(source_options.keys()),
        index=0,
        key="idsr_source_mode",
    )
    idsr_source_mode = source_options[source_label]

    df_idsr = pd.DataFrame()
    src = None

    if idsr_source_mode == "upload":
        st.caption("Téléverser un fichier IDSR agrégé (.xlsx) depuis cet onglet.")
        up = st.file_uploader("Fichier IDSR agrégé", type=["xlsx", "xls"], key="idsr_upl")
        if up is not None:
            try:
                df_idsr = _load_idsr_workbook(up)
                src = getattr(up, "name", "upload") or "upload"
            except Exception as exc:
                st.error(f"Impossible de lire le fichier IDSR téléversé : {exc}")
    else:
        available_idsr_files = list_available_idsr_files(list_available_line_list_files())
        if available_idsr_files:
            preferred_local_path = guess_preferred_included_file(
                available_idsr_files,
                disease_key="idsr",
                default_sheet="IDS_RDC",
            )
            available_local_names = [path.name for path in available_idsr_files]
            if st.session_state.get("idsr_local_file") not in available_local_names:
                st.session_state["idsr_local_file"] = (
                    preferred_local_path.name if preferred_local_path is not None else available_local_names[0]
                )

            selected_local_name = st.selectbox(
                "Fichier IDSR disponible",
                options=available_local_names,
                key="idsr_local_file",
            )
            selected_local_path = next(
                (path for path in available_idsr_files if path.name == selected_local_name),
                None,
            )
            st.caption(get_line_list_bundle_caption())

            if selected_local_path is not None:
                try:
                    df_idsr = _load_idsr_workbook(selected_local_path)
                    src = selected_local_path.name
                except Exception as exc:
                    st.error(f"Impossible de lire le fichier IDSR local : {exc}")
        else:
            st.info("Aucun fichier Excel IDSR valide n'est disponible dans `line_list/`.")

    if df_idsr.empty:
        render_reader_narrative(
            "Données IDSR attendues",
            "Chargez un fichier IDSR agrégé pour afficher les analyses hebdomadaires. "
            "Tant que le fichier n'est pas chargé, aucune conclusion ne peut être tirée sur la situation.",
            tone="missing",
        )
    else:
        st.success(f"Fichier chargé : {src} | Lignes: {len(df_idsr):,}")

        # ---------------------------------------------------------------------
        # 2) Harmonisation colonnes (BRUT vs COMPILÉ)
        # ---------------------------------------------------------------------
        df_idsr = harmonize_idsr_columns(df_idsr.copy())

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
        df_idsr = normalize_idsr_debutsem_column(df_idsr)
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
                _dt_src = parse_idsr_date_series(df_idsr["Date_debut_semaine"])
            elif "DEBUTSEM" in df_idsr.columns:
                _dt_src = parse_idsr_date_series(df_idsr["DEBUTSEM"])

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
            src_dt = parse_idsr_date_series(df_idsr["Date_debut_semaine"])
        elif "DEBUTSEM" in df_idsr.columns:
            src_dt = parse_idsr_date_series(df_idsr["DEBUTSEM"])
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
        # TIME_KEY / TIME_LAB : tri et affichage analytique en Année-Semaine quand possible.
        # Le filtre semaine BRUT reste volontairement sur le numéro de semaine uniquement (voir bloc filtre).
        wnum_num = pd.to_numeric(df_idsr.get("Num_semaine_epid"), errors="coerce")
        year_num = pd.to_numeric(df_idsr.get("Annee_epid"), errors="coerce") if "Annee_epid" in df_idsr.columns else pd.Series(np.nan, index=df_idsr.index)

        df_idsr["TIME_LAB"] = _idsr_build_year_week_label_series(year_num, wnum_num)
        df_idsr["TIME_KEY"] = _idsr_build_year_week_key_series(year_num, wnum_num)

        _has_year_week = year_num.notna() & wnum_num.notna()
        df_idsr["YW"] = pd.Series(pd.NA, index=df_idsr.index, dtype="string")
        df_idsr.loc[_has_year_week, "YW"] = df_idsr.loc[_has_year_week, "TIME_LAB"].astype("string")
        df_idsr["YW_KEY"] = pd.to_numeric(np.where(_has_year_week, year_num * 100 + wnum_num, np.nan), errors="coerce")

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

        def _build_idsr_duplicate_week_label(_df: pd.DataFrame) -> pd.Series:
            if "YW" in _df.columns:
                _yw = _df["YW"].astype("string").str.strip()
                if _yw.notna().any():
                    return _yw

            if "Date_debut_semaine_iso" in _df.columns:
                _dt_iso = parse_idsr_date_series(_df["Date_debut_semaine_iso"])
                if _dt_iso.notna().any():
                    return _dt_iso.dt.strftime("%Y-%m-%d").astype("string")

            if "DEBUTSEM" in _df.columns:
                _debutsem = _df["DEBUTSEM"]
                _dt = parse_idsr_date_series(_debutsem)
                if _dt.notna().any():
                    return _dt.dt.strftime("%Y-%m-%d").astype("string")

            if ("Annee_epid" in _df.columns) or ("Num_semaine_epid" in _df.columns):
                _year_src = _df["Annee_epid"] if "Annee_epid" in _df.columns else pd.Series(pd.NA, index=_df.index)
                _week_src = _df["Num_semaine_epid"] if "Num_semaine_epid" in _df.columns else pd.Series(pd.NA, index=_df.index)
                _year = pd.to_numeric(_year_src, errors="coerce").astype("Int64")
                _week = pd.to_numeric(_week_src, errors="coerce").astype("Int64")
                _year_s = _year.astype("string")
                _week_s = _week.astype("string").str.zfill(2)

                _label = pd.Series(pd.NA, index=_df.index, dtype="string")
                _has_yw = _year.notna() & _week.notna()
                _has_w = _week.notna()

                _label.loc[_has_yw] = _year_s.loc[_has_yw] + "-W" + _week_s.loc[_has_yw]
                _label.loc[(~_has_yw) & _has_w] = "W" + _week_s.loc[(~_has_yw) & _has_w]
                if _label.notna().any():
                    return _label

            if "TIME_LAB" in _df.columns:
                _time_lab = _df["TIME_LAB"].astype("string").str.strip()
                if _time_lab.notna().any():
                    return _time_lab

            return pd.Series(pd.NA, index=_df.index, dtype="string")

        def _prepare_idsr_duplicate_views(_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
            _work = _df.copy()
            _work["_dup_week"] = _build_idsr_duplicate_week_label(_work)

            _dup_key_cols = [c for c in ["_dup_week", COL_PROV_ID, COL_ZS_ID, COL_MAL] if c in _work.columns]
            if len(_dup_key_cols) < 4:
                return _work, pd.DataFrame(), pd.DataFrame(), _dup_key_cols, []

            for _c in _dup_key_cols:
                _work[_c] = _work[_c].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()

            _work["_dup_key"] = _work[_dup_key_cols].fillna("").agg(" | ".join, axis=1)
            _work["_dup_key_valid"] = _work[_dup_key_cols].notna().all(axis=1)
            _work["_dup_key_valid"] = _work["_dup_key_valid"] & _work[_dup_key_cols].ne("").all(axis=1)

            _metric_cols = [
                c for c in [
                    "Population",
                    "Total_cas",
                    "Total_deces",
                    "Cas_tnn",
                    "Cas_0_11mois",
                    "Cas_12_59mois",
                    "Cas_5_14ans",
                    "Cas_15plus",
                    "Deces_tnn",
                    "Deces_0_11mois",
                    "Deces_12_59mois",
                    "Deces_5_14ans",
                    "Deces_15plus",
                    "Taux_letalite",
                    "Taux_attaque",
                ]
                if c in _work.columns
            ]

            for _c in _metric_cols:
                _work[_c] = pd.to_numeric(_work[_c], errors="coerce")

            if _metric_cols:
                _work["_dup_metric_signature"] = _work[_metric_cols].astype("string").fillna("").agg("|".join, axis=1)
            else:
                _work["_dup_metric_signature"] = "NA"

            _subset_exact = _dup_key_cols + ["_dup_metric_signature"]
            _work["duplicate_idsr_potential"] = (
                _work["_dup_key_valid"]
                & _work.duplicated(subset=_dup_key_cols, keep=False)
            )
            _work["duplicate_idsr_exact"] = (
                _work["_dup_key_valid"]
                & _work.duplicated(subset=_subset_exact, keep=False)
            )

            _detail = _work[_work["duplicate_idsr_potential"]].copy()
            if _detail.empty:
                return _work, _detail, pd.DataFrame(), _dup_key_cols, _metric_cols

            _agg_kwargs = {
                "Occurrences": ("_dup_key", "size"),
                "Distinct_metric_rows": ("_dup_metric_signature", "nunique"),
                "Exact_rows": ("duplicate_idsr_exact", "sum"),
            }
            if "UniqueKey" in _detail.columns:
                _agg_kwargs["UniqueKey_nunique"] = ("UniqueKey", lambda s: int(s.dropna().nunique()))

            _summary = (
                _detail.groupby(["_dup_key", *_dup_key_cols], dropna=False, as_index=False)
                .agg(**_agg_kwargs)
            )
            _summary["Type_doublon"] = np.where(
                _summary["Distinct_metric_rows"] > 1,
                "Contradictoire",
                "Exact métier",
            )

            if _metric_cols:
                def _list_changed_metrics(_sub: pd.DataFrame) -> str:
                    _changed = []
                    for _c in _metric_cols:
                        _vals = _sub[_c]
                        if _vals.drop_duplicates().shape[0] > 1:
                            _changed.append(_c)
                    if not _changed:
                        return ""
                    if len(_changed) > 6:
                        return ", ".join(_changed[:6]) + f", +{len(_changed) - 6} autre(s)"
                    return ", ".join(_changed)

                _changed_metrics = (
                    _detail.groupby("_dup_key", dropna=False)[_metric_cols]
                    .apply(_list_changed_metrics)
                    .reset_index(name="Variables_en_ecart")
                )
                _summary = _summary.merge(_changed_metrics, on="_dup_key", how="left")
            else:
                _summary["Variables_en_ecart"] = ""

            _detail = _detail.merge(
                _summary[["_dup_key", "Occurrences", "Distinct_metric_rows", "Type_doublon", "Variables_en_ecart"]],
                on="_dup_key",
                how="left",
            )

            _summary = _summary.sort_values(
                ["Distinct_metric_rows", "Occurrences", "_dup_week"],
                ascending=[False, False, True],
            ).reset_index(drop=True)

            return _work, _detail, _summary, _dup_key_cols, _metric_cols

        def _idsr_fmt_int(_value: Any) -> str:
            if pd.isna(_value):
                return "NA"
            try:
                return f"{int(round(float(_value))):,}"
            except Exception:
                return str(_value)

        def _idsr_fmt_pct(_value: Any, decimals: int = 1) -> str:
            if pd.isna(_value):
                return "NA"
            try:
                return f"{float(_value):.{decimals}f}%"
            except Exception:
                return str(_value)

        def _idsr_join_sentences(*_parts: Any) -> str:
            return " ".join(
                str(_part).strip()
                for _part in _parts
                if isinstance(_part, str) and str(_part).strip()
            )

        def _idsr_make_narrative(
            constat: str = "",
            interpretation: str = "",
            action: str = "",
        ) -> dict[str, str]:
            _narrative: dict[str, str] = {}
            if constat.strip():
                _narrative["constat"] = constat.strip()
            if interpretation.strip():
                _narrative["interpretation"] = interpretation.strip()
            if action.strip():
                _narrative["action"] = action.strip()
            return _narrative

        def _build_idsr_scope_narrative(_df: pd.DataFrame) -> dict[str, str]:
            if _df is None or _df.empty:
                return {}

            _cases = pd.to_numeric(_df.get("Total_cas"), errors="coerce").sum(skipna=True) if "Total_cas" in _df.columns else np.nan
            _deaths = pd.to_numeric(_df.get("Total_deces"), errors="coerce").sum(skipna=True) if "Total_deces" in _df.columns else np.nan
            _cfr = (float(_deaths) / float(_cases) * 100.0) if pd.notna(_cases) and _cases > 0 and pd.notna(_deaths) else np.nan
            _n_mal = int(_df[COL_MAL].nunique(dropna=True)) if COL_MAL in _df.columns else 0
            _n_prov = int(_df[COL_PROV_ID].nunique(dropna=True)) if COL_PROV_ID in _df.columns else 0
            _n_zs = int(_df[COL_ZS_ID].nunique(dropna=True)) if COL_ZS_ID in _df.columns else 0

            _constat = _idsr_join_sentences(
                f"Au terme du filtrage appliqué, le périmètre analytique totalise {_idsr_fmt_int(_cases)} cas et {_idsr_fmt_int(_deaths)} décès, pour une létalité recalculée de {_idsr_fmt_pct(_cfr)}.",
                f"La lecture agrégée porte sur {_idsr_fmt_int(_n_mal)} maladies, {_idsr_fmt_int(_n_prov)} provinces et {_idsr_fmt_int(_n_zs)} zones de santé.",
            )
            _interpretation_parts: list[str] = []

            if (COL_MAL in _df.columns) and ("Total_cas" in _df.columns):
                _top_mal = (
                    _df.groupby(COL_MAL, as_index=False)
                    .agg(Cas=("Total_cas", "sum"))
                    .sort_values("Cas", ascending=False)
                )
                if not _top_mal.empty:
                    _leader = _top_mal.iloc[0]
                    _share = (float(_leader["Cas"]) / float(_cases) * 100.0) if pd.notna(_cases) and _cases > 0 else np.nan
                    _interpretation_parts.append(
                        f"La maladie prédominante est {str(_leader[COL_MAL])}, avec {_idsr_fmt_int(_leader['Cas'])} cas, soit {_idsr_fmt_pct(_share)} du volume observé."
                    )

            if (COL_PROV_ID in _df.columns) and ("Total_cas" in _df.columns):
                _top_prov = (
                    _df.groupby(COL_PROV_ID, as_index=False)
                    .agg(Cas=("Total_cas", "sum"))
                    .sort_values("Cas", ascending=False)
                )
                if not _top_prov.empty:
                    _leader_p = _top_prov.iloc[0]
                    _share_p = (float(_leader_p["Cas"]) / float(_cases) * 100.0) if pd.notna(_cases) and _cases > 0 else np.nan
                    _interpretation_parts.append(
                        f"La province la plus affectée est {str(_leader_p[COL_PROV_ID])}, avec {_idsr_fmt_int(_leader_p['Cas'])} cas ({_idsr_fmt_pct(_share_p)} du total)."
                    )

            _interpretation = _idsr_join_sentences(*_interpretation_parts)
            _action = (
                "Documenter ce périmètre filtré dans tout partage opérationnel et croiser sa lecture avec la qualité des dates, les doublons et la cohérence des agrégats avant diffusion."
            )
            return _idsr_make_narrative(_constat, _interpretation, _action)

        def _build_idsr_latest_week_narrative(
            _df_last_week: pd.DataFrame,
            _week_label: Any,
            _cases: Any,
            _deaths: Any,
            _cfr: Any,
            _delta_cases: Any = None,
            _delta_deaths: Any = None,
            _diff_cases: Any = np.nan,
            _diff_deaths: Any = np.nan,
        ) -> dict[str, str]:
            if _df_last_week is None or _df_last_week.empty:
                return {}

            _constat_parts = [
                f"Pour la dernière semaine disponible ({_week_label}), le niveau agrégé observé est de {_idsr_fmt_int(_cases)} cas, {_idsr_fmt_int(_deaths)} décès et une létalité de {_idsr_fmt_pct(_cfr)}."
            ]

            if _delta_cases is not None and pd.notna(_delta_cases):
                _trend = "en augmentation" if float(_delta_cases) > 0 else ("en diminution" if float(_delta_cases) < 0 else "stables")
                _constat_parts.append(f"Par rapport à la semaine précédente, les cas sont {_trend} ({float(_delta_cases):+.1f}%).")

            if _delta_deaths is not None and pd.notna(_delta_deaths):
                _trend_d = "en augmentation" if float(_delta_deaths) > 0 else ("en diminution" if float(_delta_deaths) < 0 else "stables")
                _constat_parts.append(f"L’évolution des décès est {_trend_d} ({float(_delta_deaths):+.1f}%).")

            _interpretation_parts: list[str] = []

            if (COL_MAL in _df_last_week.columns) and ("Total_cas" in _df_last_week.columns):
                _top_mal = (
                    _df_last_week.groupby(COL_MAL, as_index=False)
                    .agg(Cas=("Total_cas", "sum"))
                    .sort_values("Cas", ascending=False)
                )
                if not _top_mal.empty:
                    _interpretation_parts.append(
                        f"La maladie qui contribue le plus à la charge hebdomadaire est {str(_top_mal.iloc[0][COL_MAL])} avec {_idsr_fmt_int(_top_mal.iloc[0]['Cas'])} cas."
                    )

            if (COL_PROV_ID in _df_last_week.columns) and ("Total_cas" in _df_last_week.columns):
                _top_prov = (
                    _df_last_week.groupby(COL_PROV_ID, as_index=False)
                    .agg(Cas=("Total_cas", "sum"))
                    .sort_values("Cas", ascending=False)
                )
                if not _top_prov.empty:
                    _interpretation_parts.append(
                        f"La province la plus contributive sur cette semaine est {str(_top_prov.iloc[0][COL_PROV_ID])} avec {_idsr_fmt_int(_top_prov.iloc[0]['Cas'])} cas."
                    )

            if (pd.notna(_diff_cases) and float(_diff_cases) != 0) or (pd.notna(_diff_deaths) and float(_diff_deaths) != 0):
                _interpretation_parts.append(
                    "Le contrôle de cohérence met en évidence un écart entre les totaux notifiés et la somme des tranches d’âge ; cette semaine doit donc être interprétée avec prudence."
                )
                _action = (
                    "Documenter en priorité l’écart entre totaux et tranches d’âge, puis valider les maladies citées dans le détail avant toute diffusion définitive."
                )
            else:
                _action = (
                    "Confirmer rapidement les provinces et maladies les plus contributives afin d’orienter la réponse hebdomadaire et le message de bulletin."
                )

            return _idsr_make_narrative(
                _idsr_join_sentences(*_constat_parts),
                _idsr_join_sentences(*_interpretation_parts),
                _action,
            )

        def _build_idsr_date_quality_narrative(_counts: pd.Series) -> dict[str, str]:
            if _counts is None or _counts.empty:
                return {}
            _total = int(_counts.sum())
            _ok = int(_counts.get("✅ OK", 0))
            _ko = int(_counts.get("❌ KO", 0))
            _na = int(_counts.get("NA", 0))
            _constat = (
                f"Le contrôle de concordance entre date et semaine porte sur {_idsr_fmt_int(_total)} lignes : {_idsr_fmt_int(_ok)} sont cohérentes, {_idsr_fmt_int(_ko)} incohérentes et {_idsr_fmt_int(_na)} non évaluables."
            )
            if _ko > 0 and _total > 0:
                _interpretation = (
                    f"La proportion de lignes incohérentes est de {_idsr_fmt_pct((_ko / _total) * 100.0)} ; ces enregistrements peuvent fragiliser la lecture chronologique."
                )
                _action = (
                    "Afficher les lignes KO, corriger la date source ou la semaine épidémiologique, puis relancer la lecture temporelle avant consolidation."
                )
            elif _na > 0:
                _interpretation = (
                    "Aucune incohérence n’est visible parmi les lignes évaluables, mais une partie des enregistrements reste non évaluable faute d’information temporelle exploitable."
                )
                _action = (
                    "Compléter les champs de date ou de semaine manquants pour sécuriser la lecture des tendances et des cumuls."
                )
            else:
                _interpretation = "La concordance date-semaine est satisfaisante sur le périmètre actuellement analysé."
                _action = "Maintenir ce contrôle avant chaque diffusion hebdomadaire pour prévenir les décalages de calendrier."
            return _idsr_make_narrative(_constat, _interpretation, _action)

        def _build_idsr_duplicate_narrative(_summary: pd.DataFrame) -> dict[str, str]:
            if _summary is None or _summary.empty:
                return _idsr_make_narrative(
                    "Aucun doublon métier n’a été détecté sur le périmètre filtré.",
                    "Le jeu affiché paraît stable sur la clé Semaine + Province + Zone de santé + Maladie.",
                    "Conserver ce contrôle avant les analyses comparatives et les exports afin de détecter rapidement toute réapparition.",
                )

            _n_groups = int(len(_summary))
            _n_exact = int((_summary["Type_doublon"] == "Exact métier").sum()) if "Type_doublon" in _summary.columns else 0
            _n_contrad = int((_summary["Type_doublon"] == "Contradictoire").sum()) if "Type_doublon" in _summary.columns else 0
            _constat = _idsr_join_sentences(
                f"{_idsr_fmt_int(_n_groups)} groupe(s) de doublons métier ont été identifiés dans le périmètre filtré.",
                f"{_idsr_fmt_int(_n_exact)} groupe(s) correspondent à des répétitions exactes de même contenu." if _n_exact > 0 else "",
                f"{_idsr_fmt_int(_n_contrad)} groupe(s) présentent des valeurs contradictoires pour une même clé métier." if _n_contrad > 0 else "",
            )
            _interpretation = _idsr_join_sentences(
                "Les doublons exacts peuvent gonfler artificiellement les volumes agrégés." if _n_exact > 0 else "",
                "Les doublons contradictoires signalent plutôt des versions concurrentes d’un même enregistrement et nécessitent une validation humaine." if _n_contrad > 0 else "",
            )
            _action = (
                "Examiner en priorité les groupes contradictoires avant diffusion ; les doublons exacts peuvent être exclus des analyses sans masquer les anomalies métier."
                if _n_contrad > 0
                else "Vous pouvez exclure les doublons exacts des analyses, tout en conservant ce contrôle comme garde-fou avant export."
            )
            return _idsr_make_narrative(_constat, _interpretation, _action)

        def _build_idsr_coherence_narrative(_qc_view: pd.DataFrame) -> dict[str, str]:
            if _qc_view is None or _qc_view.empty or "QC_Global" not in _qc_view.columns:
                return {}
            _total = int(len(_qc_view))
            _ko = int((_qc_view["QC_Global"] == "❌ KO").sum())
            _ok = int((_qc_view["QC_Global"] == "✅ OK").sum())
            _constat = (
                f"Le contrôle de cohérence des agrégats couvre {_idsr_fmt_int(_total)} lignes : {_idsr_fmt_int(_ok)} sont cohérentes et {_idsr_fmt_int(_ko)} présentent au moins un écart."
            )
            if _ko > 0:
                _interpretation = (
                    f"La part de lignes incohérentes atteint {_idsr_fmt_pct((_ko / _total) * 100.0)} ; les totaux notifiés ne correspondent donc pas toujours aux sommes par tranche d’âge."
                )
                _action = (
                    "Examiner les lignes KO et les maladies responsables des écarts avant de consolider les cas et décès dans un bulletin ou un export."
                )
            else:
                _interpretation = "Les totaux notifiés concordent avec les sommes par tranche d’âge sur le périmètre analysé."
                _action = "Maintenir ce contrôle comme étape standard avant publication afin de sécuriser la lecture des agrégats."
            return _idsr_make_narrative(_constat, _interpretation, _action)

        def _build_idsr_profile_narrative(_df: pd.DataFrame) -> dict[str, str]:
            if _df is None or _df.empty:
                return {}

            _cases = pd.to_numeric(_df.get("Total_cas"), errors="coerce").sum(skipna=True) if "Total_cas" in _df.columns else np.nan
            _constat_parts: list[str] = []

            if (COL_MAL in _df.columns) and ("Total_cas" in _df.columns):
                _by_mal = (
                    _df.groupby(COL_MAL, as_index=False)
                    .agg(Cas=("Total_cas", "sum"))
                    .sort_values("Cas", ascending=False)
                )
                if not _by_mal.empty:
                    _leader = _by_mal.iloc[0]
                    _share = (float(_leader["Cas"]) / float(_cases) * 100.0) if pd.notna(_cases) and _cases > 0 else np.nan
                    _constat_parts.append(
                        f"Dans le profil global, {str(_leader[COL_MAL])} est la maladie la plus représentée, avec {_idsr_fmt_int(_leader['Cas'])} cas ({_idsr_fmt_pct(_share)} du total)."
                    )

            _age_map = {
                "Cas_tnn": "<1 mois",
                "Cas_0_11mois": "0–11 mois",
                "Cas_12_59mois": "12–59 mois",
                "Cas_5_14ans": "5–14 ans",
                "Cas_15plus": "≥15 ans",
            }
            _age_rows = []
            for _col, _label in _age_map.items():
                if _col in _df.columns:
                    _age_rows.append((_label, pd.to_numeric(_df[_col], errors="coerce").sum(skipna=True)))
            if _age_rows:
                _age_df = pd.DataFrame(_age_rows, columns=["Tranche", "Cas"]).sort_values("Cas", ascending=False)
                if not _age_df.empty and pd.notna(_age_df.iloc[0]["Cas"]) and float(_age_df.iloc[0]["Cas"]) > 0:
                    _constat_parts.append(
                        f"La tranche d’âge la plus représentée est {_age_df.iloc[0]['Tranche']}, avec {_idsr_fmt_int(_age_df.iloc[0]['Cas'])} cas."
                    )

            if (COL_PROV_ID in _df.columns) and ("Total_cas" in _df.columns):
                _by_prov = (
                    _df.groupby(COL_PROV_ID, as_index=False)
                    .agg(Cas=("Total_cas", "sum"))
                    .sort_values("Cas", ascending=False)
                )
                if not _by_prov.empty:
                    _constat_parts.append(
                        f"Sur le plan géographique, {str(_by_prov.iloc[0][COL_PROV_ID])} concentre le volume le plus élevé, avec {_idsr_fmt_int(_by_prov.iloc[0]['Cas'])} cas."
                    )

            _constat = _idsr_join_sentences(*_constat_parts)
            _interpretation = (
                "Le profil observé suggère une concentration de la charge sur les groupes et territoires cités ci-dessus ; cette lecture doit être croisée avec la tendance hebdomadaire et la létalité."
                if _constat
                else ""
            )
            _action = (
                "Orienter la revue épidémiologique, clinique et programmatique vers la maladie dominante, la tranche d’âge la plus touchée et les territoires les plus contributeurs."
                if _constat
                else ""
            )
            return _idsr_make_narrative(_constat, _interpretation, _action)

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
                _debutsem_dt = parse_idsr_date_series(_debutsem)

                years_available = sorted(idsr_year_from_debutsem(_debutsem, mode="iso").dropna().astype(int).unique().tolist())

                if years_available:
                    years_selected = st.multiselect(
                        "Année épid. (DEBUTSEM)",
                        options=years_available,
                        default=years_available,
                        key="tab9_years_debutsem",
                        help="Année ISO/épidémiologique reconstruite depuis DEBUTSEM. Exemple : lundi 29 décembre 2025 correspond à 2026-W01."
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
            _yrs = idsr_year_from_debutsem(_debutsem, mode="iso")
            df9 = df9[_yrs.isin([int(y) for y in years_selected])]
        elif years_selected and ("Annee_epid" in df9.columns):
            # fallback si DEBUTSEM absent
            df9 = df9[pd.to_numeric(df9["Annee_epid"], errors="coerce").isin([int(y) for y in years_selected])]


        # Filtre semaines selon mode
        # Copie avant filtre semaines: utile pour 'Situation épidémiologique — dernière semaine disponible' (focus sur semaine max)
        df9_base = df9.copy()

        # Filtre semaines selon mode
        if week_filter_mode == "YW":
            df9["YW_KEY"] = pd.to_numeric(df9["YW_KEY"], errors="coerce")
            df9 = df9[df9["YW_KEY"].between(min_key, max_key, inclusive="both")]

        elif week_filter_mode == "WNUM":
            df9["Num_semaine_epid"] = pd.to_numeric(df9["Num_semaine_epid"], errors="coerce")
            df9 = df9[df9["Num_semaine_epid"].between(w_min, w_max, inclusive="both")]

        df9_duplicates_scope = df9.copy()
        dup_scope_work, dup_scope_detail, dup_scope_summary, dup_scope_key_cols, _ = _prepare_idsr_duplicate_views(df9_duplicates_scope)

        exclude_exact_idsr_dups = st.checkbox(
            "Exclure les doublons exacts métier des analyses IDSR",
            value=False,
            key="tab9_exclude_exact_idsr_dups",
            help="Supprime des KPI et graphiques les répétitions strictement identiques sur une même Semaine + Province + ZS + Maladie, tout en conservant les doublons contradictoires pour revue.",
        )

        removed_exact_rows = 0
        contradictory_groups_retained = 0
        if exclude_exact_idsr_dups and dup_scope_key_cols:
            dup_base_work, _, _, dup_base_key_cols, _ = _prepare_idsr_duplicate_views(df9_base)
            if dup_base_key_cols:
                df9_base = dup_base_work.loc[
                    ~(dup_base_work["_dup_key_valid"] & dup_base_work.duplicated(subset=dup_base_key_cols + ["_dup_metric_signature"], keep="first"))
                ].copy()
                df9_base = df9_base.drop(
                    columns=[
                        "_dup_week", "_dup_key", "_dup_key_valid", "_dup_metric_signature",
                        "duplicate_idsr_potential", "duplicate_idsr_exact",
                    ],
                    errors="ignore",
                )

            df9 = dup_scope_work.loc[
                ~(dup_scope_work["_dup_key_valid"] & dup_scope_work.duplicated(subset=dup_scope_key_cols + ["_dup_metric_signature"], keep="first"))
            ].copy()
            removed_exact_rows = len(dup_scope_work) - len(df9)
            contradictory_groups_retained = int((dup_scope_summary.get("Type_doublon", pd.Series(dtype="object")) == "Contradictoire").sum())
            df9 = df9.drop(
                columns=[
                    "_dup_week", "_dup_key", "_dup_key_valid", "_dup_metric_signature",
                    "duplicate_idsr_potential", "duplicate_idsr_exact",
                ],
                errors="ignore",
            )
            if removed_exact_rows > 0:
                st.info(
                    f"Déduplication analytique appliquée : {removed_exact_rows:,} ligne(s) exacte(s) retirée(s) des analyses ; "
                    f"{contradictory_groups_retained:,} groupe(s) contradictoire(s) restent visibles pour revue."
                )
        else:
            df9 = df9.copy()

        st.caption(f"📌 Périmètre analytique filtré : {len(df9):,} lignes")
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
        # 8.b) Résumé de la période filtrée (confort utilisateur)
        # ---------------------------------------------------------------------
        if not df9.empty:
            _tot_cas = pd.to_numeric(df9.get("Total_cas"), errors="coerce").sum(skipna=True) if "Total_cas" in df9.columns else np.nan
            _tot_dec = pd.to_numeric(df9.get("Total_deces"), errors="coerce").sum(skipna=True) if "Total_deces" in df9.columns else np.nan
            _cfr = (float(_tot_dec) / float(_tot_cas) * 100.0) if (pd.notna(_tot_cas) and _tot_cas > 0 and pd.notna(_tot_dec)) else np.nan

            _n_mal = df9[COL_MAL].nunique(dropna=True) if COL_MAL in df9.columns else 0
            _n_prov = df9[COL_PROV_ID].nunique(dropna=True) if COL_PROV_ID in df9.columns else 0
            _n_zs = df9[COL_ZS_ID].nunique(dropna=True) if COL_ZS_ID in df9.columns else 0

            _period_label, _time_span = _build_idsr_period_labels(df9)

            st.markdown("### 01 · Synthèse rapide du périmètre")
            r1, r2, r3, r4, r5, r6 = st.columns(6)
            r1.metric("Cas (total)", f"{int(_tot_cas):,}" if pd.notna(_tot_cas) else "NA")
            r2.metric("Décès (total)", f"{int(_tot_dec):,}" if pd.notna(_tot_dec) else "NA")
            r3.metric("CFR (recalculé)", f"{_cfr:.2f}%" if pd.notna(_cfr) else "NA")
            r4.metric("Maladies", f"{_n_mal:,}")
            r5.metric("Provinces", f"{_n_prov:,}")
            r6.metric("Zones de santé", f"{_n_zs:,}")
            st.caption(
                f"Période couverte : **{format_range_label_for_display(_period_label)}** | "
                f"Fenêtre hebdo : **{format_range_label_for_display(_time_span)}**"
            )
            _scope_narrative = _build_idsr_scope_narrative(df9)
            if _scope_narrative:
                render_reader_narrative("Lecture du périmètre", _scope_narrative)


        if df9.empty:
            st.info("Aucune donnée n’est disponible après application des filtres analytiques.")
        else:
            st.divider()
            st.markdown("### 02 · Vue d’ensemble guidée")
            st.caption(
                "Les blocs ci-dessous sont organisés du contrôle de complétude vers l’interprétation épidémiologique, "
                "puis vers la qualité des données et les exports."
            )
            render_idsr_reading_guide()

            # -----------------------------------------------------------------
            # 8.c) Complétude IDSR : nombre de ZS rapportantes par province/semaine
            # -----------------------------------------------------------------
            render_idsr_completeness_section(
                df=df9,
                province_col=COL_PROV_ID,
                zs_col=COL_ZS_ID if COL_ZS_ID in df9.columns else "Zone_de_sante_notification",
            )
            st.divider()

            # -----------------------------------------------------------------
            # 8.d) Taux d'attaque et incidence : cas rapportés / population
            # -----------------------------------------------------------------
            render_idsr_attack_incidence_section(
                df=df9,
                province_col=COL_PROV_ID,
                zs_col=COL_ZS_ID if COL_ZS_ID in df9.columns else None,
                mal_col=COL_MAL,
            )
            st.divider()

            render_idsr_phase_header(
                "04 · Cartographie et concentration géographique",
                "Localiser les provinces et zones de santé les plus contributives après vérification de la complétude."
            )
            render_idsr_maps_section(
                df_f=df9,
                province_col=COL_PROV_ID,
                zs_col=COL_ZS_ID if COL_ZS_ID in df9.columns else None,
                cases_col="Total_cas",
            )

            st.divider()

            # -----------------------------------------------------------------
            # 9) Série temporelle : cas/décès/CFR (robuste sur TIME_KEY/LAB)
            # -----------------------------------------------------------------
            required_cols = ["Total_cas", "Total_deces"]
            missing = [c for c in required_cols if c not in df9.columns]
            if missing:
                st.error(f"Variables manquantes pour l’analyse temporelle : {', '.join(missing)}")
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
                # - Ligne 1 : somme des tranches d’âge (Cas_* / Deces_*) => détecte incohérences
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
                # KPI dernière semaine + variation vs semaine-1

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

                            lab_last = (
                                df_last_kpi["TIME_LAB"].iloc[0]
                                if "TIME_LAB" in df_last_kpi.columns and not df_last_kpi.empty
                                else _idsr_format_year_week_label(last_year, w_max)
                            )
                            last = {"TIME_LAB": lab_last, "Cas": cas_last, "Deces": dec_last, "CFR_calc_%": cfr_last}
                            prev = {"Cas": cas_prev, "Deces": dec_prev, "CFR_calc_%": cfr_prev} if not df_prev_kpi.empty else None

                # Fallback: si on n'a rien trouvé, on garde l'ancien comportement
                if last is None and len(weekly_sorted) >= 1:
                    last = weekly_sorted.iloc[-1]
                    prev = weekly_sorted.iloc[-2] if len(weekly_sorted) >= 2 else None

                d_cas = None if ("disable_deltas" in locals() and disable_deltas) else (pct_change_metric_safe(last["Cas"], prev["Cas"]) if (last is not None and prev is not None) else None)
                d_dec = None if ("disable_deltas" in locals() and disable_deltas) else (pct_change_metric_safe(last["Deces"], prev["Deces"]) if (last is not None and prev is not None) else None)
                d_cfr = None if ("disable_deltas" in locals() and disable_deltas) else (pct_change_metric_safe(last["CFR_calc_%"], prev["CFR_calc_%"]) if (last is not None and prev is not None) else None)

                st.markdown("### Dernière semaine disponible")

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

                                last_lab_focus = (
                                    df_last_week["TIME_LAB"].iloc[0]
                                    if "TIME_LAB" in df_last_week.columns and not df_last_week.empty
                                    else _idsr_format_year_week_label(last_year, w_max)
                                )

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
                        st.info("Plusieurs années ont été détectées pour cette semaine (mode brut / WNUM) : les variations par rapport à la semaine précédente sont désactivées.")
                        disable_deltas = True

                # 2) Affichage métriques "tranches d'âge" pour la semaine max
                if not df_last_week.empty and age_case_cols and age_death_cols:
                    cas_age_last = df_last_week[age_case_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0, skipna=True).sum()
                    dec_age_last = df_last_week[age_death_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0, skipna=True).sum()
                    cfr_age_last = (float(dec_age_last) / float(cas_age_last) * 100.0) if (pd.notna(cas_age_last) and cas_age_last > 0 and pd.notna(dec_age_last)) else np.nan

                    cas_age_prev = df_prev_week[age_case_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0, skipna=True).sum() if (not df_prev_week.empty) else np.nan
                    dec_age_prev = df_prev_week[age_death_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0, skipna=True).sum() if (not df_prev_week.empty) else np.nan
                    cfr_age_prev = (float(dec_age_prev) / float(cas_age_prev) * 100.0) if (pd.notna(cas_age_prev) and cas_age_prev > 0 and pd.notna(dec_age_prev)) else np.nan

                    d_cas_a = None if ("disable_deltas" in locals() and disable_deltas) else pct_change_metric_safe(cas_age_last, cas_age_prev)
                    d_dec_a = None if ("disable_deltas" in locals() and disable_deltas) else pct_change_metric_safe(dec_age_last, dec_age_prev)
                    d_cfr_a = None if ("disable_deltas" in locals() and disable_deltas) else pct_change_metric_safe(cfr_age_last, cfr_age_prev)

                    st.caption("Ligne 1 : somme des tranches d’âge (Cas_* / Deces_*)")
                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("Semaine", str(last_lab_focus))
                    a2.metric("Cas (tranches)", f"{int(cas_age_last):,}" if pd.notna(cas_age_last) else "NA", delta=None if d_cas_a is None else f"{d_cas_a:.1f}% vs semaine-1")
                    a3.metric("Décès (tranches)", f"{int(dec_age_last):,}" if pd.notna(dec_age_last) else "NA", delta=None if d_dec_a is None else f"{d_dec_a:.1f}% vs semaine-1")
                    a4.metric("CFR (tranches)", f"{cfr_age_last:.2f}%" if pd.notna(cfr_age_last) else "NA", delta=None if d_cfr_a is None else f"{d_cfr_a:.1f}% vs semaine-1")
                else:
                    st.caption("Ligne 1 : somme des tranches d’âge (Cas_* / Deces_*) — indisponible (colonnes manquantes ou aucune donnée)")

                
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

                d_cas_t = None if ("disable_deltas" in locals() and disable_deltas) else pct_change_metric_safe(tot_cas_lastwk, tot_cas_prevwk)
                d_dec_t = None if ("disable_deltas" in locals() and disable_deltas) else pct_change_metric_safe(tot_dec_lastwk, tot_dec_prevwk)
                d_cfr_t = None if ("disable_deltas" in locals() and disable_deltas) else pct_change_metric_safe(cfr_tot_lastwk, cfr_tot_prevwk)

                st.caption("Ligne 2 : totaux notifiés (TOTALCAS / TOTALDECES)")
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
                _latest_narrative = _build_idsr_latest_week_narrative(
                    df_last_week,
                    last_lab_focus,
                    tot_cas_lastwk,
                    tot_dec_lastwk,
                    cfr_tot_lastwk,
                    d_cas_t,
                    d_dec_t,
                    diff_cas,
                    diff_dec,
                )
                if _latest_narrative:
                    render_reader_narrative("Lecture de la dernière semaine", _latest_narrative, tone="decision")

                disease_gap_table = pd.DataFrame()
                if (
                    not df_last_week.empty
                    and (COL_MAL in df_last_week.columns)
                    and age_case_cols
                    and age_death_cols
                    and ("Total_cas" in df_last_week.columns)
                    and ("Total_deces" in df_last_week.columns)
                ):
                    disease_gap_table = df_last_week[[COL_MAL, "Total_cas", "Total_deces", *age_case_cols, *age_death_cols]].copy()
                    disease_gap_table["Total_cas"] = pd.to_numeric(disease_gap_table["Total_cas"], errors="coerce")
                    disease_gap_table["Total_deces"] = pd.to_numeric(disease_gap_table["Total_deces"], errors="coerce")
                    disease_gap_table["Cas_tranches"] = disease_gap_table[age_case_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
                    disease_gap_table["Deces_tranches"] = disease_gap_table[age_death_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)

                    disease_gap_table = (
                        disease_gap_table.groupby(COL_MAL, as_index=False)
                        .agg(
                            Cas_totaux=("Total_cas", "sum"),
                            Cas_tranches=("Cas_tranches", "sum"),
                            Deces_totaux=("Total_deces", "sum"),
                            Deces_tranches=("Deces_tranches", "sum"),
                        )
                    )
                    disease_gap_table["Ecart_cas"] = disease_gap_table["Cas_totaux"] - disease_gap_table["Cas_tranches"]
                    disease_gap_table["Ecart_deces"] = disease_gap_table["Deces_totaux"] - disease_gap_table["Deces_tranches"]
                    disease_gap_table["Pct_ecart_cas"] = np.where(
                        disease_gap_table["Cas_tranches"] != 0,
                        (disease_gap_table["Ecart_cas"] / disease_gap_table["Cas_tranches"]) * 100.0,
                        np.nan,
                    )
                    disease_gap_table["Pct_ecart_deces"] = np.where(
                        disease_gap_table["Deces_tranches"] != 0,
                        (disease_gap_table["Ecart_deces"] / disease_gap_table["Deces_tranches"]) * 100.0,
                        np.nan,
                    )
                    disease_gap_table = disease_gap_table[
                        (disease_gap_table["Ecart_cas"].fillna(0) != 0)
                        | (disease_gap_table["Ecart_deces"].fillna(0) != 0)
                    ].copy()

                def _format_gap_diseases(_gap_df: pd.DataFrame, _diff_col: str, _pct_col: str, _limit: int = 8) -> str:
                    if _gap_df.empty or (_diff_col not in _gap_df.columns):
                        return ""

                    _subset = _gap_df[_gap_df[_diff_col].fillna(0) != 0].copy()
                    if _subset.empty:
                        return ""

                    _subset = _subset.sort_values(_diff_col, key=lambda s: s.abs(), ascending=False)
                    _labels = []
                    for _, _row in _subset.head(_limit).iterrows():
                        _mal = str(_row.get(COL_MAL, "Maladie non renseignée")).strip() or "Maladie non renseignée"
                        _diff = _row.get(_diff_col, np.nan)
                        _pct = _row.get(_pct_col, np.nan)
                        if pd.notna(_pct):
                            _labels.append(f"{_mal} ({_diff:+,.1f}; {_pct:+.1f}%)")
                        else:
                            _labels.append(f"{_mal} ({_diff:+,.1f})")

                    _remaining = len(_subset) - min(len(_subset), _limit)
                    if _remaining > 0:
                        _labels.append(f"+{_remaining} autre(s)")

                    return ", ".join(_labels)

                if pd.notna(diff_cas) and pd.notna(diff_dec):
                    if (diff_cas == 0) and (diff_dec == 0):
                        st.success("Aucun écart détecté : TOTALCAS/TOTALDECES correspond à la somme des tranches d’âge sur la semaine maximale.")
                    else:
                        pct_cas = (diff_cas / cas_age_last * 100.0) if ("cas_age_last" in locals() and pd.notna(cas_age_last) and cas_age_last != 0) else np.nan
                        pct_dec = (diff_dec / dec_age_last * 100.0) if ("dec_age_last" in locals() and pd.notna(dec_age_last) and dec_age_last != 0) else np.nan
                        case_diseases = _format_gap_diseases(disease_gap_table, "Ecart_cas", "Pct_ecart_cas")
                        death_diseases = _format_gap_diseases(disease_gap_table, "Ecart_deces", "Pct_ecart_deces")

                        error_lines = [
                            "❌ Écart détecté (Totaux − Tranches) – semaine max : "
                            f"Cas={diff_cas:+,} ({pct_cas:.1f}%) | Décès={diff_dec:+,} ({pct_dec:.1f}%)"
                        ]
                        if case_diseases:
                            error_lines.append(f"Cas concernés : {case_diseases}")
                        if death_diseases:
                            error_lines.append(f"Décès concernés : {death_diseases}")
                        st.error("\n".join(error_lines))

                        if not disease_gap_table.empty:
                            disease_gap_display = (
                                disease_gap_table.rename(columns={
                                    COL_MAL: "Maladie",
                                    "Cas_totaux": "Cas totaux",
                                    "Cas_tranches": "Cas tranches",
                                    "Ecart_cas": "Écart cas",
                                    "Pct_ecart_cas": "% écart cas",
                                    "Deces_totaux": "Décès totaux",
                                    "Deces_tranches": "Décès tranches",
                                    "Ecart_deces": "Écart décès",
                                    "Pct_ecart_deces": "% écart décès",
                                })
                                .sort_values(["Écart cas", "Écart décès"], key=lambda s: s.abs(), ascending=False)
                            )
                            st.dataframe(
                                disease_gap_display,
                                width="stretch",
                                hide_index=True,
                                column_config={
                                    "% écart cas": st.column_config.NumberColumn(format="%.1f%%"),
                                    "% écart décès": st.column_config.NumberColumn(format="%.1f%%"),
                                    "Cas totaux": st.column_config.NumberColumn(format="%.0f"),
                                    "Cas tranches": st.column_config.NumberColumn(format="%.0f"),
                                    "Écart cas": st.column_config.NumberColumn(format="%.0f"),
                                    "Décès totaux": st.column_config.NumberColumn(format="%.0f"),
                                    "Décès tranches": st.column_config.NumberColumn(format="%.0f"),
                                    "Écart décès": st.column_config.NumberColumn(format="%.0f"),
                                },
                            )
                else:
                    st.info("Écart non calculable : variables manquantes ou données insuffisantes.")

                # Note: cette section est volontairement centrée sur la semaine max,
                # même si l'utilisateur change semaine min.
                st.divider()
                with st.expander("05 · 🚨 Alertes et classements — signaux opérationnels", expanded=False):

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

                        with st.expander("05.a · 📈 Provinces en hausse", expanded=False):
                            n_up = st.slider("Nombre d’unités à afficher", 5, 50, 15, step=5, key="tab9_n_up_prov")
                            st.dataframe(prov_delta.head(n_up), width="stretch", height=420, hide_index=True)
                    else:
                        st.info("Le classement des provinces en hausse est indisponible : variable Province absente ou historique insuffisant.")

                    c3, c4 = st.columns(2)

                    with c3:
                        if COL_PROV_ID in df9.columns and "Total_cas" in df9.columns and "Total_deces" in df9.columns:
                            top_prov = df9.groupby(COL_PROV_ID, as_index=False).agg(
                                Cas=("Total_cas", "sum"),
                                Deces=("Total_deces", "sum")
                            )
                            top_prov["CFR_%"] = np.where(top_prov["Cas"] > 0, (top_prov["Deces"] / top_prov["Cas"]) * 100, np.nan)
                            top_prov = top_prov.sort_values("Cas", ascending=False)

                            with st.expander("05.b · 🏥 Provinces les plus touchées", expanded=False):
                                n_prov = st.slider("Nombre de provinces à afficher", 10, 200, 20, step=10, key="tab9_n_top_prov")
                                st.dataframe(top_prov.head(n_prov), width="stretch", height=420, hide_index=True)
                        else:
                            top_prov = None
                            st.info("Le classement des provinces est indisponible : variables requises manquantes.")

                    with c4:
                        if (COL_PROV_ID in df9.columns) and (COL_ZS_ID in df9.columns) and ("Total_cas" in df9.columns) and ("Total_deces" in df9.columns):
                            top_zs = df9.groupby([COL_PROV_ID, COL_ZS_ID], as_index=False).agg(
                                Cas=("Total_cas", "sum"),
                                Deces=("Total_deces", "sum")
                            )
                            top_zs["CFR_%"] = np.where(top_zs["Cas"] > 0, (top_zs["Deces"] / top_zs["Cas"]) * 100, np.nan)
                            top_zs = top_zs.sort_values("Cas", ascending=False)

                            with st.expander("05.c · 🗺️ Zones de santé les plus touchées", expanded=False):
                                n_zs = st.slider("Nombre de ZS à afficher", 10, 300, 20, step=10, key="tab9_n_top_zs")
                                st.dataframe(top_zs.head(n_zs), width="stretch", height=420, hide_index=True)
                        else:
                            top_zs = None
                            st.info("Le classement des zones de santé est indisponible : variables requises manquantes.")

                st.markdown("### Qualité des données")
                st.caption(
                    "Cette partie aide à vérifier la cohérence temporelle, les doublons et les écarts entre totaux déclarés et tranches d’âge."
                )
                with st.expander("06.a · 🗓️ Qualité des dates — concordance date/semaine", expanded=False):
                    if "QC_Date_vs_Semaine" in df9.columns:
                        qc_date_counts = df9["QC_Date_vs_Semaine"].value_counts(dropna=False)
                        st.write(qc_date_counts)
                        _date_narrative = _build_idsr_date_quality_narrative(qc_date_counts)
                        if _date_narrative:
                            render_reader_narrative("Lecture des dates", _date_narrative)

                        qc_date_filter = st.radio(
                            "Afficher les lignes",
                            ["Tous", "✅ OK", "❌ KO", "NA"],
                            horizontal=True,
                            key="tab9_qc_date_filter",
                        )

                        qc_date_view = df9.copy()
                        if qc_date_filter != "Tous":
                            qc_date_view = qc_date_view[qc_date_view["QC_Date_vs_Semaine"] == qc_date_filter]

                        qc_date_show_cols = [
                            c for c in [
                                "QC_Date_vs_Semaine",
                                "TIME_LAB",
                                "Date_debut_semaine_iso",
                                "DEBUTSEM",
                                "Date_debut_semaine",
                                "Annee_epid",
                                "Num_semaine_epid",
                                "YW",
                                COL_MAL,
                                COL_PROV_ID,
                                COL_ZS_ID,
                                "Total_cas",
                                "Total_deces",
                            ]
                            if c in qc_date_view.columns
                        ]

                        if qc_date_view.empty:
                            st.caption("Aucune ligne ne correspond au filtre sélectionné.")
                        else:
                            st.caption(f"Lignes affichées : {len(qc_date_view):,}")
                            st.dataframe(
                                qc_date_view[qc_date_show_cols],
                                width="stretch",
                                height=280,
                                hide_index=True,
                            )
                            st.download_button(
                                "⬇️ Télécharger lignes QC date/semaine (CSV)",
                                data=qc_date_view[qc_date_show_cols].to_csv(index=False).encode("utf-8"),
                                file_name="idsr_qc_date_vs_semaine.csv",
                                mime="text/csv",
                                key="tab9_dl_qc_date_scope_csv",
                            )
                    else:
                        st.info("Le contrôle qualité temporel est indisponible : dates sources absentes.")

                with st.expander("06.b · 🧬 Doublons possibles — revue métier", expanded=False):
                    if len(dup_scope_key_cols) < 4:
                        st.info("Recherche de doublons indisponible : colonnes clés semaine/province/ZS/maladie insuffisantes.")
                    elif dup_scope_summary.empty:
                        st.success("Aucun doublon potentiel détecté sur le périmètre IDSR filtré.")
                    else:
                        dup_contradictions = dup_scope_detail[dup_scope_detail["Type_doublon"] == "Contradictoire"].copy()
                        k_dup1, k_dup2, k_dup3, k_dup4 = st.columns(4)
                        k_dup1.metric("Lignes dupliquées", f"{len(dup_scope_detail):,}")
                        k_dup2.metric("Groupes dupliqués", f"{len(dup_scope_summary):,}")
                        k_dup3.metric(
                            "Groupes contradictoires",
                            f"{int((dup_scope_summary['Type_doublon'] == 'Contradictoire').sum()):,}",
                        )
                        k_dup4.metric(
                            "Groupes exacts",
                            f"{int((dup_scope_summary['Type_doublon'] == 'Exact métier').sum()):,}",
                        )

                        st.caption(
                            "Clé utilisée : Semaine + Province + Zone de santé + Maladie. "
                            "`Exact métier` = mêmes indicateurs agrégés; `Contradictoire` = mêmes clés mais valeurs différentes."
                        )
                        _dup_narrative = _build_idsr_duplicate_narrative(dup_scope_summary)
                        if _dup_narrative:
                            render_reader_narrative("Lecture des doublons", _dup_narrative)
                        if exclude_exact_idsr_dups and removed_exact_rows > 0:
                            st.caption(
                                f"Les analyses affichées plus haut excluent déjà {removed_exact_rows:,} ligne(s) exact(e)s. "
                                "Cette section continue d'afficher le diagnostic sur le périmètre filtré avant déduplication analytique."
                            )

                        dup_type_sel = st.radio(
                            "Type de doublons à afficher",
                            ["Tous", "Contradictoire", "Exact métier"],
                            horizontal=True,
                            key="tab9_dup_type_sel",
                        )

                        dup_summary_view = dup_scope_summary.copy()
                        if dup_type_sel != "Tous":
                            dup_summary_view = dup_summary_view[dup_summary_view["Type_doublon"] == dup_type_sel]

                        dup_summary_display = dup_summary_view.rename(columns={
                            "_dup_week": "Semaine",
                            COL_PROV_ID: "Province",
                            COL_ZS_ID: "Zone de santé",
                            COL_MAL: "Maladie",
                            "Type_doublon": "Type de doublon",
                            "Distinct_metric_rows": "Versions métriques",
                            "UniqueKey_nunique": "UniqueKey distincts",
                            "Variables_en_ecart": "Variables en écart",
                            "Exact_rows": "Lignes exactes",
                        })
                        dup_summary_export = dup_scope_summary.rename(columns={
                            "_dup_week": "Semaine",
                            COL_PROV_ID: "Province",
                            COL_ZS_ID: "Zone de santé",
                            COL_MAL: "Maladie",
                            "Type_doublon": "Type de doublon",
                            "Distinct_metric_rows": "Versions métriques",
                            "UniqueKey_nunique": "UniqueKey distincts",
                            "Variables_en_ecart": "Variables en écart",
                            "Exact_rows": "Lignes exactes",
                        })

                        st.dataframe(
                            dup_summary_display[
                                [
                                    c for c in [
                                        "Semaine",
                                        "Province",
                                        "Zone de santé",
                                        "Maladie",
                                        "Occurrences",
                                        "Lignes exactes",
                                        "Type de doublon",
                                        "Versions métriques",
                                        "UniqueKey distincts",
                                        "Variables en écart",
                                    ]
                                    if c in dup_summary_display.columns
                                ]
                            ],
                            width="stretch",
                            height=320,
                            hide_index=True,
                        )

                        if not dup_contradictions.empty:
                            st.markdown("**Lignes contradictoires à revoir en priorité**")
                            contradiction_cols = [
                                c for c in [
                                    "_dup_week",
                                    COL_PROV_ID,
                                    COL_ZS_ID,
                                    COL_MAL,
                                    "Occurrences",
                                    "Variables_en_ecart",
                                    "Population",
                                    "Total_cas",
                                    "Total_deces",
                                    "Cas_tnn",
                                    "Cas_0_11mois",
                                    "Cas_12_59mois",
                                    "Cas_5_14ans",
                                    "Cas_15plus",
                                    "Deces_tnn",
                                    "Deces_0_11mois",
                                    "Deces_12_59mois",
                                    "Deces_5_14ans",
                                    "Deces_15plus",
                                    "Taux_letalite",
                                    "Taux_attaque",
                                    "RecStatus",
                                    "UniqueKey",
                                ]
                                if c in dup_contradictions.columns
                            ]
                            contradiction_display = dup_contradictions[contradiction_cols].rename(columns={
                                "_dup_week": "Semaine",
                                COL_PROV_ID: "Province",
                                COL_ZS_ID: "Zone de santé",
                                COL_MAL: "Maladie",
                                "Variables_en_ecart": "Variables en écart",
                            })
                            st.dataframe(
                                contradiction_display,
                                width="stretch",
                                height=260,
                                hide_index=True,
                            )

                        show_dup_lines = st.checkbox(
                            "Afficher toutes les lignes dupliquées",
                            value=False,
                            key="tab9_show_dup_lines",
                        )
                        if show_dup_lines:
                            dup_detail_display = dup_scope_detail.rename(columns={
                                "_dup_week": "Semaine",
                                COL_PROV_ID: "Province",
                                COL_ZS_ID: "Zone de santé",
                                COL_MAL: "Maladie",
                                "Type_doublon": "Type de doublon",
                                "Variables_en_ecart": "Variables en écart",
                                "Distinct_metric_rows": "Versions métriques",
                            })
                            st.dataframe(
                                dup_detail_display[
                                    [
                                        c for c in [
                                            "Semaine",
                                            "Province",
                                            "Zone de santé",
                                            "Maladie",
                                            "Type de doublon",
                                            "Occurrences",
                                            "Versions métriques",
                                            "Variables en écart",
                                            "Population",
                                            "Total_cas",
                                            "Total_deces",
                                            "Cas_tnn",
                                            "Cas_0_11mois",
                                            "Cas_12_59mois",
                                            "Cas_5_14ans",
                                            "Cas_15plus",
                                            "Deces_tnn",
                                            "Deces_0_11mois",
                                            "Deces_12_59mois",
                                            "Deces_5_14ans",
                                            "Deces_15plus",
                                            "Taux_letalite",
                                            "Taux_attaque",
                                            "RecStatus",
                                            "UniqueKey",
                                        ]
                                        if c in dup_detail_display.columns
                                    ]
                                ],
                                width="stretch",
                                height=360,
                                hide_index=True,
                            )

                        dup_summary_csv = dup_summary_export.to_csv(index=False).encode("utf-8")
                        dup_detail_csv = dup_scope_detail.rename(columns={
                            "_dup_week": "Semaine",
                            COL_PROV_ID: "Province",
                            COL_ZS_ID: "Zone de santé",
                            COL_MAL: "Maladie",
                            "Type_doublon": "Type de doublon",
                            "Variables_en_ecart": "Variables en écart",
                            "Distinct_metric_rows": "Versions métriques",
                        }).to_csv(index=False).encode("utf-8")

                        dup_buffer = BytesIO()
                        with pd.ExcelWriter(dup_buffer, engine="openpyxl") as writer:
                            dup_summary_export.to_excel(writer, sheet_name="Doublons_resume", index=False)
                            dup_scope_detail.rename(columns={
                                "_dup_week": "Semaine",
                                COL_PROV_ID: "Province",
                                COL_ZS_ID: "Zone de santé",
                                COL_MAL: "Maladie",
                                "Type_doublon": "Type de doublon",
                                "Variables_en_ecart": "Variables en écart",
                                "Distinct_metric_rows": "Versions métriques",
                            }).to_excel(writer, sheet_name="Doublons_detail", index=False)
                            if not dup_contradictions.empty:
                                dup_contradictions.rename(columns={
                                    "_dup_week": "Semaine",
                                    COL_PROV_ID: "Province",
                                    COL_ZS_ID: "Zone de santé",
                                    COL_MAL: "Maladie",
                                    "Type_doublon": "Type de doublon",
                                    "Variables_en_ecart": "Variables en écart",
                                    "Distinct_metric_rows": "Versions métriques",
                                }).to_excel(writer, sheet_name="Doublons_contrad", index=False)

                        d1, d2, d3 = st.columns(3)
                        with d1:
                            st.download_button(
                                "⬇️ Télécharger résumé doublons (CSV)",
                                data=dup_summary_csv,
                                file_name="idsr_doublons_resume.csv",
                                mime="text/csv",
                                key="tab9_dl_dup_summary_csv",
                            )
                        with d2:
                            st.download_button(
                                "⬇️ Télécharger détail doublons (CSV)",
                                data=dup_detail_csv,
                                file_name="idsr_doublons_detail.csv",
                                mime="text/csv",
                                key="tab9_dl_dup_detail_csv",
                            )
                        with d3:
                            st.download_button(
                                "⬇️ Télécharger pack doublons (Excel)",
                                data=dup_buffer.getvalue(),
                                file_name="idsr_doublons_pack.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="tab9_dl_dup_pack_xlsx",
                            )

            # -----------------------------------------------------------------
            # 12) Contrôles cohérence totaux vs tranches d’âge
            # -----------------------------------------------------------------
            with st.expander("06.c · ✅ Cohérence des totaux et des tranches d’âge", expanded=False):
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
                _coherence_narrative = _build_idsr_coherence_narrative(qc_view)
                if _coherence_narrative:
                    render_reader_narrative("Lecture de la cohérence des totaux", _coherence_narrative)

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
                st.markdown("#### Filtres")
                f1, f2, f3, f4 = st.columns(4)

                with f1:
                    qc_global_sel = st.selectbox("Contrôle qualité global", options=["Tous", "✅ OK", "❌ KO"], index=0, key="tab9_qc_global_sel")
                with f2:
                    qc_cas_sel = st.selectbox("Contrôle qualité des cas", options=["Tous", "✅ OK", "❌ KO"], index=0, key="tab9_qc_cas_sel")
                with f3:
                    qc_deces_sel = st.selectbox("Contrôle qualité des décès", options=["Tous", "✅ OK", "❌ KO"], index=0, key="tab9_qc_deces_sel")
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

                st.caption(f"📌 Lignes après application des filtres de contrôle qualité : {len(table_to_show):,}")

                # Colonnes QC à afficher
                qc_cols = ["QC_Global", "QC_Cas", "QC_Deces"]
                qc_cols = [c for c in qc_cols if c in table_to_show.columns]
                final_cols = qc_cols + cols_show

                if show_qc_tables:
                    with st.expander("🧾 Voir le détail des lignes OK/KO", expanded=False):
                        st.dataframe(
                        table_to_show[final_cols].style.apply(style_qc, axis=1),
                        width="stretch",
                        height=520,
                        hide_index=True
                    )
                
            st.markdown("### 07 · Analyses détaillées")
            st.caption(
                "Les rubriques suivantes approfondissent la lecture par âge, maladie et territoire. "
                "Elles complètent la vue d’ensemble sans la remplacer."
            )
            _profile_narrative = _build_idsr_profile_narrative(df9)
            if _profile_narrative:
                render_reader_narrative("Lecture du profil épidémiologique", _profile_narrative)

            # ---------------------------------------------------------------------
            # 14) IDSR – Spécifications des sorties
            # ---------------------------------------------------------------------
            # 14.1) Histogramme des cas + courbe de létalité (CFR%)
            with st.expander("07.a · 📈 Évolution hebdomadaire des cas, décès et létalité", expanded=True):
                if 'weekly_sorted' in locals() and isinstance(weekly_sorted, pd.DataFrame) and not weekly_sorted.empty:
                    _wk = weekly_sorted.copy()
                    # Libellé unique Année-Semaine (évite doublons W01/W02 quand plusieurs années)
                    if "YW" in _wk.columns:
                        _wk["_X_LAB"] = _wk["YW"].astype(str)
                    elif "TIME_KEY" in _wk.columns:
                        _wk["_X_LAB"] = _wk["TIME_KEY"].astype(str)
                    else:
                        _wk["_X_LAB"] = _wk.get("TIME_LAB", pd.Series(dtype="object")).astype(str)
                    # fmt_yw_label est centralisée dans dashboard_app.core
                    _wk["_X_LAB"] = _wk["_X_LAB"].map(fmt_yw_label)

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
                            bargap=0.04,
                            bargroupgap=0.02,
                            margin=dict(t=70, b=60, l=60, r=60),
                            height=420,
                        )
                        fig_cas_cfr = apply_plotly_value_annotations(fig_cas_cfr, annot_vals)
                        st.plotly_chart(fig_cas_cfr, width="stretch", key="idsr_hist_cas_cfr")
                    else:
                        st.info("Variables insuffisantes pour tracer l’évolution hebdomadaire (TIME_LAB/Cas).")
                else:
                    st.info("Aucune donnée hebdomadaire agrégée n’est disponible après filtrage.")

            # 14.2) Camembert par tranche d'âge + tableau associé
            with st.expander("07.b · 👶 Répartition des cas par âge", expanded=False):
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
                        st.info("Aucune variable 'Cas_*' par tranche d’âge n’a été trouvée dans les données IDSR.")
                else:
                    st.info("Aucune donnée n’est disponible après filtrage pour produire la répartition par âge.")


            # 14.2.b) Analyses descriptives IDSR inspirées des listes linéaires
            with st.expander("07.c · 👥 Détail par maladie, âge et semaine", expanded=False):
                st.caption(
                    "Ces analyses détaillent la distribution par maladie, âge et semaine. "
                    "Les approfondissements géographiques ont été recentrés dans la cartographie et les classements pour alléger la lecture."
                )

                # -------------------------------------------------------------
                # A) Profil par maladie
                # -------------------------------------------------------------
                st.markdown("#### A. Maladies les plus fréquentes")
                if (COL_MAL in df9.columns) and ("Total_cas" in df9.columns):
                    df_mal_profile = (
                        df9.groupby(COL_MAL, as_index=False)
                        .agg(Cas=("Total_cas", "sum"), Deces=("Total_deces", "sum") if "Total_deces" in df9.columns else ("Total_cas", "size"))
                    )
                    if "Total_deces" not in df9.columns:
                        df_mal_profile["Deces"] = 0
                    df_mal_profile["CFR_%"] = np.where(
                        df_mal_profile["Cas"] > 0,
                        (df_mal_profile["Deces"] / df_mal_profile["Cas"]) * 100.0,
                        np.nan,
                    )
                    df_mal_profile = df_mal_profile.sort_values("Cas", ascending=False)

                    c_m1, c_m2 = st.columns([1.2, 1])
                    with c_m1:
                        fig_mal = px.bar(
                            df_mal_profile,
                            x=COL_MAL,
                            y="Cas",
                            title="Cas cumulés par maladie",
                            text="Cas",
                        )
                        fig_mal.update_layout(template="plotly_white", xaxis_tickangle=-35, height=420)
                        fig_mal = apply_plotly_value_annotations(fig_mal, annot_vals)
                        st.plotly_chart(fig_mal, width="stretch", key="idsr_profile_maladie")
                    with c_m2:
                        st.dataframe(
                            df_mal_profile.assign(**{"CFR_%": df_mal_profile["CFR_%"].round(2)}),
                            width="stretch",
                            height=420,
                            hide_index=True,
                        )
                else:
                    st.info("Le profil par maladie est indisponible : colonnes Maladie ou Total_cas absentes.")

                st.divider()

                # -------------------------------------------------------------
                # B) Structure d'âge par maladie (équivalent profil population)
                # -------------------------------------------------------------
                st.markdown("#### B. Âge des cas selon la maladie")
                age_case_cols = [c for c in ["Cas_tnn", "Cas_0_11mois", "Cas_12_59mois", "Cas_5_14ans", "Cas_15plus"] if c in df9.columns]
                age_label_map = {
                    "Cas_tnn": "<1 mois",
                    "Cas_0_11mois": "0–11 mois",
                    "Cas_12_59mois": "12–59 mois",
                    "Cas_5_14ans": "5–14 ans",
                    "Cas_15plus": "≥15 ans",
                }
                if (COL_MAL in df9.columns) and age_case_cols:
                    age_mal = (
                        df9.groupby(COL_MAL, as_index=False)[age_case_cols]
                        .sum(min_count=1)
                    )
                    age_mal_long = age_mal.melt(
                        id_vars=[COL_MAL],
                        value_vars=age_case_cols,
                        var_name="Tranche_source",
                        value_name="Cas",
                    )
                    age_mal_long["Tranche_age"] = age_mal_long["Tranche_source"].map(age_label_map)
                    age_mal_long["Cas"] = pd.to_numeric(age_mal_long["Cas"], errors="coerce").fillna(0)
                    age_mal_long = age_mal_long[age_mal_long["Cas"] > 0]

                    if not age_mal_long.empty:
                        ordre_age = ["<1 mois", "0–11 mois", "12–59 mois", "5–14 ans", "≥15 ans"]
                        age_mal_long["Tranche_age"] = pd.Categorical(age_mal_long["Tranche_age"], categories=ordre_age, ordered=True)

                        c_a1, c_a2 = st.columns([1.3, 1])
                        with c_a1:
                            fig_age_mal = px.bar(
                                age_mal_long.sort_values([COL_MAL, "Tranche_age"]),
                                x=COL_MAL,
                                y="Cas",
                                color="Tranche_age",
                                barmode="stack",
                                title="Cas par maladie et tranche d’âge",
                            )
                            fig_age_mal.update_layout(template="plotly_white", xaxis_tickangle=-35, height=460)
                            st.plotly_chart(fig_age_mal, width="stretch", key="idsr_profile_age_maladie")
                        with c_a2:
                            age_mal_tab = (
                                age_mal_long.pivot_table(
                                    index=COL_MAL,
                                    columns="Tranche_age",
                                    values="Cas",
                                    aggfunc="sum",
                                    fill_value=0,
                                    observed=False,
                                )
                                .reset_index()
                            )
                            st.dataframe(age_mal_tab, width="stretch", height=460, hide_index=True)
                    else:
                        st.info("Les colonnes d’âge existent mais ne contiennent pas de volume exploitable après filtrage.")
                else:
                    st.info("La structure par tranche d’âge selon la maladie est indisponible : colonnes Cas_* ou Maladie absentes.")

                st.divider()

                st.divider()

                # -------------------------------------------------------------
                # C) Profil hebdomadaire par maladie (équivalent dynamique par groupe)
                # -------------------------------------------------------------
                st.markdown("#### C. Évolution hebdomadaire par maladie")
                if (COL_MAL in df9.columns) and ("Total_cas" in df9.columns) and ("TIME_LAB" in df9.columns) and ("TIME_KEY" in df9.columns):
                    wk_mal = (
                        df9.groupby(["TIME_LAB", "TIME_KEY", COL_MAL], as_index=False)
                        .agg(Cas=("Total_cas", "sum"))
                        .sort_values(["TIME_KEY", COL_MAL])
                    )
                    if not wk_mal.empty:
                        fig_wk_mal = px.line(
                            wk_mal,
                            x="TIME_LAB",
                            y="Cas",
                            color=COL_MAL,
                            markers=True,
                            title="Évolution hebdomadaire des cas par maladie",
                        )
                        fig_wk_mal.update_layout(template="plotly_white", xaxis_tickangle=-45, height=460)
                        st.plotly_chart(fig_wk_mal, width="stretch", key="idsr_weekly_by_disease")

                        with st.expander("Voir le tableau par maladie", expanded=False):
                            wk_mal_wide = wk_mal.pivot_table(
                                index="TIME_LAB",
                                columns=COL_MAL,
                                values="Cas",
                                aggfunc="sum",
                                fill_value=0,
                                observed=False,
                            ).reset_index()
                            st.dataframe(wk_mal_wide, width="stretch", height=420, hide_index=True)
                    else:
                        st.info("Aucune série hebdomadaire exploitable n’est disponible pour les maladies après filtrage.")
                else:
                    st.info("La dynamique hebdomadaire par maladie est indisponible : colonnes TIME_LAB/TIME_KEY/Maladie/Total_cas absentes.")

            st.markdown("### 08 · Tableaux et exports")
            st.caption(
                "Utilisez ces tableaux pour la lecture détaillée, les vérifications territoriales et les exports analytiques."
            )

            # 14.3) Tableau d’évolution par province et semaine épidémiologique
            with st.expander("08.a · 📋 Tableau par province et semaine", expanded=False):

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
                        st.info("Variables manquantes pour produire le tableau province × semaine (YW/TIME_KEY/TIME_LAB).")
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
                    st.info("Aucune donnée n’est disponible après filtrage pour produire le tableau province × semaine.")

            # 14.4) Tableau croisé – totaux mensuels (Province / ZS)
            with st.expander("08.b · 📋 Tableau mensuel par province et zone de santé", expanded=False):

                if df9.empty:
                    st.info("Aucune donnée n’est disponible après application des filtres analytiques.")
                else:
                    # ---------------------------------------------------------
                    # 1) Construire une date source robuste
                    # ---------------------------------------------------------
                    def _get_date_series(_df: pd.DataFrame) -> pd.Series:
                        # Priorité : Date_debut_semaine_iso (déjà calculée)
                        if "Date_debut_semaine_iso" in _df.columns:
                            s = parse_idsr_date_series(_df["Date_debut_semaine_iso"])
                            if s.notna().any():
                                return s

                        # Sinon Date_debut_semaine si dispo
                        if "Date_debut_semaine" in _df.columns:
                            s = parse_idsr_date_series(_df["Date_debut_semaine"])
                            if s.notna().any():
                                return s

                        # Sinon DEBUTSEM (Excel serial ou date)
                        if "DEBUTSEM" in _df.columns:
                            _debutsem = _df["DEBUTSEM"]
                            s = parse_idsr_date_series(_debutsem)
                            if s.notna().any():
                                return s

                        return pd.Series(pd.NaT, index=_df.index)

                    tmp_m = df9.copy()
                    tmp_m["_dt"] = _get_date_series(tmp_m)

                    if tmp_m["_dt"].isna().all():
                        st.warning("Impossible de construire les mois : aucune date exploitable n’a été détectée (Date_debut_semaine_iso / Date_debut_semaine / DEBUTSEM).")
                    else:
                        # ---------------------------------------------------------
                        # 1bis) (Optionnel) filtrer dates absurdes pour éviter 1965/2037
                        # ---------------------------------------------------------
                        dt_min = pd.Timestamp("2000-01-01")
                        dt_max = pd.Timestamp.today() + pd.Timedelta(days=366)
                        tmp_m = tmp_m[tmp_m["_dt"].between(dt_min, dt_max)]

                        if tmp_m.empty:
                            st.warning("Toutes les dates disponibles sont hors de la plage attendue (2000 → année courante + 1). Veuillez vérifier DEBUTSEM/Date_debut_semaine.")
                        else:
                            # Mois (timestamp)
                            tmp_m["_month"] = tmp_m["_dt"].dt.to_period("M").dt.to_timestamp()

                            # Libellé mois en FR: "janv.-2024" (IMPORTANT: %Y pour éviter collisions 1924 vs 2024)
                            mois_fr = {
                                1: "janv.", 2: "févr.", 3: "mars", 4: "avr.", 5: "mai", 6: "juin",
                                7: "juil.", 8: "août", 9: "sept.", 10: "oct.", 11: "nov.", 12: "déc."
                            }
                            tmp_m["_month_lab"] = tmp_m["_dt"].dt.month.map(mois_fr) + "-" + tmp_m["_dt"].dt.strftime("%Y")

                            # ---------------------------------------------------------
                            # 2) Choix niveau: Province / Province+ZS
                            # ---------------------------------------------------------
                            level_m = st.radio(
                                "Niveau d’affichage",
                                ["Provincial", "Zonal (Province + ZS)"],
                                horizontal=True,
                                key="idsr_month_level",
                            )

                            # Colonnes id
                            col_mal = "Maladie" if "Maladie" in tmp_m.columns else COL_MAL
                            col_prov = COL_PROV_ID
                            col_zs = COL_ZS_ID if (COL_ZS_ID in tmp_m.columns) else None

                            idx_cols = [col_mal, col_prov]
                            if (level_m.startswith("Zonal")) and (col_zs is not None):
                                idx_cols = [col_mal, col_prov, col_zs]

                            # ---------------------------------------------------------
                            # 3) Indicateurs à produire
                            # ---------------------------------------------------------
                            metrics = [
                                ("Population", "Population exposée", "max"),
                                ("Cas_0_11mois", "Cas suspects 0 à 11mois", "sum"),
                                ("Cas_12_59mois", "Cas suspects 12mois à 5ans", "sum"),
                                ("Cas_5_14ans", "Cas suspects 5 à 14ans", "sum"),
                                ("Cas_15plus", "Cas suspects Adultes", "sum"),
                                ("Total_deces", "Nombre de décès", "sum"),
                            ]

                            # Garder uniquement les métriques existantes
                            metrics_ok = [(c, lab, agg) for (c, lab, agg) in metrics if c in tmp_m.columns]
                            if not metrics_ok:
                                st.info("Aucune colonne indicateur trouvée (Population / Cas_* / Total_deces).")
                            else:
                                # Préparer valeurs numériques
                                for c, _, _ in metrics_ok:
                                    tmp_m[c] = pd.to_numeric(tmp_m[c], errors="coerce")

                                # -----------------------------------------------------
                                # 4) Construire une table longue puis pivot mensuel
                                # -----------------------------------------------------
                                pieces = []
                                group_base = idx_cols + ["_month", "_month_lab"]

                                for c, lab, agg in metrics_ok:
                                    g = tmp_m[group_base + [c]].copy()

                                    if agg == "max":
                                        out = g.groupby(group_base, as_index=False)[c].max()
                                    else:
                                        out = g.groupby(group_base, as_index=False)[c].sum(min_count=1)

                                    out = out.rename(columns={c: "Valeur"})
                                    out["Données"] = lab
                                    pieces.append(out)

                                long_df = pd.concat(pieces, ignore_index=True)

                                pivot = (
                                    long_df.pivot_table(
                                        index=idx_cols + ["Données"],
                                        columns="_month",
                                        values="Valeur",
                                        aggfunc="sum",
                                        fill_value=0,
                                        observed=False,
                                    )
                                    .reset_index()
                                )

                                # Mapping mois timestamp -> label "janv.-2024"
                                month_map = (
                                    long_df.dropna(subset=["_month"])
                                    .drop_duplicates(subset=["_month"])[["_month", "_month_lab"]]
                                    .sort_values("_month")
                                    .set_index("_month")["_month_lab"]
                                    .to_dict()
                                )

                                new_cols = []
                                for col in pivot.columns:
                                    if isinstance(col, (pd.Timestamp, datetime)):
                                        new_cols.append(month_map.get(pd.Timestamp(col), pd.Timestamp(col).strftime("%b-%Y")))
                                    else:
                                        new_cols.append(col)
                                pivot.columns = new_cols

                                # Tri logique des lignes "Données"
                                order_data = [
                                    "Population exposée",
                                    "Cas suspects 0 à 11mois",
                                    "Cas suspects 12mois à 5ans",
                                    "Cas suspects 5 à 14ans",
                                    "Cas suspects Adultes",
                                    "Nombre de décès",
                                ]
                                pivot["Données"] = pd.Categorical(pivot["Données"], categories=order_data, ordered=True)
                                pivot = pivot.sort_values(idx_cols + ["Données"]).reset_index(drop=True)

                                # -----------------------------------------------------
                                # 5) IMPORTANT: rendre les colonnes uniques (Streamlit/PyArrow)
                                # -----------------------------------------------------
                                def _make_unique(cols):
                                    seen = {}
                                    out = []
                                    for x in cols:
                                        x = str(x)
                                        if x not in seen:
                                            seen[x] = 0
                                            out.append(x)
                                        else:
                                            seen[x] += 1
                                            out.append(f"{x}__{seen[x]}")
                                    return out

                                pivot.columns = _make_unique(pivot.columns)

                                st.dataframe(pivot, width="stretch", height=520, hide_index=True)

                                # -----------------------------------------------------
                                # 6) Exportation CSV / XLSX (colonnes déjà uniques)
                                # -----------------------------------------------------
                                csv_m = pivot.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "⬇️ Télécharger (mensuel) – CSV",
                                    data=csv_m,
                                    file_name="idsr_tableau_mensuel.csv",
                                    mime="text/csv",
                                    key="tab9_dl_monthly_pivot",
                                )

                                xlsx_buffer = BytesIO()
                                with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
                                    pivot.to_excel(writer, sheet_name="Tableau_mensuel", index=False)
                                xlsx_buffer.seek(0)

                                st.download_button(
                                    "⬇️ Télécharger (mensuel) – XLSX",
                                    data=xlsx_buffer,
                                    file_name="idsr_tableau_mensuel.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="tab9_dl_monthly_pivot_xlsx",
                                )


