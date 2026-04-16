from dashboard_app.overview import *

def _make_unique_columns(cols):
    seen = {}
    out = []
    for c in cols:
        c = str(c)
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out

@st.cache_data(show_spinner=False)
def _read_excel_from_bytes_cached(
    file_bytes: bytes,
    sheet_name: str | None,
    engine: str,
    kwargs_items: tuple
) -> pd.DataFrame:
    """
    Cache interne stable:
    - clé basée sur bytes du fichier + sheet_name + engine + kwargs hashables
    """
    kwargs = dict(kwargs_items)

    bio = BytesIO(file_bytes)
    with pd.ExcelFile(bio, engine=engine) as xls:
        chosen = None

        # 1) feuille demandée (exacte ou insensible à la casse)
        if sheet_name:
            if sheet_name in xls.sheet_names:
                chosen = sheet_name
            else:
                low = {s.lower(): s for s in xls.sheet_names}
                chosen = low.get(str(sheet_name).lower())

        # 2) fallback: première feuille
        if chosen is None:
            chosen = xls.sheet_names[0] if xls.sheet_names else None

        if chosen is None:
            return pd.DataFrame()

        df = pd.read_excel(xls, sheet_name=chosen, engine=engine, **kwargs)

    df.columns = _make_unique_columns(df.columns)
    return df

@st.cache_data(show_spinner=False)
def _read_excel_from_path_cached(
    path_str: str,
    mtime_ns: int,
    sheet_name: str | None,
    engine: str,
    kwargs_items: tuple
) -> pd.DataFrame:
    """
    Cache interne pour chemins locaux:
    clé = path + mtime (change si fichier mis à jour) + params
    """
    p = Path(path_str)
    file_bytes = p.read_bytes()
    return _read_excel_from_bytes_cached(file_bytes, sheet_name, engine, kwargs_items)

def load_excel_cached(file, sheet_name=None, engine="openpyxl", **kwargs) -> pd.DataFrame:
    """
    Lecture Excel avec cache pour accélérer l'app (supporte UploadedFile, file-like ou chemin local).
    - sheet_name manquant -> fallback première feuille
    - colonnes uniques
    """
    # kwargs doivent être hashables pour cache_data
    kwargs_items = tuple(sorted(kwargs.items(), key=lambda x: x[0]))

    # Cas 1: Streamlit UploadedFile
    if hasattr(file, "getvalue") and callable(file.getvalue):
        b = file.getvalue()
        if not b:
            return pd.DataFrame()
        return _read_excel_from_bytes_cached(b, sheet_name, engine, kwargs_items)

    # Cas 2: chemin local
    if isinstance(file, (str, Path)):
        p = Path(file)
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable: {p}")
        mtime_ns = p.stat().st_mtime_ns
        return _read_excel_from_path_cached(str(p), mtime_ns, sheet_name, engine, kwargs_items)

    # Cas 3: file-like (BytesIO / handle)
    if hasattr(file, "read") and callable(file.read):
        try:
            pos = file.tell()
        except Exception:
            pos = None
        b = file.read()
        try:
            if pos is not None:
                file.seek(pos)
        except Exception:
            pass
        if not b:
            return pd.DataFrame()
        return _read_excel_from_bytes_cached(b, sheet_name, engine, kwargs_items)

    raise TypeError("load_excel_cached: 'file' doit être UploadedFile, str/Path ou file-like.")


def clean_week(series: pd.Series) -> pd.Series:
    """
    Nettoie une colonne semaine:
    - extrait digits
    - cast Int64
    - borne 1..53
    """
    s = series.astype("string").str.extract(r"(\d{1,2})", expand=False)
    w = pd.to_numeric(s, errors="coerce").astype("Int64")
    return w.where((w >= 1) & (w <= 53), pd.NA)

def clean_year(series: pd.Series) -> pd.Series:
    """
    Nettoie une colonne année:
    - extrait YYYY
    - cast Int64
    - borne 2000..2100 (adaptable)
    """
    s = series.astype("string").str.extract(r"((?:19|20)\d{2})", expand=False)
    y = pd.to_numeric(s, errors="coerce").astype("Int64")
    # RDC: on limite pour éviter 1960/1900 issus d'erreurs Excel ou parsing
    return y.where((y >= 2000) & (y <= 2100), pd.NA)

def parse_year_from_filename(path_or_name: str | None):
    """Extrait une année YYYY depuis un nom de fichier si disponible."""
    if not path_or_name:
        return None
    m = re.search(r"(19|20)\d{2}", str(path_or_name))
    return int(m.group()) if m else None

def iso_monday_from_year_week(y, w):
    """
    Construit le lundi ISO depuis (année ISO, semaine ISO).
    Renvoie pd.NaT si invalide.
    """
    try:
        if pd.isna(y) or pd.isna(w):
            return pd.NaT
        return pd.Timestamp(date.fromisocalendar(int(y), int(w), 1))
    except Exception:
        return pd.NaT

def norm_text(series: pd.Series) -> pd.Series:
    """
    Normalise du texte pour réduire les doublons:
    - cast string
    - strip
    - espaces multiples -> 1 espace
    - option: uppercase (souvent utile IDSR)
    """
    s = series.astype("string")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    # Option robuste (souvent recommandé IDSR):
    s = s.str.upper()
    # Option: enlever les points/virgules parasites (sans casser les / -)
    s = s.str.replace(r"[^\w\s\-/]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def to_numeric_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Convertit une liste de colonnes en numeric si elles existent."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _score_quantile_0_100(s: pd.Series, q_low=0.05, q_high=0.95) -> pd.Series:
    """Transforme une série numérique en score 0–100 via quantiles (robuste aux extrêmes)."""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 3:
        return pd.Series([np.nan]*len(s), index=s.index)

    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series([np.nan]*len(s), index=s.index)

    x = s.clip(lo, hi)
    return (x - lo) / (hi - lo) * 100.0

def _completeness_pct(df: pd.DataFrame, required_cols: list) -> float:
    """% de complétude moyenne sur champs clés (0–100)."""
    cols = [c for c in required_cols if c in df.columns]
    if not cols:
        return np.nan
    # proportion non-NA par colonne, puis moyenne
    per_col = [df[c].notna().mean() for c in cols]
    return float(np.mean(per_col) * 100.0)

def _timeliness_pct(df: pd.DataFrame, date_onset: str, date_notif: str, threshold_days: int = 2) -> float:
    """% de cas avec (Date_notification - Date_debut_maladie) <= threshold_days."""
    if (date_onset not in df.columns) or (date_notif not in df.columns):
        return np.nan

    d1 = pd.to_datetime(df[date_onset], errors="coerce")
    d2 = pd.to_datetime(df[date_notif], errors="coerce")
    dd = (d2 - d1).dt.days

    valid = dd.notna()
    if valid.sum() < 5:
        return np.nan

    return float((dd[valid] <= threshold_days).mean() * 100.0)

def compute_irep_province(
    df: pd.DataFrame,
    *,
    col_prov: str = "Province_notification",
    col_week: str = "Semaine_epid",         # ex: "2026-W07"
    col_cases: str = "Total_cas",           # si line list -> tu peux pré-agréger en count
    col_deaths: str = "Total_deces",
    current_week: str = None,               # ex "2026-W07" ; si None -> prend la dernière
    population_map: dict = None,            # {"Kinshasa": 17000000, ...}
    date_onset: str = "Date_debut_maladie",
    date_notif: str = "Date_notification",
    threshold_days: int = 2,
    completeness_required: list = None,
    w: dict = None                          # poids
) -> pd.DataFrame:

    population_map = population_map or {}
    completeness_required = completeness_required or [
        col_prov, "Zone_de_sante_notification", "Sexe", "Age",
        date_onset, date_notif, "Issue"
    ]
    w = w or {"trend": 0.30, "incidence": 0.25, "cfr": 0.20, "timeliness": 0.15, "completeness": 0.10}

    d = df.copy()

    # --- Déterminer semaine courante ---
    if current_week is None:
        if col_week not in d.columns or d[col_week].dropna().empty:
            raise ValueError("Impossible de déterminer la semaine courante (col_week manquante/NA).")
        current_week = sorted(d[col_week].dropna().astype(str).unique())[-1]

    # --- Séparer S0 et historique récent ---
    # On prend S0 + S-1..S-3 pour tendance
    # => nécessite un ordre des semaines; si tu as déjà YW_KEY / TIME_KEY c’est encore mieux.
    weeks = sorted(d[col_week].dropna().astype(str).unique())
    if current_week not in weeks:
        raise ValueError(f"current_week '{current_week}' non trouvée.")

    i0 = weeks.index(current_week)
    hist_weeks = weeks[max(0, i0-3): i0]   # 3 semaines avant
    # NB: si moins de 3, ça marche quand même.

    # --- agrégation hebdo provinciale ---
    # Supporte:
    # - IDSR agrégé : col_cases / col_deaths numériques existent
    # - Line list : 1 ligne = 1 cas ; décès dérivé de Issue ou is_death
    d_work = d.copy()

    # Cas
    if (col_cases in d_work.columns):
        d_work["_irep_cas"] = pd.to_numeric(d_work[col_cases], errors="coerce")
        # si la colonne existe mais est majoritairement NA/non-numérique -> fallback à 1 (line list)
        if d_work["_irep_cas"].notna().mean() < 0.2:
            d_work["_irep_cas"] = 1.0
        d_work["_irep_cas"] = d_work["_irep_cas"].fillna(0.0)
    else:
        d_work["_irep_cas"] = 1.0

    # Décès
    if (col_deaths in d_work.columns):
        d_work["_irep_deces"] = pd.to_numeric(d_work[col_deaths], errors="coerce").fillna(0.0)
    elif "is_death" in d_work.columns:
        d_work["_irep_deces"] = pd.to_numeric(d_work["is_death"], errors="coerce").fillna(0.0)
    elif "Issue" in d_work.columns:
        issue = d_work["Issue"].astype("string").str.lower().str.strip()
        death_set = {"dec", "decede", "décédé", "décédée", "decedee", "died", "dead", "décès", "deces"}
        d_work["_irep_deces"] = issue.isin(death_set).astype(float)
    else:
        d_work["_irep_deces"] = 0.0

    agg = (
        d_work.groupby([col_prov, col_week], dropna=False, as_index=False)
              .agg(cas=("_irep_cas", "sum"), deces=("_irep_deces", "sum"))
    )

    # --- S0 ---

    s0 = agg[agg[col_week].astype(str) == str(current_week)].copy()
    s0 = s0.rename(columns={"cas": "cas_S0", "deces": "deces_S0"})

    # --- moyenne 3 semaines (tendance) ---
    if hist_weeks:
        sh = agg[agg[col_week].astype(str).isin([str(x) for x in hist_weeks])].copy()
        moy = (sh.groupby(col_prov, as_index=False)["cas"].mean().rename(columns={"cas": "moy_3sem"}))
    else:
        moy = pd.DataFrame({col_prov: s0[col_prov].unique(), "moy_3sem": np.nan})

    out = s0.merge(moy, on=col_prov, how="left")

    # --- Trend metric ---
    out["trend_ratio"] = out["cas_S0"] / (out["moy_3sem"].fillna(0) + 1)

    # --- Incidence ---
    out["population"] = out[col_prov].map(population_map)
    out["incidence_100k"] = np.where(
        out["population"].notna() & (out["population"] > 0),
        (out["cas_S0"] / out["population"]) * 100000.0,
        np.nan
    )

    # --- CFR ---
    out["cfr_pct"] = np.where(out["cas_S0"] > 0, (out["deces_S0"] / out["cas_S0"]) * 100.0, np.nan)

    # --- Promptitude & Complétude (calculées sur les lignes S0, pas sur l'agrégat) ---
    d_s0 = d[d[col_week].astype(str) == str(current_week)].copy()

    tim = []
    comp = []
    for p in out[col_prov]:
        dp = d_s0[d_s0[col_prov] == p].copy()
        tim_pct = _timeliness_pct(dp, date_onset=date_onset, date_notif=date_notif, threshold_days=threshold_days)
        comp_pct = _completeness_pct(dp, required_cols=completeness_required)
        tim.append(tim_pct)
        comp.append(comp_pct)

    out["promptitude_pct_le2j"] = tim
    out["completude_pct"] = comp

    # Convertir en risques (faible % => risque élevé)
    out["timeliness_risk"] = 100.0 - out["promptitude_pct_le2j"]
    out["completeness_risk"] = 100.0 - out["completude_pct"]

    # --- scores 0-100 ---
    out["TrendScore"]       = _score_quantile_0_100(out["trend_ratio"])
    out["IncidenceScore"]   = _score_quantile_0_100(out["incidence_100k"])
    out["CFRScore"]         = _score_quantile_0_100(out["cfr_pct"])
    out["PromptitudeScore"] = _score_quantile_0_100(out["timeliness_risk"])
    out["CompletenessScore"]= _score_quantile_0_100(out["completeness_risk"])

    # --- Score global avec redistribution des poids si NaN ---
    score_cols = {
        "trend": "TrendScore",
        "incidence": "IncidenceScore",
        "cfr": "CFRScore",
        "timeliness": "PromptitudeScore",
        "completeness": "CompletenessScore",
    }

    def _row_irep(r):
        available = {k: w[k] for k, c in score_cols.items() if pd.notna(r[c])}
        if not available:
            return np.nan
        w_sum = sum(available.values())
        return sum((available[k]/w_sum) * r[score_cols[k]] for k in available)

    out["IREP"] = out.apply(_row_irep, axis=1)

    # --- Catégorisation (optionnel) ---
    out["Risque_cat"] = pd.cut(
        out["IREP"],
        bins=[-np.inf, 30, 60, 80, np.inf],
        labels=["Faible", "Modéré", "Élevé", "Très élevé"]
    )

    # Nettoyage sortie
    keep = [
        col_prov, "cas_S0", "deces_S0", "moy_3sem", "trend_ratio",
        "population", "incidence_100k", "cfr_pct",
        "promptitude_pct_le2j", "completude_pct",
        "TrendScore", "IncidenceScore", "CFRScore", "PromptitudeScore", "CompletenessScore",
        "IREP", "Risque_cat"
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].sort_values("IREP", ascending=False)

    return out

