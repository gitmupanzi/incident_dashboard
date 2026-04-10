def compute_kpis(df, col_date, col_statut):
    total = len(df)
    clotures = df[df[col_statut] == "Clôturé"].shape[0]
    non_clotures = total - clotures
    taux = (clotures / total * 100) if total > 0 else 0
    moyenne = int(total / max(1, df[col_date].nunique()))

    return total, clotures, non_clotures, taux, moyenne