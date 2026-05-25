
import pandas as pd
from dashboard_app.runtime_support import build_runtime_context
from dashboard_app.tabs.cousp import render_cousp_tab
from dashboard_app.domain import standardize_ll_by_disease, standardize_df

path = r"C:\Users\Benjamin MUPANZI\Documents\dataminsante\output\rdc_compilation_LL_Ebola_SE01_SE53_2026_05_25_19_17_10.xlsx"
raw = pd.read_excel(path, sheet_name="LL_Ebola")
df_f = standardize_df(standardize_ll_by_disease(raw, "ebola"))
ctx = build_runtime_context(
    df_f=df_f,
    disease_key="ebola",
    IDSR_MODE=False,
    annot_vals=False,
)
render_cousp_tab(ctx)
