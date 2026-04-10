import pandas as pd

def load_data(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    return df