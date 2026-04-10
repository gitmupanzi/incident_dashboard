import plotly.express as px

def chart_pathologie(df, col):
    d = df[col].value_counts().head(5).reset_index()
    d.columns = ["Pathologie", "Nombre"]
    return px.bar(d, x="Nombre", y="Pathologie", orientation="h")

def chart_categorie(df, col):
    return px.pie(df, names=col, hole=0.5)

def chart_item(df, col):
    d = df[col].value_counts().head(5).reset_index()
    d.columns = ["Item", "Nombre"]
    return px.bar(d, x="Nombre", y="Item", orientation="h")

def chart_time(df, col):
    d = df.groupby(df[col].dt.date).size().reset_index(name="Appels")
    return px.line(d, x=col, y="Appels")