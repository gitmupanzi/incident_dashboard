import geopandas as gpd
import plotly.express as px

def map_rdc(df, col_province, geojson_path):
    gdf = gpd.read_file(geojson_path)

    data = df[col_province].value_counts().reset_index()
    data.columns = ["province", "value"]

    merged = gdf.merge(data, left_on="province", right_on="province", how="left")

    fig = px.choropleth(
        merged,
        geojson=merged.geometry,
        locations=merged.index,
        color="value",
        projection="mercator"
    )

    fig.update_geos(fitbounds="locations", visible=False)
    return fig