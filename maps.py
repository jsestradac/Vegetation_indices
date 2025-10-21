import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import geodatasets

# 1. Define your data directly in Python
data = {
    "Country": [
        "United States",
        "Korea, Republic of",
        "European Patents",
        "Australia",
        "Canada",
        "Netherlands",
        "South Africa"
    ],
    "Patents": [61, 14, 10, 3, 1, 1, 1]
}

df = pd.DataFrame(data)

# 2. Clean up country names to match GeoPandas dataset
df["Country"] = df["Country"].replace({
    "United States": "United States of America",
    "Korea, Republic of": "Republic of Korea",
    "European Patents": None  # remove this because it's not a country
})

# Remove invalid entries
df = df.dropna(subset=["Country"])

# 3. Load world map data
world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

# 4. Merge your data with the world geometries
merged = world.merge(df, how="left", left_on="name", right_on="Country")

# 5. Plot the map
fig, ax = plt.subplots(figsize=(14, 8))
merged.plot(
    column="Patents",
    cmap="viridis",
    linewidth=0.8,
    ax=ax,
    edgecolor="0.8",
    legend=True,
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "white",
        "hatch": "///",
        "label": "No data"
    }
)

ax.set_title("Number of Patents by Country", fontsize=18, pad=20)
ax.axis("off")

plt.show()