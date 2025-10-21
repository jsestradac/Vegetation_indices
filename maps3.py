# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 18:00:12 2025

@author: Robotics
"""

import pandas as pd
import plotly.express as px

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

# 2. Clean names to match country names recognized by Plotly
df["Country"] = df["Country"].replace({
    "Korea, Republic of": "South Korea",
    "European Patents": None  # not a country, remove
})

df = df.dropna(subset=["Country"])

# 3. Create choropleth map
fig = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",
    color="Patents",
    color_continuous_scale="Viridis",  # try 'Reds', 'Plasma', etc.
    title="Number of Patents by Country",
)

# 4. Customize layout
fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
    coloraxis_colorbar=dict(title="Patents")
)

# 5. Show map
fig.show()