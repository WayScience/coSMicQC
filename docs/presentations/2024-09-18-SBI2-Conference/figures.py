# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Generate figures for poster

# +
import numpy as np
import plotly.graph_objects as go

# Generate random data
np.random.seed(0)

# Inlier data
inlier_x = np.random.normal(loc=0, scale=1, size=50)
inlier_y = np.random.normal(loc=0, scale=1, size=50)

# Outlier data
outlier_x = np.random.normal(loc=5, scale=1, size=10)
outlier_y = np.random.normal(loc=5, scale=1, size=10)

# Create figure
fig = go.Figure()

# Add inliers to the plot
fig.add_trace(
    go.Scatter(
        x=inlier_x,
        y=inlier_y,
        mode="markers",
        name="Inliers",
        marker={"color": "green", "size": 8},
    )
)

# Add outliers to the plot
fig.add_trace(
    go.Scatter(
        x=outlier_x,
        y=outlier_y,
        mode="markers",
        name="Outliers",
        marker={"color": "red", "size": 8},
    )
)

# Update layout
fig.update_layout(
    legend_title="Legend",
    legend={
        "x": 1,
        "y": 0,
        "traceorder": "normal",
        "orientation": "v",
        "xanchor": "right",
        "yanchor": "bottom",
    },
    width=1200,
    height=500,
)

# Save figure as an image
fig.write_image("images/example-outliers.png")

fig.show()
