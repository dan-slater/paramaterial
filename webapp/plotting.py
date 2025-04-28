# Webapp plotting functions using Plotly

import pandas as pd
import plotly.express as px
import plotly.utils
import json
import logging

def create_stress_strain_plot(df: pd.DataFrame, x_col: str, y_col: str):
    """Generates a Plotly JSON object for a plot using specified columns."""

    # Log the input parameters and DataFrame info
    logging.info(f"Plotting function called with x_col: '{x_col}', y_col: '{y_col}'")
    logging.info(f"DataFrame contains {len(df)} rows and columns: {df.columns.tolist()}")
    
    # Validate selected columns exist
    missing_cols = []
    if x_col not in df.columns:
        missing_cols.append(x_col)
        logging.error(f"Column '{x_col}' not found in DataFrame columns: {df.columns.tolist()}")
    if y_col not in df.columns:
        missing_cols.append(y_col)
        logging.error(f"Column '{y_col}' not found in DataFrame columns: {df.columns.tolist()}")
    if missing_cols:
        error_msg = f"Selected columns not found in data: {', '.join(missing_cols)}"
        logging.error(error_msg)
        raise ValueError(error_msg)

    try:
        # Ensure numeric types (best effort)
        df_plot = df[[x_col, y_col]].copy() # Work on a copy
        df_plot[x_col] = pd.to_numeric(df_plot[x_col], errors='coerce')
        df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
        df_plot = df_plot.dropna(subset=[x_col, y_col])

        if df_plot.empty:
             raise ValueError(f"No valid numeric data found in selected columns ('{x_col}', '{y_col}') after cleaning.")

        # Use selected column names for plot
        fig = px.line(df_plot, x=x_col, y=y_col,
                      title=f'{y_col} vs. {x_col}',
                      labels={x_col: x_col, y_col: y_col},
                      markers=True)

        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            margin=dict(l=40, r=20, t=40, b=30), # Adjust margins
            hovermode='closest'
        )

        # Convert figure to JSON
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return plot_json

    except Exception as e:
        # Log the specific error or re-raise a more specific exception
        print(f"Error generating plot: {e}") # Replace with proper logging later
        raise ValueError(f"Could not generate plot: {e}")
