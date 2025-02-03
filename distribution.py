def plot_campaign_metrics():
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    from scipy import stats

    # Load the data
    df = pd.read_csv("Felipe Chaves_TakeHome Exercise - Campaign_Performance.csv", parse_dates=["date_sent"])

    # Calculate summary statistics
    n_sent_mean = df["n_sent"].mean()
    n_sent_std = df["n_sent"].std()
    
    # The CSV has open_rate and ctr in decimal format, so we convert to percentages for plotting
    open_rate_mean = df["open_rate"].mean() * 100
    open_rate_std = df["open_rate"].std() * 100
    ctr_mean = df["ctr"].mean() * 100
    ctr_std = df["ctr"].std() * 100

    # Calculate percentiles for each metric
    n_sent_percentiles = np.percentile(df["n_sent"], [25, 50, 75])
    open_rate_percentiles = np.percentile(df["open_rate"] * 100, [25, 50, 75])
    ctr_percentiles = np.percentile(df["ctr"] * 100, [25, 50, 75])

    # Function to calculate KDE
    def get_kde(data):
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 100)
        kde_values = kde(x_range)
        # Scale KDE to match histogram height
        kde_values = kde_values * len(data) * (max(data) - min(data)) / 50
        return x_range, kde_values

    # Get KDE for each metric
    n_sent_kde = get_kde(df["n_sent"])
    open_rate_kde = get_kde(df["open_rate"] * 100)
    ctr_kde = get_kde(df["ctr"] * 100)

    # Create subplots with vertical layout
    fig = make_subplots(
        rows=3, cols=1,  # Changed to 3 rows, 1 column
        subplot_titles=(
            "<b>Emails Sent Distribution</b>",
            "<b>Open Rate Distribution</b>",
            "<b>CTR Distribution</b>"
        ),
        vertical_spacing=0.12  # Adjust spacing between plots
    )

    # Helper function for consistent formatting of annotations
    def add_metric_subplot(row, data, kde_data, percentiles, x_title, x_range=None, format_str='.1f', suffix=''):
        hist_values, _ = np.histogram(data, bins=30)
        max_height = max(hist_values)

        # Add histogram with improved styling
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=30,
                marker=dict(
                    color=['rgba(31, 119, 180, 0.6)', 'rgba(44, 160, 44, 0.6)', 'rgba(255, 127, 14, 0.6)'][row-1],
                    line=dict(
                        color=['rgba(31, 119, 180, 1)', 'rgba(44, 160, 44, 1)', 'rgba(255, 127, 14, 1)'][row-1],
                        width=1
                    )
                ),
                name='Distribution',
                showlegend=False
            ),
            row=row, col=1
        )

        # Add KDE curve with improved styling
        fig.add_trace(
            go.Scatter(
                x=kde_data[0],
                y=kde_data[1],
                mode='lines',
                line=dict(
                    color='rgba(0, 0, 128, 0.7)',
                    width=2.5
                ),
                name='Trend',
                showlegend=False
            ),
            row=row, col=1
        )

        # Add mean line
        mean_value = np.mean(data)
        fig.add_trace(
            go.Scatter(
                x=[mean_value, mean_value],
                y=[0, max_height],
                mode='lines',
                line=dict(
                    color='black',
                    width=2,
                    dash='dot'
                ),
                name='Mean',
                showlegend=(row == 1)
            ),
            row=row, col=1
        )

        # Add threshold lines with improved styling
        colors = ['#2ca02c', '#ff7f0e', '#d62728']
        names = ['25th Percentile', 'Median', '75th Percentile']
        
        for i, (percentile, name) in enumerate(zip(percentiles, names)):
            fig.add_trace(
                go.Scatter(
                    x=[percentile, percentile],
                    y=[0, max_height],
                    mode='lines',
                    line=dict(
                        color=colors[i],
                        width=2,
                        dash='dash'
                    ),
                    name=name,
                    showlegend=(row == 1)
                ),
                row=row, col=1
            )

            # Add label annotation with improved styling
            fig.add_annotation(
                x=percentile,
                y=max_height * 1.05,
                text=f"<b>{percentile:{format_str}}{suffix}</b>",
                showarrow=False,
                font=dict(
                    color=colors[i],
                    size=11
                ),
                xref=f'x{row}',
                yref=f'y{row}'
            )

        # Add summary statistics annotation
        fig.add_annotation(
            x=x_range[0] + (x_range[1] - x_range[0]) * 0.02,  # Position near left edge
            y=max_height * 0.95,
            text=(f"<b>Summary Statistics</b><br>" +
                  f"Mean: {mean_value:{format_str}}{suffix}<br>" +
                  f"Std Dev: {np.std(data):{format_str}}{suffix}"),
            showarrow=False,
            align='left',
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
            xref=f'x{row}',
            yref=f'y{row}'
        )

        # Update axes with improved styling
        fig.update_xaxes(
            title_text=f"<b>{x_title}</b>",
            title_font=dict(size=14),
            row=row, col=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickfont=dict(size=12),
            range=x_range if x_range else None,
            showgrid=True,
            zeroline=False
        )
        fig.update_yaxes(
            title_text="<b>Count</b>" if row == 1 else "",
            title_font=dict(size=14),
            row=row, col=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickfont=dict(size=12),
            showgrid=True,
            zeroline=False
        )

    # Add each subplot with custom ranges
    add_metric_subplot(
        1, df["n_sent"], n_sent_kde, n_sent_percentiles,
        "Number of Emails Sent",
        x_range=[4000, 21000],
        format_str=',.0f'
    )
    add_metric_subplot(
        2, df["open_rate"]*100, open_rate_kde, open_rate_percentiles,
        "Email Open Rate (%)",
        x_range=[20, 35],
        suffix='%'
    )
    add_metric_subplot(
        3, df["ctr"]*100, ctr_kde, ctr_percentiles,
        "Click-through Rate (%)",
        x_range=[8, 18],
        suffix='%'
    )

    # Update layout with improved styling
    fig.update_layout(
        title=dict(
            text="<b>Distribution of Campaign Performance Metrics</b>",
            font=dict(size=20),
            y=0.95
        ),
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12),
            title=dict(text="<b>Performance Indicators</b>", font=dict(size=13))
        ),
        height=1000,
        width=800,
        margin=dict(t=100, b=150, l=50, r=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12),
    )

    fig.show()

# Optionally, run the function
plot_campaign_metrics()