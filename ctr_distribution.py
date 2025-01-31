#!/usr/bin/env python3

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Read the data
df = pd.read_csv('Felipe Chaves_TakeHome Exercise - Campaign_Performance.csv', 
                 parse_dates=['date_sent'])

# Calculate rates as percentages
df['ctr'] = (df['n_click'] / df['n_open']) * 100
df['open_rate'] = (df['n_open'] / df['n_sent']) * 100

# Calculate performance thresholds for both metrics
metrics = {
    'ctr': {
        'name': 'CTR',
        'data': df['ctr'],
        'percentiles': {
            '25th': df['ctr'].quantile(0.25),
            'Median': df['ctr'].quantile(0.50),
            '75th': df['ctr'].quantile(0.75)
        },
        'range': [
            max(0, df['ctr'].min() - 0.1 * (df['ctr'].max() - df['ctr'].min())),
            df['ctr'].max() + 0.1 * (df['ctr'].max() - df['ctr'].min())
        ]
    },
    'open_rate': {
        'name': 'Open Rate',
        'data': df['open_rate'],
        'percentiles': {
            '25th': df['open_rate'].quantile(0.25),
            'Median': df['open_rate'].quantile(0.50),
            '75th': df['open_rate'].quantile(0.75)
        },
        'range': [
            max(0, df['open_rate'].min() - 0.1 * (df['open_rate'].max() - df['open_rate'].min())),
            df['open_rate'].max() + 0.1 * (df['open_rate'].max() - df['open_rate'].min())
        ]
    }
}

# Create subplots with more compact spacing
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        'CTR Distribution with Performance Thresholds',
        'Open Rate Distribution with Performance Thresholds'
    ),
    vertical_spacing=0.16
)

# Colors for percentile lines and styling
colors = {
    '25th': '#E74C3C',  # Red
    'Median': '#F39C12',  # Orange
    '75th': '#27AE60'  # Green
}

# Create plots for each metric
for idx, (metric_key, metric) in enumerate(metrics.items(), 1):
    # Create histogram with optimal number of bins
    hist, bins = np.histogram(metric['data'], bins='auto')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Add histogram with improved styling
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist,
            name='Distribution',
            marker_color='rgba(144, 238, 144, 0.7)',
            marker_line_color='rgba(0, 100, 0, 0.3)',
            marker_line_width=1,
            hovertemplate=f'{metric["name"]}: %{{x:.1f}}%<br>Count: %{{y}}<extra></extra>',
            showlegend=(idx == 1)
        ),
        row=idx, col=1
    )
    
    # Add KDE with improved smoothing
    kde_x = np.linspace(metric['range'][0], metric['range'][1], 200)
    kde = stats.gaussian_kde(metric['data'], bw_method='silverman')
    kde_y = kde(kde_x) * len(metric['data']) * (bins[1] - bins[0])
    
    fig.add_trace(
        go.Scatter(
            x=kde_x,
            y=kde_y,
            name='Density Curve',
            line=dict(color='#2ECC71', width=2.5, shape='spline'),
            hovertemplate=f'{metric["name"]}: %{{x:.1f}}%<br>Density: %{{y:.1f}}<extra></extra>',
            showlegend=(idx == 1)
        ),
        row=idx, col=1
    )
    
    # Calculate max y value for positioning annotations
    max_y = max(max(hist), max(kde_y))
    
    # Add percentile labels at the top of the plot
    fig.add_annotation(
        x=metric['range'][0],
        y=max_y * 1.15,
        text=f"<b>Performance Thresholds:</b> " +
             f"<span style='color: {colors['25th']}'>25th: {metric['percentiles']['25th']:.1f}%</span>  " +
             f"<span style='color: {colors['Median']}'>Median: {metric['percentiles']['Median']:.1f}%</span>  " +
             f"<span style='color: {colors['75th']}'>75th: {metric['percentiles']['75th']:.1f}%</span>",
        showarrow=False,
        xref=f'x{idx}',
        yref=f'y{idx}',
        xanchor='left',
        font=dict(size=11),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='rgba(0,0,0,0.1)',
        borderwidth=1,
        borderpad=4
    )
    
    # Add vertical lines for percentiles without text annotations
    for label, value in metric['percentiles'].items():
        fig.add_vline(
            x=value,
            line_dash="dash",
            line_color=colors[label],
            line_width=1.5,
            row=idx, col=1
        )
    
    # Add stats annotation
    stats_text = (
        f"<b>Key Statistics:</b><br>" +
        f"Mean: {metric['data'].mean():.1f}%<br>" +
        f"Std Dev: {metric['data'].std():.1f}%<br>" +
        f"Sample Size: {len(df):,}"
    )
    
    fig.add_annotation(
        x=0.99,
        y=0.95,
        xref=f'x{idx}',
        yref=f'y{idx}',
        text=stats_text,
        showarrow=False,
        font=dict(size=11, color="#34495E"),
        align='left',
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='rgba(0,0,0,0.1)',
        borderwidth=1,
        borderpad=6
    )

# Update layout with improved styling
fig.update_layout(
    height=1000,
    width=1000,
    template='plotly_white',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=1.0,
        xanchor="right",
        x=0.99,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
        orientation="h",
        itemwidth=70,
        itemsizing="constant"
    ),
    margin=dict(l=60, r=60, t=100, b=60),
    paper_bgcolor='white',
    plot_bgcolor='rgba(240,240,240,0.3)',
    font=dict(family="Arial", size=12)
)

# Update axes with focused ranges and improved styling
for idx, (metric_key, metric) in enumerate(metrics.items(), 1):
    fig.update_xaxes(
        title_text=f"{metric['name']} %",
        title_font=dict(size=13, color="#2C3E50"),
        range=metric['range'],
        gridcolor='rgba(0,0,0,0.1)',
        row=idx, col=1,
        tickformat='.1f'
    )
    fig.update_yaxes(
        title_text="Count",
        title_font=dict(size=13, color="#2C3E50"),
        gridcolor='rgba(0,0,0,0.1)',
        row=idx, col=1
    )

# Show the figure
fig.show() 

# Print summary statistics
print("\nEmail Marketing Metrics Analysis")
print("=" * 40)
print(f"Number of campaigns analyzed: {len(df):,}")

for metric_key, metric in metrics.items():
    print(f"\n{metric['name']} Performance Thresholds:")
    for label, value in metric['percentiles'].items():
        print(f"{label}: {value:.1f}%")
    print(f"\n{metric['name']} Key Statistics:")
    print(f"Mean: {metric['data'].mean():.1f}%")
    print(f"Standard Deviation: {metric['data'].std():.1f}%")
    print(f"Range: {metric['data'].min():.1f}% - {metric['data'].max():.1f}%")

