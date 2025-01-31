#!/usr/bin/env python3

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import scipy.stats as stats
from scipy.stats import mannwhitneyu

# Load and prepare data
df = pd.read_csv('Felipe Chaves_TakeHome Exercise - Campaign_Performance.csv', 
                 parse_dates=['date_sent'])

# Calculate metrics
df['open_rate'] = df['n_open']/df['n_sent']
df['ctr'] = np.where(df['n_open'] > 0, df['n_click']/df['n_open'], 0)
df['year'] = df['date_sent'].dt.year
df['day_of_year'] = df['date_sent'].dt.dayofyear

# Get the most recent date and calculate the day range for comparison
max_date = df['date_sent'].max()
current_day_of_year = max_date.dayofyear
start_day = current_day_of_year - 30

# Get data for each year's same period
def get_period_data(year, start_day, end_day):
    year_data = df[df['year'] == year]
    return year_data[
        (year_data['day_of_year'] >= start_day) & 
        (year_data['day_of_year'] <= end_day)
    ]

recent_data = get_period_data(2024, start_day, current_day_of_year)
data_2023 = get_period_data(2023, start_day, current_day_of_year)
data_2022 = get_period_data(2022, start_day, current_day_of_year)
historical_data = df  # All data for historical comparison

def perform_statistical_comparison(metric, data1, data2, period1_name, period2_name):
    """Perform comprehensive statistical analysis for a given metric between two periods"""
    if len(data1) == 0 or len(data2) == 0:
        return None
        
    # Perform t-test
    t_stat, t_pval = stats.ttest_ind(
        data1[metric],
        data2[metric]
    )
    
    # Perform Mann-Whitney U test (non-parametric)
    u_stat, u_pval = mannwhitneyu(
        data1[metric],
        data2[metric],
        alternative='two-sided'
    )
    
    # Calculate Cohen's d effect size
    pooled_std = np.sqrt((data1[metric].var() + data2[metric].var()) / 2)
    cohens_d = (data1[metric].mean() - data2[metric].mean()) / pooled_std
    
    return {
        'metric': metric,
        f'{period1_name}_mean': data1[metric].mean(),
        f'{period2_name}_mean': data2[metric].mean(),
        'change': (data1[metric].mean() - data2[metric].mean()) / data2[metric].mean(),
        't_stat': t_stat,
        't_pvalue': t_pval,
        'u_stat': u_stat,
        'u_pvalue': u_pval,
        'cohens_d': cohens_d,
        'significant': (t_pval < 0.05) or (u_pval < 0.05)
    }

# Perform analysis for both metrics
metrics = ['open_rate', 'ctr']
results = {}
for metric in metrics:
    # Current vs Historical
    results[f'{metric}_current_vs_hist'] = perform_statistical_comparison(
        metric, recent_data, historical_data, 'recent', 'historical'
    )
    # Current vs 2023
    results[f'{metric}_current_vs_2023'] = perform_statistical_comparison(
        metric, recent_data, data_2023, 'recent', '2023'
    )
    # Current vs 2022
    results[f'{metric}_current_vs_2022'] = perform_statistical_comparison(
        metric, recent_data, data_2022, 'recent', '2022'
    )

# Create the visualization
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=('Open Rate Comparison', 'CTR Comparison'),
                    horizontal_spacing=0.1)

# Colors for different years
colors = {
    'Historical': 'rgba(52, 152, 219, 0.7)',  # Blue
    '2022': 'rgba(155, 89, 182, 0.7)',        # Purple
    '2023': 'rgba(46, 204, 113, 0.7)',        # Green
    '2024': 'rgba(230, 126, 34, 0.7)'         # Orange
}

# Add bars for each metric
for i, metric in enumerate(['open_rate', 'ctr']):
    col = i + 1
    x_positions = [1, 2, 3, 4]  # One position for each period
    
    # Historical bar
    hist_mean = results[f'{metric}_current_vs_hist']['historical_mean']
    fig.add_trace(
        go.Bar(
            x=[x_positions[0]],
            y=[hist_mean],
            name='Historical',
            marker_color=colors['Historical'],
            text=[f"{hist_mean:.1%}"],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=col
    )
    
    # 2022 bar
    if len(data_2022) > 0:
        mean_2022 = data_2022[metric].mean()
        fig.add_trace(
            go.Bar(
                x=[x_positions[1]],
                y=[mean_2022],
                name='2022',
                marker_color=colors['2022'],
                text=[f"{mean_2022:.1%}"],
                textposition='auto',
                showlegend=False
            ),
            row=1, col=col
        )
    
    # 2023 bar
    if len(data_2023) > 0:
        mean_2023 = data_2023[metric].mean()
        fig.add_trace(
            go.Bar(
                x=[x_positions[2]],
                y=[mean_2023],
                name='2023',
                marker_color=colors['2023'],
                text=[f"{mean_2023:.1%}"],
                textposition='auto',
                showlegend=False
            ),
            row=1, col=col
        )
    
    # 2024 (Recent) bar
    recent_mean = results[f'{metric}_current_vs_hist']['recent_mean']
    fig.add_trace(
        go.Bar(
            x=[x_positions[3]],
            y=[recent_mean],
            name='2024',
            marker_color=colors['2024'],
            text=[f"{recent_mean:.1%}"],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=col
    )
    
    # Add significance annotations
    comparisons = [
        ('current_vs_hist', 0.5, 'Historical'),
        ('current_vs_2022', 1.5, '2022'),
        ('current_vs_2023', 2.5, '2023')
    ]
    
    # Calculate maximum y value for the current metric
    max_y = max(
        hist_mean,
        mean_2022 if len(data_2022) > 0 else 0,
        mean_2023 if len(data_2023) > 0 else 0,
        recent_mean
    )

    # Adjust y-axis range based on the metric
    y_range_multiplier = 1.45 if metric == 'open_rate' else 1.5
    
    # Add annotations with better spacing
    for idx, (comp_key, x_pos, label) in enumerate(comparisons):
        result = results[f'{metric}_{comp_key}']
        if result is not None:
            p_val = min(result['t_pvalue'], result['u_pvalue'])
            change = result['change'] * 100
            
            # Calculate y position with better spacing
            # Adjust spacing for open_rate to be more compact
            spacing_factor = 0.08 if metric == 'open_rate' else 0.1
            y_pos = max_y * (1.15 + idx * spacing_factor)
            
            # Format the annotation text more clearly
            sig_star = "*" if result['significant'] else ""
            annotation_text = (
                f"vs {label}:<br>"
                f"Δ{change:+.1f}%<br>"
                f"p={p_val:.3f}{sig_star}"
            )
            
            # Add connecting line to make it clear which comparison it refers to
            fig.add_shape(
                type="line",
                x0=4,  # Position of 2024 bar
                y0=recent_mean,
                x1=x_pos + 0.5,  # End at annotation
                y1=y_pos,
                line=dict(
                    color="rgba(128, 128, 128, 0.3)",
                    width=1,
                    dash="dot"
                ),
                row=1, col=col
            )
            
            # Add annotation with adjusted position and style
            fig.add_annotation(
                x=x_pos + 0.5,
                y=y_pos,
                text=annotation_text,
                showarrow=False,
                font=dict(size=11),
                xref=f'x{col}',
                yref=f'y{col}',
                align='left',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(128, 128, 128, 0.3)',
                borderwidth=1,
                borderpad=4
            )

# Update layout
fig.update_layout(
    title=dict(
        text='Year-over-Year Comparison (Same 30-day Period)',
        x=0.5,
        y=0.95,
        xanchor='center',
        yanchor='top',
        font=dict(size=20)
    ),
    height=700,  # Increased height to accommodate annotations
    width=1200,
    plot_bgcolor='white',
    margin=dict(t=100, r=200, b=100, l=50)  # Increased right margin for annotations
)

# Update axes with more space for annotations and different ranges for each metric
for i in range(1, 3):
    metric = 'open_rate' if i == 1 else 'ctr'
    max_y = max(
        results[f'{metric}_current_vs_hist']['historical_mean'],
        data_2022[metric].mean() if len(data_2022) > 0 else 0,
        data_2023[metric].mean() if len(data_2023) > 0 else 0,
        results[f'{metric}_current_vs_hist']['recent_mean']
    )
    
    # Set different ranges for open_rate and ctr
    y_range_multiplier = 1.45 if metric == 'open_rate' else 1.5
    
    fig.update_xaxes(
        ticktext=['Historical', '2022', '2023', '2024'],
        tickvals=[1, 2, 3, 4],
        row=1, col=i
    )
    fig.update_yaxes(
        title='Rate',
        tickformat='.1%',
        gridcolor='rgba(128,128,128,0.1)',
        zerolinecolor='rgba(128,128,128,0.1)',
        range=[0, max_y * y_range_multiplier],  # Different range multiplier for each metric
        row=1, col=i
    )

# Add footnote
fig.add_annotation(
    text="* indicates statistical significance (p < 0.05)",
    xref="paper",
    yref="paper",
    x=0,
    y=-0.15,
    showarrow=False,
    font=dict(size=10, color='gray')
)

# Show the figure
fig.show()

# Print detailed analysis
print("\nYear-over-Year Analysis (Same 30-day Period)")
print("=" * 80)

for metric in metrics:
    metric_name = "OPEN RATE" if metric == "open_rate" else "CLICK-THROUGH RATE (CTR)"
    print(f"\n{metric_name}:")
    print("-" * 50)
    
    # Print means for each period
    print(f"Mean Values:")
    print(f"  Historical: {results[f'{metric}_current_vs_hist']['historical_mean']:.1%}")
    if len(data_2022) > 0:
        print(f"  2022: {data_2022[metric].mean():.1%}")
    if len(data_2023) > 0:
        print(f"  2023: {data_2023[metric].mean():.1%}")
    print(f"  2024: {results[f'{metric}_current_vs_hist']['recent_mean']:.1%}")
    
    print("\nStatistical Comparisons (2024 vs):")
    comparisons = [
        ('current_vs_hist', 'Historical'),
        ('current_vs_2022', '2022'),
        ('current_vs_2023', '2023')
    ]
    
    for comp_key, label in comparisons:
        result = results[f'{metric}_{comp_key}']
        if result is not None:
            sig_mark = "*" if result['significant'] else ""
            print(f"\n  {label}:")
            print(f"    Change: {result['change']*100:+.1f}%")
            print(f"    P-value: {min(result['t_pvalue'], result['u_pvalue']):.4f}{sig_mark}")
            print(f"    Effect Size: {abs(result['cohens_d']):.2f} ({'very large' if abs(result['cohens_d']) > 1.2 else 'large' if abs(result['cohens_d']) > 0.8 else 'medium' if abs(result['cohens_d']) > 0.5 else 'small'})") 