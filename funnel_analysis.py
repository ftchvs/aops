#!/usr/bin/env python3

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to format large numbers
def format_number(num):
    if num >= 1_000_000:
        return f'{num/1_000_000:.1f}M'
    elif num >= 1_000:
        return f'{num/1_000:.1f}K'
    return f'{num:.0f}'

def format_change(value):
    return f"{'↑' if value > 0 else '↓'} {abs(value):.1f}%"

# Read the data
df = pd.read_csv('Felipe Chaves_TakeHome Exercise - Campaign_Performance.csv', 
                 parse_dates=['date_sent'])
df['year'] = df['date_sent'].dt.year

def create_funnel_data(year_data):
    funnel_stages = [
        {'stage': 'Sent', 'count': year_data['n_sent'].sum()},
        {'stage': 'Opened', 'count': year_data['n_open'].sum()},
        {'stage': 'Clicked', 'count': year_data['n_click'].sum()}
    ]
    funnel_data = pd.DataFrame(funnel_stages)
    
    # Calculate rates
    funnel_data['total_rate'] = funnel_data['count'] / funnel_data['count'].iloc[0]
    funnel_data['step_rate'] = funnel_data['count'].pct_change(-1).fillna(0)
    
    return funnel_data

# Get unique years
years = sorted(df['year'].unique())

# Create the main figure with subplots
fig = make_subplots(
    rows=1, cols=len(years),
    subplot_titles=[f'<b>{year}</b>' for year in years],
    specs=[[{'type': 'funnel'} for _ in years]]
)

# Colors for consistent branding
colors = ['#3498DB', '#2ECC71', '#F1C40F']

# Store metrics for comparison
yearly_metrics = []

# Create funnel for each year
for idx, year in enumerate(years, 1):
    year_data = df[df['year'] == year]
    funnel_data = create_funnel_data(year_data)
    
    # Store metrics for later use
    metrics = {
        'year': year,
        'sent': funnel_data['count'].iloc[0],
        'opened': funnel_data['count'].iloc[1],
        'clicked': funnel_data['count'].iloc[2],
        'open_rate': funnel_data['total_rate'].iloc[1] * 100,
        'click_rate': (funnel_data['count'].iloc[2] / funnel_data['count'].iloc[1]) * 100,
        'overall_rate': funnel_data['total_rate'].iloc[2] * 100
    }
    yearly_metrics.append(metrics)
    
    # Add funnel trace
    fig.add_trace(
        go.Funnel(
            name=str(year),
            y=funnel_data['stage'],
            x=funnel_data['count'],
            textposition="inside",
            textinfo="value+percent initial",
            texttemplate=[
                f"<b>{format_number(funnel_data['count'].iloc[0])}</b><br>({funnel_data['total_rate'].iloc[0]:.1%})",
                f"<b>{format_number(funnel_data['count'].iloc[1])}</b><br>({funnel_data['total_rate'].iloc[1]:.1%})",
                f"<b>{format_number(funnel_data['count'].iloc[2])}</b><br>({funnel_data['total_rate'].iloc[2]:.1%})"
            ],
            textfont=dict(family="Arial", size=14, color="white"),
            opacity=0.9,
            marker={
                "color": colors,
                "line": {"width": [1, 1, 1], "color": ["white", "white", "white"]}
            }
        ),
        row=1, col=idx
    )

# Calculate year-over-year changes
yoy_changes = []
for i in range(1, len(yearly_metrics)):
    current = yearly_metrics[i]
    previous = yearly_metrics[i-1]
    
    volume_change = (current['sent'] - previous['sent']) / previous['sent'] * 100
    open_rate_change = current['open_rate'] - previous['open_rate']
    click_rate_change = current['click_rate'] - previous['click_rate']
    
    yoy_changes.append({
        'years': f"{current['year']}/{previous['year']}",
        'volume': format_change(volume_change),
        'open_rate': format_change(open_rate_change),
        'click_rate': format_change(click_rate_change)
    })

# Add metrics boxes for each year
for idx, metrics in enumerate(yearly_metrics):
    fig.add_annotation(
        x=0.2 + (idx * 0.3),
        y=-0.15,
        text=f"<b>Metrics {metrics['year']}</b><br>" + \
             f"Open Rate: {metrics['open_rate']:.1f}%<br>" + \
             f"Click Rate: {metrics['click_rate']:.1f}%",
        showarrow=False,
        font=dict(size=12, color="#34495E"),
        xref='paper',
        yref='paper',
        align='center',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='rgba(0,0,0,0.1)',
        borderwidth=1,
        borderpad=4
    )

# Add YoY comparison boxes between years
for idx, change in enumerate(yoy_changes):
    fig.add_annotation(
        x=0.35 + (idx * 0.3),
        y=-0.3,
        text=f"<b>{change['years']} Changes</b><br>" + \
             f"Volume: {change['volume']}<br>" + \
             f"Open Rate: {change['open_rate']}<br>" + \
             f"Click Rate: {change['click_rate']}",
        showarrow=False,
        font=dict(size=12, color="#34495E"),
        xref='paper',
        yref='paper',
        align='center',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='rgba(0,0,0,0.1)',
        borderwidth=1,
        borderpad=4
    )

# Update layout with modern styling
fig.update_layout(
    title=dict(
        text='Email Marketing Performance Evolution<br>' +
             '<span style="font-size: 14px; color: #666;">Volume and Engagement Metrics Year-over-Year</span>',
        x=0.5,
        y=0.95,
        xanchor='center',
        yanchor='top',
        font=dict(size=24, family="Arial", color="#2C3E50")
    ),
    showlegend=False,
    height=650,  # Reduced height
    width=300 * len(years),
    template='plotly_white',
    margin=dict(l=30, r=30, t=120, b=120),  # Adjusted bottom margin
    paper_bgcolor='white',
    plot_bgcolor='rgba(240,240,240,0.0)',
    font=dict(family="Arial", size=12, color="#2C3E50")
)

# Print concise summary with data validation
print("\nEmail Marketing Performance Summary")
print("=" * 40)

for metrics in yearly_metrics:
    print(f"\n{metrics['year']} Performance:")
    print(f"Volume: {format_number(metrics['sent'])} emails sent")
    print(f"Opened: {format_number(metrics['opened'])} ({metrics['open_rate']:.1f}%)")
    print(f"Clicked: {format_number(metrics['clicked'])} ({metrics['click_rate']:.1f}%)")
    print(f"Overall Conversion: {metrics['overall_rate']:.1f}%")

print("\nYear-over-Year Changes")
print("=" * 40)
for change in yoy_changes:
    print(f"\n{change['years']}:")
    print(f"Volume: {change['volume']}")
    print(f"Open Rate: {change['open_rate']}")
    print(f"Click Rate: {change['click_rate']}")

# Show the figure
fig.show() 