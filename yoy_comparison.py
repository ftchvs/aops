#!/usr/bin/env python3

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Load and prepare data
def load_data():
    df = pd.read_csv('Felipe Chaves_TakeHome Exercise - Campaign_Performance.csv', 
                     parse_dates=['date_sent'])
    
    # Calculate key metrics
    df['open_rate'] = df['n_open'] / df['n_sent']
    df['ctr'] = df['n_click'] / df['n_open']
    
    # Extract month and year for grouping
    df['month'] = df['date_sent'].dt.month
    df['year'] = df['date_sent'].dt.year
    
    return df

def create_yoy_comparison():
    # Load data
    df = load_data()
    
    # Calculate monthly aggregates for each year
    monthly_stats = df.groupby(['year', 'month']).agg({
        'n_sent': 'sum',
        'open_rate': 'mean',
        'ctr': 'mean'
    }).reset_index()
    
    # Create subplots with increased spacing
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Email Campaign Volume',  # Simplified titles
            'Email Open Rate',
            'Click-Through Rate'
        ),
        vertical_spacing=0.12,  # Slightly reduced spacing
        row_heights=[0.33, 0.33, 0.33]
    )
    
    # Updated colors to match the provided scheme
    colors = {
        2022: '#B19CD9',  # Light purple
        2023: '#98D8B7',  # Mint green
        2024: '#E8B69E'   # Peach/coral
    }
    
    # Add traces for each year
    for year in sorted(monthly_stats['year'].unique()):
        year_data = monthly_stats[monthly_stats['year'] == year]
        
        # Email Send Volume
        fig.add_trace(
            go.Scatter(
                x=year_data['month'],
                y=year_data['n_sent'],
                name=str(year),
                mode='lines+markers',
                line=dict(
                    color=colors[year],
                    width=2.5  # Slightly thinner lines
                ),
                marker=dict(
                    color=colors[year],
                    size=7,  # Slightly smaller markers
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                hovertemplate=(
                    f'<b>{year}</b><br>'
                    'Month: %{x}<br>'
                    'Volume: %{y:,.0f}'
                ),
                legendgroup=str(year),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Email Open Rate
        fig.add_trace(
            go.Scatter(
                x=year_data['month'],
                y=year_data['open_rate'],
                name=str(year),
                mode='lines+markers',
                line=dict(
                    color=colors[year],
                    width=2.5  # Slightly thinner lines
                ),
                marker=dict(
                    color=colors[year],
                    size=7,  # Slightly smaller markers
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                hovertemplate=(
                    f'<b>{year}</b><br>'
                    'Month: %{x}<br>'
                    'Open Rate: %{y:.1%}'
                ),
                legendgroup=str(year),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Click-Through Rate
        fig.add_trace(
            go.Scatter(
                x=year_data['month'],
                y=year_data['ctr'],
                name=str(year),
                mode='lines+markers',
                line=dict(
                    color=colors[year],
                    width=2.5  # Slightly thinner lines
                ),
                marker=dict(
                    color=colors[year],
                    size=7,  # Slightly smaller markers
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                hovertemplate=(
                    f'<b>{year}</b><br>'
                    'Month: %{x}<br>'
                    'CTR: %{y:.1%}'
                ),
                legendgroup=str(year),
                showlegend=False
            ),
            row=3, col=1
        )
    
    # Update layout with improved formatting
    fig.update_layout(
        title=dict(
            text='Email Marketing Performance Metrics',  # Simplified title
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top',
            font=dict(size=24, color='#2F2F2F')
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.1)',
            borderwidth=1,
            font=dict(size=12),
            itemsizing='constant'
        ),
        height=900,  # Slightly reduced height
        width=1000,  # Slightly reduced width
        template='none',  # Clean template
        paper_bgcolor='white',
        plot_bgcolor='white',  # Clean white background
        margin=dict(t=120, b=50, l=80, r=80)
    )
    
    # Update axes with cleaner styling
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i in range(1, 4):
        # Update x-axes
        fig.update_xaxes(
            title_text=None,  # Remove x-axis titles
            ticktext=month_names,
            tickvals=list(range(1, 13)),
            tickangle=0,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 0, 0, 0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0, 0, 0, 0.3)',
            range=[0.5, 12.5],
            row=i,
            col=1,
            ticks="outside",
            ticklen=5
        )
        
        # Update y-axes
        if i == 1:
            fig.update_yaxes(
                title_text="Volume",
                tickformat=",",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0, 0, 0, 0.1)',
                showline=True,
                linewidth=1,
                linecolor='rgba(0, 0, 0, 0.3)',
                row=i,
                col=1,
                ticks="outside",
                ticklen=5,
                title_standoff=15
            )
        else:
            fig.update_yaxes(
                title_text="Percentage",
                tickformat='.1%',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0, 0, 0, 0.1)',
                showline=True,
                linewidth=1,
                linecolor='rgba(0, 0, 0, 0.3)',
                row=i,
                col=1,
                ticks="outside",
                ticklen=5,
                title_standoff=15
            )
    
    # Add YoY change annotations with improved styling
    for metric in ['n_sent', 'open_rate', 'ctr']:
        for year in range(2023, 2025):
            prev_year_data = monthly_stats[monthly_stats['year'] == year-1]
            curr_year_data = monthly_stats[monthly_stats['year'] == year]
            
            if not prev_year_data.empty and not curr_year_data.empty:
                prev_year_avg = prev_year_data[metric].mean()
                curr_year_avg = curr_year_data[metric].mean()
                yoy_change = (curr_year_avg - prev_year_avg) / prev_year_avg * 100
                
                row = 1 if metric == 'n_sent' else 2 if metric == 'open_rate' else 3
                
                # Add annotation with cleaner styling
                fig.add_annotation(
                    x=12.2,
                    y=curr_year_data[metric].iloc[-1],
                    text=f'{year} YoY: {yoy_change:+.1f}%',  # Simplified text
                    showarrow=False,
                    xanchor='left',
                    yanchor='middle',
                    font=dict(
                        size=11,
                        color='#2F2F2F',
                        family='Arial'
                    ),
                    bgcolor='rgba(255, 255, 255, 0.95)',
                    bordercolor='rgba(0, 0, 0, 0.1)',
                    borderwidth=1,
                    borderpad=4,
                    row=row,
                    col=1
                )
    
    return fig

def main():
    fig = create_yoy_comparison()
    fig.show()

if __name__ == "__main__":
    main() 