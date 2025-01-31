#!/usr/bin/env python3

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.interpolate import make_interp_spline

# Load and prepare data
def load_data():
    df = pd.read_csv('Felipe Chaves_TakeHome Exercise - Campaign_Performance.csv', 
                     parse_dates=['date_sent'])
    
    # Calculate metrics
    df['open_rate'] = df['n_open'] / df['n_sent']
    df['ctr'] = df['n_click'] / df['n_open']
    
    # Extract month for grouping
    df['month'] = df['date_sent'].dt.month
    df['year'] = df['date_sent'].dt.year
    
    return df

def smooth_data(x, y, smoothing_factor=300):
    """Create smoothed version of data using B-spline interpolation"""
    # Create a finer mesh for smoother curve
    x_smooth = np.linspace(min(x), max(x), smoothing_factor)
    
    # Fit spline
    try:
        spl = make_interp_spline(x, y, k=3)  # type: BSpline
        y_smooth = spl(x_smooth)
        
        # Ensure values stay within reasonable bounds
        if 'rate' in str(y.name).lower():
            y_smooth = np.clip(y_smooth, 0, 1)
        else:
            y_smooth = np.clip(y_smooth, 0, None)
            
        return x_smooth, y_smooth
    except:
        # Fallback to original data if smoothing fails
        return x, y

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
            '<b>Email Send Volume</b>',
            '<b>Open Rate</b>',
            '<b>Click-Through Rate</b>'
        ),
        vertical_spacing=0.15,
        row_heights=[0.33, 0.33, 0.33]
    )
    
    # Updated colors for a more modern, softer look
    colors = {
        2022: '#7CB9E8',  # Soft blue
        2023: '#F4A460',  # Soft orange
        2024: '#90EE90'   # Soft green
    }
    
    # Add traces for each year
    for year in sorted(monthly_stats['year'].unique()):
        year_data = monthly_stats[monthly_stats['year'] == year]
        
        # Email Send Volume with smooth lines
        x_smooth, y_smooth = smooth_data(year_data['month'], year_data['n_sent'])
        fig.add_trace(
            go.Scatter(
                x=x_smooth,
                y=y_smooth,
                name=str(year),  # Simplified legend name
                line=dict(
                    color=colors[year],
                    width=3,
                    shape='spline',
                    smoothing=1.3
                ),
                hovertemplate=(
                    '<b>Month:</b> %{x:.0f}<br>'
                    '<b>Sends:</b> %{y:,.0f}<br>'
                    f'<b>Year:</b> {year}'
                ),
                legendgroup=str(year),  # Group by year
                showlegend=True  # Show legend for all years
            ),
            row=1, col=1
        )
        
        # Add original points as markers
        fig.add_trace(
            go.Scatter(
                x=year_data['month'],
                y=year_data['n_sent'],
                name=f'{year} (Actual)',
                mode='markers',
                marker=dict(
                    color=colors[year],
                    size=8,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                showlegend=False,
                hovertemplate=(
                    '<b>Month:</b> %{x}<br>'
                    '<b>Actual Sends:</b> %{y:,.0f}<br>'
                    f'<b>Year:</b> {year}'
                ),
                legendgroup=str(year)
            ),
            row=1, col=1
        )
        
        # Open Rate with smooth lines
        x_smooth, y_smooth = smooth_data(year_data['month'], year_data['open_rate'])
        fig.add_trace(
            go.Scatter(
                x=x_smooth,
                y=y_smooth,
                name=str(year),
                line=dict(
                    color=colors[year],
                    width=3,
                    shape='spline',
                    smoothing=1.3
                ),
                hovertemplate=(
                    '<b>Month:</b> %{x:.0f}<br>'
                    '<b>Open Rate:</b> %{y:.2%}<br>'
                    f'<b>Year:</b> {year}'
                ),
                legendgroup=str(year),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add original points as markers for Open Rate
        fig.add_trace(
            go.Scatter(
                x=year_data['month'],
                y=year_data['open_rate'],
                name=f'{year} (Actual)',
                mode='markers',
                marker=dict(
                    color=colors[year],
                    size=8,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                showlegend=False,
                hovertemplate=(
                    '<b>Month:</b> %{x}<br>'
                    '<b>Actual Open Rate:</b> %{y:.2%}<br>'
                    f'<b>Year:</b> {year}'
                ),
                legendgroup=str(year)
            ),
            row=2, col=1
        )
        
        # CTR with smooth lines
        x_smooth, y_smooth = smooth_data(year_data['month'], year_data['ctr'])
        fig.add_trace(
            go.Scatter(
                x=x_smooth,
                y=y_smooth,
                name=str(year),
                line=dict(
                    color=colors[year],
                    width=3,
                    shape='spline',
                    smoothing=1.3
                ),
                hovertemplate=(
                    '<b>Month:</b> %{x:.0f}<br>'
                    '<b>CTR:</b> %{y:.2%}<br>'
                    f'<b>Year:</b> {year}'
                ),
                legendgroup=str(year),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Add original points as markers for CTR
        fig.add_trace(
            go.Scatter(
                x=year_data['month'],
                y=year_data['ctr'],
                name=f'{year} (Actual)',
                mode='markers',
                marker=dict(
                    color=colors[year],
                    size=8,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                showlegend=False,
                hovertemplate=(
                    '<b>Month:</b> %{x}<br>'
                    '<b>Actual CTR:</b> %{y:.2%}<br>'
                    f'<b>Year:</b> {year}'
                ),
                legendgroup=str(year)
            ),
            row=3, col=1
        )
    
    # Update layout with improved formatting and spacing
    fig.update_layout(
        title=dict(
            text='<b>Year-Over-Year Email Marketing Performance Comparison</b>',
            x=0.5,
            y=0.97,  # Moved title up slightly
            xanchor='center',
            yanchor='top',
            font=dict(size=24)
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,  # Increased space between title and legend
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(240, 240, 240, 1)',  # Lighter border
            borderwidth=1,
            font=dict(size=12),
            itemsizing='constant',  # Ensure consistent legend item sizes
            itemwidth=40,  # Control legend item width
            itemclick=False,  # Disable item clicking
            itemdoubleclick=False  # Disable item double clicking
        ),
        height=1000,
        width=1200,
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='rgba(250, 250, 250, 0.5)',  # Lighter background
        margin=dict(t=150, b=50, l=80, r=80)  # Increased top margin for legend
    )
    
    # Update axes with softer grid
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i in range(1, 4):
        # Update x-axes
        fig.update_xaxes(
            title_text="<b>Month</b>",
            ticktext=month_names,
            tickvals=list(range(1, 13)),
            tickangle=0,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.1)',  # Softer grid
            showline=True,
            linewidth=1,
            linecolor='rgba(0, 0, 0, 0.1)',  # Softer axis lines
            range=[0.5, 12.5],
            row=i,
            col=1
        )
        
        # Update y-axes
        if i == 1:
            fig.update_yaxes(
                title_text="<b>Number of Emails Sent</b>",
                tickformat=",",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.1)',  # Softer grid
                showline=True,
                linewidth=1,
                linecolor='rgba(0, 0, 0, 0.1)',  # Softer axis lines
                row=i,
                col=1
            )
        else:
            fig.update_yaxes(
                title_text="<b>Rate</b>",
                tickformat='.1%',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.1)',  # Softer grid
                showline=True,
                linewidth=1,
                linecolor='rgba(0, 0, 0, 0.1)',  # Softer axis lines
                row=i,
                col=1
            )
    
    # Update annotation style for a more modern look
    for metric in ['n_sent', 'open_rate', 'ctr']:
        for year in range(2023, 2025):
            prev_year_data = monthly_stats[monthly_stats['year'] == year-1]
            curr_year_data = monthly_stats[monthly_stats['year'] == year]
            
            if not prev_year_data.empty and not curr_year_data.empty:
                prev_year_avg = prev_year_data[metric].mean()
                curr_year_avg = curr_year_data[metric].mean()
                yoy_change = (curr_year_avg - prev_year_avg) / prev_year_avg * 100
                
                row = 1 if metric == 'n_sent' else 2 if metric == 'open_rate' else 3
                
                # Add annotation with softer styling
                fig.add_annotation(
                    x=12.2,
                    y=curr_year_data[metric].iloc[-1] if not curr_year_data.empty else 0,
                    text=f'<b>{year} vs {year-1}:</b><br>{yoy_change:+.1f}%',
                    showarrow=False,
                    xanchor='left',
                    yanchor='middle',
                    font=dict(
                        size=11,
                        color=colors[year],
                        family='Arial'
                    ),
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor=colors[year],
                    borderwidth=1,
                    borderpad=4,
                    row=row,
                    col=1
                )
    
    return fig

def main():
    # Create the visualization
    fig = create_yoy_comparison()
    
    # Show the plot
    fig.show()

if __name__ == "__main__":
    main() 