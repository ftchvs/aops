#!/usr/bin/env python3

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Load and prepare the campaign performance data."""
    df = pd.read_csv(file_path, parse_dates=['date_sent'])
    
    # Calculate metrics
    df['open_rate'] = df['n_open'] / df['n_sent']
    df['ctr'] = df['n_click'] / df['n_open']
    
    # Add time components
    df['quarter'] = df['date_sent'].dt.to_period('Q')
    df['month'] = df['date_sent'].dt.month
    df['year'] = df['date_sent'].dt.year
    
    return df

def create_volume_engagement_plot(file_path):
    """Create a plot comparing send volume with engagement metrics over time."""
    # Load and prepare data
    df = pd.read_csv(file_path, parse_dates=['date_sent'])
    df['open_rate'] = df['n_open'] / df['n_sent']
    df['ctr'] = df['n_click'] / df['n_open']
    df['quarter'] = df['date_sent'].dt.to_period('Q')
    
    # Create quarterly aggregates
    quarterly = df.groupby('quarter').agg({
        'n_sent': 'sum',
        'open_rate': 'mean',
        'ctr': 'mean'
    }).reset_index()
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add send volume bars
    fig.add_trace(
        go.Bar(
            x=quarterly['quarter'].astype(str),
            y=quarterly['n_sent'],
            name='Total Sent',
            marker_color='rgba(135, 206, 250, 0.8)',  # Light blue
            showlegend=True
        ),
        secondary_y=False
    )
    
    # Add Open Rate line
    fig.add_trace(
        go.Scatter(
            x=quarterly['quarter'].astype(str),
            y=quarterly['open_rate'],
            name='Open Rate',
            line=dict(color='#E74C3C', width=2),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    # Add CTR line
    fig.add_trace(
        go.Scatter(
            x=quarterly['quarter'].astype(str),
            y=quarterly['ctr'],
            name='CTR',
            line=dict(color='#2ECC71', width=2),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title='Send Volume vs Engagement Metrics by Quarter',
        xaxis_title='Quarter',
        template='plotly_white',
        height=600,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            dtick=50000,
            range=[0, 400000],  # Fixed range to match reference
            tickfont=dict(size=10),
            title_font=dict(size=12)
        ),
        yaxis2=dict(
            showgrid=False,
            range=[0.10, 0.35],  # Range from 10% to 35%
            tickformat='.0%',
            tickfont=dict(size=10),
            title_font=dict(size=12)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update axes labels
    fig.update_yaxes(
        title_text="Total Emails Sent", 
        secondary_y=False,
        tickprefix="",  # Remove any prefix
        ticksuffix="k",  # Add k suffix
        tickvals=[0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000],
        ticktext=['0', '50k', '100k', '150k', '200k', '250k', '300k', '350k', '400k']
    )
    fig.update_yaxes(title_text="Engagement Rate", secondary_y=True)
    
    # Update x-axis
    fig.update_xaxes(
        tickangle=0,
        tickfont=dict(size=10),
        title_font=dict(size=12)
    )
    
    return fig

def analyze_volume_impact(df):
    """Analyze the impact of send volume on engagement metrics."""
    # Calculate volume segments
    df['volume_segment'] = pd.qcut(df['n_sent'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Calculate metrics by volume segment
    volume_analysis = df.groupby('volume_segment').agg({
        'n_sent': ['mean', 'count'],
        'open_rate': ['mean', 'std'],
        'ctr': ['mean', 'std']
    }).round(4)
    
    # Create visualization
    fig = go.Figure()
    
    segments = volume_analysis.index
    
    # Add bars for open rate
    fig.add_trace(go.Bar(
        name='Open Rate',
        x=segments,
        y=volume_analysis['open_rate']['mean'],
        error_y=dict(
            type='data',
            array=volume_analysis['open_rate']['std'],
            visible=True
        ),
        marker_color='#E74C3C',
        opacity=0.7
    ))
    
    # Add bars for CTR
    fig.add_trace(go.Bar(
        name='CTR',
        x=segments,
        y=volume_analysis['ctr']['mean'],
        error_y=dict(
            type='data',
            array=volume_analysis['ctr']['std'],
            visible=True
        ),
        marker_color='#2ECC71',
        opacity=0.7
    ))
    
    # Update layout
    fig.update_layout(
        title='Engagement Metrics by Send Volume Segment',
        xaxis_title='Send Volume Segment',
        yaxis_title='Rate',
        template='plotly_white',
        height=500,
        width=1000,
        barmode='group',
        yaxis_tickformat='.1%'
    )
    
    return fig, volume_analysis

def create_daily_trend_analysis(df):
    """Create daily trend analysis with smoothed curves."""
    # Calculate daily metrics
    daily_stats = df.groupby('date_sent').agg({
        'n_sent': 'sum',
        'open_rate': 'mean',
        'ctr': 'mean'
    }).reset_index()
    
    # Apply smoothing
    window = min(15, len(daily_stats) - (len(daily_stats) % 2 - 1))
    if window > 2:
        daily_stats['ctr_smooth'] = savgol_filter(daily_stats['ctr'], window, 3)
        daily_stats['open_rate_smooth'] = savgol_filter(daily_stats['open_rate'], window, 3)
    else:
        daily_stats['ctr_smooth'] = daily_stats['ctr'].rolling(window=3, center=True).mean()
        daily_stats['open_rate_smooth'] = daily_stats['open_rate'].rolling(window=3, center=True).mean()
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add send volume bars
    fig.add_trace(
        go.Bar(
            x=daily_stats['date_sent'],
            y=daily_stats['n_sent'],
            name='Daily Sends',
            marker_color='#3498DB',
            opacity=0.3
        ),
        secondary_y=False
    )
    
    # Add engagement metrics
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date_sent'],
            y=daily_stats['open_rate_smooth'],
            name='Open Rate (Smoothed)',
            line=dict(color='#E74C3C', width=2)
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_stats['date_sent'],
            y=daily_stats['ctr_smooth'],
            name='CTR (Smoothed)',
            line=dict(color='#2ECC71', width=2)
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title='Daily Send Volume and Engagement Trends',
        xaxis_title='Date',
        template='plotly_white',
        height=600,
        width=1000,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    fig.update_yaxes(title_text="Daily Emails Sent", secondary_y=False)
    fig.update_yaxes(title_text="Engagement Rate", tickformat='.1%', secondary_y=True)
    
    return fig

def main():
    # Create and show the plot
    fig = create_volume_engagement_plot('Felipe Chaves_TakeHome Exercise - Campaign_Performance.csv')
    fig.show()

if __name__ == "__main__":
    main() 