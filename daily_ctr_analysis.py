#!/usr/bin/env python3

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def smooth_data(x, y, smoothing_factor=0.8):
    """Create smoothed version of data using B-spline interpolation"""
    # Handle NaN and inf values
    y = pd.Series(y)
    y = y.replace([np.inf, -np.inf], np.nan)
    
    # Interpolate NaN values
    y = y.interpolate(method='linear', limit_direction='both')
    
    # If still have NaN values after interpolation, fill with mean
    if y.isna().any():
        y = y.fillna(y.mean())
    
    # Convert to numpy array
    y = y.to_numpy()
    
    # Ensure minimum length for smoothing
    if len(y) < 4:
        # Return original data if too short
        return x, y
        
    try:
        # Convert dates to numbers for interpolation
        x_numeric = np.arange(len(x))
        
        # Create smooth curve using B-spline
        spl = make_interp_spline(x_numeric, y, k=min(3, len(y)-1))
        
        # Generate smooth points
        x_smooth = np.linspace(0, len(x)-1, 300)
        y_smooth = spl(x_smooth)
        
        # Additional Gaussian smoothing
        y_smooth = gaussian_filter1d(y_smooth, sigma=smoothing_factor*10)
        
        # Map x_smooth back to dates
        date_range = pd.date_range(start=x.min(), end=x.max(), periods=len(x_smooth))
        
        return date_range, y_smooth
    except Exception as e:
        print(f"Smoothing failed: {str(e)}")
        return x, y

def create_enhanced_daily_ctr_analysis(daily_stats):
    """Create an enhanced visualization of daily CTR analysis"""
    try:
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add smoothed CTR line
        dates_smooth, ctr_mean_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_mean'])
        _, ctr_upper_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_ci_upper'])
        _, ctr_lower_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_ci_lower'])
        _, trend_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_trend'], smoothing_factor=1.2)
        
        # Add smoothed trend for email sends
        _, sends_trend_smooth = smooth_data(daily_stats['date'], daily_stats['total_sent'], smoothing_factor=1.5)

        # Calculate year-over-year growth rates using full year data
        daily_stats['year'] = daily_stats['date'].dt.year
        
        # Group by year and calculate annual metrics
        yearly_stats = daily_stats.groupby('year').agg({
            'total_sent': 'sum',
            'ctr_mean': 'mean'
        }).reset_index()
        
        # Get current and previous year
        current_year = yearly_stats['year'].max()
        previous_year = current_year - 1
        
        # Calculate YoY growth comparing full years
        if previous_year in yearly_stats['year'].values:
            current_year_data = yearly_stats[yearly_stats['year'] == current_year]
            previous_year_data = yearly_stats[yearly_stats['year'] == previous_year]
            
            # Calculate volume growth
            volume_growth = ((current_year_data['total_sent'].iloc[0] - 
                            previous_year_data['total_sent'].iloc[0]) / 
                           previous_year_data['total_sent'].iloc[0] * 100)
            
            # Calculate CTR growth
            ctr_growth = ((current_year_data['ctr_mean'].iloc[0] - 
                          previous_year_data['ctr_mean'].iloc[0]) / 
                         previous_year_data['ctr_mean'].iloc[0] * 100)
            
            # Create period label for annotation
            period_label = f"{previous_year} vs {current_year}"
        else:
            # Fallback if we don't have full previous year data
            volume_growth = 0
            ctr_growth = 0
            period_label = f"{current_year}"

        # Add confidence interval for CTR
        fig.add_trace(
            go.Scatter(
                x=dates_smooth.tolist() + dates_smooth.tolist()[::-1],
                y=ctr_upper_smooth.tolist() + ctr_lower_smooth.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(231,234,241,0.5)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=True,
                name='95% Confidence Band'
            ),
            secondary_y=True
        )

        # Add email send volume as bars with reduced opacity
        fig.add_trace(
            go.Bar(
                x=daily_stats['date'],
                y=daily_stats['total_sent'],
                name='Daily Emails Sent',
                marker_color='rgba(52, 152, 219, 0.3)',
                hovertemplate='Date: %{x}<br>Sent: %{y:,}<extra></extra>'
            ),
            secondary_y=False
        )

        # Add trend line for email sends
        fig.add_trace(
            go.Scatter(
                x=dates_smooth,
                y=sends_trend_smooth,
                mode='lines',
                line=dict(
                    color='rgba(52, 152, 219, 0.9)',
                    width=3,
                    shape='spline'
                ),
                name='Email Volume Trend'
            ),
            secondary_y=False
        )

        # Add actual CTR values
        fig.add_trace(
            go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['ctr_mean'],
                mode='markers',
                marker=dict(
                    size=6,
                    color='rgba(255, 65, 54, 0.5)',
                ),
                name='Daily CTR',
                hovertemplate='Date: %{x}<br>CTR: %{y:.2%}<extra></extra>'
            ),
            secondary_y=True
        )

        # Add smoothed trend line for CTR
        fig.add_trace(
            go.Scatter(
                x=dates_smooth,
                y=trend_smooth,
                mode='lines',
                line=dict(
                    color='rgba(255, 65, 54, 0.8)',
                    width=3,
                    shape='spline'
                ),
                name='CTR Trend'
            ),
            secondary_y=True
        )

        # Update layout with modern styling
        fig.update_layout(
            title=dict(
                text='Daily Click-to-Open Rate and Send Volume Analysis<br><sup>Showing parallel growth in both metrics</sup>',
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20)
            ),
            xaxis_title='Date',
            yaxis_title='Number of Emails Sent',
            yaxis2_title='Click-to-Open Rate (CTR)',
            template='plotly_white',
            height=700,
            width=1200,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.1)',
                borderwidth=1
            ),
            margin=dict(l=60, r=60, t=100, b=60),  # Reduced bottom margin
            plot_bgcolor='rgba(250,250,250,0.9)',
            paper_bgcolor='white'
        )

        # Add a semi-transparent background box for annotations
        fig.add_shape(
            type="rect",
            x0=0.35,
            y0=0.85,
            x1=0.65,
            y1=0.95,
            xref="paper",
            yref="paper",
            fillcolor="white",
            opacity=0.8,
            layer="below",
            line_width=0,
        )

        # Add title for growth metrics in the center
        fig.add_annotation(
            x=0.5,
            y=0.92,
            xref='paper',
            yref='paper',
            text=f"<b>Year-over-Year Growth</b>",
            showarrow=False,
            font=dict(
                size=14,
                color='black'
            ),
            align='center'
        )

        # Add volume growth annotation
        fig.add_annotation(
            x=0.4,
            y=0.88,
            xref='paper',
            yref='paper',
            text=f"<span style='color: rgba(52, 152, 219, 1)'>Volume</span> ({period_label}):<br><b>{volume_growth:+.1f}%</b>",
            showarrow=False,
            font=dict(
                size=12,
                color='black'
            ),
            align='center'
        )

        # Add CTR growth annotation
        fig.add_annotation(
            x=0.6,
            y=0.88,
            xref='paper',
            yref='paper',
            text=f"<span style='color: rgba(255, 65, 54, 1)'>CTR</span> ({period_label}):<br><b>{ctr_growth:+.1f}%</b>",
            showarrow=False,
            font=dict(
                size=12,
                color='black'
            ),
            align='center'
        )

        # Update axes with modern styling
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.1)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)'
        )
        
        # Update primary y-axis (email volume)
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.1)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)',
            secondary_y=False
        )
        
        # Update secondary y-axis (CTR)
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.1)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)',
            tickformat='.1%',
            secondary_y=True
        )

        return fig

    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None

def main():
    # Load and prepare data
    df = pd.read_csv('Felipe Chaves_TakeHome Exercise - Campaign_Performance.csv', 
                     parse_dates=['date_sent'])
    
    # Calculate metrics
    df['open_rate'] = df['n_open']/df['n_sent']
    df['ctr'] = np.where(df['n_open'] > 0, df['n_click']/df['n_open'], 0)
    
    # Create daily stats
    daily_stats = df.groupby('date_sent').agg({
        'n_sent': 'sum',
        'n_open': 'sum',
        'n_click': 'sum'
    }).reset_index()
    
    daily_stats.columns = ['date', 'total_sent', 'total_opened', 'total_clicks']
    daily_stats['ctr_mean'] = daily_stats['total_clicks'] / daily_stats['total_sent']
    daily_stats['ctr_se'] = daily_stats['ctr_mean'] / np.sqrt(daily_stats['total_sent'])
    daily_stats['ctr_ci_lower'] = daily_stats['ctr_mean'] - 1.96 * daily_stats['ctr_se']
    daily_stats['ctr_ci_upper'] = daily_stats['ctr_mean'] + 1.96 * daily_stats['ctr_se']
    
    # Calculate trend
    window_length = min(15, len(daily_stats) - (len(daily_stats) % 2 - 1))
    if window_length > 2:
        daily_stats['ctr_trend'] = gaussian_filter1d(daily_stats['ctr_mean'], sigma=3)
    else:
        daily_stats['ctr_trend'] = daily_stats['ctr_mean'].rolling(window=3, center=True).mean()
    
    # Create and show visualization
    fig = create_enhanced_daily_ctr_analysis(daily_stats)
    if fig:
        fig.show()

if __name__ == "__main__":
    main() 