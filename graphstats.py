import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_executive_dashboard(df):
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Monthly Performance Trends",
            "Seasonal Impact on Performance",
            "Year-over-Year Comparison",
            "Recent vs Historical Performance"
        ),
        specs=[[{"secondary_y": True}, {}],
               [{}, {}]]
    )
    
    # 1. Monthly Performance Trends
    df['month_year'] = df['date_sent'].dt.to_period('M').astype(str)
    monthly_stats = df.groupby('month_year').agg({
        'open_rate': 'mean',
        'ctr': 'mean'
    }).reset_index()
    
    fig.add_trace(
        go.Scatter(
            x=monthly_stats['month_year'],
            y=monthly_stats['open_rate'] * 100,
            name="Open Rate",
            line=dict(color="#E74C3C", width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_stats['month_year'],
            y=monthly_stats['ctr'] * 100,
            name="CTR",
            line=dict(color="#2ECC71", width=2)
        ),
        row=1, col=1
    )
    
    # 2. Seasonal Impact (Monthly Averages)
    monthly_impact = df.groupby(df['date_sent'].dt.month).agg({
        'open_rate': ['mean', 'std'],
        'ctr': ['mean', 'std']
    }).reset_index()
    
    fig.add_trace(
        go.Bar(
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=monthly_impact['open_rate']['mean'] * 100,
            name="Monthly Open Rate",
            marker_color="#3498DB",
            error_y=dict(
                type='data',
                array=monthly_impact['open_rate']['std'] * 100,
                visible=True
            )
        ),
        row=1, col=2
    )
    
    # 3. Year-over-Year Comparison
    yearly_stats = df.groupby(df['date_sent'].dt.year).agg({
        'open_rate': 'mean',
        'ctr': 'mean'
    }).reset_index()
    
    fig.add_trace(
        go.Bar(
            x=yearly_stats['date_sent'],
            y=yearly_stats['open_rate'] * 100,
            name="Yearly Open Rate",
            marker_color="#E74C3C",
            width=0.4
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=yearly_stats['date_sent'],
            y=yearly_stats['ctr'] * 100,
            name="Yearly CTR",
            marker_color="#2ECC71",
            width=0.4
        ),
        row=2, col=1
    )
    
    # 4. Recent vs Historical Performance
    recent_cutoff = df['date_sent'].max() - pd.Timedelta(days=30)
    performance_comparison = pd.DataFrame({
        'Period': ['Recent', 'Historical'],
        'Open Rate': [
            df[df['date_sent'] >= recent_cutoff]['open_rate'].mean() * 100,
            df[df['date_sent'] < recent_cutoff]['open_rate'].mean() * 100
        ],
        'CTR': [
            df[df['date_sent'] >= recent_cutoff]['ctr'].mean() * 100,
            df[df['date_sent'] < recent_cutoff]['ctr'].mean() * 100
        ]
    })
    
    fig.add_trace(
        go.Bar(
            x=performance_comparison['Period'],
            y=performance_comparison['Open Rate'],
            name="Open Rate Comparison",
            marker_color="#E74C3C"
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=performance_comparison['Period'],
            y=performance_comparison['CTR'],
            name="CTR Comparison",
            marker_color="#2ECC71"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        showlegend=True,
        title_text="Email Marketing Performance Dashboard",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes labels
    fig.update_yaxes(title_text="Percentage (%)")
    fig.update_xaxes(title_text="Time Period")
    
    return fig

# Create and display the dashboard
df = pd.read_csv('Felipe Chaves_TakeHome Exercise - Campaign_Performance.csv', 
                 parse_dates=['date_sent'])
df['open_rate'] = df['n_open'] / df['n_sent']
df['ctr'] = df['n_click'] / df['n_open']

dashboard = create_executive_dashboard(df)
dashboard.show()