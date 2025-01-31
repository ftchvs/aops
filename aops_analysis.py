#!/usr/bin/env python3

# 1. Setup and Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.core.groupby")

# Set style for static plots
plt.style.use('seaborn-v0_8')  # Use the updated seaborn style name
sns.set_theme(style="whitegrid")  # Set seaborn theme

# Try importing interactive visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Plotly not available. Some interactive visualizations will be disabled.")
    PLOTLY_AVAILABLE = False

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    print("Altair not available. Some visualizations will use alternative libraries.")
    ALTAIR_AVAILABLE = False

# Load data
df = pd.read_csv('Felipe Chaves_TakeHome Exercise - Campaign_Performance.csv', 
                 parse_dates=['date_sent'])

# Add data validation after loading CSV
assert (df['n_sent'] >= df['n_open']).all(), "Invalid data: Opens exceed sent emails"
assert (df['n_open'] >= df['n_click']).all(), "Invalid data: Clicks exceed opens"

# Update metric calculations
df['open_rate'] = df['n_open']/df['n_sent']  # Explicit calculation
df['ctr'] = np.where(df['n_open'] > 0, df['n_click']/df['n_open'], 0)

# Add date component extraction
df['day_of_week'] = df['date_sent'].dt.day_name()
df['month'] = df['date_sent'].dt.month
df['year'] = df['date_sent'].dt.year

# Add new practical metrics
df['conversion_rate'] = df['n_click']/df['n_sent']  # Overall conversion
df['click_to_open_ratio'] = df['n_click']/df['n_open']  # Content effectiveness

# Update daily stats calculation
daily_stats = df.groupby('date_sent').agg({
    'n_sent': 'sum',  # Total emails sent per day
    'n_open': 'sum',
    'n_click': 'sum'
}).reset_index()

daily_stats.columns = ['date', 'total_sent', 'total_opened', 'total_clicks']
daily_stats['ctr_mean'] = daily_stats['total_clicks'] / daily_stats['total_sent']
daily_stats['ctr_se'] = daily_stats['ctr_mean'] / np.sqrt(daily_stats['total_sent'])
daily_stats['ctr_ci_lower'] = daily_stats['ctr_mean'] - 1.96 * daily_stats['ctr_se']
daily_stats['ctr_ci_upper'] = daily_stats['ctr_mean'] + 1.96 * daily_stats['ctr_se']

# Calculate trend using Savitzky-Golay filter
window_length = min(15, len(daily_stats) - (len(daily_stats) % 2 - 1))  # Must be odd and less than data length
if window_length > 2:
    daily_stats['ctr_trend'] = savgol_filter(daily_stats['ctr_mean'], window_length, 3)
else:
    # Fallback to simple moving average if not enough data
    daily_stats['ctr_trend'] = daily_stats['ctr_mean'].rolling(window=3, center=True).mean()

# Create monthly aggregates for YoY comparison
monthly_stats = df.groupby(['year', 'month']).agg({
    'open_rate': 'mean',
    'ctr': 'mean'
}).reset_index()

# Prepare data for YoY comparison plots
def create_yoy_comparison_plots():
    if PLOTLY_AVAILABLE:
        # Create subplots
        fig = make_subplots(rows=2, cols=1,
                          subplot_titles=('Year-Over-Year Open Rate Comparison',
                                        'Year-Over-Year CTR Comparison'),
                          vertical_spacing=0.15)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Default plotly colors
        
        # Plot Open Rate YoY comparison
        for i, year in enumerate(sorted(monthly_stats['year'].unique())):
            year_data = monthly_stats[monthly_stats['year'] == year]
            fig.add_trace(
                go.Scatter(x=year_data['month'],
                          y=year_data['open_rate'],
                          name=f'Open Rate {year}',
                          line=dict(color=colors[i], width=2),
                          hovertemplate='Month: %{x}<br>Open Rate: %{y:.2%}'),
                row=1, col=1
            )

        # Plot CTR YoY comparison
        for i, year in enumerate(sorted(monthly_stats['year'].unique())):
            year_data = monthly_stats[monthly_stats['year'] == year]
            fig.add_trace(
                go.Scatter(x=year_data['month'],
                          y=year_data['ctr'],
                          name=f'CTR {year}',
                          line=dict(color=colors[i], width=2),
                          hovertemplate='Month: %{x}<br>CTR: %{y:.2%}'),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Update axes
        fig.update_xaxes(title_text="Month", tickmode='linear', tick0=1, dtick=1, row=1, col=1)
        fig.update_xaxes(title_text="Month", tickmode='linear', tick0=1, dtick=1, row=2, col=1)
        fig.update_yaxes(title_text="Open Rate", tickformat='.1%', row=1, col=1)
        fig.update_yaxes(title_text="Click-Through Rate", tickformat='.1%', row=2, col=1)

        fig.show()
    else:
        # Matplotlib fallback
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Default matplotlib colors
        
        # Plot Open Rate YoY comparison
        for i, year in enumerate(sorted(monthly_stats['year'].unique())):
            year_data = monthly_stats[monthly_stats['year'] == year]
            ax1.plot(year_data['month'], year_data['open_rate'], 
                    label=f'Open Rate {year}', marker='o', color=colors[i])
        
        ax1.set_title('Year-Over-Year Open Rate Comparison')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Open Rate')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xticks(range(1, 13))
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Plot CTR YoY comparison
        for i, year in enumerate(sorted(monthly_stats['year'].unique())):
            year_data = monthly_stats[monthly_stats['year'] == year]
            ax2.plot(year_data['month'], year_data['ctr'], 
                    label=f'CTR {year}', marker='o', color=colors[i])
        
        ax2.set_title('Year-Over-Year CTR Comparison')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Click-Through Rate')
        ax2.legend()
        ax2.grid(True)
        ax2.set_xticks(range(1, 13))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        plt.tight_layout()
        plt.show()

# Call the function to create YoY comparison plots
create_yoy_comparison_plots()

# 2. Correlation Analysis
correlation_matrix = df[['n_sent', 'n_open', 'n_click', 'open_rate', 'ctr']].corr()

if PLOTLY_AVAILABLE:
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=np.round(correlation_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False))

    fig_corr.update_layout(
        title='Interactive Correlation Matrix',
        height=600,
        width=800)

    fig_corr.show()
else:
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# 3. Performance by Day Analysis
weekly_performance = df.groupby('day_of_week').agg({
    'open_rate': 'mean',
    'ctr': 'mean',
    'n_sent': 'count'
}).round(4)

# Sort days in a logical order
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_performance = weekly_performance.reindex(day_order)

if PLOTLY_AVAILABLE:
    # Create subplots for better visualization
    fig_weekly = make_subplots(specs=[[{"secondary_y": True}]])

    # Add bar for open rate
    fig_weekly.add_trace(
        go.Bar(
            x=weekly_performance.index,
            y=weekly_performance['open_rate'],
            name='Open Rate',
            marker_color='#1f77b4',  # Use hex color
            text=[f'{v:.1%}' for v in weekly_performance['open_rate']],
            textposition='auto',
        ),
        secondary_y=False,
    )

    # Add bar for CTR
    fig_weekly.add_trace(
        go.Bar(
            x=weekly_performance.index,
            y=weekly_performance['ctr'],
            name='Click-Through Rate',
            marker_color='#ff7f0e',  # Use hex color
            text=[f'{v:.1%}' for v in weekly_performance['ctr']],
            textposition='auto',
        ),
        secondary_y=False,
    )

    # Add line for number of campaigns
    fig_weekly.add_trace(
        go.Scatter(
            x=weekly_performance.index,
            y=weekly_performance['n_sent'],
            name='Number of Campaigns',
            line=dict(color='#bdbdbd', width=2, dash='dot'),  # Use hex color
            mode='lines+markers+text',
            text=weekly_performance['n_sent'],
            textposition='top center',
        ),
        secondary_y=True,
    )

    # Update layout
    fig_weekly.update_layout(
        title='Campaign Performance by Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Rate',
        yaxis2_title='Number of Campaigns',
        height=600,
        width=1000,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        yaxis=dict(
            tickformat='.1%',
            range=[0, max(weekly_performance['open_rate'].max(), 
                         weekly_performance['ctr'].max()) * 1.1]
        ),
        yaxis2=dict(
            range=[0, max(weekly_performance['n_sent']) * 1.2]
        ),
        template='plotly_white'
    )

    fig_weekly.show()
else:
    # Enhanced matplotlib fallback
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Set the width of each bar and positions of the bars
    width = 0.35
    x = np.arange(len(weekly_performance.index))
    
    # Create bars
    bars1 = ax1.bar(x - width/2, weekly_performance['open_rate'], width, 
                    label='Open Rate', color='#1f77b4')
    bars2 = ax1.bar(x + width/2, weekly_performance['ctr'], width,
                    label='Click-Through Rate', color='#ff7f0e')
    
    # Add value labels on the bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    
    # Customize primary y-axis
    ax1.set_ylabel('Rate')
    ax1.set_title('Campaign Performance by Day of Week')
    ax1.set_xticks(x)
    ax1.set_xticklabels(weekly_performance.index, rotation=45)
    ax1.legend(loc='upper left')
    
    # Create second y-axis for number of campaigns
    ax2 = ax1.twinx()
    ax2.plot(x, weekly_performance['n_sent'], '--', color='#bdbdbd', 
             label='Number of Campaigns', marker='o')
    
    # Add value labels for number of campaigns
    for i, v in enumerate(weekly_performance['n_sent']):
        ax2.text(i, v, str(v), ha='center', va='bottom')
    
    ax2.set_ylabel('Number of Campaigns')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# 4. Statistical Analysis
recent_cutoff = df['date_sent'].max() - timedelta(days=30)
recent_data = df[df['date_sent'] >= recent_cutoff]
historical_data = df[df['date_sent'] < recent_cutoff]

def perform_analysis(metric):
    t_stat, p_val = stats.ttest_ind(
        recent_data[metric],
        historical_data[metric]
    )
    return {
        'metric': metric,
        'recent_mean': recent_data[metric].mean(),
        'historical_mean': historical_data[metric].mean(),
        'change': (recent_data[metric].mean() - historical_data[metric].mean()) / historical_data[metric].mean(),
        'p_value': p_val
    }

metrics = ['open_rate', 'ctr']
analysis_results = pd.DataFrame([perform_analysis(metric) for metric in metrics])

# Print statistical analysis results
print("\nStatistical Analysis Results:")
print(analysis_results.round(4))

# 5. Recommendations
print("\nKey Recommendations:")
best_days = df.groupby('day_of_week')['ctr'].mean().sort_values(ascending=False).index[:2]

print(f"1. Best days to send (based on CTR): {', '.join(best_days)}")

# Save static plots if interactive versions are not available
if not PLOTLY_AVAILABLE:
    plt.savefig('campaign_analysis.png')
    print("\nStatic plots have been saved to 'campaign_analysis.png'")

# Enhanced smoothing for daily CTR analysis
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

if PLOTLY_AVAILABLE:
    # Create modern daily CTR analysis with smooth bands
    fig_daily_ctr = go.Figure()

    # Create smooth versions of the data
    dates_smooth, ctr_mean_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_mean'])
    _, ctr_upper_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_ci_upper'])
    _, ctr_lower_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_ci_lower'])
    _, trend_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_trend'], smoothing_factor=1.2)

    # Add smoothed confidence interval
    fig_daily_ctr.add_trace(
        go.Scatter(
            x=dates_smooth.tolist() + dates_smooth.tolist()[::-1],
            y=ctr_upper_smooth.tolist() + ctr_lower_smooth.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(231,234,241,0.5)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='95% Confidence Band'
        )
    )

    # Add actual CTR values with smaller points
    fig_daily_ctr.add_trace(
        go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['ctr_mean'],
            mode='markers',
            marker=dict(
                size=6,
                color=daily_stats['total_clicks'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title='Total Clicks',
                    titleside='right',
                    thickness=15,
                    len=0.7
                )
            ),
            name='Daily CTR',
            hovertemplate='Date: %{x}<br>CTR: %{y:.2%}<br>Clicks: %{marker.color}<extra></extra>'
        )
    )

    # Add smoothed trend line
    fig_daily_ctr.add_trace(
        go.Scatter(
            x=dates_smooth,
            y=trend_smooth,
            mode='lines',
            line=dict(
                color='rgba(255, 65, 54, 0.8)',
                width=2.5,
                shape='spline'
            ),
            name='Trend'
        )
    )

    # Update layout with modern styling
    fig_daily_ctr.update_layout(
        title=dict(
            text='Daily Click-to-Open Rate Analysis with Smooth Trends',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        xaxis_title='Date',
        yaxis_title='Click-to-Open Rate (CTR)',
        hovermode='x unified',
        template='plotly_white',
        height=700,
        width=1200,
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
        margin=dict(l=60, r=60, t=80, b=60)
    )

    # Update axes with modern styling
    fig_daily_ctr.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.1)',
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.2)'
    )
    fig_daily_ctr.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.1)',
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor='rgba(0,0,0,0.2)',
        tickformat='.1%'
    )

    # Add subtle background color
    fig_daily_ctr.update_layout(
        plot_bgcolor='rgba(250,250,250,0.9)',
        paper_bgcolor='white'
    )

    fig_daily_ctr.show()

else:
    # Enhanced matplotlib fallback
    plt.figure(figsize=(15, 8))
    
    # Create smooth versions of the data
    dates_smooth, ctr_mean_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_mean'])
    _, ctr_upper_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_ci_upper'])
    _, ctr_lower_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_ci_lower'])
    _, trend_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_trend'], smoothing_factor=1.2)
    
    # Plot smooth confidence interval
    plt.fill_between(dates_smooth, 
                    ctr_lower_smooth,
                    ctr_upper_smooth,
                    alpha=0.2,
                    color='gray',
                    label='95% Confidence Band')
    
    # Plot actual CTR values
    scatter = plt.scatter(daily_stats['date'],
                         daily_stats['ctr_mean'],
                         c=daily_stats['total_clicks'],
                         cmap='viridis',
                         s=30,
                         alpha=0.6,
                         label='Daily CTR')
    
    # Plot smooth trend line
    plt.plot(dates_smooth, trend_smooth,
             color='red',
             linewidth=2,
             label='Trend',
             alpha=0.8)
    
    plt.colorbar(scatter, label='Total Clicks')
    plt.title('Daily Click-to-Open Rate Analysis with Smooth Trends', pad=20, size=14)
    plt.xlabel('Date')
    plt.ylabel('Click-to-Open Rate (CTR)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
plt.show()

# Add Funnel Analysis section before the statistical analysis
print("\nEmail Marketing Funnel Analysis:")
funnel_stages = [
    {'stage': 'Sent', 'count': df['n_sent'].sum()},
    {'stage': 'Opened', 'count': df['n_open'].sum()},
    {'stage': 'Clicked', 'count': df['n_click'].sum()}
]
funnel_data = pd.DataFrame(funnel_stages)

# Calculate conversion rates
funnel_data['conversion_rate'] = funnel_data['count'] / funnel_data['count'].shift(-1).fillna(funnel_data['count'])
funnel_data['previous_stage_rate'] = (funnel_data['count'] / funnel_data['count'].shift(1)).fillna(1)

if PLOTLY_AVAILABLE:
    # Create modern funnel visualization
    fig_funnel = go.Figure()

    # Add main funnel with improved colors and formatting
    fig_funnel.add_trace(go.Funnel(
        name='Conversion Funnel',
        y=funnel_data['stage'],
        x=funnel_data['count'],
        textposition="inside",
        textinfo="value",
        texttemplate="<b>%{value:,.0f}</b>",  # Bold numbers with commas
        opacity=0.9,
        marker={
            "color": ["#3498DB", "#2ECC71", "#F1C40F"],  # Brighter, more vibrant colors
            "line": {"width": [2, 2, 2], "color": ["white", "white", "white"]}
        },
        connector={"line": {"color": "#E0E0E0", "width": 1}}
    ))

    # Add stage labels on the left
    for i, row in funnel_data.iterrows():
        fig_funnel.add_annotation(
            x=0,  # Left side
            y=row['stage'],
            text=f"<b>{row['stage']}</b>",
            showarrow=False,
            font=dict(size=14, color="#2C3E50", family="Arial"),
            xanchor='right',
            xshift=-10
        )

    # Add percentage labels inside the funnel
    for i, row in funnel_data.iterrows():
        percentage = 100 * row['count']/funnel_data.iloc[0]['count']
        fig_funnel.add_annotation(
            x=row['count']/2,  # Center of the bar
            y=row['stage'],
            text=f"<b>{percentage:.1f}%</b>",
            showarrow=False,
            font=dict(size=16, color="white", family="Arial"),
            xanchor='center'
        )

    # Add conversion rate annotations on the right with arrows
    for i in range(len(funnel_data)-1):
        current_stage = funnel_data.iloc[i]
        next_stage = funnel_data.iloc[i+1]
        conversion_rate = (next_stage['count'] / current_stage['count']) * 100
        
        fig_funnel.add_annotation(
            x=1.1 * max(funnel_data['count']),  # Slightly closer to the funnel
            y=current_stage['stage'],
            text=f"↓ {conversion_rate:.1f}% conversion",  # Added arrow
            showarrow=False,
            font=dict(size=14, color="#34495E", family="Arial"),
            xanchor='left',
            yanchor='bottom'
        )

    # Update layout with modern styling
    fig_funnel.update_layout(
        title=dict(
            text='Email Marketing Funnel Analysis',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(
                size=24,
                family="Arial",
                color="#2C3E50"
            )
        ),
        showlegend=False,
        width=1000,
        height=500,
        template='plotly_white',
        margin=dict(l=150, r=200, t=120, b=80),
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,240,0.0)',
        font=dict(
            family="Arial",
            size=14,
            color="#2C3E50"
        )
    )

    # Add subtitle with key metrics
    total_sent = funnel_data.iloc[0]['count']
    total_opened = funnel_data.iloc[1]['count']
    total_clicks = funnel_data.iloc[2]['count']
    
    fig_funnel.add_annotation(
        x=0.5,
        y=1.1,  # Moved up slightly
        xref='paper',
        yref='paper',
        text=f'<b>Campaign Overview:</b> {total_sent:,.0f} Total Emails • {(total_clicks/total_sent)*100:.1f}% Overall Conversion',
        showarrow=False,
        font=dict(size=16, color="#34495E", family="Arial"),
        xanchor='center'
    )

    # Add key metrics below the title
    metrics_text = (
        f"Open Rate: {(total_opened/total_sent)*100:.1f}% • "
        f"Click-to-Open Rate: {(total_clicks/total_opened)*100:.1f}% • "
        f"Overall Click Rate: {(total_clicks/total_sent)*100:.1f}%"
    )
    
    fig_funnel.add_annotation(
        x=0.5,
        y=1.02,  # Position between title and subtitle
        xref='paper',
        yref='paper',
        text=metrics_text,
        showarrow=False,
        font=dict(size=14, color="#7F8C8D", family="Arial"),
        xanchor='center'
    )

    # Add explanatory notes at the bottom
    fig_funnel.add_annotation(
        x=0.5,
        y=-0.15,
        xref='paper',
        yref='paper',
        text='Note: Percentages inside funnel show total reach at each stage. Conversion rates show step-by-step progression.',
        showarrow=False,
        font=dict(size=12, color="#95A5A6", family="Arial"),
        xanchor='center'
    )

    fig_funnel.show()

else:
    # Enhanced Matplotlib fallback for funnel visualization
    plt.figure(figsize=(12, 8))
    
    # Create funnel plot with improved colors
    bars = plt.bar(funnel_data['stage'], funnel_data['count'], 
                  color=['#3498DB', '#2ECC71', '#F1C40F'])
    
    # Add value and percentage labels
    for i, (bar, value) in enumerate(zip(bars, funnel_data['count'])):
        # Add absolute value
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{value:,.0f}',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold',
                color='#2C3E50')
        
        # Add percentage of total
        percentage = value / funnel_data.iloc[0]['count'] * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{percentage:.1f}%',
                ha='center', va='center',
                color='white',
                fontsize=14, fontweight='bold')
        
        # Add conversion rate arrow
        if i < len(funnel_data)-1:
            conv_rate = (funnel_data.iloc[i+1]['count'] / value) * 100
            plt.text(i+0.7, bar.get_height()/2,
                    f'↓ {conv_rate:.1f}%',
                    ha='left', va='center',
                    color='#34495E',
                    fontsize=12)
    
    plt.title('Email Marketing Funnel Analysis', 
              pad=20, fontsize=16, fontweight='bold',
              color='#2C3E50')
    
    # Add subtitle with metrics
    total_sent = funnel_data.iloc[0]['count']
    total_opened = funnel_data.iloc[1]['count']
    total_clicks = funnel_data.iloc[2]['count']
    
    metrics_text = (
        f'Campaign Overview: {total_sent:,.0f} Total Emails • {(total_clicks/total_sent)*100:.1f}% Overall Conversion\n'
        f'Open Rate: {(total_opened/total_sent)*100:.1f}% • '
        f'Click-to-Open Rate: {(total_clicks/total_opened)*100:.1f}% • '
        f'Overall Click Rate: {(total_clicks/total_sent)*100:.1f}%'
    )
    
    plt.suptitle(metrics_text,
                 y=0.95, fontsize=12, color='#34495E')
    
    plt.xticks(rotation=0, fontsize=12)
    plt.ylabel('Number of Events', fontsize=12, color='#2C3E50')
    
    # Add explanatory note
    plt.figtext(0.5, 0.02,
                'Note: Percentages inside funnel show total reach at each stage. Conversion rates show step-by-step progression.',
                ha='center', fontsize=10, color='#95A5A6')
    
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
plt.show()

# Print detailed funnel metrics
print("\nDetailed Funnel Metrics:")
print(f"Total Emails Sent: {funnel_data.iloc[0]['count']:,.0f}")
print(f"Total Emails Opened: {funnel_data.iloc[1]['count']:,.0f}")
print(f"Total Clicks: {funnel_data.iloc[2]['count']:,.0f}")

# Calculate additional funnel metrics
avg_daily_stats = df.groupby('date_sent').agg({
    'n_sent': 'mean',
    'n_open': 'mean',
    'n_click': 'mean'
}).mean()

print("\nAverage Daily Performance:")
print(f"Average Emails Sent per Day: {avg_daily_stats['n_sent']:,.1f}")
print(f"Average Opens per Day: {avg_daily_stats['n_open']:,.1f}")
print(f"Average Clicks per Day: {avg_daily_stats['n_click']:,.1f}")

# Add Advanced Analysis Section
print("\n=== Advanced Campaign Analysis ===")

# 1. Campaign Size Analysis
size_performance = df.copy()
size_bins = [0, 5000, 10000, 15000, df['n_sent'].max()]
size_performance['size_category'] = pd.cut(df['n_sent'], bins=size_bins, 
                                            labels=['Small', 'Medium', 'Large', 'Very Large'])
size_metrics = size_performance.groupby('size_category', observed=True).agg({
    'n_sent': 'mean',
    'open_rate': 'mean',
    'ctr': 'mean',
}).round(4)

print("\n1. Campaign Size Impact:")
print(size_metrics)
optimal_size = size_metrics['ctr'].idxmax()
print(f"\nOptimal campaign size category: {optimal_size}")

# 2. Weekly Pattern Analysis
weekly_metrics = df.groupby('day_of_week').agg({
    'open_rate': 'mean',
    'ctr': 'mean',
    'n_sent': 'count'
}).round(4)

print("\n2. Day of Week Performance:")
print(weekly_metrics)

# 3. Engagement Consistency Analysis
consistency_metrics = df.groupby('date_sent').agg({
    'open_rate': ['mean', 'std'],
    'ctr': ['mean', 'std']
}).round(4)

print("\n3. Engagement Consistency:")
print("Variability in metrics:")
print(f"Open Rate - Mean Std: {consistency_metrics['open_rate']['std'].mean():.4f}")
print(f"CTR - Mean Std: {consistency_metrics['ctr']['std'].mean():.4f}")

# 4. Calculate month-over-month growth
df['month_year'] = df['date_sent'].dt.to_period('M')
monthly_metrics = df.groupby('month_year').agg({
    'open_rate': 'mean',
    'ctr': 'mean',
    'n_sent': 'sum',
    'n_click': 'sum',
}).round(4)

monthly_growth = monthly_metrics.pct_change() * 100

print("\n4. Month-over-Month Growth:")
print(monthly_growth.tail().round(2))

# Calculate overall trend direction
recent_months = monthly_metrics.tail(3)
trend_direction = 'Improving' if recent_months['ctr'].diff().mean() > 0 else 'Declining'

# Comprehensive Recommendations
print("\n=== Comprehensive Recommendations ===")

# 1. Campaign Size Optimization
optimal_size_range = size_performance[size_performance['size_category'] == optimal_size]['n_sent']
print(f"\n1. Campaign Size Optimization:")
print(f"   - Optimal campaign size range: {optimal_size_range.min():,.0f} to {optimal_size_range.max():,.0f} recipients")
print(f"   - Expected CTR: {size_metrics.loc[optimal_size, 'ctr']:.2%}")

# 2. Timing Optimization
print("\n2. Timing Optimization:")
print(f"   - Best days: {', '.join(weekly_metrics.nlargest(2, 'ctr').index)}")
print("   - Consider A/B testing different send days")

# 3. Engagement Strategy
print("\n3. Engagement Strategy:")
high_engagement_campaigns = df[df['ctr'] > df['ctr'].quantile(0.75)]
print("   High-engagement campaign characteristics:")
print(f"   - Typical send size: {high_engagement_campaigns['n_sent'].mean():,.0f} recipients")
print(f"   - Average CTR: {high_engagement_campaigns['ctr'].mean():.2%}")

# 4. Performance Improvement Opportunities
print("\n4. Performance Improvement Opportunities:")
print(f"   - Current average CTR: {df['ctr'].mean():.2%}")
print("   - Potential improvements:")
print(f"     * Target CTR: {df['ctr'].quantile(0.75):.2%} (75th percentile)")

# 5. Risk Mitigation
print("\n5. Risk Mitigation:")
poor_performing_segments = df[df['ctr'] < df['ctr'].quantile(0.25)]
print("   Watch out for:")
if not poor_performing_segments.empty:
    print(f"   - Problematic campaign sizes: {poor_performing_segments['n_sent'].mean():,.0f} recipients")
    print("   - Consider segmenting these audiences for targeted re-engagement campaigns")

# 6. Action Items
print("\n6. Immediate Action Items:")
print("   1. Implement optimal campaign size segmentation")
print("   2. Adjust send days to match best-performing days")
print("   3. Develop re-engagement strategy for low-performing segments")
print("   4. Set up A/B testing for subject lines and content")
print("   5. Create automated alerts for engagement metrics below thresholds")

# 7. Additional Insights
print("\n7. Additional Insights:")
print("   Performance Trends:")
print(f"   - Recent trend direction: {trend_direction}")
print(f"   - Last month change: {monthly_growth['ctr'].iloc[-1]:.1f}%")

print("\n   Key Findings:")
print("   - Campaign Size Impact:")
print(f"     * Largest campaigns ({optimal_size}) show {(size_metrics.loc[optimal_size, 'ctr'] - size_metrics['ctr'].mean()) * 100:.1f}% better CTR")
print("   - Timing Patterns:")
print(f"     * Best performing day ({weekly_metrics['ctr'].idxmax()}) shows {(weekly_metrics['ctr'].max() - weekly_metrics['ctr'].mean()) * 100:.1f}% above average CTR")

print("\n   Strategic Recommendations:")
print("   1. Audience Segmentation:")
print("      - Segment by CTR level (high, medium, low)")
print("      - Create targeted content for each segment")
print("      - Implement re-engagement campaigns for inactive subscribers")

print("   2. Testing Strategy:")
print("      - A/B test subject lines")
print("      - Test different content formats")
print("      - Experiment with send days")

print("   3. Monitoring and Optimization:")
print("      - Track CTR metrics weekly")
print("      - Set up automated performance alerts")
print("      - Regular list maintenance")

print("   4. Content Strategy:")
print("      - Focus on high-performing content types")
print("      - Personalize content based on CTR history")
print("      - Optimize for mobile viewing")

# Enhanced Statistical Analysis
print("\n=== Detailed Performance Analysis ===")

# Calculate month-over-month metrics with statistical tests
df['month_year'] = df['date_sent'].dt.to_period('M')
monthly_metrics = df.groupby('month_year').agg({
    'open_rate': ['mean', 'std', 'count'],
    'ctr': ['mean', 'std', 'count'],
    'n_sent': 'sum',
    'n_click': 'sum'
}).round(4)

# Flatten column names
monthly_metrics.columns = ['_'.join(col).strip() for col in monthly_metrics.columns.values]

# Calculate month-over-month changes
monthly_changes = monthly_metrics.pct_change() * 100

# Focus on last 3 months vs previous periods
recent_months = monthly_metrics.tail(3)
historical_months = monthly_metrics.iloc[:-3]

def calculate_significance(recent_data, historical_data, metric):
    t_stat, p_val = stats.ttest_ind(
        recent_data[f'{metric}_mean'],
        historical_data[f'{metric}_mean']
    )
    effect_size = (recent_data[f'{metric}_mean'].mean() - historical_data[f'{metric}_mean'].mean()) / historical_data[f'{metric}_mean'].mean()
    return {
        'metric': metric,
        'recent_mean': recent_data[f'{metric}_mean'].mean(),
        'historical_mean': historical_data[f'{metric}_mean'].mean(),
        'change': effect_size,
        'p_value': p_val,
        'significant': p_val < 0.05
    }

metrics = ['open_rate', 'ctr']
significance_results = pd.DataFrame([calculate_significance(recent_months, historical_months, metric) 
                                   for metric in metrics])

print("\nRecent Performance Changes (Last 3 Months vs Historical):")
print(significance_results.round(4))

# Print detailed month-by-month changes
print("\nMonth-over-Month Changes (Last 3 Months):")
print(monthly_changes.tail(3).round(2))

# Calculate stability metrics
stability_metrics = df.groupby('month_year').agg({
    'open_rate': lambda x: np.std(x) / np.mean(x),  # Coefficient of variation
    'ctr': lambda x: np.std(x) / np.mean(x),
}).round(4)

print("\nPerformance Stability (Coefficient of Variation - lower is more stable):")
print("Recent months:")
print(stability_metrics.tail(3).round(4))
print("\nHistorical average:")
print(stability_metrics.iloc[:-3].mean().round(4))

# Identify significant changes
significant_changes = []
for metric in metrics:
    if significance_results[significance_results['metric'] == metric]['significant'].iloc[0]:
        direction = 'decrease' if significance_results[significance_results['metric'] == metric]['change'].iloc[0] < 0 else 'increase'
        change_pct = abs(significance_results[significance_results['metric'] == metric]['change'].iloc[0] * 100)
        significant_changes.append(f"{metric}: {direction} of {change_pct:.1f}%")

print("\nKey Findings:")
if significant_changes:
    print("Statistically Significant Changes:")
    for change in significant_changes:
        print(f"- {change}")
else:
    print("No statistically significant changes detected.")

# Add trend analysis
recent_trend = monthly_metrics.tail(3)['ctr_mean'].diff().mean()
trend_direction = "improving" if recent_trend > 0 else "declining"
trend_strength = abs(recent_trend)

print(f"\nRecent Trend Analysis:")
print(f"- Direction: {trend_direction}")
print(f"- Average monthly change: {trend_strength:.2f}%")

# Performance variability analysis
recent_variability = stability_metrics.tail(3).mean()
historical_variability = stability_metrics.iloc[:-3].mean()
variability_change = ((recent_variability - historical_variability) / historical_variability * 100).round(2)

print("\nPerformance Variability Analysis:")
print(f"- Recent vs Historical Variability Change:")
for metric in variability_change.index:
    print(f"  {metric}: {variability_change[metric]:+.1f}% {'higher' if variability_change[metric] > 0 else 'lower'} variability")

# Add Performance Decline Analysis Visualizations
print("\n=== Performance Decline Analysis ===")

# 1. Recent Decline Visualization
if PLOTLY_AVAILABLE:
    # Create subplot with 3 metrics
    fig_decline = make_subplots(rows=3, cols=1,
                               subplot_titles=('Open Rate Trend', 'CTR Trend', 'CTR Trend'),
                               vertical_spacing=0.12,
                               row_heights=[0.33, 0.33, 0.33])

    metrics_to_plot = ['open_rate', 'ctr']
    colors = ['#3498DB', '#2ECC71', '#F1C40F']
    
    for idx, (metric, color) in enumerate(zip(metrics_to_plot, colors), 1):
        monthly_data = df.groupby('month_year').agg({metric: ['mean', 'std']}).reset_index()
        monthly_data.columns = ['month_year', 'mean', 'std']
        
        # Convert Period to datetime for plotting
        monthly_data['month_year'] = monthly_data['month_year'].astype(str).apply(lambda x: pd.to_datetime(x + '-01'))
        
        # Add trace for mean
        fig_decline.add_trace(
            go.Scatter(
                x=monthly_data['month_year'],
                y=monthly_data['mean'],
                name=f'{metric} Mean',
                line=dict(color=color, width=2),
                showlegend=(idx == 1)
            ),
            row=idx, col=1
        )
        
        # Add confidence interval
        fig_decline.add_trace(
            go.Scatter(
                x=monthly_data['month_year'].tolist() + monthly_data['month_year'].tolist()[::-1],
                y=(monthly_data['mean'] + monthly_data['std']).tolist() + 
                  (monthly_data['mean'] - monthly_data['std']).tolist()[::-1],
                fill='toself',
                fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{metric} Confidence',
                showlegend=(idx == 1)
            ),
            row=idx, col=1
        )
        
        # Add vertical line for recent period start
        fig_decline.add_vline(
            x=monthly_data['month_year'].iloc[-3],
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            row=idx, col=1
        )

    fig_decline.update_layout(
        height=800,
        title_text="Recent Performance Decline Analysis",
        showlegend=True,
        template='plotly_white'
    )
    
    # Update y-axes to percentage format
    for i in range(1, 4):
        fig_decline.update_yaxes(tickformat='.1%', row=i, col=1)
    
    fig_decline.show()

    # 2. Volatility Analysis
    fig_volatility = go.Figure()

    # Calculate rolling volatility (coefficient of variation) for each metric
    window_size = 7  # 7-day rolling window
    for metric, color in zip(metrics_to_plot, colors):
        rolling_vol = df.groupby('date_sent')[metric].std().rolling(window=window_size).std() / \
                     df.groupby('date_sent')[metric].mean().rolling(window=window_size).mean()
        
        fig_volatility.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                name=f'{metric} Volatility',
                line=dict(color=color, width=2)
            )
        )

    fig_volatility.update_layout(
        title='Metric Volatility Over Time (7-Day Rolling Window)',
        xaxis_title='Date',
        yaxis_title='Coefficient of Variation',
        height=500,
        template='plotly_white',
        showlegend=True
    )

    fig_volatility.show()

    # 3. Statistical Significance Visualization
    fig_stats = go.Figure()

    # Prepare data for visualization
    for metric, color in zip(metrics_to_plot, colors):
        recent_mean = significance_results[significance_results['metric'] == metric]['recent_mean'].iloc[0]
        historical_mean = significance_results[significance_results['metric'] == metric]['historical_mean'].iloc[0]
        p_value = significance_results[significance_results['metric'] == metric]['p_value'].iloc[0]
        
        # Add bars for recent and historical means
        fig_stats.add_trace(
            go.Bar(
                name=f'{metric} Historical',
                x=[metric],
                y=[historical_mean],
                marker_color=color.replace('rgb', 'rgba').replace(')', ', 0.3)'),
                width=0.3,
                offset=-0.2
            )
        )
        
        fig_stats.add_trace(
            go.Bar(
                name=f'{metric} Recent',
                x=[metric],
                y=[recent_mean],
                marker_color=color,
                width=0.3,
                offset=0.2
            )
        )
        
        # Add p-value annotation
        fig_stats.add_annotation(
            x=metric,
            y=max(recent_mean, historical_mean) * 1.1,
            text=f'p={p_value:.4f}{"*" if p_value < 0.05 else ""}',
            showarrow=False,
            font=dict(size=12)
        )

    fig_stats.update_layout(
        title='Statistical Comparison: Recent vs Historical Performance',
        yaxis_title='Rate',
        yaxis_tickformat='.1%',
        height=500,
        template='plotly_white',
        showlegend=True,
        barmode='overlay'
    )

    fig_stats.show()

else:
    # Matplotlib fallback for decline analysis
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    metrics_to_plot = ['open_rate', 'ctr']
    colors = ['#3498DB', '#2ECC71', '#F1C40F']
    
    for ax, (metric, color) in zip(axes, zip(metrics_to_plot, colors)):
        monthly_data = df.groupby('month_year').agg({metric: ['mean', 'std']}).reset_index()
        monthly_data.columns = ['month_year', 'mean', 'std']
        
        # Convert Period to datetime for plotting
        monthly_data['month_year'] = monthly_data['month_year'].astype(str).apply(lambda x: pd.to_datetime(x + '-01'))
        
        # Plot mean line
        ax.plot(monthly_data['month_year'], monthly_data['mean'], color=color, label=metric)
        
        # Add confidence interval
        ax.fill_between(monthly_data['month_year'],
                       monthly_data['mean'] - monthly_data['std'],
                       monthly_data['mean'] + monthly_data['std'],
                       color=color, alpha=0.2)
        
        # Add vertical line for recent period
        ax.axvline(monthly_data['month_year'].iloc[-3], color='red', linestyle='--', alpha=0.5)
        
        ax.set_title(f'{metric} Trend')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.grid(True, alpha=0.3)
    
plt.tight_layout()
plt.show()

# Volatility Analysis
plt.figure(figsize=(15, 6))
window_size = 7

for metric, color in zip(metrics_to_plot, colors):
    rolling_vol = df.groupby('date_sent')[metric].std().rolling(window=window_size).std() / \
                 df.groupby('date_sent')[metric].mean().rolling(window=window_size).mean()
    plt.plot(rolling_vol.index, rolling_vol, color=color, label=f'{metric} Volatility')

plt.title('Metric Volatility Over Time (7-Day Rolling Window)')
plt.xlabel('Date')
plt.ylabel('Coefficient of Variation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Add Year-over-Year Seasonality Analysis
print("\n=== Year-over-Year Seasonality Analysis ===")

# Add year and month columns for seasonality analysis
df['year'] = df['date_sent'].dt.year
df['month'] = df['date_sent'].dt.month

# Calculate monthly averages by year
monthly_by_year = df.groupby(['year', 'month']).agg({
    'open_rate': 'mean',
    'ctr': 'mean',
    'n_sent': 'sum'
}).round(4)

# Reset index for easier plotting
monthly_by_year = monthly_by_year.reset_index()

if PLOTLY_AVAILABLE:
    # Create YoY comparison visualization
    fig_yoy = make_subplots(rows=3, cols=1,
                           subplot_titles=('Open Rate YoY', 'CTR YoY', 'CTR YoY'),
                           vertical_spacing=0.12)
    
    metrics = ['open_rate', 'ctr']
    colors = {2022: '#3498DB', 2023: '#2ECC71', 2024: '#F1C40F'}
    
    for idx, metric in enumerate(metrics, 1):
        for year in monthly_by_year['year'].unique():
            year_data = monthly_by_year[monthly_by_year['year'] == year]
            
            fig_yoy.add_trace(
                go.Scatter(
                    x=year_data['month'],
                    y=year_data[metric],
                    name=f'{year} {metric}',
                    line=dict(color=colors[year]),
                    hovertemplate=f'Month: %{{x}}<br>{metric}: %{{y:.2%}}<extra>{year}</extra>'
                ),
                row=idx, col=1
            )
    
    # Update layout
    fig_yoy.update_layout(
        height=900,
        title_text='Year-over-Year Performance Comparison',
        showlegend=True,
        template='plotly_white'
    )
    
    # Update y-axes to percentage format
    for i in range(1, 4):
        fig_yoy.update_yaxes(tickformat='.1%', row=i, col=1)
        fig_yoy.update_xaxes(dtick=1, row=i, col=1)
    
    fig_yoy.show()

# Print YoY analysis for December
print("\nDecember Performance Year-over-Year:")
december_data = monthly_by_year[monthly_by_year['month'] == 12].copy()
if not december_data.empty:
    for metric in ['open_rate', 'ctr']:
        print(f"\n{metric.upper()} Comparison:")
        for year in december_data['year'].unique():
            value = december_data[december_data['year'] == year][metric].iloc[0]
            print(f"{year} December: {value:.2%}")
        
        # Calculate YoY change if we have multiple years
        if len(december_data['year'].unique()) > 1:
            years = sorted(december_data['year'].unique())
            for i in range(1, len(years)):
                prev_year = december_data[december_data['year'] == years[i-1]][metric].iloc[0]
                curr_year = december_data[december_data['year'] == years[i]][metric].iloc[0]
                yoy_change = (curr_year - prev_year) / prev_year * 100
                print(f"{years[i-1]} to {years[i]} Change: {yoy_change:+.1f}%")

# Calculate statistical significance of December changes
if len(december_data['year'].unique()) > 1:
    print("\nStatistical Significance of December Changes:")
    for metric in ['open_rate', 'ctr']:
        years = sorted(december_data['year'].unique())
        latest_year = years[-1]
        prev_year = years[-2]
        
        # Get all December data for both years
        latest_dec = df[(df['year'] == latest_year) & (df['month'] == 12)][metric]
        prev_dec = df[(df['year'] == prev_year) & (df['month'] == 12)][metric]
        
        if len(latest_dec) > 0 and len(prev_dec) > 0:
            t_stat, p_val = stats.ttest_ind(latest_dec, prev_dec)
            print(f"\n{metric.upper()}:")
            print(f"t-statistic: {t_stat:.3f}")
            print(f"p-value: {p_val:.4f}")
            print(f"Significant change: {'Yes' if p_val < 0.05 else 'No'}")

# Add seasonality detection using time series decomposition
print("\nSeasonality Analysis:")
for metric in ['open_rate', 'ctr']:
    # Create time series by resampling to daily frequency
    daily_metric = df.groupby('date_sent', observed=True)[metric].mean().resample('ME').mean()
    
    # Fill missing values with forward fill, then backward fill
    daily_metric = daily_metric.ffill().bfill()
    
    # Calculate rolling statistics
    rolling_mean = daily_metric.rolling(window=30, center=True).mean()
    rolling_std = daily_metric.rolling(window=30, center=True).std()
    
    # Calculate coefficient of variation to measure seasonality strength
    cv = rolling_std / rolling_mean
    
    print(f"\n{metric.upper()}:")
    print(f"Average monthly variation: {cv.mean():.3f}")
    print(f"Seasonality strength: {'Strong' if cv.mean() > 0.15 else 'Moderate' if cv.mean() > 0.05 else 'Weak'}")

# Performance Analysis Functions
def analyze_performance_decline(df, metrics=['open_rate', 'ctr']):
    plt.figure(figsize=(15, 10))
    
    # Change from RGB strings to hex codes
    colors = ['#3498DB', '#2ECC71', '#F1C40F']  # Hex equivalents
    
    for i, metric in enumerate(metrics):
        # Calculate rolling volatility
        rolling_vol = df[metric].rolling(window=30).std()
        
        # Plot volatility with hex color
        plt.plot(df['date_sent'], rolling_vol, 
                color=colors[i], 
                label=f'{metric} Volatility')
    
    plt.title('Performance Volatility Over Time')
    plt.xlabel('Date')
    plt.ylabel('30-Day Rolling Standard Deviation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_seasonality(df, metrics=['open_rate', 'ctr']):
    plt.figure(figsize=(15, 10))
    
    colors = ['#3498DB', '#2ECC71', '#F1C40F']  # Hex equivalents
    
    for i, metric in enumerate(metrics):
        daily_metric = df.groupby('date_sent', observed=True)[metric].mean().resample('ME').mean()
        plt.plot(daily_metric.index, daily_metric, 
                color=colors[i], 
                label=f'{metric} Daily Average',
                marker='o')
    
    plt.title('Daily Performance Patterns')
    plt.xlabel('Date')
    plt.ylabel('Average Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_performance_trends(df, metrics=['open_rate', 'ctr']):
    plt.figure(figsize=(15, 10))
    
    colors = ['#3498DB', '#2ECC71', '#F1C40F']  # Hex equivalents
    
    for i, metric in enumerate(metrics):
        # Calculate rolling mean
        rolling_mean = df[metric].rolling(window=30).mean()
        
        # Plot trend
        plt.plot(df['date_sent'], rolling_mean, 
                color=colors[i], 
                label=f'{metric} Trend')
    
    plt.title('Performance Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('30-Day Rolling Average')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Call the functions to analyze performance
analyze_performance_decline(df)
analyze_seasonality(df)
analyze_performance_trends(df)

# Analyze performance by campaign size
def analyze_campaign_size_impact(df):
    plt.figure(figsize=(15, 10))
    
    colors = ['#3498DB', '#2ECC71', '#F1C40F']  # Hex equivalents
    
    # Create scatter plots
    plt.scatter(df['n_sent'], df['open_rate'], 
               color=colors[0], alpha=0.5, label='Open Rate')
    plt.scatter(df['n_sent'], df['ctr'], 
               color=colors[1], alpha=0.5, label='CTR')
    
    # Add trend lines
    for i, metric in enumerate(['open_rate', 'ctr']):
        z = np.polyfit(df['n_sent'], df[metric], 1)
        p = np.poly1d(z)
        plt.plot(df['n_sent'], p(df['n_sent']), 
                color=colors[i], linestyle='--', alpha=0.8)
    
    plt.title('Campaign Size Impact on Performance Metrics')
    plt.xlabel('Number of Recipients')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Call the function to analyze campaign size impact
analyze_campaign_size_impact(df)

# Add time between campaigns analysis
df = df.sort_values('date_sent')
df['days_since_last'] = df['date_sent'].diff().dt.days

# Analyze frequency impact
frequency_impact = df.groupby(pd.cut(df['days_since_last'], 
                                    bins=[0, 1, 3, 7, 14, np.inf],
                                    labels=['Daily', '2-3 Days', 'Weekly', 'Biweekly', '>2 Weeks']
                                   )).agg({
    'open_rate': 'mean',
    'ctr': 'mean',
    'n_sent': 'sum'
}).reset_index()

print("\nCampaign Frequency Impact Analysis:")
print(frequency_impact)

# Add these lines after calculating the engagement_score
df['day_of_week'] = df['date_sent'].dt.day_name()
df['month'] = df['date_sent'].dt.month
df['year'] = df['date_sent'].dt.year 

# Add K-means clustering for campaign performance
from sklearn.cluster import KMeans

# Prepare features
X = df[['n_sent', 'open_rate', 'ctr']]
X_scaled = StandardScaler().fit_transform(X)

# Cluster campaigns
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_profile = df.groupby('cluster').agg({
    'n_sent': 'mean',
    'open_rate': 'mean',
    'ctr': 'mean',
    'date_sent': 'count'
}).rename(columns={'date_sent': 'count'})

print("\nCampaign Performance Clusters:")
print(cluster_profile)

# Sort clusters before labeling
cluster_profile = cluster_profile.sort_values('ctr', ascending=False)
assert cluster_profile.index[0] == 1, "Cluster ordering mismatch - check KMeans initialization"
cluster_labels = {
    1: 'Optimal Performers', 
    0: 'High Volume-Low Engagement',
    2: 'Average Campaigns'
}
df['cluster_label'] = df['cluster'].map(cluster_labels)

print("\nCluster Characteristics:")
print(df.groupby('cluster_label').agg({
    'n_sent': ['mean', 'std'],
    'open_rate': 'mean',
    'ctr': 'mean'
}))

# Add engagement decay analysis
def calculate_engagement_decay(df):
    decay_rates = []
    for days in [1, 3, 7, 14]:
        decay_df = df[df['days_since_last'] <= days]
        decay_rate = decay_df['ctr'].mean() / df['ctr'].mean() - 1
        decay_rates.append(decay_rate)
    
    # Add visualization
    plt.figure(figsize=(10,6))
    plt.bar([1,3,7,14], decay_rates, color='#3498DB')
    plt.title('Engagement Decay Rates by Days Since Last Campaign')
    plt.xlabel('Days Since Last Campaign')
    plt.ylabel('CTR Change vs Average')
    plt.xticks([1,3,7,14])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return pd.Series(decay_rates, index=[1, 3, 7, 14])

print("\nEngagement Decay Rates:")
print(calculate_engagement_decay(df))

# Add random forest feature importance analysis
from sklearn.ensemble import RandomForestRegressor

# Prepare data
X = df[['n_sent', 'days_since_last', 'month', 'day_of_week']]
X = pd.get_dummies(X, columns=['month', 'day_of_week'])
y = df['ctr']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nCTR Predictive Factors:")
print(importance.head(10))

# Add to the recommendations section
print("\nCritical Thresholds:")
print(f"- Open Rate Alert Threshold: {df['open_rate'].quantile(0.25):.1%}")
print(f"- CTR Danger Zone: <{df['ctr'].quantile(0.25):.1%}")

# Add after date component extraction
df['open_rate_benchmark'] = df.groupby('month')['open_rate'].transform('mean')
df['ctr_benchmark'] = df.groupby('month')['ctr'].transform('mean')

# Add to visualization section
plt.figure(figsize=(12,6))
sns.histplot(df['ctr'], bins=30, kde=True, color='#2ECC71')
plt.axvline(df['ctr'].quantile(0.25), color='red', linestyle='--', label='25th Percentile')
plt.axvline(df['ctr'].median(), color='orange', linestyle='--', label='Median')
plt.axvline(df['ctr'].quantile(0.75), color='green', linestyle='--', label='75th Percentile')
plt.title('CTR Distribution with Performance Thresholds')
plt.xlabel('Click-Through Rate')
plt.ylabel('Count')
plt.legend()
plt.show()

def analyze_send_volume_evolution(df):
    # Create time series data
    monthly_sends = df.resample('ME', on='date_sent')['n_sent'].sum().reset_index()
    
    # Create figure
    fig = go.Figure()
    
    # Add send volume trace
    fig.add_trace(go.Scatter(
        x=monthly_sends['date_sent'],
        y=monthly_sends['n_sent'],
        mode='lines+markers',
        name='Total Sent',
        line=dict(color='#3498DB', width=2),
        marker=dict(size=6),
        hovertemplate='%{x|%b %Y}<br>Sent: %{y:,}'
    ))
    
    # Add 3-month moving average
    monthly_sends['ma3'] = monthly_sends['n_sent'].rolling(3).mean()
    fig.add_trace(go.Scatter(
        x=monthly_sends['date_sent'],
        y=monthly_sends['ma3'],
        mode='lines',
        name='3-Month MA',
        line=dict(color='#2ECC71', width=2, dash='dot'),
        hovertemplate='%{x|%b %Y}<br>MA3: %{y:,.0f}'
    ))
    
    # Update layout
    fig.update_layout(
        title='Email Send Volume Evolution',
        xaxis_title='Date',
        yaxis_title='Number of Emails Sent',
        template='plotly_white',
        height=500,
        width=1000,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Add growth rate annotations
    first_val = monthly_sends['n_sent'].iloc[0]
    last_val = monthly_sends['n_sent'].iloc[-1]
    growth_rate = (last_val - first_val)/first_val * 100
    
    fig.add_annotation(
        x=monthly_sends['date_sent'].iloc[-1],
        y=last_val,
        text=f"Total Growth: {growth_rate:+.1f}%",
        showarrow=True,
        arrowhead=1,
        ax=-50,
        ay=-40
    )
    
    fig.show()
    
    return monthly_sends

# Call the function after other analyses
print("\n=== Send Volume Analysis ===")
send_evolution = analyze_send_volume_evolution(df)

def analyze_volume_impact(df):
    # Create quarterly aggregates
    df['quarter'] = df['date_sent'].dt.to_period('Q')
    quarterly = df.groupby('quarter').agg({
        'n_sent': 'sum',
        'open_rate': 'mean',
        'ctr': 'mean'
    }).reset_index()
    
    # Create subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add send volume bar chart
    fig.add_trace(go.Bar(
        x=quarterly['quarter'].astype(str),
        y=quarterly['n_sent'],
        name='Total Sent',
        marker_color='#3498DB',
        opacity=0.6
    ), secondary_y=False)
    
    # Add open rate line
    fig.add_trace(go.Scatter(
        x=quarterly['quarter'].astype(str),
        y=quarterly['open_rate'],
        name='Open Rate',
        line=dict(color='#E74C3C', width=2),
        mode='lines+markers'
    ), secondary_y=True)
    
    # Add CTR line
    fig.add_trace(go.Scatter(
        x=quarterly['quarter'].astype(str),
        y=quarterly['ctr'],
        name='CTR',
        line=dict(color='#2ECC71', width=2),
        mode='lines+markers'
    ), secondary_y=True)
    
    # Update layout
    fig.update_layout(
        title='Send Volume vs Engagement Metrics',
        xaxis_title='Quarter',
        template='plotly_white',
        height=500,
        width=1000,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Set y-axes labels
    fig.update_yaxes(title_text="Total Sent", secondary_y=False)
    fig.update_yaxes(title_text="Engagement Rate", secondary_y=True, tickformat='.1%')
    
    fig.show()
    
    # Calculate correlations
    corr_sent_open = quarterly['n_sent'].corr(quarterly['open_rate'])
    corr_sent_ctr = quarterly['n_sent'].corr(quarterly['ctr'])
    
    print(f"\nVolume vs Engagement Correlations:")
    print(f"Sent vs Open Rate: {corr_sent_open:.2f}")
    print(f"Sent vs CTR: {corr_sent_ctr:.2f}")

# Call the function after volume analysis
analyze_volume_impact(df)

# Add final summary statistics
print("\nFinal Summary Statistics:")
print(f"- Total Campaigns Analyzed: {len(df):,}")
print(f"- Date Range: {df['date_sent'].min().strftime('%Y-%m-%d')} to {df['date_sent'].max().strftime('%Y-%m-%d')}")
print(f"- Average Daily Sends: {df['n_sent'].mean():,.0f}")
print(f"- Peak Day Sends: {df['n_sent'].max():,.0f}")

def create_enhanced_daily_ctr_analysis():
    if PLOTLY_AVAILABLE:
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add smoothed CTR line
        dates_smooth, ctr_mean_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_mean'])
        _, ctr_upper_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_ci_upper'])
        _, ctr_lower_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_ci_lower'])
        _, trend_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_trend'], smoothing_factor=1.2)
        
        # Add smoothed trend for email sends
        _, sends_trend_smooth = smooth_data(daily_stats['date'], daily_stats['total_sent'], smoothing_factor=1.5)

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

        # Calculate overall growth rates for annotation
        first_ctr = trend_smooth[0]
        last_ctr = trend_smooth[-1]
        ctr_growth = (last_ctr - first_ctr) / first_ctr * 100

        first_sends = sends_trend_smooth[0]
        last_sends = sends_trend_smooth[-1]
        sends_growth = (last_sends - first_sends) / first_sends * 100

        # Add annotations for growth rates
        fig.add_annotation(
            x=dates_smooth[-1],
            y=last_sends,
            text=f"+{sends_growth:.1f}% Volume Growth",
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=-40,
            font=dict(color='rgba(52, 152, 219, 1)'),
            secondary_y=False
        )

        fig.add_annotation(
            x=dates_smooth[-1],
            y=last_ctr,
            text=f"+{ctr_growth:.1f}% CTR Growth",
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=40,
            font=dict(color='rgba(255, 65, 54, 1)'),
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
            margin=dict(l=60, r=60, t=100, b=60)
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

        # Add subtle background color
        fig.update_layout(
            plot_bgcolor='rgba(250,250,250,0.9)',
            paper_bgcolor='white'
        )

        fig.show()
    else:
        # Matplotlib fallback
        fig, ax1 = plt.subplots(figsize=(15, 8))
        
        # Plot email volume bars with reduced opacity
        ax1.bar(daily_stats['date'], daily_stats['total_sent'], 
                alpha=0.2, color='#3498DB', label='Daily Emails Sent')
        
        # Add trend line for email sends
        dates_smooth, sends_trend_smooth = smooth_data(daily_stats['date'], daily_stats['total_sent'], smoothing_factor=1.5)
        ax1.plot(dates_smooth, sends_trend_smooth,
                color='#3498DB', linewidth=2, label='Email Volume Trend')
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Emails Sent', color='#3498DB')
        ax1.tick_params(axis='y', labelcolor='#3498DB')
        
        # Create second y-axis for CTR
        ax2 = ax1.twinx()
        
        # Plot CTR data
        dates_smooth, ctr_mean_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_mean'])
        _, trend_smooth = smooth_data(daily_stats['date'], daily_stats['ctr_trend'], smoothing_factor=1.2)
        
        # Plot actual CTR points
        ax2.scatter(daily_stats['date'], daily_stats['ctr_mean'], 
                   color='#E74C3C', alpha=0.3, s=30, label='Daily CTR')
        
        # Plot trend line for CTR
        ax2.plot(dates_smooth, trend_smooth, 
                 color='#E74C3C', linewidth=2, label='CTR Trend')
        
        # Calculate and add growth rates
        first_ctr = trend_smooth[0]
        last_ctr = trend_smooth[-1]
        ctr_growth = (last_ctr - first_ctr) / first_ctr * 100

        first_sends = sends_trend_smooth[0]
        last_sends = sends_trend_smooth[-1]
        sends_growth = (last_sends - first_sends) / first_sends * 100
        
        # Add growth annotations
        ax1.annotate(f'+{sends_growth:.1f}% Volume Growth',
                    xy=(dates_smooth[-1], sends_trend_smooth[-1]),
                    xytext=(10, 10), textcoords='offset points',
                    color='#3498DB')
        
        ax2.annotate(f'+{ctr_growth:.1f}% CTR Growth',
                    xy=(dates_smooth[-1], trend_smooth[-1]),
                    xytext=(10, -10), textcoords='offset points',
                    color='#E74C3C')
        
        ax2.set_ylabel('Click-to-Open Rate (CTR)', color='#E74C3C')
        ax2.tick_params(axis='y', labelcolor='#E74C3C')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title('Daily Click-to-Open Rate and Send Volume Analysis\nShowing parallel growth in both metrics')
        plt.grid(True, alpha=0.1)
        plt.tight_layout()
        plt.show()

# Call the new function to create the enhanced visualization
create_enhanced_daily_ctr_analysis() 