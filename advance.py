import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "Felipe Chaves_TakeHome Exercise - Campaign_Performance.csv"
df = pd.read_csv(file_path, parse_dates=['date_sent'])

# Compute open rate and CTR
df['open_rate'] = df['n_open'] / df['n_sent']
df['ctr'] = df['n_click'] / df['n_open']

def perform_advanced_analysis(df):
    """
    Perform advanced statistical analysis including hypothesis testing
    and regression modeling to identify key impact variables.
    """
    # 1. Prepare Data
    df['day_of_week'] = df['date_sent'].dt.dayofweek
    df['month'] = df['date_sent'].dt.month
    df['year'] = df['date_sent'].dt.year
    
    # 2. Create Features
    features = ['day_of_week', 'month', 'year', 'n_sent']
    X = df[features].copy()

    # Convert categorical features to dummy variables before scaling
    X = pd.get_dummies(X, columns=['day_of_week', 'month', 'year'], drop_first=True)
    
    # Scale 'n_sent'
    scaler = StandardScaler()
    X['n_sent'] = scaler.fit_transform(df[['n_sent']])  # Replaces n_sent, avoids redundancy
    
    # 3. Prepare Target Variables
    y_open = df['open_rate'].values
    y_click = df['ctr'].values

    # 4. Fit Linear Regression Models
    model_open = LinearRegression().fit(X, y_open)
    model_click = LinearRegression().fit(X, y_click)

    # 5. Hypothesis Tests (Recent vs. Historical Performance)
    recent_cutoff = df['date_sent'].max() - pd.Timedelta(days=30)
    recent_data = df[df['date_sent'] >= recent_cutoff]
    historical_data = df[df['date_sent'] < recent_cutoff]

    # Perform t-tests (with unequal variance assumption)
    open_rate_ttest = stats.ttest_ind(
        recent_data['open_rate'], historical_data['open_rate'], equal_var=False
    )
    ctr_ttest = stats.ttest_ind(
        recent_data['ctr'], historical_data['ctr'], equal_var=False
    )

    # 6. Seasonal Adjustments (Comparing December 2024 to Past Decembers)
    historical_december = df[(df['month'] == 12) & (df['year'] < 2024)]
    recent_december = df[(df['month'] == 12) & (df['year'] == 2024)]
    
    dec_open_rate_ttest = stats.ttest_ind(
        recent_december['open_rate'], historical_december['open_rate'], equal_var=False
    )
    dec_ctr_ttest = stats.ttest_ind(
        recent_december['ctr'], historical_december['ctr'], equal_var=False
    )

    # 7. Summary of Findings
    findings = {
        'open_rate_model': {
            'r_squared': model_open.score(X, y_open),
            'coefficients': dict(zip(X.columns, model_open.coef_))
        },
        'ctr_model': {
            'r_squared': model_click.score(X, y_click),
            'coefficients': dict(zip(X.columns, model_click.coef_))
        },
        'hypothesis_tests': {
            'open_rate_recent_vs_historical': {
                'statistic': open_rate_ttest.statistic,
                'p_value': open_rate_ttest.pvalue,
                'significant': open_rate_ttest.pvalue < 0.05
            },
            'ctr_recent_vs_historical': {
                'statistic': ctr_ttest.statistic,
                'p_value': ctr_ttest.pvalue,
                'significant': ctr_ttest.pvalue < 0.05
            },
            'open_rate_december_comparison': {
                'statistic': dec_open_rate_ttest.statistic,
                'p_value': dec_open_rate_ttest.pvalue,
                'significant': dec_open_rate_ttest.pvalue < 0.05
            },
            'ctr_december_comparison': {
                'statistic': dec_ctr_ttest.statistic,
                'p_value': dec_ctr_ttest.pvalue,
                'significant': dec_ctr_ttest.pvalue < 0.05
            },
        }
    }
    
    # Print Summary
    print("\nAdvanced Statistical Analysis Results")
    print("=" * 50)
    
    print("\n1. Open Rate Model")
    print(f"R-squared: {findings['open_rate_model']['r_squared']:.3f}")
    print("\nTop factors affecting Open Rate (positive/negative impact):")
    coeffs_open_sorted = pd.Series(findings['open_rate_model']['coefficients']).sort_values(ascending=False)
    print(coeffs_open_sorted.head(5))  # Top positive factors
    print("\nTop negative factors:")
    print(coeffs_open_sorted.tail(5))  # Top negative factors
    
    print("\n2. Click-through Rate Model")
    print(f"R-squared: {findings['ctr_model']['r_squared']:.3f}")
    print("\nTop factors affecting CTR (positive/negative impact):")
    coeffs_ctr_sorted = pd.Series(findings['ctr_model']['coefficients']).sort_values(ascending=False)
    print(coeffs_ctr_sorted.head(5))  # Top positive factors
    print("\nTop negative factors:")
    print(coeffs_ctr_sorted.tail(5))  # Top negative factors
    
    print("\n3. Hypothesis Tests")
    print("\nOpen Rate - Recent vs Historical:")
    print(f"t-statistic: {findings['hypothesis_tests']['open_rate_recent_vs_historical']['statistic']:.3f}")
    print(f"p-value: {findings['hypothesis_tests']['open_rate_recent_vs_historical']['p_value']:.3f}")
    print(f"Significant difference: {findings['hypothesis_tests']['open_rate_recent_vs_historical']['significant']}")
    
    print("\nCTR - Recent vs Historical:")
    print(f"t-statistic: {findings['hypothesis_tests']['ctr_recent_vs_historical']['statistic']:.3f}")
    print(f"p-value: {findings['hypothesis_tests']['ctr_recent_vs_historical']['p_value']:.3f}")
    print(f"Significant difference: {findings['hypothesis_tests']['ctr_recent_vs_historical']['significant']}")
    
    print("\nDecember Open Rate - 2024 vs Past Years:")
    print(f"t-statistic: {findings['hypothesis_tests']['open_rate_december_comparison']['statistic']:.3f}")
    print(f"p-value: {findings['hypothesis_tests']['open_rate_december_comparison']['p_value']:.3f}")
    print(f"Significant difference: {findings['hypothesis_tests']['open_rate_december_comparison']['significant']}")
    
    print("\nDecember CTR - 2024 vs Past Years:")
    print(f"t-statistic: {findings['hypothesis_tests']['ctr_december_comparison']['statistic']:.3f}")
    print(f"p-value: {findings['hypothesis_tests']['ctr_december_comparison']['p_value']:.3f}")
    print(f"Significant difference: {findings['hypothesis_tests']['ctr_december_comparison']['significant']}")
    
    return findings

# Run the Analysis
findings = perform_advanced_analysis(df)
