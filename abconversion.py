import math
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data from CSV file and transpose it to get the correct format
df = pd.read_csv('Felipe Chaves_TakeHome Exercise - A_BTest.csv', index_col=0).transpose()

# Clean up the column names
df.columns = df.columns.str.strip()

# Calculate all rates
df["Open_Rate"] = df["Number Opened"].astype(float) / df["Number Sent"].astype(float)
df["Click_Rate"] = df["Links Clicked"].astype(float) / df["Number Sent"].astype(float)
df["Conversion_Rate_Sent"] = df["Classes Purchased"].astype(float) / df["Number Sent"].astype(float)
df["Click_to_Open_Rate"] = df["Links Clicked"].astype(float) / df["Number Opened"].astype(float)
df["Purchase_to_Click_Rate"] = df["Classes Purchased"].astype(float) / df["Links Clicked"].astype(float)

# Calculate standard errors and CIs for all rates
for rate_col in ["Open_Rate", "Click_Rate", "Conversion_Rate_Sent", "Click_to_Open_Rate", "Purchase_to_Click_Rate"]:
    df[f"{rate_col}_SE"] = ((df[rate_col] * (1 - df[rate_col])) / df["Number Sent"].astype(float)).apply(math.sqrt)
    df[f"{rate_col}_CI"] = 1.96 * df[f"{rate_col}_SE"]

# Original p-value calculation
p1 = df.iloc[0]["Conversion_Rate_Sent"]
n1 = float(df.iloc[0]["Number Sent"])
p2 = df.iloc[1]["Conversion_Rate_Sent"]
n2 = float(df.iloc[1]["Number Sent"])
pooled_rate = (float(df.iloc[0]["Classes Purchased"]) + float(df.iloc[1]["Classes Purchased"])) / (n1 + n2)
se_diff = ((pooled_rate * (1 - pooled_rate)) * (1/n1 + 1/n2)) ** 0.5
z_score = (p1 - p2) / se_diff
p_value = stats.norm.sf(abs(z_score)) * 2

# Print intermediate calculations
print("\n----- CALCULATION VERIFICATION -----")
print(f"A. Urgency conversion rate (p1): {p1:.4f}")
print(f"B. Quality conversion rate (p2): {p2:.4f}")
print(f"Pooled rate: {pooled_rate:.4f}")
print(f"Standard error of difference: {se_diff:.6f}")
print(f"Z-score: {z_score:.4f}")
print(f"P-value: {p_value:.4f}")

# 1. Full Funnel Visualization
funnel_fig = go.Figure()

# Add traces for each version
for idx, version in enumerate(df.index):
    values = [
        df.iloc[idx]["Number Sent"],
        df.iloc[idx]["Number Opened"],
        df.iloc[idx]["Links Clicked"],
        df.iloc[idx]["Classes Purchased"]
    ]
    funnel_fig.add_trace(go.Funnel(
        name=version,
        y=["Sent", "Opened", "Clicked", "Purchased"],
        x=values,
        textinfo="value+percent initial"
    ))

funnel_fig.update_layout(title="Email Campaign Funnel Comparison")
funnel_fig.show()

# 2. Conversion Rates Comparison with Error Bars
metrics_comparison = go.Figure()

# Prepare data for plotting
metrics = ["Open_Rate", "Click_Rate", "Conversion_Rate_Sent"]
metric_labels = ["Open Rate", "Click Rate", "Conversion Rate"]

for version in df.index:
    y_values = [df.loc[version, metric] for metric in metrics]
    error_y = [df.loc[version, f"{metric}_CI"] for metric in metrics]
    
    metrics_comparison.add_trace(go.Bar(
        name=version,
        x=metric_labels,
        y=y_values,
        error_y=dict(type='data', array=error_y, visible=True)
    ))

metrics_comparison.update_layout(
    title="Conversion Metrics Comparison with 95% Confidence Intervals",
    barmode='group',
    yaxis_title="Rate",
    showlegend=True
)
metrics_comparison.show()

# 3. Relative Performance Improvement
relative_diff = pd.DataFrame({
    'Metric': metric_labels,
    'Relative_Difference': [(df.iloc[0][metric] - df.iloc[1][metric])/df.iloc[1][metric] * 100 for metric in metrics]
})

relative_fig = px.bar(
    relative_diff,
    x='Metric',
    y='Relative_Difference',
    title="Relative Performance Difference (A vs B)",
    labels={'Relative_Difference': 'Relative Difference (%)'}
)
relative_fig.show()

# Create the final conversion rate comparison plot
plot_df = df.reset_index()
fig = go.Figure()

# Add bars for each version
fig.add_trace(go.Bar(
    name='Conversion Rate',
    x=plot_df['index'],
    y=plot_df['Conversion_Rate_Sent'],
    error_y=dict(
        type='data',
        array=plot_df['Conversion_Rate_Sent_CI'],
        visible=True
    )
))

fig.update_layout(
    title=f'Conversion Rate (per Email Sent) Comparison (p-value = {p_value:.3f} | {"Statistically Significant" if p_value < 0.05 else "Not Statistically Significant"})',
    xaxis_title="Email Version",
    yaxis_title="Conversion Rate per Email Sent",
    showlegend=False
)
fig.show()

# Print full funnel metrics
print("\n----- FULL FUNNEL METRICS -----")
print("\nA. Urgency:")
print(f"Emails Sent: {int(df.iloc[0]['Number Sent']):,}")
print(f"Opened: {int(df.iloc[0]['Number Opened']):,} ({int(df.iloc[0]['Number Opened'])/float(df.iloc[0]['Number Sent']):.1%})")
print(f"Clicked: {int(df.iloc[0]['Links Clicked']):,} ({int(df.iloc[0]['Links Clicked'])/float(df.iloc[0]['Number Sent']):.1%} of sent)")
print(f"Purchased: {int(df.iloc[0]['Classes Purchased']):,} ({int(df.iloc[0]['Classes Purchased'])/float(df.iloc[0]['Number Sent']):.1%} of sent)")

print("\nB. Quality:")
print(f"Emails Sent: {int(df.iloc[1]['Number Sent']):,}")
print(f"Opened: {int(df.iloc[1]['Number Opened']):,} ({int(df.iloc[1]['Number Opened'])/float(df.iloc[1]['Number Sent']):.1%})")
print(f"Clicked: {int(df.iloc[1]['Links Clicked']):,} ({int(df.iloc[1]['Links Clicked'])/float(df.iloc[1]['Number Sent']):.1%} of sent)")
print(f"Purchased: {int(df.iloc[1]['Classes Purchased']):,} ({int(df.iloc[1]['Classes Purchased'])/float(df.iloc[1]['Number Sent']):.1%} of sent)")

# Additional metrics summary
print("\n----- DETAILED METRICS COMPARISON -----")
print("\nRelative Performance (A vs B):")
for metric, diff in zip(metric_labels, relative_diff['Relative_Difference']):
    print(f"{metric}: {'+'if diff > 0 else ''}{diff:.1f}%")

# Textual conclusion
print("\n----- STATISTICAL CONCLUSION -----")
if p_value < 0.05:
    if p1 > p2:
        print("The overall campaign shows that 'A. Urgency' has a significantly higher conversion rate per email sent.")
    else:
        print("The overall campaign shows that 'B. Quality' has a significantly higher conversion rate per email sent.")
else:
    print("The difference in conversion rates per email sent between the two email versions is not statistically significant.")
    print("This suggests that when taking the entire funnel into account (from sent to conversion),")
    print("the observed differences might be due to random chance or small influences from both open and conversion behavior.")
print("----------------------")