################################################################################################################
# A clean, executable Python script utilizing matplotlib, seaborn, and scipy 
# to generate high-quality, comprehensive statistical visualizations.
#
# This script is designed to process the exportable CSV data from the EDF Energy website.
# To obtain your data, navigate to the "Energy Hub" section, scroll to the bottom of 
# the "At a glance" tab, and click "Download your data as CSV".
#
# The script will produce the following four graphs:
# 1. Time Series Plot
# 2. Scatter Plot with Regression
# 3. Boxplots
# 4. Histograms & Density Curves
#
# Execution & Usage:
# Ensure you are running this within an active virtual environment.
# 1. Activate the environment: .venv\Scripts\activate
# 2. Execute the script: python analyze_electricity.py
#
# By default, the script saves a .png image file and opens an interactive window. 
# This interface is adjustable, scrollable, zoomable, and pannable, allowing you 
# to explore the data in detail. If you prefer to run the script silently and 
# only generate the .png file, simply comment out `plt.show()` at the end of the script.
#
# Roland Ihasz - 2026-09-30
################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Load Data
file_path = '09_2026_Timestamp_Electricity_consumption.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Could not find the file at {file_path}")
    exit(1)

# 2. Preprocessing
# Parse Timestamp correctly (DD/MM/YYYY)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y')
df.set_index('Timestamp', inplace=True)

# Rename columns for easier access
df.rename(columns={'Electricity consumption (kWh)': 'Consumption_kWh', 'Electricity cost (£)': 'Cost_GBP'}, inplace=True)

# 3. Statistical Calculations
# Mean, Standard Deviation, Range, Percentiles
stats_df = df.describe(percentiles=[.25, .5, .75, .90, .95, .99]).T
stats_df['range'] = stats_df['max'] - stats_df['min']

print("="*50)
print(" DESCRIPTIVE STATISTICS (Mean, Std, Range, Percentiles)")
print("="*50)
print(stats_df[['mean', 'std', 'min', 'max', 'range', '25%', '50%', '75%', '95%']].round(3))
print("\n")

# Correlation
correlation = df['Consumption_kWh'].corr(df['Cost_GBP'])
print("="*50)
print(" CORRELATION")
print("="*50)
print(f"Pearson Correlation between Consumption and Cost: {correlation:.4f}\n")

# Linear Regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Consumption_kWh'], df['Cost_GBP'])
print("="*50)
print(" LINEAR REGRESSION")
print("="*50)
print(f"Equation: Cost(£) = {slope:.4f} * Consumption(kWh) + {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"p-value: {p_value:.4e}\n")

# 4. Visualizations
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Electricity Consumption & Cost Statistical Analysis', fontsize=18, fontweight='bold', y=0.98)

# Plot 1: Time Series with Mean and Std Dev
ax1 = plt.subplot(2, 2, 1)
ax1.plot(df.index, df['Consumption_kWh'], marker='o', label='Daily Consumption (kWh)', color='royalblue')
ax1.axhline(df['Consumption_kWh'].mean(), color='crimson', linestyle='--', label=f"Mean: {df['Consumption_kWh'].mean():.2f}")
ax1.fill_between(df.index, 
                df['Consumption_kWh'].mean() - df['Consumption_kWh'].std(), 
                df['Consumption_kWh'].mean() + df['Consumption_kWh'].std(), 
                color='crimson', alpha=0.15, label='±1 Std Dev')
ax1.set_title('Time Series: Consumption vs Mean & Deviation', fontsize=14)
ax1.set_ylabel('Consumption (kWh)')
ax1.legend()
plt.xticks(rotation=45)

# Plot 2: Scatter Plot with Regression Line
ax2 = plt.subplot(2, 2, 2)
sns.regplot(x='Consumption_kWh', y='Cost_GBP', data=df, ax=ax2, color='indigo', 
            scatter_kws={'alpha':0.6}, line_kws={'color':'crimson', 'label':f'Regression y={slope:.2f}x+{intercept:.2f}'})
ax2.set_title(f'Regression & Correlation (r = {correlation:.3f})', fontsize=14)
ax2.set_xlabel('Consumption (kWh)')
ax2.set_ylabel('Cost (£)')
ax2.legend()

# Plot 3: Boxplots (Percentiles, Range, Outliers)
ax3 = plt.subplot(2, 2, 3)
sns.boxplot(data=df[['Consumption_kWh', 'Cost_GBP']], ax=ax3, palette='Set2', width=0.4)
ax3.set_title('Distribution: Percentiles & Range (Boxplots)', fontsize=14)
ax3.set_ylabel('Value')

# Plot 4: Histograms & Density (Distribution)
ax4 = plt.subplot(2, 2, 4)
sns.histplot(df['Consumption_kWh'], kde=True, color='teal', ax=ax4, label='Consumption (kWh)')

# Use a secondary axis for Cost to overlay distributions appropriately since scales differ slightly
ax4_2 = ax4.twiny()
sns.kdeplot(df['Cost_GBP'], color='darkorange', ax=ax4_2, label='Cost Density (£)', fill=True, alpha=0.2)
ax4.set_title('Histograms & Density Estimates', fontsize=14)
ax4.set_xlabel('Consumption (kWh)')
ax4_2.set_xlabel('Cost (£)')

# Add a combined legend for plot 4
lines, labels = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_2.get_legend_handles_labels()
ax4.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_image = 'statistical_analysis_visuals.png'
plt.savefig(output_image, dpi=300)
print(f"Visualizations successfully saved to '{output_image}'.")

plt.show()