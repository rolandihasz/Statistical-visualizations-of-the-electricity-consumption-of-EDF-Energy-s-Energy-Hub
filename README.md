# EDF Energy Statistical Visualizations

A clean, executable Python script utilizing `matplotlib`, `seaborn`, and `scipy` to generate high-quality, comprehensive statistical visualizations.

## Data Source
This script is designed to process exportable CSV data from the EDF Energy website. To obtain your data:
1. Log in and navigate to the **Energy Hub** section.
2. Go to the **At a glance** tab.
3. Scroll to the bottom and click **"Download your data as CSV"**.

## Features
The script generates a comprehensive dashboard featuring four distinct graphs:
1. **Time Series Plot:** Tracks daily consumption against the mean and standard deviation.
2. **Scatter Plot with Regression:** Visualizes the correlation between consumption and cost.
3. **Boxplots:** Displays distribution percentiles, ranges, and outliers.
4. **Histograms & Density Curves:** Overlays consumption and cost distributions.

## Execution & Usage
Ensure you are running this within an active virtual environment (your `requirements.txt` dependencies must be installed).

1. Activate the virtual environment:
   ```bash
   .venv\Scripts\activate
