# ------------------------------------------------------------
# 1. Install required packages if necessary:
#    pip install pandas seaborn matplotlib pyreadstat
# ------------------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadstat

# ------------------------------------------------------------
# 2. Load the data
#    Make sure to adjust the path to where the file is located.
# ------------------------------------------------------------
df, meta = pyreadstat.read_dta(r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\PhD\two_thirds_submission\Github_replication_files_by_paper\Paper_1\data\Eurobarometer\Eurobarometer_data\all_data.dta")

# ------------------------------------------------------------
# 3. Convert 'year' to integer (if it is not already)
# ------------------------------------------------------------
df["year"] = df["year"].astype(int)

# Debug: Print initial data info
print("Initial data shape:", df.shape)
print("Columns:", df.columns)

# Debug: Print unique nation values in the dataset
print("\nUnique nation values in dataset:")
print(df['nation'].unique())

# Create a dictionary for country names (using numeric codes)
country_names = {
    1: 'France',
    3: 'Germany',
    4: 'Italy',
    5: 'Netherlands',
    12: 'Spain'
}

# Filter and clean the data
selected_countries = list(country_names.keys())
df_filtered = df[df['nation'].isin(selected_countries)].copy()

# Debug: Check if we have data after filtering
print("\nFiltered data shape:", df_filtered.shape)
print("Nations in filtered data:", df_filtered['nation'].unique())

df_filtered['nation'] = df_filtered['nation'].map(country_names)

# Remove any NaN values before aggregating
df_filtered = df_filtered.dropna(subset=['treu', 'treu_ecb', 'year', 'nation'])

# Debug: Print sample of filtered data
print("\nSample of filtered data:")
print(df_filtered[['year', 'nation', 'treu', 'treu_ecb']].head())

# Convert year to integer and ensure it stays as integer
df_filtered["year"] = df_filtered["year"].astype(int)

# Aggregate data by year and country
df_agg = df_filtered.groupby(['year', 'nation']).agg({
    'treu': 'mean',
    'treu_ecb': 'mean'
}).reset_index()

df_agg["year"] = df_agg["year"].astype(int)

# Debug: Print aggregated data
print("\nAggregated data:")
print(df_agg.describe())

# Set plot parameters
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(12, 7))

# Define colors and line styles
colors = {
    'France': '#1f77b4',
    'Germany': '#ff7f0e',
    'Italy': '#2ca02c',
    'Spain': '#d62728',
    'Netherlands': '#9467bd'
}

# Create two separate legend handles
eu_lines = []
ecb_lines = []

# Plot lines for EU trust
for country in country_names.values():
    country_data = df_agg[df_agg['nation'] == country].sort_values('year')
    if not country_data.empty and not country_data['treu'].isna().all():
        line, = ax.plot(country_data['year'], country_data['treu'], 
                     color=colors[country], marker='o', 
                     linewidth=2, label=country)
        eu_lines.append(line)

# Plot lines for ECB trust
for country in country_names.values():
    country_data = df_agg[df_agg['nation'] == country].sort_values('year')
    if not country_data.empty and not country_data['treu_ecb'].isna().all():
        line, = ax.plot(country_data['year'], country_data['treu_ecb'], 
                     color=colors[country], marker='s', linestyle='--',
                     linewidth=2, label=country)
        ecb_lines.append(line)

# Customize the plot
ax.set_title('Trust in EU Institutions', fontsize=14, pad=15)
ax.set_xlabel('', fontsize=12)
ax.set_ylabel('Trust Level', fontsize=12)

# Set integer ticks for years
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}"))

# Set y-axis limits and ticks based on actual data range
ax.set_ylim(1.1, 1.9)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

# Add grid
ax.grid(True, linestyle='--', alpha=0.3)
ax.grid(True, which='minor', linestyle=':', alpha=0.15)

# Create two-part legend
first_legend = ax.legend(eu_lines, [line.get_label() for line in eu_lines],
                        title='European Union', bbox_to_anchor=(1.02, 1), loc='upper left')
ax.add_artist(first_legend)
ax.legend(ecb_lines, [line.get_label() for line in ecb_lines],
          title='European Central Bank', bbox_to_anchor=(1.02, 0.5), loc='upper left')

# Adjust layout to prevent legend cutoff
plt.subplots_adjust(right=0.85)

# Save as PDF with high resolution
plt.savefig(r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\PhD\two_thirds_submission\Github_replication_files_by_paper\Paper_1\output\trust_in_eu_institutions.pdf", 
            dpi=300, 
            bbox_inches='tight',
            format='pdf')

# Show the plot
plt.show()

# Print detailed summary
print("\nDetailed data summary by year:")
print(df_agg.groupby('year')[['treu', 'treu_ecb']].describe())
