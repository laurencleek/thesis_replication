##################################################################
########This timeseries analysis file consists of 5 steps#########
#step 1: loading packages and set-up working space
#step 2: loading and processing speeches
#step 3: topic model
#step 4: creation of sequence analysis dataset
#step 5: creation of TSCS dataset with additional variables
##################################################################
##################################################################

### step 1: loading packages and set-up working space
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import os
import sys
from stargazer.stargazer import Stargazer
import traceback

    # File paths
stem = "C:/Users/lclee/OneDrive - Istituto Universitario Europeo/PhD/two_thirds_submission/Github_replication_files_by_paper/Paper_1"
output = f"{stem}/output/"
data = f"{stem}/data/"

    # Ensure that output is saved in the right folders
figures = 'figures_paper'
figures_path = os.path.join(output, figures)
os.makedirs(figures_path, exist_ok=True)

appendix_fig = 'figures_appendix'
appendix_fig_path = os.path.join(output, appendix_fig)
os.makedirs(appendix_fig_path, exist_ok=True)

tables = 'tables_paper'
tables_path = os.path.join(output, tables)
os.makedirs(figures_path, exist_ok=True)

appendix_tables = 'tables_appendix'
appendix_tables_path = os.path.join(output, appendix_tables)
os.makedirs(appendix_tables_path, exist_ok=True)

sequence_dir = os.path.join(stem, "code", "py", "timeseries")

    # Add the sequence directory to the system path
sys.path.append(sequence_dir)

    # Load all the functions
try:
    from multiple_topics_figure import *
    print("Multiple topics figure module loaded.")
except ImportError as e:
    print(f"Error importing multiple topics figure module: {e}")

try:
    from table_1 import *
    print("Main aggregrate regression module loaded.")
except ImportError as e:
    print(f"Error importing main aggregrate regression module: {e}")

try:
    from create_tables import *
    print("Coefficient plot and tables module loaded.")
except ImportError as e:
    print(f"Error importing functions for coefficient plot module: {e}")

try:
    from figures_a4_to_a13 import *
    print("Interaction plot module loaded.")
except ImportError as e:
    print(f"Error importing interaction plot module: {e}")

try:
    from interaction_grid import *
    print("Multi-topic plot module loaded.")
except ImportError as e:
    print(f"Error importing multi-topic plot module: {e}")

try:
    from coefficient_plots import *
    print("Coefficient plot module loaded.")
except ImportError as e:
    print(f"Error importing coefficient plot module: {e}")

from coalition_grid import generate_coalition_grid, plot_coalition_grid
from sintra_analysis import run_sintra_regressions, create_sintra_latex_tables

# Load and preprocess data
df_quarterly = pd.read_stata(f"{output}/timeseries.dta")
df_half = pd.read_stata(f"{output}/half_timeseries.dta")

# Process quarterly data
df_quarterly['yq'] = pd.to_datetime(df_quarterly['quarter'], format='%Y-%m-%d')
df_quarterly['ECB'] = (df_quarterly['central_bank'] == 'european central bank').astype(int)
df_quarterly['NCB'] = (df_quarterly['central_bank'] != 'european central bank').astype(int)
df_quarterly['banks'] = df_quarterly['central_bank'].astype('category').cat.codes

# Process half-yearly data
df_half['yh'] = pd.to_datetime(df_half['halfyear'], format='%Y-%m-%d')
df_half['ECB'] = (df_half['central_bank'] == 'european central bank').astype(int)
df_half['NCB'] = (df_half['central_bank'] != 'european central bank').astype(int)
df_half['banks'] = df_half['central_bank'].astype('category').cat.codes

# Define topic and bank labels
topic_titles = {
    'Monetary_Policy_Central_Banking': 'Monetary Policy',
    'Economic_Analysis_Indicators': 'Economic Indicators',
    'Financial_Markets_Integration': 'Financial Markets',
    'Banking_Regulation_Supervision': 'Banking Supervision',
    'Digital_Finance_Innovation': 'Digital Finance',
    'International_Econ_Exchange': 'International Economics',
    'Crisis_Management_Stability': 'Crisis Management',
    'Sustainable_Finance_Climate': 'Climate',
    'Payment_Systems_Cash': 'Payment Systems',
    'National_Economy': 'National Economy'
}

bank_labels = {0: 'FR', 1: 'IT', 2: 'ES', 3: 'DE', 4: 'ECB', 5: 'NL'}
bank_names = list(bank_labels.values())

# Create multi-topic plot (figure 1)
create_multi_topic_plot(df_quarterly, topic_titles, bank_names, figures_path)

# Define common variables
control_vars = ['gdp_real_growth', 'unemployment_rate', 'hicp']

id_vars = ['central_bank', 'banks', 'ECB', 'NCB'] + control_vars

# Prepare quarterly data for regression
df_long_quarter = pd.melt(df_quarterly, 
                         id_vars=['yq'] + id_vars, 
                         value_vars=topic_titles, 
                         var_name='topic', 
                         value_name='value')
df_long_quarter['topic_label'] = df_long_quarter['topic'].map(topic_titles)
df_long_quarter = df_long_quarter.sort_values(['topic', 'banks', 'yq'])
df_long_quarter['value_lag'] = df_long_quarter.groupby(['topic', 'banks'])['value'].shift(1)
df_long_quarter['value_lead'] = df_long_quarter.groupby(['topic', 'banks'])['value'].shift(-1)

# Prepare half-yearly data for regression
df_long_half = pd.melt(df_half, 
                       id_vars=['yh'] + id_vars, 
                       value_vars=topic_titles, 
                       var_name='topic', 
                       value_name='value')
df_long_half['topic_label'] = df_long_half['topic'].map(topic_titles)
df_long_half = df_long_half.sort_values(['topic', 'banks', 'yh'])
df_long_half['value_lag'] = df_long_half.groupby(['topic', 'banks'])['value'].shift(1)
df_long_half['value_lead'] = df_long_half.groupby(['topic', 'banks'])['value'].shift(-1)

# Define banks and topics for both frequencies
ncb_banks = [bank for bank in df_long_quarter['banks'].unique() if bank != 4]  # Excluding ECB
topics = df_long_quarter['topic'].unique()

# Initialize results storage
all_results_quarter = pd.DataFrame()
models_quarter = {}

# Run quarterly regressions
print("Running quarterly regressions...")
for topic in topic_titles.keys():
    print(f"Processing topic: {topic}")
    topic_data = df_long_quarter[df_long_quarter['topic'] == topic].copy()
    
    if len(topic_data) > 0:
        try:
            topic_results, topic_models = run_regressions(
                topic_data, 
                ncb_banks,
                [topic], 
                control_vars, 
                bank_labels
            )
            
            if topic_results is not None and not topic_results.empty:
                topic_results['topic'] = topic
                all_results_quarter = pd.concat([all_results_quarter, topic_results], ignore_index=True)
                models_quarter.update(topic_models)
                print(f"Successfully processed {topic}")
            else:
                print(f"No results for topic {topic}")
        except Exception as e:
            print(f"Error processing topic {topic}: {e}")

# Initialize results storage for half-yearly
all_results_half = pd.DataFrame()
models_half = {}

# Run half-yearly regressions
print("Running half-yearly regressions...")
for topic in topic_titles.keys():
    print(f"Processing topic: {topic}")
    topic_data = df_long_half[df_long_half['topic'] == topic].copy()
    
    if len(topic_data) > 0:
        try:
            topic_results, topic_models = run_regressions_half(
                topic_data, 
                ncb_banks,
                [topic], 
                control_vars, 
                bank_labels
            )
            
            if topic_results is not None and not topic_results.empty:
                topic_results['topic'] = topic
                all_results_half = pd.concat([all_results_half, topic_results], ignore_index=True)
                models_half.update(topic_models)
                print(f"Successfully processed {topic}")
            else:
                print(f"No results for topic {topic}")
        except Exception as e:
            print(f"Error processing half-yearly topic {topic}: {e}")
            print("Data columns:", topic_data.columns.tolist())  # Debug print

# Crisis analysis (quarterly data only)
print("Running crisis analysis...")
all_results_crisis = pd.DataFrame()
models_crisis = {}
crisis_date = pd.to_datetime('2008-09-15')

for topic in topics:
    topic_data = df_long_quarter[df_long_quarter['topic'] == topic].copy()
    
    if len(topic_data) > 0:
        try:
            crisis_results, crisis_models = run_regressions_crisis(
                topic_data,
                ncb_banks,
                [topic],
                control_vars,
                bank_labels,
                crisis_date
            )
            
            if crisis_results is not None and not crisis_results.empty:
                crisis_results['topic'] = topic
                all_results_crisis = pd.concat([all_results_crisis, crisis_results], ignore_index=True)
                models_crisis.update(crisis_models)
        except Exception as e:
            print(f"Error processing crisis analysis for topic {topic}: {e}")

# Debug prints
print("\nResults Summary:")
print(f"Quarterly results shape: {all_results_quarter.shape}")
print(f"Half-yearly results shape: {all_results_half.shape}")
print(f"Crisis results shape: {all_results_crisis.shape}")

# Create all necessary subdirectories
figures_quarterly = os.path.join(figures_path, 'quarterly')
figures_half = os.path.join(figures_path, 'half_yearly')
appendix_quarterly = os.path.join(appendix_fig_path, 'quarterly')

os.makedirs(figures_quarterly, exist_ok=True)
os.makedirs(figures_half, exist_ok=True)
os.makedirs(appendix_quarterly, exist_ok=True)

# Create necessary subdirectories for tables
tables_quarterly = os.path.join(tables_path, 'quarterly')
tables_half = os.path.join(tables_path, 'half_yearly')

os.makedirs(tables_quarterly, exist_ok=True)
os.makedirs(tables_half, exist_ok=True)

# Generate plots and tables if results exist
if not all_results_quarter.empty:
    try:
        plot_coefficients(all_results_quarter, bank_labels, figures_quarterly, topic_titles)
        create_latex_tables(models_quarter, control_vars, bank_labels, tables_quarterly)
    except Exception as e:
        print(f"Error in plotting quarterly results: {e}")

if not all_results_crisis.empty:
    try:
        plot_coefficients_crisis_overlap(all_results_crisis, bank_labels, appendix_quarterly, topic_titles)
    except Exception as e:
        print(f"Error in plotting crisis results: {e}")

if not all_results_half.empty:
    try:
        plot_coefficients(all_results_half, bank_labels, figures_half, topic_titles)
        create_latex_tables(models_half, control_vars, bank_labels, tables_half)
    except Exception as e:
        print(f"Error in plotting half-yearly results: {e}")

#interaction plots
topic_bank_mapping = {
    'Monetary_Policy_Central_Banking': ['deutsche bundesbank', 'bank of italy', 'bank of france', 'bank of spain', 'netherlands bank'],
    'Economic_Analysis_Indicators': ['deutsche bundesbank', 'bank of italy', 'bank of france', 'bank of spain', 'netherlands bank'],
    'Sustainable_Finance_Climate': ['deutsche bundesbank', 'bank of italy', 'bank of france', 'bank of spain', 'netherlands bank'],
    'Crisis_Management_Stability': ['deutsche bundesbank', 'bank of italy', 'bank of france', 'bank of spain', 'netherlands bank'],
    'Financial_Markets_Integration': ['deutsche bundesbank', 'bank of italy', 'bank of france', 'bank of spain', 'netherlands bank'],
    'Banking_Regulation_Supervision': ['deutsche bundesbank', 'bank of italy', 'bank of france', 'bank of spain', 'netherlands bank'],
    'National_Economy': ['deutsche bundesbank', 'bank of italy', 'bank of france', 'bank of spain', 'netherlands bank'],
    'Payment_Systems_Cash': ['deutsche bundesbank', 'bank of italy', 'bank of france', 'bank of spain', 'netherlands bank'],
    'International_Econ_Exchange': ['deutsche bundesbank', 'bank of italy', 'bank of france', 'bank of spain', 'netherlands bank'],
    'Digital_Finance_Innovation': ['deutsche bundesbank', 'bank of italy', 'bank of france', 'bank of spain', 'netherlands bank'],
}

circumstances = ['google_trends_value', 'gdp_real_growth', 'treu_ecb', 'hicp']

# Prepare data for interaction analysis
df_quarterly_wide = df_quarterly.copy()
df_half_wide = df_half.copy()

# Convert long format topics back to wide format for quarterly data
for topic in topic_titles.keys():
    # Pivot the data for just this topic
    topic_pivot = df_long_quarter[df_long_quarter['topic'] == topic].pivot(
        index='yq',
        columns='central_bank',
        values='value'
    ).reset_index()
    
    # Create a temporary DataFrame for topic-specific lag values
    lag_data = pd.DataFrame({'yq': df_quarterly_wide['yq'].unique()})
    
    # Add lagged values for each bank
    for bank in topic_bank_mapping[topic]:
        bank_data = df_long_quarter[
            (df_long_quarter['topic'] == topic) & 
            (df_long_quarter['central_bank'] == bank)
        ].sort_values('yq')
        
        # Calculate lag for this bank and add to lag_data
        lag_data[f'{topic}_{bank}_lag'] = (
            bank_data.set_index('yq')['value']
            .shift(1)
            .reindex(lag_data['yq'])
        )
    
    # First merge lag data
    df_quarterly_wide = pd.merge(
        df_quarterly_wide,
        lag_data,
        on='yq',
        how='left'
    )
    
    # Then merge topic data
    df_quarterly_wide = pd.merge(
        df_quarterly_wide,
        topic_pivot,
        on='yq',
        how='left',
        suffixes=('', f'_{topic}')
    )

# Convert long format topics back to wide format for half-yearly data
for topic in topic_titles.keys():
    topic_pivot = df_long_half[df_long_half['topic'] == topic].pivot(
        index='yh',
        columns='central_bank',
        values='value'
    ).reset_index()
    
    # Create a temporary DataFrame for topic-specific lag values
    lag_data = pd.DataFrame({'yh': df_half_wide['yh'].unique()})
    
    # Add lagged values for each bank
    for bank in topic_bank_mapping[topic]:
        bank_data = df_long_half[
            (df_long_half['topic'] == topic) & 
            (df_long_half['central_bank'] == bank)
        ].sort_values('yh')
        
        # Calculate lag for this bank and add to lag_data
        lag_data[f'{topic}_{bank}_lag'] = (
            bank_data.set_index('yh')['value']
            .shift(1)
            .reindex(lag_data['yh'])
        )
    
    # First merge lag data
    df_half_wide = pd.merge(
        df_half_wide,
        lag_data,
        on='yh',
        how='left'
    )
    
    # Then merge topic data
    df_half_wide = pd.merge(
        df_half_wide,
        topic_pivot,
        on='yh',
        how='left',
        suffixes=('', f'_{topic}')
    )

# Create interaction plots for both frequencies
print("Creating quarterly interaction plots...")
create_topic_panel_plots(df_quarterly_wide, topic_bank_mapping, circumstances, output, topic_titles)

# Create interaction margin plots
print("Creating interaction margin plots...")
create_interaction_margin_plots(df_quarterly_wide, topic_bank_mapping, output)

# Create two-way interaction plots
print("Creating two-way interaction plots...")
create_two_way_margin_plots(df_quarterly_wide, topic_bank_mapping, output)

print("\nGenerating coalition interaction grid...")
try:
    # Generate quarterly coalition grid
    coalition_grid_quarter = generate_coalition_grid(
        df_quarterly, 
        topic_titles.keys(), 
        circumstances
    )
    
    # Generate half-yearly coalition grid
    coalition_grid_half = generate_coalition_grid(
        df_half, 
        topic_titles.keys(), 
        circumstances,
        time_col='yh'
    )
    
    # Create visualizations
    plot_coalition_grid(
        coalition_grid_quarter,
        os.path.join(figures_path, 'quarterly'),
        topic_titles
    )
    plot_coalition_grid(
        coalition_grid_half,
        os.path.join(figures_path, 'half_yearly'),
        topic_titles
    )
    
    print("Coalition grid analysis completed successfully")
except Exception as e:
    print(f"Error in coalition grid analysis: {e}")
    traceback.print_exc()

def main():
    # Generate the grid for quarterly data
    print("\nGenerating quarterly interaction grid...")
    grid_quarter = generate_grid(df_quarterly_wide, topic_bank_mapping, circumstances)
    print("\nQuarterly Grid:")
    print_grid(grid_quarter)
    save_color_grid(grid_quarter, os.path.join(figures_path, 'quarterly'))
    
    # Generate the grid for half-yearly data
    print("\nGenerating half-yearly interaction grid...")
    grid_half = generate_grid(df_half_wide, topic_bank_mapping, circumstances)
    print("\nHalf-yearly Grid:")
    print_grid(grid_half)
    save_color_grid(grid_half, os.path.join(figures_path, 'half_yearly'))

if __name__ == "__main__":
    main()
    # Create quarterly output table
    results, models = create_basic_output_table(
        df=df_long_quarter,
        ncb_banks=ncb_banks,
        topics=topics,
        controls=control_vars,
        bank_labels=bank_labels,
        output_path=os.path.join(output, 'tables_paper', 'basic_output_table.tex')
    )
    
    # Create half-yearly output table
    results_half, models_half = create_basic_output_table_half(
        df=df_long_half,
        ncb_banks=ncb_banks,
        topics=topics,
        controls=control_vars,
        bank_labels=bank_labels,
        output_path=os.path.join(output, 'tables_paper', 'basic_output_table_half.tex')
    )

try:
    print("\nRunning leads analysis...")
    leads_data = df_long_quarter.copy()
    leads_data = leads_data.dropna(subset=control_vars)
    
    results_leads_df, models_leads = run_regressions_leads(
        leads_data, 
        ncb_banks, 
        topics, 
        control_vars, 
        bank_labels
    )
    
    if not results_leads_df.empty:
        print(f"Generated {len(results_leads_df)} lead coefficients")
        plot_coefficients_leads(results_leads_df, bank_labels, output, topic_titles)
    else:
        print("No results available for plotting leads coefficients")
        print("Data shape:", leads_data.shape)
        print("Available topics:", leads_data['topic'].unique())
        print("Available banks:", leads_data['banks'].unique())
except Exception as e:
    print(f"Error in leads analysis: {e}")
    import traceback
    traceback.print_exc()

# Create three-way interaction plots
print("Creating three-way interaction plots...")
create_three_way_interaction_plot(df_quarterly_wide, topic_bank_mapping, output)


# Generate the grid
coalition_grid = generate_coalition_grid(df_quarterly, topic_titles.keys(), circumstances)

# Create visualization
plot_coalition_grid(
    coalition_grid,
    os.path.join(figures_path, 'quarterly'),
    topic_titles
)

# After your existing analysis
print("\nRunning Sintra analysis...")
try:
    # Run Sintra regressions with correct arguments
    sintra_results = run_sintra_regressions(
        df=df_long_quarter,
        ncb_banks=ncb_banks,      # Added this
        topics=topic_titles.keys(),
        controls=control_vars,
        bank_labels=bank_labels
    )
    
    # Create output directory for Sintra analysis
    sintra_tables_path = os.path.join(tables_path, 'sintra')
    os.makedirs(sintra_tables_path, exist_ok=True)
    
    # Generate LaTeX tables
    create_sintra_latex_tables(
        sintra_results,
        os.path.join(sintra_tables_path, 'sintra_regression_tables.tex'),
        topic_titles
    )
    
    print("Sintra analysis completed successfully")
except Exception as e:
    print(f"Error in Sintra analysis: {e}")
    traceback.print_exc()
