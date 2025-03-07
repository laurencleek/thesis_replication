##################################################################
#Who drives who? Agenda-setting within the multi-level Eurosystem#
######################Replication Code############################
###################Lauren Leek, 26/09/2024########################
##################################################################

##################################################################
##############This main file consists of 5 steps##################
#step 1: loading packages and set-up working space
#step 2: loading and processing speeches
#step 3: topic model
#step 4: creation of sequence analysis dataset
#step 5: creation of TSCS dataset with additional variables
##################################################################
##################################################################

### Step 1: loading packages and set-up working space
import os
import pandas as pd
from bertopic import BERTopic
import numpy as np
import importlib
import os
import sys
import matplotlib.pyplot as plt

    # Define the root and main directories
root_dir = os.path.abspath(r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\PhD\two_thirds_submission\Github_replication_files_by_paper\Paper_1")
main_dir = os.path.join(root_dir, "code", "py", "main")

    # Add the main directory to the system path
sys.path.append(main_dir)

    # Import the functions from modules in the main folder
try:
    from descriptive_figures import *
    print("Descriptive figures module loaded.")
except ImportError as e:
    print(f"Error importing descriptive figures module: {e}")

try:
    from topic_modelling import *
    print("Topic modeling module loaded.")
except ImportError as e:
    print(f"Error importing topic modeling module: {e}")

try:
    from dataset_creation import *
    print("Dataset creation module loaded.")
except ImportError as e:
    print(f"Error importing dataset creation module: {e}")

try:
    from UMAP_visual import *
    print("UMAP visual creation loaded.")
except ImportError as e:
    print(f"Error importing UMAP visual module: {e}")


    # Create output folder in the main working directory if it doesn't exist
output = os.path.join(root_dir, 'output')
if not os.path.exists(output):
    os.makedirs(output)

    # Ensure that output is saved in subfolders
figures = 'figures_paper'
figures_path = os.path.join(output, figures)
os.makedirs(figures_path, exist_ok=True)

appendix_fig = 'figures_appendix'
appendix_fig_path = os.path.join(output, appendix_fig)
os.makedirs(appendix_fig_path, exist_ok=True)

tables = 'tables_paper'
tables_path = os.path.join(output, tables)
os.makedirs(tables_path, exist_ok=True)

appendix_tables = 'tables_appendix'
appendix_tables_path = os.path.join(output, appendix_tables)
os.makedirs(appendix_tables_path, exist_ok=True)

### Step 2: load speeches (already cleaned), fix dates, filter eurozone
speeches_data_path = os.path.join(root_dir, 'data', 'speeches_all_country_data_except_vdem.parquet')
all_speeches = pd.read_parquet(speeches_data_path, engine='pyarrow')

    # Change date format 
all_speeches['date'] = pd.to_datetime(all_speeches['date']).dt.date
all_speeches['year'] = all_speeches['year'].astype(int)

    # Filter speeches for EU countries
EU_central_banks = ['european central bank', 'deutsche bundesbank', 'bank of france', 'netherlands bank', 'bank of italy', 'bank of spain']
filtered_df = all_speeches[all_speeches['central_bank'].isin(EU_central_banks)].reset_index(drop=True)
speeches_count_CB_EU = filtered_df.groupby(['central_bank'], sort=False).size().reset_index(name='Count')

### Step 3: descriptives

    # Call the function to plot and save the bar plot
speeches_count = all_speeches.groupby(['year'], sort=False).size().reset_index(name='Count')
filtered_df['date'] = pd.to_datetime(filtered_df['date'], format='%Y-%m-%d')
plot_stacked_speeches_over_time(all_speeches, speeches_count_CB_EU, appendix_fig_path)

    # Load pre-existing topic model if already present, otherwise run the topic model (22 minutes)
model = run_or_load_topic_modeling(filtered_df, output)

    # Create extra figures (word clouds)
create_wordclouds(model, appendix_fig_path, num_topics=45)

save_custom_latex_table(output)

### After Step 3: descriptives, modify the intra-topic analysis section:
try:
    from intra_topic_analysis import *
    print("Intra-topic analysis module loaded.")
except ImportError as e:
    print(f"Error importing intra-topic analysis module: {e}")

# Perform intra-topic analysis for selected topics
selected_topics = [0]  # Example topics to analyze
ncb_list = ['deutsche bundesbank', 'bank of france', 'netherlands bank', 'bank of italy', 'bank of spain']

# Get topic probabilities once
topic_probs = get_topic_probabilities(model, filtered_df['speech_text'].tolist())

for topic_id in selected_topics:
    # Compute embeddings for all speeches
    embeddings = compute_embeddings(filtered_df['speech_text'].tolist())
    
    # Aggregate embeddings by period
    aggregated = aggregate_embeddings_by_period(
        filtered_df,
        embeddings,
        topic_probs,
        topic_id,
        threshold=0.3
    )
    
    if aggregated:  # Only proceed if we have data
        # Compute distances
        distances = compute_ncb_ecb_distances(aggregated, ncb_list)
        
        if not distances.empty:  # Only plot if we have distances
            # Plot results
            plot_intra_topic_distances(distances, 0, figures_path)

### Step 4: create quarterly dataset (also merge eurobarometer and google trends)
chunk_size = 10000
eurobarometer_path = os.path.join(root_dir, 'data', 'Eurobarometer', 'Eurobarometer_data', 'all_data.dta')
google_trends_path = os.path.join(root_dir, 'data', 'google_trends')
FT_path = os.path.join(root_dir, 'data', 'FT_frequency.csv')

# Half yearly timeseries dataset
input_file = r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\PhD\two_thirds_submission\Github_replication_files_by_paper\Paper_1\output\timeseries.dta"
output_file = r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\PhD\two_thirds_submission\Github_replication_files_by_paper\Paper_1\output\half_timeseries.dta"
halfyear_data = convert_quarterly_to_halfyear(input_file, output_file)

# Yearly timeseries dataset
output_file2 = r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\PhD\two_thirds_submission\Github_replication_files_by_paper\Paper_1\output\year_timeseries.dta"
year_data = convert_quarterly_to_yearly(input_file, output_file2)

### Step 5: create sequence analysis dataset (speech level observations)
merged_dataset = create_and_merge_dataset_seq(
    model=model,
    filtered_df=filtered_df,
    folder_name=output,
    eurobarometer_path=eurobarometer_path,
    google_trends_path=google_trends_path,
    FT_path=FT_path,
    output_file_name="sequence.dta"  
)
# UMAP visual
fig = create_umap_visualization(model)
plt.show()
