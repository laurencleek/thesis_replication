##################################################################
########This analysis seqeuence file consists of 5 steps##########
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
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import timedelta
from hmmlearn import hmm
from scipy.stats import binom_test
import os
import sys

    # Define the root directory and set up the path to the sequence folder
root_dir = os.path.abspath(r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\PhD\two_thirds_submission\Github_replication_files_by_paper\Paper_1")
sequence_dir = os.path.join(root_dir, "code", "py", "sequence")

    # Add the sequence directory to the system path
sys.path.append(sequence_dir)

    # importing the modules
try:
    from markov_transitions import *
    print("markov transitions module loaded.")
except ImportError as e:
    print(f"Error importing dataset creation module: {e}")

try:
    from first_movers import *
    print("First movers module loaded.")
except ImportError as e:
    print(f"Error importing dataset creation module: {e}")

try:
    from transition_matrices import *
    print("transition matrices module loaded.")
except ImportError as e:
    print(f"Error importing dataset creation module: {e}")


    # Load the specified .dta file (created by main)
file_path = r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\PhD\two_thirds_submission\Github_replication_files_by_paper\Paper_1\output\sequence.dta"
df = pd.read_stata(file_path)

    # Define the path for the output figures
output_figures_paper = os.path.join(root_dir, 'output', 'figures_paper')
output_figures_appendix = os.path.join(root_dir, 'output', 'figures_appendix')

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

selected_topic_columns = ['Monetary_Policy_Central_Banking', 'Economic_Analysis_Indicators',
                          'Financial_Markets_Integration', 'Banking_Regulation_Supervision',
                          'Digital_Finance_Innovation', 'International_Econ_Exchange',
                          'Crisis_Management_Stability', 'Sustainable_Finance_Climate',
                          'Payment_Systems_Cash', 'National_Economy']

selected_NCBs = ['EA', 'DEU', 'FRA', 'NLD', 'ESP', 'ITA'] 

    # Call the function to create and save the transition matrix plot
plot_topic_transition_matrix(df, topic_titles, output_figures_paper, file_name="figure_transition")
# Call the function to create and save the climate topic transitions plot
plot_climate_topic_transitions(df, 'Sustainable_Finance_Climate', output_figures_paper, file_name="figure_climate_transition")
# Call the function to create and save the crisis management topic transitions plot
plot_crisis_topic_transitions(df, 'Crisis_Management_Stability', output_figures_paper, file_name="figure_crisis_transition")
# Call the function to create and save the Markov transition matrix plot
plot_markov_transition_matrix(df, selected_topic_columns, topic_titles, output_figures_paper, file_name="figure_markov_transition")

topic_titles_new = {
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

# Call the function for first movers graph and save it
valid_first_movers_df, first_mover_type_counts = analyze_first_movers(
    output_figures_paper,
    df,
    topic_titles_new,
    threshold=0.25,
    time_window_days=365,
    selected_NCBs=selected_NCBs
)


# Panel of transition matrices
transition_matrices = analyze_all_topics_transitions(
    df=df,
    topic_columns=selected_topic_columns,
    topic_titles=topic_titles,
    output_dir=output_figures_paper
)

create_topic_transitions_panel(
    df,
    selected_topic_columns,
    topic_titles,
    output_figures_paper,
    n_cols=4,
    window_days=30
        
)

# Create crisis period topic transitions panel
create_crisis_topic_transitions_panel(
    df=df,
    topic_columns=selected_topic_columns,
    topic_titles=topic_titles,
    output_dir=output_figures_paper,
    n_cols=4,
    window_days=30
)

#robustness in markov
results = analyze_robustness_across_windows(
    df=df,
    topic_column='Monetary_Policy_Central_Banking',  
    topic_name='Monetary Policy',
    output_dir=output_figures_paper,  
    windows=[7, 14, 30, 60, 90]
)

for topic_col in selected_topic_columns:
    topic_name = topic_titles[topic_col]
    results = analyze_robustness_across_windows(
        df=df,
        topic_column=topic_col,
        topic_name=topic_name,
        output_dir=output_figures_paper,
        windows=[7, 14, 30, 60, 90]
    )

plot_topic_transition_matrix_robustness(
    df=df,
    topic_titles=selected_topic_columns,
    output_dir=output_figures_paper,
    file_name='transition_matrix_robustness'
)