import pandas as pd
import os as os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import timedelta
from hmmlearn import hmm
from scipy.stats import binom_test
from datetime import timedelta
import math

def plot_topic_transition_matrix(df, topic_titles, output_dir, file_name="transition_matrix"):
    """
    Create and save a transition matrix heatmap for selected topics.

    Parameters:
    - df: pandas DataFrame containing the topic data.
    - topic_titles: dictionary mapping topic column names to human-readable titles.
    - output_dir: string, path to the directory where the plot will be saved.
    - file_name: string, base name for the saved plot files (default is "transition_matrix").

    Saves:
    - Heatmap plot as both PNG and PDF in the specified output directory.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Select specific topics for analysis
    selected_topic_columns = list(topic_titles.keys())

    # Recalculate the dominant topic using only the selected topic columns
    df['dominant_topic_selected'] = df[selected_topic_columns].idxmax(axis=1)

    # Shift the dominant topic column by one to get transitions from one topic to the next
    df['next_topic_selected'] = df['dominant_topic_selected'].shift(-1)

    # Remove any rows where the next topic is NaN
    df_selected = df.dropna(subset=['next_topic_selected'])

    # Create a transition matrix for the selected topics
    transition_matrix_selected = pd.crosstab(
        df_selected['dominant_topic_selected'],
        df_selected['next_topic_selected'],
        normalize='index'
    )

    # Rename the index and columns using the human-readable topic titles
    transition_matrix_selected.index = [topic_titles[topic] for topic in transition_matrix_selected.index]
    transition_matrix_selected.columns = [topic_titles[topic] for topic in transition_matrix_selected.columns]

    # Plot the heatmap for selected topics
    plt.figure(figsize=(14, 12))  # Increased figure size slightly
    sns.heatmap(transition_matrix_selected, annot=True, cmap="Blues", linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.xlabel("Next Topic", labelpad=15)
    plt.ylabel("Current Topic", labelpad=15)
    plt.title("Topic Transition Matrix", pad=20)

    # Adjust layout to ensure titles and labels are fully visible
    plt.tight_layout()

    # Save the plot as PNG and PDF
    plt.savefig(os.path.join(output_dir, f"{file_name}.png"), format="png", dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"{file_name}.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


def plot_climate_topic_transitions(df, climate_topic_column, output_dir, file_name="climate_transitions"):
    """
    Plot and save the climate topic transitions for each central bank after 2016.

    Parameters:
    - df: pandas DataFrame containing the topic data.
    - climate_topic_column: string, name of the column representing the climate topic.
    - output_dir: string, path to the directory where the plot will be saved.
    - file_name: string, base name for the saved plot files (default is "climate_transitions").

    Saves:
    - Line plot as both PNG and PDF in the specified output directory.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a new column indicating whether 'Climate' was the dominant topic in each speech
    df['is_climate'] = df[climate_topic_column] > 0.1  # Assuming threshold for topic relevance

    # Group the data by central bank and year to observe 'Climate' topic over time
    climate_transitions = df[df['is_climate']].groupby(['country', 'year']).size().unstack(fill_value=0)

    # Filter the climate transitions data for only the years after 2016
    climate_transitions_after_2016 = climate_transitions.loc[:, climate_transitions.columns > 2015]

    # Plot the climate topic transitions for each central bank after 2016
    plt.figure(figsize=(12, 6))
    for central_bank in climate_transitions_after_2016.index:
        plt.plot(
            climate_transitions_after_2016.columns,
            climate_transitions_after_2016.loc[central_bank],
            marker='o',
            label=central_bank
        )

    # Customize plot appearance
    plt.title("Climate Topic Transitions Across Central Banks (After 2015)", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Number of Speeches on Climate Topic", fontsize=14)
    plt.legend(title="Central Bank", loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as PNG and PDF
    plt.savefig(os.path.join(output_dir, f"{file_name}.png"), format="png", dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"{file_name}.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.show()

def plot_crisis_topic_transitions(df, crisis_topic_column, output_dir, file_name="crisis_transitions"):
    """
    Plot and save the crisis management topic transitions for each central bank over time.

    Parameters:
    - df: pandas DataFrame containing the topic data.
    - crisis_topic_column: string, name of the column representing the crisis management topic.
    - output_dir: string, path to the directory where the plot will be saved.
    - file_name: string, base name for the saved plot files (default is "crisis_transitions").

    Saves:
    - Line plot as both PNG and PDF in the specified output directory.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a new column indicating whether 'Crisis Management' was the dominant topic in each speech
    df['is_crisis'] = df[crisis_topic_column] > 0.1  # Assuming threshold for topic relevance

    # Group the data by central bank and year to observe 'Crisis Management' topic over time
    crisis_transitions = df[df['is_crisis']].groupby(['country', 'year']).size().unstack(fill_value=0)

    # Plot the crisis management topic transitions for each central bank over time
    plt.figure(figsize=(12, 6))
    for central_bank in crisis_transitions.index:
        plt.plot(
            crisis_transitions.columns,
            crisis_transitions.loc[central_bank],
            marker='o',
            label=central_bank
        )

    # Customize plot appearance
    plt.title("Crisis Management Topic Transitions Across Central Banks Over Time", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Number of Speeches on Crisis Management Topic", fontsize=14)
    plt.legend(title="Central Bank", loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as PNG and PDF
    plt.savefig(os.path.join(output_dir, f"{file_name}.png"), format="png", dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"{file_name}.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


def plot_markov_transition_matrix(df, selected_topic_columns, topic_titles, output_dir, file_name="markov_transition_matrix"):
    """
    Fit a Markov model on the dominant topics and plot the transition matrix as a heatmap.

    Parameters:
    - df: pandas DataFrame containing the topic data.
    - selected_topic_columns: list of strings, columns representing topics to be included in the Markov model.
    - topic_titles: dictionary mapping topic column names to human-readable titles.
    - output_dir: string, path to the directory where the plot will be saved.
    - file_name: string, base name for the saved plot files (default is "markov_transition_matrix").

    Saves:
    - Heatmap plot as both PNG and PDF in the specified output directory.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Map topics to numerical labels and back
    topic_mapping = {topic: idx for idx, topic in enumerate(selected_topic_columns)}
    inverse_topic_mapping = {idx: topic for topic, idx in topic_mapping.items()}

    # Get dominant topic for each speech
    df['dominant_topic'] = df[selected_topic_columns].idxmax(axis=1)
    df['dominant_topic_num'] = df['dominant_topic'].map(topic_mapping)

    # Drop rows with missing values in the dominant topic
    df = df.dropna(subset=['dominant_topic_num'])

    # Prepare sequences for the Markov model
    sequences = df['dominant_topic_num'].values.reshape(-1, 1)

    # Fit a Multinomial Hidden Markov Model
    model = hmm.MultinomialHMM(n_components=len(selected_topic_columns), n_iter=100, random_state=42)
    model.fit(sequences)

    # Get the transition matrix
    transmat = model.transmat_

    # Convert transition matrix to DataFrame for readability
    transmat_df = pd.DataFrame(transmat,
                               index=[topic_titles[topic] for topic in selected_topic_columns],
                               columns=[topic_titles[topic] for topic in selected_topic_columns])

    # Plot the transition matrix heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(transmat_df, annot=True, cmap="Blues", linewidths=0.5)
    plt.title("Markov Transition Matrix of Topic Transitions")
    plt.xlabel("Next Topic")
    plt.ylabel("Current Topic")
    plt.tight_layout()

    # Save the plot as PNG and PDF
    plt.savefig(os.path.join(output_dir, f"{file_name}.png"), format="png", dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"{file_name}.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.show()

# Function to calculate Markov transition matrix for a central bank
def calculate_transition_matrix(df, central_bank):
    # Filter data for the central bank
    df_cb = df[df['country'] == central_bank].copy()

    # Identify dominant topics
    df_cb['dominant_topic'] = df_cb[selected_topic_columns].idxmax(axis=1)

    # Convert topics to numerical labels
    topic_mapping = {topic: idx for idx, topic in enumerate(selected_topic_columns)}
    df_cb['dominant_topic_num'] = df_cb['dominant_topic'].map(topic_mapping)

    # Remove any rows with missing dominant topic
    df_cb = df_cb.dropna(subset=['dominant_topic_num'])

    # Prepare the sequences for the Markov model
    sequences = df_cb['dominant_topic_num'].values.reshape(-1, 1)

    # Fit the Markov model
    model = hmm.MultinomialHMM(n_components=len(selected_topic_columns), n_iter=100)
    model.fit(sequences)

    # Extract the transition matrix
    transmat = model.transmat_

    # Convert to a DataFrame for easier interpretation
    transmat_df = pd.DataFrame(transmat, index=[topic_titles[topic] for topic in selected_topic_columns],
                               columns=[topic_titles[topic] for topic in selected_topic_columns])
    return transmat_df

def calculate_transition_matrix(df, bank):
    """
    Placeholder for the transition matrix calculation for a specific central bank.
    This function should be implemented to calculate the transition matrix.
    """
    # Filter the dataset for the specific central bank
    bank_df = df[df['country'] == bank]

    # Assuming 'dominant_topic_num' is already calculated as in previous examples
    bank_df = bank_df.dropna(subset=['dominant_topic_num'])
    sequences = bank_df['dominant_topic_num'].values.reshape(-1, 1)

    # Fit the Markov model
    model = hmm.MultinomialHMM(n_components=len(selected_topic_columns), n_iter=100, random_state=42)
    model.fit(sequences)

    # Return the transition matrix as a DataFrame
    transmat = model.transmat_
    transmat_df = pd.DataFrame(
        transmat,
        index=[topic_titles[topic] for topic in selected_topic_columns],
        columns=[topic_titles[topic] for topic in selected_topic_columns]
    )
    return transmat_df

def plot_transition_matrix(transmat_df, central_bank, output_dir, file_name):
    """
    Plot and save the Markov transition matrix for a given central bank.

    Parameters:
    - transmat_df: DataFrame containing the transition matrix for a central bank.
    - central_bank: string, name of the central bank.
    - output_dir: string, directory where the plot will be saved.
    - file_name: string, base name for the saved plot files.
    """
    # Plot the transition matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(transmat_df, annot=True, cmap="Blues", linewidths=0.5)
    plt.title(f"Markov Transition Matrix for {central_bank}")
    plt.xlabel("Next Topic")
    plt.ylabel("Current Topic")
    plt.tight_layout()

    # Save the plot as PNG and PDF
    plt.savefig(os.path.join(output_dir, f"{file_name}_{central_bank}.png"), format="png", dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"{file_name}_{central_bank}.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.show()

def analyze_and_plot_transition_matrices(df, output_dir):
    """
    Calculate and plot Markov transition matrices for each central bank in the dataset.

    Parameters:
    - df: DataFrame containing the data, with a 'country' column for central bank identification.
    - output_dir: string, directory where the plots will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List of central banks to analyze
    central_banks = df['country'].unique()
    
    # For each central bank, calculate and plot the Markov transition matrix
    for bank in central_banks:
        transmat_df = calculate_transition_matrix(df, bank)
        plot_transition_matrix(transmat_df, bank, output_dir, file_name="markov_transition_matrix")


def analyze_first_movers(output, df, topic_titles, threshold=0.25, time_window_days=365, selected_NCBs=None):
    """
    Analyze and visualize first movers on emerging topics among central banks.

    Parameters:
    - df: pandas DataFrame containing the speech data.
    - topic_titles: dictionary mapping topic column names to human-readable titles.
    - threshold: float, the topic significance threshold (default is 0.1).
    - time_window_days: int, the rolling time window in days to consider other central banks' speeches (default is 365 days).
    - selected_NCBs: list of strings, the list of NCBs to highlight in the visualization (default is None).

    Returns:
    - valid_first_movers_df: DataFrame containing information about valid first movers.
    - first_mover_type_counts: Series containing counts of first mover types ('NCB', 'ECB', 'Both').
    """
    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Initialize a list to store valid first movers
    valid_first_movers_list = []

    # For each topic
    for topic_column, topic_name in topic_titles.items():
        # Identify significant speeches for the topic
        df_topic = df.copy()
        df_topic['is_significant_topic'] = df_topic[topic_column] >= threshold

        # Filter speeches where the topic is significantly addressed
        significant_speeches = df_topic[df_topic['is_significant_topic']]

        # If there are no significant speeches for the topic, skip to next topic
        if significant_speeches.empty:
            continue

        # Sort the significant speeches by date
        significant_speeches = significant_speeches.sort_values('date')

        # Initialize variables to track if a valid date is found
        valid_date_found = False
        date_index = 0

        # Get the unique dates when the topic was addressed
        unique_dates = significant_speeches['date'].unique()

        # Iterate over the unique dates
        while date_index < len(unique_dates) and not valid_date_found:
            current_date = unique_dates[date_index]

            # Identify the central bank(s) that addressed the topic on the current date
            first_movers = significant_speeches[significant_speeches['date'] == current_date]
            first_movers = first_movers[['country', 'date']].drop_duplicates()
            first_mover_countries = first_movers['country'].unique().tolist()

            # Calculate the end date for the time window
            end_date = current_date + timedelta(days=time_window_days)

            # Filter speeches by other central banks within the time window
            other_speeches_within_window = significant_speeches[
                (significant_speeches['date'] > current_date) &
                (significant_speeches['date'] <= end_date) &
                (~significant_speeches['country'].isin(first_mover_countries))
            ]

            # Count the number of other central banks
            other_central_banks = other_speeches_within_window['country'].unique()
            num_other_central_banks = len(other_central_banks)

            # Check if at least 3 other central banks discussed the topic within the time window
            if num_other_central_banks >= 3:
                valid_date_found = True

                # Determine the bank type of first movers
                first_movers['bank_type'] = first_movers['country'].apply(lambda x: 'NCB' if x != 'ECB' else 'ECB')

                # Create a dictionary for the topic with relevant information
                topic_first_mover = {
                    'topic': topic_name,
                    'earliest_date': current_date,
                    'first_mover_bank_types': first_movers['bank_type'].unique().tolist(),
                    'first_mover_countries': first_mover_countries,
                    'other_central_banks_within_window': other_central_banks.tolist()
                }

                # Append to the list
                valid_first_movers_list.append(topic_first_mover)
            else:
                # Move to the next date
                date_index += 1

        # If no valid date is found for the topic, you can choose to handle it accordingly
        # For now, we skip the topic if no valid date is found
        if not valid_date_found:
            print(f"No valid date found for topic '{topic_name}' where at least 3 other central banks discussed it within the time window.")

    # Combine all valid first movers into a single DataFrame
    if valid_first_movers_list:
        valid_first_movers_df = pd.DataFrame(valid_first_movers_list)
    else:
        print("No valid first movers found.")
        valid_first_movers_df = pd.DataFrame(columns=['topic', 'earliest_date', 'first_mover_bank_types'])

    # Function to determine the first mover type based on bank types
    def determine_first_mover_type(bank_types):
        if 'NCB' in bank_types and 'ECB' in bank_types:
            return 'Both'
        elif 'NCB' in bank_types:
            return 'NCB'
        elif 'ECB' in bank_types:
            return 'ECB'
        else:
            return 'Unknown'

    # Apply the function to determine first mover type per topic
    valid_first_movers_df['first_mover_type'] = valid_first_movers_df['first_mover_bank_types'].apply(determine_first_mover_type)

    # Count the number of topics where NCBs and ECB were first movers
    first_mover_type_counts = valid_first_movers_df['first_mover_type'].value_counts()

    # Visualize the first movers
    if not valid_first_movers_df.empty:
        plt.figure(figsize=(12, 6))

        # If selected_NCBs is None, use default values
        if selected_NCBs is None:
            selected_NCBs = ['DEU', 'FRA', 'NLD', 'ESP', 'ITA']  # Replace with actual NCB names in your data

        # Assign distinct colors to the selected NCBs
        colors_list = list(mcolors.TABLEAU_COLORS.values())  # Get a list of colors
        ncb_color_map = {ncb: colors_list[i % len(colors_list)] for i, ncb in enumerate(selected_NCBs)}

        # Initialize a set to keep track of labels added to the legend
        labels_seen = set()

        # Set font to Times New Roman
        plt.rcParams["font.family"] = "Times New Roman"

        # Plot each first mover
        for idx, row in valid_first_movers_df.iterrows():
            first_mover_countries = row['first_mover_countries']
            for country in first_mover_countries:
                if country == 'ECB':
                    # Handle ECB separately
                    label = 'ECB' if 'ECB' not in labels_seen else "_nolegend_"
                    labels_seen.add('ECB')
                    plt.scatter(row['earliest_date'], row['topic'], label=label, color='black', s=100, edgecolors='k')
                elif country in selected_NCBs:
                    # Assign color and label for the NCB
                    color = ncb_color_map.get(country, 'gray')
                    label = country if country not in labels_seen else "_nolegend_"
                    labels_seen.add(country)
                    plt.scatter(row['earliest_date'], row['topic'], label=label, color=color, s=100, edgecolors='k')
                else:
                    # For NCBs not in the selected list, you can choose to skip or handle them differently
                    continue

        plt.title("First Movers on Emerging Topics", fontsize=16)
        plt.xlabel('')
        plt.ylabel('')
        plt.legend(title="First Movers", fontsize=12, title_fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Create output directory
        plt.savefig(os.path.join(output, "figure_first_movers.png"), format="png", dpi=300)
        plt.savefig(os.path.join(output, "figure_first_movers.pdf"), format="pdf", dpi=300)
        plt.show()
    else:
        print("No valid first movers to visualize.")

    # Return the DataFrame and counts
    return valid_first_movers_df, first_mover_type_counts

def create_topic_transitions_panel(df, topic_columns, topic_titles, output_dir, n_cols, window_days=30):
    """Create a panel figure showing transition matrices for all topics in a grid layout with a configurable number of columns."""
    set_academic_style()
    
    n_topics = len(topic_columns)
    n_rows = math.ceil(n_topics / n_cols)
    
    # Adjust figure size based on the number of rows & columns
    fig_width = 5 * n_cols
    fig_height = 5 * n_rows
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    gs = plt.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.15)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    
    for idx, topic_col in enumerate(topic_columns):
        row = idx // n_cols
        col = idx % n_cols
        
        topic_name = topic_titles[topic_col]
        # --- calculate the transition matrix for the given topic ---
        topic_speeches = df[df[topic_col] > 0.2].copy()
        topic_speeches = topic_speeches.sort_values('date')
        banks = topic_speeches['country'].unique()
        transition_matrix = pd.DataFrame(0, index=banks, columns=banks)
        
        for bank in banks:
            bank_speeches = topic_speeches[topic_speeches['country'] == bank]
            for _, speech in bank_speeches.iterrows():
                next_date = speech['date'] + timedelta(days=window_days)
                next_speeches = topic_speeches[
                    (topic_speeches['date'] > speech['date']) &
                    (topic_speeches['date'] <= next_date)
                ]
                if not next_speeches.empty:
                    next_bank = next_speeches.iloc[0]['country']
                    transition_matrix.loc[bank, next_bank] += 1
        
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix.div(row_sums, axis=0).fillna(0)
        
        ax = fig.add_subplot(gs[row, col])
        sns.heatmap(transition_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlBu_r',
                    center=0.5,
                    vmin=0,
                    vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar=False,
                    ax=ax)
        ax.set_title(topic_name, fontsize=14, pad=10, fontweight='bold')
        if row == n_rows - 1:
            ax.set_xlabel('Next Central Bank', fontsize=10)
        else:
            ax.set_xlabel('')
        if col == 0:
            ax.set_ylabel('Current Central Bank', fontsize=10)
        else:
            ax.set_ylabel('')
        ax.tick_params(labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    cbar_ax.set_ylabel('Transition Probability', fontsize=14, rotation=270, labelpad=20)
    cbar_ax.tick_params(labelsize=12)
    
    fig.suptitle('Topic Transitions Between Central Banks\n' + f'({window_days}-day window)', fontsize=18, y=0.95, fontweight='bold')
    
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(output_dir, f"topic_transitions_panel{ext}"),
                    dpi=300, bbox_inches='tight', metadata={'Creator': 'Academic Visualization Tool'})
    plt.close()


def create_crisis_topic_transitions_panel(df, topic_columns, topic_titles, output_dir, n_cols=4, window_days=30):
    """
    Create a panel of topic transition matrices specifically for the euro crisis period (2010-2015).
    """
    # Ensure we're working with a copy and reset the index
    working_df = df.reset_index(drop=True).copy()
    
    # Convert date column to datetime if needed
    working_df['date'] = pd.to_datetime(working_df['date'])
    
    # Filter for crisis years
    crisis_df = working_df[
        (working_df['date'].dt.year >= 2010) & 
        (working_df['date'].dt.year <= 2015)
    ].copy()
    
    n_topics = len(topic_columns)
    n_rows = math.ceil(n_topics / n_cols)
    
    # Create figure
    fig_width = 5 * n_cols
    fig_height = 5 * n_rows
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create grid
    gs = plt.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Process each topic
    for idx, topic_col in enumerate(topic_columns):
        row = idx // n_cols
        col = idx % n_cols
        topic_name = topic_titles[topic_col]
        
        # Calculate transitions for this topic
        topic_speeches = crisis_df[crisis_df[topic_col] > 0.2].copy()
        topic_speeches = topic_speeches.sort_values('date', ignore_index=True)
        banks = topic_speeches['country'].unique()
        
        # Create transition matrix
        transition_matrix = pd.DataFrame(0, index=banks, columns=banks)
        
        for bank in banks:
            bank_speeches = topic_speeches[topic_speeches['country'] == bank]
            for _, speech in bank_speeches.iterrows():
                next_date = speech['date'] + timedelta(days=window_days)
                next_speeches = topic_speeches[
                    (topic_speeches['date'] > speech['date']) &
                    (topic_speeches['date'] <= next_date)
                ]
                if not next_speeches.empty:
                    next_bank = next_speeches.iloc[0]['country']
                    transition_matrix.loc[bank, next_bank] += 1
        
        # Normalize transition probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix.div(row_sums, axis=0).fillna(0)
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        sns.heatmap(transition_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlBu_r',
                    center=0.5,
                    vmin=0,
                    vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar=True if col == n_cols-1 else False,
                    ax=ax)
        
        ax.set_title(topic_name, fontsize=12, pad=10)
        if row == n_rows - 1:
            ax.set_xlabel('Next CB', fontsize=10)
        else:
            ax.set_xlabel('')
        if col == 0:
            ax.set_ylabel('Current CB', fontsize=10)
        else:
            ax.set_ylabel('')
            
        ax.tick_params(labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Main title
    fig.suptitle('Topic Transitions During Eurozone Crisis (2010-2015)\n' +
                 f'({window_days}-day window)', 
                 fontsize=16, y=0.95)
    
    # Save figure
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(output_dir, f"crisis_topic_transitions_panel{ext}"),
                    dpi=300, bbox_inches='tight')
    plt.close()
