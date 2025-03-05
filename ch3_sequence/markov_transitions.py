import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
from hmmlearn import hmm
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties

def set_academic_style():
    """Set publication-quality plot style."""
    plt.style.use('default')  # Reset to default style
    
    # Set figure style parameters
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'figure.dpi': 300,
        'axes.grid': False,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
    })

def plot_academic_heatmap(transition_matrix, title, output_dir, file_name, window_days=30):
    """
    Create a publication-quality heatmap for topic transitions.
    """
    set_academic_style()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap with improved aesthetics - removed mask to show all values
    sns.heatmap(transition_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                center=0.5,
                vmin=0,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Transition Probability'},
                ax=ax)

    # Improve title and labels
    ax.set_title(f'{title}\n({window_days}-day window)',
                 pad=20,
                 fontsize=16,
                 fontfamily='Times New Roman',
                 fontweight='bold')
    
    ax.set_xlabel('Next Central Bank',
                  fontsize=14,
                  fontfamily='Times New Roman',
                  fontweight='bold')
    ax.set_ylabel('Current Central Bank',
                  fontsize=14,
                  fontfamily='Times New Roman',
                  fontweight='bold')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()

    # Save with high quality
    for ext in ['.png', '.pdf']:
        plt.savefig(
            os.path.join(output_dir, f"{file_name}{ext}"),
            dpi=300,
            bbox_inches='tight',
            metadata={'Creator': 'Academic Visualization Tool'}
        )
    
    plt.close()

def plot_markov_transition_matrix(df, selected_topic_columns, topic_titles, output_dir, file_name="markov_transition_matrix"):
    """
    Fit a Markov model on the dominant topics and plot the transition matrix as a heatmap.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get dominant topic for each speech
    df['dominant_topic'] = df[selected_topic_columns].idxmax(axis=1)
    
    # Create numerical mapping for topics
    topic_mapping = {topic: idx for idx, topic in enumerate(selected_topic_columns)}
    df['dominant_topic_num'] = df['dominant_topic'].map(topic_mapping)

    # Drop rows with missing values
    df = df.dropna(subset=['dominant_topic_num'])

    # Sort by date to ensure proper sequence
    df = df.sort_values('date')

    # Calculate transition counts manually
    n_topics = len(selected_topic_columns)
    transitions = np.zeros((n_topics, n_topics))
    
    # Count transitions
    for i in range(len(df) - 1):
        current_topic = int(df.iloc[i]['dominant_topic_num'])
        next_topic = int(df.iloc[i + 1]['dominant_topic_num'])
        transitions[current_topic][next_topic] += 1

    # Convert to probabilities
    row_sums = transitions.sum(axis=1)
    transition_matrix = transitions / row_sums[:, np.newaxis]
    transition_matrix = np.nan_to_num(transition_matrix)  # Replace NaN with 0

    # Convert to DataFrame with topic names
    transmat_df = pd.DataFrame(
        transition_matrix,
        index=[topic_titles[topic] for topic in selected_topic_columns],
        columns=[topic_titles[topic] for topic in selected_topic_columns]
    )

    # Plot using the academic style
    set_academic_style()
    
    # Create heatmap
    plot_academic_heatmap(
        transition_matrix=transmat_df,
        title="",  # Changed this line to remove the title
        output_dir=output_dir,
        file_name=file_name
    )

    return transmat_df

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

def calculate_transition_matrix(df, bank, selected_topic_columns, topic_titles):
    """
    Calculate the transition matrix for a specific central bank.

    Parameters:
    - df: DataFrame containing the data for all banks.
    - bank: string, name of the central bank.
    - selected_topic_columns: list of topic columns.
    - topic_titles: dictionary of topic titles.

    Returns:
    - DataFrame containing the transition matrix.
    """
    # Filter the dataset for the specific central bank
    bank_df = df[df['country'] == bank]

    # Remove any rows with missing dominant topic
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

def analyze_and_plot_transition_matrices(df, selected_topic_columns, topic_titles, output_dir):
    """
    Calculate and plot Markov transition matrices for each central bank in the dataset.

    Parameters:
    - df: DataFrame containing the data, with a 'country' column for central bank identification.
    - selected_topic_columns: list of topic columns.
    - topic_titles: dictionary of topic titles.
    - output_dir: string, directory where the plots will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Identify dominant topics for all data
    df['dominant_topic'] = df[selected_topic_columns].idxmax(axis=1)
    topic_mapping = {topic: idx for idx, topic in enumerate(selected_topic_columns)}
    df['dominant_topic_num'] = df['dominant_topic'].map(topic_mapping)

    # List of central banks to analyze
    central_banks = df['country'].unique()
    
    # For each central bank, calculate and plot the Markov transition matrix
    for bank in central_banks:
        transmat_df = calculate_transition_matrix(df, bank, selected_topic_columns, topic_titles)
        plot_transition_matrix(transmat_df, bank, output_dir, file_name="markov_transition_matrix")

def analyze_climate_transitions_between_banks(df, climate_topic_column, output_dir, file_name="climate_transitions_between_banks"):
    """
    Analyze and plot how climate-related topics transition between different central banks.
    
    Parameters:
    - df: DataFrame containing the speeches data
    - climate_topic_column: string, name of the climate-related topic column
    - output_dir: string, directory to save the output
    - file_name: string, base name for output files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for speeches with significant climate content (e.g., > 0.2 probability)
    climate_threshold = 0.2
    climate_speeches = df[df[climate_topic_column] > climate_threshold].copy()
    
    # Sort by date
    climate_speeches = climate_speeches.sort_values('date')
    
    # Create transition matrix between banks
    banks = climate_speeches['country'].unique()
    transition_matrix = pd.DataFrame(0, index=banks, columns=banks)
    
    # Calculate transitions
    for bank in banks:
        # Get all speeches after each speech from this bank
        bank_speeches = climate_speeches[climate_speeches['country'] == bank]
        
        for idx, speech in bank_speeches.iterrows():
            # Find the next climate speech within 30 days
            next_date = speech['date'] + timedelta(days=30)
            next_speeches = climate_speeches[
                (climate_speeches['date'] > speech['date']) & 
                (climate_speeches['date'] <= next_date)
            ]
            
            if not next_speeches.empty:
                next_bank = next_speeches.iloc[0]['country']
                transition_matrix.loc[bank, next_bank] += 1
    
    # Normalize the transition matrix
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix.div(row_sums, axis=0).fillna(0)
    
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(transition_matrix, 
                annot=True, 
                cmap="YlOrRd", 
                fmt='.2f',
                linewidths=0.5)
    plt.title("")
    plt.xlabel("Next Central Bank")
    plt.ylabel("Current Central Bank")
    plt.tight_layout()
    
    # Save the plots
    plt.savefig(os.path.join(output_dir, f"{file_name}.png"), format="png", dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"{file_name}.pdf"), format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    
    return transition_matrix

def analyze_topic_transitions_between_banks(df, topic_column, topic_name, output_dir, threshold=0.2, window_days=30, file_name=None):
    """
    Analyze and plot how any given topic transitions between different central banks.
    
    Parameters:
    - df: DataFrame containing the speeches data
    - topic_column: string, name of the topic column to analyze
    - topic_name: string, human-readable name of the topic for the plot title
    - output_dir: string, directory to save the output
    - threshold: float, minimum topic probability to consider (default: 0.2)
    - window_days: int, number of days to look ahead for transitions (default: 30)
    - file_name: string, base name for output files (default: derived from topic_name)
    
    Returns:
    - DataFrame containing the transition matrix
    """
    # Set default file name if none provided
    if file_name is None:
        file_name = f"{topic_name.lower().replace(' ', '_')}_transitions_between_banks"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for speeches with significant topic content
    topic_speeches = df[df[topic_column] > threshold].copy()
    
    # Sort by date
    topic_speeches = topic_speeches.sort_values('date')
    
    # Create transition matrix between banks
    banks = topic_speeches['country'].unique()
    transition_matrix = pd.DataFrame(0, index=banks, columns=banks)
    
    # Calculate transitions
    for bank in banks:
        bank_speeches = topic_speeches[topic_speeches['country'] == bank]
        
        for idx, speech in bank_speeches.iterrows():
            next_date = speech['date'] + timedelta(days=window_days)
            next_speeches = topic_speeches[
                (topic_speeches['date'] > speech['date']) & 
                (topic_speeches['date'] <= next_date)
            ]
            
            if not next_speeches.empty:
                next_bank = next_speeches.iloc[0]['country']
                transition_matrix.loc[bank, next_bank] += 1
    
    # Normalize the transition matrix
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix.div(row_sums, axis=0).fillna(0)
    
    # Plot the heatmap
    plot_academic_heatmap(
        transition_matrix=transition_matrix,
        title=f"{topic_name} Topic Transitions Between Central Banks",
        output_dir=output_dir,
        file_name=file_name,
        window_days=window_days
    )
    
    return transition_matrix

def analyze_all_topics_transitions(df, topic_columns, topic_titles, output_dir, window_days=30):
    """
    Analyze transitions between banks for all topics with academic styling.
    
    Parameters:
    - df: DataFrame containing the speeches data
    - topic_columns: list of topic column names
    - topic_titles: dictionary mapping topic columns to human-readable titles
    - output_dir: string, directory to save the outputs
    
    Returns:
    - Dictionary of transition matrices for each topic
    """
    transition_matrices = {}
    
    for topic_col in topic_columns:
        topic_name = topic_titles[topic_col]
        transition_matrices[topic_col] = analyze_topic_transitions_between_banks(
            df=df,
            topic_column=topic_col,
            topic_name=topic_name,
            output_dir=output_dir,
            window_days=window_days
        )
    
    create_topic_transitions_panel(
        df=df,
        topic_columns=topic_columns,
        topic_titles=topic_titles,
        output_dir=output_dir,
        window_days=window_days
    )
    
    return transition_matrices

def create_topic_transitions_panel(df, topic_columns, topic_titles, output_dir, window_days=30):
    """Create a panel figure showing transition matrices for all topics in a 2x5 layout."""
    set_academic_style()
    
    # Fixed layout: 2 columns, 5 rows
    n_cols = 3
    n_rows = 4
    
    # Create figure
    fig = plt.figure(figsize=(16, 22))
    
    # Create a grid with specific spacing
    gs = plt.GridSpec(n_rows, n_cols, figure=fig,
                     hspace=0.4,    # Vertical space between plots
                     wspace=0.15)   # Horizontal space between plots
    
    # Common colorbar settings
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    
    for idx, topic_col in enumerate(topic_columns, 0):
        topic_name = topic_titles[topic_col]
        row = idx // n_cols
        col = idx % n_cols
        
        # Calculate transition matrix
        topic_speeches = df[df[topic_col] > 0.2].copy()
        topic_speeches = topic_speeches.sort_values('date')
        banks = topic_speeches['country'].unique()
        transition_matrix = pd.DataFrame(0, index=banks, columns=banks)
        
        # Calculate transitions
        # ...existing transition calculation code...
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
        
        # Normalize
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix.div(row_sums, axis=0).fillna(0)
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        # Create heatmap with shared colorbar
        sns.heatmap(transition_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlBu_r',
                    center=0.5,
                    vmin=0,
                    vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar=True if idx == len(topic_columns)-1 else False,
                    cbar_ax=cbar_ax if idx == len(topic_columns)-1 else None,
                    ax=ax,
                    annot_kws={'size': 8})
        
        # Improve subplot titles and labels
        ax.set_title(topic_name, 
                    fontsize=14,
                    pad=10,
                    fontweight='bold')
        
        # Only show x-label for bottom plots
        if row == n_rows - 1:
            ax.set_xlabel('Next Central Bank', fontsize=10)
        else:
            ax.set_xlabel('')
            
        # Only show y-label for leftmost plots
        if col == 0:
            ax.set_ylabel('Current Central Bank', fontsize=10)
        else:
            ax.set_ylabel('')
        
        # Adjust tick labels
        ax.tick_params(labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Adjust colorbar
    cbar_ax.set_ylabel('Transition Probability', fontsize=14, rotation=270, labelpad=20)
    cbar_ax.tick_params(labelsize=12)
    
    # Add main title
    fig.suptitle('Topic Transitions Between Central Banks\n'
                 f'({window_days}-day window)',
                 fontsize=18,
                 y=0.95,
                 fontweight='bold')
    
    # Save with tight layout
    for ext in ['.png', '.pdf']:
        plt.savefig(
            os.path.join(output_dir, f"topic_transitions_panel{ext}"),
            dpi=300,
            bbox_inches='tight',
            metadata={'Creator': 'Academic Visualization Tool'}
        )
    plt.close()

def analyze_topic_transitions_with_window(df, topic_column, topic_name, output_dir, window_days, threshold=0.2, file_name=None):
    """
    Analyze topic transitions for a specific time window.
    
    Parameters:
    - df: DataFrame containing the speeches data
    - topic_column: string, name of the topic column to analyze
    - topic_name: string, human-readable name of the topic
    - output_dir: string, directory to save the output
    - window_days: int, number of days to look ahead for transitions
    - threshold: float, minimum topic probability to consider
    - file_name: string, optional custom filename
    
    Returns:
    - DataFrame containing the transition matrix
    """
    if file_name is None:
        file_name = f"{topic_name.lower().replace(' ', '_')}_transitions_{window_days}days"
    
    # Filter for speeches with significant topic content
    topic_speeches = df[df[topic_column] > threshold].copy()
    topic_speeches = topic_speeches.sort_values('date')
    
    banks = topic_speeches['country'].unique()
    transition_matrix = pd.DataFrame(0, index=banks, columns=banks)
    
    for bank in banks:
        bank_speeches = topic_speeches[topic_speeches['country'] == bank]
        for idx, speech in bank_speeches.iterrows():
            next_date = speech['date'] + timedelta(days=window_days)
            next_speeches = topic_speeches[
                (topic_speeches['date'] > speech['date']) & 
                (topic_speeches['date'] <= next_date)
            ]
            if not next_speeches.empty:
                next_bank = next_speeches.iloc[0]['country']
                transition_matrix.loc[bank, next_bank] += 1
    
    # Normalize
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix.div(row_sums, axis=0).fillna(0)
    
    # Plot
    plot_academic_heatmap(
        transition_matrix=transition_matrix,
        title=f"{topic_name} Topic Transitions\n({window_days}-day window)",
        output_dir=output_dir,
        file_name=file_name,
        window_days=window_days
    )
    
    return transition_matrix

def analyze_robustness_across_windows(df, topic_column, topic_name, output_dir, windows=[7, 14, 30, 60, 90], threshold=0.2):
    """
    Analyze topic transitions across multiple time windows.
    
    Parameters:
    - df: DataFrame containing the speeches data
    - topic_column: string, name of the topic column to analyze
    - topic_name: string, human-readable name of the topic
    - output_dir: string, directory to save the output
    - windows: list of integers, different time windows to analyze in days
    - threshold: float, minimum topic probability to consider
    
    Returns:
    - Dictionary of transition matrices for each time window
    """
    results = {}
    
    for window in windows:
        results[window] = analyze_topic_transitions_with_window(
            df=df,
            topic_column=topic_column,
            topic_name=topic_name,
            output_dir=output_dir,
            window_days=window,
            threshold=threshold
        )
    
    # Create comparison visualization
    compare_transition_matrices(results, topic_name, output_dir)
    
    return results

def compare_transition_matrices(matrices_dict, topic_name, output_dir):
    """
    Create a panel plot comparing transition matrices across different time windows.
    
    Parameters:
    - matrices_dict: Dictionary of transition matrices keyed by window size
    - topic_name: string, name of the topic
    - output_dir: string, output directory for plots
    """
    set_academic_style()
    
    n_matrices = len(matrices_dict)
    n_cols = min(3, n_matrices)
    n_rows = (n_matrices + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
    gs = plt.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.2)
    
    for idx, (window, matrix) in enumerate(sorted(matrices_dict.items())):
        row = idx // n_cols
        col = idx % n_cols
        
        ax = fig.add_subplot(gs[row, col])
        
        sns.heatmap(matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlBu_r',
                    center=0.5,
                    vmin=0,
                    vmax=1,
                    square=True,
                    linewidths=0.5,
                    ax=ax,
                    annot_kws={'size': 8})
        
        ax.set_title(f'{window}-day window',
                    fontsize=12,
                    pad=10)
        
        if row == n_rows - 1:
            ax.set_xlabel('Next Central Bank', fontsize=10)
        if col == 0:
            ax.set_ylabel('Current Central Bank', fontsize=10)
            
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    fig.suptitle(f'{topic_name} Topic Transitions\nRobustness Across Time Windows',
                fontsize=16,
                y=0.95)
    
    # Save the comparison plot
    for ext in ['.png', '.pdf']:
        plt.savefig(
            os.path.join(output_dir, f"{topic_name.lower().replace(' ', '_')}_transitions_comparison{ext}"),
            dpi=300,
            bbox_inches='tight',
            metadata={'Creator': 'Academic Visualization Tool'}
        )
    plt.close()


