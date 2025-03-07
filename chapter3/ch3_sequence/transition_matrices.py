import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import matplotlib.font_manager as fm

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

def plot_topic_transition_matrix_robustness(df, topic_titles, output_dir, file_name="transition_matrix_robustness"):
    """
    Create and save a panel of transition matrix heatmaps for different time windows.
    """
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define time windows in days
    windows = [7, 14, 30, 60, 90]
    
    # Create figure with subplot grid - adjusted for better spacing
    fig = plt.figure(figsize=(20, 24))  # Increased width for better spacing
    gs = gridspec.GridSpec(3, 2, figure=fig, 
                         wspace=0.4,   # Increased horizontal space
                         hspace=0.4)   # Keep vertical space the same
    
    # Ensure DataFrame has datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df['date'])

    # Handle both dictionary and list inputs for topic_titles
    if isinstance(topic_titles, dict):
        selected_topic_columns = list(topic_titles.keys())
        topic_names = topic_titles
    else:
        selected_topic_columns = topic_titles
        # Use topic_titles_new for mapping if available
        topic_names = {col: col.replace('_', ' ') for col in topic_titles}
    
    for idx, window in enumerate(windows):
        # Calculate row and column position in grid - updated for new layout
        row = idx // 2
        col = idx % 2
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        # Create copy of DataFrame for this window
        df_window = df.copy()
        
        # Calculate dominant topic for current window
        df_window['dominant_topic'] = df_window[selected_topic_columns].idxmax(axis=1)
        
        # Create forward-looking window groups
        transitions = []
        dates = sorted(df_window.index.unique())
        
        for i, date in enumerate(dates[:-1]):  # Exclude last date as it won't have a next window
            try:
                current_topic = df_window.loc[date, 'dominant_topic']
                if isinstance(current_topic, pd.Series):
                    current_topic = current_topic.iloc[0]
                
                # Get next window dates
                window_end = date + timedelta(days=window)
                next_window_mask = (df_window.index > date) & (df_window.index <= window_end)
                next_window_data = df_window[next_window_mask]['dominant_topic']
                
                if not next_window_data.empty:
                    # Get the most common topic in the window
                    next_topic = next_window_data.value_counts().index[0]
                    transitions.append((current_topic, next_topic))
                    
            except Exception as e:
                print(f"Error processing date {date}: {str(e)}")
                continue
        
        if not transitions:
            print(f"No transitions found for {window}-day window")
            continue
        
        # Create transition matrix
        try:
            transition_df = pd.DataFrame(transitions, columns=['current', 'next'])
            transition_matrix = pd.crosstab(
                transition_df['current'],
                transition_df['next'],
                normalize='index'
            )
            
            # Ensure all topics are represented
            all_topics = sorted(set(selected_topic_columns))
            for topic in all_topics:
                if topic not in transition_matrix.index:
                    transition_matrix.loc[topic] = 0
                if topic not in transition_matrix.columns:
                    transition_matrix[topic] = 0
            
            # Sort index and columns
            transition_matrix = transition_matrix.reindex(index=all_topics, columns=all_topics, fill_value=0)
            
            # Map the indices back to topic names
            transition_matrix.index = [topic_names[t] for t in transition_matrix.index]
            transition_matrix.columns = [topic_names[t] for t in transition_matrix.columns]
            
            # Plot heatmap with consistent styling
            sns.heatmap(
                transition_matrix,
                annot=True,
                cmap="RdYlBu_r",  # Changed to match other robustness checks
                center=0.5,       # Center the colormap
                vmin=0,
                vmax=1,
                linewidths=0.5,
                ax=ax,
                fmt='.2f',
                cbar=False,
                annot_kws={'size': 8, 'family': 'Times New Roman'}  # Reduced font size for annotations
            )
            
            ax.set_title(f"{window}-Day Window", 
                        pad=10, 
                        fontsize=14, 
                        fontfamily='Times New Roman',
                        fontweight='bold')
            
            if col == 0:
                ax.set_ylabel("Current Topic", 
                            fontsize=12, 
                            fontfamily='Times New Roman')
            if row == 2:  # Updated for new layout
                ax.set_xlabel("Next Topic", 
                            fontsize=12, 
                            fontfamily='Times New Roman')
            
            # Improved tick label formatting with more space
            ax.tick_params(axis='both', which='major', labelsize=9)
            plt.setp(ax.get_xticklabels(), 
                    rotation=45, 
                    ha='right', 
                    rotation_mode='anchor',
                    fontfamily='Times New Roman')
            plt.setp(ax.get_yticklabels(), 
                    rotation=0, 
                    fontfamily='Times New Roman')
            
            # Add padding around the subplot
            ax.set_xticklabels(ax.get_xticklabels(), ha='right', rotation=45)
            ax.set_yticklabels(ax.get_yticklabels(), va='center')
            
        except Exception as e:
            print(f"Error creating transition matrix for {window}-day window: {str(e)}")
            continue

    # Add colorbar with consistent styling
    plt.subplots_adjust(right=0.85, bottom=0.1)  # Adjusted margins
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel('Transition Probability', 
                      fontsize=12, 
                      fontfamily='Times New Roman',
                      rotation=270,
                      labelpad=25)
    
    # Add overall title
    fig.suptitle("Topic Transition Matrices - Time Window Robustness Check", 
                 fontsize=16, 
                 y=0.95, 
                 fontfamily='Times New Roman',
                 fontweight='bold')
    
    # Save plots with tight layout
    plt.savefig(os.path.join(output_dir, f"{file_name}.png"), 
                format="png", 
                dpi=600, 
                bbox_inches='tight',
                pad_inches=0.5)  # Added padding
    plt.savefig(os.path.join(output_dir, f"{file_name}.pdf"), 
                format="pdf", 
                dpi=600, 
                bbox_inches='tight',
                pad_inches=0.5)  # Added padding
    plt.close()