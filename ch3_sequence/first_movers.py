import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import timedelta

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
                valid_first_movers_list.append(topic_first_mover)
            else:
                date_index += 1

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

        # Sort topics by earliest date
        topic_order = valid_first_movers_df.sort_values('earliest_date')['topic'].unique()
        
        if selected_NCBs is None:
            selected_NCBs = ['DEU', 'FRA', 'NLD', 'ESP', 'ITA']

        # Assign distinct colors to the selected NCBs
        colors_list = list(mcolors.TABLEAU_COLORS.values())
        ncb_color_map = {ncb: colors_list[i % len(colors_list)] for i, ncb in enumerate(selected_NCBs)}

        labels_seen = set()

        plt.rcParams["font.family"] = "Times New Roman"

        # Create a mapping of topics to their positions
        topic_positions = {topic: i for i, topic in enumerate(topic_order)}

        # Plot each first mover
        for idx, row in valid_first_movers_df.iterrows():
            first_mover_countries = row['first_mover_countries']
            topic_position = topic_positions[row['topic']]
            for country in first_mover_countries:
                if country == 'ECB':
                    label = 'ECB' if 'ECB' not in labels_seen else "_nolegend_"
                    labels_seen.add('ECB')
                    plt.scatter(row['earliest_date'], topic_position, label=label, color='black', s=100, edgecolors='k')
                elif country in selected_NCBs:
                    color = ncb_color_map.get(country, 'gray')
                    label = country if country not in labels_seen else "_nolegend_"
                    labels_seen.add(country)
                    plt.scatter(row['earliest_date'], topic_position, label=label, color=color, s=100, edgecolors='k')
                else:
                    continue

        plt.xlabel('', fontsize=14)
        plt.ylabel('', fontsize=14)
        plt.yticks(range(len(topic_order)), topic_order, fontsize=14)
        plt.legend(title="First Movers", fontsize=12, title_fontsize=12)
        plt.xticks(rotation=45, fontsize=14)
        plt.tight_layout()

        # Save in output directory
        plt.savefig(os.path.join(output, "figure_first_movers.png"), format="png", dpi=300)
        plt.savefig(os.path.join(output, "figure_first_movers.pdf"), format="pdf", dpi=300)
        plt.show()
    else:
        print("No valid first movers to visualize.")

    return valid_first_movers_df, first_mover_type_counts