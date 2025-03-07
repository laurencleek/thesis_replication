# descriptive_figures.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime
import matplotlib.ticker as ticker
from wordcloud import WordCloud

def is_date(variable):
    return isinstance(variable, (date, datetime))

def plot_stacked_speeches_over_time(all_speeches, speeches_count_CB_EU, full_subfolder_path):
    # Prepare the data
    speeches_by_year_and_bank = all_speeches.groupby(['year', 'central_bank']).size().unstack(fill_value=0)
    
    relevant_banks = speeches_count_CB_EU['central_bank'].tolist()
    speeches_by_year_and_bank = speeches_by_year_and_bank[relevant_banks]
    
    if 1987 in speeches_by_year_and_bank.index:
        speeches_by_year_and_bank = speeches_by_year_and_bank.drop(1987)
    
    # Set figure size to 12cm width, with a golden ratio for height
    width_cm = 12
    height_cm = width_cm / 1.618  # Golden ratio
    width_inch = width_cm / 2.54
    height_inch = height_cm / 2.54
    
    plt.figure(figsize=(width_inch, height_inch), dpi=300)
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    sns.set_style("whitegrid", {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    
    # Create the stacked bar plot
    ax = speeches_by_year_and_bank.plot(kind='bar', stacked=True, width=0.8)
    
    plt.ylabel("Number of Speeches", fontsize=8, fontweight='bold')
    plt.xlabel("Year", fontsize=8, fontweight='bold')
    plt.title("Number of Speeches per Central Bank Over Time", fontsize=10, fontweight='bold', pad=10)
    
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.yticks(fontsize=6)
    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#CCCCCC')
    
    # Add legend inside the plot in the top-left corner with improved readability
    plt.legend(title="Central Bank", loc='upper left', fontsize=6, title_fontsize=7, 
               bbox_to_anchor=(0.02, 0.98), ncol=2)  # Adjust ncol as needed
    
    plt.tight_layout()
    
    # Save as PDF
    file_path = os.path.join(full_subfolder_path, 'figure_a1.pdf')
    plt.savefig(file_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure saved as PDF at {file_path}")
    
    plt.show()


# wordcloud
def create_wordclouds(model, folder_name, num_topics=45):
    def create_wordcloud(model, topic):
        words_weights = model.get_topic(topic)
        print(f"Words and weights for Topic {topic}: {words_weights}")
        
        if not words_weights or isinstance(words_weights, bool):
            print(f"No valid data for Topic {topic}")
            return None
        
        word_dict = dict(words_weights)
        
        try:
            wordcloud = WordCloud(
                width=300, 
                height=150, 
                background_color='white',
                min_font_size=4,
                colormap='Blues',  # Professional blue colormap
                prefer_horizontal=0.7,
                relative_scaling=0.5,
                max_words=50
            ).generate_from_frequencies(word_dict)
            return wordcloud
        except ValueError as e:
            print(f"Error creating wordcloud for Topic {topic}: {e}")
            return None

    os.makedirs(folder_name, exist_ok=True)

    # Set the font style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    sns.set_style("whitegrid", {'font.family': 'serif', 'font.serif': 'Times New Roman'})

    # Create two separate figures for topics 0-22 and 23-44
    for part in range(2):
        start_topic = part * 23
        end_topic = start_topic + 23 if part == 0 else num_topics
        current_num_topics = end_topic - start_topic

        # Calculate layout with 4 columns
        num_cols = 4
        num_rows = (current_num_topics + num_cols - 1) // num_cols

        # Adjust figure size for 4 columns
        width_cm = 12
        height_cm = width_cm * (num_rows / num_cols) * 0.8  # Reduced height factor
        width_inch = width_cm / 2.54
        height_inch = height_cm / 2.54

        # Create figure
        fig = plt.figure(figsize=(width_inch, height_inch), dpi=300)

        # Even tighter layout with minimal vertical spacing
        fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, 
                          wspace=0.05, hspace=0.05)  # Reduced vertical spacing

        # Create wordclouds for this part
        for i in range(start_topic, end_topic):
            ax = fig.add_subplot(num_rows, num_cols, i - start_topic + 1)
            wordcloud = create_wordcloud(model, i)
            if wordcloud:
                ax.imshow(wordcloud, interpolation='bilinear')
            else:
                ax.text(0.5, 0.5, f'No data for Topic {i}',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=4)
            ax.axis('off')
            ax.set_title(f'Topic {i}', fontsize=6, fontweight='bold', pad=0)  # Reduced padding

        # Save each part separately
        file_path = os.path.join(folder_name, f'figure_a2_part{part + 1}.pdf')
        plt.savefig(file_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Figure part {part + 1} saved as PDF at {file_path}")
        plt.close()