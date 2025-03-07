import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import seaborn as sns
import os

def analyze_interaction(df, topic, circumstance, central_banks, time_col='yq'):
    """
    Modified to handle different time columns (yq for quarterly, yh for half-yearly)
    """
    results = {}
    for bank in central_banks:
        bank_data = df[df['central_bank'] == bank].sort_values(time_col)
        ecb_data = df[df['central_bank'] == 'european central bank'].sort_values(time_col)
        
        required_columns = [topic, circumstance, 'central_bank', time_col]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns for topic '{topic}', circumstance '{circumstance}', central bank '{bank}': {missing_columns}")
            results[bank] = 'Missing Columns'
            continue
        
        if bank_data.empty or ecb_data.empty:
            print(f"No data for central bank '{bank}' or ECB")
            results[bank] = 'No Data'
            continue
        
        bank_data[f'{topic}_lag'] = bank_data[topic].shift(1)
        
        merged_data = pd.merge(ecb_data[[time_col, topic, circumstance]], 
                             bank_data[[time_col, f'{topic}_lag']], 
                             on=time_col, 
                             suffixes=('', f'_{bank}'))
        
        valid_data = merged_data.dropna()
        
        if valid_data.empty:
            print(f"No valid data after merging for topic '{topic}', circumstance '{circumstance}', central bank '{bank}'")
            results[bank] = 'No Valid Data'
            continue
        
        formula = f"{topic} ~ {circumstance} + {topic}_lag + {circumstance}:{topic}_lag"
        
        try:
            model = ols(formula, data=valid_data).fit()
            interaction_coef = model.params[f'{circumstance}:{topic}_lag']
            interaction_pvalue = model.pvalues[f'{circumstance}:{topic}_lag']
            
            direction = '+' if interaction_coef > 0 else '-'
            significance = 'S' if interaction_pvalue < 0.05 else 'NS'
            
            results[bank] = f"{direction}{significance}"
        except Exception as e:
            print(f"Error analyzing topic '{topic}', circumstance '{circumstance}', central bank '{bank}': {str(e)}")
            results[bank] = 'Error'
    
    return results

def generate_grid(df, topic_bank_mapping, circumstances):
    """
    Detect time column and pass it to analyze_interaction
    """
    time_col = 'yq' if 'yq' in df.columns else 'yh'
    grid = {}
    for topic, central_banks in topic_bank_mapping.items():
        grid[topic] = {}
        for circumstance in circumstances:
            grid[topic][circumstance] = analyze_interaction(df, topic, circumstance, central_banks, time_col)
    return grid

def print_grid(grid):
    for topic, circumstances in grid.items():
        print(f"\nTopic: {topic}")
        banks = list(next(iter(circumstances.values())).keys())
        print("Circumstance    | " + " | ".join(banks))
        print("-" * (18 + 15 * len(banks)))
        for circumstance, results in circumstances.items():
            print(f"{circumstance:<15} | " + " | ".join(results.get(bank, 'N/A').center(13) for bank in banks))

def print_data_info(df):
    print("\nDataset Information:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")
    print("\nUnique values in 'central_bank' column:")
    print(df['central_bank'].unique())

def create_color_grid(grid):
    topics = list(grid.keys())
    circumstances = list(grid[topics[0]].keys())
    
    # Calculate optimal grid layout
    n_topics = len(topics)
    n_cols = min(3, n_topics)  # Maximum 3 topics per row
    n_rows = (n_topics + n_cols - 1) // n_cols
    
    # Create lists to store all unique banks and their data per topic
    all_data = []
    all_banks = []
    max_banks = 0
    
    for topic in topics:
        topic_banks = list(grid[topic][circumstances[0]].keys())
        all_banks.append((topic, topic_banks))
        max_banks = max(max_banks, len(topic_banks))
        
        topic_grid = np.zeros((len(circumstances), len(topic_banks)))
        for i, circumstance in enumerate(circumstances):
            for j, bank in enumerate(topic_banks):
                result = grid[topic][circumstance][bank]
                if result == '+S':
                    topic_grid[i, j] = 2
                elif result == '+NS':
                    topic_grid[i, j] = 1
                elif result == '-NS':
                    topic_grid[i, j] = -1
                elif result == '-S':
                    topic_grid[i, j] = -2
        all_data.append(topic_grid)
    
    return all_data, circumstances, all_banks, (n_rows, n_cols)

def plot_color_grid(all_data, circumstances, all_banks, grid_shape):
    n_rows, n_cols = grid_shape
    
    # Set figure style with updated approach
    plt.style.use('default')
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.titlesize': 12,  # Reduced from 14 to 12
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (12, 12),
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.color': '0.9',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5
    })
    
    # Create figure with white background
    fig = plt.figure(facecolor='white')
    
    # Create GridSpec with reduced horizontal space
    gs = plt.GridSpec(n_rows, n_cols, figure=fig,
                     hspace=0.4,
                     wspace=0.15)
    
    # Define exact colors for each significance level
    color_dict = {
        2: '#2166AC',  # Strong blue for significant positive
        1: '#92C5DE',  # Light blue for non-significant positive
        -1: '#F4A582', # Light red for non-significant negative
        -2: '#B2182B'  # Strong red for significant negative
    }
    
    # Create custom colormap from these colors
    colors = [color_dict[-2], color_dict[-1], 'white', color_dict[1], color_dict[2]]
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-2, vmax=2)
    
    # Map circumstances to readable names
    circumstance_labels = {
        'google_trends_value': 'Public Attention',
        'gdp_real_growth': 'GDP Growth',
        'treu_ecb': 'Trust in ECB',
        'hicp': 'Inflation'
    }
    
    # Simplified bank names to just country codes
    bank_to_country = {
        'deutsche bundesbank': 'DE',
        'bank of italy': 'IT',
        'bank of france': 'FR',
        'bank of spain': 'ES',
        'netherlands bank': 'NL'
    }
    
    def format_topic_title(title):
        """Format topic title to be more readable"""
        words = title.split('_')
        # Join with newlines for better spacing
        return ' '.join(words).title()
    
    for idx, ((topic_grid, (topic, banks)), subplot_idx) in enumerate(zip(zip(all_data, all_banks), range(len(all_data)))):
        row = subplot_idx // n_cols
        col = subplot_idx % n_cols
        
        ax = fig.add_subplot(gs[row, col])
        
        # Convert bank names to simplified labels
        bank_labels = [bank_to_country[bank] for bank in banks]
        
        # Plot heatmap with exact colors
        sns.heatmap(topic_grid, 
                   cmap=cmap,
                   norm=norm,
                   center=0,
                   cbar=False,
                   xticklabels=bank_labels,
                   yticklabels=[circumstance_labels[c] for c in circumstances] if col == 0 else False,
                   ax=ax,
                   square=True,
                   linewidths=1,
                   linecolor='white')
        
        # Add topic title 
        title = format_topic_title(topic)
        ax.set_title(title, pad=10, fontsize=10, fontweight='bold')
        
        # Set x-axis labels horizontal and centered
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
        
        # Ensure all cells have proper coloring by explicitly setting the color
        for i in range(topic_grid.shape[0]):
            for j in range(topic_grid.shape[1]):
                value = topic_grid[i, j]
                text = {
                    2: '+S',
                    1: '+NS',
                    -1: '-NS',
                    -2: '-S'
                }.get(value, 'N/A')
                
                # Get exact color from dictionary
                if value in color_dict:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color_dict[value]))
                
                # Use white text for better contrast on all colored backgrounds
                ax.text(j + 0.5, i + 0.5, text,
                       ha='center', va='center',
                       fontsize=10,
                       weight='bold',
                       color='white')
    
    # Update legend to use exact colors
    legend_ax = fig.add_subplot(gs[-1, -1])
    legend_ax.axis('off')
    
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=color_dict[2], label='Significant Positive (+S)'),
        plt.Rectangle((0,0), 1, 1, facecolor=color_dict[1], label='Non-significant Positive (+NS)'),
        plt.Rectangle((0,0), 1, 1, facecolor=color_dict[-1], label='Non-significant Negative (-NS)'),
        plt.Rectangle((0,0), 1, 1, facecolor=color_dict[-2], label='Significant Negative (-S)')
    ]
    
    legend = legend_ax.legend(handles=legend_elements,
                             title='Significance Levels',
                             loc='center',
                             frameon=True,
                             edgecolor='black',
                             fancybox=False)
    
    legend.get_frame().set_linewidth(1)
    
    # Remove main title and adjust layout
    plt.tight_layout()

def save_color_grid(grid, base_folder):
    all_data, circumstances, all_banks, grid_shape = create_color_grid(grid)
    plot_color_grid(all_data, circumstances, all_banks, grid_shape)
    
    # Create the folder structure
    pdf_folder = os.path.join(base_folder, 'pdf')
    png_folder = os.path.join(base_folder, 'png')
    os.makedirs(pdf_folder, exist_ok=True)
    os.makedirs(png_folder, exist_ok=True)
    
    # Save as PDF
    pdf_path = os.path.join(pdf_folder, 'interaction_grid.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Color grid saved as PDF: {pdf_path}")
    
    # Save as PNG
    png_path = os.path.join(png_folder, 'interaction_grid.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Color grid saved as PNG: {png_path}")
    
    plt.close()

def main():
    # Set up file paths
    stem = "C:/Users/lclee/OneDrive - Istituto Universitario Europeo/PhD/two_thirds_submission/Github_replication_files_by_paper/Paper_1"
    output = f"{stem}/output/"
    
    # Create output directories
    figures_path = os.path.join(output, 'figures_paper')
    os.makedirs(figures_path, exist_ok=True)
    
    # Load actual data
    df = pd.read_stata(f"{output}/timeseries.dta")
    df['yq'] = pd.to_datetime(df['quarter'], format='%Y-%m-%d')
    
    # Define topic mapping with actual central banks
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
        'Digital_Finance_Innovation': ['deutsche bundesbank', 'bank of italy', 'bank of france', 'bank of spain', 'netherlands bank']
    }
    
    # Define circumstances to analyze
    circumstances = ['google_trends_value', 'gdp_real_growth', 'treu_ecb', 'hicp']
    
    # Print data info for verification
    print_data_info(df)
    
    # Generate the grid
    grid = generate_grid(df, topic_bank_mapping, circumstances)
    
    # Print text version of the grid
    print_grid(grid)
    
    # Save visualization
    save_color_grid(grid, figures_path)
    
    print("Analysis completed. Check the figures_paper folder for the output.")

if __name__ == "__main__":
    main()