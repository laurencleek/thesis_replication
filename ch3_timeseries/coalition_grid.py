import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import os

def analyze_coalition_interaction(df, topic, circumstance, time_col='yq'):
    """
    Analyze interactions between coalition stance, circumstances, and topic coverage
    """
    try:
        # Define coalition groups
        dovish_banks = ['bank of france', 'bank of italy', 'bank of spain']
        hawkish_banks = ['deutsche bundesbank', 'netherlands bank']
        
        # Create coalition dummy (1 for dovish, 0 for hawkish)
        df = df.copy()
        df['coalition'] = df['central_bank'].str.lower().isin(dovish_banks).astype(int)
        
        # Calculate coalition means per time period
        coalition_data = df.groupby([time_col, 'coalition'])[topic].mean().reset_index()
        
        # Get ECB data
        ecb_data = df[df['central_bank'] == 'european central bank'][[time_col, topic, circumstance]]
        
        # Calculate lags for both coalitions
        coalition_pivoted = coalition_data.pivot(index=time_col, 
                                               columns='coalition', 
                                               values=topic).reset_index()
        coalition_pivoted.columns = [time_col, 'hawkish_value', 'dovish_value']
        
        # Add lags
        coalition_pivoted['hawkish_lag'] = coalition_pivoted['hawkish_value'].shift(1)
        coalition_pivoted['dovish_lag'] = coalition_pivoted['dovish_value'].shift(1)
        
        # Merge with ECB data
        merged_data = ecb_data.merge(coalition_pivoted, on=time_col, how='left')
        
        # Create interaction terms
        merged_data['hawkish_interaction'] = merged_data['hawkish_lag'] * merged_data[circumstance]
        merged_data['dovish_interaction'] = merged_data['dovish_lag'] * merged_data[circumstance]
        
        # Run regressions for each coalition
        results = {}
        
        # Hawkish regression
        hawkish_formula = f"{topic} ~ {circumstance} + hawkish_lag + hawkish_interaction"
        hawkish_model = ols(hawkish_formula, data=merged_data.dropna()).fit()
        
        # Dovish regression
        dovish_formula = f"{topic} ~ {circumstance} + dovish_lag + dovish_interaction"
        dovish_model = ols(dovish_formula, data=merged_data.dropna()).fit()
        
        # Store results
        for coalition, model in [('Hawkish', hawkish_model), ('Dovish', dovish_model)]:
            interaction_var = f"{coalition.lower()}_interaction"
            coef = model.params[interaction_var]
            pvalue = model.pvalues[interaction_var]
            
            direction = '+' if coef > 0 else '-'
            significance = 'S' if pvalue < 0.05 else 'NS'
            results[coalition] = f"{direction}{significance}"
        
        return results
        
    except Exception as e:
        print(f"Error in coalition analysis for topic {topic}, circumstance {circumstance}: {e}")
        return {'Hawkish': 'Error', 'Dovish': 'Error'}

def generate_coalition_grid(df, topics, circumstances, time_col='yq'):
    """
    Generate interaction grid for coalition analysis
    """
    grid = {}
    for topic in topics:
        grid[topic] = {}
        for circumstance in circumstances:
            if circumstance in df.columns:
                grid[topic][circumstance] = analyze_coalition_interaction(
                    df, topic, circumstance, time_col
                )
            else:
                print(f"Circumstance {circumstance} not found in data")
                grid[topic][circumstance] = {'Hawkish': 'NA', 'Dovish': 'NA'}
    return grid

def plot_coalition_grid(grid, output_path, topic_titles=None):
    """
    Create visualization of coalition interaction grid
    """
    # Setup
    plt.style.use('default')
    sns.set_style("white")
    
    # Define colors and result mapping
    color_mapping = {
        '+S': '#2166AC',   # Strong blue
        '+NS': '#92C5DE',  # Light blue
        '-NS': '#F4A582',  # Light red
        '-S': '#B2182B',   # Strong red
        'NA': '#FFFFFF',   # White
        'Error': '#CCCCCC' # Gray
    }
    
    # Define circumstance mapping
    circumstance_mapping = {
        'gdp_real_growth': 'GDP Growth',
        'google_trends_value': 'Salience',
        'hicp': 'Inflation',
        'treu_ecb': 'Trust ECB'
    }
    
    # Prepare data for plotting
    data = []
    for topic, circs in grid.items():
        display_topic = topic_titles.get(topic, topic) if topic_titles else topic
        for circ, results in circs.items():
            mapped_circ = circumstance_mapping.get(circ, circ)
            for coalition, result in results.items():
                data.append({
                    'Topic': display_topic,
                    'Coalition': coalition,
                    'Circumstance': mapped_circ,
                    'Result': result
                })
    
    # Convert to DataFrame
    df_plot = pd.DataFrame(data)
    
    # Calculate dimensions
    n_topics = len(grid)
    n_circs = len(next(iter(grid.values())))
    figsize = (n_circs * 1.5, n_topics * 0.8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create pivot table for plotting
    pivot_data = df_plot.pivot(
        index=['Topic', 'Coalition'],
        columns='Circumstance',
        values='Result'
    )
    
    # Create numerical matrix for plotting
    result_to_num = {'+S': 2, '+NS': 1, '-NS': -1, '-S': -2, 'NA': 0, 'Error': 0}
    plot_matrix = pivot_data.map(lambda x: result_to_num.get(x, 0))
    
    # Create custom colormap
    colors = ['#B2182B', '#F4A582', '#FFFFFF', '#92C5DE', '#2166AC']
    n_colors = 256
    custom_cmap = sns.blend_palette(colors, n_colors=n_colors, as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(
        plot_matrix,
        cmap=custom_cmap,
        center=0,
        vmin=-2,
        vmax=2,
        cbar=False,
        ax=ax,
        linewidths=1,
        linecolor='white'
    )
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.iloc[i, j]
            color = 'white' if val in ['+S', '-S'] else 'black'
            ax.text(j + 0.5, i + 0.5, val,
                   ha='center', va='center',
                   color=color,
                   fontsize=10,
                   fontweight='bold')
    
    # Customize axes
    ax.set_xticklabels(pivot_data.columns, rotation=45, ha='right')
    
    # Create custom y-axis labels with different colors for coalition types
    ylabels = []
    ycolors = []
    for idx in pivot_data.index:
        topic, coalition = idx
        if coalition == 'Dovish':
            ylabels.append(f"{topic} (Dovish)")
            ycolors.append('#4A90E2')  # Blue for Dovish
        else:
            ylabels.append(f"{topic} (Hawkish)")
            ycolors.append('#E24A4A')  # Red for Hawkish
    
    ax.set_yticklabels(ylabels, rotation=0)
    
    # Apply colors to y-axis labels
    for ticklabel, color in zip(ax.get_yticklabels(), ycolors):
        ticklabel.set_color(color)
    
    # Remove axes labels and title
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'coalition_interaction_grid.pdf'), 
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_path, 'coalition_interaction_grid.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

# Example usage in main script:
