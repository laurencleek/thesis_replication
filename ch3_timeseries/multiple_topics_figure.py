import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import os
from stargazer.stargazer import Stargazer
from statsmodels.nonparametric.smoothers_lowess import lowess

def create_multi_topic_plot(df, topic_titles, bank_names, output):
    """
    multi-topic plot. Input is the dataframe of numeric already processed topics,
    the topics, the banks and the output folder. The output is a saved pdf and png in
    the specified output folder.
    """
    # Set font and style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = '#cccccc'

    num_topics = len(topic_titles)
    ncols = 3
    nrows = (num_topics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, nrows * 4))
    axes = axes.flatten()
    
    handles = []
    colors = sns.color_palette("deep")
    
    for idx, (topic, title) in enumerate(topic_titles.items()):
        topic_df = df[['yq', 'banks', topic]].copy()
        topic_pivot = topic_df.pivot(index='yq', columns='banks', values=topic)
        
        ax = axes[idx]
        year_quarters = topic_pivot.index
        
        # Extract years from Timestamp objects
        years = [yq.year for yq in year_quarters]
        
        ymin, ymax = float('inf'), float('-inf')
        
        for bank_idx, col in enumerate(topic_pivot.columns):
            x = np.arange(len(topic_pivot))
            y = topic_pivot[col].values
            
            # Remove NaN values before smoothing
            valid_mask = ~np.isnan(y)
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            if len(x_valid) > 1:  # Ensure we have at least 2 points for smoothing
                # Calculate smoothed line
                smooth = lowess(y_valid, x_valid, frac=0.6, it=5)
                smooth_x, smooth_y = smooth[:, 0], smooth[:, 1]
                
                # Update ymin and ymax based on smoothed line
                ymin = min(ymin, np.min(smooth_y))
                ymax = max(ymax, np.max(smooth_y))
                
                # Plot scatter points with lower opacity
                sns.scatterplot(x=x_valid, y=y_valid, ax=ax, color=colors[bank_idx], alpha=0.3)
                
                # Plot smoothed line
                line, = ax.plot(smooth_x, smooth_y, color=colors[bank_idx])
                
                if idx == 0:
                    handles.append(line)
            else:
                print(f"Warning: Not enough valid data points for {col} in topic {title}")
        
        # Set y-axis limits with a small margin
        if ymin != float('inf') and ymax != float('-inf'):
            margin = (ymax - ymin) * 0.1
            ax.set_ylim(max(0, ymin - margin), ymax + margin)
        
        # Enhanced title and label styling
        ax.set_title(title, pad=15, fontname='Times New Roman', fontsize=16, fontweight='bold')
        ax.set_ylabel('Proportion (%)', fontname='Times New Roman', fontsize=13)
        
        # Set x-axis ticks and labels to show only 5 years
        unique_years = sorted(set(years))
        num_years = len(unique_years)
        if num_years >= 5:
            tick_indices = np.linspace(0, num_years - 1, 5, dtype=int)
            tick_years = [unique_years[i] for i in tick_indices]
            tick_positions = [years.index(year) for year in tick_years]
        else:
            tick_years = unique_years
            tick_positions = [years.index(year) for year in tick_years]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_years, rotation=0, ha='center', fontname='Times New Roman')
        
        # Enhanced tick label styling
        ax.tick_params(axis='both', labelsize=12)
        
        # Improve grid appearance
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Remove empty subplots
    for i in range(num_topics, nrows * ncols):
        fig.delaxes(axes[i])
    
    # Enhanced legend styling
    legend_ax = fig.add_subplot(nrows, ncols, (nrows * ncols - 1, nrows * ncols))
    legend_ax.axis('off')
    legend = legend_ax.legend(handles, bank_names, 
                            title="Central Banks", 
                            loc='center', 
                            fontsize=14,
                            title_fontsize=16,
                            ncol=2,
                            frameon=True,
                            edgecolor='none',
                            fancybox=True,
                            shadow=False)
    
    # Set font for legend
    plt.setp(legend.get_title(), fontname='Times New Roman', fontweight='bold')
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, 
                       wspace=0.2,
                       right=0.88)
    
    # Save figures
    pdf_path = os.path.join(output, "multiple_topics_plot.pdf")
    png_path = os.path.join(output, "multiple_topics_plot.png")

    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    plt.close(fig)
