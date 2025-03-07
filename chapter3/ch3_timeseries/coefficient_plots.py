import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import os
from stargazer.stargazer import Stargazer
from statsmodels.nonparametric.smoothers_lowess import lowess


def plot_coefficients(results_df, bank_labels, output, topic_labels):
    """
    Main coefficient plot figure with increased font sizes and custom topic labels.
    """
    plt.figure(figsize=(12, 15))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    sns.set_style("whitegrid", {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    
    topics = results_df['topic'].unique()
    # Convert dictionary labels to ordered list matching topics
    topic_label_list = [topic_labels[topic] for topic in topics]
    
    banks = results_df['bank'].unique()
    
    # Single light grey color for alternating background
    light_grey = '#f5f5f5'
    
    for i, topic in enumerate(topics):
        topic_data = results_df[results_df['topic'] == topic]
        
        # Alternate between white and light grey
        if i % 2 == 0:
            plt.axhspan(i - 0.5, i + 0.5, facecolor=light_grey, alpha=0.5)
        
        for j, bank in enumerate(banks):
            bank_data = topic_data[topic_data['bank'] == bank]
            
            y_pos = i + j / len(banks) - 0.5 + 1 / (2 * len(banks))
            plt.errorbar(x=bank_data['coefficient'], y=y_pos,
                         xerr=bank_data['std_error'],
                         fmt='o', capsize=5, capthick=2, color=f'C{j}', label=bank if i == 0 else "")
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    
    # Center y-axis labels with topic_labels
    y_ticks = list(range(len(topics)))
    topic_labels = topic_labels if len(topic_labels) == len(topics) else topics 
    plt.yticks(y_ticks, topic_label_list, fontsize=16)  
    
    plt.xlabel('Coefficient Estimate', fontsize=16)  # Increased font size
    plt.ylabel('', fontsize=16)
    plt.title('', fontsize=16)
    
    plt.legend(title='National Central Bank', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
    
    # Adjust plot limits
    plt.ylim(-0.5, len(topics) - 0.5)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()

    file_path = os.path.join(output, 'coefficient_plot_main.pdf')
    plt.savefig(file_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure saved as PDF at {file_path}")

    file_path = os.path.join(output, 'coefficient_plot_main.png')
    plt.savefig(file_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Figure saved as PNG at {file_path}")
    
    plt.close()

def plot_coefficients_crisis_overlap(results_df, bank_labels, output, topic_labels):
    """
    Plot coefficient estimates with overlap for pre- and post-crisis periods.
    """
    plt.figure(figsize=(15, 20))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    sns.set_style("whitegrid", {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    
    topics = results_df['topic'].unique()
    topic_label_list = [topic_labels[topic] for topic in topics]

    banks = results_df['bank'].unique()
    periods = results_df['period'].unique()
    
    # Single light grey color for alternating background
    light_grey = '#f5f5f5'
    
    for i, topic in enumerate(topics):
        topic_data = results_df[results_df['topic'] == topic]
        
        # Alternate between white and light grey
        if i % 2 == 0:
            plt.axhspan(i - 0.5, i + 0.5, facecolor=light_grey, alpha=0.5)
        
        for j, bank in enumerate(banks):
            bank_data = topic_data[topic_data['bank'] == bank]
            
            for k, period in enumerate(periods):
                period_data = bank_data[bank_data['period'] == period]
                
                y_pos = i + (j / len(banks)) - 0.25 + (k * 0.5 / len(banks))
                color = f'C{j}'
                marker = 'o' if period == 'pre' else 's'
                alpha = 0.7 if period == 'pre' else 1.0
                
                plt.errorbar(x=period_data['coefficient'], y=y_pos,
                             xerr=period_data['std_error'],
                             fmt=marker, capsize=5, capthick=2, color=color, alpha=alpha,
                             label=f'{bank} ({period})' if i == 0 else "")
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    
    y_ticks = list(range(len(topics)))
    topic_labels = topic_labels if len(topic_labels) == len(topics) else topics 
    plt.yticks(y_ticks, topic_label_list, fontsize=16)  
    
    plt.xlabel('Coefficient Estimate', fontsize=16) 
    plt.ylabel('', fontsize=16)
    plt.title('Coefficient Estimates Pre and Post Crisis', fontsize=18) 
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='National Central Bank (Period)', 
               bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
    
    plt.ylim(-0.5, len(topics) - 0.5)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()

    # Save plots
    file_path = os.path.join(output, 'coefficient_plot_crisis.pdf')
    plt.savefig(file_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure saved as PDF at {file_path}")

    file_path = os.path.join(output, 'coefficient_plot_crisis.png')
    plt.savefig(file_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Figure saved as PNG at {file_path}")      

    plt.close()

def plot_coefficients_leads(results_df, bank_labels, output, topic_labels):
    """
    Main coefficient plot figure for leads analysis with increased font sizes and custom topic labels.
    """
    plt.figure(figsize=(12, 15))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    sns.set_style("whitegrid", {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    
    topics = results_df['topic'].unique()
    topic_label_list = [topic_labels[topic] for topic in topics]
    
    banks = results_df['bank'].unique()
    
    light_grey = '#f5f5f5'
    
    for i, topic in enumerate(topics):
        topic_data = results_df[results_df['topic'] == topic]
        
        if i % 2 == 0:
            plt.axhspan(i - 0.5, i + 0.5, facecolor=light_grey, alpha=0.5)
        
        for j, bank in enumerate(banks):
            bank_data = topic_data[topic_data['bank'] == bank]
            
            y_pos = i + j / len(banks) - 0.5 + 1 / (2 * len(banks))
            plt.errorbar(x=bank_data['coefficient'], y=y_pos,
                         xerr=bank_data['std_error'],
                         fmt='o', capsize=5, capthick=2, color=f'C{j}', label=bank if i == 0 else "")
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    
    plt.yticks(list(range(len(topics))), topic_label_list, fontsize=16)
    
    plt.xlabel('Lead Coefficient Estimate', fontsize=16)
    plt.ylabel('', fontsize=16)
    plt.title('Forward-Looking Effects', fontsize=18)
    
    plt.legend(title='National Central Bank', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
    
    plt.ylim(-0.5, len(topics) - 0.5)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()

    file_path = os.path.join(output, 'coefficient_plot_leads.pdf')
    plt.savefig(file_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure saved as PDF at {file_path}")

    file_path = os.path.join(output, 'coefficient_plot_leads.png')
    plt.savefig(file_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Figure saved as PNG at {file_path}")
    
    plt.close()

## Add here a halfyearly and yearly coefficientplot too.