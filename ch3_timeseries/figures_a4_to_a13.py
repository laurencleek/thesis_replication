import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import os
from stargazer.stargazer import Stargazer


def create_topic_panel_plots(df, topic_bank_mapping, circumstances, output, topic_titles, topics_per_page=2):
    # Set style with smaller fonts for condensed layout
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'font.size': 10})
    
    rename = {
        'google_trends_value': 'Salience',
        'gdp_real_growth': 'GDP Growth',
        'treu_ecb': 'Trust ECB',
        'hicp': 'Inflation'
    }
    
    # Group topics into pages
    topics = list(topic_bank_mapping.keys())
    topic_groups = [topics[i:i + topics_per_page] for i in range(0, len(topics), topics_per_page)]
    
    for page_num, page_topics in enumerate(topic_groups):
        # Create figure with extra space at top for titles
        fig = plt.figure(figsize=(16, 9 * len(page_topics)))
        
        for topic_idx, topic in enumerate(page_topics):
            central_banks = topic_bank_mapping[topic]
            topic_title = topic_titles.get(topic, topic)
            
            # Calculate subplot positions with space for title
            subplot_height = 1.0 / len(page_topics)
            
            # Create main gridspec for this topic with space for title
            topic_gs = fig.add_gridspec(len(page_topics), 1, 
                                      height_ratios=[1] * len(page_topics),
                                      hspace=0.4)[topic_idx]
            
            # Create subgridspec for the 2x2 plots
            plot_gs = topic_gs.subgridspec(2, 2, hspace=0.3, wspace=0.25)
            axes = np.empty((2, 2), dtype=object)
            
            # Add topic title above the plots
            fig.text(0.5, 1.0 - topic_idx * subplot_height - 0.02,
                    topic_title,
                    ha='center',
                    va='bottom',
                    fontsize=20,
                    fontweight='bold')
            
            # Create the 2x2 subplot grid
            for i in range(2):
                for j in range(2):
                    axes[i, j] = fig.add_subplot(plot_gs[i, j])
            
            for i, circumstance in enumerate(circumstances):
                row, col = divmod(i, 2)
                ax = axes[row, col]
                
                for bank in central_banks:
                    # ... existing plotting code for each bank ...
                    bank_data = df[df['central_bank'] == bank].sort_values('yq')
                    ecb_data = df[df['central_bank'] == 'european central bank'].sort_values('yq')
                    
                    required_columns = [topic, circumstance]
                    if not all(col in bank_data.columns for col in required_columns):
                        print(f"Missing required columns for topic '{topic}', circumstance '{circumstance}', central bank '{bank}'")
                        continue
                    
                    bank_data[f'{topic}_lag'] = bank_data[topic].shift(1)
                    
                    merged_data = pd.merge(ecb_data[['yq', topic, circumstance]], 
                                           bank_data[['yq', f'{topic}_lag']], 
                                           on='yq', 
                                           suffixes=('', f'_{bank}'))
                    
                    valid_data = merged_data.dropna()
                    
                    if valid_data.empty:
                        print(f"No valid data for topic '{topic}', circumstance '{circumstance}', central bank '{bank}'")
                        continue
                    
                    formula = f"{topic} ~ {circumstance} + {topic}_lag + {circumstance}:{topic}_lag"
                    
                    try:
                        model = ols(formula, data=valid_data).fit()
                    except Exception as e:
                        print(f"Error fitting model for topic '{topic}', circumstance '{circumstance}', central bank '{bank}': {str(e)}")
                        continue
                    
                    x_range = np.linspace(valid_data[circumstance].min(), valid_data[circumstance].max(), 100)
                    topic_lag_value = valid_data[f'{topic}_lag'].median()
                    
                    X_pred = pd.DataFrame({
                        circumstance: x_range,
                        f'{topic}_lag': topic_lag_value
                    })
                    
                    try:
                        predictions = model.get_prediction(X_pred)
                        pred_mean = predictions.predicted_mean
                        pred_ci = predictions.conf_int(alpha=0.05)
                    
                        label = f"{bank}"
                        ax.plot(x_range, pred_mean, label=label)
                        ax.fill_between(x_range, pred_ci[:, 0], pred_ci[:, 1], alpha=0.1)
                    except Exception as e:
                        print(f"Error making predictions for topic '{topic}', circumstance '{circumstance}', central bank '{bank}': {str(e)}")
                        continue
                
                # Remove x-axis label and adjust titles
                ax.set_xlabel('')  # Remove x-axis label
                ax.set_ylabel("ECB Topic Value", fontsize=10)
                # Move circumstance title above plot with more space
                ax.set_title(f"{rename.get(circumstance, circumstance)}", 
                           fontsize=14, pad=10, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Adjust legend position to be more compact
            handles, labels = axes[-1, -1].get_legend_handles_labels()
            axes[0, 1].legend(handles, labels, loc='upper right', fontsize=9, 
                            bbox_to_anchor=(1.15, 1.0))
        
        # Adjust layout with tighter spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.4)
        
        # Save with page number
        for format_type, folder_name in [('pdf', 'pdf'), ('png', 'png')]:
            folder_path = os.path.join(output, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, f'topic_panel_page_{page_num + 1}.{format_type}')
            fig.savefig(file_path, format=format_type, dpi=300, bbox_inches='tight')
            print(f"Panel figure saved as {format_type.upper()} at {file_path}")
        
        plt.close()

def create_three_way_interaction_plot(df, topic_bank_mapping, output):
    """Create three-way interaction plots for media, public opinion, and salience."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'font.size': 10})
    
    # Print available columns for debugging
    print("Available columns:", df.columns.tolist())
    
    # Check if required columns exist
    required_columns = ['yq', 'central_bank', 'google_trends_value', 'treu_ecb']
    
    # Look for any column that might represent speech or document count
    count_columns = [col for col in df.columns if any(term in col.lower() 
                                                     for term in ['speech', 'doc', 'document', 'count'])]
    print("Potential count columns found:", count_columns)
    
    if not count_columns:
        print("No speech/document count column found. Please check your data.")
        return
        
    count_column = count_columns[0]  # Use the first matching column
    print(f"Using {count_column} as the count measure")
    
    required_columns.append(count_column)
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return
    
    # Focus on Monetary Policy topic
    topic = 'Monetary_Policy_Central_Banking'
    central_banks = topic_bank_mapping[topic]
    
    for bank in central_banks:
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        bank_data = df[df['central_bank'] == bank].copy()
        ecb_data = df[df['central_bank'] == 'european central bank'].copy()
        
        # Merge ECB and bank data
        try:
            merged_data = pd.merge(
                ecb_data[['yq', topic, 'google_trends_value', 'treu_ecb', count_column]], 
                bank_data[['yq', topic]], 
                on='yq', 
                suffixes=('_ecb', '_bank')
            )
        except Exception as e:
            print(f"Error merging data for bank {bank}: {str(e)}")
            print("ECB data columns:", ecb_data.columns.tolist())
            print("Bank data columns:", bank_data.columns.tolist())
            continue
        
        if merged_data.empty:
            print(f"No data available for bank: {bank}")
            continue
            
        # Rest of the plotting code remains the same, but use count_column instead of speech_count
        # Define the ranges for our three variables
        google_trends_range = np.linspace(merged_data['google_trends_value'].min(), 
                                        merged_data['google_trends_value'].max(), 
                                        20)
        treu_range = np.linspace(merged_data['treu_ecb'].min(), 
                                merged_data['treu_ecb'].max(), 
                                20)
        
        # Create meshgrid for 3D surface
        X, Y = np.meshgrid(google_trends_range, treu_range)
        
        # Fit model with three-way interaction
        formula = (f"{topic}_ecb ~ google_trends_value * treu_ecb * {count_column} + "
                  f"{topic}_bank")
        try:
            model = ols(formula, data=merged_data).fit()
            
            # Create predictions for different count levels
            count_levels = [
                merged_data[count_column].quantile(0.25),
                merged_data[count_column].median(),
                merged_data[count_column].quantile(0.75)
            ]
            count_labels = ['Low Coverage', 'Medium Coverage', 'High Coverage']
            
            # Plot surfaces for different count levels
            for i, (count_level, label) in enumerate(zip(count_levels, count_labels), 1):
                ax = fig.add_subplot(2, 2, i, projection='3d')
                
                Z = np.zeros_like(X)
                for xi in range(len(google_trends_range)):
                    for yi in range(len(treu_range)):
                        pred_data = pd.DataFrame({
                            'google_trends_value': [X[xi,yi]],
                            'treu_ecb': [Y[xi,yi]],
                            count_column: [count_level],
                            f'{topic}_bank': [merged_data[f'{topic}_bank'].median()]
                        })
                        Z[xi,yi] = model.predict(pred_data)[0]
                
                surf = ax.plot_surface(X, Y, Z, cmap='viridis')
                ax.set_xlabel('Google Trends\n(Salience)', labelpad=10)
                ax.set_ylabel('Trust in ECB', labelpad=10)
                ax.set_zlabel('ECB Topic Value', labelpad=10)
                ax.set_title(f'{label}\n(N={int(count_level)})', pad=10, fontsize=12)
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                
                # Rotate the plot for better visibility
                ax.view_init(elev=20, azim=45)
        
        except Exception as e:
            print(f"Error creating plot for bank {bank}: {str(e)}")
            continue
        
        # Add overall title
        plt.suptitle(f'Three-way Interaction Effects\n{bank.title()}', y=1.02, fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        for format_type in ['pdf', 'png']:
            folder_path = os.path.join(output, format_type)
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, f'three_way_interaction_{bank.replace(" ", "_")}.{format_type}')
            plt.savefig(file_path, format=format_type, dpi=300, bbox_inches='tight')
            print(f"Three-way interaction figure saved as {format_type.upper()} at {file_path}")
        
        plt.close()

def create_interaction_margin_plots(df, topic_bank_mapping, output):
    """Create margin plots for interactions with lagged dependent variable."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'font.size': 10})
    
    # Find available count column
    count_columns = [col for col in df.columns if any(term in col.lower() 
                                                     for term in ['speech', 'doc', 'document', 'count'])]
    if not count_columns:
        print("No speech/document count column found. Please check your data.")
        print("Available columns:", df.columns.tolist())
        return
        
    count_column = count_columns[0]
    print(f"Using {count_column} as the count measure")
    
    # Focus on Monetary Policy topic
    topic = 'Monetary_Policy_Central_Banking'
    central_banks = topic_bank_mapping[topic]
    
    for bank in central_banks:
        # Create figure for each bank
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        bank_data = df[df['central_bank'] == bank].copy()
        ecb_data = df[df['central_bank'] == 'european central bank'].copy()
        
        # Create lagged variables and merge data
        bank_data[f'{topic}_lag'] = bank_data[topic].shift(1)
        ecb_data[f'{topic}_lag'] = ecb_data[topic].shift(1)
        
        # Adjust merge columns based on available data
        try:
            merged_data = pd.merge(
                ecb_data[['yq', topic, f'{topic}_lag', 'google_trends_value', 'treu_ecb', count_column]], 
                bank_data[['yq', topic, f'{topic}_lag']], 
                on='yq', 
                suffixes=('_ecb', '_bank')
            ).dropna()
            
            if merged_data.empty:
                print(f"No data available for bank: {bank}")
                continue
            
            interaction_vars = [
                ('google_trends_value', 'Salience'),
                ('treu_ecb', 'Trust in ECB'),
                (count_column, 'Media Coverage')
            ]
            
            for idx, (var, title) in enumerate(interaction_vars):
                # Rest of plotting code remains the same...
                formula = (f"{topic}_ecb ~ {topic}_lag_ecb + {topic}_lag_bank * {var}")
                
                try:
                    model = ols(formula, data=merged_data).fit()
                    
                    # Generate prediction data
                    x_range = np.linspace(merged_data[var].min(), merged_data[var].max(), 100)
                    ncb_lag_levels = [
                        merged_data[f'{topic}_lag_bank'].quantile(0.25),
                        merged_data[f'{topic}_lag_bank'].median(),
                        merged_data[f'{topic}_lag_bank'].quantile(0.75)
                    ]
                    
                    # Plot margins
                    ax = axes[idx]
                    ecb_lag_med = merged_data[f'{topic}_lag_ecb'].median()
                    
                    for ncb_lag in ncb_lag_levels:
                        X_pred = pd.DataFrame({
                            var: x_range,
                            f'{topic}_lag_bank': ncb_lag,
                            f'{topic}_lag_ecb': ecb_lag_med
                        })
                        
                        predictions = model.get_prediction(X_pred)
                        pred_mean = predictions.predicted_mean
                        pred_ci = predictions.conf_int(alpha=0.05)
                        
                        label = f'NCB lag = {ncb_lag:.2f}'
                        ax.plot(x_range, pred_mean, label=label)
                        ax.fill_between(x_range, pred_ci[:, 0], pred_ci[:, 1], alpha=0.1)
                    
                    ax.set_xlabel(title)
                    ax.set_ylabel('ECB Topic Value')
                    ax.legend(title='NCB Topic Value (t-1)', bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.set_title(title)
                
                except Exception as e:
                    print(f"Error creating plot for {var} in bank {bank}: {str(e)}")
                    continue
            
            plt.suptitle(f'Interaction Effects for {bank.title()}\nDependent Variable: ECB {topic}', y=1.05)
            plt.tight_layout()
            
            # Save the figure
            for format_type in ['pdf', 'png']:
                folder_path = os.path.join(output, format_type)
                os.makedirs(folder_path, exist_ok=True)
                file_path = os.path.join(folder_path, f'margin_plots_{bank.replace(" ", "_")}.{format_type}')
                plt.savefig(file_path, format=format_type, dpi=300, bbox_inches='tight')
                print(f"Margin plots saved as {format_type.upper()} at {file_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"Error processing bank {bank}: {str(e)}")
            print("Available columns in ecb_data:", ecb_data.columns.tolist())
            print("Available columns in bank_data:", bank_data.columns.tolist())
            continue

def create_two_way_margin_plots(df, topic_bank_mapping, output):
    """Create margin plots comparing NCB interactions across different conditions."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'font.size': 10})
    
    # Focus on Monetary Policy topic
    topic = 'Monetary_Policy_Central_Banking'
    central_banks = topic_bank_mapping[topic]
    
    # Define conditions with labels
    conditions = [
        ('google_trends_value', 'Salience'),
        ('treu_ecb', 'Trust in ECB'),
        ('doc_count', 'Media Coverage')
    ]
    
    # Create separate figure for each condition
    for condition, condition_label in conditions:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up color palette
        colors = plt.cm.Set2(np.linspace(0, 1, len(central_banks)))
        
        for bank, color in zip(central_banks, colors):
            bank_data = df[df['central_bank'] == bank].copy()
            ecb_data = df[df['central_bank'] == 'european central bank'].copy()
            
            # Create lagged variables
            bank_data[f'{topic}_lag'] = bank_data[topic].shift(1)
            
            # Merge data
            merged_data = pd.merge(
                ecb_data[['yq', topic, condition]], 
                bank_data[['yq', topic, f'{topic}_lag']], 
                on='yq',
                suffixes=('_ecb', '_bank')
            ).dropna()
            
            if merged_data.empty:
                print(f"No data available for bank: {bank}")
                continue
            
            # Create high/low values for NCB lag (25th and 75th percentiles)
            ncb_lag_low = merged_data[f'{topic}_lag_bank'].quantile(0.25)
            ncb_lag_high = merged_data[f'{topic}_lag_bank'].quantile(0.75)
            
            # Fit model with interaction
            formula = f"{topic}_ecb ~ {topic}_lag_bank * {condition}"
            
            try:
                model = ols(formula, data=merged_data).fit()
                
                # Generate prediction data
                x_range = np.linspace(merged_data[condition].min(), 
                                    merged_data[condition].max(), 
                                    100)
                
                # Plot for both low and high NCB lag values
                for lag_value, linestyle in [(ncb_lag_low, '--'), (ncb_lag_high, '-')]: 
                    X_pred = pd.DataFrame({
                        condition: x_range,
                        f'{topic}_lag_bank': lag_value
                    })
                    
                    predictions = model.get_prediction(X_pred)
                    pred_mean = predictions.predicted_mean
                    pred_ci = predictions.conf_int(alpha=0.05)
                    
                    # Simplified bank name for legend
                    bank_label = bank.replace('bank of ', '').replace('deutsche ', '')
                    label = f"{bank_label.title()} ({lag_value:.2f})"
                    
                    # Plot line and confidence interval
                    ax.plot(x_range, pred_mean, 
                           label=label, 
                           color=color, 
                           linestyle=linestyle)
                    ax.fill_between(x_range, pred_ci[:, 0], pred_ci[:, 1], 
                                  alpha=0.1, color=color)
                
            except Exception as e:
                print(f"Error creating plot for {bank}: {str(e)}")
                continue
        
        # Customize plot
        ax.set_xlabel(condition_label)
        ax.set_ylabel('ECB Monetary Policy Topic')
        ax.legend(title='NCB (Topic Value t-1)', 
                 bbox_to_anchor=(1.05, 1), 
                 loc='upper left')
        ax.set_title(f'Effect of {condition_label} on ECB Communication\nby NCB Topic Value')
        
        # Adjust layout and save
        plt.tight_layout()
        
        for format_type in ['pdf', 'png']:
            folder_path = os.path.join(output, format_type)
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, 
                                   f'two_way_interaction_{condition.lower()}.{format_type}')
            plt.savefig(file_path, format=format_type, dpi=300, bbox_inches='tight')
            print(f"Two-way interaction plot saved as {format_type.upper()} at {file_path}")
        
        plt.close()

def create_ncb_responsiveness_plot(df, topic_bank_mapping, output):
    """Create panel plot showing NCB responsiveness across different conditions."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'font.size': 10})
    
    # Find available count column
    count_columns = [col for col in df.columns if any(term in col.lower() 
                                                     for term in ['speech', 'doc', 'document', 'count'])]
    count_column = count_columns[0] if count_columns else None
    
    # Focus on Monetary Policy topic
    topic = 'Monetary_Policy_Central_Banking'
    central_banks = topic_bank_mapping[topic]
    
    # Define conditions and their labels
    conditions = [
        ('google_trends_value', 'Salience'),
        ('treu_ecb', 'Trust in ECB'),
        (count_column, 'Media Coverage')
    ] if count_column else [
        ('google_trends_value', 'Salience'),
        ('treu_ecb', 'Trust in ECB')
    ]
    
    # Create one figure with subplots
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 5))
    if len(conditions) == 1:
        axes = [axes]
    
    # Set up color palette for banks
    colors = plt.cm.Set2(np.linspace(0, 1, len(central_banks)))
    
    for idx, (condition, condition_label) in enumerate(conditions):
        ax = axes[idx]
        bank_results = []
        
        for bank, color in zip(central_banks, colors):
            # Prepare data for each bank
            bank_data = df[df['central_bank'] == bank].copy()
            ecb_data = df[df['central_bank'] == 'european central bank'].copy()
            
            # Create lag variables before merging
            bank_data['ncb_lag'] = bank_data[topic].shift(1)
            
            # Merge data
            merged_data = pd.merge(
                ecb_data[['yq', topic, condition]], 
                bank_data[['yq', 'ncb_lag']], 
                on='yq'
            ).dropna()
            
            if merged_data.empty:
                print(f"No data available for bank: {bank}")
                continue
            
            # Standardize variables
            for col in [topic, condition, 'ncb_lag']:
                merged_data[col] = (merged_data[col] - merged_data[col].mean()) / merged_data[col].std()
            
            # Fit model with interaction
            formula = f"{topic} ~ ncb_lag * {condition}"
            
            try:
                model = ols(formula, data=merged_data).fit()
                
                # Extract interaction coefficient and CI
                interaction_var = f"ncb_lag:{condition}"
                coef = model.params[interaction_var]
                conf_int = model.conf_int().loc[interaction_var]
                
                # Simplified bank name
                bank_label = bank.replace('bank of ', '').replace('deutsche ', '')
                
                bank_results.append({
                    'bank': bank_label.title(),
                    'coef': coef,
                    'ci_lower': conf_int[0],
                    'ci_upper': conf_int[1],
                    'color': color
                })
                
            except Exception as e:
                print(f"Error for {bank} in {condition}: {str(e)}")
                continue
        
        # Plot results
        for res in bank_results:
            ax.vlines(x=res['bank'], ymin=res['ci_lower'], ymax=res['ci_upper'],
                     color=res['color'], alpha=0.6)
            ax.plot(res['bank'], res['coef'], 'o',
                   color=res['color'], label=res['bank'], markersize=8)
        
        # Customize subplot
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(condition_label)
        ax.set_ylabel('Interaction Effect' if idx == 0 else '')
        ax.tick_params(axis='x', rotation=45)
        
        if idx == 0:
            ax.legend(title='National Central Banks',
                     bbox_to_anchor=(0, -0.4),
                     loc='upper left',
                     ncol=len(central_banks))
    
    plt.suptitle('NCB Communication Responsiveness to Different Conditions\nMonetary Policy Topic',
                 y=1.05)
    plt.tight_layout()
    
    # Save figure
    for format_type in ['pdf', 'png']:
        folder_path = os.path.join(output, format_type)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f'ncb_responsiveness_panel.{format_type}')
        plt.savefig(file_path, format=format_type, dpi=300, bbox_inches='tight')
        print(f"NCB responsiveness panel saved as {format_type.upper()} at {file_path}")
    
    plt.close()

def create_ncb_responsiveness_plot(df, topic_bank_mapping, output):
    """Create panel plot showing NCB responsiveness across different conditions."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'font.size': 10})
    
    # Focus on Monetary Policy topic
    topic = 'Monetary_Policy_Central_Banking'
    central_banks = topic_bank_mapping[topic]
    
    # Define conditions and their labels
    conditions = [
        ('google_trends_value', 'Salience'),
        ('treu_ecb', 'Trust in ECB'),
        ('doc_count', 'Media Coverage')
    ]
    
    # Create one figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Set up color palette for banks
    colors = plt.cm.Set2(np.linspace(0, 1, len(central_banks)))
    
    for idx, (condition, condition_label) in enumerate(conditions):
        ax = axes[idx]
        
        # Store results for each bank
        bank_results = []
        
        for bank, color in zip(central_banks, colors):
            bank_data = df[df['central_bank'] == bank].copy()
            ecb_data = df[df['central_bank'] == 'european central bank'].copy()
            
            # Create lagged variables
            bank_data[f'{topic}_lag'] = bank_data[topic].shift(1)
            
            # Merge data
            merged_data = pd.merge(
                ecb_data[['yq', topic, condition]], 
                bank_data[['yq', topic, f'{topic}_lag']], 
                on='yq',
                suffixes=('_ecb', '_bank')
            ).dropna()
            
            if merged_data.empty:
                print(f"No data available for bank: {bank}")
                continue
            
            # Standardize variables for comparison
            merged_data[condition] = (merged_data[condition] - merged_data[condition].mean()) / merged_data[condition].std()
            merged_data[f'{topic}_lag_bank'] = (merged_data[f'{topic}_lag_bank'] - merged_data[f'{topic}_lag_bank'].mean()) / merged_data[f'{topic}_lag_bank'].std()
            merged_data[f'{topic}_ecb'] = (merged_data[f'{topic}_ecb'] - merged_data[f'{topic}_ecb'].mean()) / merged_data[f'{topic}_ecb'].std()
            
            # Fit model with interaction
            formula = f"{topic}_ecb ~ {topic}_lag_bank * {condition}"
            
            try:
                model = ols(formula, data=merged_data).fit()
                
                # Extract interaction coefficient and confidence interval
                interaction_var = f"{topic}_lag_bank:{condition}"
                coef = model.params[interaction_var]
                conf_int = model.conf_int().loc[interaction_var]
                
                # Simplified bank name
                bank_label = bank.replace('bank of ', '').replace('deutsche ', '')
                
                # Store results
                bank_results.append({
                    'bank': bank_label.title(),
                    'coef': coef,
                    'ci_lower': conf_int[0],
                    'ci_upper': conf_int[1],
                    'color': color
                })
                
            except Exception as e:
                print(f"Error creating plot for {bank}: {str(e)}")
                continue
        
        # Plot coefficients for all banks
        for res in bank_results:
            ax.plot([res['bank'], res['bank']], [res['ci_lower'], res['ci_upper']], 
                   color=res['color'], alpha=0.6)
            ax.plot([res['bank']], [res['coef']], 'o', 
                   color=res['color'], label=res['bank'])
        
        # Customize subplot
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(condition_label)
        ax.set_ylabel('Interaction Coefficient' if idx == 0 else '')
        ax.tick_params(axis='x', rotation=45)
        
        if idx == 0:  # Only show legend for first subplot
            ax.legend(title='National Central Banks', 
                     bbox_to_anchor=(0, -0.4),
                     loc='upper left',
                     ncol=len(central_banks))
    
    plt.suptitle('NCB Communication Responsiveness\nInteraction Effects with Different Conditions', 
                 y=1.05)
    plt.tight_layout()
    
    # Save the figure
    for format_type in ['pdf', 'png']:
        folder_path = os.path.join(output, format_type)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f'ncb_responsiveness_panel.{format_type}')
        plt.savefig(file_path, format=format_type, dpi=300, bbox_inches='tight')
        print(f"NCB responsiveness panel saved as {format_type.upper()} at {file_path}")
    
    plt.close()



