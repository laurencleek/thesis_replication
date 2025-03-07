import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from stargazer.stargazer import Stargazer
import os

def run_sintra_regressions(df, ncb_banks, topics, controls, bank_labels, time_col='yq'):
    """Run regressions with Sintra dummy using NCB lags"""
    results = {}
    
    # Create Sintra dummy
    df = df.copy()
    df['quarter'] = pd.to_datetime(df[time_col]).dt.quarter
    df['sintra'] = (df['quarter'] == 3).astype(int)
    
    for topic in topics:
        topic_models = {}
        
        # Filter data for current topic
        topic_data = df[df['topic'] == topic].copy()
        
        # Get ECB data
        ecb_data = topic_data[topic_data['banks'] == 4].copy()
        
        # Process each NCB
        for bank in ncb_banks:
            try:
                # Get NCB data
                bank_name = bank_labels[bank]
                ncb_data = topic_data[topic_data['banks'] == bank].copy()
                
                # Create lag for NCB
                ncb_data['value_lag'] = ncb_data['value'].shift(1)
                
                # Prepare regression data
                reg_data = pd.merge(
                    ecb_data[[time_col, 'value', 'sintra'] + controls],
                    ncb_data[[time_col, 'value_lag']],
                    on=time_col,
                    how='left'
                )
                
                # Create interaction term
                reg_data['value_lag_sintra'] = reg_data['value_lag'] * reg_data['sintra']
                
                # Drop missing values
                reg_data = reg_data.dropna()
                
                if len(reg_data) == 0:
                    print(f"No valid data for topic {topic}, bank {bank_name}")
                    continue
                
                # Create cluster variable that's unique per observation
                reg_data['topic_time'] = reg_data.index
                
                # Run regression
                formula = ("value ~ value_lag + sintra + value_lag_sintra + " + 
                         " + ".join(controls))
                
                model = ols(formula, data=reg_data).fit(
                    cov_type='cluster',
                    cov_kwds={'groups': reg_data.index}  # Use index for clustering
                )
                
                topic_models[bank_name] = model
                
            except Exception as e:
                print(f"Error in regression for topic {topic}, bank {bank_name}: {e}")
                continue
        
        if topic_models:
            results[topic] = topic_models
    
    return results

def create_sintra_latex_tables(results, output_path, topic_titles=None):
    """Create LaTeX tables for Sintra analysis"""
    latex_output = """\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{longtable}
\\usepackage{adjustbox}
\\begin{document}
"""
    
    for topic, models in results.items():
        display_topic = topic_titles.get(topic, topic) if topic_titles else topic
        
        # Create Stargazer instance
        stargazer = Stargazer(list(models.values()))
        
        # Configure table
        stargazer.title(f"Impact of NCB Communication on ECB: {display_topic}")
        stargazer.show_model_numbers(False)
        
        # Column headers (NCB names)
        stargazer.custom_columns(list(models.keys()), [1] * len(models))
        
        # Variable names in table
        rename_dict = {
            'sintra': 'Sintra Period',
            'value_lag': 'NCB Topic (t-1)',
            'value_lag_sintra': 'NCB Topic × Sintra'
        }
        
        # Set covariate order and rename variables
        ordered_covariates = ['value_lag', 'sintra', 'value_lag_sintra']
        
        for bank in models.keys():
            sint_var = f'value_lag_sintra'
            rename_dict[sint_var] = f'NCB Topic × Sintra'
        
        stargazer.covariate_order(ordered_covariates)
        stargazer.rename_covariates(rename_dict)
        
        # Generate table
        table = stargazer.render_latex()
        
        # Add note
        table = table.replace(
            "\\end{tabular}",
            "\\midrule\n"
            "\\multicolumn{" + str(len(models) + 1) + "}{l}{\\textit{Notes: }" +
            "Dependent variable is ECB topic coverage. Standard errors clustered by topic-time. " +
            "Sintra Period is a dummy variable equal to 1 in Q3 of each year. " +
            "* p<0.1, ** p<0.05, *** p<0.01} \\\\\n"
            "\\end{tabular}"
        )
        
        latex_output += table + "\n\\clearpage\n\n"
    
    latex_output += "\\end{document}"
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_output)

