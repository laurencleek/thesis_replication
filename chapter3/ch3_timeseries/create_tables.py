import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import os
from stargazer.stargazer import Stargazer
from statsmodels.nonparametric.smoothers_lowess import lowess

def run_regressions(df, ncb_banks, topics, controls, bank_labels):
    results = []
    models = {topic: {} for topic in topics}
    
    for topic in topics:
        topic_data = df[df['topic'] == topic].copy()
        ecb_data = topic_data[topic_data['banks'] == 4].copy()
        ncb_data = topic_data[topic_data['banks'] != 4].copy()
        
        # Create lagged values
        ncb_data['value_lag'] = ncb_data.groupby('banks')['value'].shift(1)
        
        # Create pivot table
        ncb_wide = ncb_data.pivot(index='yq', columns='banks', values='value_lag')
        ncb_wide.columns = [f'NCB_{bank_labels[col]}_lag' for col in ncb_wide.columns]
        
        # Merge with ECB data and ensure data alignment
        merged_data = ecb_data.merge(ncb_wide, on='yq', how='left')
        
        for bank in ncb_banks:
            try:
                # Add lagged dependent variable
                merged_data['value_lag_dep'] = merged_data.groupby('topic')['value'].shift(1)
                
                # Drop NaN values and create regression dataset
                regression_vars = ['value', 'value_lag_dep', f'NCB_{bank_labels[bank]}_lag'] + controls
                regression_data = merged_data.dropna(subset=regression_vars)
                
                if len(regression_data) == 0:
                    print(f"No valid data for topic {topic}, bank {bank}")
                    continue
                
                # Create topic_time cluster variable from cleaned data
                regression_data = regression_data.copy()  # Create a copy to avoid SettingWithCopyWarning
                regression_data['topic_time'] = regression_data['topic'].astype(str) + '_' + regression_data['yq'].astype(str)
                
                formula = f"value ~ value_lag_dep + NCB_{bank_labels[bank]}_lag + {' + '.join(controls)}"
                
                # Ensure cluster variable matches regression data
                cluster_var = regression_data['topic_time'].values
                
                model = ols(formula, data=regression_data).fit(
                    cov_type='cluster',
                    cov_kwds={'groups': cluster_var}
                )
                
                coef = model.params[f"NCB_{bank_labels[bank]}_lag"]
                std_err = model.bse[f"NCB_{bank_labels[bank]}_lag"]
                conf_int = model.conf_int().loc[f"NCB_{bank_labels[bank]}_lag"]
                
                results.append({
                    'topic': topic,
                    'bank': bank_labels[bank],
                    'coefficient': coef,
                    'std_error': std_err,
                    'conf_int_lower': conf_int[0],
                    'conf_int_upper': conf_int[1],
                    'n_obs': len(regression_data)
                })
                
                models[topic][bank] = model
                
            except Exception as e:
                print(f"Error in OLS for topic {topic}, bank {bank}: {e}")
                print("Data shape:", merged_data.shape)
                print("Available columns:", merged_data.columns.tolist())
                continue
    
    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        print("Warning: No regression results were produced!")
        
    return results_df, models

def run_regressions_half(df, ncb_banks, topics, controls, bank_labels):
    results = []
    models = {topic: {} for topic in topics}
    
    for topic in topics:
        topic_data = df[df['topic'] == topic].copy()
        ecb_data = topic_data[topic_data['banks'] == 4].copy()
        ncb_data = topic_data[topic_data['banks'] != 4].copy()
        
        # Create lagged values
        ncb_data['value_lag'] = ncb_data.groupby('banks')['value'].shift(1)
        
        # Create pivot table using 'yh' instead of 'halfyear'
        ncb_wide = ncb_data.pivot(index='yh', columns='banks', values='value_lag')
        ncb_wide.columns = [f'NCB_{bank_labels[col]}_lag' for col in ncb_wide.columns]
        
        # Merge using 'yh'
        merged_data = ecb_data.merge(ncb_wide, on='yh', how='left')
        
        for bank in ncb_banks:
            # Add lagged dependent variable
            merged_data['value_lag_dep'] = merged_data.groupby('topic')['value'].shift(1)
            
            formula = f"value ~ value_lag_dep + NCB_{bank_labels[bank]}_lag + {' + '.join(controls)}"
            try:
                # Create topic_time cluster variable
                merged_data['topic_time'] = merged_data['topic'].astype(str) + '_' + merged_data['yh'].astype(str)
                
                model = ols(formula, data=merged_data).fit(cov_type='cluster',
                                                         cov_kwds={'groups': merged_data['topic_time']})
                
                coef = model.params[f"NCB_{bank_labels[bank]}_lag"]
                std_err = model.bse[f"NCB_{bank_labels[bank]}_lag"]
                conf_int = model.conf_int().loc[f"NCB_{bank_labels[bank]}_lag"]
                
                results.append({
                    'topic': topic,
                    'bank': bank_labels[bank],
                    'coefficient': coef,
                    'std_error': std_err,
                    'conf_int_lower': conf_int[0],
                    'conf_int_upper': conf_int[1]
                })
            except Exception as e:
                print(f"Error in OLS for topic {topic}, bank {bank}: {e}")
                print("Formula:", formula)
                print("Available columns:", merged_data.columns.tolist())
    
    return pd.DataFrame(results), models


def run_regressions_crisis(df, ncb_banks, topics, controls, bank_labels, crisis_date):
    results = []
    models = {'pre': {topic: {} for topic in topics}, 'post': {topic: {} for topic in topics}}
    
    for period in ['pre', 'post']:
        period_df = df[df['yq'] < crisis_date] if period == 'pre' else df[df['yq'] >= crisis_date]
        
        for topic in topics:
            topic_data = period_df[period_df['topic'] == topic].copy()
            ecb_data = topic_data[topic_data['banks'] == 4].copy()
            ncb_data = topic_data[topic_data['banks'] != 4].copy()
            
            ncb_data['value_lag'] = ncb_data.groupby('banks')['value'].shift(1)
            ncb_wide = ncb_data.pivot(index='yq', columns='banks', values='value_lag')
            ncb_wide.columns = [f'NCB_{bank_labels[col]}_lag' for col in ncb_wide.columns]
            
            merged_data = ecb_data.merge(ncb_wide, on='yq', how='left')
            
            for bank in ncb_banks:
                # Add lagged dependent variable
                merged_data['value_lag_dep'] = merged_data.groupby('topic')['value'].shift(1)
                
                formula = f"value ~ value_lag_dep + NCB_{bank_labels[bank]}_lag + {' + '.join(controls)}"
                try:
                    # Create topic_time cluster variable
                    merged_data['topic_time'] = merged_data['topic'].astype(str) + '_' + merged_data['yq'].astype(str)
                    
                    model = ols(formula, data=merged_data).fit(cov_type='cluster',
                                                             cov_kwds={'groups': merged_data['topic_time']})
                    
                    coef = model.params[f"NCB_{bank_labels[bank]}_lag"]
                    std_err = model.bse[f"NCB_{bank_labels[bank]}_lag"]
                    conf_int = model.conf_int().loc[f"NCB_{bank_labels[bank]}_lag"]
                    
                    results.append({
                        'period': period,
                        'topic': topic,
                        'bank': bank_labels[bank],
                        'coefficient': coef,
                        'std_error': std_err,
                        'conf_int_lower': conf_int[0],
                        'conf_int_upper': conf_int[1]
                    })
                except Exception as e:
                    print(f"Error in OLS for period {period}, topic {topic}, bank {bank}: {e}")
    
    return pd.DataFrame(results), models

def run_regressions_leads(df, ncb_banks, topics, controls, bank_labels):
    """
    Run regressions using leads (future values) of NCB variables.
    """
    results = []
    models = {topic: {} for topic in topics}
    
    for topic in topics:
        topic_data = df[df['topic'] == topic].copy()
        ecb_data = topic_data[topic_data['banks'] == 4].copy()
        ncb_data = topic_data[topic_data['banks'] != 4].copy()
        
        # Calculate leads instead of lags
        ncb_data['value_lead'] = ncb_data.groupby('banks')['value'].shift(-1)
        ncb_wide = ncb_data.pivot(index='yq', columns='banks', values='value_lead')
        ncb_wide.columns = [f'NCB_{bank_labels[col]}_lead' for col in ncb_wide.columns]
        
        # Merge with ECB data
        merged_data = ecb_data.merge(ncb_wide, on='yq', how='left')
        
        for bank in ncb_banks:
            try:
                # Add lagged dependent variable for control
                merged_data['value_lag_dep'] = merged_data.groupby('topic')['value'].shift(1)
                
                # Drop NaN values and prepare regression dataset
                regression_vars = ['value', 'value_lag_dep', f'NCB_{bank_labels[bank]}_lead'] + controls
                regression_data = merged_data.dropna(subset=regression_vars).copy()
                
                if len(regression_data) == 0:
                    print(f"No valid data for topic {topic}, bank {bank}")
                    continue
                
                # Create topic_time cluster variable from cleaned data
                regression_data['topic_time'] = regression_data['topic'].astype(str) + '_' + regression_data['yq'].astype(str)
                
                # Create regression formula
                formula = f"value ~ value_lag_dep + NCB_{bank_labels[bank]}_lead + {' + '.join(controls)}"
                
                # Ensure cluster variable matches regression data
                cluster_var = regression_data['topic_time'].values
                
                # Run regression with properly aligned cluster variable
                model = ols(formula, data=regression_data).fit(
                    cov_type='cluster',
                    cov_kwds={'groups': cluster_var}
                )
                
                coef = model.params[f"NCB_{bank_labels[bank]}_lead"]
                std_err = model.bse[f"NCB_{bank_labels[bank]}_lead"]
                conf_int = model.conf_int().loc[f"NCB_{bank_labels[bank]}_lead"]
                
                results.append({
                    'topic': topic,
                    'bank': bank_labels[bank],
                    'coefficient': coef,
                    'std_error': std_err,
                    'conf_int_lower': conf_int[0],
                    'conf_int_upper': conf_int[1],
                    'n_obs': len(regression_data)
                })
                
                models[topic][bank] = model
                
            except Exception as e:
                print(f"Error in OLS for topic {topic}, bank {bank}: {e}")
                print("Regression data shape:", regression_data.shape if 'regression_data' in locals() else "No regression data")
                print("Available columns:", merged_data.columns.tolist())
                continue
    
    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        print("Warning: No regression results were produced!")
    else:
        print(f"Successfully generated {len(results_df)} regression results")
        
    return results_df, models

def create_latex_tables(models, controls, bank_labels, output):
    latex_output = "\\documentclass{article}\n\\usepackage{booktabs,longtable,adjustbox}\n\\usepackage{amsmath}\n\\begin{document}\n\n"
    
    for topic, bank_models in models.items():
        latex_output += f"\\section*{{Regression Results for Topic: {topic}}}\n\n"
        
        stargazer = Stargazer(bank_models.values())
        
        stargazer.title(f"Impact of NCB Lags on ECB Topic Value: {topic}")
        stargazer.covariate_order([f"NCB_{bank_labels[bank]}_lag" for bank in bank_models.keys()] + controls)
        stargazer.rename_covariates({f"NCB_{bank_labels[bank]}_lag": f"NCB {bank_labels[bank]} Lag" for bank in bank_models.keys()})
        stargazer.significance_levels([0.05, 0.01, 0.001])
        
        latex_table = stargazer.render_latex()
        latex_table = latex_table.replace("\\begin{table}", "\\begin{adjustbox}{width=\\textwidth,center}")
        latex_table = latex_table.replace("\\end{table}", "\\end{adjustbox}")
        
        latex_output += latex_table
        latex_output += "\n\\clearpage\n\n"
    
    latex_output += "\\end{document}"
    
    with open(os.path.join(output, 'ncb_ecb_regression_tables.tex'), 'w') as f:
        f.write(latex_output)


def create_latex_tables_crisis(models_crisis, controls, bank_labels, output):
    latex_output = """

"""
    
    for topic in models_crisis['pre'].keys():
        pre_models = models_crisis['pre'][topic]
        post_models = models_crisis['post'][topic]
        
        combined_models = []
        for bank in pre_models.keys():
            combined_models.extend([pre_models[bank], post_models[bank]])
        
        stargazer = Stargazer(combined_models)
        
        stargazer.title(f"Impact of NCB Lags on ECB Topic Value: {topic}")
        
        labels = []
        for bank in pre_models.keys():
            labels.extend([f"{bank_labels[bank]} (Pre)", f"{bank_labels[bank]} (Post)"])
        
        stargazer.custom_columns(labels, [1] * len(combined_models))
        
        covariate_order = [f"NCB_{bank_labels[bank]}_lag" for bank in pre_models.keys()] + controls
        stargazer.covariate_order(covariate_order)
        
        rename_covariates = {f"NCB_{bank_labels[bank]}_lag": f"NCB {bank_labels[bank]} Lag" for bank in pre_models.keys()}
        stargazer.rename_covariates(rename_covariates)
        
        stargazer.significance_levels([0.05, 0.01, 0.001])
        
        latex_table = stargazer.render_latex()
        
        # Modify the table environment
        latex_table = latex_table.replace("\\begin{table}", "\\begin{table}[H]\n\\small")
        latex_table = latex_table.replace("\\end{table}", "")
        
        # Add a note below the table
        note = ("\\\\[-1.8ex]\\multicolumn{" + str(len(combined_models) + 1) + "}{p{\\linewidth}}"
                "{\\footnotesize\\textit{Note:} This table presents regression results for the impact of National Central Bank (NCB) "
                "lags on European Central Bank (ECB) topic values. The columns show pre-crisis and post-crisis "
                "estimates for each NCB. Significance levels: * p<0.05, ** p<0.01, *** p<0.001.}")
        
        latex_table += note + "\n\\end{table}"
        
        latex_output += latex_table
        latex_output += "\n\\clearpage\n\n"
    
    latex_output += "\\end{document}"
    
    with open(os.path.join(output, 'ncb_ecb_regression_tables_crisis_landscape.tex'), 'w') as f:
        f.write(latex_output)

def create_basic_output_table(df, ncb_banks, topics, controls, bank_labels, output_path, time_col='yq'):
    """
    Create basic output table for regression results.
    Added time_col parameter to handle both quarterly ('yq') and half-yearly ('yh') data.
    """
    all_results = []
    all_models = {}
    
    for bank in ncb_banks:
        for topic in topics:
            # Filter data for ECB and current bank
            ecb_data = df[(df['central_bank'] == 'european central bank') & 
                         (df['topic'] == topic)].copy()
            bank_data = df[(df['central_bank'].apply(lambda x: x in [bank_labels[bank]])) & 
                          (df['topic'] == topic)].copy()
            
            # Sort by time and calculate lags and leads
            bank_data = bank_data.sort_values(time_col)
            bank_data['value_lag'] = bank_data['value'].shift(1)
            bank_data['value_lead'] = bank_data['value'].shift(-1)
            
            # Merge with ECB data
            merged_data = ecb_data.merge(
                bank_data[[time_col, 'topic', 'value', 'value_lag', 'value_lead']], 
                on=[time_col, 'topic'], 
                how='left',
                suffixes=('_ecb', f'_ncb_{bank_labels[bank]}')
            )
            
            # Add lagged dependent variable
            merged_data['value_ecb_lag'] = merged_data.groupby('topic')['value_ecb'].shift(1)
            
            # Update specifications to include lagged dependent variable
            specs = [
                ('Basic', f"value_ecb ~ value_ecb_lag + value_ncb_{bank_labels[bank]}", merged_data),
                ('With Controls', f"value_ecb ~ value_ecb_lag + value_ncb_{bank_labels[bank]} + {' + '.join(controls)}", merged_data),
                ('With Lags', f"value_ecb ~ value_ecb_lag + value_ncb_{bank_labels[bank]} + value_lag", merged_data),
                ('With Leads', f"value_ecb ~ value_ecb_lag + value_ncb_{bank_labels[bank]} + value_lead", merged_data),
                ('Full', f"value_ecb ~ value_ecb_lag + value_ncb_{bank_labels[bank]} + value_lag + value_lead + {' + '.join(controls)}", merged_data)
            ]
            
            for spec_name, formula, data in specs:
                try:
                    # Create topic_time cluster variable
                    data['topic_time'] = data['topic'].astype(str) + '_' + data[time_col].astype(str)
                    
                    model = ols(formula, data=data).fit(cov_type='cluster',
                                                      cov_kwds={'groups': data['topic_time']})
                    
                    # Store results
                    all_results.append({
                        'bank': bank_labels[bank],
                        'topic': topic,
                        'specification': spec_name,
                        'coefficient': model.params[f'value_ncb_{bank_labels[bank]}'],
                        'std_error': model.bse[f'value_ncb_{bank_labels[bank]}'],
                        'r_squared': model.rsquared,
                        'n_obs': model.nobs
                    })
                    
                    if topic not in all_models:
                        all_models[topic] = {}
                    if bank not in all_models[topic]:
                        all_models[topic][bank] = {}
                    all_models[topic][bank][spec_name] = model
                    
                except Exception as e:
                    print(f"Error in specification {spec_name} for topic {topic}, bank {bank}: {e}")
                    continue
    
    return pd.DataFrame(all_results), all_models

def create_basic_output_table_half(df, ncb_banks, topics, controls, bank_labels, output_path):
    """
    Wrapper function to call create_basic_output_table with half-yearly settings
    """
    return create_basic_output_table(
        df=df,
        ncb_banks=ncb_banks,
        topics=topics,
        controls=controls,
        bank_labels=bank_labels,
        output_path=output_path,
        time_col='yh'  # Specify half-yearly time column
    )
