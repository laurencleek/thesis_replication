import os
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from stargazer.stargazer import Stargazer
from linearmodels.panel import PanelOLS


def run_aggregate_regression_and_export(df, controls, output_folder):
    try:
        # Check if required columns exist
        required_columns = ['banks', 'yq', 'value']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"The following required columns are missing: {', '.join(missing_columns)}")

        # Prepare data
        ecb_data = df[df['banks'] == 4].copy()
        ncb_data = df[df['banks'] != 4].copy()
        
        # Function to calculate lag and lead
        def calc_lag_lead(group):
            group = group.sort_values('yq')
            group['value_lag'] = group['value'].shift(1)
            group['value_lead'] = group['value'].shift(-1)
            return group

        # Calculate lag and lead for each NCB
        ncb_data = ncb_data.groupby(['banks', 'yq']).first().reset_index()
        ncb_data = ncb_data.groupby('banks').apply(calc_lag_lead).reset_index(drop=True)
        
        # Create separate DataFrames for lag and lead
        lag_data = ncb_data[['yq', 'banks', 'value_lag']].rename(columns={'value_lag': 'value'})
        lead_data = ncb_data[['yq', 'banks', 'value_lead']].rename(columns={'value_lead': 'value'})
        
        # Merge lag and lead data with ECB data
        merged_data = ecb_data.merge(lag_data, on='yq', suffixes=('', '_lag'))
        merged_data = merged_data.merge(lead_data, on='yq', suffixes=('', '_lead'))
        
        # Rename columns
        merged_data = merged_data.rename(columns={
            'banks_lag': 'lag_bank',
            'banks_lead': 'lead_bank',
            'value_lag': 'lag_value',
            'value_lead': 'lead_value'
        })
        
        # Drop rows with NaN values
        merged_data = merged_data.dropna()
        
        # Add lagged dependent variable
        merged_data['value_lag_dep'] = merged_data.groupby('banks')['value'].shift(1)
        
        # Prepare the formula for regression
        ncb_vars = [f"lag_value_{bank} + lead_value_{bank}" for bank in ncb_data['banks'].unique()]
        formula = f"value ~ value_lag_dep + {' + '.join(ncb_vars)} + {' + '.join(controls)}"
        
        # Create a MultiIndex for clustering
        merged_data['topic_time'] = merged_data['topic'].astype(str) + '_' + merged_data['yq'].astype(str)
        
        # Run aggregate regression with clustered standard errors
        model = ols(formula, data=merged_data).fit(cov_type='cluster', 
                                                  cov_kwds={'groups': merged_data['topic_time']})
        
        # Create Stargazer object for LaTeX
        stargazer = Stargazer([model])
        stargazer.title("Aggregate Regression Results")
        stargazer.custom_columns("ECB", [1])
        stargazer.significant_digits(3)
        stargazer.show_degrees_of_freedom(False)
        
        # Generate LaTeX code
        latex_output = stargazer.render_latex()
        
        # Create 'tex' subfolder and save LaTeX file
        tex_folder = os.path.join(output_folder, 'tex')
        os.makedirs(tex_folder, exist_ok=True)
        latex_file = os.path.join(tex_folder, 'main_results.tex')
        with open(latex_file, 'w') as f:
            f.write(latex_output)
        print(f"LaTeX table saved to {latex_file}")

        # Prepare data for RTF
        results_summary = model.summary()
        results_as_html = results_summary.tables[1].as_html()
        df_results = pd.read_html(results_as_html, header=0, index_col=0)[0].reset_index()
        
        # Generate improved RTF table
        rtf_content = r"{\rtf1\ansi\deff0 {\fonttbl {\f0 Arial;}}"
        rtf_content += r"{\colortbl;\red0\green0\blue0;}"
        rtf_content += r"\paperw12240\paperh15840\margl1440\margr1440\margt1440\margb1440"
        rtf_content += r"\widowctrl\ftnbj\aenddoc\formshade\viewkind1\viewscale100\pgbrdrhead\pgbrdrfoot"
        rtf_content += r"\fs20"
        rtf_content += r"\qc\b Aggregate Regression Results\b0\par\par"
        rtf_content += r"\trowd\trgaph108\trleft-108\trqc"
        
        # Add column headers
        for col in df_results.columns:
            rtf_content += r"\clbrdrt\brdrw10\brdrs\clbrdrl\brdrw10\brdrs\clbrdrb\brdrw10\brdrs\clbrdrr\brdrw10\brdrs\cellx" + str(1500 * (df_results.columns.get_loc(col) + 1))
        rtf_content += r"\pard\intbl"
        for col in df_results.columns:
            rtf_content += r"\qc\b " + col + r"\cell"
        rtf_content += r"\row"
        
        # Add data rows
        for _, row in df_results.iterrows():
            rtf_content += r"\trowd\trgaph108\trleft-108\trqc"
            for col in df_results.columns:
                rtf_content += r"\clbrdrt\brdrw10\brdrs\clbrdrl\brdrw10\brdrs\clbrdrb\brdrw10\brdrs\clbrdrr\brdrw10\brdrs\cellx" + str(1500 * (df_results.columns.get_loc(col) + 1))
            rtf_content += r"\pard\intbl"
            for col in df_results.columns:
                rtf_content += r"\qc " + str(row[col]) + r"\cell"
            rtf_content += r"\row"
        
        rtf_content += r"\pard\par}"
        
        # Create 'rtf' subfolder and save RTF file
        rtf_folder = os.path.join(output_folder, 'rtf')
        os.makedirs(rtf_folder, exist_ok=True)
        rtf_file = os.path.join(rtf_folder, 'main_results.rtf')
        with open(rtf_file, 'w', encoding='utf-8') as f:
            f.write(rtf_content)
        print(f"RTF table saved to {rtf_file}")

        return model.summary(), latex_output, rtf_content
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return str(e), None, None


def run_aggregate_regression_and_export1(df, controls, output_folder):
    try:
        # Check if required columns exist
        required_columns = ['banks', 'yq', 'value', 'topic']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"The following required columns are missing: {', '.join(missing_columns)}")

        # Check if control variables exist in the dataset
        missing_controls = [ctrl for ctrl in controls if ctrl not in df.columns]
        if missing_controls:
            print(f"Warning: The following control variables are not in the dataset and will be excluded: {', '.join(missing_controls)}")
            controls = [ctrl for ctrl in controls if ctrl in df.columns]

        # Prepare data
        ecb_data = df[df['banks'] == 4].copy()
        ncb_data = df[df['banks'] != 4].copy()
        
        # Function to calculate lag and lead
        def calc_lag_lead(group):
            group = group.sort_values('yq')
            group['value_lag'] = group['value'].shift(1)
            group['value_lead'] = group['value'].shift(-1)
            return group

        # Calculate lag and lead for each NCB
        ncb_data = ncb_data.groupby(['banks', 'topic']).apply(calc_lag_lead).reset_index(drop=True)
        
        # Prepare models
        models = []
        ncbs = ncb_data['banks'].unique()
        
        for ncb in ncbs:
            # Filter data for the current NCB
            current_ncb_data = ncb_data[ncb_data['banks'] == ncb].copy()
            
            # Merge with ECB data
            merged_data = ecb_data.merge(current_ncb_data, on=['yq', 'topic'] + controls, suffixes=('_ecb', '_ncb'))
            
            # Ensure all necessary columns are present
            for col in ['value_ecb', 'value_lag', 'value_lead', 'topic'] + controls:
                if col not in merged_data.columns:
                    raise ValueError(f"Column '{col}' is missing from the dataset for NCB {ncb}")
            
            # Add lagged dependent variable
            merged_data['value_ecb_lag'] = merged_data.groupby('topic')['value_ecb'].shift(1)
            
            # Update formula to include lagged dependent variable
            formula = f"value_ecb ~ value_ecb_lag + value_lag + value_lead + C(topic) + {' + '.join(controls)}"
            
            # Create topic_time cluster variable
            merged_data['topic_time'] = merged_data['topic'].astype(str) + '_' + merged_data['yq'].astype(str)
            
            # Run regression with clustered standard errors
            model = ols(formula, data=merged_data).fit(cov_type='cluster',
                                                      cov_kwds={'groups': merged_data['topic_time']})
            models.append(model)
        
        # Create Stargazer object for LaTeX
        stargazer = Stargazer(models)
        stargazer.title("Regression Results")
        stargazer.custom_columns([f"NCB {ncb}" for ncb in ncbs], [1] * len(ncbs))
        stargazer.significant_digits(3)
        stargazer.show_degrees_of_freedom(False)
        
        # Customize row labels
        stargazer.rename_covariates({
            'value_lag': 'Lag',
            'value_lead': 'Lead',
            'C(topic)': 'Topic FE'
        })
        
        # Move certain variables to the top
        top_variables = ['Lag', 'Lead', 'Topic FE'] + controls
        stargazer.reorder_covariates(top_variables)
        
        # Generate LaTeX code
        latex_output = stargazer.render_latex()
        
        # Create 'tex' subfolder and save LaTeX file
        tex_folder = os.path.join(output_folder, 'tex')
        os.makedirs(tex_folder, exist_ok=True)
        latex_file = os.path.join(tex_folder, 'main_results.tex')
        with open(latex_file, 'w') as f:
            f.write(latex_output)
        print(f"LaTeX table saved to {latex_file}")

        # Prepare data for RTF
        rtf_content = r"{\rtf1\ansi\deff0 {\fonttbl {\f0 Arial;}}"
        rtf_content += r"{\colortbl;\red0\green0\blue0;}"
        rtf_content += r"\paperw12240\paperh15840\margl1440\margr1440\margt1440\margb1440"
        rtf_content += r"\widowctrl\ftnbj\aenddoc\formshade\viewkind1\viewscale100\pgbrdrhead\pgbrdrfoot"
        rtf_content += r"\fs20"
        rtf_content += r"\qc\b Regression Results\b0\par\par"
        rtf_content += r"\trowd\trgaph108\trleft-108\trqc"
        
        # Add column headers
        column_headers = ['Variable'] + [f"NCB {ncb}" for ncb in ncbs]
        for i, header in enumerate(column_headers):
            rtf_content += r"\clbrdrt\brdrw10\brdrs\clbrdrl\brdrw10\brdrs\clbrdrb\brdrw10\brdrs\clbrdrr\brdrw10\brdrs\cellx" + str(1500 * (i + 1))
        rtf_content += r"\pard\intbl"
        for header in column_headers:
            rtf_content += r"\qc\b " + header + r"\cell"
        rtf_content += r"\row"
        
        # Add data rows
        for var in top_variables:
            rtf_content += r"\trowd\trgaph108\trleft-108\trqc"
            for i in range(len(column_headers)):
                rtf_content += r"\clbrdrt\brdrw10\brdrs\clbrdrl\brdrw10\brdrs\clbrdrb\brdrw10\brdrs\clbrdrr\brdrw10\brdrs\cellx" + str(1500 * (i + 1))
            rtf_content += r"\pard\intbl"
            rtf_content += r"\ql " + var + r"\cell"
            for model in models:
                if var == 'Topic FE':
                    rtf_content += r"\qc \u10003?\cell"  # Checkmark
                elif var in model.params.index:
                    coef = model.params[var]
                    se = model.bse[var]
                    t_stat = coef / se
                    stars = ''
                    if abs(t_stat) > 2.58:
                        stars = '***'
                    elif abs(t_stat) > 1.96:
                        stars = '**'
                    elif abs(t_stat) > 1.645:
                        stars = '*'
                    rtf_content += r"\qc " + f"{coef:.3f}{stars}\par({se:.3f})" + r"\cell"
                else:
                    rtf_content += r"\qc \cell"
            rtf_content += r"\row"
        
        rtf_content += r"\pard\par}"
        
        # Create 'rtf' subfolder and save RTF file
        rtf_folder = os.path.join(output_folder, 'rtf')
        os.makedirs(rtf_folder, exist_ok=True)
        rtf_file = os.path.join(rtf_folder, 'main_results.rtf')
        with open(rtf_file, 'w', encoding='utf-8') as f:
            f.write(rtf_content)
        print(f"RTF table saved to {rtf_file}")

        return models, latex_output, rtf_content
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None, None
