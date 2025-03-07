import os
import pandas as pd
from bertopic import BERTopic
import numpy as np
import gc

def process_in_chunks(df, chunk_size=10000):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i+chunk_size]

def create_and_merge_dataset(model, filtered_df, folder_name, eurobarometer_path, google_trends_path, FT_path, chunk_size):
    try:
        print(f"Starting dataset creation and merging process with chunk size: {chunk_size}")

        # Step 1: Process BERTopic results
        print("Processing BERTopic results...")
        topics = model.topics_
        probabilities = model.probabilities_

        all_speeches = pd.DataFrame()
        chunk_count = 0
        for chunk in process_in_chunks(filtered_df, chunk_size):
            chunk_count += 1
            print(f"Processing chunk {chunk_count}")
            chunk_indices = chunk.index.tolist()
            chunk_topics = [topics[i] for i in chunk_indices if i < len(topics)]
            chunk['topic'] = chunk_topics
            prob_df = pd.DataFrame(probabilities[chunk_indices], columns=[f'Topic_{i}' for i in range(probabilities.shape[1])])
            chunk_result = pd.concat([chunk, prob_df], axis=1)
            all_speeches = pd.concat([all_speeches, chunk_result])
            del chunk, prob_df, chunk_result
            gc.collect()

        # Step 2: Prepare dates and quarters
        print("Preparing dates and quarters...")
        all_speeches['year'] = all_speeches['year'].astype(int)
        all_speeches['date'] = pd.to_datetime(all_speeches['date'])
        all_speeches['quarter'] = all_speeches['date'].dt.to_period('Q')

        # Step 3: Separate ECB and non-ECB data
        print("Separating ECB and non-ECB data...")
        ecb_data = all_speeches[all_speeches['central_bank'] == 'european central bank']
        non_ecb_data = all_speeches[all_speeches['central_bank'] != 'european central bank']
        del all_speeches
        gc.collect()

        # Step 4: Process additional variables
        print("Processing additional variables...")
        additional_vars = ['spread', 'ro_cbie_index', 'unemployment_rate', 'diff_unemployment', 'i_minus_g', 'gdp_nominal_growth', 'gdp_real_growth', 'diff_inflation', 'gdp_deflator', 'hicp', 'output_gap_pgdp', 'structural_balance_pgdp', 'total_expenditure_pgdp', 'government_debt_pgdp', 'primary_balance_pgdp', 'government_debt', 'gdp_real_ppp_capita', 'number_of_sentences', 'audience', 'position', 'speaker', 'type_of_text']

        def get_agg_method(series):
            if pd.api.types.is_numeric_dtype(series):
                return 'mean'
            else:
                return lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None

        # Step 5: Merge ECB data with aggregated values
        print("Merging ECB data with aggregated values...")
        mean_values = non_ecb_data.groupby('quarter').agg({
            var: get_agg_method(non_ecb_data[var]) for var in additional_vars
        }).reset_index()

        ecb_data_with_means = pd.merge(ecb_data, mean_values, on='quarter', suffixes=('', '_agg'))
        for var in additional_vars:
            ecb_data_with_means[var] = ecb_data_with_means[f'{var}_agg']
            ecb_data_with_means.drop(f'{var}_agg', axis=1, inplace=True)

        # Step 6: Combine data and aggregate
        print("Combining data and aggregating...")
        final_df = pd.concat([ecb_data_with_means, non_ecb_data])
        del ecb_data_with_means, non_ecb_data, ecb_data
        gc.collect()

        topic_columns = [col for col in final_df.columns if col.startswith('Topic_')]
        columns_to_aggregate = additional_vars + topic_columns

        topic_proportions_final = pd.DataFrame()
        for chunk in process_in_chunks(final_df):
            chunk_result = chunk.groupby(['quarter', 'central_bank']).agg({
                col: get_agg_method(final_df[col]) for col in columns_to_aggregate
            }).reset_index()
            topic_proportions_final = pd.concat([topic_proportions_final, chunk_result])
            del chunk_result
            gc.collect()

        topic_proportions_final['quarter'] = pd.to_datetime(topic_proportions_final['quarter'].astype(str)).dt.to_period('Q')
        del final_df
        gc.collect()

        # Step 7: Merge and rename topics NEW for topicmodel1
        print("Merging and renaming topics...")
        topic_merging = {
            'Monetary_Policy_Central_Banking': ['Topic_0', 'Topic_12', 'Topic_16', 'Topic_17', 'Topic_19', 'Topic_21', 'Topic_24'],
            'Economic_Analysis_Indicators': ['Topic_4', 'Topic_12', 'Topic_13', 'Topic_20', 'Topic_22', 'Topic_27'],
            'Financial_Markets_Integration': ['Topic_10', 'Topic_11', 'Topic_18', 'Topic_29', 'Topic_30', 'Topic_33'],
            'Banking_Regulation_Supervision': ['Topic_6', 'Topic_7', 'Topic_25', 'Topic_26', 'Topic_34', 'Topic_38'],
            'Digital_Finance_Innovation': ['Topic_2', 'Topic_28', 'Topic_36', 'Topic_40'],
            'International_Econ_Exchange': ['Topic_18', 'Topic_23', 'Topic_35'],
            'Crisis_Management_Stability': ['Topic_8', 'Topic_9', 'Topic_32', 'Topic_37', 'Topic_41'],
            'Sustainable_Finance_Climate': ['Topic_5', 'Topic_31'],
            'Payment_Systems_Cash': ['Topic_15', 'Topic_14', 'Topic_39', 'Topic_43'],
            'National_Economy': ['Topic_1', 'Topic_3', 'Topic_42', 'Topic_44']
        }

        for new_topic, old_topics in topic_merging.items():
            topic_proportions_final[new_topic] = topic_proportions_final[old_topics].sum(axis=1)

        # Calculate proportions
        sum_of_topics = topic_proportions_final[list(topic_merging.keys())].sum(axis=1)
        for topic in topic_merging.keys():
            topic_proportions_final[f'{topic}_Proportion'] = topic_proportions_final[topic] / sum_of_topics
            topic_proportions_final.drop(columns=[topic], inplace=True)

        topic_proportions_final.rename(columns={f'{topic}_Proportion': topic for topic in topic_merging.keys()}, inplace=True)
        topic_proportions_final.drop(columns=[f'Topic_{i}' for i in range(45)], inplace=True)
        gc.collect()

        # Step 8: Process Eurobarometer data
        print("Processing Eurobarometer data...")
        eurobarometer = pd.read_stata(eurobarometer_path)
        eurobarometer['year'] = pd.to_datetime(eurobarometer['year'], format='%Y').dt.year

        def generate_quarters(year):
            return [pd.Period(f"{year}-Q{q}") for q in range(1, 5)]

        quarterly_eurobarometer = pd.DataFrame()
        for _, row in eurobarometer.iterrows():
            chunk_result = pd.DataFrame({
                'year': row['year'],
                'quarter': generate_quarters(row['year']),
                'central_bank': row['central_bank'],
                **{col: row[col] for col in eurobarometer.columns if col not in ['year', 'central_bank']}
            })
            quarterly_eurobarometer = pd.concat([quarterly_eurobarometer, chunk_result])

        quarterly_eurobarometer = quarterly_eurobarometer.reset_index(drop=True)

        # Step 9: Merge with Eurobarometer data
        print("Merging with Eurobarometer data...")
        merged_df = pd.merge(topic_proportions_final, quarterly_eurobarometer, on=['quarter', 'central_bank'], how='left')
        merged_df['quarter'] = merged_df['quarter'].dt.to_timestamp()
        merged_df['quarter_str'] = merged_df['quarter'].dt.to_period('Q').astype(str)
        merged_df.columns = [col.replace(' ', '_').replace('-', '_') for col in merged_df.columns]
        del topic_proportions_final, quarterly_eurobarometer
        gc.collect()

        # Step 10: Process and merge Google Trends data
        print("Processing and merging Google Trends data...")
        google_trends = pd.read_csv(os.path.join(google_trends_path, 'ecb_trends_quarterly.csv'))
        
        print(f"Google Trends columns: {google_trends.columns}")
        print(f"Number of columns: {len(google_trends.columns)}")
        
        if 'Quarter' not in google_trends.columns:
            raise ValueError("'Quarter' column not found in Google Trends data")
        
        # Create 'central_bank' column based on existing columns
        central_banks = ['DE', 'ES', 'FR', 'IT', 'NL', 'ECB_Average']
        for bank in central_banks:
            if bank not in google_trends.columns:
                raise ValueError(f"'{bank}' column not found in Google Trends data")
        
        # Melt the dataframe to create 'central_bank' column
        google_trends_melted = pd.melt(google_trends, id_vars=['Quarter'], value_vars=central_banks, 
                                       var_name='central_bank', value_name='google_trends_value')
        
        # Map short names to full names
        bank_name_map = {
            'DE': 'deutsche bundesbank',
            'ES': 'bank of spain',
            'FR': 'bank of france',
            'IT': 'bank of italy',
            'NL': 'netherlands bank',
            'ECB_Average': 'european central bank'
        }
        google_trends_melted['central_bank'] = google_trends_melted['central_bank'].map(bank_name_map)
        
        google_trends_melted['quarter'] = pd.to_datetime(google_trends_melted['Quarter'].apply(lambda x: f"{x.split('-Q')[0]}-{int(x.split('-Q')[1])*3-2:02d}-01"))
        
        # Rename columns
        google_trends_melted.columns = ['original_quarter', 'central_bank', 'google_trends_value', 'quarter']
        
        # Ensure the number of new column names matches the number of columns
        if len(google_trends_melted.columns) != 4:
            raise ValueError(f"Column name mismatch: {len(google_trends_melted.columns)} columns")
        
        merged_df['quarter'] = pd.to_datetime(merged_df['quarter'])
        merged_df = pd.merge(merged_df, google_trends_melted, on=['quarter', 'central_bank'], how='left')
        del google_trends, google_trends_melted
        gc.collect()

        # Step 11: merge with FT data (downloaded from Factiva)
        print("Processing and merging FT data...")
        ft_frequency_df = pd.read_csv(FT_path, delimiter=';', skiprows=5)
        ft_frequency_df.columns = ['raw_data']

        # Extract start date, end date, and document count from raw data
        parsed_data = ft_frequency_df['raw_data'].str.extract(r'Start Date: (?P<start_date>.*?) End Date: (?P<end_date>.*?)\D*(?P<document_count>\d+)$')
        parsed_data['document_count'] = pd.to_numeric(parsed_data['document_count'], errors='coerce')
        parsed_data['start_date'] = pd.to_datetime(parsed_data['start_date'], format='%d %B %Y', errors='coerce')
        parsed_data['end_date'] = pd.to_datetime(parsed_data['end_date'], format='%d %B %Y', errors='coerce')

        # Create 'quarter' column and filter out 2024 data
        parsed_data['quarter'] = parsed_data['start_date'].dt.to_period('Q')
        parsed_data = parsed_data[parsed_data['start_date'].dt.year < 2024]  # Filter out 2024 data

        # Now distribute document count across all quarters of the year
        parsed_data_expanded = parsed_data.groupby(parsed_data['start_date'].dt.year).apply(
            lambda x: pd.DataFrame({
                'quarter': pd.period_range(start=f"{x.name}-Q1", end=f"{x.name}-Q4", freq='Q'),
                'document_count': x['document_count'].sum() / 4
            })
        ).reset_index(drop=True)

        # Ensure 'quarter' in merged_df is in the correct format
        merged_df['quarter'] = pd.to_datetime(merged_df['quarter']).dt.to_period('Q')

        # Merging the cleaned FT frequency data with merged_df
        merged_df = pd.merge(merged_df, parsed_data_expanded, on='quarter', how='left')

        # If there are still missing values, fill them with 0
        merged_df['document_count'] = merged_df['document_count'].fillna(0)

        # Clean-up
        del ft_frequency_df, parsed_data, parsed_data_expanded
        gc.collect()

        print("FT data merged successfully.")

        # Convert 'quarter' to datetime
        merged_df['quarter'] = merged_df['quarter'].dt.to_timestamp()

        # Step 12: Save to Stata format
        print("Saving to Stata format...")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Convert any remaining Period columns to string
        for col in merged_df.columns:
            if isinstance(merged_df[col].dtype, pd.PeriodDtype):
                merged_df[col] = merged_df[col].astype(str)

        try:
            merged_df.to_stata(os.path.join(folder_name, 'timeseries.dta'), version=118, write_index=False)
        except ValueError:
            try:
                merged_df.to_stata(os.path.join(folder_name, 'timeseries.dta'), version=117, write_index=False)
            except Exception as e:
                print(f"Error saving to Stata format: {str(e)}")
                print("Attempting to save as CSV instead...")
                merged_df.to_csv(os.path.join(folder_name, 'timeseries.csv'), index=False)

        print("Dataset creation and merging process completed successfully.")
        return merged_df

    except Exception as e:
        print(f"An error occurred during the dataset creation and merging process: {str(e)}")
        return None
    

def create_and_merge_dataset_seq(model, filtered_df, folder_name, eurobarometer_path, google_trends_path, FT_path, output_file_name="merged_dataset.dta"):
    try:
        print("Starting dataset creation and merging process")

        # Step 1: Process BERTopic results
        print("Processing BERTopic results...")
        topics = model.topics_
        probabilities = model.probabilities_

        filtered_df['topic'] = [topics[i] for i in filtered_df.index if i < len(topics)]
        prob_df = pd.DataFrame(probabilities, columns=[f'Topic_{i}' for i in range(probabilities.shape[1])])
        all_speeches = pd.concat([filtered_df.reset_index(drop=True), prob_df], axis=1)

        # Step 2: Prepare dates
        print("Preparing dates...")
        all_speeches['year'] = all_speeches['year'].astype(int)
        all_speeches['date'] = pd.to_datetime(all_speeches['date'])

        # Step 3: Process additional variables
        print("Processing additional variables...")
        additional_vars = ['spread', 'ro_cbie_index', 'unemployment_rate', 'diff_unemployment', 'i_minus_g', 'gdp_nominal_growth', 'gdp_real_growth', 'diff_inflation', 'gdp_deflator', 'hicp', 'output_gap_pgdp', 'structural_balance_pgdp', 'total_expenditure_pgdp', 'government_debt_pgdp', 'primary_balance_pgdp', 'government_debt', 'gdp_real_ppp_capita', 'number_of_sentences', 'audience', 'position', 'speaker', 'type_of_text']

        # Step 4: Merge and rename topics NEW for topicmodel1
        print("Merging and renaming topics...")
        topic_merging = {
            'Monetary_Policy_Central_Banking': ['Topic_0', 'Topic_12', 'Topic_16', 'Topic_17', 'Topic_19', 'Topic_21', 'Topic_24'],
            'Economic_Analysis_Indicators': ['Topic_4', 'Topic_12', 'Topic_13', 'Topic_20', 'Topic_22', 'Topic_27'],
            'Financial_Markets_Integration': ['Topic_10', 'Topic_11', 'Topic_18', 'Topic_29', 'Topic_30', 'Topic_33'],
            'Banking_Regulation_Supervision': ['Topic_6', 'Topic_7', 'Topic_25', 'Topic_26', 'Topic_34', 'Topic_38'],
            'Digital_Finance_Innovation': ['Topic_2', 'Topic_28', 'Topic_36', 'Topic_40'],
            'International_Econ_Exchange': ['Topic_18', 'Topic_23', 'Topic_35'],
            'Crisis_Management_Stability': ['Topic_8', 'Topic_9', 'Topic_32', 'Topic_37', 'Topic_41'],
            'Sustainable_Finance_Climate': ['Topic_5', 'Topic_31'],
            'Payment_Systems_Cash': ['Topic_15', 'Topic_14', 'Topic_39', 'Topic_43'],
            'National_Economy': ['Topic_1', 'Topic_3', 'Topic_42', 'Topic_44']
        }
        for new_topic, old_topics in topic_merging.items():
            all_speeches[new_topic] = all_speeches[old_topics].sum(axis=1)

        # Calculate proportions
        sum_of_topics = all_speeches[list(topic_merging.keys())].sum(axis=1)
        for topic in topic_merging.keys():
            all_speeches[f'{topic}_Proportion'] = all_speeches[topic] / sum_of_topics
            all_speeches.drop(columns=[topic], inplace=True)
        all_speeches.rename(columns={f'{topic}_Proportion': topic for topic in topic_merging.keys()}, inplace=True)
        all_speeches.drop(columns=[f'Topic_{i}' for i in range(45)], inplace=True)
        # Step 5: Clean column names for Stata compatibility
        print("Cleaning column names for Stata compatibility...")
        all_speeches.columns = all_speeches.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)  # Remove special characters
        all_speeches.columns = all_speeches.columns.str[:32]  # Limit column names to 32 characters
        
        # Step 6: Clean non-ASCII characters from the data itself
        print("Cleaning non-ASCII characters from data...")
        def clean_data(value):
            if isinstance(value, str):
                return value.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters
            return value
        
        all_speeches = all_speeches.applymap(clean_data)
        # Step 7: Handle problematic columns ('pegged', 'member_of_currency_union')
        problematic_columns = ['pegged', 'member_of_currency_union']
        for column in problematic_columns:
            if column in all_speeches.columns:
                print(f"Cleaning the '{column}' column...")
                # Drop the column if it's empty or has invalid data
                if all_speeches[column].isnull().all():
                    print(f"'{column}' column contains only null values, dropping it.")
                    all_speeches.drop(columns=[column], inplace=True)
                else:
                    # Convert to strings, replacing NaN with empty strings
                    all_speeches[column] = all_speeches[column].astype(str).replace('nan', '')
        # Step 8: Drop 'speech_text' column as requested
        print("Dropping 'speech_text' column...")
        if 'speech_text' in all_speeches.columns:
            all_speeches.drop(columns=['speech_text'], inplace=True)
        # Step 9: Save as a .dta file with version 117 (Stata 13+)
        print(f"Saving dataset to {output_file_name}...")
        output_path = os.path.join(folder_name, output_file_name)
        all_speeches.to_stata(output_path, version=117)
        print("Dataset creation, merging, and saving process completed successfully.")
        return all_speeches

    except Exception as e:
        print(f"An error occurred during the dataset creation and merging process: {str(e)}")
        return None

def create_and_merge_dataset_halfyear(model, filtered_df, folder_name, eurobarometer_path, google_trends_path, FT_path, chunk_size=10000):
    try:
        print(f"Starting dataset creation and merging process (half-year) with chunk size: {chunk_size}")

        # Step 1: Process BERTopic results (same as before)
        print("Processing BERTopic results...")
        topics = model.topics_
        probabilities = model.probabilities_

        all_speeches = pd.DataFrame()
        chunk_count = 0
        for chunk in process_in_chunks(filtered_df, chunk_size):
            chunk_count += 1
            print(f"Processing chunk {chunk_count}")
            chunk_indices = chunk.index.tolist()
            chunk_topics = [topics[i] for i in chunk_indices if i < len(topics)]
            chunk['topic'] = chunk_topics
            prob_df = pd.DataFrame(probabilities[chunk_indices], columns=[f'Topic_{i}' for i in range(probabilities.shape[1])])
            chunk_result = pd.concat([chunk, prob_df], axis=1)
            all_speeches = pd.concat([all_speeches, chunk_result])
            del chunk, prob_df, chunk_result
            gc.collect()

        # After processing BERTopic results, convert topic columns to numeric
        topic_columns = [col for col in all_speeches.columns if col.startswith('Topic_')]
        for col in topic_columns:
            all_speeches[col] = pd.to_numeric(all_speeches[col], errors='coerce')

        # Updated Step 2: Prepare dates and half-years
        print("Preparing dates and half-years...")
        all_speeches['date'] = pd.to_datetime(all_speeches['date'])
        all_speeches['year'] = all_speeches['date'].dt.year
        # First half: months 1-6, Second half: months 7-12
        all_speeches['half'] = (all_speeches['date'].dt.month > 6).astype(int) + 1
        all_speeches['halfyear'] = pd.to_datetime(
            all_speeches.apply(
                lambda x: f"{x['year']}-{6 if x['half'] == 1 else 12}-30", 
                axis=1
            )
        )

        # Step 3: Separate ECB and non-ECB data
        print("Separating ECB and non-ECB data...")
        ecb_data = all_speeches[all_speeches['central_bank'] == 'european central bank']
        non_ecb_data = all_speeches[all_speeches['central_bank'] != 'european central bank']
        del all_speeches
        gc.collect()

        # Modified Step 4: Define aggregation methods with better type handling
        print("Defining aggregation methods...")
        numeric_vars = ['spread', 'ro_cbie_index', 'unemployment_rate', 'diff_unemployment', 
                       'i_minus_g', 'gdp_nominal_growth', 'gdp_real_growth', 'diff_inflation', 
                       'gdp_deflator', 'hicp', 'output_gap_pgdp', 'structural_balance_pgdp', 
                       'total_expenditure_pgdp', 'government_debt_pgdp', 'primary_balance_pgdp', 
                       'government_debt', 'gdp_real_ppp_capita', 'number_of_sentences'] + topic_columns
        
        categorical_vars = ['audience', 'position', 'speaker', 'type_of_text']
        
        # Convert numeric columns and create aggregation dictionary
        agg_dict = {}
        for var in numeric_vars:
            if var in non_ecb_data.columns:
                non_ecb_data[var] = pd.to_numeric(non_ecb_data[var], errors='coerce')
                agg_dict[var] = 'mean'
        
        for var in categorical_vars:
            if var in non_ecb_data.columns:
                agg_dict[var] = lambda x: x.mode().iloc[0] if not x.empty and len(x.mode()) > 0 else None

        # First aggregate non-ECB data
        print("Aggregating non-ECB data...")
        mean_values = non_ecb_data.groupby(['halfyear', 'central_bank']).agg(agg_dict).reset_index()
        
        # Calculate mean values across all non-ECB banks for each period
        print("Calculating average values for ECB template...")
        numeric_cols = [col for col in mean_values.columns if col != 'central_bank' and pd.api.types.is_numeric_dtype(mean_values[col])]
        ecb_means = mean_values.groupby('halfyear')[numeric_cols].mean().reset_index()
        
        # Get mode for categorical variables
        categorical_cols = [col for col in mean_values.columns if col != 'central_bank' and not pd.api.types.is_numeric_dtype(mean_values[col])]
        if categorical_cols:
            cat_modes = mean_values.groupby('halfyear')[categorical_cols].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None
            ).reset_index()
            ecb_means = pd.merge(ecb_means, cat_modes, on='halfyear')

        # Create ECB template and continue with merging
        all_halfyears = pd.DataFrame({'halfyear': pd.date_range(
            start=non_ecb_data['halfyear'].min(),
            end=non_ecb_data['halfyear'].max(),
            freq='6M'
        )})

        # Create complete ECB template with mean values
        ecb_template = pd.DataFrame({
            'halfyear': all_halfyears['halfyear'],
            'central_bank': 'european central bank'
        })
        
        # Merge template with means
        ecb_template = pd.merge(ecb_template, ecb_means, on='halfyear', how='left')
        
        # Now merge with actual ECB data, preferring actual data over template data
        ecb_data_final = pd.merge(ecb_template, ecb_data, on=['halfyear', 'central_bank'], how='left', suffixes=('_template', ''))
        
        # For each column, use actual ECB data where available, fall back to template data where not
        for col in ecb_data_final.columns:
            if col.endswith('_template'):
                base_col = col[:-9]  # remove '_template' suffix
                if base_col in ecb_data_final.columns:
                    ecb_data_final[base_col] = ecb_data_final[base_col].fillna(ecb_data_final[col])
                    ecb_data_final.drop(columns=[col], inplace=True)
        
        # Combine final datasets
        print("Combining ECB and non-ECB data...")
        final_df = pd.concat([mean_values, ecb_data_final])
        del mean_values, ecb_data_final, ecb_template, non_ecb_data, ecb_data
        gc.collect()


        # Save to Stata format with improved column name handling
        print("Saving to Stata format...")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        output_file = os.path.join(folder_name, 'timeseries_halfyear.dta')
        
        # Clean up data for Stata compatibility
        print("Cleaning column names and data for Stata compatibility...")
        
        # Function to make Stata-compliant column names
        def make_stata_compliant(name):
            # Remove non-alphanumeric characters (except underscores)
            name = ''.join(c if c.isalnum() or c == '_' else '' for c in name)
            # Ensure name starts with a letter
            if not name[0].isalpha():
                name = 'v_' + name
            # Truncate to 31 characters to allow for suffixes
            return name[:31]
        
        # Clean column names
        final_df.columns = [make_stata_compliant(col) for col in final_df.columns]
        
        # Clean string data
        for col in final_df.columns:
            if final_df[col].dtype == 'object':
                final_df[col] = final_df[col].astype(str).apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
        
        # Convert any Period or Timestamp columns to datetime
        for col in final_df.columns:
            if isinstance(final_df[col].dtype, pd.PeriodDtype):
                final_df[col] = final_df[col].dt.to_timestamp()
            if pd.api.types.is_datetime64_any_dtype(final_df[col]):
                final_df[col] = pd.to_datetime(final_df[col])

        try:
            final_df.to_stata(output_file, version=118, write_index=False)
            print(f"Successfully saved to {output_file}")
        except Exception as e:
            print(f"Error with version 118, trying version 117: {str(e)}")
            try:
                final_df.to_stata(output_file, version=117, write_index=False)
                print(f"Successfully saved to {output_file} with version 117")
            except Exception as e:
                print(f"Error saving to Stata format: {str(e)}")
                # Add error handling for CSV save
                csv_file = os.path.join(folder_name, 'timeseries_halfyear.csv')
                try:
                    final_df.to_csv(csv_file, index=False, encoding='utf-8')
                    print(f"Saved as CSV instead: {csv_file}")
                except Exception as csv_error:
                    print(f"Error saving CSV: {str(csv_error)}")
                    alternative_file = os.path.join(os.path.dirname(folder_name), 'timeseries_halfyear.csv')
                    final_df.to_csv(alternative_file, index=False, encoding='utf-8')
                    print(f"Saved CSV to alternative location: {alternative_file}")

        print("Half-year dataset creation and merging process completed successfully.")
        return final_df

    except Exception as e:
        print(f"An error occurred during the half-year dataset creation and merging process: {str(e)}")
        return None


def convert_quarterly_to_halfyear(input_file, output_file):
    """
    Convert quarterly data to half-yearly by averaging two quarters.
    
    Args:
        input_file (str): Path to the input quarterly .dta file
        output_file (str): Path where to save the half-yearly .dta file
    """
    try:
        # Read the quarterly data
        df = pd.read_stata(input_file)
        
        # Convert quarter to datetime if it's not already
        df['quarter'] = pd.to_datetime(df['quarter'])
        
        # Create half-year period
        df['halfyear'] = df['quarter'].dt.year.astype(str) + 'H' + ((df['quarter'].dt.quarter + 1) // 2).astype(str)
        
        # Identify numeric columns for averaging
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Identify non-numeric columns for mode, excluding grouping columns
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        cols_to_exclude = ['halfyear', 'central_bank']
        non_numeric_columns = [col for col in non_numeric_columns if col not in cols_to_exclude]
        
        # Create aggregation dictionary
        agg_dict = {col: 'mean' for col in numeric_columns}
        agg_dict.update({col: lambda x: x.mode()[0] if not x.empty else None for col in non_numeric_columns})
        
        # Group by halfyear and central_bank, then aggregate
        halfyear_df = df.groupby(['halfyear', 'central_bank'], as_index=False).agg(agg_dict)
        
        # Convert halfyear to datetime for Stata compatibility
        halfyear_df['halfyear'] = pd.to_datetime(halfyear_df['halfyear'].apply(
            lambda x: f"{x[0:4]}-{'06' if x.endswith('H1') else '12'}-30"
        ))
        
        # Save to Stata format
        try:
            halfyear_df.to_stata(output_file, version=118, write_index=False)
        except:
            halfyear_df.to_stata(output_file, version=117, write_index=False)
        
        print(f"Successfully converted quarterly data to half-yearly and saved to {output_file}")
        return halfyear_df
        
    except Exception as e:
        print(f"Error converting to half-yearly data: {str(e)}")
        return None

# Example usage:
input_file = r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\PhD\two_thirds_submission\Github_replication_files_by_paper\Paper_1\output\timeseries.dta"
output_file = r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\PhD\two_thirds_submission\Github_replication_files_by_paper\Paper_1\output\half_timeseries.dta"
halfyear_data = convert_quarterly_to_halfyear(input_file, output_file)


def convert_quarterly_to_yearly(input_file, output_file):
    """
    Convert quarterly data to yearly by averaging four quarters.
    
    Args:
        input_file (str): Path to the input quarterly .dta file
        output_file (str): Path where to save the yearly .dta file
    """
    try:
        # Read the quarterly data
        df = pd.read_stata(input_file)
        
        # Convert quarter to datetime if it's not already
        df['quarter'] = pd.to_datetime(df['quarter'])
        
        # Create year as integer
        df['year'] = df['quarter'].dt.year.astype(int)
        
        # Identify numeric columns for averaging
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Identify non-numeric columns for mode, excluding grouping columns
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        cols_to_exclude = ['year', 'quarter', 'central_bank']
        non_numeric_columns = [col for col in non_numeric_columns if col not in cols_to_exclude]
        
        # Create aggregation dictionary
        agg_dict = {col: 'mean' for col in numeric_columns}
        agg_dict.update({col: lambda x: x.mode()[0] if not x.empty else None for col in non_numeric_columns})
        
        # Group by year and central_bank, then aggregate
        yearly_df = df.groupby(['year', 'central_bank'], as_index=False).agg(agg_dict)
        
        # Ensure year is integer and create year_date
        yearly_df['year'] = yearly_df['year'].astype(int)
        yearly_df['year_date'] = pd.to_datetime([f"{year}-12-31" for year in yearly_df['year']])
        
        # Save to Stata format
        try:
            yearly_df.to_stata(output_file, version=118, write_index=False)
        except:
            yearly_df.to_stata(output_file, version=117, write_index=False)
        
        print(f"Successfully converted quarterly data to yearly and saved to {output_file}")
        return yearly_df
        
    except Exception as e:
        print(f"Error converting to yearly data: {str(e)}")
        return None


# Example usage:
input_file = r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\PhD\two_thirds_submission\Github_replication_files_by_paper\Paper_1\output\timeseries.dta"
output_file = r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\PhD\two_thirds_submission\Github_replication_files_by_paper\Paper_1\output\year_timeseries.dta"
halfyear_data = convert_quarterly_to_yearly(input_file, output_file)


# Example usage:
# input_file = "path/to/quarterly/timeseries.dta"
# output_file = "path/to/yearly/timeseries_yearly.dta"
# yearly_data = convert_quarterly_to_yearly(input_file, output_file)
