# Thesis Replication: The Rise of Central Bank Talk: Essays in Central bank Communication and Independence

This folder has the following structure:

The project consists of 3 empirical chapters:

## Chapter 2 
This chapter is currently under review and will upon publication be made available.

### Overview
This repository contains replication code for "_How Central Bank Independence Shapes Monetary Policy Communication: A Large Language Model Application_".
It includes code to:
- Calculate yearly indices of dominance and coordination based on sentence-level classifications using the Gemini LLM
- Merge the aggregated speeches dataset with other datasets as described in the paper
- Run the empirical analysis and produce tables and figures of both the appendix and the main text

### Repository Structure
The repository is structured as follows:
```
cbi-llm
│   Readme.md
│   run_graphs_and_tables.R
│
├───codes
│   ├───constants
│   ├───dataset_processing
│   ├───figures
│   ├───functions
│   ├───llm
│   └───tables
│
├───data
│   ├───input
│   └───processed
│
└───output
    ├───figures
    └───tables
```

All codes are stored in the `codes` folder grouped by function. There are self-contained functions to produce figures and tables from the paper in the `figures` and `tables` sub-folders. Settings and variables names are stored in the `constants` folder. `dataset_processing`  contains the codes to produce our dataset, including calculating the textual measures of central bank communication from the sentence level classification and merging the further datasets mentioned in the paper. Codes to run the Gemini classification, fine-tune a model and run the topic model are in the `llm` folder. The data folder contains `input` files and intermediate processing data files which are stored under `data/processed`. Graphs and tables, are stored under `output`.


### Instructions to Run Code
Our code can be grouped into 3 parts: 
1. LLM fine-tuning, metadata extraction, and classification
2. Dataset aggregation
3. Empirical analysis

All three steps can be run independently. We provide the outputs of each step in `data/processed/`. To replicate results and work with the data from the paper, we strongly recommend to start with step 2 or 3, as running the entire Gemini process is impractical and complicated to setup (see details below). To quickly produce the tables and figures, run `run_graphs_and_tables.R`.

- The model fine-tuning, metadata extraction, and classification of the full dataset form the basis for further analysis. While the LLM code should run in principle, it is provided mainly for reference. The Gemini code requires the Google Cloud SDK to be installed locally and a cloud project configured with OAuth authentication for fine-tuning to be accessible. Also, the fine-tuning is not deterministic and thus the fine-tuned model will be marginally different from our fine-tune. It is presently not possible to share fine-tuned models across accounts. Further, classifying the entire dataset is very time and cost-intensive. (Outside of Europe, Gemini has a free tier, but with current rate limits, it takes months to process the entire dataset). Results may not exactly replicate due to potential model changes and inherent variation in model responses. However, we expect that most classifications would be the same in a complete rerun. We provide all sentence-level classifications obtained from our Gemini run in `data/processed/dom_corp_classification_sentence_level.parquet`, which can be merged with the sentences themselves in `data/processed/sentence_level_bis.parquet`.

- To produce our dataset, which forms the basis for the instrumental variable (IV) and difference-in-differences (DiD) analysis, the sentence-level classifications are aggregated and merged with additional datasets in `dataset_processing/merge_datasets.R`. The necessary files from other datasets are provided in the subfolders of `data/input/`, with the exception of the V-Dem dataset and the adjustments we make to the currency peg dataset. For these datasets, run `dataset_processing/prepare_currency_peg_file.R` and `dataset_processing/vdem.R`.

- For the empirical analysis, we provide self-contained code for each figure and table. For the difference-in-differences analysis, they load functions from the `codes/functions` folder. The `diff_in_diff_functions.R` contains most of the logic of the DiD analysis. It implements functions to run different estimators, placebo tests, treatment indicator definitions, and sample configurations, which form the basis for the numerous robustness checks conducted in our paper.
- To run the R codes the following packages need to be installed:
```R
# These packages are on CRAN:
install.packages(c("tidyverse", "readxl", "haven", "lubridate", "zoo", "arrow", "countrycode", "vdemdata", "patchwork", "binsreg", "slider", "ggpattern", "geosphere", "sf", "rnaturalearth", "knitr", "kableExtra", "ggbeeswarm", "ggridges", "did", "did2s", "fixest",  "eventstudyr"))

# For the didimputation package, the development version should be installed as otherwise specifying the leads/lags as it is done in the code will not work
devtools::install_github("kylebutts/didimputation")

# Figure 5 in the paper uses ggsankey which is not on CRAN:
devtools::install_github("davidsjoberg/ggsankey")
```
R codes should always be run with the working directory set to the root of the folder structure. Python codes expect to be run from the folder they are in. To replicate a specific figure it is recommended to look for the figure under `codes/figures`. To produce all graphs at once run `run_graphs_and_tables.R`. 


## Chapter 3
- `code/py/`: Python source code organized by functionality
  - `main/`: Core functionality modules
  - `sequence/`: Sequence analysis modules  
  - `timeseries/`: Time series analysis modules
- `data/`: Input data files
  - `Eurobarometer/`: Survey data
  - `google_trends/`: Google Trends data
  - Speech data and FT frequency data
- `output/`: Generated output files
  - Figures for paper and appendix
  - Tables for paper and appendix  
  - Various timeseries datasets
- Root level analysis scripts:
  - `main.py`: Main execution script
  - `analysis_sequence.py`: Sequence analysis
  - `analysis_timeseries.py`: Time series analysis


## Chapter 4
This repository contains replication codes for ["Introducing a Textual Measure of Central Bank Policy Interactions Using ChatGPT"](https://osf.io/preprints/socarxiv/78wnp).
It contains all the codes to pre-process the dataset, run ChatGPT on two million sentences, and finally produce our indicator and aggregated results.
Moreover, we provide our manually classified validation sample `inputdata/validation_sample_all.xlsx` and the codes to conduct prompt engineering experiments, fine-tune GPT-3.5, and assess the classification quality of various ChatGPT models and Gemini Pro against this validation set.
We share a yearly aggregation of our index `dominance_coordination_dataset.csv`. This file is sufficient to produce all charts inside the appendix and main part of the paper. Importantly, we don't include any speeches or sentence-level results. The output files are more than a gigabyte in size and too large for this repository. To rerun the full analysis, the speech data would need to be scraped with the python code [here](https://github.com/HanssonMagnus/scrape_bis). We do, however, provide the sentence-level classification of our prompt engineering results, validation exercise, and model comparisons. These are stored as Pandas DataFrames in `.pkl` format inside the `outputdata` folder.

### Instructions to run codes
- To rerun any of our analyses, an API key for ChatGPT and/or Gemini needs to be set inside the `llm_functions.py` file. Also note that these LLMs, even at a temperature set to zero, are non-deterministic. Exact results vary with each run, although with ChatGPT, usually 97%-99% of sentences are identically classified across two runs. In addition, changes to the model on OpenAI's/Google's side can impact results.
- To run R codes, the working directory should be set to the root of the project.
- Python codes expect to be run from the folder they are in.
- Validation, prompt engineering, and model comparison codes are self-contained and can be run with the inputs provided inside this repository, provided that an API key is set.

### Included files
The codes folder contains the following files:
- `0_text_preprocessing.py` This file runs the preprocessing steps described in the appendix.
- `1_chat_gpt_main_analysis.py` This code consists of the code required to run the full dataset. It requires the output produced by `0_text_preprocessing.py`.
- `2_validation_and_robustness.py` This file contains the code for the robustness checks, prompt engineering results, and different GPT versions. It requires only our validation set as input `validation_sample_all.xlsx`.
- `3_fine_tuning_and_few_shot.py` This file constructs a training dataset from our validation set, trains a fine-tuned GPT 3.5 model, and evaluates it with the remaining sample. Moreover, it contains code to run Gemini Pro using (i) the same prompts as ChatGPT and (ii) a few-shot prompting strategy.
- `llm_functions.py` Functions that are shared by the python codes are in this file. Most notably, it contains the function that takes a dataframe as input and calls either the Gemini or ChatGPT API with our prompt design. This function allows for parallel API queries to maximize rate limits.
- `merge_datasets.R` This R code calculates our relative indicator of dominance and coordination. It requires the outputs saved by `1_chat_gpt_main_analysis.py`. It also sketches how our shared dataset `dominance_coordination_dataset.csv` is produced (without including the third-party data sources).
- `run_all_charts.R` Produces all of the charts.

### Replication of Charts:
All our charts can be replicated with the R codes inside the `codes/figures` folder. Run `run_all_charts.R` to produce all charts. The R files read from the ChatGPT results provided inside the `outputdata` and the yearly aggregation of the full dataset `dominance_coordination_dataset.csv`. No access to ChatGPT is required to produce the charts. These are the files to produce the charts:
- `bin_scatter.R` Scatter charts presenting the development of dominance/coordination over time.
- `correlation.R` Pooled regressions.
- `crisis.R` Differences in fiscal dominance in crisis vs. non-crisis years.
- `levels_over_time.R` Shows the development of all three classification levels over time.
- `sentence_count_charts.R` Prompt engineering regarding the number of sentences.
- `stability.R` Stability of ChatGPT vs. Uncertainty in human coding.
- `temperature_charts.R` Prompt engineering regarding the temperature setting.
Common functions and settings to change the size of the charts are inside `functions_and_settings.R`.

### Prompts
The instructions part of our prompts are stored in the `prompts` folder. The sentences/excerpts are automatically appended to the prompt. We use a `.yaml` format to store the prompts. Our final instructions for level 1, level 2 and level 3 are in the `l1`, `l2`, `l3` subfolders. To change the prompts either modify the prompt file or modify the python code to load a different prompt.