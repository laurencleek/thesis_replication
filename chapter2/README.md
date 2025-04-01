# Overview
This repository contains replication code for "_How Central Bank Independence Shapes Monetary Policy Communication: A Large Language Model Application_".
It includes code to:
- Calculate yearly indices of dominance and coordination based on sentence-level classifications using the Gemini LLM
- Merge the aggregated speeches dataset with other datasets as described in the paper
- Run the empirical analysis and produce tables and figures of both the appendix and the main text

# Repository structure
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


# Instructions to run code
Our code can be grouped into 3 parts: 
1. LLM fine-tuning, metadata extraction, and classification
2. Dataset aggregation
3. Empirical analysis

All three steps can be run independently. We provide the outputs of each step in `data/processed/`. To replicate results and work with the data from the paper, we strongly recommend to start with step 2 or 3, as running the entire Gemini process is impractical and complicated to setup (see details below). To quickly produce the tables and figures, run `run_graphs_and_tables.R`, which will procude all tables and figures covered by the replication code except for Figure A14 and Table A8, which are in self-contained python codes `codes/figures/figure_a14.py` and `codes/tables/table_a8.py` respectively.

## LLM fine-tuning, metadata extraction, and classification
- The first step is the speeches data preprocessing. This is done in `llm/speech_preprocessing.py`. This script starts of with the [BIS speeches dataset](https://www.bis.org/cbspeeches/download.htm), merges manual fixes of some speech metadata, cleans the speech text using extensive regex-based preprocessing to remove metadata, footnotes, and non-sentence content. It tokenizes the cleaned text into individual sentences, filters for meaningful sentences based on length and character composition. Since the speeches file is too large for Github the most recent version of the speeches can either be downloaded from the BIS website or our version is taken from OSF mirror of the replication files ([Link](https://osf.io/gpk7r/files/osfstorage/67eab6793d01252c338b7fd0))
- Next, we extract structured metadata such as speaker name, position, central bank, venue, and audience from the description strings contained in the BIS speeches dataset. See `llm/metadata_extraction.py`.
- Logically next, follows the fine-tunig of the Gemini model and finally the classification of the entire dataset using the Gemini LLM. The code for this is in `llm/gemini_fine_tuning.py`. For details on the exact procedure, see the Appendix of the paper.

The model fine-tuning, metadata extraction, and classification of the full dataset form the basis for further analysis. While the LLM code should run in principle, it is provided mainly for reference. The Gemini code requires the Google Cloud SDK to be installed locally and a cloud project configured with OAuth authentication for fine-tuning to be accessible. Also, the fine-tuning is not deterministic and thus the fine-tuned model will be marginally different from our fine-tune. It is presently not possible to share fine-tuned models across accounts. Further, classifying the entire dataset is very time and cost-intensive. (Gemini has a free tier, but with current rate limits, it takes months to process the entire dataset). Results may not exactly replicate due to potential model changes and inherent variation in model responses. However, we expect that most classifications would be the same in a complete rerun. ``

## Dataset aggregation
- We provide all sentence-level classifications obtained from our Gemini run in `data/processed/dom_corp_classification_sentence_level.parquet`, which can be merged with the sentences themselves in `data/processed/sentence_level_bis.parquet`. 
- To produce our dataset, which forms the basis for the instrumental variable (IV) and difference-in-differences (DiD) analysis, the sentence-level classifications are aggregated and merged with additional datasets in `dataset_processing/merge_datasets.R`. 
- The necessary files from other datasets are provided in the subfolders of `data/input/`, with the exception of the V-Dem dataset and the adjustments we make to the currency peg dataset. For these datasets, run `dataset_processing/prepare_currency_peg_file.R` and `dataset_processing/vdem.R`.

## Empirical analysis
For the empirical analysis, we provide self-contained code for each figure and table. For the difference-in-differences analysis, they load functions from the `codes/functions` folder. 

The `diff_in_diff_functions.R` contains most of the logic of the DiD analysis. It implements functions to run different estimators, placebo tests, treatment indicator definitions, and sample configurations, which form the basis for the numerous robustness checks conducted in our paper.

To run the R codes the following packages need to be installed:
```R
# These packages are on CRAN:
install.packages(c("tidyverse", "readxl", "haven", "lubridate", "zoo", "arrow", "countrycode", "vdemdata", "patchwork", "binsreg", "slider", "ggpattern", "geosphere", "sf", "rnaturalearth", "knitr", "kableExtra", "ggbeeswarm", "ggridges", "did", "did2s", "fixest",  "eventstudyr"))

# For the didimputation package, the development version should be installed as otherwise specifying the leads/lags as it is done in the code will not work
devtools::install_github("kylebutts/didimputation")

# Figure 5 in the paper uses ggsankey which is not on CRAN:
devtools::install_github("davidsjoberg/ggsankey")
```
R codes should always be run with the working directory set to the root of the folder structure. Python codes expect to be run from the folder they are in. To replicate a specific figure it is recommended to look for the figure under `codes/figures`. To produce all R graphs at once run `run_graphs_and_tables.R`. 
