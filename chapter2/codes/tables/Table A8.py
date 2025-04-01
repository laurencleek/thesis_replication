# Python based graph and tables. This part of the code should work on its own. Reimport everything.
import sys
import os

# Get the parent directory two levels up
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../llm'))
sys.path.append(parent_dir)

from llm_functions import *

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

holdout_sample_classification = pd.read_pickle("../../data/processed/fine_tuning/holdout_validation.pkl")

# Leek et al. 2024 validation set
leek_et_al_2024_validation = pd.read_pickle("../../data/input/validation/leek_et_al_2024_validation.pkl")
leek_et_al_2024_validation = fix_labels(leek_et_al_2024_validation, mapping = {
            "monetary dominance" : "monetary dominance",
            "financial dominance" : "financial dominance",
            "fiscal dominance" : "fiscal dominance",
            "monetary-fiscal coordination" : "monetary-fiscal coordination",
            "fiscal-monetary coordination" : "monetary-fiscal coordination",
            "Monetary coordination" : "monetary-fiscal coordination",
            "monetary-financial coordination" : "monetary-financial coordination",
            "Monetary-prudential coordination" : "monetary-financial coordination",
            "none" : "none",
            "monetary" : "monetary dominance",
            "financial" : "financial dominance",
            "Financial regulation dominance" : "monetary dominance"
}, drop_unknown = False, unkown_category= "none")

validation_sample = pd.read_excel("../../data/input//validation/validation_sample_all.xlsx")

np.random.seed(3)
validation_sample = validation_sample.assign(
         A_level_1 = validation_sample[["S_level_1", "M_level_1", "L_level_1"]].mode(axis = 1).apply(lambda x: x.dropna().sample(1).item(), axis = 1),
         A_level_1_agree = 4 - validation_sample[["S_level_1", "M_level_1", "L_level_1"]].nunique(axis = 1),
         A_level_2 = validation_sample[["S_level_2", "M_level_2", "L_level_2"]].mode(axis = 1).apply(lambda x: x.dropna().sample(1).item(), axis = 1),
         A_level_2_agree = 4 - validation_sample[["S_level_2", "M_level_2", "L_level_2"]].nunique(axis = 1),
         A_level_3 = validation_sample[["S_level_3", "M_level_3", "L_level_3"]].mode(axis = 1).apply(lambda x: x.dropna().sample(1).item(), axis = 1),
         A_level_3_agree = 4 - validation_sample[["S_level_3", "M_level_3", "L_level_3"]].nunique(axis = 1),
)

# Split validation_sample
fine_tuning = validation_sample.sample(frac = 0.3,random_state=444)
holdout = validation_sample.drop(fine_tuning.index)

# Calculate numbers for LLM comparison table
calculate_scores(holdout["A_level_3"], leek_et_al_2024_validation["classification"]) # Check we get same number for previous paper.
calculate_scores(holdout["A_level_3"], holdout_sample_classification["classification"]) # New model performance.