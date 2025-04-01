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

sns.set(font_scale=1.65)
plt.rcParams["font.family"] = "Palatino Linotype"
plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(15, 7.5))  # Adjusted figure size for side-by-side plots
gs = gridspec.GridSpec(1, 2, figure=fig)  # Only one row, two columns
gs.update(wspace=0.05, hspace=0.6)  # Adjust spacing between plots, reduce width spacing

ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1], sharey=ax1)  # Share the y-axis with ax1

# Calculate the confusion matrices
conf_matrix1 = calculate_confusion(holdout["A_level_3"].reset_index(drop=True),
                                   holdout_sample_classification["classification"].reset_index(drop=True))
conf_matrix2 = calculate_confusion(holdout["A_level_3"].reset_index(drop=True),
                                   leek_et_al_2024_validation["classification"].reset_index(drop=True))

# Determine the color scale limits
vmin = min(conf_matrix1.min().min(), conf_matrix2.min().min())
vmax = max(conf_matrix1.max().max(), conf_matrix2.max().max())

ax1.set_title("Gemini Pro 1.0 fine-tune", fontsize=25, pad=15)
sns.heatmap(conf_matrix1, annot=True, cmap='Blues', cbar=True, ax=ax1, fmt=".2f", vmin=vmin, vmax=vmax)
ax1.set(xlabel='Predicted labels (share)', ylabel='True label')

ax2.set_title("ChatGPT 3.5 (Leek et al. 2024)", fontsize=25, pad=15)
sns.heatmap(conf_matrix2, annot=True, cmap='Greens', cbar=True, ax=ax2, fmt=".2f", vmin=vmin, vmax=vmax)
ax2.set(xlabel='Predicted labels (share)', ylabel='')
ax2.tick_params(axis='y', left=False, right=False, labelleft=False)

fig.savefig('../../output/figures/llm_validation/confusion_matrices.pdf',bbox_inches="tight")