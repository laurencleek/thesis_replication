# These ipython commands autoreload when changes occur to packages
#%autoreload 2
#%load_ext autoreload

import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import google.generativeai as genai
from google.oauth2.credentials import Credentials
from tenacity import retry, stop_after_attempt, wait_fixed
import ctypes
import optuna
import joblib

from llm_functions import *

# The fine tuning needs to be authenticated with OAuth (API KEY not enough!) which requires Google Cloud CLI utils to be installed and also a Cloud Project with the Gemini API activated
# gcloud auth application-default login --client-id-file=client_secret.json --scopes="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.tuning"

# This file fine-tunes a Gemini model and runs on full dataset. It provided mainly for reference as runnig the file is impractical for the following reasons
# (i) Both fine-tuning and inference take a long time to run and are non-deterministic (and potentially subject to model changes on Google's side).
# (ii) Running inference is subject to restrictive rate limits which makes it currently close to impossible to run the full dataset on the Gemini free plan. With current rate limits on the free tier it would take a couple of months to run everything
# (iii) Accessing our exact fine tune is not possible as it cannot be shared across Google accounts. It is however possible to train a fine-tune with identical settings as ours.
#
# It is therefore strongly recommended to take the classifications as is and start replicating the paper from the dataset aggregation step taking the Gemini classifications as provided.


gemini_classification_prompt = """You will be provided with excerpts from a central bank speech. If the information contained in the excerpt allows for it, label these excerpts according to the policy approach regarding monetary policy. Choose from one of the following policy approaches:
“monetary dominance”, i.e., the central bank prioritizes to maintain price stability, and its monetary policy is not subordinated to fiscal policy or to financial stability considerations.
“fiscal dominance”, i.e., the central bank accommodates its monetary policy to fiscal considerations, and its decisions are subordinated to meet the demands fiscal policy.
“financial dominance”, i.e., the central bank accommodates its monetary policy to financial considerations, and its decisions are subordinated to respond to the needs of financial markets.
“monetary-fiscal coordination”, i.e., the central bank suggests to cooperate with the governments to better mutually align policies.
“monetary-financial coordination”, i.e., the central bank suggests to cooperate with financial market participants to better mutually align policies.

If the excerpt discusses topics unrelated to the interaction of monetary policy with fiscal policy and financial markets or is purely descriptive without implying a policy approach label as: 
“none”, i.e., the excerpt does not contain information regarding the policy approach taken by the bank.

Assign one of the three dominance categories if a hierarchy of actors is apparent in the excerpt, i.e., monetary dominance if the central bank prescribes actions to others or acts irrespectively of the needs of other actors. Financial or fiscal dominance if the central bank is subordinating its monetary policy. Assign one of the two coordination categories if the central bank suggests to coordinate monetary policy with fiscal or financial policy to achieve a better policy mix."""

system_message = "You are a research assistant at a central bank."

# Load the validation data
validation_sample = pd.read_excel("../../data/input/validation/validation_sample_all.xlsx")

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

# Function to prepare finetuning data
def prepare_pandas_dataset(df, response_format, input_column, output_columns, classify_in_one_prompt = 10):
  # Create prompt reply messages in a while loop
  prompt_number = 0
  input_and_output = []
  while prompt_number < df.shape[0] / classify_in_one_prompt:
      sentences = df.iloc[prompt_number * classify_in_one_prompt:(prompt_number * classify_in_one_prompt + classify_in_one_prompt)]
      input_string = ""
      for index_sentence, sentence in sentences.reset_index().iterrows():
          input_string = input_string + f"[{index_sentence + 1}]\n{sentence[input_column]}\n\n"

      output_string = ""
      for index_sentence, sentence in sentences.reset_index().iterrows():
          columns_content = [sentence[col] for col in output_columns]
          output_string = output_string + response_format.format(index_sentence + 1, *columns_content) + "\n"  
       
      input_and_output.append(
                {
                    "input" : input_string,
                    "output" : output_string
                }
            )
      prompt_number = prompt_number + 1
  return(pd.DataFrame(input_and_output))

# Function to prepare funetining data needed by Gemini
def prepare_gemini_fine_tuning_dataset(df, base_prompt = None, additional_instructions = None):
    if base_prompt is None:
        gemini_format = df.assign(
        input = lambda df: "input: " + df["input"],
        output = lambda df: "output: " + df["output"]
        )

    elif additional_instructions is None:
        gemini_format = df.assign(
        input = lambda df: "input: " + gemini_classification_prompt + "\n" + df["input"],
        output = lambda df: "output: " + df["output"]
        )

    else:
        gemini_format = df.assign(
            input = lambda df: "input: " + gemini_classification_prompt + "\n\n" + additional_instructions_cot + "\n" + df["input"],
            output = lambda df: "output: " + df["output"]
        )

    return(gemini_format)

# Function to oversample df
def oversample_df(df, column, amount = 0.75):
    max_size = df[column].value_counts().max()
    lst = [df]
    for class_index, group in df.groupby(column):
        oversample = np.floor(np.power(max_size-len(group), amount)).astype(int)
        oversample = 0 if oversample < 2 else oversample # Hack to make 0 and very small values not oversample anything
        lst.append(group.sample(oversample, replace=True))
    return(pd.concat(lst).sample(frac=1).reset_index(drop = True))

fine_tuning_set_chatgpt = prepare_pandas_dataset(fine_tuning_set, response_format= reponse_format_cot, input_column = "three-sentence", output_columns = ["explanation", "A_level_3"], classify_in_one_prompt = 3).assign(
    input = lambda df: "input: " + gemini_classification_prompt + "\n\n" + additional_instructions_cot + "\n" + df["input"],
    output = lambda df: "output: " + df["output"]
)

fine_tuning_set_gemini = prepare_pandas_dataset(fine_tuning_set, response_format= reponse_format_cot, input_column = "three-sentence", output_columns = ["explanation_gemini", "A_level_3"], classify_in_one_prompt = 3).assign(
    input = lambda df: "input: " + gemini_classification_prompt + "\n\n" + additional_instructions_cot + "\n" + df["input"],
    output = lambda df: "output: " + df["output"]
)

# Define fine_tuning function which exposes many settings
def schedule_fine_tuninig_and_evaluate(training_set, # Needs a sentence and classification column,
                                       evaluation_set = None, # Set if evaluate = true
                                       evaluate = False,
                                       dataset_config = "both", # Only used to distringuish models. Has not effect on training.
                                       base_prompt = None,
                                       additional_instructions = None,
                                       upsample_factor = 0,
                                       epochs = 5,
                                       learning_rate = 0.001,
                                       batch_size = 4,
                                       train_classify_in_prompt = 5,
                                       inference_classify_in_prompt = 10,
                                       randomized_epochs = False,
                                       temperature = 0.5,
                                       eval_hash = None # If this is set no model is trained but inference is done on 
                                       ):
    fine_tune_config = {
        "dataset_config" : dataset_config,
        "base_prompt" : base_prompt,
        "additional_instructions" : additional_instructions,
        "upsample_factor" : upsample_factor,
        "epochs" : epochs,
        "learning_rate" : learning_rate,
        "batch_size" : batch_size,
        "train_classify_in_prompt" : train_classify_in_prompt,
        "inference_classify_in_prompt" : inference_classify_in_prompt,
        "randomized_epochs" : randomized_epochs,
        "temperature" : temperature
    }

    print(fine_tune_config)

    if eval_hash:
        config_hash = eval_hash
    else:
        config_hash = hex(ctypes.c_size_t( hash(json.dumps(fine_tune_config,sort_keys=True))).value)

    log_path = Path(f"../outputdata/fine_tuning/gemini_fine_tuning/fine_tune_logs/{config_hash}.txt")

    if not log_path.exists():
        training_data = None
        if randomized_epochs:
            for _ in range(epochs):
                training_data = pd.concat([training_data, training_set.sample(frac=1)]) # <- Inefficient copy operation, but given size of dataset okay
        else:
            training_data = training_set
        
        if upsample_factor != 0:
            training_data = oversample_df(training_data, "classification", amount= upsample_factor)

        fine_tuning_data = prepare_pandas_dataset(training_data, response_format= """[{}] {}""", input_column= "sentence", output_columns=  ["classification"], classify_in_one_prompt=train_classify_in_prompt)
        fine_tuning_data = prepare_gemini_fine_tuning_dataset(fine_tuning_data, base_prompt, additional_instructions).to_dict('records')

        print(f"Scheduling model {config_hash} for training.")

        tuning_operation = genai.create_tuned_model(
            source_model="models/gemini-1.0-pro-001",
            training_data=fine_tuning_data,
            id = f"gemini10-{config_hash}",
            epoch_count = 1 if randomized_epochs else epochs,
            batch_size= min(batch_size, len(fine_tuning_data)),
            learning_rate = learning_rate,
            input_key = "input",
            output_key = "output",
            temperature= temperature
        )

        # This blocks until the fine-tuning is complete
        result = tuning_operation.result(timeout=1800) # Wait up to 30 min
        log_path.write_text(json.dumps(fine_tune_config,sort_keys=True))

        print(f"Finetuning completed for model {config_hash}")
    else:
        print("Model already trained, moving to evaluate")


    # Stop here if evaluation is disabled
    if not evaluate:
        return
    
    evaluation_set = evaluation_set.reset_index(drop = True).assign(
        sentence_id = lambda df: df.index
    )

    prompt = base_prompt + "\n"
    if additional_instructions:
        prompt += "\n" + additional_instructions + "\n"

    eval_classified = run_classifcation(prompt,
                    evaluation_set,
                    output_regex =  r"(\[\d+\]\s?)(.*)",
                    output_regex_position= 1,
                    input_numbering_format = "[{}]\n{}\n\n",
                    model = "gemini",
                    gemini_model=f"tunedModels/gemini10-{config_hash}", 
                    maximum_sentences = inference_classify_in_prompt,
                    temperature = temperature,
                    max_retries=0,
                    parallel_prompts=2,
                    abort_if_model_failure= False,
                    allow_incorrect_format_response= True)
    
    received_unexpected_label = eval_classified["classification"].apply(lambda x: not (x in ["monetary dominance", "fiscal dominance", "financial dominance", "monetary-fiscal coordination", "monetary-financial coordination", "none"])).any().tolist()

    if received_unexpected_label:
        print("One of the classifications is a unexpected label")
    
    model_stats = {
        "model_id" : config_hash,
        "received_unexpected_label" : received_unexpected_label,
        **fine_tune_config,
        **eval_classified["classification"].value_counts(normalize = True).to_dict(),
        **calculate_scores(evaluation_set["classification"], eval_classified["classification"])
    }

    log_path.write_text(json.dumps(model_stats,sort_keys=True))

    return(model_stats)


base_fine_tune = schedule_fine_tuninig_and_evaluate(
    training_set = fine_tuning.assign(
        classification = lambda df: df["A_level_3"],
        sentence = lambda df: df["three-sentence"]
    )[["classification", "sentence"]],
    evaluation_set = holdout.assign(
        classification = lambda df: df["A_level_3"],
        sentence = lambda df: df["three-sentence"]
    )[["classification", "sentence"]],
    evaluate= True,
    dataset_config = "fine_tune_sample",
    base_prompt = gemini_classification_prompt,
    epochs = 5,
    learning_rate = 0.001,
    batch_size = 4,
    train_classify_in_prompt = 5,
    inference_classify_in_prompt = 10,
    randomized_epochs = False
)

zero_shot_instruction = """Reply only with the number and the label of each excerpt. Use the following format:
[1] label of first excerpt
[2] label of second excerpt
...
These are the excerpts:"""

def hyper_opt(trial):
    add_synthetic = trial.suggest_categorical("add_synthetic", [True, False])
    add_additional_instructions = trial.suggest_categorical("add_additional_instructions", [True, False])

    training_set = fine_tuning.assign(
        classification = lambda df: df["A_level_3"],
        sentence = lambda df: df["three-sentence"]
        )[["classification", "sentence"]]
    
    evaluation_set = holdout.assign(
        classification = lambda df: df["A_level_3"],
        sentence = lambda df: df["three-sentence"]
    )[["classification", "sentence"]]
    
    if add_synthetic:
        training_set = pd.concat(
            [
                training_set,
                pd.read_excel("../outputdata/fine_tuning/gemini_fine_tuning/synthetic_training.xlsx").rename({"three-sentence" : "sentence"}, axis = 1)[["classification", "sentence"]]
            ]
        ).sample(frac=1).reset_index(drop=True)

    per_prompt = trial.suggest_categorical("sentence_per_prompt", [5,10,25])

    results = schedule_fine_tuninig_and_evaluate(
    training_set = training_set,
    evaluation_set = evaluation_set,
    additional_instructions= zero_shot_instruction if add_additional_instructions else None,
    evaluate = True,
    dataset_config = "hyperopt",
    base_prompt = gemini_classification_prompt,
    epochs = trial.suggest_int("epochs", 1,10, log=True), # Dont train more than 10 epochs
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.01),
    batch_size = trial.suggest_int("batch_size", 2,16, step=2),
    train_classify_in_prompt = per_prompt,
    inference_classify_in_prompt = per_prompt,
    randomized_epochs = trial.suggest_categorical("randomize_epochs", [True, False]),
    upsample_factor =  trial.suggest_float("upsample_factor", 0, 1),
    temperature = trial.suggest_float("temperature", 0, 0.9),
    )

    # Store user attributes
    for key in results:
        trial.set_user_attr(key, results[key])

    return(
        2 * results["f1_macro"] + results["accuracy"]
    )

study = optuna.create_study(direction = "maximize", study_name="optimize_300_excerpts")
study.optimize(hyper_opt, timeout = 60*60*2.2) # Run until some timeout

# The sutdy can be loaded and save like this:
# study = joblib.load("../outputdata/fine_tuning/gemini_fine_tuning/study_85.pkl")
# joblib.dump(study, "../outputdata/fine_tuning/gemini_fine_tuning/study_115.pkl")

# Save after sufficient attempts were made
study.trials_dataframe().to_parquet("../../data/processed/fine_tuning/study_115.parquet")

# Gemini stuff
study = joblib.load("../../data/processed/fine_tuning/study_115.pkl")

# Final model 2024-03-15 23-31-08, #0x4e3aa25b950553ad
final_model = schedule_fine_tuninig_and_evaluate(
    training_set = fine_tuning.assign(
        classification = lambda df: df["A_level_3"],
        sentence = lambda df: df["three-sentence"]
    )[["classification", "sentence"]],
    evaluation_set = holdout.assign(
        classification = lambda df: df["A_level_3"],
        sentence = lambda df: df["three-sentence"]
    )[["classification", "sentence"]],
    evaluate= True,
    dataset_config = "finalft",
    base_prompt = gemini_classification_prompt,
    additional_instructions = zero_shot_instruction,
    epochs = 7,
    learning_rate = 0.0005,
    batch_size = 2,
    train_classify_in_prompt = 5,
    inference_classify_in_prompt = 5,
    randomized_epochs = True,
    temperature=0.0,
    upsample_factor=0
)

# Run on full dataset
zero_shot_instruction = """Reply only with the number and the label of each excerpt. Use the following format:
[1] label of first excerpt
[2] label of second excerpt
..."""

sentence_level = pd.read_pickle("../outputdata/fine_tuning/sentence_level_bis.pkl")
sentence_level_data_with_context = sentence_level.assign(
         sentence_id = lambda df: df.index,
         context_1_1 = sentence_level.groupby("speech_identifier",  sort = False, group_keys = False).apply(lambda df: add_neighbouring_sentences(df)))

# The run_classification function excepts the sentence or excerpt to be in the sentence column.
sentence_level_data_with_context["sentence"] = sentence_level_data_with_context["context_1_1"]
sentence_level_data_with_context = sentence_level_data_with_context.sample(frac=1, random_state = 42)

full_classification = run_classifcation(gemini_classification_prompt + "\n" + "\n" + zero_shot_instruction + "\n",
                    sentence_level_data_with_context,
                    output_regex =  r"(\[\d+\]\s?)(.*)",
                    output_regex_position= 1,
                    input_numbering_format = "[{}]\n{}\n\n",
                    model = "gemini",
                    gemini_model=f"tunedModels/gemini10-0x4e3aa25b950553ad",
                    continue_run=None, # Use this option to continue a initial run
                    maximum_sentences = 5,
                    temperature = 0,
                    max_retries=10,
                    parallel_prompts=3,
                    allow_reclassification = False,
                    enforce_labels=["none", "monetary-financial coordination", "monetary-fiscal coordination", "fiscal dominance", "financial dominance", "monetary dominance"], 
                    abort_if_model_failure= False,
                    allow_incorrect_format_response= False)


# Store full classification
full_classification[["sentence_id", "classification"]].to_parquet("../outputdata/fine_tuning/dom_corp_classification_sentence_level.parquet")


# Validation of model:
holdout_sample_classification = run_classifcation(gemini_classification_prompt + "\n" + "\n" + zero_shot_instruction + "\n",
                    holdout.assign(sentence = lambda df: df["three-sentence"]),
                    output_regex =  r"(\[\d+\]\s?)(.*)",
                    output_regex_position= 1,
                    input_numbering_format = "[{}]\n{}\n\n",
                    model = "gemini",
                    gemini_model=f"tunedModels/gemini10-0x4e3aa25b950553ad",
                    maximum_sentences = 5,
                    temperature = 0,
                    max_retries=10,
                    parallel_prompts=3,
                    allow_reclassification = False,
                    enforce_labels=["none", "monetary-financial coordination", "monetary-fiscal coordination", "fiscal dominance", "financial dominance", "monetary dominance"], 
                    abort_if_model_failure= False,
                    allow_incorrect_format_response= False)


holdout_sample_classification.to_pickle("../../data/processed/fine_tuning/holdout_validation.pkl")