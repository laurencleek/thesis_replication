# Query chat gpt

import pandas as pd
import datetime as pt
import tiktoken
import datetime
import json
import ruamel.yaml as yaml
import time
from pathlib import Path
import os
import openai
from openai import AzureOpenAI
from openai import OpenAI
import re
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from itertools import product
from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from nltk.metrics.agreement import AnnotationTask
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
from google.oauth2.credentials import Credentials

use_o_auth_for_Gemini = False # This is difficult to configure see here: https://ai.google.dev/palm_docs/oauth_quickstart
enc = tiktoken.get_encoding("cl100k_base")

# API Keys
client = openai.OpenAI(
   api_key = "<API KEY>"
)

if use_o_auth_for_Gemini:
   creds = Credentials.from_authorized_user_file('application_default_credentials.json', ['https://www.googleapis.com/auth/generative-language.tuning'])
   genai.configure(credentials=creds)
else:
   genai.configure(api_key="<API KEY>", transport="grpc")

safety_settings = [
{
   "category": "HARM_CATEGORY_HARASSMENT",
   "threshold": "BLOCK_ONLY_HIGH"
},
{
   "category": "HARM_CATEGORY_HATE_SPEECH",
   "threshold": "BLOCK_ONLY_HIGH"
},
{
   "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
   "threshold": "BLOCK_ONLY_HIGH"
},
{
   "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
   "threshold": "BLOCK_ONLY_HIGH"
},
]

# Some helper functions
def count_tokens(text):
    """Count tokens. Requires tiktoken and a enc loaded"""
    try:
      length = len(enc.encode(text))
    except:
      length = 0
    return(length)

def set_high_value_if_none(number):
     """Returns large number if None"""
     if number is None:
        return(1_000_000_000)
     
     else:
        return (number)

def expand_grid(dictionary):
   """Provide a dictionary where keys are variables and values are possible values. It will return a pd.DataFrame with all possible combinations of the values of the provided variables. Useful for running grid search in CV"""
   return pd.DataFrame([row for row in product(*dictionary.values())], 
                       columns=dictionary.keys())

def conditional_classification(prompt, input_data, conditional, **kwargs):
   """Run different prompts based on the value in another column. Prompt should be a dictionary containing as key all possible values of the the column specified by conditional"""
   conditional_classifiaction = []
   for label, subset in input_data.groupby(conditional):
      conditional_classifiaction.append(
         run_classifcation(prompt = prompt[label], input_data = subset, **kwargs)
      )
   
   return(pd.concat(conditional_classifiaction).sort_values("sentence_id"))

# This function has a external dependency. It requires a openai client called 'client' to be loaded.
def run_classifcation(
      prompt,
      input_data,
      model = "chatgpt", # Either chatgpt or gemini
      maximum_sentences = 10, # Use this to set a limit on the number of sentences included in a prompt. None means unlimited.
      tokens_max = None, # Use this to set a limit on the number of tokens included in a single port. None means unlimited. 
      max_queries = None, # This limits the number of queries made to the openAI api
      few_shot = False,
      continue_run = None, # This is to deal with failing runs. Put the timestamp as string and it will not rerun prompts if already run
      temperature = 0,
      top_p = 1,
      parallel_prompts = 12,
      max_retries = 10,
      wait_between_retries = 10,
      input_numbering_format = "[{}]\n{}\n\n", # The format to use to append the examples to the prompt
      output_regex =  r"(\d+\.\s)(.*)", # Regular expression to extract the response. It should match exactly #(^label:\s?)(.*)
      output_regex_position = 1, # Match group to exctract.
      enforce_labels = None, # This fails the prompt if one of the labels does not in the list
      allow_incorrect_format_response = False, # If this is set to true it will enfore a correctly formatted response and retry immediately if it does not conform until max_retries is reached. Also it will not store the response in the log if it cannot extract the number of responses.It will however return "NA" in the dataframe.
      abort_if_model_failure = True, # What to do if model did not reply max_retry times. Move on to next prompt or abort thread?
      allow_reclassification = False, # This allows already run classifcations to be saved again.
      # ChatGPT specific
      chatgpt_model = "gpt-3.5-turbo-0301",
      system_message = "You are a research assistant at a central bank.",
      few_shot_sentences = None, # This should be a pandas dataframe with the columns sentence, and classification
      few_shot_follow_up = None, # If a full prompt and answer are provided as few shot learning, it might make sense to not restate the full prompt. This could be something like: "Now classify these sentences:"
      seed = None,
      # Gemini specific
      prompt_history = None,
      gemini_model = "gemini-pro"
      ):
  """This is the main function used to classify sentences. It has sensible defaults. See comments on what options do"""
  # Handle the options that specify maximums. To deal with the Nones we just set a very high value that none of the paramters can reasonably reach.
  maximum_sentences = set_high_value_if_none(maximum_sentences)
  tokens_max = set_high_value_if_none(tokens_max)
  max_queries = set_high_value_if_none(max_queries)
  
  # Initalize counters
  current_query = ''
  sentences_current_query = 0
  number_of_queries = 0
  # Sentence ids (This is a list that keeps track of the sentence ids that have been added to the current.query)
  sentence_ids = []

  # Store the return values. These are only 'Promises', i.e. async
  prompt_results = []

  # Create a pool for multiprocessing:
  pool = Pool(parallel_prompts)
  
  # Save a timestamp that identifiers this run. A folder with this timestamp as name will be created in the /logs/ folder
  if continue_run is not None:
     timestamp_run = continue_run
  else:
    timestamp_run = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
     
  # Define a function that is used to add to the current query.
  def append_to_query(current_query, sentence, sentence_number):
     """This takes the current query as input and appends the next sentence"""

     return(current_query + input_numbering_format.format(sentence_number, sentence))
  
  # Few shot learning codes goes here:
  if few_shot:
   few_shot_prompt = prompt
   few_shot_response = ""
   for index, row in few_shot_sentences.reset_index(drop = True).iterrows():
       few_shot_prompt = append_to_query(few_shot_prompt, row.sentence, index+1)
       few_shot_response = few_shot_response + f"{index + 1}. {row.classification}\n"
  else:
     few_shot_prompt = None
     few_shot_response = None

  def process_response(text, expected_number_of_responses):
     try:
         sentence_classification = [x[output_regex_position] for x in re.findall(output_regex, text)]#
         assert len(sentence_classification) == expected_number_of_responses
     except:
         # Set to anything that isnt a list
         return(None)

     if enforce_labels:
        if not all([x in enforce_labels for x in sentence_classification]):
           return(None)
      
     return(sentence_classification)
     

  def run_query(
        query,
        sentence_ids,
        query_number,
        # ChatGPT specific
        few_shot_query = None,
        few_shot_response = None,
        system_message = None,
        # Gemini specific
        prompt_history = None
        ):
    """Nested function which executes the request itself."""
    start_time = time.time()

    # This variable is incremented when a query iteration fails
    required_attemps = 1

    if model == "chatgpt":
      if (few_shot_query is not None) and (few_shot_response is not None):
         messages = [
                     {"role": "system", "content": system_message},
                     {"role": "user", "content": few_shot_query},
                     {"role": "assistant", "content": few_shot_response},
                     {"role" : "user", "content": query}
                  ]
      else:
         messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": query}
                     ]
    elif model == "gemini":
       if prompt_history is not None:   
         prompt_parts = [
            *prompt_history,
            f"input: {query}",
            "output: "
         ]
       else:
         prompt_parts = [
             f"input: {query}",
             "output: "
         ]      

    already_classified = Path(f"../logs/{timestamp_run}/{query_number}.json").is_file()     
    if already_classified and not allow_reclassification:
      result_dict = json.loads(Path(f"../logs/{timestamp_run}/{query_number}.json").read_text())
      sentence_classification = process_response(result_dict["response_text"], len(sentence_ids))
    else:
      read_attempt_from_file = True if already_classified else False
      
      while True:
         result_dict = {}
         result_dict["prompt"] = query
         result_dict["query_number"] = query_number
         result_dict["sentence_ids"] = sentence_ids

         try:
            if read_attempt_from_file:
               result_dict = json.loads(Path(f"../logs/{timestamp_run}/{query_number}.json").read_text())
               read_attempt_from_file = False
               
            elif model == "chatgpt":
               # ChatGPT specific logging
               result_dict["system_message"] = system_message
               result_dict["few_shot_query"] = few_shot_query
               result_dict["few_shot_response"] = few_shot_response

               api_response = client.chat.completions.create(
               model = chatgpt_model,
               temperature = temperature,
               seed = seed,
               top_p = top_p,
               messages= messages).model_dump()

               # Required logs
               result_dict["response_text"] = api_response['choices'][0]['message']['content']
               result_dict["completed_response"] = True

            elif model == "gemini":
               # Gemini specific logging
               result_dict["prompt_history"] = prompt_history

               gemini_client = genai.GenerativeModel(model_name=gemini_model,
                              safety_settings = safety_settings,
                              generation_config= {
                                "temperature": temperature,
                                "top_p": top_p,
                                "top_k": 1
                                })
               api_response = gemini_client.generate_content(prompt_parts)

               # Required logs
               result_dict["response_text"] = api_response.text
               result_dict["completed_response"] = True

         except Exception as e:
            print(f"Running the prompt failed {required_attemps} times. Last error: {print(e)}")
            result_dict["completed_response"] = False
            result_dict["response_text"] = None         

         if result_dict["completed_response"] == False:
            if required_attemps <= max_retries:
               required_attemps = required_attemps + 1
               time.sleep(wait_between_retries)
               continue
            else:
               print("Exhausted max retries never got a response from the model")
               if abort_if_model_failure:
                  # This should not happen! In case it does we raise an exception and abort
                  raise
               else:
                  sentence_classification = process_response(result_dict["response_text"], len(sentence_ids))
                  break
            
         sentence_classification = process_response(result_dict["response_text"], len(sentence_ids))

         # Check if we require a correct format and if we got incorrect format. If yes, we retry
         if not allow_incorrect_format_response and not isinstance(sentence_classification, list):
            if required_attemps <= max_retries:
               print(f"Retrying as format did not match on the {required_attemps} attempt. Got:\n {result_dict['response_text']}")
               required_attemps = required_attemps + 1
               continue
            else:
               #print("Exhausted max retries and did not get a valid response format")
               break
         else:
            # We got a good response or a bad response is allowed!
            break

      # This needs to be correctly indented to be outside of the Loop and outside of. Python whitespace bugs are so annoying...
    if allow_incorrect_format_response | isinstance(sentence_classification, list): # Check if either incorrect format is allowed or the format is correct
         if not already_classified or allow_reclassification:
            # Store the returned values from the prompt
            log_file = Path(f"../logs/{timestamp_run}/{query_number}.json")
            log_file.parent.mkdir(exist_ok=True, parents=True)
            log_file.write_text(json.dumps(result_dict), encoding='UTF-8')

      # In case the classification is not valid, we assign NA. This ensures that always a dataframe is returned. However, it may not be saved and be retried when rerunning.
    if not isinstance(sentence_classification, list):
         print(f"The returned labels are not in the specified format on prompt {query_number}. Assigning NA.")
         sentence_classification = ["NA"] * len(sentence_ids)

      # ChatGPT has some interesting statistics we might care about
      # This extracts the chat completition, i.e. the reply we are insterested in
    if model == "chatgpt":
         prompt_tokens = result_dict["usage"]["prompt_tokens"]
         completion_tokens = result_dict["usage"]["completion_tokens"]
         total_tokens = result_dict["usage"]["total_tokens"]
    elif model == "gemini":
         prompt_tokens = None # Gemini does not return these
         completion_tokens = None # Gemini does not return these
         total_tokens = None # Gemini does not return these
      
    results = pd.DataFrame(
         {
         "prompt_id" : query_number,
         "time_to_run" : time.time() - start_time,
         "completed_response" : result_dict["completed_response"],
         "required_attemps" : required_attemps,
         "classification": sentence_classification,
         "sentence_id" : sentence_ids,
         "prompt_tokens" : prompt_tokens,
         "completion_tokens" : completion_tokens,
         "total_tokens" : total_tokens,
         "estimated_prompt_tokens" : count_tokens(prompt)
         })

    return(results)

  # Iterate over all the rows of the provided dataframe
  with tqdm(total=input_data.shape[0]) as pbar:
     def updateprogress(_):
        pbar.update(maximum_sentences)

     for index, row in input_data.reset_index(drop = True).iterrows():
      if current_query == '':
          current_query = few_shot_follow_up if few_shot else prompt

      sentences_current_query += 1
      appended_query = append_to_query(current_query, row.sentence, sentences_current_query)
      appended_sentence_ids = sentence_ids + [row.sentence_id] # Here we copy the list
      
      tokens = count_tokens(appended_query)

      if (tokens > tokens_max) | (sentences_current_query > maximum_sentences):

        if (number_of_queries < max_queries):
          
          prompt_results.append(pool.apply_async(run_query, callback = updateprogress, kwds = {
             "query" : current_query,
             "sentence_ids" : sentence_ids,
             "query_number" : number_of_queries,
             "few_shot_query" : few_shot_prompt,
             "few_shot_response" : few_shot_response,
             "system_message" : system_message,
             "prompt_history" : prompt_history
          }))

          number_of_queries += 1
          
          # Finally we need to reset the current_query
          sentences_current_query = 1
          current_query = append_to_query(few_shot_follow_up if few_shot else prompt, row.sentence, sentences_current_query)
          sentence_ids = [row.sentence_id]
      else:
        # Keep adding to the query:
        current_query = appended_query
        sentence_ids = appended_sentence_ids

      if (index + 1 == input_data.shape[0]) & (number_of_queries < max_queries):
         prompt_results.append(pool.apply_async(run_query, callback = updateprogress, kwds = {
             "query" : current_query,
             "sentence_ids" : sentence_ids,
             "query_number" : number_of_queries,
             "few_shot_query" : few_shot_prompt,
             "few_shot_response" : few_shot_response,
             "system_message" : system_message,
             "prompt_history" : prompt_history
          }))
         
      if(number_of_queries == max_queries):
         break
         
      # Wait for everytihng to finish       
     pool.close()
     pool.join()

  prompt_results_df = [prompt_result.get() for prompt_result in prompt_results]
  return(pd.concat(prompt_results_df, ignore_index=True))



def fix_labels(classified_sentences, #DataFrame returned from run_classifcation
               mapping = None, # Simple mapping containing labels as keys and the label mapped to as value
               regex_mapping = None, # Same as mapping except that regular expressions are used as keys
               ignore_case = True, # Should case be ignored. Only applies to mapping
               unkown_category = 'NA', # What value to set if the label in classified_sentences does not appear as key in either of the mappings
               drop_unknown = False # Should these be dropped?
               ):
   """This function is supposed to run after run_classifcation to repair labels. It takes the dataframe and modified the classification column. You can either provide lists of w"""

   # If ignore_case is set to true regex with ignore case is used
   if mapping is not None:
      if ignore_case:
         ignorecaseregex = {f"(?i)^{k}$" : v for k,v in mapping.items()} 
         classified_sentences = classified_sentences.assign(
            classification = lambda x: x["classification"].replace(ignorecaseregex, regex = True)
         )
      else:
         classified_sentences = classified_sentences.assign(
            classification = lambda x: x["classification"].replace(mapping)
         )

   if regex_mapping is not None:
      # Also apply the regular expressions
      classified_sentences = classified_sentences.assign(
            classification = lambda x: x["classification"].replace(regex_mapping, regex = True)
      )
      
   # We assigning all values that are not part of the mapping to "NA". For that  we use a regex, but we need to escape some of the characters like (
   if (regex_mapping is not None) and (mapping is not None):
      allowed_values = list(mapping.values()) + list(regex_mapping.values())
   elif mapping is not None:
      allowed_values = list(mapping.values())
   elif regex_mapping is not None:
      allowed_values = list(regex_mapping.values())

   safe_matches = [re.escape(m) for m in allowed_values]
   classified_sentences.loc[~classified_sentences["classification"].str.contains("^(" + "|".join(safe_matches) + ")$", case = False), "classification"] = unkown_category

   if drop_unknown:
      classified_sentences = classified_sentences.loc[~(classified_sentences.classification == unkown_category)]

   return(classified_sentences)

def calculate_confusion(y_true, y_pred):
  """Calculates confusion matrix based on true and predicted vectors"""
  return(pd.DataFrame(
     {
        "y_true" : y_true,
        "y_pred" : y_pred
     }
  ).groupby("y_true")["y_pred"].value_counts(normalize=True).unstack(fill_value=0))


def calculate_scores(ytrue, ypred):
   """This function defines the scores to evaluate the model. Add to the returned dictionary to report more scores whenever scores are calculated."""
   ytrue_freq = ytrue.sort_values().value_counts(normalize= True)
   ypred_freq = ypred.sort_values().value_counts(normalize= True)
   freq_deviation = (ytrue_freq - ypred_freq).abs().sum()

   return(
      {
        "accuracy" : accuracy_score(ytrue, ypred),
        "f1_weighted" : f1_score(ytrue, ypred, average = 'weighted'),
        "f1_macro" : f1_score(ytrue, ypred, average = 'macro'),
        "precision_macro" : precision_score(ytrue, ypred, average = "macro"),
        "recall_macro" : recall_score(ytrue, ypred, average = "macro"),
        "balanced_accuracy" : balanced_accuracy_score(ytrue, ypred),
        "freq_deviation" :  freq_deviation
      }
   )

def category_specific_accuracy(ytrue, ypred):
   """Calculate category specific, i.e. how accurate is the prediction by "true" tabulated by each value found in ytrue"""
   classification_report_dict = classification_report(ytrue, ypred, output_dict= True) 
   scores = []
   for label in ytrue.unique():
      scores.append(
         {
            "category" : label,  
            **classification_report_dict[label]
         })
      
   return(
      pd.DataFrame(scores)
   )

def add_neighbouring_sentences(df, before = 1, after = 1):
      """Adds the sentences before and each sentence to the sentence. This requires that sentences that are passed in df are in the correct order"""
      added_context_column = ""
      for before_index in range(before, 0, -1):
         added_context_column = added_context_column + df["sentence"].shift(before_index).fillna('') + " "


      added_context_column = added_context_column +  df["sentence"]

      for after_index in range(1, after + 1, 1):
         added_context_column = added_context_column + " " + df["sentence"].shift(-after_index).fillna('')

      added_context_column = added_context_column.str.replace(r"^\s+", "", regex = True)

      added_context_column = added_context_column.str.replace(r"\s+$", "", regex = True)

      return(added_context_column)

def entire_speech(df, sentence_id):
   """Returns the entire speech belonging to a sentence_id"""
   speech = df.query(
      f"sentence_id == @sentence_id"
   ).iloc[0]["speech_identifier"]

   return(" ".join(df[
      df.speech_identifier == speech
   ]["sentence"]))