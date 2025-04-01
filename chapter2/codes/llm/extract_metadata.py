# Minimal file to extract metadata
import pandas as pd
from pathlib import Path
import math
import google.generativeai as genai
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
from tenacity import retry, stop_after_attempt, wait_fixed
import json

# Add to parent path to allow imports from parent directories:
import os, sys
current_dir = os.path.abspath('')
parent_dir = os.path.dirname(current_dir)
if not parent_dir in sys.path: sys.path.append(parent_dir)

from constants.cbnames import *
from speech_preprocessing import *

genai.configure(api_key="<API_KEY>")
log_path = "./logs/" # Log path for the API calls. This should be a directory that exists.

prompt_metadata = """I'll provide you with metadata on central bank speeches from which you should extract the following information:
1. Speech Identifier (e.g. b050203)
2. Type of Text (e.g. Speech, Introductory statement or Introductory remarks or Interview)
3. Name of the speaker (e.g. Jean-Claude Trichet)
4. Central bank of the speaker (e.g. Bank of England)
5. Position of the Speaker (e.g. President of the Federal Reserve Bank of Kansas City)
6. Occasion (e.g. 30th Economics Conference 'Competition of Regions and Integration in EMU')
7. Venue (e.g. London School of Economics and Political Science)
8. Location (e.g Frankfurt or Vienna)\n
Reply with a JSON list with one object for each speech. Please use the following keys for the metadata:
1. "speech_identifier"
2. "type_of_text"
3. "speaker"
4. "central_bank"
5. "position"
6. "occasion"
7. "venue"
8. "location"\n
Additional instructions:
* The central bank should always be the central bank with which the speaker is affiliated. E.g. if the the governor of the Bank of England delivers a speech at an event organized by the ECB you should assign the Bank of England as the central bank.
* If the central bank is part of the US federal reserve system be specific whether it is the board of governors of the Fed or a regional Fed like the Federal Reserve Bank of Chicago.
* If a speech contains the names of two speakers separate the names with a semicolon. Do the same for their positions and central banks. See the second example given below.
* Not all fields are always contained in the data. In this cases you can leave missing values. See the second example which does not contain a occasion and venue. And the third example which does not contain a location.
* Occasion should contain the occasion on which the speech was given. This should contain most of the information that is given in the metadata. E.g. if it was part of a session of a particular conference on a topic it should include all that
* Venue is more narrow and should be a place like an institute, university, ministry or central bank or a particular congress/forum. Don't put a city as venue. The city should go to location.
* Location should always be a city. If the city is not directly given you can infer the city from e.g. the university.
* Sometimes abbreviations are used such as Ass. for association, Conf. for conference or Econ. for economics. Please spell these out these abbreviations.\n\n
Extract the metadata from the following lines. Each line starts with the speech identifer followed by a string that contains the metadata that should be extracted.
"""

examples_metadata = """b170831: Welcome address by Dr Ernest Addison, Governor of the Bank of Ghana, at the Bank of Ghana's 60th Anniversary Lectures and Exhibition, Accra, 18 August 2017.
a000303: Introductory statements given by Mr Willem F Duisenberg, President of the European Central Bank, and Mr Christian Noyer, Vice-President of the European Central Bank, held in Frankfurt on 2 March 2000.
b050203: English translation of an interview of Mr Jean-Claude Trichet, President of the European Central Bank, conducted by Ms Françoise Crouïgneau and Mr Thibaut Madelin (Les Echos), 20 January 2005.
h970403: Text of the London School of Economic and Political Science Annual Lecture on Central Banking delivered by the Deputy Goveror of the Bank of England, Mr. Howard Davies, on 4/3/97.
e140331: Speech by Dr Andreas Dombret, Member of the Executive Board of the Deutsche Bundesbank, at the Harvard Law School Symposium on "Building the Financial System of the 21st Century - An Agenda for Europe and the United States", Armonk, New York, 28 March 2014.
e220708: Speech by Ms Lael Brainard, Member of the Board of Governors of the Federal Reserve System, at the Bank of England Conference, London, 8 July 2022."""

desired_output_metadata = """[
    {
        "speech_identifier" : "b170831",
        "type_of_text" : "Welcome address",
        "speaker" : "Ernest Addison",
        "central_bank" : "Bank of Ghana",
        "position" : "Governor of the Bank of Ghana",
        "occasion" : "Bank of Ghana's 60th Anniversary Lectures and Exhibition",
        "venue" : "Bank of Ghana",
        "location" : "Accra"
    },
    {
        "speech_identifier" : "a000303",
        "type_of_text" : "Introductory statement",
        "speaker" : "Willem F Duisenberg;Christian Noyer",
        "central_bank" : "European Central Bank;European Central Bank",
        "position" : "President of the European Central Bank;Vice-President of the European Central Bank",
        "occasion" : null,
        "venue" : null,
        "location" : "Frankfurt"
    },
    {
        "speech_identifier" : "b050203",
        "type_of_text" : "Interview",
        "speaker" : "Jean-Claude Trichet",
        "central_bank" : "European Central Bank",
        "position" : "President of the European Central Bank",
        "occasion" : "Interview conducted by Ms Françoise Crouïgneau and Mr Thibaut Madelin (Les Echos)",
        "venue" : "Les Echos",
        "location" : null
    },
    {
        "speech_identifier" : "h970403",
        "type_of_text" : "Lecture",
        "speaker" : "Howard Davies",
        "central_bank" : "Bank of England",
        "position" : "Deputy Governor of the Bank of England",
        "occasion" : "London School of Economic and Political Science Annual Lecture on Central Banking",
        "venue" : "London School of Economics and Political Science",
        "location" : "London"
    },
    {
        "speech_identifier" : "e140331",
        "type_of_text" : "Speech",
        "speaker" : "Andreas Dombret",
        "central_bank" : "Deutsche Bundesbank",
        "position" : "Member of the Executive Board of the Deutsche Bundesbank",
        "occasion" : "Harvard Law School Symposium on \"Building the Financial System of the 21st Century - An Agenda for Europe and the United States\"",
        "venue" : "Harvard Law School",
        "location" : "New York"
    },
    {
        "speech_identifier" : "e220708",
        "type_of_text" : "Speech",
        "speaker" : "Lael Brainard",
        "central_bank" : "Board of Governors of the Federal Reserve",
        "position" : "Member of the Board of Governors of the Federal Reserve System",
        "occasion" : "Bank of England Conference",
        "venue" : "Bank of England",
        "location" : "London"
    }
]"""

prompt_audience = """I'll provide you with metadata from central bank speeches from which you should infer the audience of the speech. Please assign one of the following labels for each speech:
"academic", if the audience is likely to be academic, e.g. a speech at a university or a conference at a research institute
"financial_market", if the audience is likely to be financial market actors or representatives, e.g. a speech at a financial markets association
"political", if the audience is likely to be politicians, government officials or elected representatives, e.g. an address in front of parliament or a ministry
"central_bank", if the audience are central bankers or a general central bank audience, e.g. interviews with newspapers or central bank press conferences. Also assign this category if none of the other categories fit.\n
Reply with a JSON list with one object for each speech. Each object should contain two entries:
1. "identifier"
2. "audience"
Extract the audience from the following lines. Each line starts with a speech identifier, followed by a colon, followed by the metadata on the speech.
"""

examples_audience = """b060714: Speech by Mr Jean-Claude Trichet, President of the European Central Bank, at the 57. Jahresversammlung des Ifo Instituts für Wirtschaftsforschung an der Universität München, Munich, 29 June 2006.
a151202: Speech by Mr Amando M Tetangco, Jr, Governor of Bangko Sentral ng Pilipinas (BSP, the central bank of the Philippines), at the Launching of the Paranaque City Credit Surety Fund, Manila, 3 November 2015.
e180108: Address by Mr Rameswurlall Basant Roi, Governor of the Bank of Mauritius, at the annual dinner for major economic stakeholders, Flic-en-Flac, 17 November 2017.
a170831: Opening statement by Dr Andreas Dombret, Member of the Executive Board of the Deutsche Bundesbank, at the press conference presenting the results of the low-interest-rate survey conducted by the Bundesbank and BaFin, Frankfurt am Main, 30 August 2017.
a170418: Introductory statement by Mr Ignazio Visco, Governor of the Bank of Italy, at an "Open coordinators meeting" of the ECON Committee (European Parliament) for an exchange of views on the economic and financial situation of Italy and prospects for economic governance in the European Union, Brussels, 11 April 2017."""

desired_output_audience = """[
  {
    "identifier": "b060714",
    "audience": "academic"
  },
  {
    "identifier": "a151202",
    "audience": "financial_market"
  },
  {
    "identifier": "a170831",
    "audience": "central_bank"
  },
  {
    "identifier": "e180108",
    "audience": "financial_market"
  },
  {
    "identifier": "a170418",
    "audience": "political"
  }
]"""

prompt_location = """I'll provide you with identifiers and associated locations. I want you to find the geographic coordinates associated with each location. Reply with a JSON list with one object for each identifer. Each object should contain the following keys:
1. identifier
2. latitude
3. longitude\n
Extract the metadata from the following lines. Each line starts with the identifier, followed by a semicolon, followed by the location for which you should find the coordinates.
"""

examples_location = """a981012;Brussels
c090907;Buenos Aires
c050609;Frankfurt"""

desired_output_location = """[
  {
    "identifier": "a981012",
    "latitude": 50.8465,
    "longitude": 4.3517
  },
  {
    "identifier": "c090907",
    "latitude": -34.6037,
    "longitude": -58.3692
  },
  {
    "identifier": "c050609",
    "latitude": 50.1109,
    "longitude": 8.6821
  }
]"""

metadata_list = []

# Read
metadata = pd.read_parquet("../../data/processed/speech_metadata.parquet").assign(
   identifier = lambda df: df.speech_identifier,
   metadata = lambda df: df.description,
   combined = lambda df: df.speech_identifier + ": " + df.description
)[["identifier", "metadata", "combined"]]

classify_in_one_go = 6
parallel_prompts = 15
read_cache = True

results = []
with tqdm(total=math.ceil(metadata.shape[0]/classify_in_one_go)) as pbar:
      pool = Pool(parallel_prompts)
      def updateprogress(_):
        pbar.update(1)

      def failed_too_often(fail):
         print("Failed too many times; moving on!")
         print(fail)
         return("")
      
      @retry(stop=stop_after_attempt(5), wait=wait_fixed(10), retry_error_callback = failed_too_often)
      def run_in_parallel(index):
         meta_data_to_classify = metadata.iloc[(index * classify_in_one_go): (index * classify_in_one_go + classify_in_one_go)]["combined"].str.cat(sep='\n').replace("\"", "'")

         gemini_client = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                    generation_config= {
                                    "temperature": 0.5,
                                    "top_p": 1,
                                    "top_k": 1,
                                    "max_output_tokens": 2048,
                                    })
         
         if (read_cache and Path(f"{log_path}/description/{index}.txt").is_file()):
            api_response = Path(f"{log_path}/description/{index}.txt").read_text(encoding= "UTF-8")
         else:
            api_response = gemini_client.generate_content([
                  f"input: {prompt_metadata}{examples_metadata}",
                  f"output: {desired_output_metadata}",
                  f"input: {prompt_metadata}{meta_data_to_classify}",
                  f"output: "
            ])
            api_response = api_response.text

         # Validate Output:
         output_as_json = json.loads(api_response)

         # Check for keys:
         for entry in output_as_json:
            if not all(required_field in entry for required_field in ["speech_identifier","type_of_text","speaker","central_bank", "position","occasion","venue","location"]):
               print(f"{index} does not match requirements. Retry")
               raise

         # Do something with the response
         Path(f"{log_path}/description/{index}.txt").write_text(api_response, encoding="UTF-8")

         return(api_response)
      # Query jobs:
      for index in range(math.ceil(metadata.shape[0]/classify_in_one_go)):
         results.append(pool.apply_async(run_in_parallel, callback = updateprogress, kwds= {
            "index" : index,
         }))

      pool.close()
      pool.join()

processed_metadata = [result.get() for result in results]

converted_to_dicts = []
for result in processed_metadata:
   entries = json.loads(result)
   converted_to_dicts.extend(entries)

processed_metadata = pd.DataFrame(converted_to_dicts)

# Check that all indentifiers are present and that lengths match. Order doesnt matter since we will merge on identfier.
set(metadata.identifier.tolist()) - set(processed_metadata.speech_identifier.tolist())
len(metadata.identifier) == len(processed_metadata.speech_identifier)

processed_metadata.to_parquet("../../data/processed/extracted_main_metadata.parquet")

metadatatt = metadata.merge(processed_metadata, how = "left", left_on = "identifier", right_on = "speech_identifier")

testing = metadatatt.assign(
   central_bank_cc = lambda df: df.central_bank.apply(lambda x: match_central_bank(central_bank_mapping, x)),
   central_bank_naive = lambda df: df.metadata.apply(lambda x: match_central_bank(central_bank_mapping, x))
)

# Classify audience:
results_audience = []
with tqdm(total=math.ceil(metadata.shape[0]/classify_in_one_go)) as pbar:
      pool = Pool(8)
      def updateprogress(_):
        pbar.update(1)

      def failed_too_often(fail):
         print("Failed too many times; moving on!")
         print(fail)
         return("")
      
      @retry(stop=stop_after_attempt(10), wait=wait_fixed(5), retry_error_callback = failed_too_often)
      def run_in_parallel(index):
         meta_data_to_classify = metadata.iloc[(index * classify_in_one_go): (index * classify_in_one_go + classify_in_one_go)]["combined"].str.cat(sep='\n').replace("\"", "'")

         gemini_client = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                    generation_config= {
                                    "temperature": 0.5,
                                    "top_p": 1,
                                    "top_k": 1,
                                    "max_output_tokens": 2048,
                                    })
         
         if (read_cache and Path(f"{log_path}/extractaudience/{index}.txt").is_file()):
            api_response = Path(f"{log_path}/extractaudience/{index}.txt").read_text(encoding= "UTF-8")
         else:
            api_response = gemini_client.generate_content([
                  f"input: {prompt_audience}{examples_audience}",
                  f"output: {desired_output_audience}",
                  f"input: {prompt_audience}{meta_data_to_classify}",
                  f"output: "
            ])
            api_response = api_response.text

         # Validate Output:
         output_as_json = json.loads(api_response)

         # Check for keys:
         for entry in output_as_json:
            if not all(required_field in entry for required_field in ["identifier","audience"]):
               print(f"{index} Missing required field")
               raise
            if not entry["audience"] in ["political", "central_bank", "financial_market", "academic"]:
               print(f"{index} Contains not allowed value")
               raise

         # Do something with the response
         Path(f"{log_path}/extractaudience/{index}.txt").write_text(api_response, encoding="UTF-8")

         return(api_response)
      # Query jobs:
      for index in range(math.ceil(metadata.shape[0]/classify_in_one_go)):
         results_audience.append(pool.apply_async(run_in_parallel, callback = updateprogress, kwds= {
            "index" : index,
         }))

      pool.close()
      pool.join()

processed_audiences = [result.get() for result in results_audience]

converted_to_dicts = []
for result in processed_audiences:
   entries = json.loads(result)
   converted_to_dicts.extend(entries)

processed_audiences = pd.DataFrame(converted_to_dicts)
# Check that all indentifiers are present and that lengths match. Order doesnt matter since we will merge on identfier.
set(metadata.identifier.tolist()) - set(processed_audiences.identifier.tolist())
len(metadata.identifier) == len(processed_audiences.identifier)

processed_audiences.to_parquet("../../data/processed/extracted_audience.parquet")

non_null_locations = processed_metadata[processed_metadata.location.notnull()]

results_coordinates = []
with tqdm(total=math.ceil(non_null_locations.shape[0]/classify_in_one_go)) as pbar:
      pool = Pool(8)
      def updateprogress(_):
        pbar.update(1)

      def failed_too_often(fail):
         print("Failed too many times; moving on!")
         print(fail)
         return("")
      
      @retry(stop=stop_after_attempt(10), wait=wait_fixed(10), retry_error_callback = failed_too_often)
      def run_in_parallel(index):
         locations_to_find = (
            non_null_locations.iloc[(index * classify_in_one_go): (index * classify_in_one_go + classify_in_one_go)].speech_identifier +
              ";" +
            non_null_locations.iloc[(index * classify_in_one_go): (index * classify_in_one_go + classify_in_one_go)].location).str.cat(sep='\n')
         
         gemini_client = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                    generation_config= {
                                    "temperature": 0.5,
                                    "top_p": 1,
                                    "top_k": 1,
                                    "max_output_tokens": 2048,
                                    })
         
         if (read_cache and Path(f"{log_path}/extractlocation/{index}.txt").is_file()):
            api_response = Path(f"{log_path}/extractlocation/{index}.txt").read_text(encoding= "UTF-8")
         else:
            api_response = gemini_client.generate_content([
                  f"input: {prompt_location}{examples_location}",
                  f"output: {desired_output_location}",
                  f"input: {prompt_location}{locations_to_find}",
                  f"output: "
            ])
            api_response = api_response.text

         # Validate Output:
         output_as_json = json.loads(api_response)

         # Check for keys:
         for entry in output_as_json:
            if not all(required_field in entry for required_field in ["identifier","latitude","longitude"]):
               print(f"{index} does not match requirements. Retry")
               raise

         # Do something with the response
         Path(f"{log_path}/extractlocation/{index}.txt").write_text(api_response, encoding="UTF-8")

         return(api_response)
      # Query jobs:
      for index in range(math.ceil(non_null_locations.shape[0]/classify_in_one_go)):
         results_coordinates.append(pool.apply_async(run_in_parallel, callback = updateprogress, kwds= {
            "index" : index,
         }))

      pool.close()
      pool.join()

raw_coordinates = [result.get() for result in results_coordinates]
converted_to_dicts = []
for result in raw_coordinates:
   entries = json.loads(result)
   converted_to_dicts.extend(entries)

raw_coordinates = pd.DataFrame(converted_to_dicts)
raw_coordinates.to_parquet("../../data/processed/extracted_coordinates.parquet")

# Merge together metadata and write to single file.
raw_coordinates = pd.read_parquet("../../data/processed/extracted_coordinates.parquet")
processed_audiences = pd.read_parquet("../../data/processed/extracted_audience.parquet")
processed_metadata = pd.read_parquet("../../data/processed/extracted_main_metadata.parquet")
metadata_before = pd.read_parquet("../../data/processed/speech_metadata.parquet")

merged_data = metadata_before.merge(
   processed_metadata, on = "speech_identifier", how = "left"
).merge(
   processed_audiences, left_on= "speech_identifier", right_on= "identifier", how = "left"
).merge(
   raw_coordinates, left_on= "speech_identifier", right_on= "identifier", how = "left"
).drop(["identifier_x", "identifier_y"], axis = 1)

def create_replace_dict(dictionary):
   """Turns a dictionary of the format {name : [name variant 1, name variant 2, name variant 3, ...]} into a dict that can be used with pandas replace"""
   replace_dict = {}
   for key, values in dictionary.items():
      for v in values:
         replace_dict[v] = key

   return(replace_dict)

merged_data = merged_data.assign(
   position = lambda df: df.position.str.split(';').str[0], # Anything that isnt the first central bank author or position
   central_bank = lambda df: df.central_bank.str.split(';').str[0],
   # speaker = lambda df: df.speaker.str.split(';').str[0] We dont use the speaker information from Gemini but rather what is in the BIS set already
).assign(
   speaker = lambda df: df.author.replace(create_replace_dict(duplicate_names)),
   location = lambda df: df.location.replace(create_replace_dict(duplicate_cities))
).assign(
   central_bank = lambda df: df.central_bank.apply(lambda x: match_central_bank(central_bank_mapping, x, log = False))
).merge(pd.read_excel("../../data/input/metadata/speeches/post_llm_manual_fixes.xlsx", sheet_name = "central_banks"),
                      how = "left",
                      on = "speech_identifier").assign(
                                    central_bank = lambda df: df.central_bank_y.combine_first(df.central_bank_x)
                                ).drop(["central_bank_y", "central_bank_x", "comment", "author"], axis = 1)

merged_data.to_parquet("../../data/processed/complete_metadata.parquet")