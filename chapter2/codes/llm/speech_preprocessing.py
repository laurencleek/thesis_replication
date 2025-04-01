import os
import pandas as pd
from pathlib import Path
import spacy
import datetime as dt
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import tiktoken

# The first time run the following to get the required models
# spacy_model = spacy.load('en_core_web_lg')
# nltk.download('punkt')

# FUNCTIONS # -------------------------------------

def clean_identifier_from_url(url):
    """Functions to retrieve an unqiue identifier from the URL included in dataset"""
    parsed_url = url.replace(".pdf", "")
    parsed_url = parsed_url.replace(".htm", "")
    parsed_url = parsed_url.replace(".", "")

    processed_string = re.search(r"/review/r(\d+)?([a-z]+)?", parsed_url)

    try:
        if processed_string.groups()[0]:
            date = processed_string.groups()[0]
            # Some speeches have a 0 before the day
            if len(date) == 7:
                date = date[:4] + date[5:]

        if processed_string.groups()[1]:
            letter = processed_string.groups()[1]
        else:
            letter = "z"
    except:
        print(f"Could not extract identifeir from {url}")

    return(f"{letter}{date}")

def extract_sentences(text):
    """Spacy equivalent of nltk.tokenize.sent_tokenize. Much slower however ~ 130 times"""
    labelled_output = spacy_model(text)
    return(
        [sentence.text for sentence in labelled_output.sents]
    )

enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(text):
    "Count tokens. Requires tiktoken and a enc loaded"
    try:
      length = len(enc.encode(text))
    except:
      print(f"something went wrong with the input {text}")
      length = 0
    return(length)

# Vectorized
def remove_before_pattern(series, pattern):
    """Removes anything that comes before the pattern. If the pattern is not found it preserves the series"""
    return(series.str.split(pattern).str[1].combine_first(series))

def frequency_count(series, pattern):
    """Counts the frequency of a regex pattern or word"""
    return(series.str.split(pattern).map(len) -1)

# END FUNCTIONS # -------------------------------------

# Run this only when calling the script directly
if '__main__' == __name__:
    # Actual code which loads the speeches
    bis_speeches = pd.read_csv("../../data/input/speeches.csv").assign(
        speech_identifier = lambda df: df.url.apply(clean_identifier_from_url),
    )

    # Attempt to remove metadata at beginning of speech
    bis_speeches["text"] = bis_speeches["text"].str.split(r"\* \* \*\n").str[1].combine_first(bis_speeches["text"])

    bis_speeches["text"] = (
        bis_speeches["text"].
        replace(r"\n\d+\n\n", "" ,regex= True). # Page numbers
        replace(r"(\n\d\n\n(.+\n)+)+\n", r"\n" ,regex= True). # Footnotes
        replace(r"\n.*\f", "" ,regex= True). # New page characters
        replace(r"\f", "" ,regex= True). # Delete remaininig \f entries
        replace(r"\d+\[\d+\]", "" ,regex= True). # Footnote
        replace(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "" ,regex= True). # URLs
        replace(r"BIS Review \d+/\d+", "", regex= True).
        replace(r"BIS central bankers' speeches", "", regex= True).
        replace(r"BIS central bankers. speeches", "", regex= True). 

        replace(r"(?<=[^0-9]\.)\d+", "", regex= True). # letter followed by dot followed by number. This is often a footnote
        replace(r"(?<=\.)\[\d+\]", "", regex= True). # Also footnotes like this .[digit]
        replace(r"(?<=[\!\.\?])\n\d+", r"\n", regex= True). # Sentence after line break starting with number is likely a footnote.
        replace(r"[\!\.\?]\n\d+", r"\n", regex= True).

        replace(r"\n[^\!\.\?]{1,15}\n", r"\n" ,regex= True). # Less than 15 characters in a line and nothing that termiantes a sentence.
        replace(r"\n+", r"\n" ,regex= True). # More than one line break in sequence
        replace(r"\n\.\n", r"\n" ,regex= True). # Lines that only contain a dot
        replace(r"\n\d+\.\n", r"\n", regex= True). # Lines that only contain a number followed by a dot
        replace(r"\n\d+\n", r"\n", regex= True) # Lines that only contain a digit
    )

    sentence_level = bis_speeches.assign(
        sentence = lambda x: x["text"].astype(str).map(sent_tokenize)
    ).drop(["text"], axis = 1).explode("sentence").dropna( #There are 34 senteces which are na.
        subset=["sentence"]
    ).assign(
        sentence = lambda x: x["sentence"].replace(r"\s+", " " ,regex= True),
        token_count = lambda x: x["sentence"].map(count_tokens),len = lambda x: x["sentence"].str.len(),
        ascii_letter_count = lambda x: x["sentence"].str.count('[a-zA-Z]'),
        share_ascii_letter = lambda x: x["ascii_letter_count"] / x["len"]
    )[
        lambda x : 
        (x["share_ascii_letter"] > 0.66) & 
        (x["token_count"] > 5)  & 
        (x["token_count"] < 200)  & 
        (x["len"] > 20) 
    ].reset_index(drop = True)

    sentence_level = sentence_level[["speech_identifier","sentence", "token_count",	"len",	"share_ascii_letter"]]

    sentence_level.to_pickle("../../data/processed/sentence_level_bis.pkl")

    # Prepare metadata for further processing

    bis_speeches_metadata = bis_speeches[["url", "title", "description", "date", "author", "speech_identifier"]].assign(
        description = lambda df: df.description.astype(str)
    )   

    # Fix missing/broken authors:
    bis_speeches_metadata = bis_speeches_metadata.merge(pd.read_excel("../../data/input/metadata/speeches/fixes_bis.xlsx", sheet_name = "missing_author"),
                                how = "left",
                                on = "url").assign(
                                    author = lambda df: df.author_y.combine_first(df.author_x)
                                ).drop(["author_y", "author_x"], axis = 1)

    # Fix broken descriptions:
    bis_speeches_metadata = bis_speeches_metadata.merge(pd.read_excel("../../data/input/metadata/speeches/fixes_bis.xlsx", sheet_name = "missing_description"),
                                how = "left",
                                on = "url").assign(
                                    description = lambda df: df.description_y.combine_first(df.description_x)
                                ).drop(["description_y", "description_x"], axis = 1)

    bis_speeches_metadata.to_parquet("../../data/processed/speech_metadata.parquet", index = None)