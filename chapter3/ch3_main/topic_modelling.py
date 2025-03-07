# topic_modeling.py
import os
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import umap as UMAP
import numpy as np
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer


def run_or_load_topic_modeling(filtered_df, output):
    
    #make a new folder for the output if it doesn't exist already
    topic_info_folder = os.path.join(output, 'topic_info')
    os.makedirs(topic_info_folder, exist_ok=True)

# Define the model save path within the 'topic_info' folder
    model_path = os.path.join(topic_info_folder, 'my_topics_model')
    
    if os.path.exists(model_path):
        print("Loading existing topic model...")
        model = BERTopic.load(model_path)
    else:
        print("Running topic modeling (this may take up to 22 minutes)...")
        model = run_topic_modeling(filtered_df, output)
    
    return model

def run_topic_modeling(filtered_df, output):
    docs = filtered_df.speech_text.to_list()

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(
    docs, 
    show_progress_bar=True,
    batch_size=32,  # Adjust based on your available memory
    device="cuda" if torch.cuda.is_available() else "cpu",  # Use GPU if available
    convert_to_numpy=True,  # For faster processing if you don't need torch tensors
    normalize_embeddings=True  # Normalize the embeddings to unit length
)

    extended_stop_words = [
            "the", "of", "to", "and", "a", "in", "is", "it", "you", "that", "he", "was", "for", "on", "are", "with", "as", "I", "his", "they",
            "be", "at", "one", "have", "this", "from", "or", "had", "by", "not", "word", "but", "what", "some", "we", "can", "out", "other", "were", "all",
            "there", "when", "up", "use", "your", "how", "said", "an", "each", "she", "which", "do", "their", "time", "if", "will", "way", "about", "many", "then",
            "them", "write", "would", "like", "so", "these", "her", "long", "make", "thing", "see", "him", "two", "has", "look", "more", "day", "could", "go", "come",
            "did", "number", "sound", "no", "most", "people", "my", "over", "know", "water", "than", "call", "first", "who", "may", "down", "side", "been", "now", "find",
            "any", "new", "work", "part", "take", "get", "place", "made", "live", "where", "after", "back", "little", "only", "round", "man", "year", "came", "show", "every",
            "good", "me", "give", "our", "under", "name", "very", "through", "just", "form", "sentence", "great", "think", "say", "help", "low", "line", "differ", "turn", "cause",
            "much", "mean", "before", "move", "right", "de", "old", "too", "same", "tell", "does", "set", "three", "want", "air", "well", "also", "play", "small", "end",
            "put", "home", "read", "hand", "port", "large", "spell", "add", "even", "land", "here", "must", "big", "high", "such", "follow", "act", "why", "ask", "men",
            "change", "went", "light", "kind", "off", "need", "house", "picture", "try", "us", "again", "animal", "point", "mother", "world", "near", "build", "self", "earth", "father",
            "therefore", "however", "nevertheless", "nonetheless", "although", "consequently", "furthermore", "moreover", "meanwhile", "whereas", "hereby", "thereby", "whereby", "therein", "thereupon",
            "whereupon", "la", "et", "hereto", "hereof", "les", "hereafter", "heretofore", "hereunder", "hereinafter", "hereinabove", "hereinbefore", "hereinbelow"
        ]

    # Add this to remove stopwords
    vectorizer_model = CountVectorizer(
        min_df=10, 
        max_df=0.5,  
        ngram_range=(2, 4), 
        stop_words=extended_stop_words,
        max_features=5000  
    )

    # Add this to set seed (reduce dimensionality)
    umap_models = UMAP.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.01,
        metric='cosine',
        low_memory=False,
        random_state=1337
    )

    # Run BERTopic model, auto topics limit
    model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        language="multilingual",
        calculate_probabilities=True,
        verbose=True,
        umap_model=umap_models,
        #nr_topics="auto",
        low_memory=False,
        min_topic_size=16
    )
    topics, probs = model.fit_transform(docs, embeddings)
    initial_topic_info = model.get_topic_info()
    
   # Reduce outliers
    new_topics = model.reduce_outliers(docs, topics, strategy="c-tf-idf", threshold=0.2)
    new_topics = model.reduce_outliers(docs, new_topics, probabilities=probs, 
                             threshold=0.05, strategy="probabilities")

    # Update the model with new topics
    model.update_topics(docs, topics=new_topics)
    model.update_topics(docs, vectorizer_model=vectorizer_model)

    # add timestamp for to examine over time change
    timestamps = filtered_df.date.to_list()
    timestamps = pd.to_datetime(timestamps)
    # Extract just the year
    years = [ts.year for ts in timestamps]
    years_series = pd.Series(years)
    years_list = years_series.to_list()

    # Save topic info
    topic_info = model.get_topic_info()
    excel_path = os.path.join(topic_info_folder, "topic_info.xlsx")
    pickle_path = os.path.join(topic_info_folder, "topic_info.pkl")

    # Save the DataFrame as an Excel file
    topic_info.to_excel(excel_path, index=False)
    print(f"Topic info saved to {excel_path}")

    # Save the DataFrame as a pickle file
    topic_info.to_pickle(pickle_path)
    print(f"Topic info saved to {pickle_path}")

    model_save_path = os.path.join(topic_info_folder, "my_topics_model")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Generate and save the Intertopic Distance Map
    fig = model.visualize_topics()
    topic_info_folder = os.path.join(output, "topic_info")

    # Save as interactive HTML
    html_path = os.path.join(topic_info_folder, "intertopic_distance_map.html")
    fig.write_html(html_path)
    print(f"Intertopic Distance Map saved to {html_path}")

    return model

def save_custom_latex_table(output, filename="topic_info_table.tex"):
    # Load the DataFrame from the .pkl file
    pkl_file_path = os.path.join(output, "topic_info", "topic_info.pkl")
    topic_info_full = pd.read_pickle(pkl_file_path)

    # Select necessary columns for LaTeX table
    summary_df = topic_info_full[['Topic', 'Count', 'Representation']]
    
    # Path to save LaTeX file
    latex_table_path = os.path.join(output, "topic_info", filename)
    
    # Custom LaTeX table with width control for full Representation text
    with open(latex_table_path, "w") as f:
        f.write("\\begin{longtable}{p{1cm} p{2cm} p{12cm}}\n")
        f.write("\\hline\n")
        f.write("Topic & Count & Representation \\\\\n")
        f.write("\\hline\n")
        
        # Write each row with complete Representation data
        for _, row in summary_df.iterrows():
            f.write(f"{row['Topic']} & {row['Count']} & {row['Representation']} \\\\\n")
            f.write("\\hline\n")
        
        f.write("\\end{longtable}\n")

    print(f"Custom LaTeX table saved to {latex_table_path}")

    # Dynamic topic model
    #timestamps = filtered_df['date'].dt.date.to_list()
    #topics_over_time = model.topics_over_time(docs, timestamps)

    # Visualize this
   # model.visualize_topics_over_time(topics_over_time, topics=[4, 5, 9])

    # Examine the topics
   # print(model.get_topic_freq().head(100))

    # Visualize topics
  #  model.visualize_topics()

    # Visualize topics bar chart
    # new_topics.visualize_barchart(width=580, height=630, top_n_topics=74, n_words=10)
    # folder_name = 'figures'
    # file_path = os.path.join(folder_name, 'Topics_all.png')
    # plt.savefig(file_path)
    # print(f"Figure saved at {file_path}")
    # plt.show()
