import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple
from bertopic import BERTopic

def compute_embeddings(texts: List[str]) -> np.ndarray:
    """Compute embeddings for a list of texts using SentenceBERT."""
    model = SentenceTransformer('all-mpnet-base-v2')
    return model.encode(texts, show_progress_bar=True)

def get_topic_probabilities(
    model: BERTopic,
    texts: List[str]
) -> pd.DataFrame:
    """Get topic probabilities for each document."""
    topics, probs = model.transform(texts)
    probs_array = np.array(probs)
    return pd.DataFrame(
        probs_array,
        columns=[f'topic_{i}' for i in range(probs_array.shape[1])]
    )

def aggregate_embeddings_by_period(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    topic_probs: pd.DataFrame,
    topic_id: int,
    threshold: float = 0.3
) -> Dict[Tuple[str, str], np.ndarray]:
    """Aggregate embeddings by central bank and quarter."""
    df['quarter'] = pd.to_datetime(df['date']).dt.to_period('Q')
    
    aggregated = {}
    for bank in df['central_bank'].unique():
        for quarter in df['quarter'].unique():
            mask = (df['central_bank'] == bank) & \
                   (df['quarter'] == quarter) & \
                   (topic_probs[f'topic_{topic_id}'] > threshold)
            
            if mask.any():
                bank_quarter_embeddings = embeddings[mask]
                avg_embedding = np.mean(bank_quarter_embeddings, axis=0)
                aggregated[(bank, str(quarter))] = avg_embedding
    
    return aggregated

def compute_ncb_ecb_distances(
    aggregated_embeddings: Dict[Tuple[str, str], np.ndarray],
    ncb_list: List[str]
) -> pd.DataFrame:
    """Compute distances between ECB and NCB embeddings."""
    quarters = sorted(list(set(q for _, q in aggregated_embeddings.keys())))
    distances = []
    
    for q_idx in range(1, len(quarters)):
        current_q = quarters[q_idx]
        prev_q = quarters[q_idx-1]
        
        # Get ECB embedding for current quarter
        ecb_current = aggregated_embeddings.get(('european central bank', current_q))
        if ecb_current is None:
            continue
            
        # Average NCB embeddings from previous quarter
        ncb_prev_embeddings = []
        for ncb in ncb_list:
            emb = aggregated_embeddings.get((ncb, prev_q))
            if emb is not None:
                ncb_prev_embeddings.append(emb)
        
        if ncb_prev_embeddings:
            ncb_avg = np.mean(ncb_prev_embeddings, axis=0)
            distance = cosine(ecb_current, ncb_avg)
            distances.append({
                'quarter': current_q,
                'distance': distance
            })
    
    return pd.DataFrame(distances)

def plot_intra_topic_distances(
    distances: pd.DataFrame,
    topic_id: int,
    output_path: str
) -> None:
    """Plot the evolution of distances over time with professional styling."""
    plt.rcParams['font.family'] = 'Times New Roman'
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Apply smoothing using rolling average
    distances['smoothed_distance'] = distances['distance'].rolling(window=4, center=True).mean()
    
    sns.scatterplot(
        data=distances, 
        x='quarter', 
        y='distance',
        color='#8ab0c1',  
        alpha=0.3,
        s=30
    )
    
    sns.lineplot(
        data=distances, 
        x='quarter', 
        y='smoothed_distance',
        color='#2c5d73',  
        linewidth=2.5
    )
    
   
    plt.ylabel('Distance', fontsize=11)
    plt.xlabel('')  # Remove x-axis label
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(
        f'{output_path}/intra_topic_distance_topic_{topic_id}.png',
        bbox_inches='tight',
        dpi=300
    )
    plt.close()
