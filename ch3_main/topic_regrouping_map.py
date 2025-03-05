import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Load the provided Excel file
file_path = "topic_info.xlsx"  # Update the path if needed
xls = pd.ExcelFile(file_path)

df = pd.read_excel(xls, sheet_name="Sheet1")

# Extract topic numbers and representative words
# Use only one or two representative words per topic for readability
topic_dict = dict(zip(df['Topic'].astype(str), df['Representation'].apply(lambda x: ", ".join(eval(x)[:2]))))

# Define the regrouping structure
topic_merging = {
    'Monetary Policy': ['0', '12', '16', '17', '19', '21', '24'],
    'Economic Indicators': ['4', '12', '13', '20', '22', '27'],
    'Financial Markets': ['10', '11', '18', '29', '30', '33'],
    'Banking Supervision': ['6', '7', '25', '26', '34', '38'],
    'Digital Finance': ['2', '28', '36', '40'],
    'International Economics': ['18', '23', '35'],
    'Crisis Management': ['8', '9', '32', '37', '41'],
    'Climate Finance': ['5', '31'],
    'Payment Systems': ['15', '14', '39', '43'],
    'National Economy': ['1', '3', '42', '44']
}

# Create a graph
G = nx.Graph()

# Add topic group nodes
for group in topic_merging.keys():
    G.add_node(group, size=5000, color="lightblue")

# Add topic nodes and edges with reduced labels
for group, topics in topic_merging.items():
    for topic in topics:
        if topic in topic_dict:
            rep_words = topic_dict[topic]  # Use top 2 words
            G.add_node(topic, size=2000, color="lightgray", label=rep_words)
            G.add_edge(group, topic)

# Create a layout to minimize overlap
pos = nx.kamada_kawai_layout(G)

# Extract node sizes and colors
node_sizes = [G.nodes[n].get("size", 1000) for n in G.nodes]
node_colors = [G.nodes[n].get("color", "gray") for n in G.nodes]

# Draw the base graph
plt.figure(figsize=(14, 10))
nx.draw(G, pos, with_labels=False, node_size=node_sizes, node_color=node_colors, edge_color="gray", alpha=0.6)

# Add labels for topic groups (larger labels, positioned clearly)
for node in topic_merging.keys():
    x, y = pos[node]
    plt.text(x, y, node, fontsize=12, fontweight='bold', ha='center', va='center', 
             bbox=dict(facecolor="lightblue", edgecolor="black", boxstyle="round,pad=0.3"))

# Add labels for individual topics with representative words, avoiding overlap
labels = {node: G.nodes[node].get("label", node) for node in G.nodes}
for node, label in labels.items():
    if node not in topic_merging:  # These are individual topics
        x, y = pos[node]
        plt.text(x, y, label, fontsize=9, ha='center', va='center', 
                 bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.2"))

# Add arrows from larger topic groups to individual topics
for group, topics in topic_merging.items():
    for topic in topics:
        if topic in pos:  # Ensure the topic exists in the graph
            x1, y1 = pos[group]
            x2, y2 = pos[topic]
            plt.arrow(x1, y1, (x2 - x1) * 0.7, (y2 - y1) * 0.7, head_width=0.02, head_length=0.03, 
                      fc='black', ec='black', alpha=0.7)

# Finalize the visualization
plt.title("Topic Regrouping with Representative Words", fontsize=14, fontweight='bold')
plt.axis("off")
plt.show()
