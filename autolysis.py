import os
import sys
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from dotenv import load_dotenv
from PIL import Image
import chardet  # To detect file encoding
from io import BytesIO
import argparse

# Load environment variables from .env file
load_dotenv()

# Fetch AI Proxy token from .env file
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN not found in .env file. Please add it.")
    sys.exit(1)

# Define headers for API request
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {AIPROXY_TOKEN}'
}

# Function to request AI to generate the narrative story
def get_ai_story(dataset_summary, dataset_info, visualizations):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    prompt = f"""
    Below is a detailed summary and analysis of a dataset. Please generate a **rich and engaging narrative** about this dataset analysis, including:

    1. **The Data Received**: Describe the dataset with vivid language. What does the data represent? What are its features? Create a story around it.
    2. **The Analysis Carried Out**: Explain the analysis methods used—highlighting techniques like missing value handling, outlier detection, clustering, etc.
    3. **The Insights Discovered**: What were the key findings? What trends or patterns emerged that can be interpreted as discoveries?
    4. **The Implications of Findings**: How do these insights influence decisions? What actions can be taken based on the analysis? What recommendations would you give?

    **Dataset Summary**:
    {dataset_summary}

    **Dataset Info**:
    {dataset_info}

    **Visualizations**:
    {visualizations}
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Will raise HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        sys.exit(1)

    return response.json()['choices'][0]['message']['content'] if response.status_code == 200 else None

# Function to load dataset with automatic encoding detection
def load_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())  # Detect the encoding of the file
        encoding = result['encoding']
        data = pd.read_csv(file_path, encoding=encoding)
        print(f"Data loaded with {encoding} encoding.")
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)

# Perform basic analysis
def basic_analysis(data):
    summary = data.describe(include='all').to_dict()
    missing_values = data.isnull().sum().to_dict()
    column_info = data.dtypes.to_dict()
    return {"summary": summary, "missing_values": missing_values, "column_info": column_info}

# Outlier detection using IQR
def outlier_detection(data):
    numeric_data = data.select_dtypes(include=np.number)
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum().to_dict()
    return {"outliers": outliers}

# Generate visualizations dynamically
def save_plot(fig, plot_name):
    plot_path = f"{plot_name}.png"
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    return plot_path

# Correlation heatmap
def generate_correlation_matrix(data):
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    return save_plot(fig, "correlation_matrix")

# PCA Visualization
def generate_pca_plot(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(StandardScaler().fit_transform(numeric_data))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], ax=ax)
    ax.set_title("PCA Plot")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    return save_plot(fig, "pca_plot")

# DBSCAN clustering
def dbscan_clustering(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_data)
    numeric_data['cluster'] = clusters
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=numeric_data.iloc[:, 0], y=numeric_data.iloc[:, 1], hue=numeric_data['cluster'], palette="viridis", ax=ax)
    ax.set_title("DBSCAN Clustering")
    return save_plot(fig, "dbscan_clusters")

# Hierarchical clustering
def hierarchical_clustering(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()
    linked = linkage(numeric_data, 'ward')
    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(linked, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")
    return save_plot(fig, "hierarchical_clustering")

# Save README file
def save_readme(content):
    try:
        readme_path = "README.md"
        with open(readme_path, "w") as f:
            f.write(content)
        print(f"README saved in the current directory.")
    except Exception as e:
        print(f"Error saving README: {e}")
        sys.exit(1)

# Full analysis workflow
def analyze_and_generate_output(file_path):
    data = load_data(file_path)
    analysis = basic_analysis(data)
    outliers = outlier_detection(data)
    combined_analysis = {**analysis, **outliers}

    image_paths = {
        'correlation_matrix': generate_correlation_matrix(data),
        'pca_plot': generate_pca_plot(data),
        'dbscan_clusters': dbscan_clustering(data),
        'hierarchical_clustering': hierarchical_clustering(data)
    }

    data_info = {
        "filename": file_path,
        "summary": combined_analysis["summary"],
        "missing_values": combined_analysis["missing_values"],
        "outliers": combined_analysis["outliers"]
    }

    narrative = get_ai_story(data_info["summary"], data_info["missing_values"], image_paths)
    save_readme(f"Dataset Analysis: {narrative}")
    return narrative, image_paths

# Main entry point
def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_and_generate_output(file_path)

if __name__ == "__main__":
    main()
