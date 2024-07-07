import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the src directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.clustering import Clustering
from src.evaluation import Evaluation
from src.visualization import Visualization

def plot_elbow_method(wcss):
    plt.figure(figsize=(10, 7))
    plt.plot(range(2, len(wcss) + 2), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

def plot_silhouette_analysis(silhouette_scores):
    plt.figure(figsize=(10, 7))
    plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
    plt.title('Silhouette Analysis')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

def main():
    # Initialize clustering
    clustering = Clustering(n_clusters=5)
    
    # Load abstracts from CSV
    abstracts_df = pd.read_csv(os.path.join(parent_dir, 'data/processed/abstracts.csv'))
    abstracts = abstracts_df['Abstract'].tolist()
    file_names = abstracts_df['File Name'].tolist()
    
    # Determine the optimal number of clusters
    max_clusters = 50  # Test up to 50 clusters
    wcss, silhouette_scores = clustering.determine_optimal_clusters(abstracts, max_clusters=max_clusters)
    
    # Plot elbow method
    plot_elbow_method(wcss)
    
    # Plot silhouette analysis
    plot_silhouette_analysis(silhouette_scores)
    
    # Choose the optimal number of clusters based on silhouette score
    optimal_clusters = 20  # Setting to 20 clusters as per the discussion
    print(f"Setting the number of clusters to {optimal_clusters}")
    
    clustering = Clustering(n_clusters=optimal_clusters)
    
    # Vectorize abstracts
    X = clustering.vectorize_texts(abstracts)
    
    # Fit and predict clusters
    labels, score = clustering.fit_predict(X)
    
    # Save clustering results
    output_path = os.path.join(parent_dir, 'data/processed/clustering_results.csv')
    clustering.save_results(file_names, abstracts, labels, output_path)
    print(f"Clustering results saved to {output_path}")
    print(f"Silhouette Score: {score}")

    # Generate and save summary report
    summary_output_path = os.path.join(parent_dir, 'data/processed/summary_report.csv')
    clustering.generate_summary_report(file_names, abstracts, labels, summary_output_path)
    print(f"Summary report saved to {summary_output_path}")

    # Visualize clusters
    visualization = Visualization()
    visualization.visualize_clusters(X, labels)

if __name__ == "__main__":
    main()
