import os
import sys

# Ensure the src directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.clustering import Clustering
from src.evaluation import Evaluation
from src.visualization import Visualization

def main():
    # Initialize clustering
    clustering = Clustering(n_clusters=5)

    # Directory containing processed data
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(parent_dir, '../data/processed')

    # Load abstracts from CSV
    abstracts = clustering.load_abstracts(os.path.join(processed_data_dir, 'abstracts.csv'))

    # Vectorize abstracts
    X = clustering.vectorize_texts(abstracts)

    # Perform clustering
    labels, score = clustering.fit_predict(X)

    # Save results to CSV
    clustering.save_results(os.path.join(processed_data_dir, 'clustering_results.csv'), labels, score)

    # Evaluate clustering
    print(f"Silhouette Score: {score}")

    # Visualize clusters
    visualization = Visualization()
    visualization.visualize_clusters(X, labels)

if __name__ == "__main__":
    main()
