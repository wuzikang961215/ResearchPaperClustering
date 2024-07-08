import os
import pandas as pd

# Ensure the src directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

from data_processing import DataProcessing
from clustering import Clustering
from evaluation import Evaluation
from visualization import Visualization

def extract_abstracts():
    dp = DataProcessing()
    raw_data_dir = os.path.join(parent_dir, 'data/raw')
    results = dp.extract_abstracts(raw_data_dir)
    output_csv = os.path.join(parent_dir, 'data/processed/abstracts.csv')
    dp.save_abstracts_to_csv(results, output_csv)
    print(f"Abstracts saved to {output_csv}")

def main():
    extract_abstracts()

    clustering = Clustering(n_clusters=5)
    abstracts_df = pd.read_csv(os.path.join(parent_dir, 'data/processed/abstracts.csv'))
    abstracts = abstracts_df['Abstract'].tolist()
    file_names = abstracts_df['File Name'].tolist()

    max_clusters = 50  # Test up to 50 clusters
    wcss, silhouette_scores = clustering.determine_optimal_clusters(abstracts, max_clusters=max_clusters)

    Evaluation.plot_elbow_method(wcss)
    Evaluation.plot_silhouette_analysis(silhouette_scores)

    optimal_clusters = 20  # Setting to 20 clusters as per the discussion
    print(f"Setting the number of clusters to {optimal_clusters}")

    clustering = Clustering(n_clusters=optimal_clusters)
    X = clustering.vectorize_texts(abstracts)
    labels, score = clustering.fit_predict(X)

    output_path = os.path.join(parent_dir, 'data/processed/clustering_results.csv')
    clustering.save_results(file_names, abstracts, labels, output_path)
    print(f"Clustering results saved to {output_path}")
    print(f"Silhouette Score: {score}")

    summary_output_path = os.path.join(parent_dir, 'data/processed/summary_report.csv')
    clustering.generate_summary_report(file_names, abstracts, labels, summary_output_path)
    print(f"Summary report saved to {summary_output_path}")

    Visualization.visualize_clusters(X, labels)

if __name__ == "__main__":
    main()
