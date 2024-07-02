from src.data_processing import DataProcessing
from src.clustering import Clustering
from src.evaluation import Evaluation
from src.visualization import Visualization

def main():
    # Initialize classes
    data_processing = DataProcessing()
    clustering = Clustering(n_clusters=5)
    evaluation = Evaluation()
    visualization = Visualization()
    
    # Step 1: Extract abstracts
    file_paths = ['data/raw/paper1.pdf', 'data/raw/paper2.pdf']
    abstracts = data_processing.extract_abstracts(file_paths)
    
    # Step 2: Preprocess text
    processed_abstracts = [data_processing.preprocess_text(text) for text in abstracts]
    
    # Step 3: Vectorize text
    X = clustering.vectorize_texts(processed_abstracts)
    
    # Step 4: Cluster texts
    labels, score = clustering.fit_predict(X)
    print("Labels:", labels)
    print("Silhouette Score:", score)
    
    # Step 5: Visualize clusters (optional)
    visualization.visualize_clusters(X, labels)
    
    # Save results
    # Save the processed abstracts, vectors, labels, etc.

if __name__ == "__main__":
    main()
