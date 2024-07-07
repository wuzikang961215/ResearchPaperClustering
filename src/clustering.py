import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Clustering:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)

    def vectorize_texts(self, texts):
        """
        Vectorizes the input texts using TF-IDF.

        Args:
            texts (list): List of texts to be vectorized.

        Returns:
            sparse matrix: TF-IDF features.
        """
        return self.vectorizer.fit_transform(texts)

    def fit_predict(self, X):
        """
        Fits the KMeans model and predicts cluster labels.

        Args:
            X (sparse matrix): TF-IDF features.

        Returns:
            tuple: Cluster labels and silhouette score.
        """
        labels = self.model.fit_predict(X)
        score = silhouette_score(X, labels)
        return labels, score

    def load_abstracts(self, file_path):
        """
        Loads the abstracts from the CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            list: List of abstracts.
        """
        df = pd.read_csv(file_path)
        return df['Abstract'].tolist()

    def save_results(self, file_path, labels, score):
        """
        Saves the clustering results to a CSV file.

        Args:
            file_path (str): Path to the output CSV file.
            labels (list): List of cluster labels.
            score (float): Silhouette score.
        """
        df = pd.DataFrame({'cluster': labels})
        df.to_csv(file_path, index=False)
        print(f"Clustering results saved to {file_path}")
        print(f"Silhouette Score: {score}")