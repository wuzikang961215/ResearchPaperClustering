import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Clustering:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        # Ensure only alphabetic words are considered as features
        self.vectorizer = TfidfVectorizer(max_features=1000, token_pattern=r'(?u)\b[A-Za-z]+\b')
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)

    def determine_optimal_clusters(self, texts, max_clusters=20):
        """
        Determines the optimal number of clusters using the elbow method and silhouette score.

        Args:
            texts (list): List of texts to be clustered.
            max_clusters (int): Maximum number of clusters to consider.

        Returns:
            tuple: WCSS values and silhouette scores for each number of clusters.
        """
        X = self.vectorize_texts(texts)
        wcss = []
        silhouette_scores = []

        for n in range(2, max_clusters + 1):
            model = KMeans(n_clusters=n, random_state=42)
            labels = model.fit_predict(X)
            wcss.append(model.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))

        return wcss, silhouette_scores

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

    def save_results(self, file_names, abstracts, labels, output_path):
        """
        Save the clustering results to a CSV file.

        Args:
            file_names (list): List of file names.
            abstracts (list): List of abstracts.
            labels (list): List of cluster labels.
            output_path (str): Path to save the CSV file.
        """
        results = []
        for file_name, abstract, label in zip(file_names, abstracts, labels):
            first_10_words = ' '.join(abstract.split()[:10])
            results.append((file_name, label, first_10_words))

        df = pd.DataFrame(results, columns=['File Name', 'Cluster', 'First 10 Words of Abstract'])
        df.to_csv(output_path, index=False)


    def generate_summary_report(self, file_names, abstracts, labels, output_path):
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        summary = []
        
        for cluster in range(self.n_clusters):
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster]
            cluster_abstracts = [abstracts[i] for i in cluster_indices]
            cluster_filenames = [file_names[i] for i in cluster_indices]

            # Get top terms for the cluster
            X_cluster = self.vectorizer.transform(cluster_abstracts)
            terms = self.vectorizer.get_feature_names_out()
            top_terms = self._get_top_terms(X_cluster, terms)

            summary.append({
                'Cluster': cluster,
                'Number of Papers': len(cluster_abstracts),
                'Top Terms': ', '.join(top_terms),
                'Papers': cluster_filenames
            })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_path, index=False)

    def _get_top_terms(self, X, terms, top_n=10):
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in self.vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        top_terms = [word for word, freq in words_freq[:top_n]]
        return top_terms