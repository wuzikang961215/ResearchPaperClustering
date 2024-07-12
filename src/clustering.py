import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture 
from sklearn.metrics import silhouette_score, davies_bouldin_score
from transformers import BertTokenizer, BertModel
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

class Clustering:
    def __init__(self, n_clusters=9, vectorizer_type='word2vec', algorithm='kmeans', n_init=100):
        self.n_clusters = n_clusters
        self.vectorizer_type = vectorizer_type
        self.algorithm = algorithm
        self.vectorizer = self._initialize_vectorizer(vectorizer_type)
        self.model = self._initialize_model(algorithm, n_clusters, n_init)
        

    def _initialize_vectorizer(self, vectorizer_type):
        if vectorizer_type == 'tfidf':
            return TfidfVectorizer(max_features=1000, token_pattern=r'(?u)\b[A-Za-z]+\b')
        elif vectorizer_type == 'spacy_word2vec':
            return SpacyWord2VecVectorizer()
        elif vectorizer_type == 'bert':
            return SBERTVectorizer()  # Changed from BERTVectorizer to SBERTVectorizer
        else:
            raise ValueError("Unsupported vectorizer type. Choose from 'tfidf', 'spacy_word2vec', or 'bert'.")
        
    def _initialize_model(self, algorithm, n_clusters, n_init):
        if algorithm == 'kmeans':
            return KMeans(n_clusters=n_clusters, random_state=42, n_init=n_init)
        elif algorithm == 'hierarchical':
            return AgglomerativeClustering(n_clusters=n_clusters)
        elif algorithm == 'dbscan':
            return DBSCAN(eps=0.5, min_samples=5)
        elif algorithm == 'spectral':
            return SpectralClustering(n_clusters=n_clusters, random_state=42)
        elif algorithm == 'gmm':
            return GaussianMixture(n_components=n_clusters, random_state=42)
        else:
            raise ValueError("Unsupported algorithm. Choose from 'kmeans', 'hierarchical', 'dbscan', 'spectral', or 'gmm'.")


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
        davies_bouldin_scores = []

        for n in range(2, max_clusters + 1):
            if self.algorithm == 'kmeans':
                model = KMeans(n_clusters=n, random_state=42)
            elif self.algorithm == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n)
            elif self.algorithm == 'spectral':
                model = SpectralClustering(n_clusters=n, random_state=42)
            elif self.algorithm == 'gmm':
                model = GaussianMixture(n_components=n, random_state=42)
            elif self.algorithm == 'dbscan':
                # DBSCAN does not use a predefined number of clusters, setting eps for each iteration
                model = DBSCAN(eps=0.5)
            labels = model.fit_predict(X)
            wcss.append(model.inertia_ if hasattr(model, 'inertia_') else 0)
            silhouette_scores.append(silhouette_score(X, labels))
            # Convert sparse matrix to dense array for davies_bouldin_score
            X_dense = X.toarray() if hasattr(X, 'toarray') else X
            davies_bouldin_scores.append(davies_bouldin_score(X_dense, labels))

        return wcss, silhouette_scores, davies_bouldin_scores
    

    def vectorize_texts(self, texts):
        """
        Vectorizes the input texts using the specified vectorizer.

        Args:
            texts (list): List of texts to be vectorized.

        Returns:
            array: Vectorized text features.
        """
        return self.vectorizer.fit_transform(texts)

    def fit_predict(self, X):
        """
        Fits the KMeans model and predicts cluster labels.

        Args:
            X (array): Vectorized text features.

        Returns:
            tuple: Cluster labels and silhouette score.
        """
        if self.algorithm in ['kmeans', 'spectral', 'gmm']:
            labels = self.model.fit_predict(X)
        elif self.algorithm == 'hierarchical':
            labels = self.model.fit_predict(X)
        elif self.algorithm == 'dbscan':
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
            if self.vectorizer_type == 'tfidf':
                X_cluster = self.vectorizer.transform(cluster_abstracts)
                terms = self.vectorizer.get_feature_names_out()
                top_terms = self._get_top_terms(X_cluster, terms)
            else:
                top_terms = self._get_top_terms_word2vec(cluster_abstracts)

            summary.append({
                'Cluster': cluster,
                'Number of Papers': len(cluster_abstracts),
                'Top Terms': ', '.join(top_terms),
                'Papers': cluster_filenames
            })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_path, index=False)

    def _get_top_terms(self, X, terms, top_n=4):
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in self.vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        top_terms = [word for word, freq in words_freq[:top_n]]
        return top_terms



    def _get_top_terms_word2vec(self, cluster_abstracts, top_n=4):
        # Load the pre-trained spaCy model
        nlp = spacy.load('en_core_web_md')
        docs = [nlp(abstract) for abstract in cluster_abstracts]
        word_vectors = np.array([token.vector for doc in docs for token in doc if not token.is_stop and not token.is_punct])
        avg_vector = word_vectors.mean(axis=0)
        similar_words = nlp.vocab.vectors.most_similar(avg_vector.reshape(1, -1), n=top_n)
        top_terms = [nlp.vocab.strings[word_id] for word_id in similar_words[0][0]]
        return top_terms
    
    

class BERTVectorizer:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
    
    def fit_transform(self, texts):
        vectors = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
            vectors.append(cls_embedding.flatten())
        return np.array(vectors)

    def transform(self, texts):
        vectors = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
            vectors.append(cls_embedding.flatten())
        return np.array(vectors)

    def get_feature_names_out(self):
        return np.array([f'feature_{i}' for i in range(self.model.config.hidden_size)])

class SpacyWord2VecVectorizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.size = self.nlp.vocab.vectors_length

    def fit_transform(self, texts):
        return np.array([self.nlp(text).vector for text in texts])

    def transform(self, texts):
        return np.array([self.nlp(text).vector for text in texts])

    def get_feature_names_out(self):
        return np.array([f'feature_{i}' for i in range(self.size)])

class SBERTVectorizer:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def fit_transform(self, texts):
        vectors = self.model.encode(texts)
        return np.array(vectors)

    def transform(self, texts):
        vectors = self.model.encode(texts)
        return np.array(vectors)

    def get_feature_names_out(self):
        return np.array([f'feature_{i}' for i in range(self.model.get_sentence_embedding_dimension())])
