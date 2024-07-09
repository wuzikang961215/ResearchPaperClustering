import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

class Clustering:
    def __init__(self, n_clusters=5, vectorizer_type='tfidf', n_init=100):
        self.n_clusters = n_clusters
        self.vectorizer_type = vectorizer_type
        self.vectorizer = self._initialize_vectorizer(vectorizer_type)
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=n_init)

    def _initialize_vectorizer(self, vectorizer_type):
        if vectorizer_type == 'tfidf':
            return TfidfVectorizer(max_features=1000, token_pattern=r'(?u)\b[A-Za-z]+\b')
        elif vectorizer_type == 'word2vec':
            from gensim.models import Word2Vec
            return Word2VecVectorizer()
        elif vectorizer_type == 'doc2vec':
            from gensim.models import Doc2Vec
            from gensim.models.doc2vec import TaggedDocument
            return Doc2VecVectorizer()
        elif vectorizer_type == 'bert':
            from transformers import BertTokenizer, BertModel
            return BERTVectorizer()
        else:
            raise ValueError("Unsupported vectorizer type. Choose from 'tfidf', 'word2vec', 'doc2vec', or 'bert'.")

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

# class Word2VecVectorizer:
#     def __init__(self, size=100, window=5, min_count=1, workers=4):
#         self.size = size
#         self.window = window
#         self.min_count = min_count
#         self.workers = workers
#         self.model = None

#     def fit_transform(self, texts):
#         from gensim.models import Word2Vec
#         sentences = [text.split() for text in texts]
#         self.model = Word2Vec(sentences, vector_size=self.size, window=self.window, min_count=self.min_count, workers=self.workers)
#         vectors = [self._vectorize(text) for text in sentences]
#         return np.array(vectors)

#     def _vectorize(self, text):
#         vector = np.mean([self.model.wv[word] for word in text if word in self.model.wv], axis=0)
#         return vector if vector.size else np.zeros(self.size)


# class Doc2VecVectorizer:
#     def __init__(self, vector_size=100, window=5, min_count=1, workers=4, epochs=20):
#         self.vector_size = vector_size
#         self.window = window
#         self.min_count = min_count
#         self.workers = workers
#         self.epochs = epochs
#         self.model = None

#     def fit_transform(self, texts):
#         from gensim.models import Doc2Vec
#         from gensim.models.doc2vec import TaggedDocument
#         documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(texts)]
#         self.model = Doc2Vec(documents, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers, epochs=self.epochs)
#         vectors = [self.model.infer_vector(doc.words) for doc in documents]
#         return np.array(vectors)


# class BERTVectorizer:
#     def __init__(self, model_name='bert-base-uncased'):
#         from transformers import BertTokenizer, BertModel
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = BertModel.from_pretrained(model_name)
    
#     def fit_transform(self, texts):
#         vectors = []
#         for text in texts:
#             inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
#             outputs = self.model(**inputs)
#             cls_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
#             vectors.append(cls_embedding.flatten())
#         return np.array(vectors)
