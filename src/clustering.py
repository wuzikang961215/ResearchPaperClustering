from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Clustering:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)

    def vectorize_texts(self, texts):
        return self.vectorizer.fit_transform(texts)

    def fit_predict(self, X):
        labels = self.model.fit_predict(X)
        score = silhouette_score(X, labels)
        return labels, score
