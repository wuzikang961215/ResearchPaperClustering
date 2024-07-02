from sklearn.metrics import silhouette_score

class Evaluation:
    @staticmethod
    def evaluate_clustering(labels, X):
        return silhouette_score(X, labels)
