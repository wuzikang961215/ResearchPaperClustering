from sklearn.metrics import silhouette_score

class Evaluation:
    @staticmethod
    def evaluate_clustering(labels, X):
        return silhouette_score(X, labels)

    @staticmethod
    def plot_elbow_method(wcss):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 7))
        plt.plot(range(2, len(wcss) + 2), wcss, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.show()

    @staticmethod
    def plot_silhouette_analysis(silhouette_scores):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 7))
        plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
        plt.title('Silhouette Analysis')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
