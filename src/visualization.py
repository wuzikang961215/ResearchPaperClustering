from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

class Visualization:
    @staticmethod
    def visualize_clusters(X, labels):
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(X)
        plt.figure(figsize=(10, 7))
        sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=labels, palette='viridis')
        plt.show()
