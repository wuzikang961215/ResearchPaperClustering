from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

class Visualization:
    @staticmethod
    def visualize_clusters(X, labels):
        tsne = TSNE(n_components=2, random_state=42, init='random')
        X_embedded = tsne.fit_transform(X)
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, palette='viridis')
        plt.show()
