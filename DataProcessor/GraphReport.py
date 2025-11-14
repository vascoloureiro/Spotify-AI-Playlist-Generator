import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class GraphReport:
    """
    Classe para gerar gráficos e relatórios visuais:
    - Confusion Matrix
    - Feature Importance
    - Elbow Curve (KMeans)
    - Silhouette Score (KMeans)
    - PCA 2D
    - Distribuição de Emoções
    """
    def __init__(self, X_test=None, y_test=None, dt_pred=None, rf_pred=None, rf_model=None):
        self.X_test = X_test
        self.y_test = y_test
        self.dt_pred = dt_pred
        self.rf_pred = rf_pred
        self.rf_model = rf_model

    def plot_confusion_matrices(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(confusion_matrix(self.y_test, self.dt_pred), annot=True, fmt='d', cmap='Blues', ax=ax[0])
        ax[0].set_title('Decision Tree')

        sns.heatmap(confusion_matrix(self.y_test, self.rf_pred), annot=True, fmt='d', cmap='Greens', ax=ax[1])
        ax[1].set_title('Random Forest')

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        if self.rf_model is None:
            print("Random Forest model não fornecido.")
            return
        
        importances = pd.Series(
            self.rf_model.feature_importances_,
            index=self.X_test.columns
        ).sort_values(ascending=False)

        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances, y=importances.index)
        plt.title("Importância das Features (Random Forest)")
        plt.tight_layout()
        plt.show()

    def plot_elbow(self, df_scaled, feature_cols, k_min=2, k_max=8):
        inertias = []
        Ks = range(k_min, k_max+1)

        for k in Ks:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(df_scaled[feature_cols])
            inertias.append(km.inertia_)

        plt.figure(figsize=(7, 4))
        plt.plot(Ks, inertias, marker='o')
        plt.title("Elbow Method")
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.tight_layout()
        plt.show()

    def plot_silhouette(self, df_scaled, feature_cols, k_min=2, k_max=8):
        sil_scores = []
        Ks = range(k_min, k_max+1)

        for k in Ks:
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(df_scaled[feature_cols])
            sil_scores.append(silhouette_score(df_scaled[feature_cols], labels))

        plt.figure(figsize=(7, 4))
        plt.plot(Ks, sil_scores, marker='o')
        plt.title("Silhouette Score")
        plt.xlabel("k")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.show()

    def plot_pca(self, df_scaled, feature_cols, cluster_col):
        pca = PCA(n_components=2)
        Z = pca.fit_transform(df_scaled[feature_cols])

        plt.figure(figsize=(8,6))
        plt.scatter(Z[:,0], Z[:,1], c=df_scaled[cluster_col], s=5)
        plt.title("Clusters (PCA 2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.show()

    def plot_emotion_distribution(self, df_result):
        emo_counts = df_result['emotion'].value_counts().sort_values(ascending=False)

        plt.figure(figsize=(7,4))
        emo_counts.plot(kind='bar')
        plt.title("Distribuição de emoções")
        plt.xlabel("Emoção")
        plt.ylabel("Contagem")
        plt.tight_layout()
        plt.show()
