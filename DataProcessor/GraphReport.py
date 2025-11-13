import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

class GraphReport:
    """
    Classe para gerar gráficos de relatórios de classificação:
    - Confusion Matrix
    - Importância das Features (Random Forest)
    """
    def __init__(self, X_test: pd.DataFrame, y_test: pd.Series, dt_pred: list, rf_pred: list, rf_model=None):
        self.X_test = X_test
        self.y_test = y_test
        self.dt_pred = dt_pred
        self.rf_pred = rf_pred
        self.rf_model = rf_model

    def plot_confusion_matrices(self):
        """Gera as matrizes de confusão para Decision Tree e Random Forest."""
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(confusion_matrix(self.y_test, self.dt_pred), annot=True, fmt='d', cmap='Blues', ax=ax[0])
        ax[0].set_title('Decision Tree')
        ax[0].set_xlabel('Previsto')
        ax[0].set_ylabel('Real')

        sns.heatmap(confusion_matrix(self.y_test, self.rf_pred), annot=True, fmt='d', cmap='Greens', ax=ax[1])
        ax[1].set_title('Random Forest')
        ax[1].set_xlabel('Previsto')
        ax[1].set_ylabel('Real')

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        """Gera o gráfico de importância das features do Random Forest."""
        if self.rf_model is None:
            print("Random Forest model não fornecido. Não é possível plotar feature importance.")
            return
        
        importances = pd.Series(self.rf_model.feature_importances_, index=self.X_test.columns).sort_values(ascending=False)
        print("\nImportância das variáveis no modelo Random Forest:")
        print(importances)

        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances, y=importances.index, palette="viridis")
        plt.title("Importância das Features (Random Forest)")
        plt.xlabel("Importância")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()
