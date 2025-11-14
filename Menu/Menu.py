import numpy as np
import pandas as pd

from DataProcessor.DataProcessor import DataProcessor
from DataProcessor.Moodify import MoodifyMerger
from DataProcessor.GraphReport import GraphReport
from Classification.Classificator import Classificator
from Classification.RulesAssociation import RulesAssociation
from Classification.EmotionImputer import EmotionImputer


class Menu:
    def __init__(self, lang='PT', show_graphs=True):
        """
        lang: 'PT' (Português) or 'EN' (English)
        show_graphs: True para mostrar gráficos, False para apenas logs
        """
        self.lang = lang
        self.show_graphs = show_graphs
        self.df_normalized = None
        self.X_train_data = None
        self.y_train_data = None
        self.classficator = None
        self.rules = None
        self.moodify = None
        self.df_result = None
        
        self.dt_model = None
        self.rf_model = None
        self.dt_pred = None
        self.rf_pred = None
        self.X_test = None
        self.y_test = None
        self.dataMusic = None  

    def _t(self, pt, en):
        return pt if self.lang.upper() == 'PT' else en

    def prepare_data(self):
        print(self._t("=== ANÁLISE DE MÚSICAS SPOTIFY ===", "=== SPOTIFY MUSIC ANALYSIS ==="))

        columnsToKeep = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'duration_ms', 'track_popularity'
        ]
        
        self.dataMusic = DataProcessor('./Data/spotify_songs.csv', 'Unnamed: 0', columnsToKeep)
        self.dataMusic.cleanCsv(outlier=True)
        self.df_normalized = self.dataMusic.normalizeCleanData(save=True)

        print("\n" + self._t("=== DIAGNÓSTICO PRÉ-LABEL ===", "=== PRE-LABEL DIAGNOSIS ==="))
        print(f"track_popularity - Min: {self.df_normalized['track_popularity'].min():.2f}")
        print(f"track_popularity - Max: {self.df_normalized['track_popularity'].max():.2f}")
        print(f"track_popularity - {self._t('Média', 'Mean')}: {self.df_normalized['track_popularity'].mean():.2f}")
        
        # Create column for popular_label and popular
        threshold = self.df_normalized['track_popularity'].quantile(0.75)
        print(f"\n{self._t('Usando threshold dinâmico (75º percentil)', 'Using dynamic threshold (75th percentile)')}: {threshold:.2f}")
        
        self.df_normalized['popular_label'] = (self.df_normalized['track_popularity'] > threshold).astype(int)
        self.df_normalized['popular'] = self.df_normalized['popular_label']

        # Verification of the dataset
        print("\n" + self._t("=== ANÁLISE DO DATASET ===", "=== DATASET ANALYSIS ==="))
        print(f"{self._t('Total de músicas', 'Total songs')}: {len(self.df_normalized)}")
        print(f"\n{self._t('Distribuição de popularidade', 'Popularity distribution')}:")
        print(self.df_normalized['popular_label'].value_counts())
        print(f"\n{self._t('Percentagem de músicas populares', 'Percentage of popular songs')}: {self.df_normalized['popular_label'].mean():.2%}")

        # Select numeric features for training
        cols_to_remove = [
            'track_popularity', 'cluster', 'popular_label', 'popular',
            'track_id', 'track_name', 'track_artist', 'playlist_genre'
        ]
        
        existing_cols_to_remove = [col for col in cols_to_remove if col in self.df_normalized.columns]
        self.X_train_data = self.df_normalized.drop(existing_cols_to_remove, axis=1)
        self.X_train_data = self.X_train_data.select_dtypes(include=[np.number])
        self.y_train_data = self.df_normalized['popular_label']

    def train_models(self):
        print("\n" + self._t("=== TREINAMENTO DOS MODELOS ===", "=== TRAINING MODELS ==="))
        self.classficator = Classificator(
            self.X_train_data,
            self.y_train_data,
            testSize=0.2,
            randomState=42,
            stratify='y'
        )
        
        # Obter dados de teste e predições
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_train_data,
            self.y_train_data,
            test_size=0.2,
            random_state=42,
            stratify=self.y_train_data
        )
        
        dt_report, dt_acc = self.classficator.reportDecisionTree()
        print("\n--- Decision Tree ---")
        print(f"{self._t('Acurácia', 'Accuracy')}: {dt_acc:.4f}")
        print(dt_report)
        
        # Random Forest
        rf_report, rf_acc = self.classficator.reportRandomForest()
        print("\n--- Random Forest ---")
        print(f"{self._t('Acurácia', 'Accuracy')}: {rf_acc:.4f}")
        print(rf_report)
        
        from Classification.Algorithm import AlgorithmModels
        algo = AlgorithmModels(self.X_train, self.X_test, self.y_train)
        self.dt_pred, self.dt_model = algo.DecisonTree()
        self.rf_pred, self.rf_model = algo.RandomForest()
        
        if self.show_graphs:
            grapher = GraphReport(
                X_test=self.X_test,
                y_test=self.y_test,
                dt_pred=self.dt_pred,
                rf_pred=self.rf_pred,
                rf_model=self.rf_model
            )
            
            # Confusion Matrices
            print(self._t("  → Matrizes de Confusão", "  → Confusion Matrices"))
            grapher.plot_confusion_matrices()
            
            # Feature Importance
            print(self._t("  → Importância das Features", "  → Feature Importance"))
            grapher.plot_feature_importance()

    def association_rules(self):
        print("\n" + self._t("=== REGRAS DE ASSOCIAÇÃO ===", "=== ASSOCIATION RULES ==="))
        colunmsBinary = ['danceability', 'energy', 'valence', 'acousticness', 'loudness']
        self.rules = RulesAssociation(
            df=self.df_normalized,
            columnsBind=colunmsBinary,
            binaryColumnName='popular',
            parmColumn='track_popularity',
            valueParm=70
        )
        
        self.rules.bind()
        binary_df = self.rules.getBinaryDF()
        print(f"{self._t('Binary DataFrame shape', 'Binary DataFrame shape')}: {binary_df.shape}")

        frequent_itemsets = self.rules.applyApriori()
        print(f"{self._t('Frequent itemsets encontrados', 'Frequent itemsets found')}: {len(frequent_itemsets)}")

        rules_df = self.rules.rules()
        
        print(f"{self._t('Regras de associação encontradas', 'Association rules found')}: {len(rules_df)}")
        
        if len(rules_df) > 0:
            print("\nTop 5 rules:")
            print(rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

    def clustering_analysis(self):
        """Nova seção para análise de clustering"""
        print("\n" + self._t("=== ANÁLISE DE CLUSTERING ===", "=== CLUSTERING ANALYSIS ==="))
        
        if not self.show_graphs:
            print(self._t("  Gráficos desativados.", "  Graphs disabled."))
            return
        
        # Verificar se temos dados normalizados e clusters
        if 'cluster' not in self.df_normalized.columns:
            print(self._t("Coluna 'cluster' não encontrada. Pulando análise.", 
                         "Column 'cluster' column not found. Skipping analysis."))
            return
        
        # Features para clustering (mesmas usadas no treino)
        feature_cols = self.X_train_data.columns.tolist()
        
        grapher = GraphReport()
        
        # Elbow Method
        print(self._t("  → Método do Cotovelo (Elbow)", "  → Elbow Method"))
        grapher.plot_elbow(self.df_normalized, feature_cols, k_min=2, k_max=10)
        
        # Silhouette Score
        print(self._t("  → Silhouette Score", "  → Silhouette Score"))
        grapher.plot_silhouette(self.df_normalized, feature_cols, k_min=2, k_max=10)
        
        # PCA 2D
        print(self._t("  → Visualização PCA 2D", "  → PCA 2D Visualization"))
        grapher.plot_pca(self.df_normalized, feature_cols, 'cluster')

    def moodify_analysis(self):
        print("\n" + self._t("=== INTEGRAÇÃO MOODIFY ===", "=== MOODIFY INTEGRATION ==="))
        
        label_map = {0: "calm", 1: "happy", 2: "sad", 3: "energetic"}
        self.moodify = MoodifyMerger(
            moodify_csv="./Data/278k_labelled_uri.csv",
            label_map=label_map
        )
        
        self.moodify.load()
        self.moodify.preprocess()
        self.df_result = self.moodify.merge(self.df_normalized)

        # Imputer
        emotion_imputer = EmotionImputer(mood_dataset="./Data/278k_song_labelled.csv")
        self.df_result = emotion_imputer.knn_impute(self.df_result)
        self.df_result = emotion_imputer.fallback_name_artist(self.df_result)

        # Report
        self.moodify.report(self.df_result)
        self.moodify.generate_playlists(self.df_result)
        
        if self.show_graphs:
            grapher = GraphReport()
            print(self._t("  → Distribuição de Emoções", "  → Emotion Distribution"))
            grapher.plot_emotion_distribution(self.df_result)

    def run(self, skip_clustering=False):
        """
        skip_clustering: Se True, pula a análise de clustering
        """
        self.prepare_data()
        self.train_models()
        self.association_rules()
        
        # Análise de clustering (opcional)
        if not skip_clustering:
            self.clustering_analysis()
        
        self.moodify_analysis()
        
        print("\n" + self._t(
            "=== PROCESSO CONCLUÍDO ===\nAnálise finalizada com sucesso!", 
            "=== PROCESS FINISHED ===\nAnalysis completed successfully!"
        ))
