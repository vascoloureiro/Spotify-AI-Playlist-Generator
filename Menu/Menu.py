import pandas as pd
import numpy as np

# Components
from DataProcessor.DataProcessor import DataProcessor
from DataProcessor.Moodify import MoodifyMerger
from Classification.Classificator import Classificator
from Classification.RulesAssociation import RulesAssociation
from Classification.EmotionImputer import EmotionImputer


class Menu:
    def __init__(self, lang='PT'):
        """
        lang: 'PT' (Português) ou 'EN' (English)
        """
        self.lang = lang
        self.df_normalized = None
        self.X_train_data = None
        self.y_train_data = None
        self.classficator = None
        self.rules = None
        self.moodify = None
        self.df_result = None

    def _t(self, pt, en):
        return pt if self.lang.upper() == 'PT' else en

    def prepare_data(self):
        print(self._t("=== ANÁLISE DE MÚSICAS SPOTIFY ===", "=== SPOTIFY MUSIC ANALYSIS ==="))

        columnsToKeep = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'duration_ms', 'track_popularity'
        ]
        colunmsBinary = ['danceability', 'energy', 'valence', 'acousticness', 'loudness']

        dataMusic = DataProcessor('./Data/spotify_songs.csv', 'Unnamed: 0', columnsToKeep)
        dataMusic.cleanCsv(outlier=True)
        self.df_normalized = dataMusic.normalizeCleanData(save=True)

        # Criar coluna popular_label
        threshold = self.df_normalized['track_popularity'].quantile(0.75)
        self.df_normalized['popular_label'] = (self.df_normalized['track_popularity'] > threshold).astype(int)
        self.df_normalized['popular'] = self.df_normalized['popular_label']

        # Selecionar features numéricas para treino
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
        dt_report, dt_acc = self.classficator.reportDecisionTree()
        rf_report, rf_acc = self.classficator.reportRandomForest()

        print("\n--- Decision Tree ---")
        print(f"{self._t('Acurácia', 'Accuracy')}: {dt_acc:.4f}")
        print(dt_report)
        print("\n--- Random Forest ---")
        print(f"{self._t('Acurácia', 'Accuracy')}: {rf_acc:.4f}")
        print(rf_report)

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

    def run(self):
        self.prepare_data()
        self.train_models()
        self.association_rules()
        self.moodify_analysis()
        print("\n" + self._t("=== PROCESSO CONCLUÍDO ===\nAnálise finalizada com sucesso!", "=== PROCESS FINISHED ===\nAnalysis completed successfully!"))

