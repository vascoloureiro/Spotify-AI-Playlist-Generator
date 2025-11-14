class EmotionImputer:
    def __init__(self, mood_dataset="278k_song_labelled.csv"):
        self.mood_dataset = mood_dataset

    def knn_impute(self, df_result):
        return df_result

    def fallback_name_artist(self, df_result):
        return df_result
