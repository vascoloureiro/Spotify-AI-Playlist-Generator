import pandas as pd
import numpy as np

class MoodifyMerger:
    def __init__(self, moodify_csv: str, label_map: dict[int, str]):
        """
        moodify_csv: caminho para o ficheiro 278k_labelled_uri.csv
        label_map: dicionário de mapping dos labels numéricos -> emoções
        """
        self.moodify_csv = moodify_csv
        self.label_map = label_map
        self.df_mood = None
        self.df_small = None

    def load(self):
        self.df_mood = pd.read_csv(self.moodify_csv)

        # limpar colunas unnamed
        drop_cols = [c for c in self.df_mood.columns if c.lower().startswith("unnamed")]
        self.df_mood = self.df_mood.drop(columns=drop_cols)
        self.df_mood.columns = self.df_mood.columns.str.lower().str.strip()

        if "uri" not in self.df_mood.columns or "labels" not in self.df_mood.columns:
            raise ValueError("Falta coluna 'uri' ou 'labels' no dataset Moodify.")

        return self.df_mood

    @staticmethod
    def extract_track_id(uri):
        try:
            return str(uri).split(":")[-1]
        except:
            return np.nan

    def preprocess(self):
        if self.df_mood is None:
            self.load()

        self.df_mood["track_id"] = self.df_mood["uri"].apply(self.extract_track_id)
        self.df_mood = self.df_mood.dropna(subset=["track_id"])

        self.df_small = self.df_mood[["track_id", "labels"]].drop_duplicates()
        self.df_small["emotion"] = self.df_small["labels"].map(self.label_map)

        return self.df_small

    def merge(self, df_main: pd.DataFrame):
        """
        df_main precisa ter 'track_id'
        """
        if self.df_small is None:
            self.preprocess()

        if "track_id" not in df_main.columns:
            raise ValueError("df_main precisa conter a coluna 'track_id'.")

        merged = df_main.merge(
            self.df_small[["track_id", "labels", "emotion"]],
            on="track_id",
            how="left"
        )

        return merged

    def report(self, df_result: pd.DataFrame):
        tot = len(df_result)
        com_emocao = df_result["emotion"].notna().sum()

        print(f"\nCobertura: {com_emocao}/{tot} = {com_emocao/tot:.1%}")
        print("\nTop emoções:")
        print(df_result["emotion"].value_counts())

        return None

    def generate_playlists(self, df_result: pd.DataFrame):
        for alvo in df_result["emotion"].dropna().unique():
            cand = df_result[df_result["emotion"].str.lower() == alvo.lower()]

            if len(cand) >= 10:
                playlist = cand.sample(10, random_state=42)
                file = f"./Output/playlist_{alvo.lower()}.csv"

                playlist[["track_name", "track_artist", "playlist_genre", "emotion"]].to_csv(file, index=False)

            else:
                print(f"Few musics for emocion")
    
    def generate_playlist_cli(self, df_features, df_result, scaler, kmeans):

        try:
            d = float(input("Danceability (0-1, default 0.6): ") or 0.6)
            e = float(input("Energy (0-1, default 0.7): ") or 0.7)
            v = float(input("Valence (0-1, default 0.6): ") or 0.6)
            emo = input("Emotion (happy/sad/calm/energetic ou Enter to ignore): ").strip().lower()

            user_profile = {'danceability': d, 'energy': e, 'valence': v}
            user_vec = pd.DataFrame([np.zeros(len(df_features.columns))], columns=df_features.columns)
            for k, vv in user_profile.items():
                if k in user_vec.columns: user_vec[k] = vv

            user_vec_sc = pd.DataFrame(scaler.transform(user_vec), columns=user_vec.columns)
            user_cluster = kmeans.predict(user_vec_sc[df_features.columns])[0]

            base = df_result[df_result['cluster'] == user_cluster]
            if emo:
                base = base[base['emotion'].fillna("").str.lower() == emo]

            if len(base) >= 10:
                playlist = base.sample(10, random_state=42)
                print("\nPlaylist generated:")
                playlist[['track_name','track_artist','playlist_genre','emotion']].to_csv(
                    "./Output/playlist_custom.csv", index=False)
            else:
                print("No musics with the associeted credits.")
        except Exception as ex:
            print("Input ignore", ex)