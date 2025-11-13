import pandas as pd
import numpy as np

""" from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
 """
 
# Components
from DataProcessor.DataProcessor import DataProcessor
from Classification.Classificator import Classificator


""" Preparation Data"""
columnsToKeep = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms', 'track_popularity'
]
dataMusic = DataProcessor('./Data/spotify_songs.csv', 'Unnamed: 0', columnsToKeep)

dataMusic.cleanCsv(outlier=True)
df_normalized = dataMusic.normalizeCleanData(save=True)  #K-Means works better if the data is in the same scale

#dataMusic.debugger()
#print(df_normalized.head()) # Debbug

""" Training """
classficator = Classificator(
    df_normalized.drop(['track_popularity', 'cluster', 'popular_label'], axis=1, errors='ignore'),
    df_normalized['popular_label'],
    0.2,
    42,
    'y'
)

# Train and retrive the Reports of diferent algorithms
dt_report,dt_acc = classficator.reportDecisionTree()
rf_report, rf_acc= classficator.reportRandomForest()

#debugg
print("Decision Tree Accuracy:", dt_acc)
print(dt_report)
print("Random Forest Accuracy:", rf_acc)
print(rf_report)    