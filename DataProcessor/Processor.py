import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class Processor:
    
    def __init__(self, filters: list[str], filterValues: list[float], df: pd.DataFrame, clusters: int = 5, randomState:int = 42):
        
        self.df = df 
        self.filters = filters
        self.filterValues = filterValues
        self.clusters = clusters
        self.randomState = randomState
        
    def outlierFilter(self):
        if not self.filters or not self.filterValues:
            return self.df
        
        if len(self.filters) != len(self.filterValues):
            return self.df
        
        for index, column in enumerate(self.filters):
            if column in self.df.columns:
                min_val, max_val = self.filters[index]
                before = len(self.df)
                self.df = self.df[(self.df[column] > min_val) & (self.df[column] < max_val)]    
                after = len(self.df)
        return self.df            

    def normalizeData(self):
        if self.df is None:
            raise ValueError("DataFrame not defined")
        
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        df_scaled = self.df.copy()
        df_scaled[numeric_cols] = StandardScaler().fit_transform(self.df[numeric_cols])
        return df_scaled


    def KMeans(self):
        if self.df is None:
            raise ValueError("DataFrame not defined")
        
        kmeans = KMeans(n_clusters=self.clusters, random_state=self.randomState)
        self.df['cluster'] = kmeans.fit_predict(self.df[self.df.columns])
        return self.df